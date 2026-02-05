from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from deap import gp
from data.util import load_dataset
from data.linear_elasticity.linear_elasticity_dataset import data_path
from dctkit.mesh import util
from flex.gp import regressor as gps
from flex.gp.util import load_config_data, compile_individuals
from flex.gp.primitives import add_primitives_to_pset_from_dict
from dctkit import config
import dctkit
import ray
import numpy as np
import jax.numpy as jnp
import math
import time
import sys
import yaml
from typing import Tuple, Callable, Dict
import numpy.typing as npt
import pygmsh
from util import get_features_batch, get_LE_boundary_values

residual_formulation = False

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()


def eval_MSE_sol(
    func: Callable,
    X: npt.NDArray,
    y: Dict,
    S: SimplicialComplex,
    gamma: float,
    u_0: C.CochainP0,
) -> Tuple[float, npt.NDArray]:

    num_data, num_nodes, dim_embedded_space = X.shape

    num_faces = S.S[2].shape[0]

    # need to call config again before using JAX in energy evaluations to
    # make sure that the current worker has initialized JAX
    config()

    # create objective function and set its energy function
    def total_energy(x, curr_bvalues):
        x_reshaped = x.reshape(S.node_coords.shape)
        penalty = 0.0
        for key in curr_bvalues:
            idx, values = curr_bvalues[key]
            if key == ":":
                penalty += jnp.sum((x_reshaped[idx, :] - values) ** 2)
            else:
                penalty += jnp.sum((x_reshaped[idx, int(key)] - values) ** 2)
        penalty *= gamma
        F = C.CochainD0(S, S.get_deformation_gradient(x_reshaped))
        total_energy = func(F) + penalty
        return total_energy

    prb = oc.OptimizationProblem(
        dim=num_nodes * dim_embedded_space,
        state_dim=num_nodes * dim_embedded_space,
        objfun=total_energy,
    )

    total_err = 0.0

    best_sols = []

    # for i, x in enumerate(X):
    for i in range(num_data):
        # extract current bvalues
        curr_bvalues = y[i]

        args = {"curr_bvalues": curr_bvalues}
        prb.set_obj_args(args)

        # minimize the objective
        x_flatten = prb.solve(
            x0=u_0.coeffs.flatten(), maxeval=5000, ftol_abs=1e-12, ftol_rel=1e-12
        )

        if (
            prb.last_opt_result == 1
            or prb.last_opt_result == 3
            or prb.last_opt_result == 4
        ):

            current_err = np.linalg.norm(x_flatten - X[i, :].flatten()) ** 2
            x_reshaped = x_flatten.reshape(S.node_coords.shape)
            curr_nodes = C.CochainP0(S, x_reshaped)
            F = C.deformation_gradient(curr_nodes)
            W = jnp.stack([jnp.array([[0, jnp.e], [-jnp.e, 0]])] * num_faces)
            F_plus_W = C.CochainD0(S, F.coeffs + W)
            current_err += (func(F) - func(F_plus_W)) ** 2
        else:
            current_err = math.nan

        if math.isnan(current_err):
            total_err = 1e5
            break

        total_err += current_err

        best_sols.append(x_flatten.reshape(S.node_coords.shape))

    total_err *= 1 / X.shape[0]

    return 1000.0 * total_err, best_sols


def predict(individuals_str, toolbox, X, y, S, gamma, u_0, penalty):

    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, X, y, S, gamma, u_0)

    return u


def score(
    individuals_str,
    toolbox,
    X: npt.NDArray,
    y: npt.NDArray,
    S: SimplicialComplex,
    gamma: float,
    u_0: npt.NDArray,
    penalty: dict,
):

    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, X, y, S, gamma, u_0)

    return MSE


def fitness(
    individuals_str,
    toolbox,
    X: npt.NDArray,
    y: npt.NDArray,
    S: SimplicialComplex,
    gamma: float,
    u_0: npt.NDArray,
    penalty: dict,
):

    callables = compile_individuals(toolbox, individuals_str)
    individ_length = get_features_batch(individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, X, y, S, gamma, u_0)

        fitnesses[i] = (MSE + penalty["reg_param"] * individ_length[i],)

    return fitnesses


def stgp_linear_elasticity(config_file, output_path=None):
    global residual_formulation
    # generate mesh
    lc = 0.2
    L = 2.0
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[0.0, 0.0], [L, 0.0], [L, L], [0.0, L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        geom.add_physical(p.lines[0], label="down")
        geom.add_physical(p.lines[2], label="up")
        geom.add_physical(p.lines[1], label="right")
        geom.add_physical(p.lines[3], label="left")
        mesh = geom.generate_mesh()

    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_DPD_weights()
    S.get_flat_DPP_weights()

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_path, "npy")

    # set bc
    num_faces = S.S[2].shape[0]
    ref_node_coords = S.node_coords

    left_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "left")
    right_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "right")
    down_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "down")
    up_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "up")

    # FIXME: just to initialize ref_metric_contravariant.
    # Write a routine in simplex that does it
    _ = S.get_deformation_gradient(ref_node_coords)

    # define a dictionary containing boundary nodes information (needed to set properly
    #  boundary_values)
    boundary_nodes_info = {
        "left_bnd_nodes_idx": left_bnd_nodes_idx,
        "right_bnd_nodes_idx": right_bnd_nodes_idx,
        "up_bnd_nodes_idx": up_bnd_nodes_idx,
        "down_bnd_nodes_idx": down_bnd_nodes_idx,
    }

    # extract boundary values
    bvalues_train = get_LE_boundary_values(
        X=X_train,
        y=y_train,
        ref_node_coords=ref_node_coords,
        boundary_nodes_info=boundary_nodes_info,
    )
    bvalues_val = get_LE_boundary_values(
        X=X_val,
        y=y_val,
        ref_node_coords=ref_node_coords,
        boundary_nodes_info=boundary_nodes_info,
    )
    bvalues_test = get_LE_boundary_values(
        X=X_test,
        y=y_test,
        ref_node_coords=ref_node_coords,
        boundary_nodes_info=boundary_nodes_info,
    )

    # penalty parameter for the Dirichlet bcs
    gamma = 1000000.0

    # initial guess for the solution of the problem
    u_0 = C.CochainP0(S, ref_node_coords)

    residual_formulation = config_file["gp"]["residual_formulation"]

    # define primitive set and add primitives and terminals
    if residual_formulation:
        print("Using residual formulation.")
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], C.Cochain)
        # ones cochain
        pset.addTerminal(
            C.Cochain(
                S.num_nodes, True, S, np.ones(S.num_nodes, dtype=dctkit.float_dtype)
            ),
            C.Cochain,
            name="F",
        )
    else:
        pset = gp.PrimitiveSetTyped("F", [C.CochainD0T], float)

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1.0, float, name="-1.")
    pset.addTerminal(2.0, float, name="2.")
    pset.addTerminal(10.0, float, name="10.")
    pset.addTerminal(0.1, float, name="0.1")

    identity = jnp.stack([jnp.identity(2)] * num_faces)
    identity_coch = C.CochainD0T(S, identity)
    pset.addTerminal(identity_coch, C.CochainD0T, name="I")

    # rename arguments
    pset.renameArguments(ARG0="F")

    penalty = config_file["gp"]["penalty"]

    common_params = {"S": S, "penalty": penalty, "gamma": gamma, "u_0": u_0}

    # create symbolic regression problem instance
    gpsr = gps.GPSymbolicRegressor(
        pset=pset,
        fitness=fitness.remote,
        error_metric=eval_MSE.remote,
        predict_func=eval_best_sols.remote,
        feature_extractors=[len],
        print_log=True,
        common_data=common_params,
        config_file_data=config_file,
        save_best_individual=True,
        save_train_fit_history=True,
        plot_best=False,
        plot_best_individual_tree=True,
        output_path="./",
    )

    # datasets = {'train': [X_train, bvalues_train],
    #             'val': [X_val, bvalues_val],
    #             'test': [X_test, bvalues_test]}

    train_data = Dataset("D", [X_train, bvalues_train], y_train)

    start = time.perf_counter()
    # epsilon = "SubCD0T(symD0T(F), I)"
    # opt_string_eps = "AddF(MulF(2., InnD0T(epsilon, epsilon)),
    # MulF(10., InnD0T(MvD0VT(trD0T(epsilon), I), epsilon)))"
    # opt_string = opt_string_eps.replace("epsilon", epsilon)
    # opt_string = ""
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]
    gpsr.fit(train_data)

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")

    ray.shutdown()


if __name__ == "__main__":
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))

    # path for output data speficified
    if n_args >= 3:
        output_path = sys.argv[2]
    else:
        output_path = "."

    stgp_linear_elasticity(config_file, output_path)
