from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from matplotlib import tri
from deap import gp
from dctkit.mesh import util
from flex.gp.regressor import GPSymbolicRegressor
from flex.gp.util import load_config_data, compile_individuals
from flex.gp.primitives import add_primitives_to_pset_from_dict
from dctkit import config
import dctkit
import warnings
import numpy as np
import jax.numpy as jnp
import math
import time
import sys
from typing import Callable
import numpy.typing as npt
import os
from functools import partial
import ray
from util import get_features_batch, load_dataset, load_noise


residual_formulation = False

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
os.environ["JAX_PLATFORMS"] = "cpu"
config()

data_path = os.path.join(os.getcwd(), "data/poisson")
noise = np.load(os.path.join(data_path, "noise_poisson.npy"))


def is_valid_energy(
    u: npt.NDArray, prb: oc.OptimizationProblem, bnodes: npt.NDArray
) -> bool:
    # perturb solution and check whether the gradient still vanishes
    # (i.e. constant energy)
    u_noise = u + noise * np.mean(u)
    u_noise[bnodes] = u[bnodes]
    grad_u_noise = jnp.linalg.norm(prb.solver.gradient(u_noise))
    is_valid = grad_u_noise >= 1e-6
    return is_valid


def eval_MSE_sol(
    individual: Callable,
    X: npt.NDArray,
    y: npt.NDArray,
    S: SimplicialComplex,
    bnodes: npt.NDArray,
    gamma: float,
    u_0: C.CochainP0,
):

    warnings.filterwarnings("ignore")
    num_nodes = X.shape[1]

    # need to call config again before using JAX in energy evaluations to make
    # sure that  the current worker has initialized JAX
    config()

    # create objective function and set its energy function
    def total_energy(y, curr_x, curr_bvalues):
        penalty = 0.5 * gamma * jnp.sum((y[bnodes] - curr_bvalues) ** 2)
        c = C.CochainP0(S, y)
        fk = C.CochainP0(S, curr_x)
        if residual_formulation:
            total_energy = C.inner(individual(c, fk), individual(c, fk)) + penalty
        else:
            total_energy = individual(c, fk) + penalty
        return total_energy

    prb = oc.OptimizationProblem(
        dim=num_nodes, state_dim=num_nodes, objfun=total_energy
    )

    MSE = 0.0

    us = []

    # Dirichlet boundary conditions for all the samples
    bvalues = y[:, bnodes]

    # loop over dataset samples
    for i, curr_x in enumerate(X):

        curr_bvalues = bvalues[i, :]

        args = {"curr_x": curr_x, "curr_bvalues": curr_bvalues}
        prb.set_obj_args(args)

        # minimize the objective
        y_pred = prb.solve(
            x0=u_0.coeffs.flatten(), ftol_abs=1e-12, ftol_rel=1e-12, maxeval=1000
        )

        if (
            prb.last_opt_result == 1
            or prb.last_opt_result == 3
            or prb.last_opt_result == 4
        ):
            # check whether the energy is "admissible" (i.e. exclude
            # constant energies)
            valid_energy = is_valid_energy(u=y_pred, prb=prb, bnodes=bnodes)

            if valid_energy:
                current_err = np.linalg.norm(y_pred - y[i, :]) ** 2
            else:
                current_err = math.nan
        else:
            current_err = math.nan

        if math.isnan(current_err):
            MSE = 1e5
            us = [u_0.coeffs.flatten()] * X.shape[0]
            break

        MSE += current_err

        us.append(y_pred)

    MSE *= 1 / X.shape[0]

    return MSE, us


def predict(individuals_str, toolbox, X, y, S, bnodes, gamma, u_0, penalty):

    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, X, y, S, bnodes, gamma, u_0)

    return u


def score(
    individuals_str,
    toolbox,
    X: npt.NDArray,
    y: npt.NDArray,
    S: SimplicialComplex,
    bnodes: npt.NDArray,
    gamma: float,
    u_0: npt.NDArray,
    penalty: dict,
):

    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, X, y, S, bnodes, gamma, u_0)

    return MSE


def fitness(
    individuals_str,
    toolbox,
    X: npt.NDArray,
    y: npt.NDArray,
    S: SimplicialComplex,
    bnodes: npt.NDArray,
    gamma: float,
    u_0: npt.NDArray,
    penalty: dict,
):

    callables = compile_individuals(toolbox, individuals_str)
    individ_length = get_features_batch(individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, X, y, S, bnodes, gamma, u_0)

        fitnesses[i] = (MSE + penalty["reg_param"] * individ_length[i],)

    return fitnesses


def stgp_poisson(output_path=None):
    global residual_formulation
    regressor_params, config_file_data = load_config_data("poisson.yaml")

    # generate mesh and get data
    mesh, _ = util.generate_square_mesh(0.08)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    bnodes = mesh.cell_sets_dict["boundary"]["line"]
    num_nodes = S.num_nodes

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_path, "csv")

    # penalty parameter for the Dirichlet bcs
    gamma = 1000.0

    # initial guess for the solution of the Poisson problem
    u_0_vec = np.zeros(num_nodes, dtype=dctkit.float_dtype)
    u_0 = C.CochainP0(S, u_0_vec)

    residual_formulation = config_file_data["gp"]["residual_formulation"]

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
            name="ones",
        )
    else:
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], float)

    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1.0, float, name="-1.")
    pset.addTerminal(2.0, float, name="2.")

    pset.renameArguments(ARG0="u")
    pset.renameArguments(ARG1="f")

    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])

    penalty = config_file_data["gp"]["penalty"]
    common_params = {
        "S": S,
        "penalty": penalty,
        "bnodes": bnodes,
        "gamma": gamma,
        "u_0": u_0,
    }

    # seed = ["SubF(MulF(InnP1(cobP0(u), cobP0(u)),1/2), InnP0(u,f))"]

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        predict_func=partial(predict, y=y_test),
        score_func=score,
        print_log=True,
        common_data=common_params,
        save_best_individual=True,
        save_train_fit_history=True,
        output_path=output_path,
        **regressor_params,
    )

    start = time.perf_counter()

    gpsr.fit(X_train, y_train, X_val, y_val)

    print("Best MSE on the test set: ", gpsr.score(X_test, y_test))

    # PLOTS
    if config_file_data["gp"]["plot_best"]:
        u_pred = gpsr.predict(X_test)
        triang = tri.Triangulation(S.node_coords[:, 0], S.node_coords[:, 1], S.S[2])

        plt.figure(1, figsize=(12, 10))
        plt.clf()
        fig = plt.gcf()
        _, axes = plt.subplots(y_test.shape[0], 2, num=1)
        for i in range(0, y_test.shape[0]):
            pltobj = axes[i, 0].tricontourf(triang, y_test[i], cmap="RdBu", levels=20)
            axes[i, 1].tricontourf(triang, u_pred[i], cmap="RdBu", levels=20)
            axes[i, 0].set_box_aspect(1)
            axes[i, 1].set_box_aspect(1)
            # set x,y labels
            axes[i, 0].set_xlabel("x")
            axes[i, 0].set_ylabel("y")
            axes[i, 1].set_xlabel("x")
            axes[i, 1].set_ylabel("y")

        # set titles
        axes[0, 0].set_title("Data")
        axes[0, 1].set_title("Prediction")
        plt.colorbar(pltobj, ax=axes)
        fig.canvas.draw()
        plt.show()

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")

    ray.shutdown()


if __name__ == "__main__":
    n_args = len(sys.argv)
    # path for output data speficified
    if n_args >= 2:
        output_path = sys.argv[1]
    else:
        output_path = "."

    stgp_poisson(output_path)
