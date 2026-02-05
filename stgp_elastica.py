import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, Tuple
import jax.numpy as jnp
from jax import grad, Array
from deap import gp
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_line_mesh, build_complex_from_mesh
from dctkit.dec import cochain as C
from dctkit import config
from dctkit.math.opt import optctrl as oc
import dctkit as dt
from data.util import load_dataset
from data.elastica.elastica_dataset import data_path
from flex.gp import regressor as gps
from flex.gp.util import load_config_data, compile_individuals
from flex.gp.primitives import add_primitives_to_pset_from_dict
import matplotlib.pyplot as plt
import math
import sys
import yaml
import time
import ray
from scipy.linalg import block_diag
from scipy import sparse
from itertools import chain
from functools import partial
import os
from util import *


residual_formulation = False

# choose precision and whether to use GPU or CPU
#  needed for context of the plots at the end of the evolution
os.environ["JAX_PLATFORMS"] = "cpu"
config()

NUM_NODES = 11
LENGTH = 1.0


def is_valid_energy(
    theta_in: npt.NDArray, theta: npt.NDArray, prb: oc.OptimizationProblem
) -> bool:
    dim = len(theta_in)
    noise = 0.0001 * np.ones(dim).astype(dt.float_dtype)
    theta_in_noise = theta_in + noise
    theta_noise = prb.solve(
        x0=theta_in_noise, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12
    )
    is_valid = np.allclose(theta, theta_noise, rtol=1e-6, atol=1e-6)
    return is_valid


class Objectives:
    def __init__(self, S: SimplicialComplex) -> None:
        self.S = S

    def set_residual(self, func: Callable) -> None:
        """Set the energy function to be used for the computation of the objective
        function."""
        self.residual = func

    def set_energy_func(self, func: Callable) -> None:
        """Set the energy function to be used for the computation of the objective
        function."""
        self.energy_func = func

    # elastic energy including Dirichlet BC by elimination of the prescribed dofs
    def total_energy(
        self, theta_vec: npt.NDArray, FL2_EI0: float, theta_0: npt.NDArray
    ) -> Array:
        # extend theta on the boundary w.r.t boundary conditions
        theta_vec = jnp.insert(theta_vec, 0, theta_0)
        theta = C.CochainD0(self.S, theta_vec)
        FL2_EI0_coch = C.CochainD0(
            self.S, FL2_EI0 * jnp.ones(self.S.num_nodes - 1, dtype=dt.float_dtype)
        )
        if residual_formulation:
            residual = self.residual(theta, FL2_EI0_coch)
            energy = jnp.linalg.norm(residual.coeffs[1:]) ** 2
        else:
            energy = self.energy_func(theta, FL2_EI0_coch)
        return energy

    # state function: stationarity conditions for the total energy
    def total_energy_grad(self, x: npt.NDArray, theta_0: float) -> Array:
        theta = x[:-1]
        FL2_EI0 = x[-1]
        if residual_formulation:
            # FIXME: not sure why we are not applying grad to total_energy
            theta_vec = jnp.insert(theta, 0, theta_0)
            theta = C.CochainD0(self.S, theta_vec)
            FL2_EI0_coch = C.CochainD0(
                self.S, FL2_EI0 * jnp.ones(self.S.num_nodes - 1, dtype=dt.float_dtype)
            )
            return self.residual(theta, FL2_EI0_coch).coeffs[1:]
        else:
            return grad(self.total_energy)(theta, FL2_EI0, theta_0)

    # objective function for the parameter EI0 identification problem
    def MSE_theta(self, x: npt.NDArray, theta_true: npt.NDArray) -> Array:
        theta = x[:-1]
        theta = jnp.insert(theta, 0, theta_true[0])
        return jnp.sum(jnp.square(theta - theta_true))


def tune_EI0(
    func: Callable,
    toolbox,
    theta_true: npt.NDArray,
    FL2: float,
    EI0_guess: float,
    theta_guess: npt.NDArray,
    S: SimplicialComplex,
) -> float:

    # number of unknowns angles
    dim = S.num_nodes - 2
    obj = Objectives(S=S)
    if residual_formulation:
        obj.set_residual(func)
    else:
        obj.set_energy_func(func)

    # prescribed angle at x=0
    theta_0 = theta_true[0]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    constraint_args = {"theta_0": theta_0}
    obj_args = {"theta_true": theta_true}

    prb = oc.OptimalControlProblem(
        objfun=obj.MSE_theta,
        statefun=obj.total_energy_grad,
        state_dim=dim,
        nparams=S.num_nodes - 1,
        constraint_args=constraint_args,
        obj_args=obj_args,
    )

    def get_bounds():
        lb = -100 * np.ones(dim + 1, dt.float_dtype)
        ub = 100 * np.ones(dim + 1, dt.float_dtype)
        lb[-1] = -100
        ub[-1] = -1e-3
        return (lb, ub)

    prb.get_bounds = get_bounds

    FL2_EI0 = FL2 / EI0_guess

    x0 = np.append(theta_guess, FL2_EI0)

    x = prb.solve(x0=x0, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12)

    # theta = x[:-1]
    FL2_EI0 = x[-1]

    EI0 = FL2 / FL2_EI0

    # compute violation norm
    viol = prb.constr(x, **prb.constr_args)
    viol = np.asarray(viol)

    viol_norm = np.linalg.norm(viol, ord=2)

    # define optimization success 1
    success_case_1 = (
        prb.last_opt_result == 1 or prb.last_opt_result == 3 or prb.last_opt_result == 4
    )
    success_case_2 = prb.last_opt_result == 5 and viol_norm <= 1e-12

    if success_case_1 or success_case_2:
        return EI0
    # in this case optimization is failed -> return a bad value
    return -1.0


def eval_MSE_sol(
    func: Callable,
    EI0: float,
    thetas_true: npt.NDArray,
    Fs: npt.NDArray,
    theta_in_all: npt.NDArray,
    S: SimplicialComplex,
) -> Tuple[float, npt.NDArray]:

    # number of unknown angles
    dim = S.num_nodes - 2

    total_err = 0.0

    obj = Objectives(S=S)
    if residual_formulation:
        obj.set_residual(func)
    else:
        obj.set_energy_func(func)

    X_dim = thetas_true.shape[0]
    best_theta = np.zeros((X_dim, S.num_nodes - 1), dtype=dt.float_dtype)

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    if EI0 < 0.0:
        total_err = 40.0
    else:
        for i, theta_true in enumerate(thetas_true):
            # extract prescribed value of theta at x = 0 from the dataset
            theta_0 = theta_true[0]

            theta_in = theta_in_all[i, :]

            FL2 = Fs[i]
            FL2_EI0 = FL2 / EI0

            prb = oc.OptimizationProblem(
                dim=dim, state_dim=dim, objfun=obj.total_energy
            )
            args = {"FL2_EI0": FL2_EI0, "theta_0": theta_0}
            prb.set_obj_args(args)
            theta = prb.solve(x0=theta_in, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12)

            # check whether the energy is "admissible" (i.e. exclude constant energies
            # and energies with minima that are too sensitive to the initial guess)
            valid_energy = is_valid_energy(theta_in=theta_in, theta=theta, prb=prb)

            if (
                prb.last_opt_result == 1
                or prb.last_opt_result == 3
                or prb.last_opt_result == 4
            ) and valid_energy:
                x = np.append(theta, FL2_EI0)
                fval = float(obj.MSE_theta(x, theta_true))
            else:
                fval = math.nan

            if math.isnan(fval):
                total_err = 40.0
                break

            total_err += fval

            # extend theta
            theta = np.insert(theta, 0, theta_0)

            # update best_theta
            best_theta[i, :] = theta

    total_err *= 1 / (X_dim)

    # round total_err to 5 decimal digits
    total_err = float("{:.5f}".format(total_err))

    return 10.0 * total_err, best_theta


def fitness(
    individuals_str: list[str],
    toolbox,
    X,
    y,
    theta_in_all: npt.NDArray,
    S: SimplicialComplex,
    penalty: dict,
) -> Tuple[float,]:

    callables = compile_individuals(toolbox, individuals_str)
    indlen = get_features_batch(individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(
            ind, individuals_str[i].EI0, X, y, theta_in_all["train"], S
        )

        # add penalty on length of the tree to promote simpler solutions
        fitnesses[i] = (MSE + penalty["reg_param"] * indlen[i],)

    return fitnesses


def predict(
    individuals_str: list[str],
    toolbox,
    X,
    y,
    theta_in_all: npt.NDArray,
    S: SimplicialComplex,
    penalty: dict,
) -> list:

    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(
            ind, individuals_str[i].EI0, X, y, theta_in_all["test"], S
        )

    return u


def score(
    individuals_str: list[str],
    toolbox,
    X,
    y,
    theta_in_all: npt.NDArray,
    S: SimplicialComplex,
    penalty: dict,
) -> list:

    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(
            ind, individuals_str[i].EI0, X, y, theta_in_all["val"], S
        )

    return MSE


def print_EI0(best_individuals):
    for ind in best_individuals:
        print(f"The best individual's EI0 is: {ind.EI0}", flush=True)


def preprocess_callback_func(individuals, EI0s):
    for ind, EI0 in zip(individuals, EI0s):
        ind.EI0 = EI0


def evaluate_EI0s(
    individuals_str,
    toolbox,
    theta_true,
    FL2,
    EI0_guess,
    theta_guess,
    S,
):
    EI0s = [None] * len(individuals_str)

    callables = compile_individuals(toolbox, individuals_str)
    for i, ind in enumerate(callables):
        EI0s[i] = tune_EI0(ind, toolbox, theta_true, FL2, EI0_guess, theta_guess, S)

    return EI0s


def stgp_elastica(output_path=None):
    global residual_formulation
    regressor_params, config_file_data = load_config_data("elastica.yaml")
    thetas_train, thetas_val, thetas_test, Fs_train, Fs_val, Fs_test = load_dataset(
        data_path, "csv"
    )

    # TODO: how can we extract these numbers from the dataset (especially length)?
    mesh, _ = generate_line_mesh(num_nodes=NUM_NODES, L=LENGTH)
    S = build_complex_from_mesh(mesh)
    S.get_hodge_star()

    x_all, y_all = get_positions_from_angles((thetas_train, thetas_val, thetas_test))

    theta_in_all = get_angles_initial_guesses(x_all, y_all)

    residual_formulation = config_file_data["gp"]["residual_formulation"]

    if residual_formulation:
        print("Using residual formulation.")
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainD0, C.CochainD0], C.CochainD0)
    else:
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainD0, C.CochainD0], float)

    # add internal cochain as a terminal
    internal_vec = np.ones(S.num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0.0
    internal_vec[-1] = 0.0
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)
    pset.addTerminal(internal_coch, C.CochainP0, name="int_coch")
    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1.0, float, name="-1.")
    pset.addTerminal(2.0, float, name="2.")

    pset.renameArguments(ARG0="theta")
    pset.renameArguments(ARG1="FL2_EI0")

    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])

    penalty = config_file_data["gp"]["penalty"]

    common_params = {"S": S, "penalty": penalty, "theta_in_all": theta_in_all}

    # define preprocess function to estimate EI0
    preprocess_func_args = {
        "theta_true": thetas_train[0, :],
        "FL2": Fs_train[0],
        "EI0_guess": 1.0,
        "theta_guess": theta_in_all["train"][0, :],
        "S": S,
    }

    preprocess_args = {
        "func": evaluate_EI0s,
        "func_args": preprocess_func_args,
        "callback": preprocess_callback_func,
    }

    # opt_string = "SubF(MulF(1/2, InnP0(CMulP0(int_coch, St1D1(cobD0(theta))),
    # CMulP0(int_coch, St1D1(cobD0(theta))))), InnD0(FL2_EI0, SinD0(theta)))"
    # seed = [opt_string]

    gpsr = gps.GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=score,
        predict_func=partial(predict, y=Fs_test),
        print_log=True,
        common_data=common_params,
        seed_str=None,
        save_best_individual=True,
        save_train_fit_history=True,
        output_path=output_path,
        preprocess_args=preprocess_args,
        custom_logger=print_EI0,
        num_cpus=1,
        **regressor_params,
    )

    start = time.perf_counter()

    gpsr.fit(X=thetas_train, y=Fs_train, X_val=thetas_val, y_val=Fs_val)

    # -- PLOTS --
    theta_pred_test = gpsr.predict(thetas_test)
    # reconstruct (x,y) pred and true
    x_pred, y_pred = get_positions_from_angles((theta_pred_test,))
    x_true, y_true = get_positions_from_angles((thetas_test,))
    dim = len(x_pred[0])

    plt.figure(1, figsize=(12, 4))
    _, axes = plt.subplots(1, dim, num=1)
    for i in range(dim):
        # plot the results
        axes[i].scatter(x_pred[0][i], y_pred[0][i], c="r", label="Predicted")
        axes[i].plot(x_true[0][i], y_true[0][i], c="b", label="True")
        axes[i].grid()
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].legend()

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

    stgp_elastica(output_path)
