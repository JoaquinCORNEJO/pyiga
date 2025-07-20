from . import *
import time
from typing import Callable, Union, Dict


class nonlinsolver:

    def __init__(
        self,
        max_iters: int = 20,
        outer_tolerance: float = 1e-6,
        inner_tolerance: float = 1e-8,
    ):
        self._max_iters = max_iters
        self._outer_tolerance = outer_tolerance
        self._inner_tolerance = inner_tolerance
        self._safeguard = 1e-14

    def solve(
        self,
        inital_guess: np.ndarray,
        compute_resisual: Callable,
        solve_linearization: Callable,
        anderson_iters: int = 1,
        inner_tolerance: Union[None, Callable] = None,
        inner_tolerance_args: Union[None, Dict] = None,
        linsolv_args: Dict = {},
        residual_args: Dict = {},
    ):
        "Solver of the nonlinear problem res(x) = 0."

        assert anderson_iters >= 0, "History size m must be >= 0"
        assert np.ndim(inital_guess) == 1, "Initial guess x0 must be a 1D array"
        auto_inner_tolerance = callable(inner_tolerance)
        norm_incr, norm_solu = 1.0, 1.0

        solution = np.copy(inital_guess)
        G_history = np.zeros((solution.shape, anderson_iters))
        R_history = np.zeros((solution.shape, anderson_iters))
        nonlinear_residual_list = []
        nonlinear_time_list = []
        nonlinear_rate_list = []
        linear_tolerance_list = []
        solution_history_list = {}

        start = time.process_time()
        for iteration in range(self._max_iters):

            residual, linsolv_args_extra = compute_resisual(
                solution, args=residual_args
            )
            norm_residual = np.linalg.norm(residual)
            print(f"Nonlinear error: {norm_residual:.3e}")
            nonlinear_residual_list.append(norm_residual)
            solution_history_list[f"noniter_{iteration}"] = np.copy(solution)

            finish = time.process_time()
            nonlinear_time_list.append(finish - start)

            if iteration == 0:
                norm_residual_ref = norm_residual
            else:
                if norm_residual <= max(
                    self._safeguard, self._outer_tolerance * norm_residual_ref
                ):
                    break
                if norm_incr <= max(self._safeguard, self._outer_tolerance * norm_solu):
                    break

            if auto_inner_tolerance and iteration > 0:
                inner_tolerance_val = inner_tolerance(
                    norm_residual,
                    norm_residual_old,
                    inner_tolerance_old,
                    inner_tolerance_args,
                )
                norm_residual_old = norm_residual
                inner_tolerance_old = inner_tolerance_val
            else:
                inner_tolerance_val = self._inner_tolerance
            linear_tolerance_list.append(inner_tolerance_val)

            incr_k = solve_linearization(
                residual, args=linsolv_args | linsolv_args_extra
            )

            newx_k = solution + incr_k
            if iteration < anderson_iters:
                R_history[:, iteration] = incr_k
                G_history[:, iteration] = newx_k
            else:
                R_history = np.roll(R_history, -1, axis=1)
                G_history = np.roll(G_history, -1, axis=1)
                R_history[:, -1] = incr_k
                G_history[:, -1] = newx_k

            if anderson_iters > 0 and iteration >= anderson_iters:
                A = R_history[:, 1:] - R_history[:, [0]]
                b = -R_history[:, 0]
                try:
                    c = np.linalg.lstsq(A, b, rcond=None)[0]
                    alpha = np.zeros(anderson_iters)
                    alpha[0] = 1.0 - np.sum(c)
                    alpha[1:] = c
                    incr_k = G_history @ alpha - solution
                except np.linalg.LinAlgError:
                    print(
                        "LinAlgError in Anderson acceleration, fallback to simple increment"
                    )

            # Update active control points
            solution += incr_k
            norm_incr = np.linalg.norm(incr_k)
            norm_solu = np.linalg.norm(solution)
            nonlinear_rate_list.append(norm_incr / norm_solu)

        return incr_k
