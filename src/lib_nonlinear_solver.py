from . import *
from typing import Callable, Union, Dict, List
import time


class nonlinsolver:

    _safeguard = 1e-14
    _maxiters_linesearch = 4
    _anderson_history_size = 3

    def __init__(
        self,
        maxiters: int = 20,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-8,
    ):
        self._max_iters = maxiters
        self._outer_tolerance = tolerance
        self._inner_tolerance = linear_solver_tolerance
        self.modify_solver()

    def modify_solver(
        self,
        allow_fixed_point_acceleration: bool = False,
        allow_linesearch: bool = False,
        **args,
    ):
        self._allow_acceleration = allow_fixed_point_acceleration
        self._allow_linesearch = allow_linesearch
        self._maxiters_linesearch = args.get("iters_linesearch", 4)
        self._anderson_history_size = args.get("anderson_size", 3)

    def _compute_anderson_step(self, R_hist: List, G_hist: List, dotfun: Callable):
        m = len(R_hist)
        mat = np.zeros((m - 1, m - 1))
        vec = np.zeros(m - 1)
        alpha = np.zeros(m)
        for i in range(1, m):
            Acol = R_hist[i] - R_hist[0]
            vec[i - 1] = -dotfun(Acol, R_hist[0])
            mat[i - 1, i - 1] = dotfun(Acol, Acol)
            for j in range(i + 1, m):
                Arow = R_hist[j] - R_hist[0]
                mat[i - 1, j - 1] = dotfun(Acol, Arow)
                mat[j - 1, i - 1] = mat[i - 1, j - 1]
        beta = np.linalg.lstsq(mat, vec, rcond=None)[0]
        alpha[0] = 1.0 - np.sum(beta)
        alpha[1:] = beta
        increment = alpha[0] * G_hist[0]
        for i in range(1, m):
            increment += alpha[i] * G_hist[i]
        return increment

    def _secant_line_search(
        self, xk: np.ndarray, pk: np.ndarray, F: Callable, args: Dict = {}
    ):
        def fun(alpha: float):
            return 0.5 * np.linalg.norm(F(xk + alpha * pk, args)[0]) ** 2

        def dersfun(alpha: float, h: float = 1e-4):
            return (fun(alpha + h) - fun(alpha - h)) / (2 * h)

        alpha0, alpha1, alpha2 = 1e-4, 1.0, 1.0  # Initial guesses
        for _ in range(self._maxiters_linesearch):
            dphi0 = dersfun(alpha0)
            dphi1 = dersfun(alpha1)
            denom = dphi1 - dphi0
            if abs(denom) < self._safeguard:
                break
            alpha2 = alpha1 - dphi1 * (alpha1 - alpha0) / denom
            if abs(alpha2 - alpha1) < self._outer_tolerance:
                return alpha2
            alpha0, alpha1 = alpha1, alpha2
        return alpha1  # or alpha2 if it's guaranteed to be defined

    def solve(
        self,
        primal_guess: np.ndarray,
        external_force: np.ndarray,
        compute_resisual: Callable,
        solve_linearization: Callable,
        #
        compute_inner_tolerance: Union[None, Callable] = None,
        dotproduct: Union[None, Callable] = None,
        #
        residual_args: Dict = {},
        linsolv_args: Dict = {},
        inner_tolerance_args: Dict = {},
        save_data=False,
    ):
        "Solver of the nonlinear problem res(x) = 0."

        dotproduct = np.dot if dotproduct is None else dotproduct
        assert callable(dotproduct), "Define a function"

        # Variables for inner and outer tolerance
        auto_inner_tolerance = callable(compute_inner_tolerance)
        norm_incr, norm_solu = 1.0, 1.0

        solution_k = np.copy(primal_guess)
        G_history: List = []
        R_history: List = []

        if save_data:
            nonlinear_residual_list = []
            solution_history_list = {}
            nonlinear_time_list = []
            nonlinear_rate_list = []
            linear_tolerance_list = []

        start = time.process_time()
        for iteration in range(self._max_iters):

            residual, linsolv_args_extra = compute_resisual(
                solution_k, external_force, **residual_args
            )

            norm_residual = np.linalg.norm(residual)
            print(f"Nonlinear error: {norm_residual:.3e}")
            if save_data:
                nonlinear_residual_list.append(norm_residual)
                solution_history_list[f"noniter_{iteration}"] = np.copy(solution_k)

            finish = time.process_time()
            if save_data:
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
                inner_tolerance = compute_inner_tolerance(
                    norm_residual,
                    norm_residual_old,
                    inner_tolerance_old,
                    inner_tolerance_args,
                )
                norm_residual_old = norm_residual
                inner_tolerance_old = inner_tolerance
            else:
                inner_tolerance = self._inner_tolerance
            if save_data:
                linear_tolerance_list.append(inner_tolerance)

            incr_k = solve_linearization(
                residual,
                args=linsolv_args
                | linsolv_args_extra
                | {"inner_tolerance": inner_tolerance},
            )

            if self._allow_acceleration:
                newx_k = solution_k + incr_k
                if iteration >= self._anderson_history_size:
                    R_history.pop(0)
                    G_history.pop(0)
                R_history.append(incr_k)
                G_history.append(newx_k)

                if iteration >= self._anderson_history_size:
                    try:
                        incr_k = (
                            self._compute_anderson_step(
                                R_history, G_history, dotproduct
                            )
                            - solution_k
                        )

                    except np.linalg.LinAlgError:
                        print(
                            "LinAlgError in Anderson acceleration, fallback to simple increment"
                        )

            if self._allow_linesearch and iteration > 0:
                alpha = self._secant_line_search(solution_k, incr_k, compute_resisual)
                incr_k = alpha * incr_k

            # Update active control points
            solution_k += incr_k
            norm_incr = np.linalg.norm(incr_k)
            norm_solu = np.linalg.norm(solution_k)
            if save_data:
                nonlinear_rate_list.append(norm_incr / norm_solu)

        extra_args = dict(
            nonlinear_residual_list=nonlinear_residual_list,
            nonlinear_time_list=nonlinear_time_list,
            nonlinear_rate_list=nonlinear_rate_list,
            linear_tolerance_list=linear_tolerance_list,
            solution_history_list=solution_history_list,
        )

        return (solution_k, extra_args) if save_data else solution_k
