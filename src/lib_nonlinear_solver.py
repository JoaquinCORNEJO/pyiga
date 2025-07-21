from . import *
import time
from typing import Callable, Union, Dict, List


class nonlinsolver:
    def __init__(
        self,
        max_iters: int = 20,
        outer_tolerance: float = 1e-6,
        inner_tolerance: float = 1e-8,
        allow_acceleration: bool = False,
        allow_linesearch: bool = False,
    ):
        self._max_iters = max_iters
        self._outer_tolerance = outer_tolerance
        self._inner_tolerance = inner_tolerance
        self._allow_acceleration = allow_acceleration
        self._allow_linesearch = allow_linesearch
        self._max_iters_linesearch = 4
        self._safeguard = 1e-14

    def _compute_anderson_step(self, R_hist: List, G_hist: List, dotfun: Callable):
        # mat = np.array([R_history[i] - R_history[0] for i in range(1, anderson_iters)]).T
        # vec = -R_history[0]
        # beta = np.linalg.lstsq(mat, vec, rcond=None)[0]
        # alpha = np.zeros(anderson_iters)
        # alpha[0] = 1.0 - np.sum(beta)
        # alpha[1:] = beta
        # incr_k = np.array(G_history).T @ alpha
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
        for _ in range(self._max_iters_linesearch):
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
        inital_guess: np.ndarray,
        compute_resisual: Callable,
        solve_linearization: Callable,
        anderson: int = 1,
        #
        compute_inner_tolerance: Union[None, Callable] = None,
        dotfun: Union[None, Callable] = None,
        #
        inner_tolerance_args: Dict = {},
        linsolv_args: Dict = {},
        residual_args: Dict = {},
    ):
        "Solver of the nonlinear problem res(x) = 0."

        dotfun = np.dot if dotfun is None else dotfun
        assert callable(dotfun), "Define a function"
        assert anderson > 0, "History size must be positive"

        # Variables for inner and outer tolerance
        auto_inner_tolerance = callable(compute_inner_tolerance)
        norm_incr, norm_solu = 1.0, 1.0

        solution_k = np.copy(inital_guess)
        G_history: List = []
        R_history: List = []
        nonlinear_residual_list = []
        nonlinear_time_list = []
        nonlinear_rate_list = []
        linear_tolerance_list = []
        solution_history_list = {}

        start = time.process_time()
        for iteration in range(self._max_iters):

            residual, linsolv_args_extra = compute_resisual(
                solution_k, args=residual_args
            )

            norm_residual = np.linalg.norm(residual)
            print(f"Nonlinear error: {norm_residual:.3e}")
            nonlinear_residual_list.append(norm_residual)
            solution_history_list[f"noniter_{iteration}"] = np.copy(solution_k)

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
            linear_tolerance_list.append(inner_tolerance)

            incr_k = solve_linearization(
                residual,
                args=linsolv_args
                | linsolv_args_extra
                | {"inner_tolerance": inner_tolerance},
            )

            if self._allow_acceleration:
                newx_k = solution_k + incr_k
                if iteration >= anderson:
                    R_history.pop(0)
                    G_history.pop(0)
                R_history.append(incr_k)
                G_history.append(newx_k)

                if iteration >= anderson:
                    try:
                        incr_k = (
                            self._compute_anderson_step(R_history, G_history, dotfun)
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
            nonlinear_rate_list.append(norm_incr / norm_solu)

        extra_args = dict(
            nonlinear_residual_list = nonlinear_residual_list,
            nonlinear_time_list = nonlinear_time_list,
            nonlinear_rate_list = nonlinear_rate_list,
            linear_tolerance_list = linear_tolerance_list,
            solution_history_list = solution_history_list
        )
        return solution_k, extra_args
