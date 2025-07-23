from . import *
from typing import Callable, Union, Dict, List
import time


class nonlinsolver:

    _safeguard = 1e-14
    _maxiters_linesearch = 3
    _anderson_history_size = 3
    _tiny_increment = 1e-4

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
    ):
        self._allow_acceleration = allow_fixed_point_acceleration
        self._allow_linesearch = allow_linesearch

    def _compute_anderson_step(self, R_hist: List, G_hist: List, dotfun: Callable):
        m = len(R_hist)
        mat = np.zeros((m - 1, m - 1))
        vec = np.zeros(m - 1)
        for i in range(1, m):
            Acol = R_hist[i] - R_hist[0]
            vec[i - 1] = -dotfun(Acol, R_hist[0])
            mat[i - 1, i - 1] = dotfun(Acol, Acol)
            for j in range(i + 1, m):
                Arow = R_hist[j] - R_hist[0]
                mat[i - 1, j - 1] = dotfun(Acol, Arow)
                mat[j - 1, i - 1] = mat[i - 1, j - 1]
        beta = np.linalg.lstsq(mat, vec, rcond=None)[0]
        alpha = np.zeros(m)
        alpha[0] = 1.0 - np.sum(beta)
        alpha[1:] = beta
        increment = alpha[0] * G_hist[0]
        for i in range(1, m):
            increment += alpha[i] * G_hist[i]
        return increment

    def _secant_line_search(
        self,
        sol_current: np.ndarray,
        incr_current: np.ndarray,
        external_force: np.ndarray,
        residual: Callable,
        update_variables: Callable,
        **res_args,
    ):

        sol_copy = deepcopy(sol_current)
        res_args_copy = deepcopy(res_args)

        def fun(alpha: float):
            update_variables(sol_copy, alpha * incr_current, **res_args_copy)
            output_value = (
                0.5
                * np.linalg.norm(residual(sol_copy, external_force, **res_args_copy)[0])
                ** 2
            )
            update_variables(sol_copy, -alpha * incr_current, **res_args_copy)
            return output_value

        def dersfun(alpha: float):
            return (
                fun(alpha + self._tiny_increment) - fun(alpha - self._tiny_increment)
            ) / (2 * self._tiny_increment)

        alpha0, alpha1, alpha2 = self._tiny_increment, 1.0, 1.0  # Initial guesses
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
        solution: np.ndarray,
        external_force: np.ndarray,
        compute_residual: Callable,
        solve_linearization: Callable,
        #
        select_inner_tolerance: Union[None, Callable] = None,
        update_variables: Union[None, Callable] = None,
        dotproduct: Union[None, Callable] = None,
        #
        residual_args: Dict = {},
        linsolv_args: Dict = {},
        inner_tolerance_args: Dict = {},
        save_information: bool = False,
    ):
        "Solver of the nonlinear problem res(x) = 0."

        def default_update_variables(x_curr, incr_curr, **kwargs):
            assert isinstance(x_curr, np.ndarray), "Define a numpy array"
            assert isinstance(incr_curr, np.ndarray), "Define a numpy array"
            x_curr += incr_curr
            return deepcopy(x_curr), deepcopy(kwargs)

        assert callable(compute_residual), "Define a function"
        assert callable(solve_linearization), "Define a function"

        update_variables = (
            default_update_variables if update_variables is None else update_variables
        )
        assert callable(update_variables), "Define a function"

        dotproduct = np.dot if dotproduct is None else dotproduct
        assert callable(dotproduct), "Define a function"

        # Variables for inner and outer tolerance
        auto_inner_tolerance = callable(select_inner_tolerance)
        norm_incr, norm_solu = 1.0, 1.0
        norm_residual_old = None
        inner_tolerance_old = None
        inner_tolerance = self._inner_tolerance

        G_history: List = []
        R_history: List = []

        # Extra data
        nonlinear_residual_list = []
        solution_history_list = {}
        nonlinear_time_list = []
        nonlinear_rate_list = []
        linear_tolerance_list = []

        start = time.process_time()
        for iteration in range(self._max_iters):

            residual, mf_args = compute_residual(
                solution, external_force, **residual_args
            )

            norm_residual = np.linalg.norm(residual)
            print(f"Nonlinear error: {norm_residual:.3e}")
            if save_information:
                nonlinear_residual_list.append(norm_residual)
                solution_history_list[f"noniter_{iteration}"] = np.copy(solution)

            finish = time.process_time()
            if save_information:
                nonlinear_time_list.append(finish - start)

            if iteration == 0:
                norm_residual_ref = norm_residual
            if norm_residual <= max(
                self._safeguard, self._outer_tolerance * norm_residual_ref
            ):
                break
            if norm_incr <= max(self._safeguard, self._outer_tolerance * norm_solu):
                break

            if auto_inner_tolerance:
                inner_tolerance = select_inner_tolerance(
                    norm_residual,
                    norm_residual_old,
                    inner_tolerance_old,
                    inner_tolerance_args,
                )
                norm_residual_old = norm_residual
                inner_tolerance_old = inner_tolerance

            if save_information:
                linear_tolerance_list.append(inner_tolerance)

            kwargs = linsolv_args | {
                "mf_args": mf_args,
                "inner_tolerance": inner_tolerance,
            }
            incr_k: np.ndarray = solve_linearization(residual, **kwargs)

            if self._allow_acceleration:
                new_solution = solution + incr_k
                if len(R_history) == self._anderson_history_size:
                    R_history.pop(0)
                R_history.append(incr_k.copy())
                if len(G_history) == self._anderson_history_size:
                    G_history.pop(0)
                G_history.append(new_solution.copy())

                if (
                    len(R_history) == self._anderson_history_size
                    and len(G_history) == self._anderson_history_size
                    and self._anderson_history_size > 1
                ):
                    try:
                        incr_k = (
                            self._compute_anderson_step(
                                R_history, G_history, dotproduct
                            )
                            - solution
                        )

                    except np.linalg.LinAlgError:
                        print(
                            "LinAlgError in Anderson acceleration, fallback to simple increment"
                        )

            if self._allow_linesearch and iteration > 0:
                alpha = self._secant_line_search(
                    solution,
                    incr_k,
                    external_force,
                    compute_residual,
                    update_variables,
                    **residual_args,
                )
                incr_k *= alpha

            # Update active control points
            update_variables(solution, incr_k, **residual_args)

            # Compute norms for outer stopping criterion
            norm_incr = np.linalg.norm(incr_k)
            norm_solu = np.linalg.norm(solution)
            if save_information:
                nonlinear_rate_list.append(norm_incr / norm_solu)

        extra_args = dict(
            nonlinear_residual_list=nonlinear_residual_list,
            nonlinear_time_list=nonlinear_time_list,
            nonlinear_rate_list=nonlinear_rate_list,
            linear_tolerance_list=linear_tolerance_list,
            solution_history_list=solution_history_list,
        )

        return extra_args
