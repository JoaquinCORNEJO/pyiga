from .__init__ import *
from typing import List, Dict, Callable, Union, Tuple


class problem:
    def __init__(
        self, patch: singlepatch, boundary: boundary_condition, solver_args: dict
    ):
        self._add_solver_parameters(args=solver_args)
        self.part: singlepatch = patch
        self.nbvars: int = boundary.nbvars
        self.sp_free_ctrlpts, self.sp_constraint_ctrlpts = (
            boundary.select_nodes4solving()
        )
        self.sp_table_dirichlet: np.ndarray = boundary.table_dirichlet
        self._sp_boundary: boundary_condition = boundary
        self._linear_tolerance_list: List[float] = []
        self._nonlinear_rate_list: List[float] = []
        self._linear_residual_list: List[np.ndarray] = []
        self._nonlinear_residual_list: List[float] = []
        self._solution_history_list: Dict[str, np.ndarray] = {}

    def _add_solver_parameters(self, args: dict):
        self._linear_solver: str = args.get("solver", "GMRES")
        self._maxiters_linear: int = args.get("iters_linear", 100)
        self._tolerance_linear: float = args.get("tol_linear", 1e-8)
        self._maxiters_nonlinear: int = args.get("iters_nonlinear", 10)
        self._tolerance_nonlinear: float = args.get("tol_nonlinear", 1e-6)
        self._use_preconditioner: bool = args.get("use_preconditioner", True)
        self._update_preconditioner: bool = args.get("update_preconditioner", True)
        self._safeguard: float = 1e-14

    def _solve_linear_system(
        self,
        Afun: Callable,
        bvec: np.ndarray,
        Pfun: Union[None, Callable] = None,
        dotfun: Union[None, Callable] = None,
        cleanfun: Union[None, Callable] = None,
        dod: list = None,
        args: dict = {},
        max_iters: Union[None, int] = None,
        tolerance: Union[None, float] = None,
    ) -> Dict[str, np.ndarray]:
        if max_iters is None:
            max_iters = self._maxiters_linear
        if tolerance is None:
            tolerance = self._tolerance_linear
        solv: solver = solver(max_iters=max_iters, tolerance=tolerance)
        solver_methods = {"BICG": solv.BiCGSTAB, "CG": solv.CG, "GMRES": solv.GMRES}
        solve_method = solver_methods.get(self._linear_solver)
        if solve_method:
            if not self._use_preconditioner:
                Pfun = None
            output = solve_method(
                Afun,
                bvec,
                Pfun=Pfun,
                dotfun=dotfun,
                cleanfun=cleanfun,
                dod=dod,
                args=args,
            )
        else:
            raise ValueError(f"Unknown linear solver method: {self._linear_solver}")
        return output

    def _calculate_norm(
        self,
        u: np.ndarray,
        uders: Union[np.ndarray, None],
        det_jac: np.ndarray,
        weights: List[np.ndarray],
        norm_type: str = "l2",
    ) -> float:
        ndim = len(weights)
        shape = [len(wgt) for wgt in weights]
        u2_l2, u2_sh1 = 0.0, 0.0
        if norm_type.lower() in ["l2", "h1"]:
            u2_l2 += np.sum(u**2, axis=0)
        if norm_type.lower() in ["h1", "semih1"]:
            u2_sh1 += np.sum(uders**2, axis=(0, 1))

        u2_det_jac = np.reshape((u2_l2 + u2_sh1) * det_jac, tuple(shape), order="F")
        text = f"{','.join('abcdef'[:ndim])},{'abcdef'[:ndim]}->"
        return np.sqrt(np.einsum(text, *weights, u2_det_jac))

    def _compute_exact_values_for_norm(
        self,
        norm_args: dict,
        quadpts_phy: Union[np.ndarray, List[np.ndarray]],
        quadpts_param: List[np.ndarray],
        prepare_norm_computation: Callable,
        allow_spacetime: bool,
    ) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:

        norm_type = str(norm_args.get("type", "l2")).lower()
        u_exact, uders_exact = None, None

        # First method
        exact_fun = norm_args.get("exact_function", None)
        if callable(exact_fun):
            if isinstance(quadpts_phy, np.ndarray):
                quadpts_phy = [quadpts_phy]
            exact_args = norm_args.get("exact_args", {})
            assert isinstance(exact_args, dict), "Error type of extra args"
            extra_args = {"position": quadpts_phy[0]} | (
                {"time": quadpts_phy[1]} if allow_spacetime else {}
            )
            exact_args = extra_args | exact_args
            u_exact = np.atleast_2d(exact_fun(exact_args))
            if norm_type != "l2":
                exact_dersfun = norm_args.get("exact_function_ders", None)
                assert callable(exact_dersfun), "Try other norm"
                uders_exact = np.atleast_3d(exact_dersfun(exact_args))

        # Second method
        part_ref: Union[None, singlepatch] = norm_args.get("part_ref", None)
        time_ref: Union[None, singlepatch] = norm_args.get("time_ref", None)
        u_ref: Union[None, np.ndarray] = norm_args.get("u_ref", None)
        if (
            allow_spacetime
            and isinstance(part_ref, singlepatch)
            and isinstance(time_ref, singlepatch)
        ):
            assert isinstance(u_ref, np.ndarray), "Solution should be numpy array"
            u_exact, uders_exact = prepare_norm_computation(
                part_ref,
                time_ref,
                u_ref,
                knots_ref=quadpts_param,
                norm_type=norm_type,
            )[:2]
        elif isinstance(part_ref, singlepatch):
            assert isinstance(u_ref, np.ndarray), "Solution should be numpy array"
            u_exact, uders_exact = prepare_norm_computation(
                part_ref, u_ref, knots_ref=quadpts_param, norm_type=norm_type
            )[:2]

        if u_exact is None or (norm_type != "l2" and uders_exact is None):
            raise ValueError(
                "Exact solution or its derivatives are not properly defined."
            )

        return u_exact, uders_exact


class space_problem(problem):
    def __init__(
        self,
        patch: singlepatch,
        boundary: boundary_condition,
        solver_args: dict,
        allow_lumping: bool = False,
    ):
        super().__init__(patch, boundary, solver_args)
        self.allow_lumping = allow_lumping

    def _verify_fun_args(self, args: dict) -> dict:
        assert isinstance(
            args, dict
        ), "allowed extra arguments should be in dictionnary"
        args = {
            "position": self.part.qp_phy,
            "shape_quadpts": (self.part.nbqp_total,),
        } | args
        return args

    def _prepare_surface_force_computation(
        self, location: dict
    ) -> Tuple[List[quadrature_rule], np.ndarray, np.ndarray, List[int]]:
        assert self.part.ndim > 1, "Method for multivariate problem"
        output = self._sp_boundary._recognize_constraint([location])
        selected_nodes, table = sorted(output[0][0]), output[1][0, ...]
        idx_direction = np.where(table)[0]
        assert idx_direction.size == 1, "Force should be applied only on one side"
        dir_range = [i for i in range(self.part.ndim)]
        dir_range.pop(idx_direction[0])

        quadrature_rules = [self.part.quadrule_list[_] for _ in dir_range]
        inpts = [quadrature_rules, self.part.ctrlpts[:, selected_nodes]]
        jac = bspline_operations.eval_jacobien(*inpts)
        quadpts_phy = bspline_operations.interpolate_meshgrid(*inpts)
        return quadrature_rules, jac, quadpts_phy, selected_nodes

    def assemble_surface_force(
        self, fun: Callable, location: dict, args: dict = {}
    ) -> np.ndarray:
        """Computes the surface foce over the boundary of a geometry.
        The surffun is a Neumann like function, ie, in transfer heat q = -(k grad(T)).normal
        and in elasticity t = sigma.normal
        """
        quadrature_rules, jac, quadpts_phy, selected_nodes = (
            self._prepare_surface_force_computation(location)
        )
        args = args | {"position": quadpts_phy}
        fun_args = np.atleast_2d(fun(args))
        prop = np.zeros_like(fun_args)
        for i in range(np.size(jac, axis=2)):
            if self.part.ndim == 2:
                v1 = jac[:, 0, i]
                dsurf = np.linalg.norm(v1)
            elif self.part.ndim == 3:
                v1 = jac[:, 0, i]
                v2 = jac[:, 1, i]
                v3 = np.cross(v1, v2)
                dsurf = np.linalg.norm(v3)
            prop[:, i] = fun_args[:, i] * dsurf
        nr = np.size(prop, axis=0)
        array_out = np.zeros((nr, self.part.nbctrlpts_total))
        array_out[:, selected_nodes] = bspline_operations.assemble_scalar_u_force(
            quadrature_rules, prop
        )
        if nr == 1:
            array_out = np.ravel(array_out)
        return array_out

    def assemble_volumetric_force(self, fun: Callable, args: dict = {}) -> np.ndarray:
        "Computes the volume force over a geometry."
        args = self._verify_fun_args(args)
        prop = np.atleast_2d(fun(args)) * self.part.det_jac
        nr = np.size(prop, axis=0)
        array_out = bspline_operations.assemble_scalar_u_force(
            self.part.quadrule_list, prop
        )
        if nr == 1:
            array_out = np.ravel(array_out)
        return array_out

    def _prepare_norm_computation(
        self,
        part_ref: singlepatch,
        u_ref: np.ndarray,
        knots_ref: List[Union[None, np.ndarray]],
        norm_type: str = "l2",
    ) -> Tuple[
        np.ndarray,
        Union[np.ndarray, None],
        List[np.ndarray],
        np.ndarray,
        np.ndarray,
        List[np.ndarray],
    ]:
        parametric_position, parametric_weights = [], []
        for i in range(part_ref.ndim):
            quadrule = gauss_quadrature(
                part_ref.degree[i], part_ref.knotvector[i], quad_args={"type": "leg"}
            )
            quadrule.export_quadrature_rules()
            parametric_position.append(
                quadrule.quadpts
                if not isinstance(knots_ref[i], np.ndarray)
                else np.copy(knots_ref[i])
            )
            parametric_weights.append(quadrule._parametric_weights)

        quadpts_phy = bspline_operations.interpolate_meshgrid(
            part_ref.quadrule_list, part_ref.ctrlpts, parametric_position
        )
        jacobien = bspline_operations.eval_jacobien(
            part_ref.quadrule_list, part_ref.ctrlpts, parametric_position
        )
        det_jac, inv_jac = eval_inverse_and_determinant(jacobien)
        u_interp = bspline_operations.interpolate_meshgrid(
            part_ref.quadrule_list, np.atleast_2d(u_ref), parametric_position
        )
        uders_interp = None
        if norm_type.lower() != "l2":
            derstemp = bspline_operations.eval_jacobien(
                part_ref.quadrule_list, np.atleast_2d(u_ref), parametric_position
            )
            uders_interp = np.atleast_3d(
                np.einsum("ijl,jkl->ikl", derstemp, inv_jac, optimize=True)
            )

        return (
            u_interp,
            uders_interp,
            parametric_position,
            quadpts_phy,
            det_jac,
            parametric_weights,
        )

    def norm_of_error(
        self, u_ctrlpts: np.ndarray, norm_args: dict
    ) -> Tuple[float, float]:
        """Computes the norm L2 or H1 of the error. The exactfun is the function of the exact solution.
        and u_ctrlpts is the field at the control points. We compute the integral using Gauss Quadrature
        whether or not the default quadrature is weighted quadrature.
        """

        norm_type = str(norm_args.get("type", "l2")).lower()
        if all(norm != norm_type for norm in ["l2", "h1", "semih1"]):
            raise Warning("Unknown norm")

        # Compute u interp
        (
            u_interp,
            uders_interp,
            parametric_position,
            quadpts_phy,
            det_jac,
            parametric_weights,
        ) = self._prepare_norm_computation(
            self.part,
            u_ctrlpts,
            knots_ref=[None] * self.part.ndim,
            norm_type=norm_type,
        )

        # Compute u exact
        u_exact, uders_exact = self._compute_exact_values_for_norm(
            norm_args,
            quadpts_phy,
            parametric_position,
            self._prepare_norm_computation,
            allow_spacetime=False,
        )

        if norm_type == "l2":
            abserror = self._calculate_norm(
                u_exact - u_interp,
                None,
                det_jac,
                parametric_weights,
                norm_type=norm_type,
            )
            tmp2 = self._calculate_norm(
                u_exact, None, det_jac, parametric_weights, norm_type=norm_type
            )
        else:
            abserror = self._calculate_norm(
                u_exact - u_interp,
                uders_exact - uders_interp,
                det_jac,
                parametric_weights,
                norm_type=norm_type,
            )
            tmp2 = self._calculate_norm(
                u_exact, uders_exact, det_jac, parametric_weights, norm_type=norm_type
            )

        relerror = abserror / tmp2 if tmp2 != 0 else abserror
        if tmp2 == 0:
            print("Warning: Dividing by zero")

        return abserror, relerror

    def compute_L2projection(self, u_at_quadpts: np.ndarray) -> np.ndarray:

        def mass(x_in, _):
            x_out = bspline_operations.compute_mf_scalar_u_v(
                self.part.quadrule_list, self.part.det_jac, x_in, allow_lumping=False
            )
            return x_out

        array_in = self.assemble_volumetric_force(lambda _: u_at_quadpts, {})
        array_in = np.atleast_2d(array_in)
        nm = np.size(array_in, axis=0)

        fastdiag = fastdiagonalization()
        fastdiag.compute_space_eigendecomposition(
            self.part.quadrule_list, np.zeros((1, 3, 2))
        )
        fastdiag.add_free_controlpoints(
            [np.arange(self.part.nbctrlpts_total, dtype=int)]
        )
        fastdiag.update_space_eigenvalues(scalar_coefs=(1, 0))

        array_out = np.zeros_like(array_in)
        for i in range(nm):
            output = self._solve_linear_system(
                mass, array_in[i, :], Pfun=fastdiag.apply_scalar_preconditioner
            )
            array_out[i, :] = output["sol"]
        if nm == 1:
            array_out = np.ravel(array_out)
        return array_out


class spacetime_problem(problem):
    def __init__(
        self,
        patch: singlepatch,
        time_patch: singlepatch,
        boundary: boundary_condition,
        solver_args: dict,
    ):
        super().__init__(patch, boundary, solver_args)
        assert time_patch.ndim == 1, "Time is only one dimension"
        self.time: singlepatch = time_patch
        self._quadrature_list: List[quadrature_rule] = (
            self.part.quadrule_list + self.time.quadrule_list
        )
        self._propagate_constraint_in_time()
        return

    def _propagate_constraint_in_time(self):
        nbctrlpts_sp = self.part.nbctrlpts_total
        nbctrlpts_tm = self.time.nbctrlpts_total

        ctrlpts_dirichlet = [set(range(nbctrlpts_sp)) for _ in range(self.nbvars)]
        for var in range(self.nbvars):
            ctrlpts_dirichlet_space = set(self.sp_constraint_ctrlpts[var])
            for j in range(1, nbctrlpts_tm):
                for i in ctrlpts_dirichlet_space:
                    ctrlpts_dirichlet[var].add(i + j * nbctrlpts_sp)

        free_nodes = [
            sorted(
                set(range(nbctrlpts_sp * nbctrlpts_tm)).difference(
                    ctrlpts_dirichlet[var]
                )
            )
            for var in range(self.nbvars)
        ]
        constraint_nodes = [
            sorted(ctrlpts_dirichlet[var]) for var in range(self.nbvars)
        ]

        self.sptm_constraint_ctrlpts = constraint_nodes
        self.sptm_free_ctrlpts = free_nodes

    def _verify_fun_args(self, args: dict) -> dict:
        assert isinstance(
            args, dict
        ), "allowed extra arguments should be in dictionnary"
        args = {
            "position": self.part.qp_phy,
            "shape_quadpts": (self.part.nbqp_total * self.time.nbqp_total,),
            "time": np.ravel(self.time.qp_phy),
        } | args
        return args

    def assemble_volumetric_force(self, fun: Callable, args: dict = {}) -> np.ndarray:
        "Computes the volume force over a geometry"
        args = self._verify_fun_args(args)
        prop = np.atleast_2d(fun(args)) * np.kron(self.time.det_jac, self.part.det_jac)
        nr = np.size(prop, axis=0)
        array_out = bspline_operations.assemble_scalar_u_force(
            self._quadrature_list, prop
        )
        if nr == 1:
            array_out = np.ravel(array_out)
        return array_out

    def _prepare_norm_computation(
        self,
        part_ref: singlepatch,
        time_ref: singlepatch,
        u_ref: np.ndarray,
        knots_ref: List[Union[None, np.ndarray]],
        norm_type: str = "l2",
    ) -> Tuple[
        np.ndarray,
        Union[np.ndarray, None],
        List[np.ndarray],
        List[np.ndarray],
        np.ndarray,
        List[np.ndarray],
    ]:
        parametric_position, parametric_weights = [], []
        # For space variables
        for i in range(part_ref.ndim):
            quadrule = gauss_quadrature(
                part_ref.degree[i], part_ref.knotvector[i], quad_args={"type": "leg"}
            )
            quadrule.export_quadrature_rules()
            parametric_position.append(
                quadrule.quadpts
                if not isinstance(knots_ref[i], np.ndarray)
                else np.copy(knots_ref[i])
            )
            parametric_weights.append(quadrule._parametric_weights)

        # For time variables
        quadrule = gauss_quadrature(
            time_ref.degree[0], time_ref.knotvector[0], quad_args={"type": "leg"}
        )
        quadrule.export_quadrature_rules()
        parametric_position.append(
            quadrule.quadpts
            if not isinstance(knots_ref[-1], np.ndarray)
            else np.copy(knots_ref[-1])
        )
        parametric_weights.append(quadrule._parametric_weights)

        quadpts_phy_space = bspline_operations.interpolate_meshgrid(
            part_ref.quadrule_list, part_ref.ctrlpts, list(parametric_position[:-1])
        )
        jacobien_space = bspline_operations.eval_jacobien(
            part_ref.quadrule_list, part_ref.ctrlpts, list(parametric_position[:-1])
        )
        det_jac_space, inv_jac_space = eval_inverse_and_determinant(jacobien_space)

        quadpts_phy_time = np.ravel(
            bspline_operations.interpolate_meshgrid(
                time_ref.quadrule_list, time_ref.ctrlpts, list(parametric_position[-1:])
            )
        )
        jacobien_time = bspline_operations.eval_jacobien(
            time_ref.quadrule_list, time_ref.ctrlpts, list(parametric_position[-1:])
        )
        det_jac_time, inv_jac_time = eval_inverse_and_determinant(jacobien_time)

        u_interp = bspline_operations.interpolate_meshgrid(
            part_ref.quadrule_list + time_ref.quadrule_list,
            np.atleast_2d(u_ref),
            parametric_position,
        )
        uders_interp = None
        if norm_type != "l2":
            nm = np.shape(u_interp)[0]
            derstmp = bspline_operations.eval_jacobien(
                part_ref.quadrule_list + time_ref.quadrule_list,
                np.atleast_2d(u_ref),
                parametric_position,
            )
            derstmp_reshaped = np.reshape(
                derstmp,
                newshape=(nm, part_ref.ndim + 1, -1, len(parametric_position[-1])),
                order="F",
            )
            uders_interp = np.zeros_like(derstmp_reshaped)
            uders_interp[:, :-1, :, :] = np.einsum(
                "mikp,ijk->mjkp",
                derstmp_reshaped[:, :-1, :, :],
                inv_jac_space,
                optimize=True,
            )
            uders_interp[:, -1, :, :] = np.einsum(
                "mkp,p->mkp",
                derstmp_reshaped[:, -1, :, :],
                np.ravel(inv_jac_time),
                optimize=True,
            )
            uders_interp = np.reshape(
                uders_interp, newshape=(nm, self.part.ndim + 1, -1), order="F"
            )

        quadpts_phy = [quadpts_phy_space, quadpts_phy_time]
        det_jac = (np.kron(det_jac_time, det_jac_space),)

        return (
            u_interp,
            uders_interp,
            parametric_position,
            quadpts_phy,
            det_jac,
            parametric_weights,
        )

    def norm_of_field(self, u_ctrlpts: np.ndarray, norm_type: str = "l2") -> float:
        output = self._prepare_norm_computation(
            self.part,
            self.time,
            u_ctrlpts,
            knots_ref=[None for _ in range(self.part.ndim + 1)],
            norm_type=norm_type,
        )
        u_interp, uders_interp, _, _, _, det_jac, parametric_weights = output
        norm = self._calculate_norm(
            u_interp, uders_interp, det_jac, parametric_weights, norm_type=norm_type
        )
        return norm

    def norm_of_error(
        self, u_ctrlpts: np.ndarray, norm_args: dict
    ) -> Tuple[float, float]:
        """Computes the norm L2 or H1 of the error. The exactfun is the function of the exact solution.
        and u_ctrlpts is the field at the control points. We compute the integral using Gauss Quadrature
        whether or not the default quadrature is weighted quadrature.
        """

        norm_type = str(norm_args.get("type", "l2")).lower()
        if all(norm != norm_type for norm in ["l2", "h1", "semih1"]):
            raise Warning("Unknown norm")

        # Compute u interp
        output = self._prepare_norm_computation(
            self.part,
            self.time,
            u_ctrlpts,
            knots_ref=[None for _ in range(self.part.ndim + 1)],
            norm_type=norm_type,
        )
        (
            u_interp,
            uders_interp,
            parametric_position,
            quadpts_phy,
            det_jac,
            parametric_weights,
        ) = output

        # Compute u exact
        u_exact, uders_exact = self._compute_exact_values_for_norm(
            norm_args,
            quadpts_phy,
            parametric_position,
            self._prepare_norm_computation,
            allow_spacetime=True,
        )

        if norm_type == "l2":
            abserror = self._calculate_norm(
                u_exact - u_interp,
                None,
                det_jac,
                parametric_weights,
                norm_type=norm_type,
            )
            tmp2 = self._calculate_norm(
                u_exact, None, det_jac, parametric_weights, norm_type=norm_type
            )
        else:
            abserror = self._calculate_norm(
                u_exact - u_interp,
                uders_exact - uders_interp,
                det_jac,
                parametric_weights,
                norm_type=norm_type,
            )
            tmp2 = self._calculate_norm(
                u_exact, uders_exact, det_jac, parametric_weights, norm_type=norm_type
            )

        relerror = abserror / tmp2 if tmp2 != 0 else abserror
        if tmp2 == 0:
            print("Warning: Dividing by zero")

        return abserror, relerror
