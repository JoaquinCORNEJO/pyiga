from .lib_job import *
from typing import Union, Callable, List, Tuple
import time


class heat_transfer_problem(space_problem):
    def __init__(
        self,
        material: heat_transfer_mat,
        patch: singlepatch,
        boundary: boundary_condition,
        solver_args: dict = {},
    ):
        super().__init__(
            patch, boundary, solver_args, allow_lumping=False
        )  # We consider no lumping in heat transfer
        self.material = material
        self.preconditioner = self.activate_preconditioner()
        self._capacity_property: Union[Callable, None] = None
        self._conductivity_property: Union[Callable, None] = None
        self._scalar_mean_capacity: List[float] = [1]
        self._scalar_mean_conductivity: List[np.ndarray] = [np.ones(self.part.ndim)]

    def activate_preconditioner(self) -> fastdiagonalization:
        fastdiag = fastdiagonalization()
        fastdiag.add_free_controlpoints(self.sp_free_ctrlpts)
        fastdiag.compute_space_eigendecomposition(
            self.part.quadrule_list, self.sp_table_dirichlet
        )
        return fastdiag

    def clear_properties(self):
        self._capacity_property = None
        self._conductivity_property = None

    def compute_mf_capacity(
        self, array_in: np.ndarray, mf_args: dict = {}
    ) -> np.ndarray:
        mf_args = self._verify_fun_args(mf_args)
        mf_args = {"temperature": np.ones(self.part.nbqp_total)} | mf_args
        if self._capacity_property is None:
            self._capacity_property = (
                self.material.capacity(mf_args) * self.part.det_jac
            )
            self._scalar_mean_capacity = [np.mean(self._capacity_property)]
        array_out = bspline_operations.compute_mf_scalar_u_v(
            self.part.quadrule_list,
            self._capacity_property,
            array_in,
            allow_lumping=self.allow_lumping,
        )
        return array_out

    def compute_mf_conductivity(
        self, array_in: np.ndarray, mf_args: dict = {}
    ) -> np.ndarray:
        mf_args = self._verify_fun_args(mf_args)
        mf_args = {"temperature": np.ones(self.part.nbqp_total)} | mf_args
        if self._conductivity_property is None:
            self._conductivity_property = np.einsum(
                "ilk,lmk,jmk,k->ijk",
                self.part.inv_jac,
                self.material.conductivity(mf_args),
                self.part.inv_jac,
                self.part.det_jac,
                optimize=True,
            )
            self._scalar_mean_conductivity = [
                np.array(
                    [
                        np.mean(self._conductivity_property[i, i, :])
                        for i in range(self.part.ndim)
                    ]
                )
            ]
        array_out = bspline_operations.compute_mf_scalar_gradu_gradv(
            self.part.quadrule_list, self._conductivity_property, array_in
        )
        return array_out

    def interpolate_temperature(self, u_ctrlpts: np.ndarray) -> np.ndarray:
        u_interp = bspline_operations.interpolate_meshgrid(
            self.part.quadrule_list, np.atleast_2d(u_ctrlpts)
        )
        return np.ravel(u_interp, order="F")

    def _assemble_internal_force(
        self,
        temperature: np.ndarray,
        flux: np.ndarray,
        scalar_coefs: tuple,
        mf_args: dict = {},
    ) -> np.ndarray:
        assert (
            scalar_coefs is not None
        ), "Define contribution from capacity and conductivity"
        array_out = 0.0
        if scalar_coefs[0] != 0:
            array_out += scalar_coefs[0] * self.compute_mf_capacity(flux, mf_args)
        if scalar_coefs[1] != 0:
            array_out += scalar_coefs[1] * self.compute_mf_conductivity(
                temperature, mf_args
            )
        return array_out

    def _compute_residual(
        self, temperature: np.ndarray, external_force: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, dict]:

        flux: np.ndarray = kwargs.get("flux")
        scalar_coefs: tuple = kwargs.get("scalar_coefs")

        if self.update_properties:
            self.clear_properties()
        mf_args = {"temperature": self.interpolate_temperature(temperature)}
        residual = external_force - self._assemble_internal_force(
            temperature, flux, scalar_coefs, mf_args
        )
        clean_dirichlet(residual, self.sp_constraint_ctrlpts)
        return residual, mf_args

    def _solve_linearized_system(self, array_in: np.ndarray, **kwargs) -> np.ndarray:

        mf_args: dict = kwargs.get("mf_args")
        scalar_coefs: tuple = kwargs.get("scalar_coefs")
        inner_tolerance: float = kwargs.get("inner_tolerance")

        def compute_mf_tangent(array_in, args):
            array_out = 0.0
            if scalar_coefs[0] != 0:
                array_out += scalar_coefs[0] * self.compute_mf_capacity(array_in, args)
            if scalar_coefs[1] != 0:
                array_out += scalar_coefs[1] * self.compute_mf_conductivity(
                    array_in, args
                )
            return array_out

        if self._update_preconditioner:
            self.preconditioner.add_scalar_space_time_correctors(
                mass_corrector=self._scalar_mean_capacity,
                stiffness_corrector=self._scalar_mean_conductivity,
            )

        self.preconditioner.update_space_eigenvalues(scalar_coefs=scalar_coefs)
        output = super()._solve_linear_system(
            compute_mf_tangent,
            array_in,
            Pfun=self.preconditioner.apply_scalar_preconditioner,
            cleanfun=clean_dirichlet,
            dod=self.sp_constraint_ctrlpts,
            args=mf_args,
            tolerance=inner_tolerance,
        )
        self._linear_residual_list.append(output["res"])
        return output["sol"]

    def _update_transient_variables(self, temperature, increment, **kwargs):
        assert isinstance(temperature, np.ndarray), "Define a numpy object"
        assert isinstance(increment, np.ndarray), "Define a numpy object"
        # Get other variables
        flux = kwargs.get("flux", None)
        flux_factor = kwargs.get("flux_factor", None)
        # Update
        temperature += increment
        if flux is not None and flux_factor is not None:
            flux += increment * flux_factor
        kwargs.update({"flux": flux})

    def solve_heat_transfer(
        self, temperature_list: np.ndarray, external_force_list: np.ndarray, **kwargs
    ):

        allow_fixed_point_acceleration = kwargs.get(
            "allow_anderson_acceleration", False
        )
        allow_linesearch = kwargs.get("allow_line_search", False)

        # Decide if it is a linear or nonlinear problem
        self.update_properties = not (
            self.material._has_uniform_capacity
            and self.material._has_uniform_conductivity
        )

        nonlinsolv = nonlinsolver(
            maxiters=self._maxiters_nonlinear,
            tolerance=self._tolerance_nonlinear,
            linear_solver_tolerance=self._tolerance_linear,
        )
        nonlinsolv.modify_solver(
            allow_fixed_point_acceleration=allow_fixed_point_acceleration,
            allow_line_search=allow_linesearch,
        )

        if external_force_list.ndim == 1 and temperature_list.ndim == 1:

            print(f"Steady heat transfer solver")
            nonlinsolv.solve(
                temperature_list,
                external_force_list,
                self._compute_residual,
                self._solve_linearized_system,
                residual_args={"scalar_coefs": (0, 1)},
                linsolv_args={"scalar_coefs": (0, 1)},
            )

            return

        def predict_temperature(
            histtemp: List[np.ndarray],
            norder: int,
        ):
            assert len(histtemp) == norder, "Size problem."
            if norder == 1:
                return np.copy(histtemp[0])
            elif norder == 2:
                [y1, y2] = histtemp
                return (4 * y2 - y1) / 3
            elif norder == 3:
                [y1, y2, y3] = histtemp
                return (18 * y3 - 9 * y2 + 2 * y1) / 11
            elif norder == 4:
                [y1, y2, y3, y4] = histtemp
                return (48 * y4 - 36 * y3 + 16 * y2 - 3 * y1) / 25
            else:
                raise ValueError("Order not supported")

        def select_snapshots(i, y, norder):
            "Creates a list of previous solution values."
            return [y[:, i - k] for k in range(norder, 0, -1)]

        def select_parameter(norder):
            assert norder in [1, 2, 3, 4], "Order not supported"
            parameters = [1.0, 2.0 / 3.0, 6.0 / 11.0, 12.0 / 25.0]
            return parameters[norder - 1]

        # This is a transient problem
        type_solver = kwargs.get("type_solver", "alpha")
        assert type_solver in ["alpha", "bdf"], "Method unknown"

        if type_solver == "alpha":
            # Get variables for alpha (or theta) method
            time_list: Union[np.ndarray, list] = kwargs.get("time_list")
            alpha: float = kwargs.get("alpha", 1.0)
            assert all(
                x is not None for x in [time_list, alpha]
            ), "time_list or alpha are not defined"
            nsteps = len(time_list) - 1
        else:
            # Get variables for BDF (or theta) method
            tspan: Tuple[float] = kwargs.get("tspan")
            nsteps: int = kwargs.get("nsteps")
            norder: int = kwargs.get("norder", 1)
            assert all(
                x is not None for x in [tspan, nsteps, norder]
            ), "tspan or nsteps or norder are not defined"

        assert nsteps >= 1, "At least 1 step required"

        print(f"Transient heat transfer solver")

        # Get inactive control points
        constraint_ctrlpts = self.sp_constraint_ctrlpts

        # Initialize time and solution arrays
        dt = (
            time_list[1] - time_list[0]
            if type_solver == "alpha"
            else (tspan[1] - tspan[0]) / nsteps
        )
        v_n0 = np.zeros_like(temperature_list[:, 0])
        v_n0[constraint_ctrlpts[0]] = (
            temperature_list[constraint_ctrlpts[0], 1]
            - temperature_list[constraint_ctrlpts[0], 0]
        ) / dt

        residual, mf_args = self._compute_residual(
            temperature_list[:, 0],
            external_force_list[:, 0],
            flux=v_n0,
            scalar_coefs=(1, 1),
        )
        v_n0 += self._solve_linearized_system(
            residual, scalar_coefs=(1, 0), mf_args=mf_args
        )

        # Main loop to solve the ODE using the BDF method
        for i in range(1, nsteps + 1):

            if type_solver == "alpha":
                # Get delta time
                dt = time_list[i] - time_list[i - 1]
                # Predict current temperature
                dj_n1 = np.copy(temperature_list[:, i - 1])
            else:
                # Get current order and ensure it does not exceed norder
                currorder = min(i, norder)
                alpha = select_parameter(currorder)
                # Get values of last steps
                d_list = select_snapshots(i, temperature_list, currorder)
                # Predict current temperature
                dj_n1 = predict_temperature(d_list, currorder)

            # Update
            dj_n1 += (1 - alpha) * dt * v_n0 if i == 1 else (1 - alpha) * dt * vj_n1

            # Predict values of new step
            Fext = external_force_list[:, i]
            vj_n1 = np.zeros_like(Fext)

            # Apply boundary conditions
            vj_n1[constraint_ctrlpts[0]] = (
                temperature_list[constraint_ctrlpts[0], i]
                - dj_n1[constraint_ctrlpts[0]]
            ) / dt
            dj_n1[constraint_ctrlpts[0]] = temperature_list[constraint_ctrlpts[0], i]

            print(f"Time step: {i}")
            residual_args = {
                "scalar_coefs": (1.0, 1.0),
                "flux": vj_n1,
                "flux_factor": 1.0 / (alpha * dt),
            }
            nonlinsolv.solve(
                dj_n1,
                Fext,
                self._compute_residual,
                self._solve_linearized_system,
                update_variables=self._update_transient_variables,
                residual_args=residual_args,
                linsolv_args={"scalar_coefs": (1.0 / (alpha * dt), 1)},
            )
            temperature_list[:, i] = np.copy(dj_n1)
            vj_n1 = np.copy(residual_args.get("flux"))


class st_heat_transfer_problem(spacetime_problem):
    def __init__(
        self,
        material: heat_transfer_mat,
        patch: singlepatch,
        time_patch: singlepatch,
        boundary: boundary_condition,
        solver_args: dict = {},
    ):
        super().__init__(patch, time_patch, boundary, solver_args)
        self.material = material
        self.preconditioner = self.activate_preconditioner()
        self._capacity_property: Union[None, Callable] = None
        self._conductivity_property: Union[None, Callable] = None
        self._ders_capacity_property: Union[None, Callable] = None
        self._ders_conductivity_property: Union[None, Callable] = None
        self._scalar_mean_capacity: List[float] = [1]
        self._scalar_mean_conductivity: List[np.ndarray] = [np.ones(self.part.ndim)]
        return

    def activate_preconditioner(self) -> fastdiagonalization:
        fastdiag = fastdiagonalization()
        fastdiag.add_free_controlpoints(self.sptm_free_ctrlpts)
        fastdiag.compute_space_eigendecomposition(
            self.part.quadrule_list, self.sp_table_dirichlet
        )
        fastdiag.compute_time_schurdecomposition(self.time.quadrule_list[0])
        return fastdiag

    def clear_properties(self):
        self._capacity_property = None
        self._conductivity_property = None
        self._ders_capacity_property = None
        self._ders_conductivity_property = None

    def compute_mf_sptm_capacity(
        self, array_in: np.ndarray, mf_args: dict = {}
    ) -> np.ndarray:
        mf_args = self._verify_fun_args(mf_args)
        mf_args = {
            "temperature": np.ones(self.part.nbqp_total * self.time.nbqp_total)
        } | mf_args
        if self._capacity_property is None:
            self._capacity_property = self.material.capacity(mf_args) * np.kron(
                np.ones_like(self.time.det_jac), self.part.det_jac
            )
            self._scalar_mean_capacity = [np.mean(self._capacity_property)]
        array_out = bspline_operations.compute_mf_scalar_u_v(
            self._quadrature_list,
            self._capacity_property,
            array_in,
            allow_lumping=False,
            enable_spacetime=True,
            time_ders=(0, 1),
        )
        return array_out

    def compute_mf_sptm_ders_capacity(
        self, array_in: np.ndarray, mf_args: dict = {}
    ) -> np.ndarray:
        mf_args = self._verify_fun_args(mf_args)
        mf_args = {
            "temperature": np.ones(self.part.nbqp_total * self.time.nbqp_total),
            "gradient": np.ones(
                (self.part.ndim + 1, self.part.nbqp_total * self.time.nbqp_total)
            ),
        } | mf_args
        grad_temperature = mf_args["gradient"]
        if self._ders_capacity_property is None:
            self._ders_capacity_property = (
                self.material.ders_capacity(mf_args)
                * grad_temperature[-1, :]
                * np.kron(self.time.det_jac, self.part.det_jac)
            )
        array_out = bspline_operations.compute_mf_scalar_u_v(
            self._quadrature_list,
            self._ders_capacity_property,
            array_in,
            allow_lumping=False,
            enable_spacetime=True,
            time_ders=(0, 0),
        )
        return array_out

    def compute_mf_sptm_conductivity(
        self, array_in: np.ndarray, mf_args: dict = {}
    ) -> np.ndarray:
        mf_args = self._verify_fun_args(mf_args)
        mf_args = {
            "temperature": np.ones(self.part.nbqp_total * self.time.nbqp_total)
        } | mf_args
        if self._conductivity_property is None:
            tmp1 = self.material.conductivity(mf_args) * np.kron(
                self.time.det_jac, self.part.det_jac
            )
            tmp1_reshaped = np.reshape(
                tmp1,
                newshape=(self.part.ndim, self.part.ndim, self.part.nbqp_total, -1),
                order="F",
            )
            tmp2 = np.einsum(
                "ilk,lmkp,jmk->ijkp",
                self.part.inv_jac,
                tmp1_reshaped,
                self.part.inv_jac,
                optimize=True,
            )
            self._conductivity_property = np.reshape(
                tmp2, newshape=(self.part.ndim, self.part.ndim, -1), order="F"
            )
            self._scalar_mean_conductivity = [
                np.array(
                    [
                        np.mean(self._conductivity_property[i, i, :])
                        for i in range(self.part.ndim)
                    ]
                )
            ]
        array_out = bspline_operations.compute_mf_scalar_gradu_gradv(
            self._quadrature_list,
            self._conductivity_property,
            array_in,
            enable_spacetime=True,
            time_ders=(0, 0),
        )
        return array_out

    def compute_mf_sptm_ders_conductivity(
        self, array_in: np.ndarray, mf_args: dict = {}
    ) -> np.ndarray:
        mf_args = self._verify_fun_args(mf_args)
        mf_args = {
            "temperature": np.ones(self.part.nbqp_total * self.time.nbqp_total),
            "gradient": np.ones(
                (self.part.ndim + 1, self.part.nbqp_total * self.time.nbqp_total)
            ),
        } | mf_args
        grad_temperature = mf_args["gradient"]
        if self._ders_conductivity_property is None:
            tmp1 = np.einsum(
                "ijk,jk,k->ik",
                self.material.ders_conductivity(mf_args),
                grad_temperature[:-1, :],
                np.kron(self.time.det_jac, self.part.det_jac),
                optimize=True,
            )
            tmp1_reshaped = np.reshape(
                tmp1, newshape=(self.part.ndim, self.part.nbqp_total, -1), order="F"
            )
            tmp2 = np.einsum(
                "ilk,lkp->ikp", self.part.inv_jac, tmp1_reshaped, optimize=True
            )
            self._ders_conductivity_property = np.reshape(
                tmp2, newshape=(self.part.ndim, -1), order="F"
            )
        array_out = bspline_operations.compute_mf_scalar_gradu_v(
            self._quadrature_list,
            self._ders_conductivity_property,
            array_in,
            enable_spacetime=True,
            time_ders=(0, 0),
        )
        return array_out

    def interpolate_sptm_temperature(
        self, u_ctrlpts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        u_interp = np.ravel(
            bspline_operations.interpolate_meshgrid(
                self._quadrature_list, np.atleast_2d(u_ctrlpts)
            ),
            order="F",
        )
        derstmp = bspline_operations.eval_jacobien(
            self._quadrature_list, np.atleast_2d(u_ctrlpts)
        )[0, :, :]
        derstmp_reshaped = np.reshape(
            derstmp, newshape=(self.part.ndim + 1, self.part.nbqp_total, -1), order="F"
        )
        uders_interp = np.zeros_like(derstmp_reshaped)
        uders_interp[:-1, :, :] = np.einsum(
            "ikp,ijk->jkp",
            derstmp_reshaped[:-1, :, :],
            self.part.inv_jac,
            optimize=True,
        )
        uders_interp[-1, :, :] = np.einsum(
            "kp,p->kp",
            derstmp_reshaped[-1, :, :],
            np.ravel(self.time.inv_jac),
            optimize=True,
        )
        return u_interp, np.reshape(
            uders_interp, newshape=(self.part.ndim + 1, -1), order="F"
        )

    def _solve_linearized_system(self, array_in: np.ndarray, **kwargs) -> np.ndarray:

        mf_args: dict = kwargs.get("mf_args")
        inner_tolerance: float = kwargs.get("inner_tolerance")
        solver_kind: str = str(kwargs.get("solver_kind", "picard")).lower()

        def compute_mf_tangent(array_in, args):
            array_out = self.compute_mf_sptm_capacity(
                array_in, args
            ) + self.compute_mf_sptm_conductivity(array_in, args)
            if solver_kind == "newton":
                array_out += self.compute_mf_sptm_ders_capacity(
                    array_in, args
                ) + self.compute_mf_sptm_ders_conductivity(array_in, args)
            return array_out

        if self._update_preconditioner:
            self.preconditioner.add_scalar_space_time_correctors(
                stiffness_corrector=self._scalar_mean_conductivity,
                advection_corrector=self._scalar_mean_capacity,
            )
        self.preconditioner.update_space_eigenvalues(scalar_coefs=(0, 1))
        output = super()._solve_linear_system(
            compute_mf_tangent,
            array_in,
            Pfun=self.preconditioner.apply_spacetime_scalar_preconditioner,
            cleanfun=clean_dirichlet,
            dod=self.sptm_constraint_ctrlpts,
            args=mf_args,
            tolerance=inner_tolerance,
        )
        self._linear_residual_list.append(output["res"])
        return output["sol"]

    def _compute_residual(
        self,
        temperature: np.ndarray,
        external_force: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:

        if self.update_properties:
            self.clear_properties()
        output = self.interpolate_sptm_temperature(temperature)
        args = {"temperature": output[0], "gradient": output[1]}
        internal_force = self.compute_mf_sptm_capacity(
            temperature, args
        ) + self.compute_mf_sptm_conductivity(temperature, args)
        residual = external_force - internal_force
        clean_dirichlet(residual, self.sptm_constraint_ctrlpts)
        return residual, args

    def solve_heat_transfer(
        self,
        temperature: np.ndarray,
        external_force: np.ndarray,
        auto_inner_tolerance: bool = True,
        auto_outer_tolerance: bool = False,
        inner_tolerance_args: dict = {"solver_kind": "picard"},
        **kwargs,
    ):

        assert (
            inner_tolerance_args.get("solver_kind") is not None
        ), "Define the type of solver"

        allow_fixed_point_acceleration = kwargs.get(
            "allow_anderson_acceleration", False
        )
        allow_linesearch = kwargs.get("allow_line_search", False)

        # Decide if linear or nonlinear problem
        self.update_properties = (
            False
            if self.material._has_uniform_capacity
            and self.material._has_uniform_conductivity
            else True
        )

        # Initialize stopping criteria parameters
        outer_tolerance = (
            super().select_outer_tolerance(self.part, time_patch=self.time)
            if auto_outer_tolerance
            else self._tolerance_nonlinear
        )

        # Solve using a nonlinear solver
        nonlinsolv = nonlinsolver(
            maxiters=self._maxiters_nonlinear,
            tolerance=outer_tolerance,
            linear_solver_tolerance=self._tolerance_linear,
        )

        nonlinsolv.modify_solver(
            allow_fixed_point_acceleration=allow_fixed_point_acceleration,
            allow_line_search=allow_linesearch,
        )

        output = nonlinsolv.solve(
            temperature,
            external_force,
            self._compute_residual,
            self._solve_linearized_system,
            select_inner_tolerance=(
                super().select_inner_tolerance if auto_inner_tolerance else None
            ),
            residual_args={},
            linsolv_args={"solver_kind": inner_tolerance_args.get("solver_kind")},
            inner_tolerance_args=inner_tolerance_args,
            save_information=True,
        )

        self._nonlinear_residual_list = output["nonlinear_residual_list"]
        self._nonlinear_time_list = output["nonlinear_time_list"]
        self._nonlinear_rate_list = output["nonlinear_rate_list"]
        self._linear_tolerance_list = output["linear_tolerance_list"]
        self._solution_history_list = output["solution_history_list"]
