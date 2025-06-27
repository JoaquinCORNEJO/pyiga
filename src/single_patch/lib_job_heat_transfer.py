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
        self._scalar_mean_conductivity: List[np.ndarray] = [np.ones(self.nbvars)]

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

    def compute_mf_capacity(self, array_in: np.ndarray, args: dict = {}) -> np.ndarray:
        args = self._verify_fun_args(args)
        args = {"temperature": np.ones(self.part.nbqp_total)} | args
        if self._capacity_property is None:
            self._capacity_property = self.material.capacity(args) * self.part.det_jac
            self._scalar_mean_capacity = [np.mean(self._capacity_property)]
        array_out = bspline_operations.compute_mf_scalar_u_v(
            self.part.quadrule_list,
            self._capacity_property,
            array_in,
            allow_lumping=False,
        )
        return array_out

    def compute_mf_conductivity(
        self, array_in: np.ndarray, args: dict = {}
    ) -> np.ndarray:
        args = self._verify_fun_args(args)
        args = {"temperature": np.ones(self.part.nbqp_total)} | args
        if self._conductivity_property is None:
            self._conductivity_property = np.einsum(
                "ilk,lmk,jmk,k->ijk",
                self.part.inv_jac,
                self.material.conductivity(args),
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
        args: dict = {},
    ) -> np.ndarray:
        assert (
            scalar_coefs is not None
        ), "Define contribution from capacity and conductivity"
        array_out = 0.0
        if scalar_coefs[0] != 0:
            array_out += scalar_coefs[0] * self.compute_mf_capacity(flux, args)
        if scalar_coefs[1] != 0:
            array_out += scalar_coefs[1] * self.compute_mf_conductivity(
                temperature, args
            )
        return array_out

    def _compute_heat_transfer_residual(
        self,
        temperature: np.ndarray,
        flux: np.ndarray,
        external_force: np.ndarray,
        scalar_coefs: tuple,
        update_properties: bool = False,
    ) -> Tuple[np.ndarray, dict]:

        if update_properties:
            self.clear_properties()
        args = {"temperature": self.interpolate_temperature(temperature)}
        residual = external_force - self._assemble_internal_force(
            temperature, flux, scalar_coefs, args
        )

        return residual, args

    def _linearized_heat_trasfer_solver(
        self, array_in: np.ndarray, scalar_coefs: tuple, args: dict = {}
    ) -> np.ndarray:
        assert (
            scalar_coefs is not None
        ), "Define contribution from capacity and conductivity"

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
        output = self._solve_linear_system(
            compute_mf_tangent,
            array_in,
            Pfun=self.preconditioner.apply_scalar_preconditioner,
            cleanfun=clean_dirichlet,
            dod=self.sp_constraint_ctrlpts,
            args=args,
        )
        self._linear_residual_list.append(output["res"])
        return output["sol"]

    def solve_heat_transfer(
        self,
        temperature_list: np.ndarray,
        external_force_list: np.ndarray,
        time_list: Union[None, np.ndarray] = None,
        alpha: float = 0.5,
    ):

        # Decide if it is a linear or nonlinear problem
        update_properties = not (
            self.material._has_uniform_capacity
            and self.material._has_uniform_conductivity
        )

        # Get inactive control points
        constraint_ctrlpts = self.sp_constraint_ctrlpts

        if external_force_list.ndim == 1 and temperature_list.ndim == 1:

            # This is a steady heat problem
            for j in range(self._maxiters_nonlinear):

                residual, args = self._compute_heat_transfer_residual(
                    temperature_list,
                    None,
                    external_force_list,
                    scalar_coefs=(0, 1),
                    update_properties=update_properties,
                )
                clean_dirichlet(residual, constraint_ctrlpts)

                norm_residual = np.linalg.norm(residual)
                if j == 0:
                    ref_norm_residual = norm_residual
                print(f"Non linear error: {norm_residual:.5e}")
                if norm_residual <= max(
                    self._safeguard, self._tolerance_nonlinear * ref_norm_residual
                ):
                    break

                increment = self._linearized_heat_trasfer_solver(
                    residual, scalar_coefs=(0, 1), args=args
                )
                temperature_list += increment

        else:

            # This is a transient problem
            assert len(time_list) >= 2, "At least 2 steps required"
            Fext = np.copy(external_force_list[:, 0])
            d_n0 = np.copy(temperature_list[:, 0])
            v_n0 = np.zeros_like(d_n0)
            v_n0[constraint_ctrlpts[0]] = (
                1.0
                / (time_list[1] - time_list[0])
                * (
                    temperature_list[constraint_ctrlpts[0], 1]
                    - temperature_list[constraint_ctrlpts[0], 0]
                )
            )

            residual, args = self._compute_heat_transfer_residual(
                d_n0,
                v_n0,
                Fext,
                scalar_coefs=(1, 1),
                update_properties=update_properties,
            )
            clean_dirichlet(residual, constraint_ctrlpts)
            v_n0 += self._linearized_heat_trasfer_solver(
                residual, scalar_coefs=(1, 0), args=args
            )

            for i in range(1, len(time_list)):

                # Get delta time
                dt = time_list[i] - time_list[i - 1]

                # Get values of last step
                d_n0 = np.copy(temperature_list[:, i - 1])
                if i > 1:
                    v_n0 = np.copy(vj_n1)

                # Predict values of new step
                Fext = np.copy(external_force_list[:, i])
                dj_n1 = d_n0 + (1 - alpha) * dt * v_n0
                vj_n1 = np.zeros_like(d_n0)

                # Apply boundary conditions
                vj_n1[constraint_ctrlpts[0]] = (
                    1.0
                    / (alpha * dt)
                    * (
                        temperature_list[constraint_ctrlpts[0], i]
                        - dj_n1[constraint_ctrlpts[0]]
                    )
                )
                dj_n1[constraint_ctrlpts[0]] = temperature_list[
                    constraint_ctrlpts[0], i
                ]

                print(f"Step: {i}")
                for j in range(self._maxiters_nonlinear):

                    residual, args = self._compute_heat_transfer_residual(
                        dj_n1,
                        vj_n1,
                        Fext,
                        scalar_coefs=(1, 1),
                        update_properties=update_properties,
                    )
                    clean_dirichlet(residual, constraint_ctrlpts)
                    self._solution_history_list[f"step_{i}_noniter_{j}"] = np.copy(
                        dj_n1
                    )

                    norm_residual = np.linalg.norm(residual)
                    if j == 0:
                        ref_norm_residual = norm_residual
                    print(f"Non linear error: {norm_residual:.5e}")
                    if norm_residual <= max(
                        self._safeguard, self._tolerance_nonlinear * ref_norm_residual
                    ):
                        break

                    increment = self._linearized_heat_trasfer_solver(
                        residual, scalar_coefs=(1, alpha * dt), args=args
                    )
                    vj_n1 += increment
                    dj_n1 += alpha * dt * increment

                # Save data for next step
                temperature_list[:, i] = np.copy(dj_n1)

    def solve_heat_transfer_bdf(
        self,
        temperature_list: np.ndarray,
        external_force_list: np.ndarray,
        tspan: Tuple[float],
        nsteps: int,
        norder: int = 1,
    ):
        """
        Solves an heat transfer equation using the Backward Differentiation Formula (BDF) method.

        Parameters:
        temperature_list : Array of initial temperature values at control points.
        external_force_list : Array of external forces applied at control points.
        tspan : A tuple containing the start and end times, (t0, tf).
        nsteps : Number of time steps to use in the solution.
        norder : Order of the BDF method (default is 1).
        """
        # Check input parameters
        assert norder in [1, 2, 3, 4], "Order not supported"
        assert nsteps >= 2, "At least 2 steps required"
        assert (
            np.size(temperature_list, 1) >= nsteps + 1
        ), "Temperature list must have enough columns for time steps"

        # Decide if it is a linear or nonlinear problem
        update_properties = not (
            self.material._has_uniform_capacity
            and self.material._has_uniform_conductivity
        )

        # Get inactive control points
        constraint_ctrlpts = self.sp_constraint_ctrlpts

        def compute_flux_current(
            histtemp: List[np.ndarray],
            currtemp: np.ndarray,
            dt: float,
            norder: int,
        ):
            assert len(histtemp) == norder, "Size problem."
            if norder == 1:
                [y1] = histtemp
                flux = (currtemp - y1) / dt
            elif norder == 2:
                [y1, y2] = histtemp
                flux = (3 * currtemp - 4 * y2 + y1) / (2 * dt)
            elif norder == 3:
                [y1, y2, y3] = histtemp
                flux = (11 * currtemp - 18 * y3 + 9 * y2 - 2 * y1) / (6 * dt)
            elif norder == 4:
                [y1, y2, y3, y4] = histtemp
                flux = (25 * currtemp - 48 * y4 + 36 * y3 - 16 * y2 + 3 * y1) / (
                    12 * dt
                )
            else:
                raise ValueError("Order not supported")
            return flux

        def compute_flux_prediction(
            histflux: List[np.ndarray],
            norder: int,
        ):
            assert len(histflux) == norder, "Size problem."
            if norder == 1:
                [w1] = histflux
                currflux = w1
            elif norder == 2:
                [w1, w2] = histflux
                currflux = (3 * w1 - w2) / 2
            elif norder == 3:
                [w1, w2, w3] = histflux
                currflux = (23 * w1 - 16 * w2 + 5 * w3) / 12
            elif norder == 4:
                [w1, w2, w3, w4] = histflux
                currflux = (55 * w1 - 59 * w2 + 37 * w3 - 9 * w4) / 24
            else:
                raise ValueError("Order not supported")
            return currflux

        def select_snapshots(i, y, norder):
            "Creates a list of previous solution values."
            return [y[:, i - k] for k in range(norder, 0, -1)]

        def select_parameter(norder):
            assert norder in [1, 2, 3, 4], "Order not supported"
            parameters = [1.0, 3.0 / 2.0, 11.0 / 6.0, 25.0 / 12.0]
            return parameters[norder - 1]

        # Initialize time and solution arrays
        dt = (tspan[1] - tspan[0]) / nsteps
        flux_list = np.zeros_like(temperature_list)
        flux_list[constraint_ctrlpts[0], 0] = (
            temperature_list[constraint_ctrlpts[0], 1]
            - temperature_list[constraint_ctrlpts[0], 0]
        ) / dt
        residual, args = self._compute_heat_transfer_residual(
            temperature_list[:, 0],
            flux_list[:, 0],
            external_force_list[:, 0],
            scalar_coefs=(1, 1),
            update_properties=update_properties,
        )
        clean_dirichlet(residual, constraint_ctrlpts)
        flux_list[:, 0] += self._linearized_heat_trasfer_solver(
            residual, scalar_coefs=(1, 0), args=args
        )

        # Main loop to solve the ODE using the BDF method
        for i in range(1, nsteps + 1):

            # Get current order and ensure it does not exceed norder
            currorder = min(i, norder)

            # Determine the order to use based on the current step
            v_list = select_snapshots(i, flux_list, currorder)
            d_list = select_snapshots(i, temperature_list, currorder)

            # Predict the new solution value using the previous step
            vj_n1 = compute_flux_prediction(v_list, currorder)
            dj_n1 = temperature_list[:, i - 1] + dt * vj_n1

            # Apply boundary conditions
            vj_n1[constraint_ctrlpts[0]] = (
                temperature_list[constraint_ctrlpts[0], i]
                - dj_n1[constraint_ctrlpts[0]]
            ) / dt
            dj_n1[constraint_ctrlpts[0]] = temperature_list[constraint_ctrlpts[0], i]

            # Solve the BDF residual equation to find the new solution value
            print(f"Step: {i}")
            for j in range(self._maxiters_nonlinear):

                residual, args = self._compute_heat_transfer_residual(
                    dj_n1,
                    vj_n1,
                    external_force_list[:, i],
                    scalar_coefs=(1, 1),
                    update_properties=update_properties,
                )
                clean_dirichlet(residual, constraint_ctrlpts)
                self._solution_history_list[f"step_{i}_noniter_{j}"] = np.copy(dj_n1)

                norm_residual = np.linalg.norm(residual)
                if j == 0:
                    ref_norm_residual = norm_residual
                print(f"Non linear error: {norm_residual:.5e}")
                if norm_residual <= max(
                    self._safeguard, self._tolerance_nonlinear * ref_norm_residual
                ):
                    break

                increment = self._linearized_heat_trasfer_solver(
                    residual,
                    scalar_coefs=(select_parameter(currorder) / dt, 1),
                    args=args,
                )
                dj_n1 += increment
                vj_n1 = compute_flux_current(d_list, dj_n1, dt, currorder)

            # Save data for next step
            temperature_list[:, i] = np.copy(dj_n1)
            flux_list[:, i] = np.copy(vj_n1)


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
        self._scalar_mean_conductivity: List[np.ndarray] = [np.ones(self.nbvars)]
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
        self, array_in: np.ndarray, args: dict = {}
    ) -> np.ndarray:
        args = self._verify_fun_args(args)
        args = {
            "temperature": np.ones(self.part.nbqp_total * self.time.nbqp_total)
        } | args
        if self._capacity_property is None:
            self._capacity_property = self.material.capacity(args) * np.kron(
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
        self, array_in: np.ndarray, args: dict = {}
    ) -> np.ndarray:
        args = self._verify_fun_args(args)
        args = {
            "temperature": np.ones(self.part.nbqp_total * self.time.nbqp_total),
            "gradient": np.ones(
                (self.part.ndim + 1, self.part.nbqp_total * self.time.nbqp_total)
            ),
        } | args
        grad_temperature = args["gradient"]
        if self._ders_capacity_property is None:
            self._ders_capacity_property = (
                self.material.ders_capacity(args)
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
        self, array_in: np.ndarray, args: dict = {}
    ) -> np.ndarray:
        args = self._verify_fun_args(args)
        args = {
            "temperature": np.ones(self.part.nbqp_total * self.time.nbqp_total)
        } | args
        if self._conductivity_property is None:
            tmp1 = self.material.conductivity(args) * np.kron(
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
        self, array_in: np.ndarray, args: dict = {}
    ) -> np.ndarray:
        args = self._verify_fun_args(args)
        args = {
            "temperature": np.ones(self.part.nbqp_total * self.time.nbqp_total),
            "gradient": np.ones(
                (self.part.ndim + 1, self.part.nbqp_total * self.time.nbqp_total)
            ),
        } | args
        grad_temperature = args["gradient"]
        if self._ders_conductivity_property is None:
            tmp1 = np.einsum(
                "ijk,jk,k->ik",
                self.material.ders_conductivity(args),
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

    def _linearized_spacetime_heat_transfer_solver(
        self,
        external_force: np.ndarray,
        args: dict = {},
        use_picard: bool = True,
        inner_tolerance: bool = None,
    ) -> np.ndarray:

        def compute_mf_tangent(array_in, args):
            array_out = self.compute_mf_sptm_capacity(
                array_in, args
            ) + self.compute_mf_sptm_conductivity(array_in, args)
            if not use_picard:
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
        output = self._solve_linear_system(
            compute_mf_tangent,
            external_force,
            Pfun=self.preconditioner.apply_spacetime_scalar_preconditioner,
            cleanfun=clean_dirichlet,
            dod=self.sptm_constraint_ctrlpts,
            args=args,
            tolerance=inner_tolerance,
        )
        self._linear_residual_list.append(output["res"])
        return output["sol"]

    def _compute_heat_transfer_residual(
        self,
        temperature: np.ndarray,
        external_force: np.ndarray,
        update_properties: bool = False,
    ) -> Tuple[np.ndarray, dict]:

        if update_properties:
            self.clear_properties()
        output = self.interpolate_sptm_temperature(temperature)
        args = {"temperature": output[0], "gradient": output[1]}
        residual = external_force - (
            self.compute_mf_sptm_capacity(temperature, args)
            + self.compute_mf_sptm_conductivity(temperature, args)
        )

        return residual, args

    def solve_heat_transfer(
        self,
        temperature: np.ndarray,
        external_force: np.ndarray,
        use_picard: bool = True,
        auto_inner_tolerance: bool = True,
        auto_outer_tolerance: bool = False,
        nonlinear_args: dict = {},
    ):

        def select_outer_tolerance(
            problem: spacetime_problem, factor: float = 0.5
        ) -> float:
            meshparameter_part = problem.part._compute_global_mesh_parameter()
            meshparameter_time = problem.time._compute_global_mesh_parameter()
            meshsize = max(meshparameter_part, meshparameter_time)
            degree = min(min(problem.part.degree), problem.time.degree)
            outer_tolerance = factor * (0.5**degree) * meshsize
            return outer_tolerance

        def select_inner_tolerance(
            use_picard: bool,
            res_new: float,
            res_old: float,
            inner_tolerance: float,
            condition: bool,
            solver_args: dict,
        ) -> float:
            eps_kr0 = solver_args.get("initial", 0.5)
            if use_picard:
                threshold_ref = solver_args.get("static", 0.25)
            else:
                gamma_kr = solver_args.get("coefficient", 1.0)
                omega_kr = solver_args.get("exponential", 2.0)
                if condition:
                    ratio = res_new / res_old
                    eps_kr_k = gamma_kr * np.power(ratio, omega_kr)
                    eps_kr_r = gamma_kr * np.power(inner_tolerance, omega_kr)
                    threshold_ref = (
                        min(eps_kr_k, eps_kr_r) if eps_kr_r > 0.1 else eps_kr_k
                    )
                else:
                    threshold_ref = eps_kr0
            return max(self._tolerance_linear, min(eps_kr0, threshold_ref))

        # Decide if linear or nonlinear problem
        update_properties = (
            False
            if self.material._has_uniform_capacity
            and self.material._has_uniform_conductivity
            else True
        )

        # Get inactive control points
        constraint_ctrlpts = self.sptm_constraint_ctrlpts

        # Initialize stopping criteria parameters
        outer_tolerance = (
            select_outer_tolerance(self)
            if auto_outer_tolerance
            else self._tolerance_nonlinear
        )
        norm_increment, norm_temperature = 1.0, 1.0
        norm_residual_old = None
        inner_tolerance_old = None

        start = time.process_time()
        for iteration in range(self._maxiters_nonlinear):

            residual, args = self._compute_heat_transfer_residual(
                temperature, external_force, update_properties=update_properties
            )
            clean_dirichlet(residual, constraint_ctrlpts)

            norm_residual = np.linalg.norm(residual)
            print(f"Nonlinear error: {norm_residual:.3e}")
            self._nonlinear_residual_list.append(norm_residual)
            self._solution_history_list[f"noniter_{iteration}"] = np.copy(temperature)

            finish = time.process_time()
            self._nonlinear_time_list.append(finish - start)

            if iteration == 0:
                norm_residual_ref = norm_residual
            else:
                if norm_residual <= max(
                    self._safeguard, outer_tolerance * norm_residual_ref
                ):
                    break
                if norm_increment <= max(
                    self._safeguard, outer_tolerance * norm_temperature
                ):
                    break

            # Update inner threshold
            if auto_inner_tolerance:
                inner_tolerance = select_inner_tolerance(
                    use_picard,
                    norm_residual,
                    norm_residual_old,
                    inner_tolerance_old,
                    iteration > 0,
                    nonlinear_args,
                )
                norm_residual_old = norm_residual
                inner_tolerance_old = inner_tolerance
            else:
                inner_tolerance = self._tolerance_linear
            self._linear_tolerance_list.append(inner_tolerance)

            # Solve for active control points
            increment = self._linearized_spacetime_heat_transfer_solver(
                residual,
                args=args,
                use_picard=use_picard,
                inner_tolerance=inner_tolerance,
            )

            # Update active control points
            temperature += increment
            norm_increment = (
                self.norm_of_field(increment, norm_type="l2")
                if auto_outer_tolerance
                else np.linalg.norm(increment)
            )
            norm_temperature = (
                self.norm_of_field(temperature, norm_type="l2")
                if auto_outer_tolerance
                else np.linalg.norm(temperature)
            )
            self._nonlinear_rate_list.append(norm_increment / norm_temperature)
