from .lib_job import *
from typing import Union, Callable


def block_dot_product(A: np.ndarray, B: np.ndarray) -> float:
    """Computes dot product of A and B.
    Both are actually vectors arranged following each dimension
    A = [Au, Av, Aw] and B = [Bu, Bv, Bw]. Then A.B = Au.Bu + Av.Bv + Aw.Bw
    """
    return sum(np.dot(A[i], B[i]) for i in range(min(len(A), len(B))))


class mechanical_problem(space_problem):
    def __init__(
        self,
        material: plasticity,
        patch: singlepatch,
        boundary: boundary_condition,
        solver_args: dict = {},
        allow_lumping: bool = False,
    ):
        super().__init__(patch, boundary, solver_args, allow_lumping=allow_lumping)
        self.material = material
        self.preconditioner = self.activate_preconditioner()
        self._mass_property: Union[None, Callable] = None
        self._stiffness_property: Union[None, Callable] = None
        self._scalar_mean_mass: List[float] = [1.0] * self.nbvars
        self._scalar_mean_stiffness: List[np.ndarray] = [
            np.ones(self.nbvars)
        ] * self.nbvars
        return

    def activate_preconditioner(self) -> fastdiagonalization:
        fastdiag = fastdiagonalization()
        fastdiag.add_free_controlpoints(self.sp_free_ctrlpts)
        fastdiag.compute_space_eigendecomposition(
            self.part.quadrule_list, self.sp_table_dirichlet
        )
        return fastdiag

    def clear_properties(self):
        self._mass_property = None
        self._stiffness_property = None
        if self.allow_lumping:
            self._diagonal_mass = None

    def activate_explicit_dynamics(self):
        if self.allow_lumping:
            print("Solver will use diagonal matrix")
            self._diagonal_mass = None
        else:
            print("Solver will use plain matrix")
        return

    def compute_mf_mass(self, array_in, args={}):
        args = self._verify_fun_args(args)
        if self._mass_property is None:
            self._mass_property = self.material.density(args) * self.part.det_jac
            mass_mean = np.mean(self._mass_property)
            self._scalar_mean_mass = [mass_mean] * self.nbvars
        array_out = np.zeros_like(array_in)
        for i in range(self.nbvars):
            array_out[i, :] = bspline_operations.compute_mf_scalar_u_v(
                self.part.quadrule_list,
                self._mass_property,
                array_in[i, :],
                allow_lumping=self.allow_lumping,
            )
        return array_out

    def compute_mf_stiffness(self, array_in, args={}):
        "In this algorithm we only consider the linear elastic case"
        args = self._verify_fun_args(args)
        if self._stiffness_property is None:
            tangent = args.get(
                "consistent_tangent",
                self.material.set_linear_elastic_tensor(
                    self.part.nbqp_total, self.nbvars
                ),
            )
            self._stiffness_property = np.zeros(
                (
                    self.nbvars,
                    self.nbvars,
                    self.nbvars,
                    self.nbvars,
                    self.part.nbqp_total,
                )
            )
            for i in range(self.nbvars):
                for j in range(self.nbvars):
                    self._stiffness_property[i, j, ...] = np.einsum(
                        "ilk,lmk,jmk,k->ijk",
                        self.part.inv_jac,
                        tangent[i, j, : self.nbvars, : self.nbvars, ...],
                        self.part.inv_jac,
                        self.part.det_jac,
                        optimize=True,
                    )
            self._scalar_mean_stiffness = [
                np.array(
                    [
                        np.mean(self._stiffness_property[i, i, j, j, :])
                        for j in range(self.nbvars)
                    ]
                )
                for i in range(self.nbvars)
            ]

        array_out = np.zeros_like(array_in)
        for i in range(self.nbvars):
            array_out[i, :] = sum(
                bspline_operations.compute_mf_scalar_gradu_gradv(
                    self.part.quadrule_list,
                    self._stiffness_property[i, j, ...],
                    array_in[j, :],
                )
                for j in range(self.nbvars)
            )

        return array_out

    def interpolate_strain(self, displacement, convert_to_3d=False):
        "Compute strain field from displacement field"
        ders_par = bspline_operations.eval_jacobien(
            self.part.quadrule_list, displacement
        )
        ders_phy = np.einsum("ijl,jkl->ikl", ders_par, self.part.inv_jac, optimize=True)
        strain = 0.5 * (ders_phy + np.einsum("ijl->jil", ders_phy, optimize=True))
        if convert_to_3d:
            output = np.zeros((3, 3, np.shape(strain)[-1]))
            output[: self.nbvars, : self.nbvars, :] = strain
            return output
        else:
            return strain

    def _assemble_internal_force(self, stress):
        mf = matrixfree(
            np.array(
                [quadrule.nbquadpts for quadrule in self.part.quadrule_list], dtype=int
            )
        )
        prop = np.einsum(
            "ilk,ljk,k->ijk",
            self.part.inv_jac,
            stress[: self.nbvars, : self.nbvars, :],
            self.part.det_jac,
            optimize=True,
        )
        array_out = np.zeros((self.nbvars, self.part.nbctrlpts_total))
        for i in range(self.nbvars):
            for k in range(self.nbvars):
                alpha_list = np.ones(self.nbvars, dtype=int)
                alpha_list[k] = 3
                array_out[i, :] += mf.sumfactorization(
                    [
                        quadrule.weights[alpha]
                        for quadrule, alpha in zip(self.part.quadrule_list, alpha_list)
                    ],
                    prop[k, i, :],
                    istranspose=False,
                )
        return array_out

    def _compute_elastoplastic_residual(
        self, displacement, external_force, old_plastic_vars, update_properties=False
    ):

        if update_properties:
            self.clear_properties()
        convert_to_3d = False if self.nbvars == 1 else True
        strain = self.interpolate_strain(displacement, convert_to_3d=convert_to_3d)
        stress, mech_args, new_plastic_vars = self.material.return_mapping(
            strain, old_plastic_vars
        )
        residual = external_force - self._assemble_internal_force(stress)
        clean_dirichlet(residual, self.sp_constraint_ctrlpts)
        return residual, mech_args, new_plastic_vars

    def _linearized_elastoplastic_solver(self, array_in, args={}):
        if self._update_preconditioner:
            self.preconditioner.add_scalar_space_time_correctors(
                stiffness_corrector=self._scalar_mean_stiffness
            )
        self.preconditioner.update_space_eigenvalues(scalar_coefs=(0, 1))
        output = self._solve_linear_system(
            self.compute_mf_stiffness,
            array_in,
            Pfun=self.preconditioner.apply_vectorial_preconditioner,
            dotfun=block_dot_product,
            cleanfun=clean_dirichlet,
            dod=self.sp_constraint_ctrlpts,
            args=args,
        )
        self._linear_residual_list.append(output["res"])
        return output["sol"]

    def solve_elastoplasticity(
        self,
        displacement_list: np.ndarray,
        external_force_list: np.ndarray,
        save_plastic_vars=False,
    ):

        # Decide if it is a linear or nonlinear problem
        update_properties = self.material._activated_plasticity
        plastic_vars = []

        # Get inactive control points
        constraint_ctrlpts = self.sp_constraint_ctrlpts

        if external_force_list.ndim == 2 and displacement_list.ndim == 2:

            # Single step problem
            for j in range(self._maxiters_nonlinear):
                residual, mech_args, _ = self._compute_elastoplastic_residual(
                    displacement_list,
                    external_force_list,
                    {},
                    update_properties=update_properties,
                )

                residual_norm = np.linalg.norm(residual)
                if j == 0:
                    ref_residual_norm = residual_norm
                print(f"Non linear error: {residual_norm:.5e}")
                if residual_norm <= max(
                    [self._safeguard, self._tolerance_nonlinear * ref_residual_norm]
                ):
                    break

                displacement_list += self._linearized_elastoplastic_solver(
                    residual, args=mech_args
                )
        else:

            # Time-stepping problem
            new_plastic_vars = {}
            for i in range(1, external_force_list.shape[-1]):

                # Predict values of new step
                Fext_n1 = np.copy(external_force_list[:, :, i])
                dj_n1 = np.copy(displacement_list[:, :, i - 1])
                for j, dod in enumerate(constraint_ctrlpts):
                    dj_n1[j, dod] = displacement_list[j, dod, i]
                old_plastic_vars = deepcopy(new_plastic_vars)

                print(f"Step: {i}")
                for j in range(self._maxiters_nonlinear):
                    (
                        residual,
                        mech_args,
                        new_plastic_vars,
                    ) = self._compute_elastoplastic_residual(
                        dj_n1,
                        Fext_n1,
                        old_plastic_vars,
                        update_properties=update_properties,
                    )

                    residual_norm = np.linalg.norm(residual)
                    if j == 0:
                        ref_residual_norm = residual_norm
                    print(f"Non linear error: {residual_norm:.5e}")
                    if residual_norm <= max(
                        [self._safeguard, self._tolerance_nonlinear * ref_residual_norm]
                    ):
                        break

                    dj_n1 += self._linearized_elastoplastic_solver(
                        residual, args=mech_args
                    )

                displacement_list[:, :, i] = dj_n1
                if save_plastic_vars:
                    plastic_vars.append(new_plastic_vars)

        return plastic_vars

    def _linearized_explicit_dynamic_solver(self, array_in, args={}):

        if self.allow_lumping:
            if self._diagonal_mass is None or self._mass_property is None:
                args = self._verify_fun_args(args)
                self._mass_property = self.material.density(args) * self.part.det_jac
                self._diagonal_mass = bspline_operations.assemble_scalar_u_v(
                    self.part.quadrule_list, self._mass_property, allow_lumping=True
                )

            array_out = np.zeros_like(array_in)
            for i in range(self.nbvars):
                free_nodes = self.sp_free_ctrlpts[i]
                array_out[i, free_nodes] = (
                    array_in[i, free_nodes] / self._diagonal_mass[free_nodes]
                )

            return array_out
        else:
            if self._update_preconditioner:
                self.preconditioner.add_scalar_space_time_correctors(
                    mass_corrector=self._scalar_mean_mass
                )
            self.preconditioner.update_space_eigenvalues(scalar_coefs=(1, 0))
            output = self._solve_linear_system(
                self.compute_mf_mass,
                array_in,
                Pfun=self.preconditioner.apply_vectorial_preconditioner,
                dotfun=block_dot_product,
                cleanfun=clean_dirichlet,
                dod=self.sp_constraint_ctrlpts,
                args=args,
            )
            self._linear_residual_list.append(output["res"])
        return output["sol"]

    def solve_explicit_linear_dynamics(
        self, displacement_list, external_force_list, time_list, initial_velocity=None
    ):
        "Solves linear explicit dynamic problem."
        update_properties = not (
            self.material._has_uniform_density
        )  # True only if density depends on current plastic variables

        assert len(time_list) > 3, "At least 2 steps"
        constraint_ctrlpts = self.sp_constraint_ctrlpts
        old_plastic_vars, new_plastic_vars = {}, {}
        velocity_has_been_given = True
        if initial_velocity is None:
            velocity_has_been_given = False
            initial_velocity = np.zeros_like(displacement_list[:, :, 0])

        def update_force_nonhomogeneous_bc(
            problem: mechanical_problem, acc, Fext_in, args
        ):
            Fext_out = Fext_in - problem.compute_mf_mass(acc, args)
            return Fext_out

        def predict_displacement(dis, vel, acc, dt):
            return dis + dt * vel + 0.5 * dt**2 * acc

        def update_velocity(vel, acc_old, acc_new, dt):
            return vel + 0.5 * dt * (acc_old + acc_new)

        def compute_acceleration(problem: mechanical_problem, res, args):
            return problem._linearized_explicit_dynamic_solver(res, args)

        args = self._verify_fun_args({})
        self.activate_explicit_dynamics()

        # Compute initial acceleration considering static problem
        Fext = np.copy(external_force_list[:, :, 0])
        d_n0 = np.copy(displacement_list[:, :, 0])
        v_n0 = np.copy(initial_velocity)
        if not velocity_has_been_given:
            for k in range(self.nbvars):
                dt1 = time_list[1] - time_list[0]
                d0 = np.copy(displacement_list[k, constraint_ctrlpts[k], 0])
                d1 = np.copy(displacement_list[k, constraint_ctrlpts[k], 1])
                v_n0[k, constraint_ctrlpts[k]] = (d1 - d0) / dt1

        a_n0 = np.zeros_like(d_n0)
        for k in range(self.nbvars):
            dt1 = time_list[1] - time_list[0]
            dt2 = time_list[2] - time_list[0]
            d0 = np.copy(displacement_list[k, constraint_ctrlpts[k], 0])
            d1 = np.copy(displacement_list[k, constraint_ctrlpts[k], 1])
            d2 = np.copy(displacement_list[k, constraint_ctrlpts[k], 2])
            a_n0[k, constraint_ctrlpts[k]] = (
                -d1 * dt2 + d2 * dt1 - d0 * (dt1 - dt2)
            ) / (dt1 * dt2 * (dt2 - dt1))

        Fext = update_force_nonhomogeneous_bc(self, a_n0, Fext, args)
        residual, _, new_plastic_vars = self._compute_elastoplastic_residual(
            d_n0, Fext, old_plastic_vars, update_properties=update_properties
        )
        old_plastic_vars = deepcopy(new_plastic_vars)
        a_n0 += compute_acceleration(self, residual, args)

        for i in range(1, len(time_list)):

            # Get delta time
            dt = time_list[i] - time_list[i - 1]

            # Get values of last step
            d_n0 = np.copy(displacement_list[:, :, i - 1])
            if i > 1:
                v_n0 = np.copy(v_n1)
                a_n0 = np.copy(a_n1)

            # Predict values of new step
            Fext = np.copy(external_force_list[:, :, i])
            d_n1 = predict_displacement(d_n0, v_n0, a_n0, dt)
            a_n1 = np.zeros_like(d_n0)

            # Apply boundary conditions
            if i == len(time_list) - 1:
                for k in range(self.nbvars):
                    # dtm1 = time_list[-1] - time_list[-2]
                    # dtm2 = time_list[-1] - time_list[-3]
                    # d_m1 = np.copy(displacement_list[k, constraint_ctrlpts[k], -1])
                    # d_m2 = np.copy(displacement_list[k, constraint_ctrlpts[k], -2])
                    # d_m3 = np.copy(displacement_list[k, constraint_ctrlpts[k], -3])
                    a_n1[
                        k, constraint_ctrlpts[k]
                    ] = 0  # TODO: remake for non homogeneous B.C.
            else:
                for k in range(self.nbvars):
                    # dtm1 = time_list[i] - time_list[i-1]
                    # dtp1 = time_list[i+1] - time_list[i]
                    # d_m1 = np.copy(displacement_list[k, constraint_ctrlpts[k], i-1])
                    # d_m0 = np.copy(displacement_list[k, constraint_ctrlpts[k], i])
                    # d_p1 = np.copy(displacement_list[k, constraint_ctrlpts[k], i+1])
                    a_n1[
                        k, constraint_ctrlpts[k]
                    ] = 0  # TODO: remake for non homogeneous B.C.

            for k in range(self.nbvars):
                d_n1[k, constraint_ctrlpts[k]] = displacement_list[
                    k, constraint_ctrlpts[k], i
                ]

            # Compute residual and update plastic variables
            Fext = update_force_nonhomogeneous_bc(self, a_n1, Fext, args)
            residual, _, new_plastic_vars = self._compute_elastoplastic_residual(
                d_n1, Fext, old_plastic_vars, update_properties=update_properties
            )
            old_plastic_vars = deepcopy(new_plastic_vars)
            a_n1 += compute_acceleration(self, residual, args)
            v_n1 = update_velocity(v_n0, a_n0, a_n1, dt)

            # Save data for next step
            displacement_list[:, :, i] = np.copy(d_n1)

        return

    def solve_eigenvalue_problem(self, args={}, which="LM", k=None):

        original_lumping_condition = deepcopy(self.allow_lumping)
        self.allow_lumping = False  # We only solve eigenproblem with plain matrix

        fastdiag = fastdiagonalization()
        fastdiag.compute_space_eigendecomposition(
            self.part.quadrule_list, np.zeros((self.nbvars, 3, 2))
        )
        fastdiag.add_free_controlpoints(
            [
                np.arange(self.part.nbctrlpts_total, dtype=int)
                for _ in range(self.nbvars)
            ]
        )
        fastdiag.update_space_eigenvalues(scalar_coefs=(1, 0))

        solv = linsolver(tolerance=1e-2)  # High tolerance
        sizeofvector = self.part.ndim * self.part.nbctrlpts_total

        def mass(x):
            x_in = np.reshape(x, newshape=(self.nbvars, -1), order="F")
            x_out = self.compute_mf_mass(x_in, args=args)
            return np.ravel(x_out, order="F")

        def preconditioner(x):
            output = solv.GMRES(
                self.compute_mf_mass,
                np.reshape(x, newshape=(self.nbvars, -1), order="F"),
                Pfun=fastdiag.apply_vectorial_preconditioner,
                dotfun=block_dot_product,
                args=args,
            )
            return np.ravel(output["sol"], order="F")

        def stiffness(x):
            x_in = np.reshape(x, newshape=(self.nbvars, -1), order="F")
            x_out = self.compute_mf_stiffness(x_in, args=args)
            return np.ravel(x_out, order="F")

        if k is None:
            k = sizeofvector - 2
        eigenvalues, eigenvectors = solv.eigs(
            N=sizeofvector,
            Afun=stiffness,
            Bfun=mass,
            Pfun=preconditioner,
            k=k,
            which=which,
        )
        self.allow_lumping = original_lumping_condition

        return eigenvalues, eigenvectors
