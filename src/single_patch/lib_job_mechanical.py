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
        self._scalar_mean_mass: List[float] = [1.0] * self.part.ndim
        self._scalar_mean_stiffness: List[np.ndarray] = [
            np.ones(self.part.ndim)
        ] * self.part.ndim

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

    def compute_mf_mass(self, array_in: np.ndarray, mf_args: dict = {}) -> np.ndarray:
        mf_args = self._verify_fun_args(mf_args)
        if self._mass_property is None:
            self._mass_property = self.material.density(mf_args) * self.part.det_jac
            mass_mean = np.mean(self._mass_property)
            self._scalar_mean_mass = [mass_mean] * self.part.ndim
        array_out = np.zeros_like(array_in)
        for i in range(self.part.ndim):
            array_out[i, :] = bspline_operations.compute_mf_scalar_u_v(
                self.part.quadrule_list,
                self._mass_property,
                array_in[i, :],
                allow_lumping=self.allow_lumping,
            )
        return array_out

    def compute_mf_stiffness(
        self, array_in: np.ndarray, mf_args: dict = {}
    ) -> np.ndarray:
        "In this algorithm we only consider the linear elastic case"
        mf_args = self._verify_fun_args(mf_args)
        if self._stiffness_property is None:
            tangent = mf_args.get(
                "consistent_tangent",
                self.material.set_linear_elastic_tensor(
                    self.part.nbqp_total, self.part.ndim
                ),
            )
            self._stiffness_property = np.zeros(
                (
                    self.part.ndim,
                    self.part.ndim,
                    self.part.ndim,
                    self.part.ndim,
                    self.part.nbqp_total,
                )
            )
            for i in range(self.part.ndim):
                for j in range(self.part.ndim):
                    self._stiffness_property[i, j, ...] = np.einsum(
                        "ilk,lmk,jmk,k->ijk",
                        self.part.inv_jac,
                        tangent[i, j, : self.part.ndim, : self.part.ndim, ...],
                        self.part.inv_jac,
                        self.part.det_jac,
                        optimize=True,
                    )
            self._scalar_mean_stiffness = [
                np.array(
                    [
                        np.mean(self._stiffness_property[i, i, j, j, :])
                        for j in range(self.part.ndim)
                    ]
                )
                for i in range(self.part.ndim)
            ]

        array_out = np.zeros_like(array_in)
        for i in range(self.part.ndim):
            array_out[i, :] = sum(
                bspline_operations.compute_mf_scalar_gradu_gradv(
                    self.part.quadrule_list,
                    self._stiffness_property[i, j, ...],
                    array_in[j, :],
                )
                for j in range(self.part.ndim)
            )

        return array_out

    def interpolate_strain(
        self, displacement: np.ndarray, convert_to_3d: bool = False
    ) -> np.ndarray:
        "Compute strain field from displacement field"
        ders_par = bspline_operations.eval_jacobien(
            self.part.quadrule_list, displacement
        )
        ders_phy = np.einsum("ijl,jkl->ikl", ders_par, self.part.inv_jac, optimize=True)
        strain = 0.5 * (ders_phy + np.einsum("ijl->jil", ders_phy, optimize=True))
        if convert_to_3d:
            strain_3d = np.zeros((3, 3, np.shape(strain)[-1]))
            strain_3d[: self.part.ndim, : self.part.ndim, :] = strain
            return strain_3d
        else:
            return strain

    def _assemble_internal_force(self, stress: np.ndarray) -> np.ndarray:
        mf = matrixfree(
            np.array(
                [quadrule.nbquadpts for quadrule in self.part.quadrule_list], dtype=int
            )
        )
        prop = np.einsum(
            "ilk,ljk,k->ijk",
            self.part.inv_jac,
            stress[: self.part.ndim, : self.part.ndim, :],
            self.part.det_jac,
            optimize=True,
        )
        array_out = np.zeros((self.part.ndim, self.part.nbctrlpts_total))
        for i in range(self.part.ndim):
            for k in range(self.part.ndim):
                alpha_list = np.ones(self.part.ndim, dtype=int)
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

    def _compute_residual(
        self, displacement: np.ndarray, external_force: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, dict]:

        plastic_vars: dict = kwargs.get("plastic_vars")

        if self.update_properties:
            self.clear_properties()
        convert_to_3d = False if self.part.ndim == 1 else True
        strain = self.interpolate_strain(displacement, convert_to_3d=convert_to_3d)
        stress, mf_args = self.material.return_mapping(strain, plastic_vars)
        residual = external_force - self._assemble_internal_force(stress)
        clean_dirichlet(residual, self.sp_constraint_ctrlpts)
        return residual, mf_args

    def _solve_linearized_system(self, array_in: np.ndarray, **kwargs) -> np.ndarray:

        mf_args: dict = kwargs.get("mf_args")
        inner_tolerance: float = kwargs.get("inner_tolerance")

        if self._update_preconditioner:
            self.preconditioner.add_scalar_space_time_correctors(
                stiffness_corrector=self._scalar_mean_stiffness
            )
        self.preconditioner.update_space_eigenvalues(scalar_coefs=(0, 1))
        output = super()._solve_linear_system(
            self.compute_mf_stiffness,
            array_in,
            Pfun=self.preconditioner.apply_vectorial_preconditioner,
            dotfun=block_dot_product,
            cleanfun=clean_dirichlet,
            dod=self.sp_constraint_ctrlpts,
            args=mf_args,
            tolerance=inner_tolerance,
        )
        self._linear_residual_list.append(output["res"])
        return output["sol"]

    def solve_elastoplasticity(
        self,
        displacement_list: np.ndarray,
        external_force_list: np.ndarray,
        save_plastic_vars: bool = False,
    ) -> dict:

        # Decide if it is a linear or nonlinear problem
        self.update_properties = self.material._activated_plasticity

        nonlinsolv = nonlinsolver(
            maxiters=self._maxiters_nonlinear,
            tolerance=self._tolerance_nonlinear,
            linear_solver_tolerance=self._tolerance_linear,
        )

        if external_force_list.ndim == 2 and displacement_list.ndim == 2:

            print(f"Static linear elastic solver")
            nonlinsolv.solve(
                displacement_list,
                external_force_list,
                self._compute_residual,
                self._solve_linearized_system,
                residual_args={"plastic_vars": {}},
            )

            return

        print(f"Quasi-static elastoplastic solver")
        # Get inactive control points
        constraint_ctrlpts = self.sp_constraint_ctrlpts

        # Time-stepping problem
        all_plastic_vars: List[dict] = []
        plastic_vars: dict = {}
        for i in range(1, external_force_list.shape[-1]):

            # Predict values of new step
            Fext_n1 = np.copy(external_force_list[:, :, i])
            dj_n1 = np.copy(displacement_list[:, :, i - 1])
            for j, dod in enumerate(constraint_ctrlpts):
                dj_n1[j, dod] = displacement_list[j, dod, i]

            print(f"(Pseudo) Time-step: {i}")
            residual_args = {"plastic_vars": plastic_vars}
            nonlinsolv.solve(
                dj_n1,
                Fext_n1,
                self._compute_residual,
                self._solve_linearized_system,
                residual_args=residual_args,
            )

            displacement_list[:, :, i] = np.copy(dj_n1)
            plastic_vars = deepcopy(residual_args["plastic_vars"])
            if save_plastic_vars:
                all_plastic_vars.append(plastic_vars)

        return all_plastic_vars

    def _linearized_explicit_dynamic_solver(
        self, array_in: np.ndarray, mf_args: dict = {}
    ) -> np.ndarray:

        if self.allow_lumping:
            if self._diagonal_mass is None or self._mass_property is None:
                mf_args = self._verify_fun_args(mf_args)
                self._mass_property = self.material.density(mf_args) * self.part.det_jac
                self._diagonal_mass = bspline_operations.assemble_scalar_u_v(
                    self.part.quadrule_list, self._mass_property, allow_lumping=True
                )

            array_out = np.zeros_like(array_in)
            for i in range(self.part.ndim):
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
            output = super()._solve_linear_system(
                self.compute_mf_mass,
                array_in,
                Pfun=self.preconditioner.apply_vectorial_preconditioner,
                dotfun=block_dot_product,
                cleanfun=clean_dirichlet,
                dod=self.sp_constraint_ctrlpts,
                args=mf_args,
            )
            self._linear_residual_list.append(output["res"])
        return output["sol"]

    def solve_explicit_linear_dynamics(
        self,
        displacement_list: np.ndarray,
        external_force_list: np.ndarray,
        time_list: Union[list, np.ndarray],
    ):
        "Solves linear explicit dynamic problem."
        self.update_properties = not (
            self.material._has_uniform_density
        )  # True only if density depends on current plastic variables

        assert len(time_list) > 3, "At least 2 steps"
        self.activate_explicit_dynamics()
        plastic_vars: dict = {}

        def predict_displacement(dis, vel, acc, dt):
            return dis + dt * vel + 0.5 * dt**2 * acc

        def update_velocity(vel, acc_old, acc_new, dt):
            return vel + 0.5 * dt * (acc_old + acc_new)

        def compute_acceleration(problem: mechanical_problem, res, mf_args):
            return problem._linearized_explicit_dynamic_solver(res, mf_args)

        # Clean displacement array
        constraint_ctrlpts = self.sp_constraint_ctrlpts
        for k in range(self.part.ndim):
            displacement_list[k, constraint_ctrlpts[k], :] = 0.0

        # Solve initial static problem
        Fext = np.copy(external_force_list[..., 0])
        d_n0 = np.copy(displacement_list[..., 0])
        v_n0, a_n0 = np.zeros_like(d_n0), np.zeros_like(d_n0)

        for i in range(1, len(time_list)):

            # Get delta time
            dt = time_list[i] - time_list[i - 1]

            # Predict values of new step
            Fext = np.copy(external_force_list[:, :, i])
            d_n1 = predict_displacement(d_n0, v_n0, a_n0, dt)
            a_n1 = np.zeros_like(d_n0)

            # Compute residual and update plastic variables
            residual_args = {"plastic_vars": plastic_vars}
            residual, mf_args = self._compute_residual(d_n1, Fext, **residual_args)
            a_n1 += compute_acceleration(self, residual, mf_args)
            v_n1 = update_velocity(v_n0, a_n0, a_n1, dt)

            # Save data for next step
            displacement_list[:, :, i] = np.copy(d_n1)
            plastic_vars = deepcopy(residual_args["plastic_vars"])
            d_n0 = np.copy(d_n1)
            v_n0 = np.copy(v_n1)
            a_n0 = np.copy(a_n1)

    def solve_eigenvalue_problem(
        self, mf_args: dict = {}, which: str = "LM", k: int = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert np.isscalar(k), "It should be scalar"
        original_condition = deepcopy(self.allow_lumping)
        self.allow_lumping = False  # We only solve eigenproblem with plain matrix

        fastdiag = fastdiagonalization()
        fastdiag.compute_space_eigendecomposition(
            self.part.quadrule_list, np.zeros((self.part.ndim, self.part.ndim, 2))
        )
        fastdiag.add_free_controlpoints(
            [
                np.arange(self.part.nbctrlpts_total, dtype=int)
                for _ in range(self.part.ndim)
            ]
        )
        fastdiag.update_space_eigenvalues(scalar_coefs=(1, 0))

        solv = linsolver(
            tolerance=1e-2
        )  # Use a rough tolerance for the pseudo-preconditioner
        size_array = self.part.ndim * self.part.nbctrlpts_total

        def mass(x):
            x_in = np.reshape(x, newshape=(self.part.ndim, -1), order="F")
            x_out = self.compute_mf_mass(x_in, mf_args=mf_args)
            return np.ravel(x_out, order="F")

        def preconditioner(x):
            output = solv.GMRES(
                self.compute_mf_mass,
                np.reshape(x, newshape=(self.part.ndim, -1), order="F"),
                Pfun=fastdiag.apply_vectorial_preconditioner,
                dotfun=block_dot_product,
                args=mf_args,
            )
            return np.ravel(output["sol"], order="F")

        def stiffness(x):
            x_in = np.reshape(x, newshape=(self.part.ndim, -1), order="F")
            x_out = self.compute_mf_stiffness(x_in, mf_args=mf_args)
            return np.ravel(x_out, order="F")

        k = min(k, size_array - 2)
        eigenvalues, eigenvectors = solv.eigs(
            N=size_array,
            Afun=stiffness,
            Bfun=mass,
            Pfun=preconditioner,
            k=k,
            which=which,
        )
        self.allow_lumping = original_condition

        return eigenvalues, eigenvectors
