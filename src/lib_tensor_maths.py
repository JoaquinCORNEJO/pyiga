from .__init__ import *
from .lib_quadrules import quadrature_rule
from typing import Union, List, Tuple


class matrixfree:
    def __init__(self, nclist: list):
        self._tensor_ndim: int = 4
        self._tensor_original_shape: list = [
            int(nclist[i]) if i < len(nclist) else 1 for i in range(self._tensor_ndim)
        ]
        self._tensor_value: Union[None, np.ndarray] = None

    def _tensor_matrix_product(
        self,
        matrix: Union[sp.csr_matrix, np.ndarray],
        mode: int,
        istranspose: bool = False,
    ):
        if sp.issparse(matrix):

            if istranspose:
                matrix = matrix.T
            old_shape = self._tensor_value.shape
            order = [mode] + [i for i in range(self._tensor_ndim) if i != mode]
            tensor_reshaped = np.reshape(
                np.transpose(self._tensor_value, axes=order),
                newshape=(old_shape[mode], -1),
            )
            tensor_transformed = matrix @ tensor_reshaped
            new_shape = [matrix.shape[0]] + [
                old_shape[i] for i in range(self._tensor_ndim) if i != mode
            ]
            rearanged_order = np.argsort(order)
            self._tensor_value = np.transpose(
                np.reshape(tensor_transformed, newshape=new_shape), axes=rearanged_order
            )

        else:

            einsum_patterns = [
                ("im,mjkl->ijkl", "mi,mjkl->ijkl"),
                ("jm,imkl->ijkl", "mj,imkl->ijkl"),
                ("km,ijml->ijkl", "mk,ijml->ijkl"),
                ("lm,ijkm->ijkl", "ml,ijkm->ijkl"),
            ]

            pattern = einsum_patterns[mode][1 if istranspose else 0]
            self._tensor_value = np.einsum(
                pattern, matrix, self._tensor_value, optimize=True
            )

    def sumfactorization(
        self,
        matrix_list: List[Union[sp.csr_matrix, np.ndarray]],
        array_in: np.ndarray,
        istranspose: bool = False,
    ) -> np.ndarray:
        assert len(matrix_list) <= self._tensor_ndim, "Dimension problem"
        self._tensor_value = np.reshape(
            array_in, self._tensor_original_shape, order="F"
        )

        for i, matrix in enumerate(matrix_list):
            self._tensor_matrix_product(matrix, mode=i, istranspose=istranspose)
        return np.ravel(self._tensor_value, order="F")


def eval_inverse_and_determinant(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert matrix.ndim > 2, "Only for 3-dimensional matrices"
    nm, ndim = np.shape(matrix)[:2]
    other_dim = matrix.shape[2:]
    inv = np.zeros(shape=(ndim, nm) + other_dim)
    if nm == ndim:
        if nm == 1:
            det = np.abs(matrix[0, 0, ...])
            inv = np.ones((1, 1, *np.shape(det)))
        elif nm == 2:
            det = (
                matrix[0, 0, ...] * matrix[1, 1, ...]
                - matrix[0, 1, ...] * matrix[1, 0, ...]
            )
            inv[0, 0, ...] = matrix[1, 1, ...]
            inv[1, 1, ...] = matrix[0, 0, ...]
            inv[0, 1, ...] = -matrix[0, 1, ...]
            inv[1, 0, ...] = -matrix[1, 0, ...]
        elif nm == 3:
            det = (
                matrix[0, 1, ...] * matrix[1, 2, ...] * matrix[2, 0, ...]
                - matrix[0, 2, ...] * matrix[1, 1, ...] * matrix[2, 0, ...]
                + matrix[0, 2, ...] * matrix[1, 0, ...] * matrix[2, 1, ...]
                - matrix[0, 0, ...] * matrix[1, 2, ...] * matrix[2, 1, ...]
                + matrix[0, 0, ...] * matrix[1, 1, ...] * matrix[2, 2, ...]
                - matrix[0, 1, ...] * matrix[1, 0, ...] * matrix[2, 2, ...]
            )
            inv[0, 0, ...] = (
                matrix[1, 1, ...] * matrix[2, 2, ...]
                - matrix[1, 2, ...] * matrix[2, 1, ...]
            )
            inv[0, 1, ...] = (
                matrix[0, 2, ...] * matrix[2, 1, ...]
                - matrix[0, 1, ...] * matrix[2, 2, ...]
            )
            inv[0, 2, ...] = (
                matrix[0, 1, ...] * matrix[1, 2, ...]
                - matrix[0, 2, ...] * matrix[1, 1, ...]
            )
            inv[1, 0, ...] = (
                matrix[1, 2, ...] * matrix[2, 0, ...]
                - matrix[1, 0, ...] * matrix[2, 2, ...]
            )
            inv[1, 1, ...] = (
                matrix[0, 0, ...] * matrix[2, 2, ...]
                - matrix[0, 2, ...] * matrix[2, 0, ...]
            )
            inv[1, 2, ...] = (
                matrix[0, 2, ...] * matrix[1, 0, ...]
                - matrix[0, 0, ...] * matrix[1, 2, ...]
            )
            inv[2, 0, ...] = (
                matrix[1, 0, ...] * matrix[2, 1, ...]
                - matrix[1, 1, ...] * matrix[2, 0, ...]
            )
            inv[2, 1, ...] = (
                matrix[0, 1, ...] * matrix[2, 0, ...]
                - matrix[0, 0, ...] * matrix[2, 1, ...]
            )
            inv[2, 2, ...] = (
                matrix[0, 0, ...] * matrix[1, 1, ...]
                - matrix[0, 1, ...] * matrix[1, 0, ...]
            )
        else:
            raise Warning("Not coded")
        inv /= det
    else:
        det = np.zeros(shape=other_dim)
        for idx in np.ndindex(other_dim):
            inv[(...,) + idx] = np.linalg.pinv(matrix[(...,) + idx])
            det[idx] = np.sqrt(
                np.linalg.det(matrix[(...,) + idx].T @ matrix[(...,) + idx])
            )
    return det, inv


class bspline_operations:

    def _set_matrix_size_from_quadrature(
        quadrule_list: List[quadrature_rule],
    ) -> Tuple[np.ndarray, np.ndarray]:
        nr_list = np.array(
            [quadrule.nbctrlpts for quadrule in quadrule_list], dtype=int
        )
        nc_list = np.array([quadrule.nbqp for quadrule in quadrule_list], dtype=int)
        return nr_list, nc_list

    def eval_jacobien(
        quadrule_list: List[quadrature_rule],
        u_ctrlpts: np.ndarray,
        knots_list: Union[None, List[np.ndarray]] = None,
    ) -> np.ndarray:
        if knots_list is None:
            knots_list = [quadrule.quadpts for quadrule in quadrule_list]
            basis_list = [quadrule.basis for quadrule in quadrule_list]
        else:
            basis_list = [
                quadrule.get_sample_basis(knots)
                for quadrule, knots in zip(quadrule_list, knots_list)
            ]

        nm = np.shape(u_ctrlpts)[0]
        ndim = len(quadrule_list)
        nr_list = bspline_operations._set_matrix_size_from_quadrature(quadrule_list)[0]
        nc_list = np.array([len(knots) for knots in knots_list], dtype=int)
        mf = matrixfree(nr_list)
        jac = np.zeros((nm, ndim, np.prod(nc_list)))
        for j in range(ndim):
            beta_list = np.zeros(ndim, dtype=int)
            beta_list[j] = 1
            for i in range(nm):
                jac[i, j, :] = mf.sumfactorization(
                    [basis[beta] for basis, beta in zip(basis_list, beta_list)],
                    u_ctrlpts[i, :],
                    istranspose=True,
                )

        return jac

    def interpolate_meshgrid(
        quadrule_list: List[quadrature_rule],
        u_ctrlpts: np.ndarray,
        knots_list: Union[None, List[np.ndarray]] = None,
    ) -> np.ndarray:
        if knots_list is None:
            knots_list = [quadrule.quadpts for quadrule in quadrule_list]
            basis_list = [quadrule.basis for quadrule in quadrule_list]
        else:
            basis_list = [
                quadrule.get_sample_basis(knots)
                for quadrule, knots in zip(quadrule_list, knots_list)
            ]

        nm = np.shape(u_ctrlpts)[0]
        nr_list = bspline_operations._set_matrix_size_from_quadrature(quadrule_list)[0]
        nc_list = np.array([len(knots) for knots in knots_list], dtype=int)
        mf = matrixfree(nr_list)
        u_interp = np.zeros((nm, np.prod(nc_list)))
        for i in range(nm):
            u_interp[i, :] = mf.sumfactorization(
                [basis[0] for basis in basis_list], u_ctrlpts[i, :], istranspose=True
            )

        return u_interp

    def _spkron_product_quadrature(
        quadrule_list: List[quadrature_rule], idx_list: list, product_type: str
    ) -> sp.csr_matrix:
        assert len(quadrule_list) == len(idx_list), "Size problem"
        if product_type.lower() == "basis":
            matrix = quadrule_list[-1].basis[idx_list[-1]]
            for i in range(len(idx_list) - 2, -1, -1):
                matrix = sp.kron(quadrule_list[i].basis[idx_list[i]], matrix)
        elif product_type.lower() == "weights":
            matrix = quadrule_list[-1].weights[idx_list[-1]]
            for i in range(len(idx_list) - 2, -1, -1):
                matrix = sp.kron(quadrule_list[i].weights[idx_list[i]], matrix)
        return matrix

    def assemble_scalar_u_v(
        quadrule_list: List[quadrature_rule],
        coefficients: np.ndarray,
        allow_lumping: bool,
    ) -> sp.csr_matrix:
        if allow_lumping:
            nc_list = bspline_operations._set_matrix_size_from_quadrature(
                quadrule_list
            )[1]
            mf = matrixfree(nc_list)
            matrix = mf.sumfactorization(
                [quadrule.weights[0] for quadrule in quadrule_list],
                coefficients,
                istranspose=False,
            )
        else:
            zero_list = np.zeros(len(quadrule_list), dtype=int)
            tmp1 = bspline_operations._spkron_product_quadrature(
                quadrule_list, zero_list, product_type="basis"
            ).T
            tmp2 = sp.diags(coefficients) @ tmp1
            matrix = (
                bspline_operations._spkron_product_quadrature(
                    quadrule_list, zero_list, product_type="weights"
                )
                @ tmp2
            )
        return matrix

    def assemble_scalar_gradu_gradv(
        quadrule_list: List[quadrature_rule], coefficients: np.ndarray
    ) -> sp.csr_matrix:
        ndim = len(quadrule_list)
        nr_list = bspline_operations._set_matrix_size_from_quadrature(quadrule_list)[0]
        matrix = sp.csr_matrix((np.product(nr_list), np.product(nr_list)))
        for j in range(ndim):
            beta_list = np.zeros(ndim, dtype=int)
            beta_list[j] = 1
            tmp1 = bspline_operations._spkron_product_quadrature(
                quadrule_list, beta_list, product_type="basis"
            ).T
            for i in range(ndim):
                alpha_list = np.zeros(ndim, dtype=int)
                alpha_list[i] = 1
                zeta_list = beta_list + 2 * alpha_list
                tmp2 = sp.diags(coefficients[i, j, :]) @ tmp1
                matrix += (
                    bspline_operations._spkron_product_quadrature(
                        quadrule_list, zeta_list, product_type="weights"
                    )
                    @ tmp2
                )
        return matrix

    def assemble_scalar_u_force(
        quadrule_list: List[quadrature_rule], coefficients: np.ndarray
    ) -> np.ndarray:
        nm = np.shape(coefficients)[0]
        nr_list, nc_list = bspline_operations._set_matrix_size_from_quadrature(
            quadrule_list
        )
        mf = matrixfree(nc_list)
        array_out = np.zeros((nm, np.product(nr_list)))
        for i in range(nm):
            array_out[i, :] = mf.sumfactorization(
                [quadrule.weights[0] for quadrule in quadrule_list],
                coefficients[i, :],
                istranspose=False,
            )
        return array_out

    def compute_mf_scalar_u_v(
        quadrule_list: List[quadrature_rule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        allow_lumping: bool,
        enable_spacetime: bool = False,
        time_ders: Union[tuple, None] = None,
    ) -> np.ndarray:
        nr_list, nc_list = bspline_operations._set_matrix_size_from_quadrature(
            quadrule_list
        )
        mf_cols = matrixfree(nc_list)

        if allow_lumping:
            matrix_lumped = mf_cols.sumfactorization(
                [quadrule.weights[0] for quadrule in quadrule_list],
                coefficients,
                istranspose=False,
            )
            array_out = matrix_lumped * array_in
        else:
            if enable_spacetime:
                assert isinstance(
                    time_ders, tuple
                ), "Derivatives of time functions should be tuple"
            mf_rows = matrixfree(nr_list)
            zero_list = np.zeros(len(quadrule_list), dtype=int)
            if enable_spacetime:
                zero_list[-1] = time_ders[1]
            array_tmp = mf_rows.sumfactorization(
                [
                    quadrule.basis[beta]
                    for quadrule, beta in zip(quadrule_list, zero_list)
                ],
                array_in,
                istranspose=True,
            )

            if enable_spacetime:
                zero_list[-1] = time_ders[0]
            array_out = mf_cols.sumfactorization(
                [
                    quadrule.weights[alpha]
                    for quadrule, alpha in zip(quadrule_list, zero_list)
                ],
                array_tmp * coefficients,
                istranspose=False,
            )
        return array_out

    def compute_mf_scalar_gradu_gradv(
        quadrule_list: List[quadrature_rule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        enable_spacetime: bool = False,
        time_ders: Union[tuple, None] = None,
    ) -> np.ndarray:
        if enable_spacetime:
            assert isinstance(
                time_ders, tuple
            ), "Derivatives of time functions should be tuple"
        ndim = len(quadrule_list)
        sp_ndim = ndim - 1 if enable_spacetime else ndim
        nr_list, nc_list = bspline_operations._set_matrix_size_from_quadrature(
            quadrule_list
        )
        mf_rows = matrixfree(nr_list)
        mf_cols = matrixfree(nc_list)
        array_out = np.zeros_like(array_in)
        for j in range(sp_ndim):
            beta_list = np.zeros(ndim, dtype=int)
            beta_list[j] = 1
            if enable_spacetime:
                beta_list[-1] = time_ders[1]
            array_tmp = mf_rows.sumfactorization(
                [
                    quadrule.basis[beta]
                    for quadrule, beta in zip(quadrule_list, beta_list)
                ],
                array_in,
                istranspose=True,
            )
            for i in range(sp_ndim):
                alpha_list = np.zeros(ndim, dtype=int)
                alpha_list[i] = 1
                if enable_spacetime:
                    alpha_list[-1] = time_ders[0]
                zeta_list = beta_list + 2 * alpha_list
                array_out += mf_cols.sumfactorization(
                    [
                        quadrule.weights[zeta]
                        for quadrule, zeta in zip(quadrule_list, zeta_list)
                    ],
                    array_tmp * coefficients[i, j, :],
                    istranspose=False,
                )
        return array_out

    def compute_mf_scalar_gradu_v(
        quadrule_list: List[quadrature_rule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        enable_spacetime: bool = False,
        time_ders: Union[tuple, None] = None,
    ) -> np.ndarray:
        if enable_spacetime:
            assert isinstance(
                time_ders, tuple
            ), "Derivatives of time functions should be tuple"
        ndim = len(quadrule_list)
        sp_ndim = ndim - 1 if enable_spacetime else ndim
        nr_list, nc_list = bspline_operations._set_matrix_size_from_quadrature(
            quadrule_list
        )
        mf_rows = matrixfree(nr_list)
        mf_cols = matrixfree(nc_list)
        array_out = np.zeros_like(array_in)

        beta_list = np.zeros(ndim, dtype=int)
        if enable_spacetime:
            beta_list[-1] = time_ders[1]
        array_tmp = mf_rows.sumfactorization(
            [quadrule.basis[beta] for quadrule, beta in zip(quadrule_list, beta_list)],
            array_in,
            istranspose=True,
        )
        for i in range(sp_ndim):
            alpha_list = np.zeros(ndim, dtype=int)
            alpha_list[i] = 1
            if enable_spacetime:
                alpha_list[-1] = time_ders[0]
            zeta_list = beta_list + 2 * alpha_list
            array_out += mf_cols.sumfactorization(
                [
                    quadrule.weights[zeta]
                    for quadrule, zeta in zip(quadrule_list, zeta_list)
                ],
                array_tmp * coefficients[i, :],
                istranspose=False,
            )
        return array_out


def combine_arrays(array_list: List[np.ndarray], coefs: np.ndarray) -> np.ndarray:
    ndim = len(array_list)
    nnz = [len(v) for v in array_list]
    if ndim == 1:
        v_out = coefs[0] * array_list[0]
    elif ndim == 2:
        v_out = coefs[0] * np.kron(np.ones(nnz[1]), array_list[0]) + coefs[1] * np.kron(
            array_list[1], np.ones(nnz[0])
        )
    elif ndim == 3:
        v_out = (
            coefs[0] * np.kron(np.ones(nnz[2]), np.kron(np.ones(nnz[1]), array_list[0]))
            + coefs[1]
            * np.kron(np.ones(nnz[2]), np.kron(array_list[1], np.ones(nnz[0])))
            + coefs[2]
            * np.kron(array_list[2], np.kron(np.ones(nnz[1]), np.ones(nnz[0])))
        )
    return v_out


class fastdiagonalization:

    def __init__(self):
        self._nbvars: int = 0
        self._nbdirs: int = 0
        self._nnz_space: List[int] = []
        self._nnz_time: List[int] = []
        self._free_ctrlpts: List[list] = [[]]
        #
        self.eigenvec_by_dir_space: List[np.ndarray] = []
        self.eigenval_by_dir_space: List[np.ndarray] = []
        self._space_eigenvalues_mixed_original: List[np.ndarray] = []
        self.fastdiag_eigenvalues: List[np.ndarray] = []
        #
        self._mass_space_corrector: List[float] = []
        self._stiff_space_corrector: List[np.ndarray] = []
        self._advection_time_corrector: List[float] = []
        #
        self._activate_stiffness_correction: bool = False
        self._activate_mass_correction: bool = False
        self._activate_advection_correction: bool = False

    def add_free_controlpoints(self, free_ctrlpts):
        self._free_ctrlpts = free_ctrlpts

    def compute_space_eigendecomposition(
        self,
        space_quadrule_list: List[quadrature_rule],
        space_table_dirichlet: np.ndarray,
    ):
        mass_list: List[Union[sp.csr_matrix, np.ndarray]] = [
            quadrule.weights[0] @ quadrule.basis[0].T
            for quadrule in space_quadrule_list
        ]
        stiff_list: List[Union[sp.csr_matrix, np.ndarray]] = [
            quadrule.weights[-1] @ quadrule.basis[-1].T
            for quadrule in space_quadrule_list
        ]
        self._nbvars, nbdirs, _ = np.shape(space_table_dirichlet)
        self._nbdirs = np.min([nbdirs, len(space_quadrule_list)])
        for ii in range(self._nbvars):
            eigvecs_by_dir, eigvals_by_dir, nnz_by_dir = [], [], []
            self._mass_space_corrector.append(1.0)
            self._stiff_space_corrector.append(np.ones(self._nbdirs))
            for jj in range(self._nbdirs):
                mass: Union[sp.csr_matrix, np.ndarray] = mass_list[jj]
                stiff: Union[sp.csr_matrix, np.ndarray] = stiff_list[jj]
                if not isinstance(stiff, np.ndarray):
                    stiff = stiff.toarray()
                if not isinstance(mass, np.ndarray):
                    mass = mass.toarray()
                inf_index, sup_index = 0, np.shape(mass)[0]
                if space_table_dirichlet[ii, jj, 0]:
                    inf_index += 1
                if space_table_dirichlet[ii, jj, 1]:
                    sup_index -= 1
                indices = np.arange(inf_index, sup_index, dtype=int)
                eigvals, eigvecs, _ = sclin.lapack.dsygvd(
                    a=stiff[np.ix_(indices, indices)], b=mass[np.ix_(indices, indices)]
                )
                eigvecs_by_dir.append(np.real(eigvecs))
                eigvals_by_dir.append(np.real(eigvals))
                nnz_by_dir.append(len(indices))
            self.eigenvec_by_dir_space.append(eigvecs_by_dir)
            self.eigenval_by_dir_space.append(eigvals_by_dir)
            self._space_eigenvalues_mixed_original.append(
                combine_arrays(eigvals_by_dir, np.ones(self._nbdirs))
            )
            self._nnz_space.append(nnz_by_dir)

    def compute_time_schurdecomposition(self, quadrule: quadrature_rule):
        def zselect(alpha, beta):
            return zselect

        mass: Union[sp.csr_matrix, np.ndarray] = (
            quadrule.weights[0] @ quadrule.basis[0].T
        )
        adv: Union[sp.csr_matrix, np.ndarray] = (
            quadrule.weights[1] @ quadrule.basis[1].T
        )
        if not isinstance(adv, np.ndarray):
            adv = adv.toarray()
        if not isinstance(mass, np.ndarray):
            mass = mass.toarray()
        indices = np.arange(
            1, np.shape(mass)[0], dtype=int
        )  # We always assume that time is constraint at the begining
        output = sclin.lapack.zgges(
            zselect, a=adv[np.ix_(indices, indices)], b=mass[np.ix_(indices, indices)]
        )
        self._schur_VSL = output[5]
        self._schur_VSR = output[6]
        self._schur_adv = output[0]
        self._schur_mass = output[1]
        self._nnz_time.append(len(indices))

    def update_space_eigenvalues(self, scalar_coefs: tuple):
        self.fastdiag_eigenvalues = []
        for var in range(self._nbvars):
            if self._activate_stiffness_correction:
                eigenvalues_mixed = combine_arrays(
                    self.eigenval_by_dir_space[var], self._stiff_space_corrector[var]
                )
            else:
                eigenvalues_mixed = self._space_eigenvalues_mixed_original[var]

            if self._activate_mass_correction:
                scalar_coefs_0 = scalar_coefs[0] * self._mass_space_corrector[var]
            else:
                scalar_coefs_0 = scalar_coefs[0]

            self.fastdiag_eigenvalues.append(
                scalar_coefs_0 * np.ones_like(eigenvalues_mixed)
                + scalar_coefs[1] * eigenvalues_mixed
            )

    def add_scalar_space_time_correctors(
        self,
        mass_corrector: Union[List[float], None] = None,
        stiffness_corrector: Union[List[np.ndarray], None] = None,
        advection_corrector: Union[List[float], None] = None,
    ):
        if stiffness_corrector is not None:
            assert (
                len(stiffness_corrector) >= self._nbvars
            ), "Stiffness corrector length must be at least the number of variables"
            self._stiff_space_corrector = stiffness_corrector
            self._activate_stiffness_correction = True
        if mass_corrector is not None:
            assert (
                len(mass_corrector) >= self._nbvars
            ), "Mass corrector length must be at least the number of variables"
            self._mass_space_corrector = mass_corrector
            self._activate_mass_correction = True
        if advection_corrector is not None:
            assert (
                len(advection_corrector) >= self._nbvars
            ), "Advection corrector length must be at least the number of variables"
            self._advection_time_corrector = advection_corrector
            self._activate_advection_correction = True

    def apply_scalar_preconditioner(self, array_in: np.ndarray) -> np.ndarray:
        array_out = np.zeros_like(array_in)
        mf = matrixfree(self._nnz_space[0])
        array = mf.sumfactorization(
            self.eigenvec_by_dir_space[0],
            array_in[self._free_ctrlpts[0]],
            istranspose=True,
        )
        array /= self.fastdiag_eigenvalues[0]
        array_out[self._free_ctrlpts[0]] = mf.sumfactorization(
            self.eigenvec_by_dir_space[0], array, istranspose=False
        )
        return array_out

    def apply_vectorial_preconditioner(self, array_in: np.ndarray) -> np.ndarray:
        array_out = np.zeros_like(array_in)
        for i in range(self._nbvars):
            mf = matrixfree(self._nnz_space[i])
            array = mf.sumfactorization(
                self.eigenvec_by_dir_space[i],
                array_in[i, self._free_ctrlpts[i]],
                istranspose=True,
            )
            array /= self.fastdiag_eigenvalues[i]
            array_out[i, self._free_ctrlpts[i]] = mf.sumfactorization(
                self.eigenvec_by_dir_space[i], array, istranspose=False
            )
        return array_out

    def apply_spacetime_scalar_preconditioner(self, array_in: np.ndarray) -> np.ndarray:
        adv_corrector = (
            self._advection_time_corrector[0]
            if self._activate_advection_correction
            else 1.0
        )
        array_out = np.zeros_like(array_in)
        mf = matrixfree(tuple(self._nnz_space[0] + self._nnz_time))
        array1 = mf.sumfactorization(
            self.eigenvec_by_dir_space[0] + [np.conj(self._schur_VSL)],
            array_in[self._free_ctrlpts[0]],
            istranspose=True,
        )

        array1_reshape = np.reshape(array1, newshape=(-1, self._nnz_time[0]), order="F")
        array2_reshape = np.zeros_like(array1_reshape)
        for idx, row in enumerate(array1_reshape):
            mat = (
                adv_corrector * self._schur_adv
                + self.fastdiag_eigenvalues[0][idx] * self._schur_mass
            )
            array2_reshape[idx, :] = sclin.lapack.ztrtrs(mat, row, lower=False)[0]

        array_out[self._free_ctrlpts[0]] = np.real(
            mf.sumfactorization(
                self.eigenvec_by_dir_space[0] + [self._schur_VSR],
                np.ravel(array2_reshape, order="F"),
                istranspose=False,
            )
        )
        return array_out
