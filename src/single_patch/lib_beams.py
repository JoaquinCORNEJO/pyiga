from .lib_job import *


class pipe_problem(problem):

    def __init__(
        self,
        axis_patch: singlepatch,
        circ_patch: singlepatch,
        radial_quadrule: gauss_quadrature,
        boundary: boundary_condition,
        solver_args: dict,
    ):
        assert axis_patch.ndim == 1, "Beam patch can only be one dimensional"
        super().__init__(axis_patch, boundary, solver_args)
        assert circ_patch.ndim == 1, "Circunferential patch can only be one dimensional"
        self.circ_part: singlepatch = circ_patch
        self._quadrature_list: List[quadrature_rule] = (
            self.part.quadrule_list + self.circ_part.quadrule_list
        )
        self.__sin_theta = None
        self.__cos_theta = None

    def _compute_strain_membrane(
        self,
        a: float,
        U_0: np.ndarray,
        V_0: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        w0: np.ndarray,
    ):
        """
        U0 and V0 are axis displacements (only depends on x)
        and u0, v0, w0 are mid-surface displacements (depends on x and theta).
        The first element of quadrule_list constaints the quadrature rule following x and
        the second element the quadrature rule following theta
        """
        axis_nbqp = self.part.nbqp_total
        circ_nbqp = self.circ_part.nbqp_total
        one_like_circ = np.ones(circ_nbqp)

        dU_0dx = self._quadrature_list[0].basis[1].T @ U_0
        d2V_0dx2 = self._quadrature_list[0].basis[2].T @ V_0

        mf = matrixfree([self.part.nbctrlpts_total, self.circ_part.nbctrlpts_total])
        pv0 = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[0]],
            v0,
            istranspose=True,
        )
        pw0 = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[0]],
            w0,
            istranspose=True,
        )
        du0dx = mf.sumfactorization(
            [self._quadrature_list[0].basis[1], self._quadrature_list[1].basis[0]],
            u0,
            istranspose=True,
        )
        dv0dx = mf.sumfactorization(
            [self._quadrature_list[0].basis[1], self._quadrature_list[1].basis[0]],
            v0,
            istranspose=True,
        )
        du0dth = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[1]],
            u0,
            istranspose=True,
        )
        dv0dth = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[1]],
            v0,
            istranspose=True,
        )

        eps = np.zeros((2, 2, axis_nbqp * circ_nbqp))
        eps[0, 0, :] = (
            np.kron(one_like_circ, dU_0dx)
            + du0dx
            - np.kron(self.__sin_theta, d2V_0dx2) * (a + pw0)
            - np.kron(self.__cos_theta, d2V_0dx2) * pv0
        )
        eps[1, 1, :] = 1 / a * (pw0 + dv0dth)
        eps[0, 1, :] = 1 / 2 * (1 / a * du0dth + dv0dx)
        eps[1, 0, :] = eps[0, 1, :]
        return eps

    def _compute_strain_z_linear(
        self,
        a: float,
        U_0: np.ndarray,
        V_0: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        w0: np.ndarray,
    ):

        axis_nbqp = self.part.nbqp_total
        circ_nbqp = self.circ_part.nbqp_total
        one_like_circ = np.ones(circ_nbqp)

        d2V_0dx2 = self._quadrature_list[0].basis[2].T @ V_0

        mf = matrixfree([self.part.nbctrlpts_total, self.circ_part.nbctrlpts_total])
        pv0 = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[0]],
            v0,
            istranspose=True,
        )
        pw0 = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[0]],
            w0,
            istranspose=True,
        )
        dv0dx = mf.sumfactorization(
            [self._quadrature_list[0].basis[1], self._quadrature_list[1].basis[0]],
            v0,
            istranspose=True,
        )
        du0dth = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[1]],
            u0,
            istranspose=True,
        )
        dw0dth = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[1]],
            w0,
            istranspose=True,
        )
        d2w0dx2 = mf.sumfactorization(
            [self._quadrature_list[0].basis[2], self._quadrature_list[1].basis[0]],
            w0,
            istranspose=True,
        )
        d2w0dth2 = mf.sumfactorization(
            [self._quadrature_list[0].basis[0], self._quadrature_list[1].basis[2]],
            w0,
            istranspose=True,
        )
        d2w0dxdth = mf.sumfactorization(
            [self._quadrature_list[0].basis[1], self._quadrature_list[1].basis[1]],
            w0,
            istranspose=True,
        )

        eps = np.zeros((2, 2, axis_nbqp * circ_nbqp))
        eps[0, 0, :] = (
            -np.kron(self.__sin_theta, d2V_0dx2)
            - np.kron(one_like_circ, d2w0dx2)
            + 1 / a * np.kron(self.__cos_theta, d2V_0dx2) * (dw0dth - pv0)
        )
        eps[1, 1, :] = -1 / a**2 * (d2w0dth2 + pw0)
        eps[0, 1, :] = -1 / (2 * a) * (2 * d2w0dxdth + 1 / a * du0dth - dv0dx)
        eps[1, 0, :] = eps[0, 1, :]
        return eps

    def _compute_mf_inertia(self, array_in, a: float):
        "Here u represents the acceleration at control points"
        axis_nbqp = self.part.nbqp_total
        circ_nbqp = self.circ_part.nbqp_total
        one_like_axis = np.ones(axis_nbqp)
        one_like_circ = np.ones(circ_nbqp)
        mf_rows = matrixfree(
            [self.part.nbctrlpts_total, self.circ_part.nbctrlpts_total]
        )
        mf_cols = matrixfree([axis_nbqp, circ_nbqp])
        t_0, t_1, t_2, t_3, t_4, t_5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        t_0 += (
            self._quadrature_list[0].weights[0]
            @ ...
            @ (self._quadrature_list[0].basis[0].T @ array_in[0])
        )
        #
        t_1 += (
            self._quadrature_list[0].weights[0]
            @ ...
            @ (self._quadrature_list[0].basis[0].T @ array_in[1])
        )
        t_1 += -(
            self._quadrature_list[0].weights[0]
            @ ...
            @ (self._quadrature_list[0].basis[2].T @ array_in[1])
        )
        t_1 += (
            self._quadrature_list[0].weights[0]
            @ ...
            @ (
                np.kron(self.__sin_theta, one_like_axis)
                * mf_rows.sumfactorization(
                    [
                        self._quadrature_list[0].basis[0],
                        self._quadrature_list[1].basis[0],
                    ],
                    array_in[4],
                    istranspose=True,
                )
                + np.kron(self.__cos_theta, one_like_axis)
                * mf_rows.sumfactorization(
                    [
                        self._quadrature_list[0].basis[0],
                        self._quadrature_list[1].basis[0],
                    ],
                    array_in[3],
                    istranspose=True,
                )
            )
        )
        #
        t_2 += mf_cols.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[0],
            ],
            ...
            @ mf_rows.sumfactorization(
                [
                    self._quadrature_list[0].basis[0],
                    self._quadrature_list[1].basis[0],
                ],
                array_in[2],
                istranspose=True,
            ),
        )
        #
        t_3 += mf_cols.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[0],
            ],
            ...
            @ mf_rows.sumfactorization(
                [
                    self._quadrature_list[0].basis[0],
                    self._quadrature_list[1].basis[0],
                ],
                array_in[3],
                istranspose=True,
            ),
        )
        t_3 += mf_cols.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[0],
            ],
            ... @ (self._quadrature_list[0].basis[0].T @ array_in[1]),
        )
        t_3 += -mf_cols.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[0],
            ],
            ...
            @ (
                mf_rows.sumfactorization(
                    [
                        self._quadrature_list[0].basis[0],
                        self._quadrature_list[1].basis[1],
                    ],
                    array_in[4],
                    istranspose=True,
                )
                - mf_rows.sumfactorization(
                    [
                        self._quadrature_list[0].basis[0],
                        self._quadrature_list[1].basis[0],
                    ],
                    array_in[3],
                    istranspose=True,
                )
            ),
        )
        #
        t_4 += mf_cols.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[0],
            ],
            ...
            @ mf_rows.sumfactorization(
                [
                    self._quadrature_list[0].basis[0],
                    self._quadrature_list[1].basis[0],
                ],
                array_in[4],
                istranspose=True,
            ),
        )
        t_4 += mf_cols.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[0],
            ],
            ... @ (self._quadrature_list[0].basis[0].T @ array_in[1]),
        )
        t_4 += -mf_cols.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[0],
            ],
            ...
            @ mf_rows.sumfactorization(
                [
                    self._quadrature_list[0].basis[2],
                    self._quadrature_list[1].basis[0],
                ],
                array_in[4],
                istranspose=True,
            ),
        )
        t_4 += -mf_cols.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[0],
            ],
            ...
            @ (
                mf_rows.sumfactorization(
                    [
                        self._quadrature_list[0].basis[0],
                        self._quadrature_list[1].basis[2],
                    ],
                    array_in[4],
                    istranspose=True,
                )
                - mf_rows.sumfactorization(
                    [
                        self._quadrature_list[0].basis[0],
                        self._quadrature_list[1].basis[1],
                    ],
                    array_in[3],
                    istranspose=True,
                )
            ),
        )
        return [t_0, t_1, t_2, t_3, t_4, t_5]

    def _compute_mf_lagrange(self, array_in, istranspose=False):
        "Here u represents the Lagrange multipliers"
        return

    def _assemble_internal_force(self, stress_m, stress_k, array_in):
        axis_nbqp = self.part.nbqp_total
        circ_nbqp = self.circ_part.nbqp_total
        one_like_axis = np.ones(axis_nbqp)
        one_like_circ = np.ones(circ_nbqp)
        mf = matrixfree([axis_nbqp, circ_nbqp])
        t_0, t_1, t_2, t_3, t_4, t_5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        t_0 += mf.sumfactorization(
            [
                self._quadrature_list[0].weights[1],
                self._quadrature_list[1].weights[0],
            ],
            (... @ stress_m[0, 0, :]),
        )
        #
        t_1 += -mf.sumfactorization(
            [
                self._quadrature_list[0].weights[2],
                self._quadrature_list[1].weights[0],
            ],
            (... @ (stress_m[0, 0, :] + stress_k[0, 0, :])),
        )
        #
        t_2 += mf.sumfactorization(
            [
                self._quadrature_list[0].weights[1],
                self._quadrature_list[1].weights[0],
            ],
            (... @ stress_m[0, 0, :]),
        )
        t_2 += mf.sumfactorization(
            [
                self._quadrature_list[0].weights[0],
                self._quadrature_list[1].weights[1],
            ],
            (... @ stress_m[0, 1, :] - ... @ stress_k[0, 1, :]),
        )
        return
