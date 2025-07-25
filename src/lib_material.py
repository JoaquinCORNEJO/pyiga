from .__init__ import *
from typing import Callable, Union, Tuple
from abc import ABC, abstractmethod
from src.lib_nonlinear_solver import nonlinsolver


def compute_double_contraction(
    tensor_1: np.ndarray, tensor_2: np.ndarray
) -> np.ndarray:
    assert tensor_1.shape == tensor_2.shape, "Array shapes must match"
    return np.sum(tensor_1 * tensor_2, axis=(0, 1))


def compute_norm_tensor(tensor: np.ndarray) -> np.ndarray:
    return np.sqrt(compute_double_contraction(tensor, tensor))


def compute_trace(tensor: np.ndarray) -> np.ndarray:
    assert tensor.shape[0] == tensor.shape[1], "Tensor must be square"
    return sum(tensor[i, i, ...] for i in range(tensor.shape[0]))


def compute_deviatoric(tensor: np.ndarray) -> np.ndarray:
    assert tensor.shape[0] == tensor.shape[1], "Tensor must be square"
    trace: np.ndarray = compute_trace(tensor) / 3.0
    deviatoric: np.ndarray = np.copy(tensor)
    for i in range(tensor.shape[0]):
        deviatoric[i, i, ...] -= trace[...]
    return deviatoric


class material:
    def __init__(self):
        self._has_uniform_density: bool = False
        self.add_density(1.0, is_uniform=True)  # By default

    def set_scalar_property(
        self, inpt: Union[Callable, float], is_uniform: bool = False
    ) -> Callable:
        if is_uniform:
            if np.isscalar(inpt):
                func: Callable = lambda args: inpt * np.ones(
                    shape=args["shape_quadpts"]
                )
            elif callable(inpt):
                func: Callable = lambda args: inpt(args)
            else:
                raise Warning("Not implemented")
        else:
            assert callable(inpt), "Not implemented"
            func: Callable = lambda args: inpt(args)
        return func

    def set_tensor_property(
        self,
        inpt: Union[Callable, float, np.ndarray],
        ndim: int = 2,
        is_uniform: bool = False,
    ) -> Callable:
        def broadcast(inpt: np.ndarray, shape_tensor: int, shape_quadpts: np.ndarray):
            tensor = np.zeros(shape=(shape_tensor, shape_tensor, *shape_quadpts))
            for i in range(shape_tensor):
                for j in range(shape_tensor):
                    tensor[i, j, ...] = inpt[i, j]
            return tensor

        if is_uniform:
            if np.isscalar(inpt):
                func: Callable = lambda args: broadcast(
                    inpt * np.eye(ndim), ndim, args["shape_quadpts"]
                )
            elif isinstance(inpt, np.ndarray):
                func: Callable = lambda args: broadcast(
                    inpt, ndim, args["shape_quadpts"]
                )
            elif callable(inpt):
                func: Callable = lambda args: inpt(args)
            else:
                raise Warning("Not implemented")
        else:
            assert callable(inpt), "Not implemented"
            func: Callable = lambda args: inpt(args)
        return func

    def add_density(self, inpt: Union[Callable, float], is_uniform: bool):
        if is_uniform:
            self._has_uniform_density = True
        self.density: Callable = self.set_scalar_property(inpt, is_uniform=is_uniform)


class heat_transfer_mat(material):
    """
    In our work we consider nonlinar materials, i.e. its thermal properties
    could change depending on the position, temperature, etc.
    """

    def __init__(self):
        super().__init__()
        self.capacity: Union[Callable, None] = None
        self.conductivity: Union[Callable, None] = None
        self._has_uniform_capacity: bool = False
        self._has_uniform_conductivity: bool = False

    def add_capacity(self, inpt: Union[Callable, float], is_uniform: bool):
        if is_uniform:
            self._has_uniform_capacity = True
        self.capacity = self.set_scalar_property(inpt, is_uniform=is_uniform)

    def add_conductivity(
        self,
        inpt: Union[Callable, float, np.ndarray],
        is_uniform: bool,
        ndim: int,
    ):
        if is_uniform:
            self._has_uniform_conductivity = True
        self.conductivity = self.set_tensor_property(
            inpt, ndim=ndim, is_uniform=is_uniform
        )

    def add_ders_capacity(self, inpt: Union[Callable, float], is_uniform: bool):
        self.ders_capacity = self.set_scalar_property(inpt, is_uniform=is_uniform)

    def add_ders_conductivity(
        self,
        inpt: Union[Callable, float, np.ndarray],
        is_uniform: bool,
        ndim: int,
    ):
        self.ders_conductivity = self.set_tensor_property(
            inpt, ndim=ndim, is_uniform=is_uniform
        )


class isotropic_hardening:
    def __init__(self, elastic_limit: float, iso_args: dict):
        self._elaslim: float = elastic_limit
        (
            self.iso_hardening_function,
            self.iso_dershardening_function,
        ) = self._select_model(iso_args)

    def _select_model(self, iso_args: dict) -> Tuple[Callable, Callable]:
        models = {
            "none": self._model_none,
            "linear": self._model_linear,
            "swift": self._model_swift,
            "voce": self._model_voce,
        }
        model_name = str(iso_args.get("name", "none")).lower()
        if model_name not in models:
            raise ValueError("Unknown hardening model")
        return models[model_name](iso_args)

    def _model_none(self, args: dict) -> Tuple[Callable, Callable]:
        factor = 1e8 * self._elaslim
        return lambda a: factor * np.ones_like(a), lambda a: np.zeros_like(a)

    def _model_linear(self, args: dict) -> Tuple[Callable, Callable]:
        Eiso = args.get("Eiso")
        return lambda a: self._elaslim + Eiso * a, lambda a: Eiso * np.ones_like(a)

    def _model_swift(self, args: dict) -> Tuple[Callable, Callable]:
        e0, n = args.get("e0"), args.get("n")
        return lambda a: self._elaslim * (
            1 + a / e0
        ) ** n, lambda a: self._elaslim * n / e0 * (1 + a / e0) ** (n - 1)

    def _model_voce(self, args: dict) -> Tuple[Callable, Callable]:
        ssat, beta = args.get("ssat"), args.get("beta")
        return lambda a: self._elaslim + ssat * (
            1 - np.exp(-beta * a)
        ), lambda a: ssat * beta * np.exp(-beta * a)


class kinematic_hardening:
    def __init__(self, kine_args: dict):
        chaboche_table: np.ndarray = np.atleast_2d(
            kine_args.get("parameters", [[0, 0]])
        )
        self._chaboche_table = chaboche_table
        self._nb_chpar = chaboche_table.shape[0]

    def sum_chaboche_terms(
        self, dgamma: np.ndarray, back: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sum_back, hat_back = np.zeros_like(back[0, ...]), np.zeros_like(back[0, ...])
        const_1, const_2 = np.zeros_like(dgamma), np.zeros_like(dgamma)

        for i in range(self._nb_chpar):
            [c, d] = self._chaboche_table[i, :]
            term = 1 / (1 + d * dgamma)
            sum_back += back[i, ...] / term
            hat_back += d * back[i, ...] / term**2
            const_1 += c * term
            const_2 += c / term**2
        return sum_back, hat_back, const_1, const_2

    def update_back_stress(
        self,
        idx_scalar: np.ndarray,
        back_n1: np.ndarray,
        normal: np.ndarray,
        dgamma: np.ndarray,
        is_unidimensional: bool = True,
    ):
        for i in range(self._nb_chpar):
            [c, d] = self._chaboche_table[i, :]
            idx = (slice(i), slice(None), slice(None), *idx_scalar)
            factor = c if is_unidimensional else c * np.sqrt(1.5)
            back_n1[idx] = (back_n1[idx] + factor * normal * dgamma) / (
                1.0 + d * dgamma
            )


class plasticity(material, ABC):

    maxiters: int = 50
    threshold: float = 1e-8
    safeguard: float = 1e-12

    def __init__(self, mat_args: dict):
        super().__init__()
        self._initialize_properties(mat_args)
        self._initialize_hardening(mat_args)

    def _initialize_properties(self, mat_args: dict):
        self.elastic_modulus: float = mat_args.get("elastic_modulus", 0.0)
        self.poisson_ratio: float = mat_args.get("poisson_ratio", 0.0)
        self.elastic_limit: float = mat_args.get("elastic_limit", 1.0e15)
        self.lame_lambda = (
            self.poisson_ratio
            * self.elastic_modulus
            / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        )
        self.lame_mu = self.elastic_modulus / (2 * (1 + self.poisson_ratio))

    def _initialize_hardening(self, mat_args: dict):
        iso_args = mat_args.get("iso_hardening", {})
        self.isotropic_hardening = isotropic_hardening(self.elastic_limit, iso_args)

        kine_args = mat_args.get("kine_hardening", {})
        self.kinematic_hardening = kinematic_hardening(kine_args)
        self._activated_plasticity = (
            False if (len(iso_args) == 0 and len(kine_args) == 0) else True
        )

    def _solve_nonlinearity(
        self, plastic_equivalent: np.ndarray, compute_residual: Callable
    ) -> Tuple[np.ndarray, dict]:
        def solve_linearization(yield_fun: np.ndarray, **kwargs: dict):
            ders_yield_fun = kwargs.get("ders_yield_fun")
            assert ders_yield_fun is not None
            return yield_fun / ders_yield_fun

        dplseq = np.zeros_like(plastic_equivalent)
        nonlinsolv = nonlinsolver(tolerance=self.threshold, maxiters=self.maxiters)
        output = nonlinsolv.solve(
            dplseq, None, compute_residual, solve_linearization, verbose=False
        )
        return dplseq, output["extra_args"]

    @abstractmethod
    def eval_elastic_stress(self):
        pass

    @abstractmethod
    def set_linear_elastic_tensor(self):
        pass

    @abstractmethod
    def return_mapping(self):
        pass


class J2plasticity(plasticity):
    def __init__(self, mat_args: dict, is_unidimensional: bool = False):
        super().__init__(mat_args=mat_args)
        self._is_unidim = is_unidimensional

    def set_linear_elastic_tensor(self, shape: Union[int, list], ndim: int):
        if np.isscalar(shape):
            shape = np.array([shape], dtype=int)
        tensor = self.elastic_modulus * np.ones((ndim, ndim, ndim, ndim, *shape))
        if not self._is_unidim:
            identity = np.identity(ndim)
            tensor = np.zeros((ndim, ndim, ndim, ndim, *shape))
            indices = np.indices((ndim, ndim, ndim, ndim), dtype=int)
            for i, j, l, m in zip(*[arr.flatten() for arr in indices]):
                tensor[i, j, l, m, ...] = self.lame_lambda * identity[i, l] * identity[
                    j, m
                ] + self.lame_mu * (
                    identity[i, m] * identity[j, l] + identity[i, j] * identity[l, m]
                )
        return tensor

    def eval_elastic_stress(self, strain: np.ndarray) -> np.ndarray:
        if self._is_unidim:
            stress = self.elastic_modulus * strain
        else:
            trace_strain = compute_trace(strain)
            stress = 2 * self.lame_mu * strain
            for i in range(strain.shape[0]):
                stress[i, i, ...] += self.lame_lambda * trace_strain
        return stress

    def eval_von_mises_stress(self, stress: np.ndarray) -> np.ndarray:
        if self._is_unidim:
            return compute_norm_tensor(stress)
        else:
            stress_dev = compute_deviatoric(stress)
            return np.sqrt(1.5) * compute_norm_tensor(stress_dev)

    def _compute_trials(self, strain_n1, plasticstrain_n0, back_n0, plseq_n0):
        # Compute trial stress
        strain_trial = strain_n1 - plasticstrain_n0
        stress_trial = self.eval_elastic_stress(strain_trial)

        # Compute shifted stress
        vonmises_trial = self.eval_von_mises_stress(
            stress_trial - np.sum(back_n0, axis=0)
        )
        if self._is_unidim:
            vonmises_trial = np.ravel(vonmises_trial)

        # Check yield status
        J2_trial = vonmises_trial - self.isotropic_hardening.iso_hardening_function(
            plseq_n0
        )
        return stress_trial, J2_trial

    def _prepare_parameters(
        self,
        stress_trial: np.ndarray,
        back_n0: np.ndarray,
        plseq_n0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def compute_residual(dg: np.ndarray, f: None):
            output = self.kinematic_hardening.sum_chaboche_terms(dg, back_n0)
            shft_stress = stress_trial - output[0]
            if not self._is_unidim:
                shft_stress = compute_deviatoric(shft_stress)
            shft_stress_norm = compute_norm_tensor(shft_stress)
            normal_shft_stress = (
                np.sign(shft_stress)
                if self._is_unidim
                else shft_stress / shft_stress_norm
            )
            plseq_n1 = plseq_n0 + dg
            if self._is_unidim:
                yield_fun = -shft_stress_norm + (self.elastic_modulus + output[2]) * dg
            else:
                yield_fun = (
                    -np.sqrt(1.5) * shft_stress_norm
                    + 1.5 * (2 * self.lame_mu + output[2]) * dg
                )
            yield_fun += self.isotropic_hardening.iso_hardening_function(plseq_n1)
            if self._is_unidim:
                ders_yield_fun = compute_double_contraction(
                    normal_shft_stress, output[1]
                ) - (self.elastic_modulus + output[3])
            else:
                ders_yield_fun = np.sqrt(1.5) * compute_double_contraction(
                    normal_shft_stress, output[1]
                ) - 1.5 * (2 * self.lame_mu + output[3])
            ders_yield_fun -= self.isotropic_hardening.iso_dershardening_function(
                plseq_n1
            )
            extra_args = dict(
                hat_back=output[1],
                shifted_stress_norm=shft_stress_norm,
                normal_shifted_stress=normal_shft_stress,
                ders_yield_fun=ders_yield_fun,
            )
            return yield_fun, extra_args

        dgamma, extra_args = super()._solve_nonlinearity(plseq_n0, compute_residual)
        ders_yield_fun = extra_args.get("ders_yield_fun")
        shifted_stress_norm = extra_args.get("shifted_stress_norm")
        normal_shifted_stress = extra_args.get("normal_shifted_stress")
        hat_back = extra_args.get("hat_back")

        if self._is_unidim:
            if ders_yield_fun is not None:
                theta = -self.elastic_modulus / ders_yield_fun
        else:
            if ders_yield_fun is not None:
                theta_1 = -3 * self.lame_mu / ders_yield_fun
            if shifted_stress_norm is not None:
                theta_2 = 2 * self.lame_mu * dgamma * np.sqrt(1.5) / shifted_stress_norm
            theta = (theta_1, theta_2)

        return dgamma, hat_back, normal_shifted_stress, theta

    def return_mapping(
        self, strain_n1: np.ndarray, plastic_vars: dict, update_tangent: bool = True
    ) -> Tuple[np.ndarray, dict, dict]:
        """Return mapping algorithm for multidimensional rate-independent plasticity."""

        assert np.ndim(strain_n1) > 2, "At least 3d array"
        ndim = np.shape(strain_n1)[0]
        assert ndim in [1, 3], "Only for 1d or 3d methods"

        # Recover last values of internal variables
        strain_shape = tuple(np.shape(strain_n1)[2:])
        nb_chpar = self.kinematic_hardening._nb_chpar
        plasticstrain_n0 = plastic_vars.get("plastic_strain", np.zeros_like(strain_n1))
        plseq_n0 = plastic_vars.get("plastic_equivalent", np.zeros(shape=strain_shape))
        back_n0 = plastic_vars.get(
            "back_stress",
            np.zeros(shape=(nb_chpar, ndim, ndim, *strain_shape)),
        )

        # Compute trial stress and trial J2 yield function
        stress_trial, J2_trial = self._compute_trials(
            strain_n1, plasticstrain_n0, back_n0, plseq_n0
        )

        # Set default values
        stress_n1 = np.copy(stress_trial)
        plasticstrain_n1 = np.copy(plasticstrain_n0)
        plseq_n1 = np.copy(plseq_n0)
        back_n1 = np.copy(back_n0)
        consistent_tangent = self.set_linear_elastic_tensor(strain_shape, ndim=ndim)

        if not self._is_unidim:
            lame_lambda = self.lame_lambda * np.ones_like(J2_trial)
            lame_mu = self.lame_mu * np.ones_like(J2_trial)

        if np.any(J2_trial > super().threshold * self.elastic_limit):

            # Select the quadrature points
            idx_scalar = np.nonzero(J2_trial > super().threshold * self.elastic_limit)
            idx_ten2d = (slice(None), slice(None), *idx_scalar)
            idx_ten3d = (slice(None), slice(None), slice(None), *idx_scalar)

            # Compute plastic-strain increment
            dgamma, hatback, normal, theta = self._prepare_parameters(
                stress_trial[idx_ten2d], back_n0[idx_ten3d], plseq_n0[idx_scalar]
            )

            # Update internal hardening variable
            plseq_n1[idx_scalar] += dgamma

            # Update stress
            factor = (
                self.elastic_modulus
                if self._is_unidim
                else 2.0 * self.lame_mu * np.sqrt(1.5)
            )
            stress_n1[idx_ten2d] -= factor * dgamma * normal

            # Update plastic strain
            factor = 1.0 if self._is_unidim else np.sqrt(1.5)
            plasticstrain_n1[idx_ten2d] += factor * dgamma * normal

            # Update backstress
            self.kinematic_hardening.update_back_stress(
                idx_scalar, back_n1, normal, dgamma, is_unidimensional=self._is_unidim
            )

            # Update stiffness tensor
            if update_tangent:
                if self._is_unidim:
                    consistent_tangent[
                        0, 0, 0, 0, idx_scalar
                    ] = self.elastic_modulus * (1 - theta)
                else:
                    identity = np.identity(ndim)
                    omega_1 = -2 * self.lame_mu * (theta[0] - theta[1])
                    omega_2 = -np.sqrt(2.0 / 3.0) * theta[0] * theta[1]
                    lame_lambda[idx_scalar] += 2.0 / 3.0 * self.lame_mu * theta[1]
                    lame_mu[idx_scalar] -= self.lame_mu * theta[1]
                    indices = np.indices((ndim, ndim, ndim, ndim), dtype=int)
                    for i, j, l, m in zip(*[arr.flatten() for arr in indices]):
                        consistent_tangent[i, j, l, m, idx_scalar] = (
                            lame_lambda[idx_scalar] * identity[i, l] * identity[j, m]
                            + lame_mu[idx_scalar]
                            * (
                                identity[i, m] * identity[j, l]
                                + identity[i, j] * identity[l, m]
                            )
                            + omega_1 * (normal[i, l, ...] * normal[j, m, ...])
                            + omega_2
                            * (
                                hatback[i, l, ...] * normal[j, m, ...]
                                - normal[i, l, ...] * hatback[j, m, ...]
                            )
                        )

        new_plastic_vars = {
            "plastic_strain": plasticstrain_n1,
            "plastic_equivalent": plseq_n1,
            "back_stress": back_n1,
        }
        return stress_n1, {
            "consistent_tangent": consistent_tangent,
            "new_plastic_vars": new_plastic_vars,
        }
