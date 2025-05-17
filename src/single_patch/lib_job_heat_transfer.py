from .lib_job import *

class heat_transfer_problem(space_problem):
	def __init__(
		self,
		material: heat_transfer_mat,
		patch: singlepatch,
		boundary: boundary_condition,
		solver_args={},
	):
		super().__init__(
			patch, boundary, solver_args, allow_lumping=False
		)  # We consider no lumping in heat transfer
		self.material = material
		self.preconditioner = self.activate_preconditioner()
		self._capacity_property = None
		self._conductivity_property = None
		self._scalar_mean_capacity = [1]
		self._scalar_mean_conductivity = [np.ones(self.nbvars)]
		return

	def activate_preconditioner(self):
		fastdiag = fastdiagonalization()
		fastdiag.compute_space_eigendecomposition(
			self.part.quadrule_list, self.sp_table_dirichlet
		)
		fastdiag.add_free_controlpoints(self.sp_free_ctrlpts)
		return fastdiag

	def clear_properties(self):
		self._capacity_property = None
		self._conductivity_property = None
		return

	def compute_mf_capacity(self, array_in, args={}):
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

	def compute_mf_conductivity(self, array_in, args={}):
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

	def interpolate_temperature(self, u_ctrlpts):
		u_interp = bspline_operations.interpolate_meshgrid(
			self.part.quadrule_list, np.atleast_2d(u_ctrlpts)
		)
		return np.ravel(u_interp, order='F')

	def _assemble_internal_force(self, temperature, flux, scalar_coefs, args={}):
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
		self, temperature, flux, external_force, scalar_coefs, update_properties=False
	):

		if update_properties:
			self.clear_properties()
		args = {"temperature": self.interpolate_temperature(temperature)}
		residual = external_force - self._assemble_internal_force(
			temperature, flux, scalar_coefs, args
		)

		return residual, args

	def _linearized_heat_trasfer_solver(self, array_in, scalar_coefs, args={}):
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
		time_list=None,
		alpha=0.5,
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
					self._solution_history_list[f'step_{i}_noniter_{j}'] = np.copy(dj_n1)

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

		return


class st_heat_transfer_problem(spacetime_problem):

	def __init__(
		self,
		material: heat_transfer_mat,
		patch: singlepatch,
		time_patch: singlepatch,
		boundary: boundary_condition,
		solver_args={},
	):
		super().__init__(patch, time_patch, boundary, solver_args)
		self.material = material
		self.preconditioner = self.activate_preconditioner()
		self._capacity_property = None
		self._conductivity_property = None
		self._ders_capacity_property = None
		self._ders_conductivity_property = None
		self._scalar_mean_capacity = [1]
		self._scalar_mean_conductivity = [np.ones(self.nbvars)]
		return

	def activate_preconditioner(self):
		fastdiag = fastdiagonalization()
		fastdiag.compute_space_eigendecomposition(
			self.part.quadrule_list, self.sp_table_dirichlet
		)
		fastdiag.compute_time_schurdecomposition(self.time.quadrule_list[0])
		fastdiag.add_free_controlpoints(self.sptm_free_ctrlpts)
		return fastdiag

	def clear_properties(self):
		self._capacity_property = None
		self._conductivity_property = None
		self._ders_capacity_property = None
		self._ders_conductivity_property = None
		return

	def compute_mf_sptm_capacity(self, array_in, args=None):
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

	def compute_mf_sptm_ders_capacity(self, array_in, args=None):
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

	def compute_mf_sptm_conductivity(self, array_in, args=None):
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

	def compute_mf_sptm_ders_conductivity(self, array_in, args=None):
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
				"ilk,lkp->ikp",
				self.part.inv_jac,
				tmp1_reshaped,
				optimize=True
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

	def interpolate_sptm_temperature(self, u_ctrlpts):
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
		self, external_force, args={}, use_picard=True, inner_tolerance=None
	):

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
		self, temperature, external_force, update_properties=False
	):

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
		temperature,
		external_force,
		use_picard=True,
		auto_inner_tolerance=True,
		auto_outer_tolerance=False,
		nonlinear_args={},
	):

		def select_outer_tolerance(problem: spacetime_problem, factor=0.5):
			meshparameter_part = problem.part._compute_global_mesh_parameter()
			meshparameter_time = problem.time._compute_global_mesh_parameter()
			meshsize = max(meshparameter_part, meshparameter_time)
			degree = min(min(problem.part.degree), problem.time.degree)
			outer_tolerance = factor * (0.5**degree) * meshsize
			return outer_tolerance

		def select_inner_tolerance(
			use_picard, res_new, res_old, inner_tolerance, condition, solver_args: dict
		):
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

		for iteration in range(self._maxiters_nonlinear):

			residual, args = self._compute_heat_transfer_residual(
				temperature, external_force, update_properties=update_properties
			)
			clean_dirichlet(residual, constraint_ctrlpts)

			norm_residual = np.linalg.norm(residual)
			print(f"Nonlinear error: {norm_residual:.3e}")
			self._nonlinear_residual_list.append(norm_residual)
			self._solution_history_list[f'noniter_{iteration}'] = np.copy(temperature)

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

		return
