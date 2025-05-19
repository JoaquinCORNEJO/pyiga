from .__init__ import *

def find_interpolation_span(array:np.ndarray, x:float, threshold=1e-8):
	"""
	Find the span in the array where the value x fits within the given threshold.
	"""
	span = 1
	while (span < (len(array) - 1) and (array[span] - x <= threshold)):
		span += 1
	return span - 1

def find_multiplicity(knotvector:np.ndarray, knot:float, threshold=1e-8):
	"""
	Find the multiplicity of a knot in the knot vector within the given threshold.
	"""
	return np.sum(np.abs(knotvector - knot) <= threshold)

def increase_multiplicity_to_knotvector(repeat:int, degree:int, knotvector:np.ndarray):
	"""
	Returns an open knot vector with higher multiplicity.
	"""
	kv_out = list([knotvector[0]] * (degree + 1))
	kv_unique = np.unique(knotvector)[1:-1]

	for knot in kv_unique:
		m = find_multiplicity(knotvector, knot) + repeat
		kv_out.extend([knot] * m)

	kv_out.extend([knotvector[-1]] * (degree + 1))
	return np.array(kv_out)

def eval_ders_basis_COO_format(degree:int, knotvector:np.ndarray, knots:np.ndarray):
	""" Evaluates B-spline functions and its first derivative at given knots.
		It returns matrices in CSR format.
	"""
	assert degree >=0, 'Degree must be a positive integer'

	nbctrlpts = len(knotvector) - degree - 1
	basis, indices_i, indices_j = [], [], []

	for j, knot in enumerate(knots):
		knot_span = helpers.find_span_linear(degree, knotvector, nbctrlpts, knot)
		basis_ders = helpers.basis_function_ders(degree, knotvector, knot_span, knot, 1)

		for i, (b0, b1) in enumerate(zip(*basis_ders)):
			basis.append([b0, b1])
			indices_i.append((knot_span - degree + i)%nbctrlpts)
			indices_j.append(j)

	return np.array(basis), np.array(indices_i), np.array(indices_j)

def eval_ders_basis_sparse(degree:int, knotvector:np.ndarray, knots:np.ndarray):
	""" Evaluates B-spline functions and its first derivative at given knots.
		It returns matrices as scipy CSR objects.
	"""
	nbctrlpts = len(knotvector) - degree - 1
	basis_coo, indi_coo, indj_coo = eval_ders_basis_COO_format(degree, knotvector, knots)
		
	# Loop through the basis functions and derivatives
	basis_list = [
		sp.coo_matrix(
			(basis_coo[:, i], (indi_coo, indj_coo)),
			shape=(nbctrlpts, len(knots))
		).tocsr() 
		for i in range(basis_coo.shape[1])
	]

	return basis_list

class quadrature_rule:

	def __init__(self, degree, knotvector):
		self.degree:int = degree
		self.knotvector:np.ndarray = knotvector
		self.nbctrlpts:int = len(self.knotvector) - self.degree - 1
		self._unique_kv:np.ndarray = np.unique(self.knotvector)
		self._nbel:int = len(self._unique_kv) - 1

		self._coo_indices:np.ndarray = np.array([])
		self._coo_basis:np.ndarray = np.array([])
		self._coo_weigts:np.ndarray = np.array([])
		self.quadpts:np.ndarray = np.array([])
		self.basis:list = []
		self.weights:list = []
		self.nbqp:int = 0

	def _set_quadrature_points(self, quadpts:np.ndarray):
		self.quadpts = quadpts
		self.nbqp = len(quadpts)
		return

	def _set_coo_basis_weights(self, basis:np.ndarray, weights:np.ndarray, indices:np.ndarray):
		self._coo_indices = indices
		self._coo_basis   = basis
		self._coo_weights = weights
		return

	def _assemble_csr_basis_weights(self):
		basis, weights = [], []
		indi, indj = self._coo_indices
		for i in range(np.size(self._coo_basis, axis=1)):
			basis.append(
				sp.coo_matrix(
					(self._coo_basis[:, i], (indi, indj)),
					shape=(self.nbctrlpts, self.nbqp)
				).tocsr()
			)

		for i in range(np.size(self._coo_weights, axis=1)):
			weights.append(
				sp.coo_matrix(
					(self._coo_weights[:, i], (indi, indj)),
					shape=(self.nbctrlpts, self.nbqp)
				).tocsr()
			)
		self.basis, self.weights = basis, weights
		return

	def get_sample_basis(self, knots_to_interp:np.ndarray):
		return eval_ders_basis_sparse(self.degree, self.knotvector, knots_to_interp)

class gauss_quadrature(quadrature_rule):

	def __init__(self, degree:int, knotvector:np.ndarray, quad_args:dict):
		super().__init__(degree, knotvector)
		self._quadrature_type = str(quad_args.get('type', 'leg')).lower()
		assert self._quadrature_type in ['leg', 'lob'], f"Unknown quadrature type: {self._quadrature_type}"

		if self._quadrature_type == 'leg': # Legendre
			self._order = quad_args.get('default_order', self.degree + 1)
			self._table_function = legendre_table
		elif self._quadrature_type == 'lob': # Lobatto
			self._order = quad_args.get('default_order', self.degree + 2)
			self._table_function = lobatto_table
		
		self._parametric_weights:np.ndarray = np.array([])
		return

	def _get_isoparametric_variables(self):
		" Gets the position of Gauss quadrature points in isoparametric space and its weights using known tables "
		self._isoparametric_positions, self._isoparametric_weights = self._table_function(self._order)
		return

	def _set_quadrature_points(self):
		" Gets the position of Gauss quadrature points in parametric space "
		knots = self._unique_kv
		quadpts = np.concatenate([
			0.5 * ((knots[i + 1] - knots[i]) * self._isoparametric_positions + knots[i] + knots[i + 1])
			for i in range(self._nbel)
		])
		super()._set_quadrature_points(quadpts)
		return

	def _compute_parametric_weights(self):
		" Gets the weight of Gauss quadrature points in parametric space "
		knots = self._unique_kv
		self._parametric_weights = np.concatenate([
			0.5 * (knots[i + 1] - knots[i]) * self._isoparametric_weights
			for i in range(self._nbel)
		])
		return

	def _set_coo_basis_weights(self):
		" Gets the basis and weights evaluated at the Gauss quadrature points. "
		basis, indi, indj = eval_ders_basis_COO_format(self.degree, self.knotvector, self.quadpts)
		nnz = np.shape(basis)[0]
		weights = np.zeros((nnz, 4))
		weights[:, 0] = basis[:, 0] * self._parametric_weights[indj]
		weights[:, 3] = basis[:, 1] * self._parametric_weights[indj]
		weights[:, 1] = weights[:, 0]
		weights[:, 2] = weights[:, 3]
		super()._set_coo_basis_weights(basis, weights, [indi, indj])
		return

	def export_quadrature_rules(self):
		self._get_isoparametric_variables()
		self._set_quadrature_points()
		self._compute_parametric_weights()
		self._set_coo_basis_weights()
		self._assemble_csr_basis_weights()
		return

class weighted_quadrature(quadrature_rule):

	def __init__(self, degree:int, knotvector:np.ndarray, quad_args:dict):
		super().__init__(degree, knotvector)
		self._quadrature_type = quad_args.get('type', 1)
		self._position_rule, default_extra_args = self._get_position_rule_and_defaults(self._quadrature_type)
		self._extra_args = {**default_extra_args, **quad_args.get('rule_parameters', {})}
		self._basis_shape:np.ndarray = np.array([])

	def _get_position_rule_and_defaults(self, quadrature_type):
		if quadrature_type == 1:
			return 'midpoint', {'s': 1, 'r': 3}
		elif quadrature_type == 2:
			return 'midpoint', {'s': 2, 'r': 3}
		elif quadrature_type == 3:
			return 'internal', {'s': 3, 'r': 2}
		else:
			raise ValueError(f"Unknown quadrature type: {quadrature_type}")

	def _set_quadrature_points(self):
		s = self._extra_args.get('s', 1)
		r = self._extra_args.get('r', 2)
		if self._position_rule == 'midpoint':
			quadpts = self._midpoint_rule(s=s, r=r)
		elif self._position_rule == 'internal':
			quadpts = self._internal_rule(s=s, r=r)
		else:
			raise ValueError(f"Unknown position rule: {self._position_rule}")

		super()._set_quadrature_points(quadpts)
		self._basis_shape = self._get_basis_shape(quadpts)
		return

	def _midpoint_rule(self, s=2, r=3):
		return self._generate_quadrature_points(s, r, include_boundaries=True)

	def _internal_rule(self, s=2, r=4):
		return self._generate_quadrature_points(s, r, include_boundaries=False)

	def _generate_quadrature_points(self, s:int, r:int, include_boundaries:bool):
		quadpts = np.array([])
		knots = self._unique_kv

		# First span
		tmp = np.linspace(knots[0], knots[1], self.degree + r)
		if not include_boundaries:
			tmp = tmp[1:-1]
		quadpts = np.append(quadpts, tmp)

		# Last span
		tmp = np.linspace(knots[-2], knots[-1], self.degree + r)
		if not include_boundaries:
			tmp = tmp[1:-1]
		quadpts = np.append(quadpts, tmp)

		# Inner spans
		for i in range(1, self._nbel - 1):
			tmp = np.linspace(knots[i], knots[i + 1], 2 + s)
			if not include_boundaries:
				tmp = tmp[1:-1]
			quadpts = np.append(quadpts, tmp)

		quadpts = np.unique(quadpts)
		quadpts.sort()
		return quadpts

	def _solve_linear_system(self, 
		integral_shape:np.ndarray, 
		basis_shape:np.ndarray, 
		integral:np.ndarray, 
		basis:np.ndarray
		):

		weights = np.zeros((basis_shape.shape[0], basis.shape[1]))
		for i in range(basis_shape.shape[0]):
			Pmin, Pmax = basis_shape[i, :]
			Fmin = np.argmax(integral_shape[:, i] > 0)
			Fmax = integral_shape.shape[0] - np.argmax(integral_shape[::-1, i] > 0)
			weights[i, Pmin:Pmax+1] = np.ravel(
				np.linalg.lstsq(
					basis[Fmin:Fmax, Pmin:Pmax+1],
					integral[Fmin:Fmax, i], rcond=None
					)[0]
			)
		return weights

	def _get_basis_shape(self, knots:np.ndarray):
		" Return the shape of basis in WQ approach. "
		B0shape = np.zeros((self.nbctrlpts, 2), dtype=int)
		B1shape = np.zeros((self.nbctrlpts, 2), dtype=int)

		tableKVSpan = np.zeros((self._nbel, 2))
		for i in range(self._nbel):
			tableKVSpan[i, :] = [self._unique_kv[i], self._unique_kv[i+1]]
		tablePointsOverKVSpan = np.zeros((self._nbel, 2), dtype=int)
		for i in range(self._nbel):
			left = self._unique_kv[i]
			right = self._unique_kv[i+1]
			boolean = (knots>=left)*(knots<=right)
			tablePointsOverKVSpan[i, 0] = np.nonzero(boolean)[0][0]
			tablePointsOverKVSpan[i, 1] = np.nonzero(boolean)[0][-1]

		tableFunctionsOverKVSpan = np.zeros((self._nbel, self.degree+1), dtype=int)
		for j in range(self.degree+1): tableFunctionsOverKVSpan[0, j] = j
		for i in range(1, self._nbel):
			multiplicity = find_multiplicity(self.knotvector, self._unique_kv[i])
			tableFunctionsOverKVSpan[i, 0] = tableFunctionsOverKVSpan[i-1, 0] + multiplicity
			for j in range(1, self.degree+1):
				tableFunctionsOverKVSpan[i, j] = tableFunctionsOverKVSpan[i, 0] + j

		tableFunctionSpans = np.zeros((self.nbctrlpts, 2), dtype=int)
		for i in range(self.nbctrlpts):
			minFuncSpan = 1
			for j in range(self._nbel):
				if np.any(tableFunctionsOverKVSpan[j, :]==i):
					minFuncSpan = j; break
			maxFuncSpan = self._nbel
			for j in range(self._nbel-1, -1, -1):
				if np.any(tableFunctionsOverKVSpan[j, :]==i):
					maxFuncSpan = j; break
			tableFunctionSpans[i, :] = [minFuncSpan, maxFuncSpan]

		## FOR B0
		# For the first function
		minFuncSpan, maxFuncSpan = tableFunctionSpans[0, :]
		minKnotOverFuncSpan = tableKVSpan[minFuncSpan, 0]
		maxKnotOverFuncSpan = tableKVSpan[maxFuncSpan, 1]
		boolean = (knots>=minKnotOverFuncSpan)*(knots<maxKnotOverFuncSpan)
		B0shape[0, 0] = np.nonzero(boolean)[0][0]
		B0shape[0, 1] = np.nonzero(boolean)[0][-1]

		# For the inner functions
		for i in range(1, self.nbctrlpts-1):
			minFuncSpan, maxFuncSpan = tableFunctionSpans[i, :]
			minKnotOverFuncSpan = tableKVSpan[minFuncSpan, 0]
			maxKnotOverFuncSpan = tableKVSpan[maxFuncSpan, 1]
			boolean = (knots>minKnotOverFuncSpan)*(knots<maxKnotOverFuncSpan)
			B0shape[i, 0] = np.nonzero(boolean)[0][0]
			B0shape[i, 1] = np.nonzero(boolean)[0][-1]

		# For the last function
		minFuncSpan, maxFuncSpan = tableFunctionSpans[self.nbctrlpts-1, :]
		minKnotOverFuncSpan = tableKVSpan[minFuncSpan, 0]
		maxKnotOverFuncSpan = tableKVSpan[maxFuncSpan, 1]
		boolean = (knots>minKnotOverFuncSpan)*(knots<=maxKnotOverFuncSpan)
		B0shape[self.nbctrlpts-1, 0] = np.nonzero(boolean)[0][0]
		B0shape[self.nbctrlpts-1, 1] = np.nonzero(boolean)[0][-1]

		## FOR B1
		B1shape = np.copy(B0shape) # It should be normally equal

		# Except for the second and next-to-last (where the derivative in 0 and 1 are different from 0)
		# For the second function
		minFuncSpan, maxFuncSpan = tableFunctionSpans[1, :]
		minKnotOverFuncSpan = tableKVSpan[minFuncSpan, 0]
		maxKnotOverFuncSpan = tableKVSpan[maxFuncSpan, 1]
		boolean = (knots>=minKnotOverFuncSpan)*(knots<maxKnotOverFuncSpan)
		B1shape[1, 0] = np.nonzero(boolean)[0][0] # Just the left

		# For the next-to-last function
		minFuncSpan, maxFuncSpan = tableFunctionSpans[self.nbctrlpts-2, :]
		minKnotOverFuncSpan = tableKVSpan[minFuncSpan, 0]
		maxKnotOverFuncSpan = tableKVSpan[maxFuncSpan, 1]
		boolean = (knots>minKnotOverFuncSpan)*(knots<=maxKnotOverFuncSpan)
		B1shape[self.nbctrlpts-2, 1] = np.nonzero(boolean)[0][-1] # Just the right
		return [B0shape, B1shape]

	def _compute_parametric_weights(self, method:str):
		" Computes the weights at quadrature points in WQ approach using specified trial space. "

		# Space S^p_r
		gauss_p0 = gauss_quadrature(self.degree, self.knotvector, {'type':'leg'})
		gauss_p0.export_quadrature_rules()
		B0cgg_p0, B1cgg_p0 = gauss_p0.basis
		basis_csr, indi_coo, indj_coo = eval_ders_basis_COO_format(self.degree, self.knotvector, self.quadpts)
		B0wq_p0 = sp.coo_matrix((basis_csr[:, 0], (indi_coo, indj_coo))).tocsr()

		if method == '1':
			# Space S^[p-1]_[r-1]
			degree_p1 = self.degree - 1
			knotvector_p1 = self.knotvector[1:-1]
		elif method == '2':
			# Space S^[p]_[r-1]
			degree_p1 = self.degree
			knotvector_p1 = increase_multiplicity_to_knotvector(1, degree_p1, self.knotvector)
		else:
			raise ValueError(f"Unknown trial space: {method}")

		gauss_p1 = quadrature_rule(degree_p1, knotvector_p1)
		B0cgg_p1 = eval_ders_basis_sparse(degree_p1, knotvector_p1, gauss_p0.quadpts)[0]
		B0wq_p1 = eval_ders_basis_sparse(degree_p1, knotvector_p1, self.quadpts)[0]

		# Compute Integrals
		weights_csr = np.zeros((np.size(basis_csr, axis=0), 4))
		Bcgg_p0_int = np.zeros((gauss_p0.nbctrlpts, gauss_p0.nbqp))
		Bcgg_p1_int = np.zeros((gauss_p1.nbctrlpts, gauss_p0.nbqp))
		Bcgg_p0_int[np.where(np.abs(B0cgg_p0.toarray())>1e-8)] = 1
		Bcgg_p1_int[np.where(np.abs(B0cgg_p1.toarray())>1e-8)] = 1

		list_weights = []
		if method == '1':
			# Computation of W00
			Ishape = Bcgg_p0_int @ Bcgg_p0_int.T
			integral = B0cgg_p0 @ sp.diags(gauss_p0._parametric_weights) @ B0cgg_p0.T
			list_weights.append(
				self._solve_linear_system(Ishape, self._basis_shape[0], integral.toarray(), B0wq_p0.toarray())
			)

		# Computation of W01 for method 1 or W00 for method 2
		Ishape = Bcgg_p1_int @ Bcgg_p0_int.T
		integral = B0cgg_p1 @ sp.diags(gauss_p0._parametric_weights) @ B0cgg_p0.T
		list_weights.append(
			self._solve_linear_system(Ishape, self._basis_shape[0], integral.toarray(), B0wq_p1.toarray())
		)

		if method == '1':
			# Computation of W10
			Ishape = Bcgg_p0_int @ Bcgg_p0_int.T
			integral = B0cgg_p0 @ sp.diags(gauss_p0._parametric_weights) @ B1cgg_p0.T
			list_weights.append(
				self._solve_linear_system(Ishape, self._basis_shape[1], integral.toarray(), B0wq_p0.toarray())
			)

		# Computation of W11
		Ishape = Bcgg_p1_int @ Bcgg_p0_int.T
		integral = B0cgg_p1 @ sp.diags(gauss_p0._parametric_weights) @ B1cgg_p0.T
		list_weights.append(
			self._solve_linear_system(Ishape, self._basis_shape[1], integral.toarray(), B0wq_p1.toarray())
		)

		for idx, (i, j) in enumerate(zip(indi_coo, indj_coo)):
			list_of_indices = [0, 1, 2, 3] if method == '1' else [0, 0, 1, 1]
			weights_csr[idx, :] = [
				list_weights[list_of_indices[k]][i, j] for k in range(len(list_of_indices))
			]

		return basis_csr, weights_csr, indi_coo, indj_coo

	def _set_coo_basis_weights(self):
		method = '2' if self._quadrature_type in [2, 3] or self.degree == 1 else '1'
		basis, weights, indi, indj = self._compute_parametric_weights(method=method)
		super()._set_coo_basis_weights(basis, weights, [indi, indj])
		return

	def export_quadrature_rules(self):
		self._set_quadrature_points()
		self._set_coo_basis_weights()
		super()._assemble_csr_basis_weights()
		return

def legendre_table(order:int):
	" Computes Gauss-Legendre weights and positions in isoparametric space for a given degree "
	if order == 1:
		pos = [0.0]
		wgt = [2.0]
	elif order == 2:
		pos = [ -0.577_350_269_189_625_76, 0.577_350_269_189_625_76]
		wgt = [ 1.0, 1.0]
	elif order == 3:
		pos = [ -0.774_596_669_241_483_37, 0.0, 0.774_596_669_241_483_37]
		wgt = [ 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
	elif order == 4:
		pos = [ -0.861_136_311_594_0526, -0.339_981_043_584_8563,
				0.339_981_043_584_8563, 0.861_136_311_594_0526]
		wgt = [ 0.347_854_845_137_4539, 0.652_145_154_862_5461,
				0.652_145_154_862_5461, 0.347_854_845_137_4539]
	elif order == 5:
		pos = [ -0.906_179_845_938_6640,-0.538_469_310_105_6831,
				0.0, 0.538_469_310_105_6831, 0.906_179_845_938_6640]
		wgt = [ 0.236_926_885_056_1891,	0.478_628_670_499_3665,
				0.568_888_888_888_8889,	0.478_628_670_499_3665, 0.236_926_885_056_1891]
	elif order == 6:
		pos = [ -0.932_469_514_203_1520,-0.661_209_386_466_2645,
				-0.238_619_186_083_1969, 0.238_619_186_083_1969,
				0.661_209_386_466_2645, 0.932_469_514_203_1520]
		wgt = [ 0.171_324_492_379_1703, 0.360_761_573_048_1386,
				0.467_913_934_572_6910, 0.467_913_934_572_6910,
				0.360_761_573_048_1386, 0.171_324_492_379_1703]
	elif order == 7:
		pos = [ -0.949_107_912_342_7585, -0.741_531_185_599_3944, -0.405_845_151_377_3972,
				0.0, 0.405_845_151_377_3972,  0.741_531_185_599_3944, 0.949_107_912_342_7585]
		wgt = [ 0.129_484_966_168_8697, 0.279_705_391_489_2767, 0.381_830_050_505_1189,
				0.417_959_183_673_4694, 0.381_830_050_505_1189, 0.279_705_391_489_2767, 0.129_484_966_168_8697]
	elif order == 8:
		pos = [ -0.960_289_856_497_5362, -0.796_666_477_413_6267, -0.525_532_409_916_3290, -0.183_434_642_495_6498,
				0.183_434_642_495_6498, 0.525_532_409_916_3290, 0.796_666_477_413_6267, 0.960_289_856_497_5362]
		wgt = [ 0.101_228_536_290_3763, 0.222_381_034_453_3745, 0.313_706_645_877_8873, 0.362_683_783_378_3620,
				0.362_683_783_378_3620, 0.313_706_645_877_8873, 0.222_381_034_453_3745, 0.101_228_536_290_3763]
	elif order == 9:
		pos = [ -0.968_160_239_507_6261, -0.836_031_107_326_6358, -0.613_371_432_700_5904, -0.324_253_423_403_8089,
				0.0, 0.324_253_423_403_8089, 0.613_371_432_700_5904, 0.836_031_107_326_6358, 0.968_160_239_507_6261]
		wgt = [ 0.081_274_388_361_5744, 0.180_648_160_694_8574, 0.260_610_696_402_9354, 0.312_347_077_040_0028,
				0.330_239_355_001_2597, 0.312_347_077_040_0028, 0.260_610_696_402_9354, 0.180_648_160_694_8574,
				0.081_274_388_361_5744]
	elif order == 10:
		pos = [ -0.973_906_528_517_1717, -0.865_063_366_688_9845, -0.679_409_568_299_0244, -0.433_395_394_129_2472,
				-0.148_874_338_981_6312, 0.148_874_338_981_6312, 0.433_395_394_129_2472,
				0.679_409_568_299_0244, 0.865_063_366_688_9845, 0.973_906_528_517_1717]
		wgt = [ 0.066_671_344_308_6881, 0.149_451_349_150_5806,	0.219_086_362_515_9820, 0.269_266_719_309_9963,
				0.295_524_224_714_7529, 0.295_524_224_714_7529, 0.269_266_719_309_9963,
				0.219_086_362_515_9820, 0.149_451_349_150_5806, 0.066_671_344_308_6881]
	else: raise Warning('Not degree found')
	return np.array(pos), np.array(wgt)

def lobatto_table(order:int):
	" Computes Gauss-Lobatto weights and positions in isoparametric space for a given degree "
	if order == 2:
		pos = [-1.0, 1.0]
		wgt = [1.0, 1.0]
	elif order == 3:
		pos = [-1.0, 0.0, 1.0]
		wgt = [1.0/3.0, 4.0/3.0, 1.0/3.0]
	elif order == 4:
		pos = [-1.0,
			-0.447_213_595_499_958,
			0.447_213_595_499_958,
			1.0]
		wgt = [1.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/6.0]
	elif order == 5:
		pos = [-1.0,
			-0.654_653_670_707_977,
			0.0,
			0.654_653_670_707_977,
			1.0]
		wgt = [0.1, 4.9/9.0, 6.4/9.0, 4.9/9.0, 0.1]
	elif order == 6:
		pos = [-1.0, -0.765_055_323_929_465, -0.285_231_516_480_645,
			0.285_231_516_480_645, 0.765_055_323_929_465, 1.0]
		wgt = [0.066_666_666_666_667, 0.378_474_956_297_847,
			0.554_858_377_035_486, 0.554_858_377_035_486,
			0.378_474_956_297_847, 0.066_666_666_666_667]
	elif order == 7:
		pos = [-1.0, -0.830_223_896_278_5670, -0.468_848_793_470_7142, 0.0,
			0.468_848_793_470_7142, 0.830_223_896_278_5670, 1.0]
		wgt = [0.047_619_047_619_0476, 0.276_826_047_361_5659, 0.431_745_381_209_8627,
				0.487_619_047_619_0476, 0.431_745_381_209_8627, 0.276_826_047_361_5659,
				0.047_619_047_619_0476]
	elif order == 8:
		pos = [-1.0, -0.871_740_148_509_6066, -0.591_700_181_433_1423,
				-0.209_299_217_902_4789, 0.209_299_217_902_4789, 0.591_700_181_433_1423,
				0.871_740_148_509_6066, 1.0]
		wgt = [0.035_714_285_714_2857, 0.210_704_227_143_5061, 0.341_122_692_483_5044,
				0.412_458_794_658_7038, 0.412_458_794_658_7038, 0.341_122_692_483_5044,
				0.210_704_227_143_5061, 0.035_714_285_714_2857]
	elif order == 9:
		pos = [-1.0, -0.899_757_995_411_4602, -0.677_186_279_510_7377,
				-0.363_117_463_826_1782, 0.0, 0.363_117_463_826_1782,
				0.677_186_279_510_7377, 0.899_757_995_411_4602, 1.0]
		wgt = [0.027_777_777_777_7778, 0.165_495_361_560_8055, 0.274_538_712_500_1617,
				0.346_428_510_973_0463, 0.371_519_274_376_4172, 0.346_428_510_973_0463,
				0.274_538_712_500_1617, 0.165_495_361_560_8055, 0.027_777_777_777_7778]
	elif order == 10:
		pos = [-1.0, -0.919_533_908_166_4589, -0.738_773_865_105_5050,
				-0.477_924_949_810_4445, -0.165_278_957_666_3870, 0.165_278_957_666_3870,
				0.477_924_949_810_4445, 0.738_773_865_105_5050, 0.919_533_908_166_4589, 1.0]
		wgt = [0.022_222_222_222_2222, 0.133_305_990_851_0701, 0.224_889_342_063_1264,
				0.292_042_683_679_6838, 0.327_539_761_183_8976, 0.327_539_761_183_8976,
				0.292_042_683_679_6838, 0.224_889_342_063_1264, 0.133_305_990_851_0701,
				0.022_222_222_222_2222]
	elif order == 11:
		pos = [-1.0, -0.934_001_430_408_0592, -0.784_483_473_663_1444,-0.565_235_326_996_2050,
				-0.295_758_135_586_9394, 0.0, 0.295_758_135_586_9394, 0.565_235_326_996_2050,
				0.784_483_473_663_1444, 0.934_001_430_408_0592, 1.0]
		wgt = [0.018_181_818_181_8182, 0.109_612_273_266_9949, 0.187_169_881_780_3052,
				0.248_048_104_264_0284, 0.286_879_124_779_0080, 0.300_217_595_455_6907,
				0.286_879_124_779_0080, 0.248_048_104_264_0284, 0.187_169_881_780_3052,
				0.109_612_273_266_9949, 0.018_181_818_181_8182]
	elif order == 12:
		pos = [-1.0, -0.944_899_272_222_8822, -0.819_279_321_644_0067,
				-0.632_876_153_031_8606, -0.399_530_940_965_3489, -0.136_552_932_854_9276,
				0.136_552_932_854_9276, 0.399_530_940_965_3489, 0.632_876_153_031_8606,
				0.819_279_321_644_0067, 0.944_899_272_222_8822, 1.0]
		wgt = [0.015_151_515_151_5152, 0.091_684_517_413_1962, 0.157_974_705_564_3701,
				0.212_508_417_761_0211, 0.251_275_603_199_2013, 0.271_405_240_910_6962,
				0.271_405_240_910_6962, 0.251_275_603_199_2013, 0.212_508_417_761_0211,
				0.157_974_705_564_3701, 0.091_684_517_413_1962, 0.015_151_515_151_5152]
	else: raise Warning('Not defined')
	return np.array(pos), np.array(wgt)
