from .__init__ import *

def clean_dirichlet(array, idx_nodes):
	""" 
	Set specified indices of an array to 0 to apply Dirichlet boundary conditions.

	Parameters:
	array (np.ndarray): The input array which can be 1D or 2D.
	idx_nodes (list or np.ndarray): Indices in each dimension where the Dirichlet condition should be applied.

	Returns:
	None

	Raises:
	AssertionError: If the input array has more than 2 dimensions or if the number of dimensions of idx_nodes 
					is greater than the number of dimensions of the array.
	"""
	ndim_array = np.ndim(array)
	ndim_idx = len(idx_nodes)
	assert ndim_array < 3, 'Only for 1d or 2d arrays'
	if ndim_idx == 1 and ndim_array == 1:
			array[idx_nodes[0]] = 0.0
	elif ndim_idx == 1 and ndim_array == 2:
			array[0, idx_nodes[0]] = 0.0
	elif ndim_idx > 1:
		assert ndim_array >= ndim_idx, 'Dimension problem'
		for i in range(ndim_idx): 
			array[i, idx_nodes[i]] = 0.0
	return

class solver():
	
	def __init__(self, max_iters=100, tolerance=1e-12):
		self._max_iters = max_iters
		self._tolerance = tolerance
		self._Pfun = lambda x: x
		self._safeguard = 1.e-14
		return
	
	def eigs(self, N, Afun, Bfun=None, Pfun=None, k=6, which='LM'):
		"""
		Compute the eigenvalues and eigenvectors of a linear operator.

		Parameters:
		-----------
		N : int
			The size of the square matrix.
		Afun : callable
			A function that performs the matrix-vector multiplication for the matrix A.
		Bfun : callable, optional
			A function that performs the matrix-vector multiplication for the matrix B (default is None).
		Pfun : callable, optional
			A function that performs the matrix-vector multiplication for the preconditioner matrix P (default is None).

		Returns:
		--------
		eigvals_sorted : numpy.ndarray
			The sorted eigenvalues of the matrix.
		eigvecs_sorted : numpy.ndarray
			The eigenvectors corresponding to the sorted eigenvalues.

		Notes:
		------
		This function uses the `scipy.sparse.linalg.eigs` method to compute the eigenvalues and eigenvectors.
		The eigenvalues and eigenvectors are sorted in ascending order based on the eigenvalues.
		
		"""
		if k > N-2: k = N - 2
		ALinOp = scsplin.LinearOperator((N, N), matvec=Afun)
		BLinOp = None if Bfun is None else scsplin.LinearOperator((N, N), matvec=Bfun)
		PLinOp = None if Pfun is None else scsplin.LinearOperator((N, N), matvec=Pfun)
		eigvals, eigvecs = scsplin.eigs(A=ALinOp, M=BLinOp, Minv=PLinOp, k=k, which=which)

		eigvals = np.real(eigvals); eigvecs = np.real(eigvecs)
		sorted_indices = np.argsort(eigvals) 
		eigvals_sorted = eigvals[sorted_indices]
		eigvecs_sorted = eigvecs[:, sorted_indices]
		return eigvals_sorted, eigvecs_sorted
	
	def CG(self, Afun, b, Pfun=None, dotfun=None, cleanfun=None, dod=None, args={}):
		"""
		Conjugate Gradient (CG) solver for solving the linear system Ax = b.

		Parameters:
		-----------
		Afun : callable
			Function that computes the matrix-vector product A*p.
		b : array_like
			Right-hand side vector.
		Pfun : callable, optional
			Preconditioner function. Default is the identity function.
		dotfun : callable, optional
			Function to compute the dot product of two vectors. Default is numpy.dot.
		cleanfun : callable, optional
			Function to clean up vectors (e.g., apply boundary conditions). Default is a no-op.
		dod : any, optional
			Additional data for the cleanfun function. Default is None.
		args : dict, optional
			Additional arguments to pass to Afun. Default is an empty dictionary.

		Returns:
		--------
		dict
			A dictionary with the following keys:
			- 'sol': The solution vector x.
			- 'res': List of relative residuals at each iteration.

		"""
		assert callable(Afun), 'Define a linear operator'
		Pfun = self._Pfun if Pfun is None else Pfun
		assert callable(Pfun), 'Define a linear operator'
		dotfun = np.dot if dotfun is None else dotfun
		assert callable(dotfun), 'Define a function'
		cleanfun = (lambda x, y: x) if cleanfun is None else cleanfun
		assert callable(cleanfun), 'Define a function'

		x = np.zeros_like(b)
		r = b.copy(); cleanfun(r, dod)
		norm_0 = np.linalg.norm(r)
		if norm_0 <= self._safeguard:
			return {'sol': x, 'res': [0]}

		ptilde = Pfun(r); cleanfun(ptilde, dod)
		p = ptilde.copy()
		rsold = dotfun(r, ptilde)
		res = [1.0]

		for _ in range(self._max_iters):
			Ap = Afun(p, args); cleanfun(Ap, dod)
			alpha = rsold/dotfun(p, Ap)
			r -= alpha*Ap
			x += alpha*p

			norm_1 = np.linalg.norm(r)
			res.append(norm_1/norm_0)
			if (norm_1 <= max([self._tolerance*norm_0, self._safeguard])): break

			ptilde = Pfun(r); cleanfun(ptilde, dod)
			rsnew = dotfun(r, ptilde)
			p = ptilde + rsnew/rsold*p
			rsold = rsnew.copy()

		return {'sol': x, 'res': res}
	
	def BiCGSTAB(self, Afun, b, Pfun=None, dotfun=None, cleanfun=None, dod=None, args={}):
		"""
		BiConjugate Gradient Stabilized (BiCGSTAB) method for solving linear systems.

		Parameters:
		-----------
		Afun : callable
			Function that computes the matrix-vector product A*x.
		b : array_like
			Right-hand side vector.
		Pfun : callable, optional
			Preconditioner function. Default is the identity function.
		dotfun : callable, optional
			Function to compute the dot product of two vectors. Default is numpy.dot.
		cleanfun : callable, optional
			Function to clean or preprocess vectors. Default is the identity function.
		dod : any, optional
			Additional data for the cleanfun function. Default is None.
		args : dict, optional
			Additional arguments to pass to Afun. Default is an empty dictionary.

		Returns:
		--------
		dict
			Dictionary containing:
			- 'sol' : array_like
				Solution vector.
			- 'res' : list
				List of relative residuals at each iteration.

		Notes:
		------
		This method iteratively solves the linear system A*x = b using the BiCGSTAB algorithm.
		The algorithm is suitable for large, sparse, and non-symmetric linear systems.

		"""
		assert callable(Afun), 'Define a linear operator'
		Pfun = self._Pfun if Pfun is None else Pfun
		assert callable(Pfun), 'Define a linear operator'
		dotfun = np.dot if dotfun is None else dotfun
		assert callable(dotfun), 'Define a function'
		cleanfun = (lambda x, y: x) if cleanfun is None else cleanfun
		assert callable(cleanfun), 'Define a function'

		x = np.zeros_like(b)
		r = b.copy(); cleanfun(r, dod)
		norm_0 = np.linalg.norm(r)
		if norm_0 <= self._safeguard:
			return {'sol': x, 'res': [0]}

		rhat = r.copy()
		p = r.copy()
		rsold = dotfun(r, rhat)
		res = [1.0]

		for _ in range(self._max_iters):
			ptilde = Pfun(p); cleanfun(ptilde, dod)
			Aptilde = Afun(ptilde, args); cleanfun(Aptilde, dod)
			alpha = rsold/dotfun(Aptilde, rhat)
			s = r - alpha*Aptilde
			x += alpha*ptilde

			norm_1 = np.linalg.norm(s)
			res.append(norm_1/norm_0)
			if norm_1 <= max([self._tolerance*norm_0, self._safeguard]): break

			stilde = Pfun(s); cleanfun(stilde, dod)
			Astilde = Afun(stilde, args); cleanfun(Astilde, dod)
			omega = dotfun(Astilde, s)/dotfun(Astilde, Astilde)
			r = s - omega*Astilde
			x += omega*stilde

			norm_1 = np.linalg.norm(r)
			res.append(norm_1/norm_0)
			if norm_1 <= max([self._tolerance*norm_0, self._safeguard]): break

			rsnew = dotfun(r, rhat)
			beta = (alpha/omega)*(rsnew/rsold)
			p = r + beta*(p - omega*Aptilde)
			rsold = rsnew.copy()

		return {'sol': x, 'res': res}
	
	def GMRES(self, Afun, b, Pfun=None, dotfun=None, cleanfun=None, dod=None, args={}):
		"""
		Generalized Minimal Residual Method (GMRES) for solving a linear system of equations.
		Parameters:
		-----------
		Afun : callable
			Function that represents the matrix-vector product A*x.
		b : array_like
			Right-hand side vector of the linear system.
		Pfun : callable, optional
			Preconditioner function. Default is the identity function.
		dotfun : callable, optional
			Function to compute the dot product of two vectors. Default is numpy.dot.
		cleanfun : callable, optional
			Function to clean or preprocess vectors. Default is the identity function.
		dod : any, optional
			Additional data or parameters for the cleanfun function.
		args : dict, optional
			Additional arguments to pass to Afun.
		Returns:
		--------
		dict
			A dictionary containing:
			- 'sol': The solution vector.
			- 'res': List of residuals at each iteration.
		Notes:
		------
		This implementation uses the Arnoldi process to build an orthonormal basis for the Krylov subspace
		and solves the least squares problem to minimize the residual.

		"""
		assert callable(Afun), 'Define a linear operator'
		Pfun = self._Pfun if Pfun is None else Pfun
		assert callable(Pfun), 'Define a linear operator'
		dotfun = np.dot if dotfun is None else dotfun
		assert callable(dotfun), 'Define a function'
		cleanfun = (lambda x, y: x) if cleanfun is None else cleanfun
		assert callable(cleanfun), 'Define a function'

		x = np.zeros_like(b)
		hessenberg = np.zeros((self._max_iters + 1, self._max_iters))
		old_vectors = np.zeros((self._max_iters + 1, *b.shape))
		new_vectors = np.zeros_like(old_vectors)

		r = b.copy(); cleanfun(r, dod)
		norm_0 = np.linalg.norm(r)
		if norm_0 <= self._safeguard:
			return {'sol': x, 'res': [0]}
		res = [1.0]

		old_vectors[0] = r / norm_0
		e1 = np.zeros(self._max_iters + 1); e1[0] = norm_0
		
		for k in range(self._max_iters):
			new_vectors[k] = Pfun(old_vectors[k]); cleanfun(new_vectors[k], dod)
			w = Afun(new_vectors[k], args); cleanfun(w, dod)

			for j in range(k + 1):
				hessenberg[j, k] = dotfun(w, old_vectors[j])
				w -= hessenberg[j, k] * old_vectors[j]

			hessenberg[k + 1, k] = np.linalg.norm(w)
			if hessenberg[k + 1, k] != 0:
				old_vectors[k + 1] = w / hessenberg[k + 1, k]
			y = np.linalg.lstsq(hessenberg[:k + 2, :k + 1], e1[:k + 2], rcond=None)[0]
			norm_1 = np.linalg.norm(hessenberg[:k + 2, :k + 1] @ y - e1[:k + 2])
			res.append(norm_1/norm_0)
			if norm_1 <= max([self._tolerance*norm_0, self._safeguard]): break
		x += sum(new_vectors[j] * y[j] for j in range(k + 1))

		return {'sol': x, 'res': res}