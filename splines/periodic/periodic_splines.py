from src.lib_quadrules import *

def extend_knotvector(kv_in, degree):
	assert len(kv_in) >= 2, "Knot vector must contain at least 2 elements"
	assert degree >= 1, "Degree must be positive"
	kv_in = np.asarray(kv_in)
	n = len(kv_in)
	m = degree // (n - 1) + 1

	kv_right = kv_in[1:] + 1.0
	kv_left = kv_in[-2::-1] - 1.0
	for i in range(1, m):
		kv_right = np.concatenate((kv_right, kv_right[:n-1] + i))
		kv_left = np.concatenate((kv_left, kv_left[:n-1] - i))

	kv_out = np.concatenate((kv_left[:degree][::-1], kv_in, kv_right[:degree]))
	return kv_out

degree = 2
kv = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.])
knotvector = extend_knotvector(kv, degree)
nbctrlpts = len(knotvector) - degree - 1
# ctrlpts = np.array([sum(knotvector[i+j] for j in range(degree))/degree
# 				for i in range(1, nbctrlpts+1)])

knots = np.linspace(0, 1, 101)
basis_0, basis_1 = eval_ders_basis_sparse(degree, knotvector, knots)
basis_0 = basis_0.toarray()
basis_1 = basis_1.toarray()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
for i in range(np.shape(basis_0)[0]):
	ax.plot(knots, basis_0[i, :])
ax.plot([], [], linewidth=1, color='k', label='B-Spline basis')
ax.set_xlabel(r'$\xi$')
ax.set_xticks(np.linspace(0, 1, 5))
ax.set_yticks([0, 0.5, 1])
ax.set_ylabel(r'$\hat{b}_{A,\,p}(\xi)$')
ax.legend()
fig.tight_layout()
fig.savefig("periodic_splines", dpi=300)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
for i in range(np.shape(basis_1)[0]):
	ax.plot(knots, basis_1[i, :])
ax.plot([], [], linewidth=1, color='k', label='B-Spline basis')
ax.set_xlabel(r'$\xi$')
ax.set_xticks(np.linspace(0, 1, 5))
ax.set_yticks([0, 0.5, 1])
ax.set_ylabel(r'$\hat{b}_{A,\,p}(\xi)$')
ax.legend()
fig.tight_layout()
fig.savefig("periodic_splines_ders", dpi=300)

ctrlpts = np.array([
	[0, 0], [1, 2], [2, 4], [5, 0.5], [3, 0], [0, 0], [1, 2]
])
evalpts = basis_0.T @ ctrlpts

fig, ax = plt.subplots()
ax.plot(ctrlpts[:, 0], ctrlpts[:, 1])
ax.plot(evalpts[:, 0], evalpts[:, 1], '--')
fig.tight_layout()
fig.savefig("periodic_curve", dpi=300)
