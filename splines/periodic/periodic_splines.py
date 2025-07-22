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
        kv_right = np.concatenate((kv_right, kv_right[: n - 1] + i))
        kv_left = np.concatenate((kv_left, kv_left[: n - 1] - i))

    kv_out = np.concatenate((kv_left[:degree][::-1], kv_in, kv_right[:degree]))
    return kv_out


idx = 3
degree = 3
kv = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
knotvector = extend_knotvector(kv, degree)
quadrule = weighted_quadrature(
    degree, knotvector, {"type": 1, "rule_parameters": {"s": 1, "r": 3}}
)
quadrule.export_quadrature_rules()
knots = np.linspace(0, 1, 101)
basis = quadrule.get_sample_basis(knots)
basis_0, basis_1 = basis[0].toarray(), basis[1].toarray()
weights_0, weights_1 = quadrule.weights[0].toarray(), quadrule.weights[-1].toarray()

# Fig 1
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
for i in range(np.shape(basis_0)[0]):
    ax1.plot(knots, basis_0[i, :])
ax1.plot([], [], linewidth=1, color="k")
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$\hat{b}_{A,\,p}(\xi)$")
ax1.set_ylim([0, 1])

ax1.fill_between(x=knots, y1=basis_0[idx, :], color="g", alpha=0.2)
ax2 = ax1.twinx()
ax2.plot(quadrule.quadpts, weights_0[idx, :], "ko")
ax2.grid(None)
ax2.set_ylim([0, 0.1])

fig.tight_layout()
fig.savefig("periodic_splines", dpi=300)

# Fig 2
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
for i in range(np.shape(basis_1)[0]):
    ax1.plot(knots, basis_1[i, :])
ax1.plot([], [], linewidth=1, color="k", label="B-Spline basis")
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$\hat{b}_{A,\,p}(\xi)$")

ax2 = ax1.twinx()
ax2.plot(quadrule.quadpts, weights_1[idx, :], "ko")
ax2.grid(None)
ax2.set_ylim([-1, 1])

fig.tight_layout()
fig.savefig("periodic_splines_ders", dpi=300)

# Fig 3
ctrlpts = np.array([[0, 0], [1, 2], [2, 4], [5, 0.5], [3, 0], [0, 0], [1, 2], [2, 4]])
evalpts = basis_0.T @ ctrlpts
fig, ax1 = plt.subplots()
ax1.plot(ctrlpts[:, 0], ctrlpts[:, 1])
ax1.plot(evalpts[:, 0], evalpts[:, 1], "--")
fig.tight_layout()
fig.savefig("periodic_curve", dpi=300)
