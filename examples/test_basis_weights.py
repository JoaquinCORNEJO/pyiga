from src.__init__ import *
from src.lib_mygeometry import create_uniform_knotvector
from src.lib_quadrules import *


def relative_error(array_interp: sp.csr_matrix, array_th: sp.csr_matrix):
    error = array_th - array_interp
    return np.linalg.norm(error.toarray()) / np.linalg.norm(array_th.toarray())

# EXAMPLE 1
degree, nbel = 2, 3
knotvector = create_uniform_knotvector(degree, nbel)
nb_ctrlpts = len(knotvector) - degree - 1

quadrule = weighted_quadrature(
    degree,
    knotvector,
    {
        "type": 2,
        "position_rule": "midpoint",
    },
)

quadrule.export_quadrature_rules()
basis, weights = quadrule.basis, quadrule.weights
quadpts = quadrule.quadpts

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(quadrule._unique_kv, np.zeros_like(quadrule._unique_kv), marker="s", color="k")
weightsmatrix = weights[0].todense()
for i in range(quadrule.nbctrlpts):
    ax.plot(quadpts, np.ravel(weightsmatrix[i, :]))
fig.savefig(f"{RESULT_FOLDER}tweights_W0_{degree}.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(quadrule._unique_kv, np.zeros_like(quadrule._unique_kv), marker="s", color="k")
weightsmatrix = weights[-1].todense()
for i in range(quadrule.nbctrlpts):
    ax.plot(quadpts, np.ravel(weightsmatrix[i, :]))
fig.savefig(f"{RESULT_FOLDER}tweights_W1_{degree}.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(quadrule._unique_kv, np.zeros_like(quadrule._unique_kv), marker="s", color="k")
basismatrix = basis[0].todense()
for i in range(quadrule.nbctrlpts):
    ax.plot(quadpts, np.ravel(basismatrix[i, :]))
fig.savefig(f"{RESULT_FOLDER}tweights_B0_{degree}.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(quadrule._unique_kv, np.zeros_like(quadrule._unique_kv), marker="s", color="k")
basismatrix = basis[-1].todense()
for i in range(quadrule.nbctrlpts):
    ax.plot(quadpts, np.ravel(basismatrix[i, :]))
fig.savefig(f"{RESULT_FOLDER}tweights_B1_{degree}.png")


# EXAMPLE 2
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

nbel = 3
for degree in [2]:
    knotvector = create_uniform_knotvector(degree, nbel)
    nb_ctrlpts = len(knotvector) - degree - 1

    quadrule = weighted_quadrature(degree, knotvector, {"type": 2})
    quadrule.export_quadrature_rules()
    basis, weights = quadrule.basis, quadrule.weights
    quadpts = quadrule.quadpts

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        quadrule._unique_kv, np.zeros_like(quadrule._unique_kv), marker="s", color="k"
    )
    weightsmatrix = weights[0].todense()
    for i in range(quadrule.nbctrlpts):
        ax.plot(quadpts, np.ravel(weightsmatrix[i, :]))
    fig.savefig(f"{RESULT_FOLDER}tweights_W0_{degree}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        quadrule._unique_kv, np.zeros_like(quadrule._unique_kv), marker="s", color="k"
    )
    weightsmatrix = weights[-1].todense()
    for i in range(quadrule.nbctrlpts):
        ax.plot(quadpts, np.ravel(weightsmatrix[i, :]))
    fig.savefig(f"{RESULT_FOLDER}tweights_W1_{degree}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        quadrule._unique_kv, np.zeros_like(quadrule._unique_kv), marker="s", color="k"
    )
    basismatrix = basis[0].todense()
    for i in range(quadrule.nbctrlpts):
        ax.plot(quadpts, np.ravel(basismatrix[i, :]))
    fig.savefig(f"{RESULT_FOLDER}tweights_B0_{degree}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        quadrule._unique_kv, np.zeros_like(quadrule._unique_kv), marker="s", color="k"
    )
    basismatrix = basis[-1].todense()
    for i in range(quadrule.nbctrlpts):
        ax.plot(quadpts, np.ravel(basismatrix[i, :]))
    fig.savefig(f"{RESULT_FOLDER}tweights_B1_{degree}.png")

# EXAMPLE 3
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
nbel_list = [2**i for i in np.arange(2, 6)]

for ax, var_name in zip(np.ravel(axs), ["I00", "I01", "I10", "I11"]):
    for degree in range(2, 5):

        error_list = []

        for nbel in nbel_list:
            knotvector = create_uniform_knotvector(degree, nbel)
            nb_ctrlpts = len(knotvector) - degree - 1

            # Weighted quadrature
            quadrule_wq = weighted_quadrature(degree, knotvector, {"type": 1})
            quadrule_wq.export_quadrature_rules()
            basis, weights = quadrule_wq.basis, quadrule_wq.weights
            quadpts = quadrule_wq.quadpts
            [B0f, B1f] = basis
            [W00f, W01f, W10f, W11f] = weights
            I00f = W00f @ B0f.T
            I01f = W01f @ B1f.T
            I10f = W10f @ B0f.T
            I11f = W11f @ B1f.T

            # Gauss quadrature
            quadrule_gs = gauss_quadrature(degree, knotvector, {})
            quadrule_gs.export_quadrature_rules()
            basis, weights = quadrule_wq.basis, quadrule_wq.weights
            [B0, B1] = basis
            [W00, W01, W10, W11] = weights

            I00 = W00 @ B0.T
            I01 = W01 @ B1.T
            I10 = W10 @ B0.T
            I11 = W11 @ B1.T

            # Compare results
            if var_name == "I00":
                var1 = I00
                var2 = I00f
            elif var_name == "I01":
                var1 = I01
                var2 = I01f
            elif var_name == "I10":
                var1 = I10
                var2 = I10f
            elif var_name == "I11":
                var1 = I11
                var2 = I11f

            error = relative_error(var2, var1)
            print(error)
            if error > 1e-5:
                raise Warning("Something wrong happend")
            error_list.append(error)

        label = "Degree $p = $ " + str(degree)
        ax.loglog(nbel_list, error_list, "-o", label=label)

    ax.set_xlabel("Discretization level " + r"$h^{-1}$")
    ax.set_ylabel("Relative error")

axs[-1, -1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
fig.tight_layout()
fig.savefig(RESULT_FOLDER + "terrorweights" + ".png")
