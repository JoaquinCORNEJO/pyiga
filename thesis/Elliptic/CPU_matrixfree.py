from thesis.Elliptic.__init__ import *
from src.lib_mygeometry import create_uniform_knotvector
from src.lib_quadrules import weighted_quadrature, gauss_quadrature
from src.lib_tensor_maths import bspline_operations
import time

# Set global variables
RUNSIMU = False
degree_list = range(1, 11)
NDIM = 3
nbel = 64  # 64 or 20

if RUNSIMU:

    timeMF_conductivity = np.zeros((len(degree_list), 2))
    timeMF_conductivity[:, 0] = degree_list

    timeMF_stiffness = np.zeros((len(degree_list), 2))
    timeMF_stiffness[:, 0] = degree_list

    timeMF_capacity = np.zeros((len(degree_list), 2))
    timeMF_capacity[:, 0] = degree_list

    for quadrule, quadtype in zip(["gs", "wq", "wq"], ["leg", 2, 1]):

        quad_args = {"quadrule": quadrule, "type": quadtype}

        for i, degree in enumerate(degree_list):

            knotvector = create_uniform_knotvector(degree, nbel)
            if quad_args["quadrule"] == "wq":
                quadrature = weighted_quadrature(
                    degree, knotvector, quad_args=quad_args
                )
            if quad_args["quadrule"] == "gs":
                quadrature = gauss_quadrature(degree, knotvector, quad_args=quad_args)
            quadrature.export_quadrature_rules()
            nbctrlpts_total = np.product(
                np.array([quadrature.nbctrlpts for i in range(NDIM)])
            )
            nbqp_total = np.product(np.array([quadrature.nbqp for i in range(NDIM)]))
            inv_jac, det_jac = np.ones((NDIM, NDIM, nbqp_total)), np.ones(nbqp_total)

            # ------------------
            # Compute MF product
            # ------------------
            if quadrule != "gs" or degree < 11:
                prop = np.ones(nbqp_total)
                start = time.process_time()
                bspline_operations.compute_mf_scalar_u_v(
                    [quadrature] * NDIM,
                    prop,
                    np.random.random(nbctrlpts_total),
                    allow_lumping=False,
                )
                finish = time.process_time()
                print("Time Capacity:%.2e" % (finish - start))
                timeMF_capacity[i, 1] = finish - start
                np.savetxt(
                    f"{FOLDER2DATA}MF_capacity_{nbel}_{quadrule}_{str(quadtype)}.dat",
                    timeMF_capacity,
                )

            if quadrule != "gs" or degree < 7:

                prop = np.ones((NDIM, NDIM, nbqp_total))
                start = time.process_time()
                bspline_operations.compute_mf_scalar_gradu_gradv(
                    [quadrature] * NDIM, prop, np.random.random(nbctrlpts_total)
                )
                finish = time.process_time()
                print("Time Conductivity:%.2e" % (finish - start))
                timeMF_conductivity[i, 1] = finish - start
                np.savetxt(
                    f"{FOLDER2DATA}MF_conductivity_{nbel}_{quadrule}_{str(quadtype)}.dat",
                    timeMF_conductivity,
                )

            if quadrule != "gs" or degree < 6:
                prop = np.ones((NDIM, NDIM, nbqp_total))
                array_in = np.random.random((NDIM, nbctrlpts_total))
                array_out = np.zeros_like(array_in)
                start = time.process_time()
                for k in range(NDIM):
                    array_out[k, :] = sum(
                        bspline_operations.compute_mf_scalar_gradu_gradv(
                            [quadrature] * NDIM, prop, array_in[j, :]
                        )
                        for j in range(NDIM)
                    )

                finish = time.process_time()
                print("Time Stiffness:%.2e" % (finish - start))
                timeMF_stiffness[i, 1] = finish - start
                np.savetxt(
                    f"{FOLDER2DATA}MF_stiffness_{nbel}_{quadrule}_{str(quadtype)}.dat",
                    timeMF_stiffness,
                )

FILENAME = "MF_stiffness_64"
sufix_list = ["gs_leg", "wq_1"]
label_list = ["MF-Gauss quadrature", "MF-Weighted quadrature"]
plotoptions = [CONFIGLINE_IGA, CONFIGLINE_WQ]
annotation = ["190 s", "5 s"]

# Load data
fig, ax = plt.subplots(figsize=(5, 5))
for label, sufix, plotops, annot in zip(label_list, sufix_list, plotoptions, annotation):
    file = np.loadtxt(f"{FOLDER2DATA}{FILENAME}_{sufix}.dat")
    degree_list = file[:, 0]
    time_elapsed = file[:, 1]
    if sufix.split('_')[0] == 'wq':
        ax.semilogy(degree_list[1:], time_elapsed[1:], label=label, **plotops)
    else:
        ax.semilogy(degree_list, time_elapsed, label=label, **plotops)
        
    ax.text(
        degree_list[-1],
        0.5 * time_elapsed[-1],
        f"{annot}",
        fontsize=12,
        ha="center",
        va="bottom",
        bbox=dict(
            facecolor="white", alpha=0.7, boxstyle="round,pad=0.3", edgecolor="gray"
        ),
    )

ax.legend(loc="upper left")
ax.set_xlabel("Degree")
ax.set_ylabel("CPU time (s)")
ax.set_xlim([0, 11])
ax.set_ylim([1e0, 1e3])
fig.tight_layout()
fig.savefig(f"{FOLDER2RESU}{FILENAME}.pdf")

FILENAME = "MF_capacity_20"
sufix_list = ["gs_leg", "wq_1"]
label_list = ["MF-Gauss quadrature", "MF-Weighted quadrature"]
annotation = ["0.27 s", "4.6 ms"]
plotoptions = [CONFIGLINE_IGA, CONFIGLINE_WQ]

# Load data
fig, ax = plt.subplots(figsize=(5, 5))
for label, sufix, plotops, annot in zip(
    label_list, sufix_list, plotoptions, annotation
):
    file = np.loadtxt(f"{FOLDER2DATA}{FILENAME}_{sufix}.dat")
    degree_list = file[:, 0]
    time_elapsed = file[:, 1]
    if sufix.split('_')[0] == 'wq':
        ax.semilogy(degree_list[1:], time_elapsed[1:], label=label, **plotops)
    else:
        ax.semilogy(degree_list, time_elapsed, label=label, **plotops)
        

    ax.text(
        degree_list[-1],
        1.5 * time_elapsed[-1],
        f"{annot}",
        fontsize=12,
        ha="center",
        va="bottom",
        bbox=dict(
            facecolor="white", alpha=0.7, boxstyle="round,pad=0.3", edgecolor="gray"
        ),
    )

ax.legend(loc="lower left")
ax.set_xlabel("Degree")
ax.set_ylabel("CPU time (s)")
ax.set_xlim([0, 11])
ax.set_ylim([1e-4, 1e0])
fig.tight_layout()
fig.savefig(f"{FOLDER2RESU}{FILENAME}.pdf")
