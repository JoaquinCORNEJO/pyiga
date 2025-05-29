from thesis.Elliptic.__init__ import *
from src.lib_part import vtk2png
import pickle

RUNSIMU = False
deg_list = np.arange(1, 6)
cut_list = np.arange(1, 8)
nbel_list = 2**cut_list

if RUNSIMU:

    degree, cuts = 8, 8
    problem, displacement = simulate_el(degree, cuts)
    np.save(f"{FOLDER2DATA}dispel.npy", displacement)
    with open(f"{FOLDER2DATA}refpartel.pkl", "wb") as outp:
        pickle.dump(problem.part, outp, pickle.HIGHEST_PROTOCOL)

    disp_ref = np.load(f"{FOLDER2DATA}dispel.npy")
    with open(f"{FOLDER2DATA}refpartel.pkl", "rb") as inp:
        part_ref = pickle.load(inp)

    abs_error_list = np.ones((len(deg_list), len(cut_list)))
    rel_error_list = np.ones_like(abs_error_list)

    for quadrule, quadtype in zip(["wq", "wq", "gs"], [1, 2, "leg"]):
        quad_args = {"quadrule": quadrule, "type": quadtype}

        for i, degree in enumerate(deg_list):
            for j, cuts in enumerate(cut_list):
                problem, displacement = simulate_el(degree, cuts, quad_args)
                abs_error_list[i, j], rel_error_list[i, j] = problem.norm_of_error(
                    u_ctrlpts=displacement,
                    norm_args={"type": "H1", "part_ref": part_ref, "u_ref": disp_ref},
                )
            np.savetxt(
                f"{FOLDER2DATA}abs_error_el_{quadrule}_{str(quadtype)}.dat",
                abs_error_list,
            )
            np.savetxt(
                f"{FOLDER2DATA}rel_error_el_{quadrule}_{str(quadtype)}.dat",
                rel_error_list,
            )

# problem, displacement = simulate_el(4, 6, {"quadrule": "wq", "type": 2})
# strain_3d = problem.interpolate_strain(displacement, convert_to_3d=True)
# stress_3d = problem.material.eval_elastic_stress(strain_3d)
# vmstress = problem.material.eval_von_mises_stress(stress_3d)
# problem.part.postprocessing_dual(
#     folder=FOLDER2RESU, fields={"vms": vmstress}, name="ellipticel"
# )
# vtk2png(
#     folder=FOLDER2RESU,
#     filename="ellipticel",
#     fieldname="vms",
#     cmap="coolwarm",
#     title="Von Mises stress",
# )

from mpltools import annotation

fig, ax = plt.subplots(figsize=(5, 5))
figname = f"{FOLDER2RESU}ConvergenceH1_el.pdf"
for quadrule, quadtype, plotopts in zip(
    ["gs", "wq"], ["leg", 1], [CONFIGLINE_IGA, CONFIGLINE_WQ]
):

    error_list = np.loadtxt(f"{FOLDER2DATA}rel_error_el_{quadrule}_{str(quadtype)}.dat")

    for i, degree in enumerate(deg_list):
        color = COLORLIST[i]

        if quadrule == "gs":
            ax.loglog(
                nbel_list,
                error_list[i, :],
                color=color,
				**plotopts,
            )
            slope = round(
                np.polyfit(np.log(nbel_list[2:]), np.log(error_list[i, 2:]), 1)[0], 1
            )
            annotation.slope_marker(
                (nbel_list[-2], error_list[i, -2]*2),
                slope,
                poly_kwargs={"facecolor": (0.73, 0.8, 1)},
                ax=ax,
            )
        else:
            if degree != 1:
                ax.loglog(nbel_list, error_list[i, :], color=color, **plotopts)

ax.loglog([], [], color="k", **CONFIGLINE_IGA, label="Gauss quadrature")
ax.loglog([], [], color="k", **CONFIGLINE_WQ, label="Weighted quadrature")

ax.set_ylabel(r"Relative $H^1(\Omega)$ error")
ax.set_xlabel("Number of elements by direction")
ax.set_ylim(top=1e0, bottom=1e-12)
ax.set_xlim(left=1, right=256)
ax.legend(title=r"Degree $p=1,\ldots,5$", loc="lower left")
fig.tight_layout()
fig.savefig(figname)
