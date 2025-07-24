from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem


def power_density(args: dict):
    x = args["position"]
    f = 4 * np.pi**2 * np.sin(2 * np.pi * x)
    return f


def exact_temperature(args: dict):
    x = args["position"]
    u = np.sin(2 * np.pi * x)
    return u


def simulate(degree, nbel, quad_args):
    # Create geometry
    geometry = mygeomdl(
        {"name": "line", "degree": degree, "nbel": nbel, "geo_parameters": {"L": 1.0}}
    ).export_geometry()
    patch = singlepatch(geometry, quad_args=quad_args)

    # Create material
    material = heat_transfer_mat()
    material.add_conductivity(1.0, is_uniform=True, ndim=1)

    # Create boundary condition
    boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=1)
    boundary.add_constraint(
        location_list=[{"direction": "x", "face": "both"}], constraint_type="dirichlet"
    )

    # Set transient heat transfer problem
    problem = heat_transfer_problem(material, patch, boundary)

    # Create external force
    external_force = problem.assemble_volumetric_force(power_density)

    # Solve problem
    temperature = np.zeros_like(external_force)
    problem.solve_heat_transfer(temperature, external_force)
    return problem, temperature


degree_list = np.arange(1, 5)
cuts_list = np.arange(1, 8)
relerror_list = np.zeros_like(cuts_list, dtype=float)
color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots(figsize=(5.5, 5))
gauss_plot = {"marker": "s", "linestyle": "-", "markersize": 10}
wq1_plot = {"marker": "o", "linestyle": "--", "markersize": 4}
wq2_plot = {"marker": "x", "linestyle": ":", "markersize": 4}

figname = RESULT_FOLDER + "convergence_steady_1d"
for quadrule, quadtype, plotops in zip(
    ["gs", "wq", "wq"], ["leg", 1, 2], [gauss_plot, wq1_plot, wq2_plot]
):
    quad_args = {"quadrule": quadrule, "type": quadtype}
    for i, degree in enumerate(degree_list):
        color = color_list[i]
        for j, cuts in enumerate(cuts_list):
            problem, temperature = simulate(degree, 2**cuts, quad_args)
            _, relerror_list[j] = problem.norm_of_error(
                temperature,
                norm_args={
                    "type": "L2",
                    "exact_function": exact_temperature,
                },
            )

        label = f"IGA-GL deg. {degree}" if quadtype == "leg" else None
        ax.loglog(
            2**cuts_list,
            relerror_list,
            label=label,
            color=color,
            markerfacecolor="w",
            **plotops,
        )
        fig.savefig(figname)

ax.loglog(
    [],
    [],
    color="k",
    markerfacecolor="w",
    **wq1_plot,
    label="IGA-WQ 1",
)
ax.loglog(
    [],
    [],
    color="k",
    markerfacecolor="w",
    **wq2_plot,
    label="IGA-WQ 2",
)

ax.set_ylim([1e-10, 1])
ax.set_ylabel(r"Relative $||u-u^h||_{L^2(\Omega)}$")
ax.set_xlabel("Number of elements")
ax.legend()
fig.tight_layout()
fig.savefig(figname)
