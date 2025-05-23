from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem


def power_density(args: dict):
    position = args.get("position")
    x = position[0, :]
    y = position[1, :]

    f = (
        3
        * np.pi**2
        * np.sin(np.pi * x)
        * np.sin(np.pi * y)
        * (x**2 + y**2 - 1)
        * (x**2 + y**2 - 4)
        - 16 * y**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        - 6 * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        - 6 * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        - 8 * x * y * np.sin(np.pi * x) * np.sin(np.pi * y)
        - np.pi**2
        * np.cos(np.pi * x)
        * np.cos(np.pi * y)
        * (x**2 + y**2 - 1)
        * (x**2 + y**2 - 4)
        - 4 * x * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        - 2 * x * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (x**2 + y**2 - 1)
        - 4 * x * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        - 2 * x * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (x**2 + y**2 - 4)
        - 2 * y * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        - 8 * y * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (x**2 + y**2 - 1)
        - 2 * y * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        - 8 * y * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (x**2 + y**2 - 4)
        - 8 * x**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    )
    return f


def exact_temperature(args: dict):
    position = args.get("position")
    x = position[0, :]
    y = position[1, :]
    u = (
        np.sin(np.pi * x)
        * np.sin(np.pi * y)
        * (x**2 + y**2 - 1.0)
        * (x**2 + y**2 - 4.0)
    )
    return u


def ders_exact_temperature(args: dict):
    position = args.get("position")
    x = position[0, :]
    y = position[1, :]
    uders = np.zeros((1, 2, np.size(position, axis=1)))
    uders[0, 0, :] = (
        2 * x * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        + 2 * x * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        + np.pi
        * np.cos(np.pi * x)
        * np.sin(np.pi * y)
        * (x**2 + y**2 - 1)
        * (x**2 + y**2 - 4)
    )
    uders[0, 1, :] = (
        2 * y * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        + 2 * y * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        + np.pi
        * np.cos(np.pi * y)
        * np.sin(np.pi * x)
        * (x**2 + y**2 - 1)
        * (x**2 + y**2 - 4)
    )
    return uders


def simulate(degree, nbel, quad_args=None):
    geo_args = {
        "name": "QA",
        "degree": degree,
        "nbel": nbel,
        "geo_parameters": {"Rin": 1.0, "Rex": 2.0},
    }

    if quad_args is None:
        quad_args = {"quadrule": "gs", "type": "leg"}
    material = heat_transfer_mat()
    material.add_conductivity(
        np.array([[1, 0.5], [0.5, 2]]), is_uniform=True, shape_tensor=2
    )
    geometry = mygeomdl(geo_args).export_geometry()
    patch = singlepatch(geometry, quad_args=quad_args)

    # Set Dirichlet boundaries
    boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nbvars=1)
    boundary.add_constraint(
        location_list=[{"direction": "x,y", "face": "both,both"}],
        constraint_type="dirichlet",
    )

    # Solve elastic problem
    problem = heat_transfer_problem(material, patch, boundary)
    external_force = problem.assemble_volumetric_force(power_density)
    temperature = np.zeros_like(external_force)
    problem.solve_heat_transfer(temperature, external_force)
    return problem, temperature


problem, temperature = simulate(4, 8)
problem.part.postprocessing_primal(
    fields={"temp": temperature}, name="steady", folder=RESULT_FOLDER
)

degree_list = np.arange(1, 5)
cuts_list = np.arange(1, 7)
relerror_list = np.zeros_like(cuts_list, dtype=float)
color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots(figsize=(5.5, 5))
gauss_plot = {"marker": "s", "linestyle": "-", "markersize": 10}
wq1_plot = {"marker": "o", "linestyle": "--", "markersize": 4}
wq2_plot = {"marker": "x", "linestyle": ":", "markersize": 4}

figname = RESULT_FOLDER + "convergence_steady_2d_l2"
for quadrule, quadtype, plotpars in zip(
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
                    "type": "l2",
                    "exact_function": exact_temperature,
                    "exact_function_ders": ders_exact_temperature,
                },
            )

        label = f"IGA-GL deg. {degree}" if quadtype == "leg" else None
        ax.loglog(
            2**cuts_list,
            relerror_list,
            label=label,
            color=color,
            marker=plotpars["marker"],
            markerfacecolor="w",
            markersize=plotpars["markersize"],
            linestyle=plotpars["linestyle"],
        )
        fig.savefig(figname)

ax.loglog(
    [],
    [],
    color="k",
    marker=wq1_plot["marker"],
    markerfacecolor="w",
    markersize=wq1_plot["markersize"],
    linestyle=wq1_plot["linestyle"],
    label="IGA-WQ 1",
)
ax.loglog(
    [],
    [],
    color="k",
    marker=wq2_plot["marker"],
    markerfacecolor="w",
    markersize=wq2_plot["markersize"],
    linestyle=wq2_plot["linestyle"],
    label="IGA-WQ 2",
)

ax.set_ylim([1e-8, 1])
ax.set_ylabel(r"Relative $||u-u^h||_{L^2(\Omega)}$")
ax.set_xlabel("Number of elements")
ax.legend()
fig.tight_layout()
fig.savefig(figname)
