from thesis.Incremental.__init__ import *
from src.lib_tensor_maths import bspline_operations
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem
from thesis.Incremental.norm_incremental_methods import norm_of_error
from numpy import sin, cos, pi, tanh

CST = 100


def conductivity_property(args: dict):
    temperature = args.get("temperature")
    conductivity = np.zeros(shape=(1, 1, *np.shape(temperature)))
    conductivity[0, 0, ...] = 3.0 + 2.0 * tanh(temperature / 50)
    return conductivity


def exact_temperature(args: dict):
    t = args["time"]
    x = args["position"][0, :]
    u = CST * sin(2 * pi * x) * sin(pi / 2 * t) * (1 + 0.75 * cos(3 * pi / 2 * t))
    return u


def power_density(args: dict):
    t = args["time"]
    x = args["position"]
    u = CST * sin(2 * pi * x) * sin(pi / 2 * t) * (1 + 0.75 * cos(3 * pi / 2 * t))
    f = (
        (
            CST
            * pi
            * cos((pi * t) / 2)
            * sin(2 * pi * x)
            * ((3 * cos((3 * pi * t) / 2)) / 4 + 1)
        )
        / 2
        - (9 * CST * pi * sin((pi * t) / 2) * sin((3 * pi * t) / 2) * sin(2 * pi * x))
        / 8
        + 4 * pi**2 * u * (2 * tanh(u / 50) + 3)
        + (
            4
            * CST**2
            * pi**2
            * cos(2 * pi * x) ** 2
            * sin((pi * t) / 2) ** 2
            * ((3 * cos((3 * pi * t) / 2)) / 4 + 1) ** 2
            * (tanh(u / 50) ** 2 - 1)
        )
        / 25
    )
    return f


def simulate_ht(degree, cuts, nbel_time, quad_args, ivp="alpha"):

    # Create geometry
    geometry = mygeomdl(
        {
            "name": "line",
            "degree": degree,
            "nbel": int(2**cuts),
            "geo_parameters": {"L": 1.0},
        }
    ).export_geometry()
    patch = singlepatch(geometry, quad_args=quad_args)
    time_inc = np.linspace(0.0, 1.0, nbel_time + 1)

    # Add material
    material = heat_transfer_mat()
    material.add_capacity(1.0, is_uniform=True)
    material.add_conductivity(conductivity_property, is_uniform=False, ndim=1)

    # Block boundaries
    boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=1)
    boundary.add_constraint(
        location_list=[{"direction": "x", "face": "both"}], constraint_type="dirichlet"
    )

    # Transient model
    problem = heat_transfer_problem(material, patch, boundary)
    temperature = np.zeros((problem.part.nbctrlpts_total, len(time_inc)))
    # Create external force
    external_heat_source = np.zeros_like(temperature)
    for i, t in enumerate(time_inc):
        external_heat_source[:, i] = problem.assemble_volumetric_force(
            power_density, args={"time": t}
        )

    if ivp == "alpha":

        # Solve problem
        problem._tolerance_nonlinear = 1e-8
        problem.solve_heat_transfer(
            temperature, external_heat_source, time_inc, alpha=0.5
        )

    else:
        assert ivp in ["BDF1", "BDF2", "BDF3", "BDF4"], "Invalid method"

        # Solve problem
        problem._tolerance_nonlinear = 1e-8
        problem.solve_heat_transfer(
            temperature,
            external_heat_source,
            tspan=(time_inc[0], time_inc[-1]),
            nsteps=nbel_time,
            norder=int(ivp[-1]),
            type_solver="bdf",
        )

    return problem, time_inc, temperature


# Set global variables
PLOTRELATIVE = True
RUNSIMU = True

degree, cuts = 8, 6
quad_args = {"quadrule": "gs", "type": "leg"}
IVP_method_list = ["BDF1", "BDF2", "BDF3", "alpha"]
nbel_time_list = np.array([2**cuts for cuts in range(2, 9)])

if RUNSIMU:

    for IVP_method in IVP_method_list:

        abserror_list = np.ones(len(nbel_time_list))
        relerror_list = np.ones(len(nbel_time_list))

        for i, nbel_time in enumerate(nbel_time_list):

            problem_inc, time_inc, temp_inc = simulate_ht(
                degree, cuts, nbel_time=nbel_time, quad_args=quad_args, ivp=IVP_method
            )

            abserror_list[i], relerror_list[i] = norm_of_error(
                problem_inc,
                temp_inc,
                time_inc,
                exact_fun=exact_temperature,
            )

            np.savetxt(f"{FOLDER2DATA}abserrorstag_inc_{IVP_method}.dat", abserror_list)
            np.savetxt(f"{FOLDER2DATA}relerrorstag_inc_{IVP_method}.dat", relerror_list)

from mpltools import annotation

IVP_method_list = ["BDF1", "BDF2", "BDF3", "alpha"]
fig, ax = plt.subplots(figsize=(5, 5))
for i, IVP_method in enumerate(IVP_method_list):
    label = (
        "Crank-Nicolson" if IVP_method == "alpha" else f"Implicit BDF-{IVP_method[-1]}"
    )
    if PLOTRELATIVE:
        error_list = np.loadtxt(f"{FOLDER2DATA}relerrorstag_inc_{IVP_method}.dat")
    else:
        error_list = np.loadtxt(f"{FOLDER2DATA}abserrorstag_inc_{IVP_method}.dat")
    nbctrlpts = nbel_time_list + 1
    if i < 3:
        ax.loglog(nbctrlpts[1:], error_list[1:], **CONFIGLINE_BDF, label=label)
    elif i == 3:
        ax.loglog(
            nbctrlpts[1:], error_list[1:], **CONFIGLINE_INC, color="k", label=label
        )

    slope = np.polyfit(np.log10(nbctrlpts[3:]), np.log10(error_list[3:]), 1)[0]
    slope = round(slope, 1)
    if i != 2:
        annotation.slope_marker(
            (nbctrlpts[-2], error_list[-2]),
            slope,
            poly_kwargs={"facecolor": (0.73, 0.8, 1)},
            ax=ax,
        )
    else:
        annotation.slope_marker(
            (nbctrlpts[-1], error_list[-1]),
            slope,
            poly_kwargs={"facecolor": (0.73, 0.8, 1)},
            ax=ax,
            invert=True,
        )

if PLOTRELATIVE:
    ax.set_ylabel(r"Relative $L^2(\Pi)$ error")
    ax.set_ylim(top=1e-1, bottom=1e-7)

ax.set_xlabel("Number of time-steps")
ax.set_xlim(left=5, right=500)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(f"{FOLDER2RESU}stagnation_error_inc.pdf")
