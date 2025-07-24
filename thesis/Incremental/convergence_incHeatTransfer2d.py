from thesis.SpaceTime.__init__ import *
from thesis.SpaceTime.input_data import *
from thesis.Incremental.norm_incremental_methods import norm_of_error
import time


def simulate_incremental_bdf(
    degree,
    cuts,
    powerdensity,
    nbel_time=None,
    quad_args=None,
    solve_system=True,
    ivp="alpha",
):

    # Create geometry
    if quad_args is None:
        quad_args = {"quadrule": "gs", "type": "leg"}
    if nbel_time is None:
        nbel_time = int(2**CUTS_TIME)

    geometry = mygeomdl(
        {"name": "QA", "degree": degree, "nbel": int(2**cuts)}
    ).export_geometry()
    patch = singlepatch(geometry, quad_args=quad_args)
    time_inc = np.linspace(0, 1.0, nbel_time + 1)

    # Add material
    material = heat_transfer_mat()
    material.add_capacity(1, is_uniform=True)
    material.add_conductivity(conductivity_property, is_uniform=False, ndim=2)

    # Block boundaries
    boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=1)
    boundary.add_constraint(
        location_list=[{"direction": "x,y", "face": "both,both"}],
        constraint_type="dirichlet",
    )

    # Transient model
    problem_inc = heat_transfer_problem(material, patch, boundary)
    temperature_inc = np.zeros((patch.nbctrlpts_total, len(time_inc)))
    if not solve_system:
        return problem_inc, time_inc, temperature_inc

    # Add external force
    external_force = np.zeros((problem_inc.part.nbctrlpts_total, len(time_inc)))
    for i, t in enumerate(time_inc):
        external_force[:, i] = problem_inc.assemble_volumetric_force(
            powerdensity, args={"time": t}
        )

    # Solve
    problem_inc._maxiters_nonlinear = 50
    problem_inc._tolerance_linear = 1e-12
    problem_inc._tolerance_nonlinear = 1e-12
    if ivp == "alpha":
        problem_inc.solve_heat_transfer(
            temperature_inc, external_force, time_inc, alpha=0.5
        )
    else:
        assert ivp in ["BDF1", "BDF2", "BDF3", "BDF4"], "Invalid method"
        problem_inc.solve_heat_transfer(
            temperature_inc,
            external_force,
            tspan=(time_inc[0], time_inc[-1]),
            nsteps=nbel_time,
            norder=int(ivp[-1]),
            type_solver="bdf",
        )

    return problem_inc, time_inc, temperature_inc


RUNSIMU = False
PLOTRELATIVE = True

degree, cuts = 8, 6
quad_args = {"quadrule": "gs", "type": "leg"}
IVP_method_list = ["BDF1", "BDF2", "BDF3", "alpha"]
nbel_time_list = np.array([2**cuts for cuts in range(2, 8)])
abserror_inc1 = np.ones_like(nbel_time_list, dtype=float)
relerror_inc1 = np.ones_like(abserror_inc1)
abserror_inc2 = np.ones_like(abserror_inc1)
relerror_inc2 = np.ones_like(abserror_inc1)
time_inc_list = np.ones_like(nbel_time_list)

if RUNSIMU:

    for IVP_method in IVP_method_list:

        for i, nbel_time in enumerate(nbel_time_list):

            start = time.process_time()
            problem_inc, time_inc, temp_inc = simulate_incremental_bdf(
                degree,
                cuts,
                powerDensityRing_inc,
                nbel_time=nbel_time,
                quad_args=quad_args,
                ivp=IVP_method,
            )
            finish = time.process_time()

            time_inc_list[i] = finish - start

            abserror_inc1[i], relerror_inc1[i] = norm_of_error(
                problem_inc, temp_inc, time_inc, exactTemperatureRing_inc
            )

            abserror_inc2[i], relerror_inc2[i] = problem_inc.norm_of_error(
                temp_inc[:, -1],
                norm_args={
                    "type": "L2",
                    "exact_function": exactTemperatureRing_inc,
                    "exact_args": {"time": time_inc[-1]},
                },
            )

            np.savetxt(
                f"{FOLDER2DATA}abserrorstag_inc1_{IVP_method}.dat", abserror_inc1
            )
            np.savetxt(
                f"{FOLDER2DATA}relerrorstag_inc1_{IVP_method}.dat", relerror_inc1
            )
            np.savetxt(
                f"{FOLDER2DATA}abserrorstag_inc2_{IVP_method}.dat", abserror_inc2
            )
            np.savetxt(
                f"{FOLDER2DATA}relerrorstag_inc2_{IVP_method}.dat", relerror_inc2
            )
            np.savetxt(f"{FOLDER2DATA}timestag_inc1_{IVP_method}.dat", time_inc_list)


from mpltools import annotation

fig, ax = plt.subplots(figsize=(5, 5))

for i, IVP_method in enumerate(IVP_method_list):
    label = (
        "Crank-Nicolson" if IVP_method == "alpha" else f"Implicit BDF-{IVP_method[-1]}"
    )
    if PLOTRELATIVE:
        error_list = np.loadtxt(f"{FOLDER2DATA}relerrorstag_inc1_{IVP_method}.dat")
    else:
        error_list = np.loadtxt(f"{FOLDER2DATA}abserrorstag_inc1_{IVP_method}.dat")
    nbctrlpts = nbel_time_list + 1
    if i < 3:
        ax.loglog(nbctrlpts, error_list, **CONFIGLINE_BDF, label=label)
    elif i == 3:
        ax.loglog(nbctrlpts, error_list, **CONFIGLINE_INC, color="k", label=label)

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
    ax.set_ylim(top=1e0, bottom=1e-8)
else:
    ax.set_ylabel(r"$L^2(\Pi)$ error")
    ax.set_ylim(top=1e1, bottom=1e-6)

ax.set_xlabel("Number of control points in time \n(or number of time-steps)")
ax.set_xlim(left=5, right=200)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(f"{FOLDER2RESU}StagnationError2d.pdf")
