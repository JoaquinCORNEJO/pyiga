"""
.. Test of mecanical dynamics 1D
.. Author: Fabio MADIE
.. Joaquin Cornejo added some corrections 28 nov. 2024
"""

from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import J2plasticity1d
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_mechanical import mechanical_problem

# Set global variables
YOUNG, RHO, LENGTH = 2.0e3, 7.8e-6, 1000
WAVEVEL = np.sqrt(YOUNG / RHO)


def compute_initial_displacement(args: dict):
    x = args["position"]
    u = np.exp(-1.0e-3 / 2 * (x - LENGTH / 2) ** 2)
    return u


# Create geometry
degree, nbel = 3, 256
geometry = mygeomdl(
    {"name": "line", "degree": degree, "nbel": nbel, "geo_parameters": {"L": LENGTH}}
).export_geometry()
patch = singlepatch(geometry, quad_args={"quadrule": "gs"})

# Create boundary condition
boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nbvars=1)

# Set material
material = J2plasticity1d({"elastic_modulus": YOUNG})
material.add_density(RHO, is_uniform=True)

# Set mechanical problem
problem = mechanical_problem(material, patch, boundary, allow_lumping=True)
timespan = LENGTH / WAVEVEL
freqmax = np.sqrt(np.max(problem.solve_eigenvalue_problem(which="LM", k=2)[0]))
nbsteps_min = int(1.005 * np.ceil(timespan * freqmax / 2))

# Create external force
time_list = np.linspace(0, timespan, nbsteps_min)
external_force = np.zeros((1, patch.nbctrlpts_total, len(time_list)))
displacement = np.zeros_like(external_force)

# Compute initial values
displacement[:, :, 0] = problem.compute_L2projection(
    compute_initial_displacement({"position": problem.part.qp_phy})
)

# Solve linear dynamics with newmark scheme
problem.solve_explicit_linear_dynamics(displacement, external_force, time_list)

# Post processing
from src.lib_tensor_maths import bspline_operations

fig, ax = plt.subplots()
knots_interp = np.linspace(0, 1, 501)

disp_interp_initial = np.ravel(
    bspline_operations.interpolate_meshgrid(
        quadrule_list=patch.quadrule_list,
        knots_list=[knots_interp],
        u_ctrlpts=np.atleast_2d(displacement[0, :, 0]),
    )
)
ax.plot(knots_interp * LENGTH, np.ravel(disp_interp_initial), "--", label="Initial")

disp_interp_final = np.ravel(
    bspline_operations.interpolate_meshgrid(
        quadrule_list=patch.quadrule_list,
        knots_list=[knots_interp],
        u_ctrlpts=np.atleast_2d(displacement[0, :, -1]),
    )
)
ax.plot(knots_interp * LENGTH, np.ravel(disp_interp_final), ".", label="Final")

disp_exact = compute_initial_displacement({"position": knots_interp * LENGTH})
ax.plot(knots_interp * LENGTH, disp_exact, label="Exact", alpha=0.5)

ax.set_xlabel("Position")
ax.set_ylabel("Displacement")
ax.legend()
fig.savefig(f"{RESULT_FOLDER}wave_propagation")
