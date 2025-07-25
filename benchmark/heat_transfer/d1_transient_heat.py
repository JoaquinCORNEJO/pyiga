from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem


def conductivity_property(args: dict):
    temperature = args.get("temperature")
    conductivity = np.zeros(shape=(1, 1, *np.shape(temperature)))
    conductivity[0, 0, ...] = 3.0 + 2.0 * np.tanh(temperature / 50)
    return conductivity


def power_density(args: dict):
    t = args["time"]
    x = args["position"]
    f = (
        np.pi * np.cos((np.pi * t) / 2) * np.sin(2 * np.pi * x)
    ) / 2 + 8 * np.pi**2 * np.sin((np.pi * t) / 2) * np.sin(2 * np.pi * x)
    return f


# Create geometry
degree, nbel, length = 6, 5, 1.0
geometry = mygeomdl(
    {"name": "line", "degree": degree, "nbel": nbel, "geo_parameters": {"L": length}}
).export_geometry()
patch = singlepatch(geometry, quad_args={"quadrule": "wq"})

# Create material
material = heat_transfer_mat()
material.add_conductivity(conductivity_property, is_uniform=False, ndim=1)
material.add_capacity(1.0, is_uniform=True)

# Create boundary condition
boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=1)
boundary.add_constraint(
    location_list=[{"direction": "x", "face": "both"}], constraint_type="dirichlet"
)

# Set transient heat transfer problem
problem = heat_transfer_problem(material, patch, boundary)
problem._tolerance_nonlinear = 1e-10

# Create external force
time_list = np.linspace(0, 1, 21)
external_heat_source = np.zeros((patch.nbctrlpts_total, len(time_list)))
for i, t in enumerate(time_list):
    external_heat_source[:, i] = problem.assemble_volumetric_force(
        power_density, args={"time": t}
    )

# Solve problem
temperature = np.zeros_like(external_heat_source)
# temperature[-1, 1:] = 1.0
problem.solve_heat_transfer(
    temperature, external_heat_source, time_list=time_list, alpha=1.0
)

# Post-processing
from src.lib_tensor_maths import bspline_operations

fig, ax = plt.subplots(figsize=(8, 4))
knots_interp = np.linspace(0, 1, 101)
temperature_interp = bspline_operations.interpolate_meshgrid(
    quadrule_list=patch.quadrule_list,
    knots_list=[knots_interp],
    u_ctrlpts=temperature.T,
)
POSITION, TIME = np.meshgrid(knots_interp * length, time_list)
im = ax.contourf(POSITION, TIME, temperature_interp, 20, cmap="viridis")
cbar = plt.colorbar(im)
cbar.set_label("Temperature (°C)")

ax.grid(False)
ax.set_ylabel("Time (s)")
ax.set_xlabel("Position (m)")
fig.tight_layout()
fig.savefig(f"{RESULT_FOLDER}transient_heat")

#### SOLVE USING BDF
norder = 1
# Solve problem
temperature1 = np.zeros_like(external_heat_source)
# temperature1[-1, 1:] = 1.0 # Add if we impose a constant temperature
problem.solve_heat_transfer(
    temperature1,
    external_heat_source,
    tspan=(time_list[0], time_list[-1]),
    nsteps=len(time_list) - 1,
    norder=norder,
    type_solver="bdf",
)

# Post-processing
from src.lib_tensor_maths import bspline_operations

fig, ax = plt.subplots(figsize=(8, 4))
temperature_interp = bspline_operations.interpolate_meshgrid(
    quadrule_list=patch.quadrule_list,
    knots_list=[knots_interp],
    u_ctrlpts=temperature1.T,
)
POSITION, TIME = np.meshgrid(knots_interp * length, time_list)
im = ax.contourf(POSITION, TIME, temperature_interp, 20, cmap="viridis")
cbar = plt.colorbar(im)
cbar.set_label("Temperature (°C)")

ax.grid(False)
ax.set_ylabel("Time (s)")
ax.set_xlabel("Position (m)")
fig.tight_layout()
fig.savefig(f"{RESULT_FOLDER}transient_heat_bdf_{norder}")

print("Difference between BDF and direct method:")
print("error:", np.linalg.norm(temperature - temperature1))
