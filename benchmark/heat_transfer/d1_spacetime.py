from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import st_heat_transfer_problem
from numpy import sin, cos, tanh, pi, cosh


def conductivity_property(args: dict):
    temperature = args.get("temperature")
    conductivity = np.zeros(shape=(1, 1, *np.shape(temperature)))
    conductivity[0, 0, ...] = 3.0 + 2.0 * np.tanh(temperature / 50)
    return conductivity


def exact_temperature(args: dict):
    t_list = args["time"]
    position = args["position"]
    x = position[0, :]
    u = np.zeros((len(x), len(t_list)))
    for i, t in enumerate(t_list):
        u[:, i] = sin(2 * pi * x) * sin(0.5 * pi * t) * (1.0 + 0.75 * cos(1.5 * pi * t))
    return np.ravel(u, order="F")


def power_density(args: dict):
    t_list = args["time"]
    x = args["position"][0, :]
    f = np.zeros((len(x), len(t_list)))
    for i, t in enumerate(t_list):
        f[:, i] = (
            -pi
            * (
                0.32
                * pi
                * (0.75 * cos(3 * pi * t / 2) + 1.0) ** 2
                * sin(pi * t / 2) ** 2
                * cos(2 * pi * x) ** 2
                / cosh(
                    (0.015 * cos(3 * pi * t / 2) + 0.02)
                    * sin(pi * t / 2)
                    * sin(2 * pi * x)
                )
                ** 2
                - 8
                * pi
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * (
                    2.0
                    * tanh(
                        (0.015 * cos(3 * pi * t / 2) + 0.02)
                        * sin(pi * t / 2)
                        * sin(2 * pi * x)
                    )
                    + 3.0
                )
                * sin(pi * t / 2)
                * sin(2 * pi * x)
                - (0.75 * cos(3 * pi * t / 2) + 1.0) * sin(2 * pi * x) * cos(pi * t / 2)
                + 2.25 * sin(pi * t / 2) * sin(3 * pi * t / 2) * sin(2 * pi * x)
            )
            / 2
        )
    return np.ravel(f, order="F")


# Create geometry
DEGREE, NBEL, LENGTH = 8, 128, 1.0
geometry = mygeomdl(
    {"name": "line", "degree": DEGREE, "nbel": NBEL, "geo_parameters": {"L": LENGTH}}
).export_geometry()
patch = singlepatch(geometry, quad_args={"quadrule": "wq"})

# Create time span
NBEL_TIME = 16
quad_args = {"quadrule": "gs", "type": "leg"}
time_interval = mygeomdl(
    {"name": "line", "degree": 1, "nbel": NBEL_TIME}
).export_geometry()
time_patch = singlepatch(time_interval, quad_args=quad_args)

# Create material
material = heat_transfer_mat()
material.add_conductivity(conductivity_property, is_uniform=False, shape_tensor=1)
material.add_capacity(1.0, is_uniform=True)

# Create boundary condition
boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=1)
boundary.add_constraint(
    location_list=[{"direction": "x", "face": "both"}], constraint_type="dirichlet"
)

# Set transient heat transfer problem
problem = st_heat_transfer_problem(material, patch, time_patch, boundary)
problem._tolerance_nonlinear = 1e-12

# Create external force
external_force = problem.assemble_volumetric_force(power_density)

# Solve problem
temperature = np.zeros_like(external_force)
problem.solve_heat_transfer(temperature, external_force)

# Post processing
error = problem.norm_of_error(
    temperature, norm_args={"type": "L2", "exact_function": exact_temperature}
)[1]
print(f"Relative error is {error:.3e}")
