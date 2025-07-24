from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import st_heat_transfer_problem
from numpy import sin, cos, tanh, pi, cosh
import time

# Global constants
CST = 50
DEGREE, NBEL = 8, 16


def exact_temperature(args: dict):
    t_list = args["time"]
    position = args["position"]
    x = position[0, :]
    y = position[1, :]
    u = np.zeros((len(x), len(t_list)))
    for i, t in enumerate(t_list):
        u[:, i] = (
            CST
            * sin(pi * x**2)
            * sin(3 * pi * y**2)
            * cos(2 * pi * x * y)
            * tanh(0.5 * pi * t)
            * (1.0 + 0.75 * cos(1.5 * pi * t))
        )
    return np.ravel(u, order="F")


def power_density(args: dict):
    position = args["position"]
    t_list = args["time"]
    x = position[0, :]
    y = position[1, :]
    f = np.zeros((len(x), len(t_list)))
    for i, t in enumerate(t_list):
        f[:, i] = (
            pi
            * CST
            * (
                12.0
                * pi
                * x**2
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(pi * x**2)
                * sin(3 * pi * y**2)
                * cos(2 * pi * x * y)
                * tanh(pi * t / 2)
                + 4.0
                * pi
                * x**2
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(3 * pi * y**2)
                * sin(2 * pi * x * y)
                * cos(pi * x**2)
                * tanh(pi * t / 2)
                + 4.0
                * pi
                * x
                * y
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(pi * x**2)
                * sin(3 * pi * y**2)
                * cos(2 * pi * x * y)
                * tanh(pi * t / 2)
                + 48.0
                * pi
                * x
                * y
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(pi * x**2)
                * sin(2 * pi * x * y)
                * cos(3 * pi * y**2)
                * tanh(pi * t / 2)
                + 8.0
                * pi
                * x
                * y
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(3 * pi * y**2)
                * sin(2 * pi * x * y)
                * cos(pi * x**2)
                * tanh(pi * t / 2)
                - 12.0
                * pi
                * x
                * y
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * cos(pi * x**2)
                * cos(3 * pi * y**2)
                * cos(2 * pi * x * y)
                * tanh(pi * t / 2)
                + 76.0
                * pi
                * y**2
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(pi * x**2)
                * sin(3 * pi * y**2)
                * cos(2 * pi * x * y)
                * tanh(pi * t / 2)
                + 12.0
                * pi
                * y**2
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(pi * x**2)
                * sin(2 * pi * x * y)
                * cos(3 * pi * y**2)
                * tanh(pi * t / 2)
                + 2.0
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(pi * x**2)
                * sin(3 * pi * y**2)
                * sin(2 * pi * x * y)
                * tanh(pi * t / 2)
                - 12.0
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(pi * x**2)
                * cos(3 * pi * y**2)
                * cos(2 * pi * x * y)
                * tanh(pi * t / 2)
                - 2.0
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(3 * pi * y**2)
                * cos(pi * x**2)
                * cos(2 * pi * x * y)
                * tanh(pi * t / 2)
                + 1.0
                * (0.75 * cos(3 * pi * t / 2) + 1.0)
                * sin(pi * x**2)
                * sin(3 * pi * y**2)
                * cos(2 * pi * x * y)
                / (cosh(pi * t) + 1)
                - 1.125
                * sin(3 * pi * t / 2)
                * sin(pi * x**2)
                * sin(3 * pi * y**2)
                * cos(2 * pi * x * y)
                * tanh(pi * t / 2)
            )
        )
    return np.ravel(f, order="F")


# Create model
quad_args = {"quadrule": "wq", "type": 2}
geometry = mygeomdl({"name": "SQ", "degree": DEGREE, "nbel": NBEL}).export_geometry()
space_patch = singlepatch(geometry, quad_args=quad_args)

# Create time span
NBEL_TIME = 16
quad_args = {"quadrule": "gs", "type": "lob"}
time_interval = mygeomdl(
    {"name": "line", "degree": 1, "nbel": NBEL_TIME, "geo_parameters": {"L": 4.0}}
).export_geometry()
time_patch = singlepatch(time_interval, quad_args=quad_args)

# Add material
material = heat_transfer_mat()
material.add_capacity(1, is_uniform=True)
material.add_conductivity(np.array([[1.0, 0.5], [0.5, 2.0]]), is_uniform=True, ndim=2)
material.add_ders_capacity(1, is_uniform=True)
material.add_ders_conductivity(
    np.array([[1.0, 0.5], [0.5, 2.0]]), is_uniform=True, ndim=2
)

# Block boundaries
boundary = boundary_condition(nbctrlpts=space_patch.nbctrlpts, nb_vars_per_ctrlpt=1)
boundary.add_constraint(
    location_list=[{"direction": "x,y", "face": "both,both"}],
    constraint_type="dirichlet",
)

# Define space time problem
problem = st_heat_transfer_problem(material, space_patch, time_patch, boundary)

start = time.time()
# Add external heat force
external_force = problem.assemble_volumetric_force(power_density)

# Solve space time problem
temperature = np.zeros_like(external_force)
problem.solve_heat_transfer(
    temperature, external_force, use_picard=True, auto_inner_tolerance=False
)
finish = time.time()

# Post processing
error = problem.norm_of_error(
    temperature, norm_args={"type": "L2", "exact_function": exact_temperature}
)[1]
print(f"Relative error is {error:.3e} in {finish-start:.2e}s")
