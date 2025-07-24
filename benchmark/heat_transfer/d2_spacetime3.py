from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import st_heat_transfer_problem
from numpy import sin, cos, tanh, pi, cosh
import time

# Global constants
CST = 25
DEGREE, NBEL = 4, 16


def exact_temperature(args: dict):
    time = args["time"]
    position = args["position"]
    x = position[0, :]
    y = position[1, :]
    u = np.zeros((len(x), len(time)))
    for i, t in enumerate(time):
        u[:, i] = (
            CST
            * sin(pi * y)
            * sin(pi * (y + 0.75 * x - 0.5) * (-y + 0.75 * x - 0.5))
            * sin(5 * pi * x)
            * sin(pi / 2 * t)
            * (1 + 0.75 * cos(3 * pi / 2 * t))
        )
    return np.ravel(u, order="F")


def power_density(args: dict):
    position = args["position"]
    time = args["time"]
    x = position[0, :]
    y = position[1, :]
    f = np.zeros((len(x), len(time)))
    for i, t in enumerate(time):
        u1 = pi * (y - (3 * x) / 4 + 1 / 2) * ((3 * x) / 4 + y - 1 / 2)
        u2 = sin((pi * t) / 2)
        u3 = sin(5 * pi * x)
        u4 = (3 * cos((3 * pi * t) / 2)) / 4 + 1
        u5 = pi * (y - (3 * x) / 4 + 1 / 2) + pi * ((3 * x) / 4 + y - 1 / 2)
        u6 = (3 * pi * (y - (3 * x) / 4 + 1 / 2)) / 4 - (
            3 * pi * ((3 * x) / 4 + y - 1 / 2)
        ) / 4
        f[:, i] = (
            (23 * CST * pi * cos(u1) * u2 * u3 * sin(pi * y) * u4) / 4
            - (CST * pi * sin(u1) * cos((pi * t) / 2) * u3 * sin(pi * y) * u4) / 2
            - 4 * CST * sin(u1) * u2 * u3 * sin(pi * y) * (u5) ** 2 * u4
            - 2 * CST * sin(u1) * u2 * u3 * sin(pi * y) * (u6) ** 2 * u4
            + 10 * CST * pi**2 * sin(u1) * cos(5 * pi * x) * cos(pi * y) * u2 * u4
            + (9 * CST * pi * sin(u1) * u2 * sin((3 * pi * t) / 2) * u3 * sin(pi * y))
            / 8
            - 54 * CST * pi**2 * sin(u1) * u2 * u3 * sin(pi * y) * u4
            - 2 * CST * sin(u1) * u2 * u3 * sin(pi * y) * (u5) * (u6) * u4
            + 10 * CST * pi * cos(u1) * cos(5 * pi * x) * u2 * sin(pi * y) * (u5) * u4
            + 8 * CST * pi * cos(u1) * cos(pi * y) * u2 * u3 * (u5) * u4
            + 20 * CST * pi * cos(u1) * cos(5 * pi * x) * u2 * sin(pi * y) * (u6) * u4
            + 2 * CST * pi * cos(u1) * cos(pi * y) * u2 * u3 * (u6) * u4
        )
    return np.ravel(f, order="F")


# Create model
quad_args = {"quadrule": "wq", "type": 2}
geometry = mygeomdl({"name": "tp", "degree": DEGREE, "nbel": NBEL}).export_geometry()
space_patch = singlepatch(geometry, quad_args=quad_args)

# Create time span
NBEL_TIME = 16
quad_args = {"quadrule": "gs", "type": "lob"}
time_interval = mygeomdl(
    {"name": "line", "degree": 1, "nbel": NBEL_TIME}
).export_geometry()
time_patch = singlepatch(time_interval, quad_args=quad_args)

# Add material
material = heat_transfer_mat()
material.add_capacity(1, is_uniform=True)
material.add_conductivity(
    2 * np.array([[1.0, 0.5], [0.5, 2.0]]), is_uniform=True, ndim=2
)

# Block boundaries
boundary = boundary_condition(nbctrlpts=space_patch.nbctrlpts, nb_vars_per_ctrlpt=1)
boundary.add_constraint(
    location_list=[{"direction": "all", "face": "both"}],
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
