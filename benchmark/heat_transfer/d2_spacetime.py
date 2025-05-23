from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import st_heat_transfer_problem
from numpy import sin, cos, tanh, pi

# Global constants
CST = 50
DEGREE, NBEL = 3, 8


def exact_temperature(args: dict):
    time = args["time"]
    position = args["position"]
    x = position[0, :]
    y = position[1, :]
    u = np.zeros((len(x), len(time)))
    for i, t in enumerate(time):
        u[:, i] = (
            CST
            * tanh(pi * (1.0 - x**2 - y**2))
            * sin(pi * (x**2 + y**2 - 0.25**2))
            * sin(pi * x * y)
            * sin(0.5 * pi * t)
            * (1.0 + 0.75 * cos(1.5 * pi * t))
        )
    return np.ravel(u, order="F")


def power_density(args: dict):
    position = args["position"]
    time = args["time"]
    x = position[0, :]
    y = position[1, :]
    f = np.zeros((len(x), len(time)))
    for i, t in enumerate(time):
        f[:, i] = (
            8.0
            * pi**2
            * CST
            * x**2
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            + 2.0
            * pi**2
            * CST
            * x**2
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * cos(pi * x * y)
            + 8.0
            * pi**2
            * CST
            * x**2
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * x * y)
            * cos(pi * (x**2 + y**2 - 0.0625))
            + 6.0
            * pi**2
            * CST
            * x**2
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            - 2.0
            * pi**2
            * CST
            * x**2
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * cos(pi * (x**2 + y**2 - 0.0625))
            * cos(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            + 8.0
            * pi**2
            * CST
            * x
            * y
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            + 12.0
            * pi**2
            * CST
            * x
            * y
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * cos(pi * x * y)
            + 8.0
            * pi**2
            * CST
            * x
            * y
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * x * y)
            * cos(pi * (x**2 + y**2 - 0.0625))
            + 5.0
            * pi**2
            * CST
            * x
            * y
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            - 12.0
            * pi**2
            * CST
            * x
            * y
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * cos(pi * (x**2 + y**2 - 0.0625))
            * cos(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            + 16.0
            * pi**2
            * CST
            * y**2
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            + 2.0
            * pi**2
            * CST
            * y**2
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * cos(pi * x * y)
            + 16.0
            * pi**2
            * CST
            * y**2
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * x * y)
            * cos(pi * (x**2 + y**2 - 0.0625))
            + 9.0
            * pi**2
            * CST
            * y**2
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            - 2.0
            * pi**2
            * CST
            * y**2
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * cos(pi * (x**2 + y**2 - 0.0625))
            * cos(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            + 6.0
            * pi
            * CST
            * (1 - tanh(pi * (-(x**2) - y**2 + 1.0)) ** 2)
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            - 1.0
            * pi
            * CST
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * cos(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            - 6.0
            * pi
            * CST
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * t / 2)
            * sin(pi * x * y)
            * cos(pi * (x**2 + y**2 - 0.0625))
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            + 0.5
            * pi
            * CST
            * (0.75 * cos(3 * pi * t / 2) + 1.0)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            * cos(pi * t / 2)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
            - 1.125
            * pi
            * CST
            * sin(pi * t / 2)
            * sin(3 * pi * t / 2)
            * sin(pi * (x**2 + y**2 - 0.0625))
            * sin(pi * x * y)
            * tanh(pi * (-(x**2) - y**2 + 1.0))
        )
    return np.ravel(f, order="F")


# Create model
quad_args = {"quadrule": "gs", "type": "leg"}
geometry = mygeomdl(
    {"name": "QA", "degree": DEGREE, "nbel": NBEL + 1}
).export_geometry()
space_patch = singlepatch(geometry, quad_args=quad_args)

# Create time span
time_interval = mygeomdl(
    {"name": "line", "degree": DEGREE, "nbel": NBEL}
).export_geometry()
time_patch = singlepatch(time_interval, quad_args=quad_args)

# Add material
material = heat_transfer_mat()
material.add_capacity(1, is_uniform=True)
material.add_conductivity(
    np.array([[1.0, 0.5], [0.5, 2.0]]), is_uniform=True, shape_tensor=2
)
material.add_ders_capacity(1, is_uniform=True)
material.add_ders_conductivity(
    np.array([[1.0, 0.5], [0.5, 2.0]]), is_uniform=True, shape_tensor=2
)

# Block boundaries
boundary = boundary_condition(nbctrlpts=space_patch.nbctrlpts, nbvars=1)
boundary.add_constraint(
    location_list=[{"direction": "x,y", "face": "both,both"}],
    constraint_type="dirichlet",
)

# Define space time problem
problem = st_heat_transfer_problem(material, space_patch, time_patch, boundary)

# Add external heat force
external_force = problem.assemble_volumetric_force(power_density)

# Solve space time problem
temperature = np.zeros_like(external_force)
problem.solve_heat_transfer(
    temperature, external_force, use_picard=True, auto_inner_tolerance=False
)

# Post processing
error = problem.norm_of_error(
    temperature, norm_args={"type": "L2", "exact_function": exact_temperature}
)[0]
print(f"Absolute error is {error:.2e}")
temperature_cutted = np.reshape(
    temperature, newshape=(-1, time_patch.nbctrlpts_total), order="F"
)
space_patch.postprocessing_primal(
    fields={"temperature": temperature_cutted[:, -1]},
    name="spacetime",
    folder=RESULT_FOLDER,
)
