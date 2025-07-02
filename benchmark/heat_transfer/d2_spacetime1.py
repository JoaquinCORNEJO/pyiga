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
DEGREE, NBEL = 8, 32


def exact_temperature(args: dict):
    t_list = args["time"]
    position = args["position"]
    x = position[0, :]
    y = position[1, :]
    u = np.zeros((len(x), len(t_list)))
    for i, t in enumerate(t_list):
        u[:, i] = (
            CST
            * tanh(pi * (1.0 - x**2 - y**2))
            * sin(pi * (x**2 + y**2 - 0.25**2))
            * sin(2 * pi * x * y)
            * sin(pi * t / 2)
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
            pi*CST*(-24.0*pi*x**2*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*sin(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0)) - 16.0*pi*x**2*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*sin(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0))/cosh(pi*(x**2 + y**2 - 1.0))**2 + 8.0*pi*x**2*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*cos(2*pi*x*y)/cosh(pi*(x**2 + y**2 - 1.0))**2 + 16.0*pi*x**2*sin(pi*t/2)*sin(2*pi*x*y)*cos(pi*(x**2 + y**2 - 0.0625))/cosh(pi*(x**2 + y**2 - 1.0))**2 + 8.0*pi*x**2*sin(pi*t/2)*cos(pi*(x**2 + y**2 - 0.0625))*cos(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0)) - 16.0*pi*x*y*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*sin(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0)) - 16.0*pi*x*y*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*sin(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0))/cosh(pi*(x**2 + y**2 - 1.0))**2 + 48.0*pi*x*y*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*cos(2*pi*x*y)/cosh(pi*(x**2 + y**2 - 1.0))**2 + 16.0*pi*x*y*sin(pi*t/2)*sin(2*pi*x*y)*cos(pi*(x**2 + y**2 - 0.0625))/cosh(pi*(x**2 + y**2 - 1.0))**2 + 48.0*pi*x*y*sin(pi*t/2)*cos(pi*(x**2 + y**2 - 0.0625))*cos(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0)) - 24.0*pi*y**2*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*sin(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0)) - 32.0*pi*y**2*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*sin(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0))/cosh(pi*(x**2 + y**2 - 1.0))**2 + 8.0*pi*y**2*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*cos(2*pi*x*y)/cosh(pi*(x**2 + y**2 - 1.0))**2 + 32.0*pi*y**2*sin(pi*t/2)*sin(2*pi*x*y)*cos(pi*(x**2 + y**2 - 0.0625))/cosh(pi*(x**2 + y**2 - 1.0))**2 + 8.0*pi*y**2*sin(pi*t/2)*cos(pi*(x**2 + y**2 - 0.0625))*cos(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0)) + 12.0*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*sin(2*pi*x*y)/cosh(pi*(x**2 + y**2 - 1.0))**2 + 4.0*sin(pi*t/2)*sin(pi*(x**2 + y**2 - 0.0625))*cos(2*pi*x*y)*tanh(pi*(x**2 + y**2 - 1.0)) + 12.0*sin(pi*t/2)*sin(2*pi*x*y)*cos(pi*(x**2 + y**2 - 0.0625))*tanh(pi*(x**2 + y**2 - 1.0)) - sin(pi*(x**2 + y**2 - 0.0625))*sin(2*pi*x*y)*cos(pi*t/2)*tanh(pi*(x**2 + y**2 - 1.0)))/2
        )
    return np.ravel(f, order="F")


# Create model
quad_args = {"quadrule": "wq", "type": 2}
geometry = mygeomdl({"name": "QA", "degree": DEGREE, "nbel": NBEL}).export_geometry()
space_patch = singlepatch(geometry, quad_args=quad_args)

# Create time span
NBEL_TIME = 64
quad_args = {"quadrule": "gs", "type": "lob"}
time_interval = mygeomdl(
    {"name": "line", "degree": 1, "nbel": NBEL_TIME}
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
)[0]
print(f"Relative error is {error:.3e} in {finish-start:.2e}s")
