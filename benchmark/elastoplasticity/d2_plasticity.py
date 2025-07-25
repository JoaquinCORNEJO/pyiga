from src.__init__ import *
from src.lib_material import J2plasticity
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_mechanical import mechanical_problem

# Set global variables
TRACTION = 400.0
YOUNG, POISSON = 2500, 0.25
NBSTEPS = 101


def surface_force(args: dict):
    position = args["position"]
    x = position[0, :]
    nnz = np.size(position, axis=1)
    tmp = np.zeros((2, nnz))
    tmp[1, :] = x**2 - 1 / 4
    force = np.zeros((2, nnz))
    force[1, :] = -TRACTION * (np.min(tmp, axis=0)) ** 2
    return force


# Create geometry
degree, nbel = 2, 8
material = J2plasticity(
    {
        "elastic_modulus": YOUNG,
        "elastic_limit": 1,
        "poisson_ratio": POISSON,
        "iso_hardening": {"name": "linear", "Eiso": 0.0},
        "kine_hardening": {"parameters": np.array([[500, 0]])},
    }
)
geometry = mygeomdl({"name": "SQ", "degree": degree, "nbel": nbel}).export_geometry()
patch = singlepatch(geometry, quad_args={"quadrule": "gs"})

# Set Dirichlet boundaries
boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=2)
boundary.add_constraint(
    location_list=[
        {"direction": "x", "face": "left"},
        {"direction": "y", "face": "bottom"},
    ],
    constraint_type="dirichlet",
)

# Solve elastic problem
problem = mechanical_problem(material, patch, boundary)
time_list = np.linspace(0, 1.0, NBSTEPS)
force_ref = problem.assemble_surface_force(
    surface_force, location={"direction": "y", "face": "top"}
)
external_force = np.tensordot(force_ref, time_list / time_list[-1], axes=0)
displacement = np.zeros_like(external_force)
problem.solve_elastoplasticity(displacement, external_force)
