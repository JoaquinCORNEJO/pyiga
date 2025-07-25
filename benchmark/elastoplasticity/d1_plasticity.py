from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import J2plasticity1d
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_mechanical import mechanical_problem


def volume_force(args: dict):
    position = args["position"]
    force = 1e2 * np.ones_like(position)
    return force


# Global variables
DEGREE, NBEL = 2, 16
YOUNG, LENGTH = 1e6, 1
NBSTEPS = 21

# Define geometry
geometry = mygeomdl(
    {"name": "line", "degree": DEGREE, "nbel": NBEL, "geo_parameters": {"L": LENGTH}}
).export_geometry()
patch = singlepatch(geometry, {"quadrule": "wq", "type": 2})

# Define plasticity model
material = J2plasticity1d(
    {
        "elastic_modulus": YOUNG,
        "elastic_limit": 1.0e1,
        "iso_hardening": {"name": "linear", "Eiso": YOUNG / 5},
    }
)


# Set boundary conditions
boundary = boundary_condition(patch.nbctrlpts)
boundary.add_constraint(
    location_list=[{"direction": "x", "face": "left"}], constraint_type="dirichlet"
)

# Define problem
problem = mechanical_problem(material=material, patch=patch, boundary=boundary)

# Add external force
time_list = np.linspace(0, 1, NBSTEPS)
force_ref = problem.assemble_volumetric_force(volume_force)
external_force = np.einsum("i,j->ij", force_ref, time_list)[np.newaxis, :]

# Solve problem
displacement = np.zeros_like(external_force)
problem.solve_elastoplasticity(displacement, external_force)
