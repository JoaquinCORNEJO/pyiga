from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import J2plasticity3d
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_mechanical import mechanical_problem
from time import time

# Set global variables
DEGREE, NBEL = 4, 4
TRACTION, RINT, REXT = 1.0, 1.0, 2.0
YOUNG, POISSON = 1e3, 0.25


def surface_force(args: dict):
    position = args["position"]
    x = position[0, :]
    y = position[1, :]

    r_square = x**2 + y**2
    b = RINT**2 / r_square
    b2 = b**2

    r = np.sqrt(r_square)
    cos_theta = x / r
    sin_theta = y / r

    cos_3theta = 4 * cos_theta**3 - 3 * cos_theta
    sin_3theta = 3 * sin_theta - 4 * sin_theta**3

    force = np.zeros_like(position)
    force[0, :] = (
        TRACTION
        / 2
        * (2 * cos_theta - b * (2 * cos_theta + 3 * cos_3theta) + 3 * b2 * cos_3theta)
    )
    force[1, :] = TRACTION / 2 * 3 * sin_3theta * (b2 - b)
    return force


# Create geometry
geo_parameters = {
    "name": "QA",
    "degree": DEGREE,
    "nbel": NBEL,
    "geo_parameters": {"Rin": RINT, "Rex": REXT},
}
material = J2plasticity3d(
    {"elastic_modulus": YOUNG, "elastic_limit": 1e10, "poisson_ratio": POISSON}
)
geometry = mygeomdl(geo_parameters).export_geometry()
patch = singlepatch(geometry, quad_args={"quadrule": "gs", "type": "leg"})

# Set Dirichlet boundaries
boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=2)
boundary.add_constraint(
    location_list=[
        {"direction": "y", "face": "top"},
        {"direction": "y", "face": "bottom"},
    ],
    constraint_type="dirichlet",
)

# Solve elastic problem
problem = mechanical_problem(material, patch, boundary)
external_force = problem.assemble_surface_force(
    surface_force, location={"direction": "x", "face": "right"}
)
displacement = np.zeros_like(external_force)
start = time()
problem.solve_elastoplasticity(displacement, external_force)
finish = time()
print("Problem solved in %.2f seconds" % (finish - start))
strain_3d = problem.interpolate_strain(displacement, convert_to_3d=True)
stress_3d = material.eval_elastic_stress(strain_3d)
vmstress = material.eval_von_mises_stress(stress_3d)
problem.part.postprocessing_dual(
    fields={"vms": vmstress}, name="elasticity", folder=RESULT_FOLDER
)
