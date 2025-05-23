from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import J2plasticity3d
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_mechanical import mechanical_problem
from time import time

# Create subfolder
subfolder = f"{RESULT_FOLDER}/dyn2d/"
if not os.path.isdir(subfolder):
    os.mkdir(subfolder)

# Set global variables
DEGREE, NBEL = 4, 32
TRACTION, RINT, REXT = 1.0, 0.2, 1.0
YOUNG, POISSON, RHO = 2.0e3, 0.25, 7.8e-6
WAVEVEL = np.sqrt(YOUNG / RHO * (1 - POISSON) / ((1 + POISSON) * (1 - 2 * POISSON)))


def regroupe_vtk_files(filename, folder, nbfiles):
    print("Creating group...")
    g = VtkGroup(folder)
    for i in range(0, nbfiles + 1):
        g.addFile(filepath=f"{folder}{filename}_{i}.vts", sim_time=i)
    g.save()
    return


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
geometry = mygeomdl(geo_parameters).export_geometry()
patch = singlepatch(geometry, quad_args={"quadrule": "gs", "type": "leg"})

# Set Dirichlet boundaries
boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nbvars=2)
boundary.add_constraint(
    location_list=[
        {"direction": "y", "face": "top"},
        {"direction": "y", "face": "bottom"},
    ],
    constraint_type="dirichlet",
)

# Set material
material = J2plasticity3d({"elastic_modulus": YOUNG, "poisson_ratio": POISSON})
material.add_density(RHO, is_uniform=True)

# Set mechanical problem
problem = mechanical_problem(material, patch, boundary, allow_lumping=True)
timespan = (REXT - RINT + np.pi / 2 * REXT) / WAVEVEL
freqmax = np.max(problem.solve_eigenvalue_problem(which="LM", k=2)[0])
nbsteps_min = int(1.01 * np.ceil(timespan * np.sqrt(freqmax) / 2))

# Create external force
time_list = np.linspace(0, timespan, nbsteps_min)
force_ref = problem.assemble_surface_force(
    surface_force, location={"direction": "x", "face": "right"}
)
external_force = np.tensordot(force_ref, np.ones_like(time_list), axes=0)
displacement = np.zeros_like(external_force)
acceleration = np.zeros_like(external_force)

# Solve linear dynamics with newmark scheme
start = time()
problem.solve_explicit_linear_dynamics(
    displacement, acceleration, external_force, time_list
)
finish = time()
print("Solve problem in %.2f seconds" % (finish - start))


# Post processing
def export_vtk(disp, name):
    strain_3d = problem.interpolate_strain(disp, convert_to_3d=True)
    stress_3d = material.eval_elastic_stress(strain_3d)
    vmstress = material.eval_von_mises_stress(stress_3d)
    patch.postprocessing_dual(folder=subfolder, fields={"vms": vmstress}, name=name)
    return


export_vtk(displacement[:, :, 1], f"dyn_{0}")
for counter, idx in enumerate(range(2, len(time_list) - 1, 15)):
    export_vtk(displacement[:, :, idx], f"dyn_{counter+1}")
export_vtk(displacement[:, :, -1], f"dyn_{counter+1}")
regroupe_vtk_files("dyn", subfolder, nbfiles=counter + 1)
