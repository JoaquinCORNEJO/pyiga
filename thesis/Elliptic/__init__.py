from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat, J2plasticity3d
from src.lib_boundary import boundary_condition, flatten_list
from src.lib_tensor_maths import bspline_operations
from src.single_patch.lib_job import space_problem
from src.single_patch.lib_job_mechanical import mechanical_problem
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem
from src.lib_linear_solver import linsolver

FOLDER2RESU = os.path.dirname(os.path.realpath(__file__)) + "/results/"
FOLDER2DATA = os.path.dirname(os.path.realpath(__file__)) + "/data/"
if not os.path.isdir(FOLDER2RESU):
    os.mkdir(FOLDER2RESU)
if not os.path.isdir(FOLDER2DATA):
    os.mkdir(FOLDER2DATA)

MARKERLIST = ["o", "v", "s", "X", "+", "p", "*"]
COLORLIST = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#C7C7C7",
    "#DBDB8D",
    "#9EDAE5",
]

CONFIGLINE_IGA = {
    "marker": "s",
    "linestyle": "-",
    "markersize": 10,
    "markerfacecolor": "w",
}
CONFIGLINE_WQ = {
    "marker": "o",
    "linestyle": "--",
    "markersize": 6,
    "markerfacecolor": None,
}
CONFIGLINE_INC = {
    "marker": "d",
    "linestyle": "-.",
    "markersize": 6,
    "markerfacecolor": None,
}
CONFIGLINE_BDF = {
    "marker": "+",
    "linestyle": ":",
    "markersize": 6,
    "markerfacecolor": None,
}

# Set global variables
GEONAME = "QA"
TRACTION, RINT, REXT = 1.0, 1.0, 2.0
YOUNG, POISSON = 1e3, 0.25


def surface_force(args: dict):
    position = args["position"]
    x = position[0, :]
    y = position[1, :]

    r_square = x**2 + y**2
    r = np.sqrt(r_square)
    b = RINT**2 / r_square
    b2 = b**2

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


def exact_displacement(args: dict):
    position = args.get("position")
    x = position[0, :]
    y = position[1, :]
    r_square = x**2 + y**2
    r = np.sqrt(r_square)

    b = RINT**2 / r_square
    c = TRACTION * (1.0 + POISSON) * r / (2 * YOUNG)

    cos_theta = x / r
    sin_theta = y / r

    cos_3theta = 4 * cos_theta**3 - 3 * cos_theta
    sin_3theta = 3 * sin_theta - 4 * sin_theta**3

    disp = np.zeros_like(position)
    disp[0, :] = c * (
        2 * (1 - POISSON) * cos_theta
        + b * (4 * (1 - POISSON) * cos_theta + cos_3theta)
        - b**2 * cos_3theta
    )
    disp[1, :] = c * (
        -2 * POISSON * sin_theta
        + b * (2 * (-1 + 2 * POISSON) * sin_theta + sin_3theta)
        - b**2 * sin_3theta
    )

    return disp


def power_density(args: dict):
    position = args.get("position")
    x = position[0, :]
    y = position[1, :]

    f = (
        3
        * np.pi**2
        * np.sin(np.pi * x)
        * np.sin(np.pi * y)
        * (x**2 + y**2 - 1)
        * (x**2 + y**2 - 4)
        - 16 * y**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        - 6 * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        - 6 * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        - 8 * x * y * np.sin(np.pi * x) * np.sin(np.pi * y)
        - np.pi**2
        * np.cos(np.pi * x)
        * np.cos(np.pi * y)
        * (x**2 + y**2 - 1)
        * (x**2 + y**2 - 4)
        - 4 * x * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        - 2 * x * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (x**2 + y**2 - 1)
        - 4 * x * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        - 2 * x * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (x**2 + y**2 - 4)
        - 2 * y * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        - 8 * y * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (x**2 + y**2 - 1)
        - 2 * y * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        - 8 * y * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (x**2 + y**2 - 4)
        - 8 * x**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    )
    return f


def exact_temperature(args: dict):
    position = args.get("position")
    x = position[0, :]
    y = position[1, :]
    u = (
        np.sin(np.pi * x)
        * np.sin(np.pi * y)
        * (x**2 + y**2 - 1.0)
        * (x**2 + y**2 - 4.0)
    )
    return u


def ders_exact_temperature(args: dict):
    position = args.get("position")
    x = position[0, :]
    y = position[1, :]
    uders = np.zeros((1, 2, np.size(position, axis=1)))
    uders[0, 0, :] = (
        2 * x * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        + 2 * x * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        + np.pi
        * np.cos(np.pi * x)
        * np.sin(np.pi * y)
        * (x**2 + y**2 - 1)
        * (x**2 + y**2 - 4)
    )
    uders[0, 1, :] = (
        2 * y * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 1)
        + 2 * y * np.sin(np.pi * x) * np.sin(np.pi * y) * (x**2 + y**2 - 4)
        + np.pi
        * np.cos(np.pi * y)
        * np.sin(np.pi * x)
        * (x**2 + y**2 - 1)
        * (x**2 + y**2 - 4)
    )
    return uders


def solve_dense_system(problem: space_problem, A, b):
    dod = flatten_list(problem.sp_constraint_ctrlpts, problem.part.nbctrlpts_total)
    spilu = scsplin.spilu(A)

    def cleanfun(array_in, dod):
        array_in[dod] = 0.0
        return

    def Afun(array_in, args):
        array_out = A @ array_in
        return array_out

    def Pfun(array_in):
        array_out = spilu.solve(array_in)
        return array_out

    output = linsolver().GMRES(Afun, b, Pfun=Pfun, cleanfun=cleanfun, dod=dod)
    return output["sol"], output["res"]


def buildmatrix_el(problem: mechanical_problem):
    ndim = problem.part.ndim
    elastictensor = problem.material.set_linear_elastic_tensor(
        shape=np.shape(problem.part.det_jac), nbvars=ndim
    )
    submatrices = []
    for i in range(ndim):
        for j in range(ndim):
            pseudo_conductivity = elastictensor[i, j, ...]
            prop = np.einsum(
                "ilk,jmk,lmk,k->ijk",
                problem.part.inv_jac,
                problem.part.inv_jac,
                pseudo_conductivity,
                problem.part.det_jac,
            )
            submatrices.append(
                bspline_operations.assemble_scalar_gradu_gradv(
                    problem.part.quadrule_list, prop
                )
            )
    matrix = sp.bmat(
        [submatrices[i : i + ndim] for i in range(0, len(submatrices), ndim)]
    )
    return matrix


def simulate_el(degree, cuts, quad_args=None, preconditioner="jm", linsolver="GMRES"):
    geo_parameters = {
        "name": GEONAME,
        "degree": degree,
        "nbel": int(2**cuts),
        "geo_parameters": {"Rin": RINT, "Rex": REXT},
    }
    assert preconditioner.lower() in [
        "ilu",
        "fd",
        "jm",
    ], f"Preconditioner {preconditioner} not implemented"
    if quad_args is None:
        quad_args = {"quadrule": "gs", "type": "leg"}

    material = J2plasticity3d({"elastic_modulus": YOUNG, "poisson_ratio": POISSON})
    geometry = mygeomdl(geo_parameters).export_geometry()
    patch = singlepatch(geometry, quad_args=quad_args)

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

    if preconditioner == "ilu":
        stiffnessmatrix = buildmatrix_el(problem)
        displacement, residual = solve_dense_system(
            problem, stiffnessmatrix, np.ravel(external_force)
        )
        problem._linear_residual_list = [residual]

    else:
        problem._tolerance_linear = 1.0e-12
        problem._maxiters_linear = 500
        problem._linear_solver = linsolver
        problem._use_preconditioner = False if preconditioner.lower() == "wp" else True
        problem._update_preconditioner = (
            True if preconditioner.lower() == "jm" else False
        )
        problem.solve_elastoplasticity(displacement, external_force)

    return problem, displacement


def buildmatrix_ht(problem: heat_transfer_problem):
    args = problem._verify_fun_args({})
    prop = np.einsum(
        "ilk,jmk,lmk,k->ijk",
        problem.part.inv_jac,
        problem.part.inv_jac,
        problem.material.conductivity(args),
        problem.part.det_jac,
    )
    matrix = bspline_operations.assemble_scalar_gradu_gradv(
        problem.part.quadrule_list, prop
    )
    return matrix


def simulate_ht(degree, cuts, quad_args=None, preconditioner="fd", linsolver="GMRES"):
    geo_args = {
        "name": GEONAME,
        "degree": degree,
        "nbel": int(2**cuts),
        "geo_parameters": {"Rin": 1.0, "Rex": 2.0},
    }
    assert preconditioner.lower() in [
        "wp",
        "ilu",
        "fd",
        "jm",
    ], f"Preconditioner {preconditioner} not implemented"
    if quad_args is None:
        quad_args = {"quadrule": "gs", "type": "leg"}
    material = heat_transfer_mat()
    material.add_conductivity(
        np.array([[1, 0.5], [0.5, 2]]), is_uniform=True, shape_tensor=2
    )
    geometry = mygeomdl(geo_args).export_geometry()
    patch = singlepatch(geometry, quad_args=quad_args)

    # Set Dirichlet boundaries
    boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=1)
    boundary.add_constraint(
        location_list=[{"direction": "x,y", "face": "both,both"}],
        constraint_type="dirichlet",
    )

    # Solve elastic problem
    problem = heat_transfer_problem(material, patch, boundary)
    external_force = problem.assemble_volumetric_force(power_density)
    temperature = np.zeros_like(external_force)

    if preconditioner.lower() == "ilu":
        conductivitymatrix = buildmatrix_ht(problem)
        temperature, residual = solve_dense_system(
            problem, conductivitymatrix, external_force
        )
        problem._linear_residual_list = [residual]

    else:
        problem._tolerance_linear = 1.0e-12
        problem._maxiters_linear = 100
        problem._linear_solver = linsolver
        problem._use_preconditioner = False if preconditioner.lower() == "wp" else True
        problem._update_preconditioner = (
            True if preconditioner.lower() == "jm" else False
        )
        temperature = problem.solve_heat_transfer(temperature, external_force)

    return problem, temperature
