from thesis.StateOfArt.__init__ import *
from src.lib_mygeometry import *
from src.lib_part import *
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import (
    heat_transfer_problem,
    st_heat_transfer_problem,
)
from numpy import sin, cos, pi, tanh

CST = 100


def conductivity_property(args: dict):
    temperature = args.get("temperature")
    conductivity = np.zeros(shape=(1, 1, *np.shape(temperature)))
    conductivity[0, 0, ...] = 3.0 + 2.0 * tanh(temperature / 50)
    return conductivity


def exact_temperature(args: dict):
    t = args["time"]
    x = args["position"]
    u = CST * sin(2 * pi * x) * sin(pi / 2 * t) * (1 + 0.75 * cos(3 * pi / 2 * t))
    return u


def power_density_inc(args: dict):
    t = args["time"]
    x = args["position"]
    u = CST * sin(2 * pi * x) * sin(pi / 2 * t) * (1 + 0.75 * cos(3 * pi / 2 * t))
    f = (
        (
            CST
            * pi
            * cos((pi * t) / 2)
            * sin(2 * pi * x)
            * ((3 * cos((3 * pi * t) / 2)) / 4 + 1)
        )
        / 2
        - (9 * CST * pi * sin((pi * t) / 2) * sin((3 * pi * t) / 2) * sin(2 * pi * x))
        / 8
        + 4 * pi**2 * u * (2 * tanh(u / 50) + 3)
        + (
            4
            * CST**2
            * pi**2
            * cos(2 * pi * x) ** 2
            * sin((pi * t) / 2) ** 2
            * ((3 * cos((3 * pi * t) / 2)) / 4 + 1) ** 2
            * (tanh(u / 50) ** 2 - 1)
        )
        / 25
    )
    return f


def power_density_spt(args: dict):
    time = args["time"]
    position = args["position"]
    nc_sp = np.size(position, axis=1)
    nc_tm = np.size(time)
    f = np.zeros((nc_sp, nc_tm))
    for i in range(nc_tm):
        f[:, i] = power_density_inc(args={"time": time[i], "position": position})
    return np.ravel(f, order="F")


def simulate_incremental(degree, cuts, nbel_time):

    # Create geometry
    geometry = mygeomdl(
        {"name": "line", "degree": degree, "nbel": int(2**cuts)}
    ).export_geometry()
    space_patch = singlepatch(geometry, quad_args={"quadrule": "wq", "type": 2})
    time_patch = np.linspace(0.0, 1.0, nbel_time + 1)

    # Add material
    material = heat_transfer_mat()
    material.add_capacity(1.0, is_uniform=True)
    material.add_conductivity(conductivity_property, is_uniform=False, shape_tensor=1)

    # Block boundaries
    boundary = boundary_condition(nbctrlpts=space_patch.nbctrlpts, nb_vars_per_ctrlpt=1)
    boundary.add_constraint(
        location_list=[{"direction": "x", "face": "both"}], constraint_type="dirichlet"
    )

    # Transient model
    problem = heat_transfer_problem(material, space_patch, boundary)
    temperature = np.zeros((problem.part.nbctrlpts_total, len(time_patch)))

    # Create external force
    external_heat_source = np.zeros_like(temperature)
    for i, t in enumerate(time_patch):
        external_heat_source[:, i] = problem.assemble_volumetric_force(
            power_density_inc, args={"time": t}
        )

    # Solve problem
    problem._tolerance_nonlinear = 1e-3
    problem.solve_heat_transfer(
        temperature, external_heat_source, time_patch, alpha=0.5
    )

    return problem, time_patch, temperature


def simulate_spacetime(degree, cuts, nbel_time, auto_inner_tolerance=True):

    # Create geometry
    geometry = mygeomdl(
        {"name": "line", "degree": degree, "nbel": int(2**cuts)}
    ).export_geometry()
    space_patch = singlepatch(geometry, quad_args={"quadrule": "wq", "type": 2})
    time_interval = mygeomdl(
        {"name": "line", "degree": 1, "nbel": nbel_time}
    ).export_geometry()
    time_patch = singlepatch(time_interval, quad_args={"quadrule": "wq", "type": 2})

    # Add material
    material = heat_transfer_mat()
    material.add_capacity(1, is_uniform=True)
    material.add_conductivity(conductivity_property, is_uniform=False, shape_tensor=1)

    # Block boundaries
    boundary = boundary_condition(nbctrlpts=space_patch.nbctrlpts, nb_vars_per_ctrlpt=1)
    boundary.add_constraint(
        location_list=[{"direction": "x", "face": "both"}], constraint_type="dirichlet"
    )

    # Define space time problem
    problem_spt = st_heat_transfer_problem(material, space_patch, time_patch, boundary)

    # Add external force
    external_force = problem_spt.assemble_volumetric_force(power_density_spt)
    temperature = np.zeros_like(external_force)

    problem_spt.solve_heat_transfer(
        temperature,
        external_force,
        use_picard=True,
        auto_inner_tolerance=auto_inner_tolerance,
    )

    return problem_spt, time_patch, temperature


degree = 6
cuts = 3
nbel_time = 6
time_span = np.linspace(0, 1, nbel_time + 1)

spacetime_patch = singlepatch(
    mygeomdl(
        {
            "name": "square",
            "degree": [degree, 1],
            "nbel": [int(2**cuts), nbel_time],
            "geo_parameters": {
                "XY": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]])
            },
        }
    ).export_geometry(),
    quad_args={"quadrule": "gs", "type": "leg"},
)

# problem = simulate_spacetime(degree, cuts, nbel_time)[0]
# solutions = problem._solution_history_list
# for name, temperature in zip(solutions.keys(), solutions.values()):
# 	step = str(name.split('_')[1])
# 	spacetime_patch.postprocessing_primal(fields={'temp':temperature},
# 										name=f'spacetime_{step}',
# 										folder=FOLDER2RESU,
# 										sample_size=101)

# for i in range(10):
# 	filepath = vtk2png(filename=f'spacetime_{i}', folder=FOLDER2RESU, title='Temperature',
# 				camera_position='xy', position_y=0.2, clim=[-100, 100],)


# problem = simulate_incremental(degree, cuts, nbel_time)[0]
# solutions = problem._solution_history_list
# matrix = np.nan*np.ones((problem.part.nbctrlpts_total, len(time_span)))
# matrix[:, 0] = 0.0
# index = 0
# for name, temperature in zip(solutions.keys(), solutions.values()):
# 	step = int(name.split('_')[1])
# 	matrix[:, step] = temperature
# 	spacetime_patch.postprocessing_primal(fields={'temp':np.ravel(matrix, order='F')},
# 										name=f'inctime_{index}',
# 										folder=FOLDER2RESU,
# 										sample_size=101)
# 	index += 1

for i in range(19, 30):
    filepath = vtk2png(
        filename=f"inctime_{i}",
        folder=FOLDER2RESU,
        title="Temperature",
        camera_position="xy",
        position_y=0.2,
        clim=[-100, 100],
    )
