import numpy as np
from scipy.interpolate import make_interp_spline
from src.lib_part import singlepatch
from src.lib_mygeometry import mygeomdl
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem


def norm_of_error(problem: heat_transfer_problem, u_ctrlpts, t_nodes, exact_fun):
    """
    Compute the absolute and relative norm of the error between the interpolated solution
    and the exact solution for a heat transfer problem.

    Parameters:
            problem (heat_transfer_problem): The heat transfer problem instance.
            u_ctrlpts (np.ndarray): Control points for the solution.
            t_nodes (np.ndarray): Time nodes for interpolation.
            exact_fun (callable): Function to compute the exact solution.

    Returns:
            tuple: Absolute error and relative error.
    """
    norm_type = "l2"

    # Prepare spatial norm computation
    output = problem._prepare_norm_computation(
        problem.part,
        u_ctrlpts[:, 0],
        knots_ref=[None] * problem.part.ndim,
        norm_type=norm_type,
    )
    quadpts_phy, det_jac_sp, parametric_weights_sp = output[3:]

    # Create a single patch for the time domain
    time_patch = singlepatch(
        mygeomdl(
            {
                "name": "line",
                "degree": 1,
                "nbel": len(t_nodes) - 1,
                "geo_parameters": {"L": t_nodes[-1] - t_nodes[0]},
            }
        ).export_geometry(),
        quad_args={"quadtype": "gs", "default_order": 4},
    )
    parametric_position_tm = time_patch.quadrule_list[0].quadpts
    parametric_weights_tm = time_patch.quadrule_list[0].parametric_weights
    det_jac_tm = time_patch.det_jac

    # Initialize arrays for interpolated and exact solutions
    num_time_points = np.shape(u_ctrlpts)[1]
    num_space_points = len(det_jac_sp)
    num_time_quad_points = time_patch.nbqp_total

    U_interp_space = np.empty((num_space_points, num_time_points))
    U_interp_full = np.empty((num_space_points, num_time_quad_points))
    U_exact_full = np.empty((num_space_points, num_time_quad_points))

    # Compute the exact solution at all quadrature points in time
    for j, t in enumerate(parametric_position_tm):
        args = {"time": t, "position": quadpts_phy}
        U_exact_full[:, j] = exact_fun(args)

    # Interpolate the solution in space
    for j in range(num_time_points):
        output = problem._prepare_norm_computation(
            problem.part,
            u_ctrlpts[:, j],
            knots_ref=[None] * problem.part.ndim,
            norm_type=norm_type,
        )
        U_interp_space[:, j] = output[0]

    # Interpolate the solution in time
    for i in range(num_space_points):
        spline = make_interp_spline(t_nodes, U_interp_space[i, :], k=3)
        U_interp_full[i, :] = spline(parametric_position_tm)

    # Compute the determinant of the Jacobian and parametric weights for the full domain
    complete_det_jac = np.kron(det_jac_tm, det_jac_sp)
    complete_parametric_weights = parametric_weights_sp + [parametric_weights_tm]

    # Flatten the arrays for norm computation
    u_exact = np.ravel(U_exact_full, order="F")
    u_diff = np.ravel(U_interp_full, order="F") - u_exact

    # Compute absolute and relative errors
    abs_error = problem._calculate_norm(
        u_diff, None, complete_det_jac, complete_parametric_weights, norm_type=norm_type
    )
    exact_norm = problem._calculate_norm(
        u_exact,
        None,
        complete_det_jac,
        complete_parametric_weights,
        norm_type=norm_type,
    )
    rel_error = abs_error / exact_norm if exact_norm != 0 else abs_error

    if exact_norm == 0:
        print(
            "Warning: Exact solution norm is zero, relative error is set to absolute error."
        )

    return abs_error, rel_error
