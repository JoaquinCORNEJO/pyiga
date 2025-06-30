from thesis.SpaceTime.__init__ import *
from thesis.SpaceTime.input_data import *
import time

# Set global variables
PLOTRELATIVE = True
RUNSIMU = True
FIG_CASE = 2

if FIG_CASE == 1:
    pass

    if RUNSIMU:
        degree_list = np.arange(1, 6)
        cut_list = np.arange(1, 7)
        for quadrule, quadtype in zip(["gs", "wq", "wq"], ["leg", 2, 1]):
            sufix = f"_{quadrule}_{str(quadtype)}"
            quad_args = {"quadrule": quadrule, "type": quadtype}
            abserror_list = np.zeros((len(degree_list) + 1, len(cut_list) + 1))
            abserror_list[0, 1:] = cut_list
            abserror_list[1:, 0] = degree_list
            relerror_list = np.copy(abserror_list)
            filenameA1 = f"{FOLDER2DATA}1l2abserror{sufix}.dat"
            filenameR1 = f"{FOLDER2DATA}1l2relerror{sufix}.dat"
            for j, cuts in enumerate(cut_list):
                for i, degree in enumerate(degree_list):
                    problem_spt, time_spt, temp_spt = simulate_spacetime(
                        degree,
                        cuts,
                        powerDensityRing_spt,
                        quad_args=quad_args,
                        degree_time=degree,
                        nbel_time=2**cuts,
                    )
                    abserror_list[i + 1, j + 1], relerror_list[i + 1, j + 1] = (
                        problem_spt.norm_of_error(
                            temp_spt,
                            norm_args={
                                "type": "L2",
                                "exact_function": exactTemperatureRing_spt,
                            },
                        )
                    )
                    np.savetxt(filenameA1, abserror_list)
                    np.savetxt(filenameR1, relerror_list)

    from mpltools import annotation

    plotoptions = [CONFIGLINE_IGA, CONFIGLINE_WQ]
    figname = f"{FOLDER2RESU}1_L2Convergence.pdf"
    if PLOTRELATIVE:
        filenames = ["1l2relerror_gs_leg", "1l2relerror_wq_1"]
    else:
        filenames = ["1l2abserror_gs_leg", "1l2abserror_wq_1"]

    fig, ax = plt.subplots(figsize=(5, 5))
    for filename, plotopts in zip(filenames, plotoptions):
        quadrule = filename.split("_")[1]
        table = np.loadtxt(f"{FOLDER2DATA}{filename}.dat")
        nbel_list = 2 ** (table[0, 1:])
        degree_list = table[1:, 0]
        error_list = table[1:, 1:]
        for i, degree in enumerate(degree_list):
            color = COLORLIST[i]
            if quadrule == "gs":
                ax.loglog(
                    nbel_list,
                    error_list[i, :],
                    # label=rf"ST-IGA-GL $p={int(degree)}$",
                    color=color,
                    **plotopts,
                )
                slope = np.polyfit(
                    np.log10(nbel_list[2:]), np.log10(error_list[i, 2:]), 1
                )[0]
                slope = int(round(slope, 1))
                annotation.slope_marker(
                    (nbel_list[-2], error_list[i, -2]),
                    slope,
                    poly_kwargs={"facecolor": (0.73, 0.8, 1)},
                    ax=ax,
                )

            else:
                if degree > 1:
                    ax.loglog(nbel_list, error_list[i, :], color=color, **plotopts)
            fig.savefig(figname)
    ax.loglog([], [], color="k", **CONFIGLINE_IGA, label="ST Gauss quadrature")
    ax.loglog([], [], color="k", **CONFIGLINE_WQ, label="ST Weighted quadrature")

    if PLOTRELATIVE:
        ax.set_ylabel(r"Relative $L^2(\Pi)$ error")
        ax.set_ylim(top=1e0, bottom=1e-12)
    else:
        ax.set_ylabel(r"$L^2(\Pi)$ error")
        ax.set_ylim(top=1e1, bottom=1e-8)

    ax.set_xlabel("Number of elements by space-time direction")
    ax.set_xlim(left=1, right=100)
    ax.legend(title=r"Degree $p=1,\ldots,5$", loc="lower left")
    fig.tight_layout()
    fig.savefig(figname)

elif FIG_CASE == 2:

    pass

    degree, cuts = 8, 6
    nbel_time_list = np.array([2**cuts for cuts in range(2, 8)], dtype=int)
    degree_time_list = np.arange(1, 5)
    #
    abserror_inc1 = np.ones_like(nbel_time_list, dtype=float)
    relerror_inc1 = np.ones_like(abserror_inc1)
    abserror_inc2 = np.ones_like(abserror_inc1)
    relerror_inc2 = np.ones_like(abserror_inc1)
    time_inc_list = np.ones_like(abserror_inc1)
    #
    abserror_spt1 = np.ones((len(degree_time_list), len(nbel_time_list)))
    relerror_spt1 = np.ones_like(abserror_spt1)
    abserror_spt2 = np.ones_like(abserror_spt1)
    relerror_spt2 = np.ones_like(abserror_spt1)
    time_spt_list = np.ones_like(abserror_spt1)

    if RUNSIMU:

        for quadrule, quadtype in zip(["wq", "wq", "gs"], [1, 2, "leg"]):
                label_tol = "exact"
                auto_inner_tolerance = False
            # for label_tol, auto_inner_tolerance in zip(
            #     ["exact", "inexact"], [False, True]
            # ):

                quad_args = {"quadrule": quadrule, "type": quadtype}
                sufix = f"_{quadrule}_{quadtype}"
                for i, nbel_time in enumerate(nbel_time_list):

                    problem_spt_inc = simulate_spacetime(
                        degree,
                        cuts,
                        powerDensityRing_spt,
                        degree_time=1,
                        nbel_time=nbel_time,
                        quad_args={"quadrule": "gs"},
                        solve_system=False,
                    )[0]

                    start = time.process_time()
                    problem_inc, time_inc, temp_inc = simulate_incremental(
                        degree,
                        cuts,
                        powerDensityRing_inc,
                        nbel_time=nbel_time,
                        quad_args=quad_args,
                    )
                    finish = time.process_time()

                    time_inc_list[i] = finish - start

                    abserror_inc1[i], relerror_inc1[i] = problem_spt_inc.norm_of_error(
                        np.ravel(temp_inc, order="F"),
                        norm_args={
                            "type": "L2",
                            "exact_function": exactTemperatureRing_spt,
                        },
                    )

                    abserror_inc2[i], relerror_inc2[i] = problem_inc.norm_of_error(
                        temp_inc[:, -1],
                        norm_args={
                            "type": "L2",
                            "exact_function": exactTemperatureRing_inc,
                            "exact_args": {"time": time_inc[-1]},
                        },
                    )

                    np.savetxt(
                        f"{FOLDER2DATA}2abserrorstag_inc1{sufix}.dat", abserror_inc1
                    )
                    np.savetxt(
                        f"{FOLDER2DATA}2relerrorstag_inc1{sufix}.dat", relerror_inc1
                    )
                    np.savetxt(
                        f"{FOLDER2DATA}2abserrorstag_inc2{sufix}.dat", abserror_inc2
                    )
                    np.savetxt(
                        f"{FOLDER2DATA}2relerrorstag_inc2{sufix}.dat", relerror_inc2
                    )
                    np.savetxt(f"{FOLDER2DATA}2timestag_inc1{sufix}.dat", time_inc_list)

                    for j, degree_spt in enumerate(degree_time_list):

                        start = time.process_time()
                        problem_spt, time_spt, temp_spt = simulate_spacetime(
                            degree,
                            cuts,
                            powerDensityRing_spt,
                            degree_time=degree_spt,
                            nbel_time=nbel_time,
                            quad_args=quad_args,
                            auto_inner_tolerance=auto_inner_tolerance,
                        )
                        finish = time.process_time()
                        time_spt_list[j, i] = finish - start

                        abserror_spt1[j, i], relerror_spt1[j, i] = (
                            problem_spt.norm_of_error(
                                temp_spt,
                                norm_args={
                                    "type": "L2",
                                    "exact_function": exactTemperatureRing_spt,
                                },
                            )
                        )

                        abserror_spt2[j, i], relerror_spt2[j, i] = (
                            problem_inc.norm_of_error(
                                np.reshape(
                                    temp_spt,
                                    order="F",
                                    newshape=(
                                        problem_spt.part.nbctrlpts_total,
                                        time_spt.nbctrlpts_total,
                                    ),
                                )[:, -1],
                                norm_args={
                                    "type": "L2",
                                    "exact_function": exactTemperatureRing_inc,
                                    "exact_args": {"time": time_inc[-1]},
                                },
                            )
                        )

                        np.savetxt(
                            f"{FOLDER2DATA}2abserrorstag_spt1{sufix}_{label_tol}.dat",
                            abserror_spt1,
                        )
                        np.savetxt(
                            f"{FOLDER2DATA}2relerrorstag_spt1{sufix}_{label_tol}.dat",
                            relerror_spt1,
                        )
                        np.savetxt(
                            f"{FOLDER2DATA}2abserrorstag_spt2{sufix}_{label_tol}.dat",
                            abserror_spt2,
                        )
                        np.savetxt(
                            f"{FOLDER2DATA}2relerrorstag_spt2{sufix}_{label_tol}.dat",
                            relerror_spt2,
                        )
                        np.savetxt(
                            f"{FOLDER2DATA}2timestag_spt1{sufix}_{label_tol}.dat",
                            time_spt_list,
                        )

    quadrule, quadtype, label_tol = "wq", 1, "exact"
    sufix = f"_{quadrule}_{quadtype}"

    from mpltools import annotation

    fig, ax = plt.subplots(figsize=(5, 5))

    if PLOTRELATIVE:
        error_list = np.loadtxt(
            f"{FOLDER2DATA}2relerrorstag_spt1{sufix}_{label_tol}.dat"
        )
    else:
        error_list = np.loadtxt(
            f"{FOLDER2DATA}2abserrorstag_spt1{sufix}_{label_tol}.dat"
        )
    for i, deg in enumerate(degree_time_list):
        nbctrlpts_list = nbel_time_list + deg
        if deg == 1:
            ax.loglog(
                nbctrlpts_list,
                error_list[i, :],
                color=COLORLIST[i],
                **CONFIGLINE_IGA,
                label=rf"ST-MF-GL $p_t={int(deg)}$",
            )
        else:
            ax.loglog(
                nbctrlpts_list,
                error_list[i, :],
                color=COLORLIST[i],
                **CONFIGLINE_WQ,
                label=rf"ST-MF-WQ $p_t={int(deg)}$",
            )
        slope = np.polyfit(
            np.log10(nbctrlpts_list[2:]), np.log10(error_list[i, 2:]), 1
        )[0]
        slope = round(slope, 1)
        if deg != 2:
            annotation.slope_marker(
                (nbctrlpts_list[-2], error_list[i, -2]),
                slope,
                poly_kwargs={"facecolor": (0.73, 0.8, 1)},
                ax=ax,
            )
        else:
            annotation.slope_marker(
                (nbctrlpts_list[-1], error_list[i, -1]),
                slope,
                poly_kwargs={"facecolor": (0.73, 0.8, 1)},
                ax=ax,
                invert=True,
            )

    if PLOTRELATIVE:
        error_list = np.loadtxt(f"{FOLDER2DATA}2relerrorstag_inc1{sufix}.dat")
    else:
        error_list = np.loadtxt(f"{FOLDER2DATA}2abserrorstag_inc1{sufix}.dat")

    nbctrlpts_list = nbel_time_list + 1
    ax.loglog(
        nbctrlpts_list, error_list, color="k", **CONFIGLINE_INC, label="Crank-Nicolson"
    )
    slope = np.polyfit(np.log10(nbctrlpts_list[:]), np.log10(error_list[:]), 1)[0]
    slope = round(slope, 1)
    annotation.slope_marker(
        (nbctrlpts_list[-2], error_list[-2]),
        slope,
        poly_kwargs={"facecolor": (0.73, 0.8, 1)},
        ax=ax,
    )

    if PLOTRELATIVE:
        ax.set_ylabel(r"Relative $L^2(\Pi)$ error")
        ax.set_ylim(top=1e0, bottom=1e-10)
    else:
        ax.set_ylabel(r"$L^2(\Pi)$ error")
        ax.set_ylim(top=1e1, bottom=1e-8)

    ax.set_xlabel("Number of control points in time \n(or number of time-steps)")
    ax.set_xlim(left=2, right=100)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FOLDER2RESU}2_StagnationError.pdf")

    ###################

    fig, ax = plt.subplots(figsize=(5, 5))

    time_list = np.loadtxt(f"{FOLDER2DATA}2timestag_spt1{sufix}_{label_tol}.dat")
    for i, deg in enumerate(degree_time_list):
        nbctrlpts_list = nbel_time_list + deg
        if deg == 1:
            ax.loglog(
                nbctrlpts_list,
                time_list[i, :],
                color=COLORLIST[i],
                **CONFIGLINE_IGA,
                label=rf"ST-MF-GL $p_t={int(deg)}$",
            )
        else:
            ax.loglog(
                nbctrlpts_list,
                time_list[i, :],
                color=COLORLIST[i],
                **CONFIGLINE_WQ,
                label=rf"ST-MF-WQ $p_t={int(deg)}$",
            )
        slope = np.polyfit(np.log10(nbctrlpts_list[3:]), np.log10(time_list[i, 3:]), 1)[
            0
        ]
        slope = round(slope, 1)
        if deg == 1:
            annotation.slope_marker(
                (nbctrlpts_list[-2], time_list[i, -2]),
                slope,
                poly_kwargs={"facecolor": (0.73, 0.8, 1)},
                ax=ax,
            )

    time_list = np.loadtxt(f"{FOLDER2DATA}2timestag_inc1{sufix}.dat")

    nbctrlpts_list = nbel_time_list + 1
    ax.loglog(
        nbctrlpts_list, time_list, color="k", **CONFIGLINE_INC, label="Crank-Nicolson"
    )
    slope = np.polyfit(np.log10(nbctrlpts_list[:]), np.log10(time_list[:]), 1)[0]
    slope = round(slope, 1)
    annotation.slope_marker(
        (nbctrlpts_list[2], time_list[2]),
        slope,
        invert=True,
        poly_kwargs={"facecolor": (0.73, 0.8, 1)},
        ax=ax,
    )

    ax.set_ylabel("CPU time (s)")
    ax.set_ylim(bottom=1e0, top=1e3)

    ax.set_xlabel("Number of control points in time \n(or number of time-steps)")
    ax.set_xlim(left=2, right=100)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FOLDER2RESU}2_StagnationTime.pdf")

elif FIG_CASE == 3:
    pass

    degree, cuts = 10, 6
    subfolderfolder = f"{FOLDER2DATA}{degree}_{cuts}/"
    if not os.path.isdir(subfolderfolder):
        os.mkdir(subfolderfolder)

    if RUNSIMU:
        for [i, is_adaptive], prefix1 in zip(
            enumerate([False, True]), ["exact", "inexact"]
        ):
            for [j, isnewton], prefix2 in zip(
                enumerate([True, False]), ["newton", "picard"]
            ):
                prefix = f"{prefix1}_{prefix2}_"
                start = time.process_time()
                problem_spt = simulate_spacetime(
                    degree,
                    cuts,
                    powerDensityRing_spt,
                    degree_time=degree,
                    nbel_time=2**cuts,
                    auto_inner_tolerance=is_adaptive,
                    use_newton=isnewton,
                )[0]
                stop = time.process_time()
                print("Method %s %s %.3f" % (prefix1, prefix2, stop - start))

                linres_list = problem_spt._linear_residual_list
                counter_list = np.cumsum([0] + [len(_) for _ in linres_list])
                nonlinres_list = problem_spt._nonlinear_residual_list
                nonlintime_list = problem_spt._nonlinear_time_list

                np.savetxt(f"{subfolderfolder}{prefix}linear_iters.dat", counter_list)
                np.savetxt(
                    f"{subfolderfolder}{prefix}nonlinear_res.dat", nonlinres_list
                )
                np.savetxt(
                    f"{subfolderfolder}{prefix}nonlinear_time.dat", nonlintime_list
                )

    fig0, ax0 = plt.subplots(figsize=(4.5, 4.5))
    fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    figs = [fig0, fig1, fig2]
    axs = [ax0, ax1, ax2]
    linestyle_list = ["-", "--", "-", "--"]
    marker_list = ["o", "o", "s", "s"]

    from matplotlib.ticker import MultipleLocator

    for [i, is_adaptive], prefix1 in zip(
        enumerate([True, False]), ["inexact", "exact"]
    ):
        for [j, isnewton], prefix2 in zip(
            enumerate([True, False]), ["newton", "picard"]
        ):
            l = j + i * 2
            label = prefix1.capitalize() + " " + prefix2.capitalize()
            prefix = f"{prefix1}_{prefix2}_"
            nb_linear_iterations = np.loadtxt(
                f"{subfolderfolder}{prefix}linear_iters.dat"
            )
            nonlinear_time = np.loadtxt(f"{subfolderfolder}{prefix}nonlinear_time.dat")
            nonlinear_residual = np.loadtxt(
                f"{subfolderfolder}{prefix}nonlinear_res.dat"
            )
            nonlinear_residual = nonlinear_residual / nonlinear_residual[0]

            for figcase, fig, ax in zip([0, 1, 2], figs, axs):
                ylim = 1e-12

                if figcase == 0:
                    yy = nonlinear_residual
                    xx = nonlinear_time[: len(nonlinear_residual)]
                    xlim = 150
                    ylabel = "Relative nonlinear residue"
                    xlabel = "CPU time (s)"
                    locatornum = 30

                elif figcase == 1:
                    yy = nonlinear_residual
                    xx = nb_linear_iterations[: len(nonlinear_residual)]
                    xlim = 250
                    ylabel = "Relative nonlinear residue"
                    xlabel = "Number of matrix-vector products"
                    locatornum = 50

                elif figcase == 2:
                    yy = nonlinear_residual
                    xx = np.arange(0, len(nonlinear_residual))
                    xlim = 20
                    ylabel = "Relative nonlinear residue"
                    xlabel = "Number of nonlinear iterations"
                    locatornum = 4

                ax.semilogy(
                    xx,
                    yy,
                    label=label,
                    marker=marker_list[l],
                    linestyle=linestyle_list[l],
                )

                ax.set_xlim(right=xlim, left=0)
                ax.set_ylim(top=1e0, bottom=ylim)
                ax.xaxis.set_major_locator(MultipleLocator(locatornum))
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if figcase == 2:
                    ax.legend()
                fig.tight_layout()
                fig.savefig(
                    f"{FOLDER2RESU}3_NLConvergence_iters_{degree}_{cuts}_{figcase}.pdf"
                )

elif FIG_CASE == 4:

    degree_list = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    cuts_list = np.arange(2, 7)

    filenameA3 = f"{FOLDER2DATA}4sptheatAbs"
    filenameR3 = f"{FOLDER2DATA}4sptheatRel"
    filenameT3 = f"{FOLDER2DATA}4sptheatTim"

    if RUNSIMU:

        A3errorList = np.ones((len(degree_list), len(cuts_list)))
        R3errorList = np.ones((len(degree_list), len(cuts_list)))
        T3timeList = np.ones((len(degree_list), len(cuts_list)))

        for label_tol, auto_inner_tolerance in zip(["exact", "inexact"], [False, True]):
            for quadrule, quadtype in zip(["wq", "wq"], [1, 2]):
                quad_args = {"quadrule": quadrule, "type": quadtype}
                sufix = f"_{quadrule}_{quadtype}_{label_tol}.dat"
                for j, cuts in enumerate(cuts_list):
                    for i, degree in enumerate(degree_list):

                        start = time.process_time()
                        problem_spt, time_spt, temp_spt = simulate_spacetime(
                            degree,
                            cuts,
                            powerDensityRing_spt,
                            degree_time=degree,
                            nbel_time=2**cuts,
                            quad_args=quad_args,
                            auto_inner_tolerance=auto_inner_tolerance,
                        )

                        end = time.process_time()
                        T3timeList[i, j] = end - start

                        A3errorList[i, j], R3errorList[i, j] = (
                            problem_spt.norm_of_error(
                                temp_spt,
                                norm_args={
                                    "type": "L2",
                                    "exact_function": exactTemperatureRing_spt,
                                },
                            )
                        )

                        np.savetxt(f"{filenameA3}{sufix}", A3errorList)
                        np.savetxt(f"{filenameR3}{sufix}", R3errorList)
                        np.savetxt(f"{filenameT3}{sufix}", T3timeList)

    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    cmap = plt.get_cmap("RdYlGn", 8)
    label_tol = "inexact"

    for quadrule, quadtype, plotopts, ax in zip(["wq"], [1], [CONFIGLINE_WQ], [ax2]):
        sufix = f"_{quadrule}_{quadtype}"
        error_list = np.loadtxt(f"{filenameR3}{sufix}_{label_tol}.dat")
        time_list = np.loadtxt(f"{filenameT3}{sufix}_{label_tol}.dat")
        for pos in range(1, np.size(error_list, axis=1)):
            im = ax.scatter(
                time_list[: len(degree_list), pos],
                error_list[: len(degree_list), pos],
                c=degree_list,
                cmap=cmap,
                marker=plotopts["marker"],
                s=15 * plotopts["markersize"],
            )

            ax.loglog(
                time_list[: len(degree_list), pos],
                error_list[: len(degree_list), pos],
                color="k",
                marker="",
                linestyle=plotopts["linestyle"],
                alpha=0.3,
            )
            ax.text(
                time_list[-1, pos],
                error_list[-1, pos] / 32,
                str(int(2 ** (pos + 2))) + r"$^3$" + " el.",
                ha="center",
                va="bottom",
            )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Degree")
        tick_locs = 1 + (np.arange(len(degree_list)) + 0.5) * (
            len(degree_list) - 1
        ) / len(degree_list)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(degree_list)

    for fig, ax, sufix in zip([fig2], [ax2], ["WQ"]):
        ax.grid(False)
        ax.set_ylabel(r"Relative $L^2(\Pi)$ error")
        ax.set_ylim(top=1e0, bottom=1e-15)
        ax.set_xlim(left=0.5e-1, right=1e2)
        ax.set_xlabel("CPU time (s)")
        fig.tight_layout()
        fig.savefig(f"{FOLDER2RESU}4_SPTINC_CPUError{sufix}.pdf")

elif FIG_CASE == 5:

    FOLDER2RESU = f"{FOLDER2RESU}/sptm/"
    if not os.path.isdir(FOLDER2RESU):
        os.mkdir(FOLDER2RESU)

    time_list = [
        0,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.6,
        0.65,
        0.7,
        0.8,
        0.85,
        0.9,
        0.925,
        0.95,
        0.975,
        1,
    ]

    degree, cuts = 8, 3
    problem_spt = simulate_spacetime(
        degree,
        cuts,
        powerDensityRing_spt,
        degree_time=degree,
        nbel_time=2**cuts,
        solve_system=False,
    )[0]
    for i, t in enumerate(time_list):
        problem_spt.part.postprocessing_primal(
            fields={"temp": exactTemperatureRing_inc},
            folder=FOLDER2RESU,
            name=f"out_{i}",
            extra_args={"time": t, "temperature": exactTemperatureRing_inc},
        )

    import imageio
    from src.lib_part import vtk2png

    images = []
    for i in range(len(time_list)):
        filepath = vtk2png(
            folder=FOLDER2RESU,
            filename=f"out_{i}",
            fieldname="temp",
            clim=(0, 15),
            cmap="coolwarm",
            title="Temperature",
            camera_position="xy",
        )
        images.append(imageio.imread(filepath))
    images.append(imageio.imread(filepath))
    images.append(imageio.imread(filepath))
    images = images[-1::-1] + images
    imageio.mimsave(f"{FOLDER2RESU}temperature.gif", images)
