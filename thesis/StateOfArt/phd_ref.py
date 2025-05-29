from thesis.StateOfArt.__init__ import *
import pandas as pd

EXTENSION = ".pdf"
FIGCASE = 2

if FIGCASE == 0:

    # Set filename
    filename_ref = f"{FOLDER2DATA}weightedquadrature.csv"
    filename_fig = f"{FOLDER2RESU}weightedquadrature{EXTENSION}"
    table_literature = pd.read_csv(filename_ref)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.semilogy(
        table_literature.degree,
        table_literature.standard,
        **CONFIGLINE_IGA,
        label="Element-loop w. \nGauss quadrature",
    )
    ax.semilogy(
        table_literature.degree,
        table_literature.weighted,
        **CONFIGLINE_WQ,
        label="Row-loop w. \nWeighted quadrature",
    )

    for xx, yy, text in zip(
        [table_literature.degree[8], table_literature.degree[8]],
        [table_literature.standard[8], table_literature.weighted[8]],
        ["60 h", "27 s"],
    ):
        ax.text(
            xx,
            2 * yy,
            f"{text}",
            fontsize=12,
            ha="center",
            va="bottom",
            bbox=dict(
                facecolor="white", alpha=0.7, boxstyle="round,pad=0.3", edgecolor="gray"
            ),
        )

    ax.legend()
    ax.set_xlim([0, 11])
    ax.set_ylim([1e0, 1e6])
    ax.set_ylabel("Setup time (s)")
    ax.set_xlabel("Degree")
    fig.tight_layout()
    fig.savefig(filename_fig)

elif FIGCASE == 1:
    fig, ax = plt.subplots(figsize=(5, 5))
    filename_list = ["fem", "iga2", "iga3"]
    label_list = ["FE radial return", f"ST-IGA quadratic", f"ST-IGA cubic"]
    for filename, label in zip(filename_list, label_list):
        table = pd.read_csv(f"{FOLDER2DATA}st_plasticity_{filename}.csv")
        ax.loglog(1 / table.timesize, table.error, label=label)

    ax.set_ylim(top=1e2, bottom=1e-4)
    ax.set_xlim(left=1e1, right=1e3)
    ax.set_ylabel(r"$L^2(\Pi)$ error on stress")
    ax.set_xlabel("Number of control points in time\n(or time steps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FOLDER2RESU}spt_plasticity{EXTENSION}")

elif FIGCASE == 2:

    plotops = CONFIGLINE_WQ
    filename_error = f"{FOLDER2DATA}matrixfree_error.csv"
    filename_time = f"{FOLDER2DATA}matrixfree_time.csv"

    fig, ax = plt.subplots(figsize=(5.5, 4))
    cmap = plt.get_cmap("RdYlGn", 10)

    error_file = pd.read_csv(filename_error)
    time_file = pd.read_csv(filename_time)
    degree_list = error_file.deg

    for pos in range(1, len(error_file.columns)):
        im = ax.scatter(
            time_file.iloc[:, pos],
            error_file.iloc[:, pos],
            c=degree_list,
            cmap=cmap,
            marker=plotops["marker"],
            s=15 * plotops["markersize"],
        )

        ax.loglog(
            time_file.iloc[:, pos],
            error_file.iloc[:, pos],
            color="k",
            marker="",
            linestyle=plotops["linestyle"],
            alpha=0.5,
        )

        if pos > 1:
            ax.text(
                time_file.iloc[-1, pos],
                error_file.iloc[-1, pos] / 40,
                str(int(2 ** (pos + 3))) + r"$^3$" + " el.",
                ha="center",
				va="bottom",
            )
        else:
            ax.text(
                time_file.iloc[int(len(degree_list) / 2), pos],
                error_file.iloc[-1, pos] / 40,
                str(int(2 ** (pos + 3))) + r"$^3$" + " el.",
                ha="center",
				va="bottom",
            )

    cbar = plt.colorbar(im)
    cbar.set_label("Degree")
    tick_locs = 1 + (np.arange(len(degree_list)) + 0.5) * (len(degree_list) - 1) / len(
        degree_list
    )
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(degree_list)

    ax.grid(False)
    ax.set_ylim(top=1e0, bottom=1e-15)
    ax.set_xlim(left=1e-1, right=1e4)
    ax.set_ylabel(r"Relative $H^1(\Omega)$ error")
    ax.set_xlabel("CPU time (s)")
    fig.tight_layout()
    fig.savefig(f"{FOLDER2RESU}matrixfree{EXTENSION}")
