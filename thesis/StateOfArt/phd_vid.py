from thesis.StateOfArt.__init__ import *
from thesis.StateOfArt.phd_fig import plot_2d_geometry
from src.lib_quadrules import *
from src.lib_mygeometry import *
from src.lib_part import singlepatch
from matplotlib.colors import ListedColormap
import imageio

try:
    from pdf2image import convert_from_path
except:
    print("pdf2image module not found. Images should be saved in .png format")

# Set global variables
FOLDER = f"{FOLDER2RESU}/videos/"
VIDCASE = 1
EXTENSION = ".pdf"

if VIDCASE == 0:  # B-spline curve

    def case0(folder, extension, offset=0):
        # Set filename
        filename = f"{folder}BSplinecurve{extension}"

        # Create the curve
        crv = BSpline.Curve()
        crv.degree = 3
        crv.ctrlpts = [
            [-1, 1, 0],
            [-0.5, 0.25, 0],
            [0, 2 + offset, 0],
            [0.5, 1.0, 0.0],
            [0.75, -0.5, 0],
            [1.5, 1, 0],
            [2, 0, 0],
        ]
        crv.knotvector = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0]
        )

        # Get data
        evalpts = np.asarray(crv.evalpts)
        ctrlpts = np.asarray(crv.ctrlpts)
        basis = eval_ders_basis_sparse(
            crv.degree, crv.knotvector, np.unique(crv.knotvector)
        )
        u_knots = basis[0].T @ np.array(crv.ctrlpts)

        fig, ax = plt.subplots()
        ax.plot(evalpts[:, 0], evalpts[:, 1], label="B-Spline curve")
        ax.plot(
            ctrlpts[:, 0],
            ctrlpts[:, 1],
            "o--",
            markersize=10,
            label="Control points net",
        )
        ax.plot(
            u_knots[:, 0],
            u_knots[:, 1],
            color="k",
            marker="s",
            linestyle="",
            label="Knots",
        )
        ax.set_xlim([-1.5, 2.5])
        ax.set_ylim([-1.0, 2.5])
        ax.legend(loc="upper right")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        return filename

    images = []
    for i, offset in enumerate(np.linspace(-2.5, 0.0, 8)):
        filename = case0(FOLDER, f"{i:d}{EXTENSION}", offset=offset)
        if EXTENSION == ".png":
            images.append(imageio.imread(filename))
        elif EXTENSION == ".pdf":
            images.append(convert_from_path(filename, dpi=300)[0])
    images += images[-2::-1]
    imageio.mimsave(f"{FOLDER}BSplinecurve.gif", images)

    def case1(folder, extension):
        # Set filename
        filename = f"{folder}BSplinebasis1D{extension}"

        # B-spline properties
        degree = 3
        knotvector = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0]
        )
        quadrature = quadrature_rule(degree, knotvector)
        knots = np.linspace(0, 1, 201)
        basis = quadrature.get_sample_basis(knots)
        B0 = basis[0].toarray()

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        for i in range(np.shape(B0)[0]):
            ax1.plot(knots, B0[i, :], linewidth=1, color="k")
        ax1.plot([], [], linewidth=1, color="k", label="B-Spline basis")
        ax1.plot(
            quadrature._unique_kv,
            np.zeros(len(quadrature._unique_kv)),
            color="k",
            marker="s",
            linestyle="",
            label="Knots",
        )

        ax1.set_xlabel(r"$\xi$")
        ax1.set_xticks(np.linspace(0, 1, 5))
        ax1.set_yticks([0, 0.5, 1])
        ax1.set_ylabel(r"$\hat{b}_{A,\,p}(\xi)$")
        ax1.legend()
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        return

    case1(FOLDER, EXTENSION)

elif VIDCASE == 1:  # Bivariate functions

    def case2(folder, extension):
        # Set filename
        filename = f"{folder}BSplinebasis2D{extension}"

        # B-Spline properties
        degree1, degree2, nbel = 1, 3, 2
        knotvector1 = create_uniform_knotvector(degree1, nbel)
        knotvector2 = create_uniform_knotvector(degree2, nbel)
        quadrature1 = quadrature_rule(degree1, knotvector1)
        quadrature2 = quadrature_rule(degree2, knotvector2)
        knots = np.linspace(0, 1, 201)
        basis1 = quadrature1.get_sample_basis(knots)
        basis2 = quadrature2.get_sample_basis(knots)

        # B-Spline 2D
        X, Y = np.meshgrid(
            np.unique(quadrature1.knotvector), np.unique(quadrature2.knotvector)
        )
        Xb, Yb = np.meshgrid(knots, knots)
        Zb = np.kron(basis1[0].toarray()[1, :], basis2[0].toarray()[2, :]).reshape(
            (len(knots), len(knots))
        )

        fig, axs = plt.subplots(
            2,
            2,
            sharex="col",
            sharey="row",
            gridspec_kw=dict(height_ratios=[1, 3.2], width_ratios=[3.2, 1]),
            figsize=(5, 5),
        )

        axs[0, 1].set_visible(False)
        axs[0, 0].set_box_aspect(1 / 3)
        axs[1, 0].set_box_aspect(1)
        axs[1, 1].set_box_aspect(3 / 1)
        axs[1, 0].grid(None)
        axs[1, 0].pcolormesh(
            Xb, Yb, Zb.T, cmap="GnBu", shading="gouraud", rasterized=True
        )
        axs[1, 0].plot(X, Y, color="k", marker="s", linestyle="--")
        axs[1, 0].plot(X.T, Y.T, color="k", marker="s", linestyle="--")

        axs[1, 0].set_yticks([0, 0.5, 1])
        axs[1, 0].set_xticks([0, 0.5, 1])

        for i in range(degree1 + nbel):
            axs[0, 0].plot(
                knots,
                np.ravel(basis1[0].toarray()[i, :]),
                linewidth=1,
                color="k",
                alpha=0.8,
            )
        uvk = np.unique(quadrature1.knotvector)
        axs[0, 0].plot(uvk, np.zeros(len(uvk)), color="k", marker="s", linestyle="")

        for i in range(degree2 + nbel):
            axs[1, 1].plot(
                np.ravel(basis2[0].toarray()[i, :]),
                knots,
                linewidth=1,
                color="k",
                alpha=0.8,
            )
        uvk = np.unique(quadrature2.knotvector)
        axs[1, 1].plot(np.zeros(len(uvk)), uvk, color="k", marker="s", linestyle="")

        axs[0, 0].axis(ymin=0, ymax=1)
        axs[0, 0].set_xlabel(r"$\xi_1$")
        axs[0, 0].set_ylabel(r"$\hat{b}_{A_1,\,p_1}$")
        axs[1, 1].axis(xmin=0, xmax=1)
        axs[1, 1].set_ylabel(r"$\xi_2$")
        axs[1, 1].set_xlabel(r"$\hat{b}_{A_2,\,p_2}$")
        axs[1, 1].set_xticks([0, 1])
        axs[0, 0].set_yticks([0, 1])
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        return

    case2(FOLDER, EXTENSION)

    # Set filename
    filename = f"{FOLDER}BSplinesurface"

    # Surface properties
    modelGeo = mygeomdl(
        geo_args={
            "name": "quarter_annulus",
            "degree": np.array([1, 3, 1]),
            "nbel": np.array([2, 2, 1]),
        }
    )
    modelIGA = modelGeo.export_geometry()
    modelPhy = singlepatch(modelIGA, quad_args={"quadrule": "gs"})
    XoriginalPos = modelPhy.ctrlpts[1, 8]
    YoriginalPos = modelPhy.ctrlpts[0, 8]

    images = []
    for i, offset in enumerate(np.linspace(-0.25, 0.1, 8)):
        modelPhy.ctrlpts[0, 8] = XoriginalPos + offset
        modelPhy.ctrlpts[1, 8] = YoriginalPos + offset
        fig = plot_2d_geometry(modelPhy, plotaxis=False)
        newfilename = f"{filename}{i:d}{EXTENSION}"
        fig.savefig(newfilename, dpi=300)
        if EXTENSION == ".png":
            images.append(imageio.imread(newfilename))
        elif EXTENSION == ".pdf":
            images.append(convert_from_path(newfilename, dpi=300)[0])

    images += images[-2::-1]
    imageio.mimsave(FOLDER + "bspline2D.gif", images)

elif VIDCASE == 2:  # Element-wise

    def case3(folder, extension, el=1):
        # Set filename
        filename = f"{folder}BSknin1_{el:d}"

        degree = 2
        nbel, multiplicity = 4, 1
        knotvector = create_uniform_knotvector(degree, nbel, multiplicity=multiplicity)
        quadrature = gauss_quadrature(degree, knotvector, quad_args={})
        quadrature.export_quadrature_rules()
        knots = np.linspace(0, 1, 201)
        basis = quadrature.get_sample_basis(knots)
        B0 = basis[0].toarray()

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
        axs[0].axis("off")
        axs[1].axis("off")

        axs[0].plot(
            quadrature._unique_kv,
            np.zeros(len(quadrature._unique_kv)),
            color="k",
            marker="s",
            linestyle="",
            label="Knots",
        )
        for i in range(np.shape(B0)[0]):
            axs[0].plot(knots, B0[i, :], linewidth=1, color="k")

        ukv = np.unique(knotvector)
        elleft, elright = ukv[el], ukv[el + 1]
        indices = np.where((knots >= elleft) & (knots <= elright))[0]
        for i in range(np.shape(B0)[0]):
            axs[0].plot(knots[indices], B0[i, indices], linewidth=2, color="tab:blue")
            axs[0].fill_between(
                x=knots[indices], y1=B0[i, indices], color="tab:blue", alpha=0.2
            )

        # Matrix
        cmap = ListedColormap(["white", "tab:gray", "tab:blue"])
        A = quadrature.weights[0] @ quadrature.basis[0].T
        A = A.toarray()
        A = np.where(A != 0, 1, 0)
        B = B0[:, indices] @ B0[:, indices].T
        B = np.where(B != 0, 1, 0)
        axs[1].imshow(A + B, cmap=cmap, interpolation=None)

        # Add gridlines for imshow
        filename += extension
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        return filename

    def case4(folder, extension, ctrlpts=1):

        # Set filename
        filename = f"{folder}BSknin2_{ctrlpts:d}"

        degree = 2
        nbel, multiplicity = 4, 1
        knotvector = create_uniform_knotvector(degree, nbel, multiplicity=multiplicity)
        quadrature = gauss_quadrature(degree, knotvector, quad_args={})
        quadrature.export_quadrature_rules()
        knots = np.linspace(0, 1, 201)
        basis = quadrature.get_sample_basis(knots)
        B0 = basis[0].toarray()

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
        axs[0].axis("off")
        axs[1].axis("off")

        axs[0].plot(
            quadrature._unique_kv,
            np.zeros(len(quadrature._unique_kv)),
            color="k",
            marker="s",
            linestyle="",
            label="Knots",
        )
        for i in range(np.shape(B0)[0]):
            axs[0].plot(knots, B0[i, :], linewidth=1, color="k")
        axs[0].plot(knots, B0[ctrlpts, :], linewidth=2, color="tab:blue")
        axs[0].fill_between(x=knots, y1=B0[ctrlpts, :], color="tab:blue", alpha=0.2)

        # Matrix
        cmap = ListedColormap(["white", "tab:gray", "tab:blue"])
        A = quadrature.weights[0] @ quadrature.basis[0].T
        A = A.toarray()
        A = np.where(A != 0, 1, 0)
        B = np.zeros(shape=np.shape(A))
        B[ctrlpts, :] = A[ctrlpts, :]
        axs[1].imshow(A + B, cmap=cmap, interpolation=None)

        # Add gridlines for imshow
        filename += extension
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        return filename

    images = []
    for i in range(4):
        filename = case3(FOLDER, EXTENSION, el=i)
        if EXTENSION == ".png":
            images.append(imageio.imread(filename))
        elif EXTENSION == ".pdf":
            images.append(convert_from_path(filename, dpi=300)[0])
    images += images[-1::-1]
    imageio.mimsave(FOLDER + "elbyel.gif", images, duration=0.6)

    images = []
    for i in range(6):
        filename = case4(FOLDER, EXTENSION, ctrlpts=i)
        if EXTENSION == ".png":
            images.append(imageio.imread(filename))
        elif EXTENSION == ".pdf":
            images.append(convert_from_path(filename, dpi=300)[0])
    images += images[-1::-1]
    imageio.mimsave(FOLDER + "rowbyrow.gif", images, duration=0.4)

elif VIDCASE == 3:  # Welding

    from src.lib_part import vtk2png
    from src.single_patch.lib_job_heat_transfer import *

    def powerDensity(args: dict):
        POWER = 20
        RADIUS = 0.25
        VELOCITY = 0.5
        position = args["position"]
        t = args["time"]
        x = position[0, :]
        y = position[1, :]
        nc_sp = np.size(position, axis=1)
        f = np.zeros(nc_sp)
        if t > 16:
            return f
        rsquared = (x - VELOCITY * t) ** 2 + y**2
        f = POWER * np.exp(-rsquared / (RADIUS**2))
        return f

    def simulate(degree, nbel, nbel_time=None):

        geo_parameters = {
            "name": "TP",
            "degree": degree,
            "nbel": nbel,
            "geo_parameters": {
                "XY": np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
            },
        }
        geometry = mygeomdl(geo_parameters).export_geometry()
        patch = singlepatch(geometry, quad_args={"quadrule": "gs", "type": "leg"})

        if nbel_time is None:
            nbel_time = np.copy(nbel)
        timespan = 20
        time_inc = np.linspace(0, timespan, nbel_time + 1)

        # Add material
        material = heat_transfer_mat()
        material.add_capacity(1, is_uniform=True)
        material.add_conductivity(2, is_uniform=True, ndim=2)

        # Block boundaries
        boundary_inc = boundary_condition(
            nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=1
        )
        boundary_inc.add_constraint(
            location_list=[{"direction": "y", "face": "top"}],
            constraint_type="dirichlet",
        )

        # Transient model
        problem_inc = heat_transfer_problem(material, patch, boundary_inc)
        temperature = np.zeros((patch.nbctrlpts_total, len(time_inc)))

        # Add external force
        external_force = np.zeros_like(temperature)
        for i, t in enumerate(time_inc):
            external_force[:, i] = problem_inc.assemble_volumetric_force(
                powerDensity, args={"time": t}
            )

        # Solve
        problem_inc.solve_heat_transfer(
            temperature, external_force, time_list=time_inc, alpha=0.5
        )

        return problem_inc, time_inc, temperature

    degree, nbel = 4, 16
    problem_inc, time_inc, output = simulate(degree, nbel, nbel_time=40)
    for k, i in enumerate(range(0, np.size(output, axis=1), 3)):
        problem_inc.part.postprocessing_primal(
            fields={"temp": output[:, i]}, name=f"out_{k}", folder=FOLDER2RESU
        )

    import imageio

    images = []
    for i in range(14):
        filepath = vtk2png(
            filename=f"out_{i}",
            folder=FOLDER2RESU,
            title="Temperature",
            camera_position="xy",
            position_y=0.2,
            clim=[0, 1],
        )
        images.append(imageio.imread(filepath))
    images += images[-2::-1]
    imageio.mimsave(f"{FOLDER2RESU}welding.gif", images, duration=0.4)
