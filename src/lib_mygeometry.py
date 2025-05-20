from .__init__ import *
from .lib_quadrules import gauss_quadrature
from typing import Union, Tuple


def create_uniform_knotvector(
    degree: int, nbel: int, multiplicity: int = 1
) -> np.ndarray:
    knotvector = np.concatenate(
        (
            np.zeros(degree + 1),
            np.repeat(np.linspace(0.0, 1.0, nbel + 1)[1:-1], multiplicity),
            np.ones(degree + 1),
        )
    )
    return knotvector


def make_quarter_circle(degree: int, nbel: int) -> Tuple[np.ndarray, np.ndarray]:
    # Create a uniform knot vector for the given degree and number of elements
    knotvector = create_uniform_knotvector(degree, nbel)

    # Perform Gaussian quadrature on the knot vector
    quadrature = gauss_quadrature(degree, knotvector, quad_args={"type": "leg"})
    quadrature.export_quadrature_rules()

    # Compute the matrix using the weights and basis functions from the quadrature
    matrix = quadrature.weights[0] @ quadrature.basis[0].T

    # Compute the vector using the weights and quadrature points, applying a cosine function
    vector = quadrature.weights[0] @ np.cos(np.pi / 2 * quadrature.quadpts)

    # Extract submatrices and vectors
    Ann = matrix[1:-1, 1:-1]
    And = matrix[1:-1, [0, -1]]
    bn = vector[1:-1]

    # Initialize solution vector
    u = np.zeros_like(vector)
    u[0] = 1.0  # Boundary condition
    ud = u[[0, -1]]  # Dirichlet boundary values

    # Solve the linear system for the interior points
    u[1:-1] = scsplin.spsolve(Ann, bn - And @ ud)

    # Construct control points
    ctrlpts = np.zeros((len(vector), 2))
    ctrlpts[:, 0] = u
    ctrlpts[:, 1] = np.flip(u)

    return knotvector, ctrlpts


def make_line(degree: int, nbel: int) -> Tuple[np.ndarray, np.ndarray]:
    knotvector = create_uniform_knotvector(degree, nbel)
    nbctrlpts = len(knotvector) - degree - 1
    ctrlpts = np.array(
        [
            sum(knotvector[i + j] for j in range(degree)) / degree
            for i in range(1, nbctrlpts + 1)
        ]
    )
    return knotvector, ctrlpts


class mygeomdl:
    def __init__(self, geo_args: dict):
        self._ndim: int = 0
        self._name: str = str(geo_args.get("name", "")).lower()
        self._degree: Union[int, np.ndarray] = geo_args.get("degree")
        if np.isscalar(self._degree):
            self._degree = np.array([self._degree] * 3)
        self._nbel: Union[int, np.ndarray] = geo_args.get("nbel")
        if np.isscalar(self._nbel):
            self._nbel = np.array([self._nbel] * 3)
        self._extra_args: dict = geo_args.get("geo_parameters", {})
        assert isinstance(
            self._extra_args, dict
        ), "Extra arguments should be dictionary"

    def export_geometry(self) -> Union[BSpline.Curve, BSpline.Surface, BSpline.Volume]:

        geometry_map = {
            "line": (1, {"L": 1.0}, self._create_line),
            "ln": (1, {"L": 1.0}, self._create_line),
            "quarter_annulus": (
                2,
                {"Rin": 0.25, "Rex": 1.0},
                self._create_quarter_annulus,
            ),
            "qa": (2, {"Rin": 0.25, "Rex": 1.0}, self._create_quarter_annulus),
            "square": (
                2,
                {"XY": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])},
                self._create_quadrilateral,
            ),
            "sq": (
                2,
                {"XY": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])},
                self._create_quadrilateral,
            ),
            "trapezium": (
                2,
                {"XY": np.array([[0.0, -0.5], [0.4, -0.2], [0.4, 0.2], [0.0, 0.5]])},
                self._create_quadrilateral,
            ),
            "tp": (
                2,
                {"XY": np.array([[0.0, -0.5], [0.4, -0.2], [0.4, 0.2], [0.0, 0.5]])},
                self._create_quadrilateral,
            ),
            "cube": (3, {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0}, self._create_parallelepiped),
            "cb": (3, {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0}, self._create_parallelepiped),
            "thick_ring": (
                3,
                {"Rin": 1.0, "Rex": 2.0, "height": 1.0},
                self._create_thick_ring,
            ),
            "tr": (3, {"Rin": 1.0, "Rex": 2.0, "height": 1.0}, self._create_thick_ring),
            "rotated_quarter_annulus": (
                3,
                {"Rin": 1.0, "Rex": 2.0, "exc": 1.0},
                self._create_rotated_quarter_annulus,
            ),
            "rqa": (
                3,
                {"Rin": 1.0, "Rex": 2.0, "exc": 1.0},
                self._create_rotated_quarter_annulus,
            ),
            "prism": (
                3,
                {
                    "XY": np.array([[0.0, -7.5], [6.0, -2.5], [6.0, 2.5], [0.0, 7.5]]),
                    "height": 1.0,
                },
                self._create_prism,
            ),
            "vb": (
                3,
                {
                    "XY": np.array([[0.0, -7.5], [6.0, -2.5], [6.0, 2.5], [0.0, 7.5]]),
                    "height": 1.0,
                },
                self._create_prism,
            ),
        }

        if self._name not in geometry_map:
            raise Warning("Not developed in this library")

        ndim, default_extra_args, func = geometry_map[self._name]
        self._ndim = ndim
        self._extra_args = {**default_extra_args, **self._extra_args}
        self._degree = self._degree[: self._ndim]
        self._nbel = self._nbel[: self._ndim]
        return func(*self._degree, *self._nbel, args=self._extra_args)

    # ----------------
    # CREATE GEOMETRY
    # ----------------

    # 1D
    def _create_line(self, degree_u: int, nbel_u: int, args: dict) -> BSpline.Curve:
        "Creates a line segment"

        length = args.get("L")

        # Get uniform control points
        knotvector_u, ctrlpts_u = make_line(degree_u, nbel_u)
        ctrlpts_u = length * ctrlpts_u

        # Create curve
        obj = BSpline.Curve()
        obj.degree = degree_u
        obj.ctrlpts = [[x, 0.0] for x in ctrlpts_u]
        obj.knotvector = knotvector_u
        return obj

    # 2D
    def _create_quarter_annulus(
        self, degree_u: int, degree_v: int, nbel_u: int, nbel_v: int, args: dict
    ) -> BSpline.Surface:
        "Creates a quarter of a ring (or annulus)"

        Rin, Rex = args.get("Rin"), args.get("Rex")

        # Construction of the arc
        nb_ctrlpts_v = degree_v + nbel_v
        knotvector_v, ctrlpts_arc = make_quarter_circle(degree_v, nbel_v)

        # Construction of line
        nb_ctrlpts_u = degree_u + nbel_u
        knotvector_u, ctrlpts_line = make_line(degree_u, nbel_u)
        ctrlpts_line = Rin + ctrlpts_line * (Rex - Rin)

        # Construction of annulus sector
        ctrlpts = [
            [x_line * x_arc, x_line * y_arc, 0.0]
            for x_line in ctrlpts_line
            for x_arc, y_arc in ctrlpts_arc
        ]

        # Create surface
        obj = BSpline.Surface()
        obj.degree_u = degree_u
        obj.degree_v = degree_v
        obj.ctrlpts_size_u, obj.ctrlpts_size_v = int(nb_ctrlpts_u), int(nb_ctrlpts_v)
        obj.set_ctrlpts(ctrlpts, nb_ctrlpts_u, nb_ctrlpts_v)
        obj.knotvector_u = knotvector_u
        obj.knotvector_v = knotvector_v

        return obj

    def _create_quadrilateral(
        self, degree_u: int, degree_v: int, nbel_u: int, nbel_v: int, args: dict
    ) -> BSpline.Surface:
        "Creates a quadrilateral given coordinates in counterclockwise direction"

        xy = args.get("XY")

        # Set reference position and real position
        x0 = [0.0, 1.0, 1.0, 0.0]
        y0 = [0.0, 0.0, 1.0, 1.0]
        x1 = xy[:, 0]
        y1 = xy[:, 1]

        # Transformation of control points
        # x1 = ax1 x0 + ax2 y0 + ax3 x0 y0 + ax4
        # y1 = ay1 x0 + ay2 y0 + ay3 x0 y0 + ay4
        T = [[x0[i], y0[i], x0[i] * y0[i], 1] for i in range(4)]
        ax = np.linalg.solve(T, x1)
        ay = np.linalg.solve(T, y1)

        # Set control points
        nb_ctrlpts_u = degree_u + nbel_u
        nb_ctrlpts_v = degree_v + nbel_v
        knotvector_u, ctrlpts_u = make_line(degree_u, nbel_u)
        knotvector_v, ctrlpts_v = make_line(degree_v, nbel_v)

        ctrlpts = [
            [
                ax[0] * xt + ax[1] * yt + ax[2] * xt * yt + ax[3],
                ay[0] * xt + ay[1] * yt + ay[2] * xt * yt + ay[3],
                0.0,
            ]
            for xt in ctrlpts_u
            for yt in ctrlpts_v
        ]

        # Create surface
        obj = BSpline.Surface()
        obj.degree_u = degree_u
        obj.degree_v = degree_v
        obj.ctrlpts_size_u, obj.ctrlpts_size_v = int(nb_ctrlpts_u), int(nb_ctrlpts_v)
        obj.set_ctrlpts(ctrlpts, nb_ctrlpts_u, nb_ctrlpts_v)
        obj.knotvector_u = knotvector_u
        obj.knotvector_v = knotvector_v

        return obj

    # 3D
    def _create_parallelepiped(
        self,
        degree_u: int,
        degree_v: int,
        degree_w: int,
        nbel_u: int,
        nbel_v: int,
        nbel_w: int,
        args: dict,
    ) -> BSpline.Volume:
        "Creates a brick (or parallelepiped)"

        Lx, Ly, Lz = args.get("Lx"), args.get("Ly"), args.get("Lz")

        # Set number of control points
        nb_ctrlpts_u = degree_u + nbel_u
        nb_ctrlpts_v = degree_v + nbel_v
        nb_ctrlpts_w = degree_w + nbel_w

        # Get uniform control points
        knotvector_u, ctrlpts_u = make_line(degree_u, nbel_u)
        knotvector_v, ctrlpts_v = make_line(degree_v, nbel_v)
        knotvector_w, ctrlpts_w = make_line(degree_w, nbel_w)

        # Create control points of the volume
        ctrlpts = [
            [cptu * Lx, cptv * Ly, cptw * Lz]
            for cptw in ctrlpts_w
            for cptu in ctrlpts_u
            for cptv in ctrlpts_v
        ]

        # Create a B-spline volume
        obj = BSpline.Volume()
        obj.degree_u, obj.degree_v, obj.degree_w = degree_u, degree_v, degree_w
        obj.ctrlpts_size_u, obj.ctrlpts_size_v, obj.ctrlpts_size_w = (
            int(nb_ctrlpts_u),
            int(nb_ctrlpts_v),
            int(nb_ctrlpts_w),
        )
        obj.set_ctrlpts(ctrlpts, nb_ctrlpts_u, nb_ctrlpts_v, nb_ctrlpts_w)
        obj.knotvector_u = knotvector_u
        obj.knotvector_v = knotvector_v
        obj.knotvector_w = knotvector_w

        return obj

    def _create_thick_ring(
        self,
        degree_u: int,
        degree_v: int,
        degree_w: int,
        nbel_u: int,
        nbel_v: int,
        nbel_w: int,
        args: dict,
    ) -> BSpline.Volume:
        "Creates a thick ring (quarter of annulus extruded)"

        Rin, Rex, height = args.get("Rin"), args.get("Rex"), args.get("height")

        # construction of the arc
        nb_ctrlpts_v = degree_v + nbel_v
        knotvector_v, ctrlpts_arc = make_quarter_circle(degree_v, nbel_v)

        # construction of line
        nb_ctrlpts_u = degree_u + nbel_u
        knotvector_u, ctrlpts_line = make_line(degree_u, nbel_u)
        ctrlpts_line = Rin + ctrlpts_line * (Rex - Rin)

        nb_ctrlpts_w = degree_w + nbel_w
        knotvector_w, ctrlpts_height = make_line(degree_w, nbel_w)
        ctrlpts_height = height * ctrlpts_height

        # construction of annulus sector
        ctrlpts = [
            [x_line * x_arc, x_line * y_arc, z]
            for z in ctrlpts_height
            for x_line in ctrlpts_line
            for x_arc, y_arc in ctrlpts_arc
        ]

        # Create volume
        obj = BSpline.Volume()
        obj.degree_u, obj.degree_v, obj.degree_w = degree_u, degree_v, degree_w
        obj.ctrlpts_size_u, obj.ctrlpts_size_v, obj.ctrlpts_size_w = (
            int(nb_ctrlpts_u),
            int(nb_ctrlpts_v),
            int(nb_ctrlpts_w),
        )
        obj.set_ctrlpts(ctrlpts, nb_ctrlpts_u, nb_ctrlpts_v, nb_ctrlpts_w)
        obj.knotvector_u = knotvector_u
        obj.knotvector_v = knotvector_v
        obj.knotvector_w = knotvector_w

        return obj

    def _create_rotated_quarter_annulus(
        self,
        degree_u: int,
        degree_v: int,
        degree_w: int,
        nbel_u: int,
        nbel_v: int,
        nbel_w: int,
        args: dict,
    ) -> BSpline.Volume:
        "Creates a quarter of a ring rotated (or revolted)"

        Rin, Rex, exc = args.get("Rin"), args.get("Rex"), args.get("exc")

        # construction of the arc 1
        nb_ctrlpts_v = degree_v + nbel_v
        knotvector_v, ctrlpts_arc_1 = make_quarter_circle(degree_v, nbel_v)

        # construction of line
        nb_ctrlpts_u = degree_u + nbel_u
        knotvector_u, ctrlpts_line = make_line(degree_u, nbel_u)
        ctrlpts_line = Rin + ctrlpts_line * (Rex - Rin)

        # construction of the arc 2
        nb_ctrlpts_w = degree_w + nbel_w
        knotvector_w, ctrlpts_arc_2 = make_quarter_circle(degree_w, nbel_w)

        # Get control points
        ctrlpts = [
            [
                x_line * x_arc_1,
                (x_line * y_arc_1 + exc) * y_arc_2,
                (x_line * y_arc_1 + exc) * z_arc_2,
            ]
            for y_arc_2, z_arc_2 in ctrlpts_arc_2
            for x_line in ctrlpts_line
            for x_arc_1, y_arc_1 in ctrlpts_arc_1
        ]

        # Create volume
        obj = BSpline.Volume()
        obj.degree_u, obj.degree_v, obj.degree_w = degree_u, degree_v, degree_w
        obj.ctrlpts_size_u, obj.ctrlpts_size_v, obj.ctrlpts_size_w = (
            int(nb_ctrlpts_u),
            int(nb_ctrlpts_v),
            int(nb_ctrlpts_w),
        )
        obj.set_ctrlpts(ctrlpts, nb_ctrlpts_u, nb_ctrlpts_v, nb_ctrlpts_w)
        obj.knotvector_u = knotvector_u
        obj.knotvector_v = knotvector_v
        obj.knotvector_w = knotvector_w

        return obj

    def _create_prism(
        self,
        degree_u: int,
        degree_v: int,
        degree_w: int,
        nbel_u: int,
        nbel_v: int,
        nbel_w: int,
        args: dict,
    ) -> BSpline.Volume:
        """Creates a prism using a quadrilateral as a base.
        The quadrilateral coordinates are given in counterclockwise direction"""

        xy, height = args.get("XY"), args.get("height")

        # Set reference position and real position
        x0 = [0.0, 1.0, 1.0, 0.0]
        y0 = [0.0, 0.0, 1.0, 1.0]
        x1 = xy[:, 0]
        y1 = xy[:, 1]

        # Transformation of control points
        # x1 = ax1 x0 + ax2 y0 + ax3 x0 y0 + ax4
        # y1 = ay1 x0 + ay2 y0 + ay3 x0 y0 + ay4
        T = [[x0[i], y0[i], x0[i] * y0[i], 1] for i in range(4)]
        ax = np.linalg.solve(T, x1)
        ay = np.linalg.solve(T, y1)

        # Set control points
        nb_ctrlpts_u = degree_u + nbel_u
        nb_ctrlpts_v = degree_v + nbel_v
        nb_ctrlpts_w = degree_w + nbel_w
        knotvector_u, ctrlpts_u = make_line(degree_u, nbel_u)
        knotvector_v, ctrlpts_v = make_line(degree_v, nbel_v)
        knotvector_w, ctrlpts_w = make_line(degree_w, nbel_w)

        ctrlpts = [
            [
                ax[0] * xt + ax[1] * yt + ax[2] * xt * yt + ax[3],
                ay[0] * xt + ay[1] * yt + ay[2] * xt * yt + ay[3],
                zt * height,
            ]
            for zt in ctrlpts_w
            for xt in ctrlpts_u
            for yt in ctrlpts_v
        ]

        # Create volume
        obj = BSpline.Volume()
        obj.degree_u, obj.degree_v, obj.degree_w = degree_u, degree_v, degree_w
        obj.ctrlpts_size_u, obj.ctrlpts_size_v, obj.ctrlpts_size_w = (
            int(nb_ctrlpts_u),
            int(nb_ctrlpts_v),
            int(nb_ctrlpts_w),
        )
        obj.set_ctrlpts(ctrlpts, nb_ctrlpts_u, nb_ctrlpts_v, nb_ctrlpts_w)
        obj.knotvector_u = knotvector_u
        obj.knotvector_v = knotvector_v
        obj.knotvector_w = knotvector_w

        return obj
