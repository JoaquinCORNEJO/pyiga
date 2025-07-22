from . import *
from .lib_tensor_maths import bspline_operations, eval_inverse_and_determinant
from .lib_quadrules import weighted_quadrature, gauss_quadrature
from typing import Union, List, Tuple


class singlepatch:
    def __init__(
        self,
        obj: Union[BSpline.Curve, BSpline.Surface, BSpline.Volume],
        quad_args: dict,
    ):
        verification_list = [
            isinstance(obj, BSpline.Curve),
            isinstance(obj, BSpline.Surface),
            isinstance(obj, BSpline.Volume),
        ]
        assert any(verification_list), "Geometry is not supported"
        dim_list = [1, 2, 3]
        self.ndim: int = dim_list[verification_list.index(True)]
        self.degree: int = self._read_degree(obj)
        self.knotvector: np.ndarray = self._read_knotvector(obj)

        self.nbctrlpts = np.array(
            [len(self.knotvector[i]) - self.degree[i] - 1 for i in range(self.ndim)],
            dtype=int,
        )
        self.nbctrlpts_total: int = np.prod(self.nbctrlpts)
        self.ctrlpts: np.ndarray = self._read_control_points(obj)
        self.nbqp: np.ndarray = np.zeros(self.ndim, dtype=int)
        self.nbqp_total: int = 0
        self.quadrule_list: List[Union[gauss_quadrature, weighted_quadrature]] = []

        self._set_quadrature_rules(quad_args)
        self._set_jacobien_physical_points()

    def _read_degree(
        self, obj: Union[BSpline.Curve, BSpline.Surface, BSpline.Volume]
    ) -> np.ndarray:
        "Reads degree from model"
        degree = np.ones(self.ndim, dtype=int)
        if self.ndim == 1:
            degree[0] = obj.degree
        else:
            degree[0] = obj.degree_u
            degree[1] = obj.degree_v
            if self.ndim == 3:
                degree[2] = obj.degree_w
        return degree

    def _read_knotvector(
        self, obj: Union[BSpline.Curve, BSpline.Surface, BSpline.Volume]
    ) -> List[np.ndarray]:
        "Reads knot-vector from model"
        knotvector = []
        if self.ndim == 1:
            knotvector.append(obj.knotvector)
        else:
            knotvector.append(obj.knotvector_u)
            knotvector.append(obj.knotvector_v)
            if self.ndim == 3:
                knotvector.append(obj.knotvector_w)
        return knotvector

    def _read_control_points(
        self, obj: Union[BSpline.Curve, BSpline.Surface, BSpline.Volume]
    ) -> np.ndarray:
        "Reads control points from model"

        nbctrlpts = self.nbctrlpts
        ctrlpts_old = obj.ctrlpts
        ctrlpts_new = np.zeros((self.ndim, self.nbctrlpts_total))

        idx_new = 0
        if self.ndim == 1:
            for i in range(nbctrlpts[0]):
                idx_old = i
                ctrlpts_new[:, idx_new] = np.copy(ctrlpts_old[idx_old])[: self.ndim]
                idx_new += 1

        elif self.ndim == 2:
            for j in range(nbctrlpts[1]):
                for i in range(nbctrlpts[0]):
                    idx_old = j + i * nbctrlpts[1]
                    ctrlpts_new[:, idx_new] = np.copy(ctrlpts_old[idx_old])[: self.ndim]
                    idx_new += 1

        elif self.ndim == 3:
            for k in range(nbctrlpts[2]):
                for j in range(nbctrlpts[1]):
                    for i in range(nbctrlpts[0]):
                        idx_old = j + i * nbctrlpts[1] + k * nbctrlpts[1] * nbctrlpts[0]
                        ctrlpts_new[:, idx_new] = np.copy(ctrlpts_old[idx_old])[
                            : self.ndim
                        ]
                        idx_new += 1

        return ctrlpts_new

    def _set_quadrature_rules(self, quad_args: dict):
        self.quadrule_list = []
        quadrule_name = str(quad_args.get("quadrule", "gs")).lower()
        assert quadrule_name in ["gs", "wq"], "Unknown method"
        quadrule_class = (
            gauss_quadrature if quadrule_name == "gs" else weighted_quadrature
        )
        for i in range(self.ndim):
            quadrule = quadrule_class(
                self.degree[i], self.knotvector[i], quad_args=quad_args
            )
            quadrule.export_quadrature_rules()
            self.nbqp[i] = quadrule.nbquadpts
            self.quadrule_list.append(quadrule)
        self.nbqp_total = np.prod(self.nbqp)

    def _set_jacobien_physical_points(self):
        "Computes jacobien and physical position"
        jac = bspline_operations.eval_jacobien(self.quadrule_list, self.ctrlpts)
        self.det_jac, self.inv_jac = eval_inverse_and_determinant(jac)
        self.qp_phy = bspline_operations.interpolate_meshgrid(
            self.quadrule_list, self.ctrlpts
        )
        assert np.all(
            self.det_jac > 0.0
        ), "Geometry problem. See control points positions"

    def _compute_global_mesh_parameter(self) -> float:
        parametric_distance = []
        for i in range(self.ndim):
            kvunique = np.unique(self.knotvector[i])
            parametric_distance.append(np.max(np.abs(np.diff(kvunique))))
        return np.max(parametric_distance)

    def interpolate_field(
        self,
        knots_list: np.ndarray,
        u_ctrlpts: Union[None, np.ndarray] = None,
        eval_geometry: bool = True,
    ) -> Tuple[
        Union[None, np.ndarray], Union[None, np.ndarray], Union[None, np.ndarray]
    ]:
        u_interp, pts_phy, det_jac = None, None, None
        if isinstance(u_ctrlpts, np.ndarray):
            u_interp = bspline_operations.interpolate_meshgrid(
                self.quadrule_list, np.atleast_2d(u_ctrlpts), knots_list
            )
        if eval_geometry:
            pts_phy = bspline_operations.interpolate_meshgrid(
                self.quadrule_list, self.ctrlpts, knots_list
            )
            jac = bspline_operations.eval_jacobien(
                self.quadrule_list, self.ctrlpts, knots_list
            )
            det_jac = eval_inverse_and_determinant(jac)[0]
        return u_interp, pts_phy, det_jac

    def postprocessing_primal(
        self,
        folder: Union[str, None] = None,
        name: str = "output",
        fields: dict = {},
        extra_args: dict = {},
        sample_size: Union[int, None] = None,
        write_file: bool = True,
    ):
        """Export solution in VTK format.
        It is possible to use Paraview to visualize data
        """
        from pyevtk.hl import gridToVTK

        if write_file:
            if folder is None:
                dirname = os.path.dirname
                folder = f"{dirname(dirname(os.path.realpath(__file__)))}/results/"
            if not os.path.isdir(folder):
                os.mkdir(folder)
            print("File saved in %s" % folder)

        if sample_size is None:
            sample_size = np.max(self.nbqp) * np.ones(self.ndim, dtype=int)
        elif np.isscalar(sample_size):
            sample_size = int(np.copy(sample_size)) * np.ones(self.ndim, dtype=int)
        knots_list = [np.linspace(0, 1, sample_size[i]) for i in range(self.ndim)]

        # Interpolate physical points and compute determinant of Jacobian
        _, pts_phy, det_jac = self.interpolate_field(
            knots_list=knots_list, eval_geometry=True
        )

        # Extract x, y, and z coordinates from physical points
        x_data = pts_phy[0, :]
        y_data = pts_phy[1, :]
        z_data = np.zeros_like(x_data) if self.ndim < 3 else pts_phy[2, :]

        # Reshape data into the required format for visualization
        position_data = [
            np.reshape(data, newshape=(sample_size[0], sample_size[1], -1))
            for data in [x_data, y_data, z_data, det_jac]
        ]

        # Create point data
        point_data = {}
        for fieldname, fieldvalue in fields.items():

            if isinstance(fieldvalue, np.ndarray):
                fieldinterp = self.interpolate_field(
                    knots_list=knots_list,
                    u_ctrlpts=np.atleast_2d(fieldvalue),
                    eval_geometry=False,
                )[0]
            elif callable(fieldvalue):
                extra_args = {"position": pts_phy} | extra_args
                fieldinterp = fieldvalue(extra_args)
                fieldinterp = np.atleast_2d(fieldinterp)
            else:
                continue

            nrows = np.size(fieldinterp, axis=0)
            for idx_row in range(nrows):
                newfieldname = (
                    fieldname if nrows == 1 else f"{fieldname}{'_'}{idx_row+1}"
                )
                point_data[newfieldname] = np.reshape(
                    fieldinterp[idx_row, :],
                    newshape=(sample_size[0], sample_size[1], -1),
                )

        point_data["det_jac"] = position_data[3]
        if write_file:
            gridToVTK(
                f"{folder}{name}",
                position_data[0],
                position_data[1],
                position_data[2],
                pointData=point_data,
            )
        return position_data, point_data

    def postprocessing_dual(
        self,
        folder: Union[str, None] = None,
        name: str = "output",
        fields: dict = {},
        write_file: bool = True,
    ):

        from pyevtk.hl import gridToVTK

        if write_file:
            if folder is None:
                dirname = os.path.dirname
                folder = f"{dirname(dirname(os.path.realpath(__file__)))}/results/"
            if not os.path.isdir(folder):
                os.mkdir(folder)
            print("File saved in %s" % folder)

        sample_size = np.copy(self.nbqp)

        # Extract x, y, and z coordinates from physical points
        x_data = self.qp_phy[0, :]
        y_data = self.qp_phy[1, :]
        z_data = np.zeros_like(x_data) if self.ndim < 3 else self.qp_phy[2, :]

        # Reshape data into the required format for visualization
        position_data = [
            np.reshape(data, newshape=(sample_size[0], sample_size[1], -1))
            for data in [x_data, y_data, z_data, self.det_jac]
        ]

        # Create point data
        point_data = {}
        for fieldname, fieldvalue in fields.items():
            if fieldvalue is None:
                continue
            fieldinterp: np.ndarray = np.atleast_2d(fieldvalue)
            nrows = fieldinterp.shape[0]
            for idx_row in range(nrows):
                newfieldname = (
                    f"{fieldname}{'_'}{idx_row+1}" if nrows > 1 else fieldname
                )
                point_data[newfieldname] = np.reshape(
                    fieldinterp[idx_row, :],
                    newshape=(sample_size[0], sample_size[1], -1),
                )

        point_data["det_jac"] = position_data[3]
        if write_file:
            gridToVTK(
                folder + name,
                position_data[0],
                position_data[1],
                position_data[2],
                pointData=point_data,
            )

        return position_data, point_data


def crop_image(filename: str):
    from PIL import Image

    im = np.array(Image.open(filename).convert("RGB"))
    colorY, colorX = np.where(np.all(im != [255, 255, 255], axis=2))
    top, bottom = min(colorY), max(colorY)
    left, right = min(colorX), max(colorX)
    Image.fromarray(im[top:bottom, left:right]).save(filename)


def vtk2png(
    filename: str,
    folder: Union[str, None] = None,
    title: Union[str, None] = "add_title",
    fieldname: str = "temp",
    clim: Union[list, tuple, None] = None,
    cmap: str = "viridis",
    fmt: str = "%.1f",
    n_labels: int = 3,
    position_x: float = 0.2,
    position_y: float = 0.1,
    n_colors: int = 101,
    camera_position: str = "yx",
) -> str:

    import pyvista as pv

    if folder is None:
        dirname = os.path.dirname
        folder = f"{dirname(dirname(os.path.realpath(__file__)))}/results/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    print("File saved in %s" % folder)

    reader = pv.get_reader(f"{folder}{filename}.vts")
    assert fieldname in reader.point_array_names, "Unknown fieldname"
    reader.disable_all_point_arrays()
    reader.enable_point_array(fieldname)
    scalars_name = deepcopy(fieldname)

    grid = reader.read()
    scalars_values = grid[scalars_name]
    valid_mask = ~np.isnan(scalars_values)
    nan_mask = np.isnan(scalars_values)
    has_nan = np.any(nan_mask)

    grid_valid = grid.extract_points(valid_mask)
    grid_nan = grid.extract_points(nan_mask) if has_nan else None

    sargs = dict(
        title=title,
        title_font_size=50,
        label_font_size=40,
        shadow=True,
        n_labels=n_labels,
        fmt=fmt,
        position_x=position_x,
        position_y=position_y,
    )

    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(
        grid_valid,
        cmap=cmap,
        clim=clim,
        reset_camera=True,
        scalar_bar_args=sargs,
        scalars=scalars_name,
        n_colors=n_colors,
    )

    if has_nan:
        plotter.add_mesh(grid_nan, color="lightgray", show_scalar_bar=False)

    plotter.camera_position = camera_position
    plotter.camera.zoom(0.9)
    plotter.background_color = "white"
    plotter.window_size = [1600, 1600]
    path = f"{folder}{fieldname}_{filename}.png"
    plotter.screenshot(path)
    crop_image(path)
    return path


def regroupe_vtk_files(filename: str, folder: str, nbfiles: int, format: str = "vts"):
    from pyevtk.vtk import VtkGroup

    print("Creating group...")
    group = VtkGroup(folder)
    for i in range(0, nbfiles + 1):
        group.addFile(filepath=f"{folder}{filename}_{i}.{format}", sim_time=i)
    group.save()
