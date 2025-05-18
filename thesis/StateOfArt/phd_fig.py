"""
.. Isogeometric analysis figures
.. Select case:
.. CASE 0: B-spline curve
.. CASE 1: Univariate B-spline functions in parametric space
.. CASE 2: Bivariate B-spline functions in parametric space
.. CASE 3: Quadrature points in IGA-Galerkin approach
.. CASE 4: Quadrature points in IGA-WQ approach
.. CASE 5: B-spline surface 2D
.. CASE 6: Quadrature rules W00 or W11
"""

from thesis.StateOfArt.__init__ import *
from src.lib_quadrules import *
from src.lib_mygeometry import *
from src.lib_part import singlepatch

def plot_2d_geometry(patch:singlepatch, sample_size=101, plotaxis=True):

	def plot_mesh(pts, shape, ax):
		"Plots mesh of control points"

		if pts.shape[0] == 3: pts = pts[:2, :]

		pts2D = []
		for j in range(shape[1]):
			pts2D_temp = []
			for i in range(shape[0]):
				pos = i + j*shape[0]
				pts2D_temp.append(pts[:, pos].tolist())
			pts2D.append(pts2D_temp)
		pts2D = np.asarray(pts2D)

		# In the first direction
		for _ in range(shape[1]):
			x = pts2D[_, :, 0]; y = pts2D[_, :, 1]
			ax.plot(x, y, color=COLORLIST[1], linestyle='--')

		# In the second direction
		for _ in range(shape[0]):
			x = pts2D[:, _, 0]; y = pts2D[:, _, 1]
			ax.plot(x, y, color=COLORLIST[1], linestyle='--')

		return

	ctrlpts = patch.ctrlpts
	evalpts = patch.interpolate_field(knots_list=[np.linspace(0, 1, sample_size) for _ in range(patch.ndim)])[1]
	u_knots = patch.interpolate_field(knots_list=[np.unique(knotvector) for knotvector in patch.knotvector])[1]

	X = np.asarray(evalpts[0, :].reshape((sample_size, sample_size)).tolist())
	Y = np.asarray(evalpts[1, :].reshape((sample_size, sample_size)).tolist())
	Z = np.zeros_like(X)

	fig, ax = plt.subplots(figsize=(5, 5))
	ax.grid(None)
	plot_mesh(ctrlpts, patch.nbctrlpts, ax)
	ax.pcolormesh(X, Y, Z, cmap=plt.cm.Paired, shading='gouraud')
	ax.plot([], [], label='B-Spline surface')
	ax.plot(ctrlpts[0, :], ctrlpts[1, :], 'o', label='Control points net')
	ax.plot(u_knots[0, :], u_knots[1, :], color='k', marker='s', linestyle='', label='Knots')

	ax.set_xticks(np.arange(0, np.ceil(max(evalpts[:, 0]))+2, .5))
	ax.set_yticks(np.arange(0, np.ceil(max(evalpts[:, 1]))+1, .5))
	ax.set_xlim(left=-0.1, right=1.25)
	ax.set_ylim(bottom=-0.1, top=1.25)

	ax.legend()
	ax.set_aspect('equal', adjustable='box')
	if plotaxis:
		ax.set_xlabel(r'$\xi_1$')
		ax.set_ylabel(r'$\xi_2$')
	else:
		ax.axis('off')
	fig.tight_layout()

	return fig

def plot_vertical_lines(x, y, ax=None, color='k'):
	for xi, yi in zip(x, y):
		ax.plot([xi, xi], [0, yi], color=color)
	return

# Set global variables
FIGCASE = 7
EXTENSION = '.pdf'

if FIGCASE == 0: # B-spline curve

	def case0(folder, extension, refinement=2):

		# Set filename
		filename = f"{folder}BsplineCurve{refinement:d}{extension}"

		# Create the curve
		crv            = BSpline.Curve()
		crv.degree     = 3
		crv.ctrlpts    = [[-1, 1, 0], [-0.5, 0.25, 0], [0, 2, 0], [0.5, 1., 0.],
							[0.75, -0.5, 0], [1.5, 1, 0], [2, 0, 0]]
		crv.knotvector = np.array([0., 0., 0., 0., 0.25, 0.75, 0.75, 1., 1., 1., 1.])

		if refinement == 1:
			# Knot insertion
			uniqueKV = np.unique(crv.knotvector)
			newknots = [(uniqueKV[i]+uniqueKV[i+1])/2 for i in range(0, len(uniqueKV)-1)]
			for knot in newknots: crv.insert_knot(knot)

		elif refinement == 2:
			# Degree elevation
			del crv
			crv = BSpline.Curve()
			crv.degree = 4
			crv.ctrlpts = [[-1, 1], [-0.625, 0.4375], [-0.4167, 0.5417], [-0.0417, 1.625], [0.3333, 1.3333],
					[0.5417,0.75], [0.7292, -0.375], [1.125, 0.25], [1.625, 0.75], [2, 0]]
			crv.knotvector = np.array([0., 0., 0., 0., 0., 0.25, 0.25, 0.75, 0.75, 0.75, 1., 1., 1., 1., 1.])

		# Get data
		evalpts = np.asarray(crv.evalpts); ctrlpts = np.asarray(crv.ctrlpts)
		basis = eval_ders_basis_sparse(crv.degree, crv.knotvector, np.unique(crv.knotvector))
		u_knots = basis[0].T @ np.array(crv.ctrlpts)

		fig, ax = plt.subplots(figsize=(5,3))
		ax.plot(evalpts[:, 0], evalpts[:, 1], label='B-Spline curve')
		ax.plot(ctrlpts[:, 0], ctrlpts[:, 1], 'o--', markersize=10, label='Control points net')
		ax.plot(u_knots[:, 0], u_knots[:, 1], color='k', marker='s', linestyle='', label='Knots')
		ax.set_xlim([-1.5, 2.5]); ax.set_ylim([-1., 2.5])
		ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
		ax.axis('off')

		fig.tight_layout()
		fig.savefig(filename, dpi=300)

		return

	for ii in range(3): case0(FOLDER2RESU, EXTENSION, refinement=ii)

elif FIGCASE == 1: # Univariate functions

	def case1(folder, extension, addDers=True):

		# Set filename
		filename = f"{folder}BSkninw"

		# # B-spline properties
		# degree = 2
		# nbel, multiplicity = 4, 1
		# knotvector = create_uniform_knotvector(degree, nbel, multiplicity=multiplicity)

		# Other configurations
		degree = 3
		knotvector = np.array([0., 0., 0., 0., 0.25, 0.75, 0.75, 1., 1., 1., 1.])
		# knotvector = np.array([0., 0., 0., 0., 0.125, 0.25, 0.5, 0.75, 0.75, 0.875, 1., 1., 1., 1.])
		# knotvector = np.array([0., 0., 0., 0., 0., 0.25, 0.25, 0.75, 0.75, 0.75, 1., 1., 1., 1., 1.])
		# knotvector = np.array([0., 0., 0., 0., 0., 0.125, 0.25, 0.5, 0.75, 0.75, 0.875, 1., 1., 1., 1., 1.])
		# knotvector = np.array([0., 0., 0., 0., 0., 0.125, 0.125, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 0.875, 0.875, 1., 1., 1., 1., 1.])
		nbel = np.unique(knotvector).shape[0] - degree - 1

		quadRule = quadrature_rule(degree, knotvector)
		knots = np.linspace(0, 1, 101)
		basis = quadRule.get_sample_basis(knots)
		B0 = basis[0].toarray(); B1 = basis[1].toarray()

		if addDers:
			filename = f"{filename}Ders"
			fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
			for i in range(np.shape(B0)[0]):
				ax1.plot(knots, B0[i, :], linewidth=2)
				ax2.plot(knots, B1[i, :], linewidth=2)

			for ax in [ax1, ax2]:
				ax.set_xlabel(r'$\xi$')
				ax.set_xticks(np.linspace(0, 1, nbel+1))
			ax1.set_ylabel(r'$\hat{b}_{A,\,p}(\xi)$')
			ax2.set_ylabel(r"${\hat{b}'}_{A,\,p}(\xi)$")
			ax1.axis('off')
			ax2.axis('off')

		else:
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
			for i in range(np.shape(B0)[0]):
				ax.plot(knots, B0[i, :], color='k')
			ax.plot([], [], linewidth=1, color='k', label='B-Spline basis')
			ax.plot(quadRule._unique_kv, np.zeros(len(quadRule._unique_kv)), color='k', marker='s', linestyle='', label='Knots')

			ax.set_xlabel(r'$\xi$')
			ax.set_xticks(np.linspace(0, 1, 5))
			ax.set_yticks([0, 0.5, 1])
			ax.set_ylabel(r'$\hat{b}_{A,\,p}(\xi)$')
			ax.legend()
			# ax.axis('off')

		filename = f"{filename}{extension}"
		fig.tight_layout()
		fig.savefig(filename, dpi=300)
		return

	for addDers in [True, False]: case1(FOLDER2RESU, EXTENSION, addDers=addDers)

elif FIGCASE == 2: # Bivariate functions

	def case2(folder, extension, is2D=True):

		# Set filename
		filename = f"{folder}BivariateFunctions"

		# B-Spline properties
		degree, nbel = 2, 4
		knotvector = create_uniform_knotvector(degree, nbel)
		quadrature = quadrature_rule(degree, knotvector)
		knots = np.linspace(0, 1, 201)
		basis = quadrature.get_sample_basis(knots)
		B0 = basis[0].toarray()
		B02plot = B0[2, :]

		# B-Spline 2D
		X, Y = np.meshgrid(knots, knots)
		Z = np.kron(B02plot, B02plot).reshape((len(knots), len(knots)))

		if is2D:
			filename += '2D'
			from mpl_toolkits.axes_grid1 import make_axes_locatable
			fig, axs = plt.subplots(2, 2, sharex="col", sharey="row",
									gridspec_kw=dict(height_ratios=[1, 3.2],
													width_ratios=[3.2, 1]), figsize=(5, 5))

			axs[0,1].set_visible(False)
			axs[0,0].set_box_aspect(1/3)
			axs[1,0].set_box_aspect(1)
			axs[1,1].set_box_aspect(3/1)
			axs[1,0].grid(None)
			im = axs[1,0].pcolormesh(X, Y, Z, cmap='GnBu', shading='gouraud', rasterized=True)
			divider = make_axes_locatable(axs[1, 0])
			fig.colorbar(im, cax=axs[1, 0].inset_axes((0.8, 0.55, 0.025, 0.4)))

			axs[1,0].set_yticks([0, 0.5, 1])
			axs[1,0].set_xticks([0, 0.5, 1])

			for i in range(degree+nbel):
				axs[0, 0].plot(knots, B0[i, :], color="0.8")
				axs[1, 1].plot(B0[i, :], knots, color="0.8")

			axs[0,0].plot(knots, B02plot); axs[0, 0].axis(ymin=0, ymax=1)
			axs[0,0].set_xlabel(r'$\xi_1$')
			axs[0,0].set_ylabel(r'$\hat{b}_{A_1,\,p_1}$')
			axs[1,1].plot(B02plot, knots); axs[1, 1].axis(xmin=0, xmax=1)
			axs[1,1].set_ylabel(r'$\xi_2$')
			axs[1,1].set_xlabel(r'$\hat{b}_{A_2,\,p_2}$')
			axs[1,1].set_xticks([0, 1])
			axs[0,0].set_yticks([0, 1])

		else:
			filename += '3D'
			fig = plt.figure(figsize=(6, 4))
			ax = fig.add_subplot(111, projection='3d')
			ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='GnBu', edgecolor='k', lw=0.02, rasterized=True)
			for i in range(degree+nbel):
				ax.plot3D(knots, 1.1*np.ones(len(knots)), B0[i, :], color="0.8")
				ax.plot3D(1.1*np.ones(len(knots)), knots, B0[i, :], color="0.8")

			ax.grid(False)
			ax.plot3D(knots, 1.1*np.ones(len(knots)), B02plot, color=COLORLIST[0])
			ax.plot3D(1.1*np.ones(len(knots)), knots, B02plot, color=COLORLIST[0])
			ax.set_xticks([0, 0.5, 1])
			ax.set_yticks([0, 0.5, 1])
			ax.set_zticks([0, 0.5, 1])
			ax.set_xlabel(r'$\xi_1$')
			ax.set_ylabel(r'$\xi_2$')
			ax.invert_xaxis()

		filename = f"{filename}{extension}"
		fig.tight_layout()
		fig.savefig(filename, dpi=300)
		return

	for is2D in [True, False]: case2(FOLDER2RESU, EXTENSION, is2D=is2D)

elif FIGCASE == 3: # Quadrature points in IGA

	def case3(folder, extension='.png'):

		# Set filename
		filename = f"{folder}QuadPtsIGA{extension}"

		fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
		degree = 4
		for ax, nbel in zip([ax1, ax2], [8, 32]):
			knotvector  = create_uniform_knotvector(degree, nbel)
			quadRule    = gauss_quadrature(degree, knotvector, quad_args={})
			quadRule.export_quadrature_rules()
			XX, YY      = np.meshgrid(quadRule.quadpts, quadRule.quadpts)
			ax.plot(XX, YY, 'ko', markersize=0.5)

			grid = np.linspace(0, 1, nbel+1)
			for i in grid:
				ax.plot([i, i], [0, 1], 'grey', linewidth=0.5, alpha=0.8)
				ax.plot([0, 1], [i, i], 'grey', linewidth=0.5, alpha=0.8)

			ax.set_xticks([0, 0.5, 1])
			ax.set_yticks([0, 0.5, 1])
			ax.axis('equal')
			ax.set_ylabel(r'$\xi_2$')
			ax.set_xlabel(r'$\xi_1$')

		fig.tight_layout()
		fig.savefig(filename, dpi=300)
		return

	case3(FOLDER2RESU)

elif FIGCASE == 4: # Quadrature points in WQ

	def case4(folder, extension='.png'):

		# Set filename
		filename = f"{folder}QuadPtsWQ{extension}"

		fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
		degree = 4
		for ax, nbel in zip([ax1, ax2], [8, 32]):
			knotvector  = create_uniform_knotvector(degree, nbel)
			quadRule    = weighted_quadrature(degree, knotvector, quad_args={})
			quadRule.export_quadrature_rules()
			XX, YY      = np.meshgrid(quadRule.quadpts, quadRule.quadpts)
			ax.plot(XX, YY, 'ko', markersize=0.5)

			grid = np.linspace(0.,1,nbel+1)
			for i in grid:
				ax.plot([i, i], [0, 1], 'grey', linewidth=0.5, alpha=0.8)
				ax.plot([0, 1], [i, i], 'grey', linewidth=0.5, alpha=0.8)

			ax.set_xticks([0, 0.5, 1])
			ax.set_yticks([0, 0.5, 1])
			ax.axis('equal')
			ax.set_ylabel(r'$\xi_2$')
			ax.set_xlabel(r'$\xi_1$')

		fig.tight_layout()
		fig.savefig(filename, dpi=300)
		return

	case4(FOLDER2RESU)

elif FIGCASE == 5: # B-spline surface

	# Set filename
	filename = f"{FOLDER2RESU}BsplineSurface{EXTENSION}"

	# Surface properties
	geometry = mygeomdl(geo_args={
						'name':'quarter_annulus', 'degree': np.array([1, 3, 1]),
						'nbel': np.array([2, 2, 1]),
						}
					)
	modelIGA = geometry.export_geometry()
	modelPhy = singlepatch(modelIGA, quad_args={'quadrule':'gs'})
	fig = plot_2d_geometry(modelPhy)
	fig.savefig(filename, dpi=300)

elif FIGCASE == 6: # Weights W00 and W11

	def case6(folder, extension, ders_idx=1):

		if ders_idx not in [0, 1]: raise Warning('Not possible')
		if ders_idx == 0: ylim1 = [0, 1];  ylim2 = [0, 0.25]
		else: ylim1 = [-6, 6]; ylim2 = [-0.6, 0.6]

		# Set filename
		filename = f"{folder}WeightsW{ders_idx:d}{extension}"

		# B-spline properties
		basis_idx = 2
		degree, nbel = 2, 3
		knotvector   = create_uniform_knotvector(degree, nbel)
		quadrature   = weighted_quadrature(degree, knotvector, {'type': 2})
		quadrature.export_quadrature_rules()
		knots = np.linspace(0, 1, 300)
		basis = quadrature.get_sample_basis(knots)[ders_idx].toarray()

		# Get weights
		weights = quadrature.weights[-ders_idx].toarray()

		fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
		for i in range(np.shape(basis)[0]):
			ax1.plot(knots, np.ravel(basis[i, :]), linewidth=2)

		# Fill basis chosen
		ax1.fill_between(x=knots, y1=basis[basis_idx, :], color="g", alpha=0.2)
		ax1.set_ylim(ylim1)

		ax2 = ax1.twinx()
		ax2.plot(quadrature.quadpts, weights[basis_idx, :], 'ko')
		plot_vertical_lines(quadrature.quadpts, weights[basis_idx, :], ax2)
		ax2.set_ylim(ylim2)
		ax2.grid(None)

		ax1.set_xlabel(r'$\xi$')
		ax1.set_ylabel('Basis')
		ax2.set_ylabel('Weights')
		ax1.set_xticks(np.linspace(0, 1, nbel+1), ['0', '1/3', '2/3', '1'])
		fig.tight_layout()
		fig.savefig(filename, dpi=300)
		return

	for ders_idx in [0, 1]:
		case6(FOLDER2RESU, EXTENSION, ders_idx=ders_idx)
	
elif FIGCASE == 7:
	
	def case7(folder, extension):

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2))

		# Set filename
		filename = f"{folder}Bandwitdh{extension}"

		for degree, color, label in zip([5, 1], ['tab:green', 'tab:blue'], ['Quintic', 'Linear']):
			nbel = 8
			knotvector = create_uniform_knotvector(degree, nbel)
			quadrature = weighted_quadrature(degree, knotvector, {'type': 2})
			quadrature.export_quadrature_rules()
			knots = np.linspace(0, 1, 301)
			basis = quadrature.get_sample_basis(knots)[0].toarray()
			basis_idx = int(np.floor((nbel+degree)/2))
			basis_toplot = basis[basis_idx, :]
			index_toplot = np.where(basis_toplot > 0)[0]
			knots_toplot = knots[index_toplot[0]-1:index_toplot[-1]+2]
			basis_toplot = basis_toplot[index_toplot[0]-1:index_toplot[-1]+2]

			ax.plot(knots_toplot, basis_toplot, color=color, linewidth=2)
			ax.fill_between(x=knots, y1=basis[basis_idx, :], color=color, alpha=0.2, label=f'{label} basis')

		ax.plot(quadrature._unique_kv, np.zeros_like(quadrature._unique_kv), 
				color='k', marker='s', linestyle='None', markersize=3, label='Knots')
		
		ax.grid(None)
		ax.axis('off')
		ax.legend()
		fig.tight_layout()
		fig.savefig(filename, dpi=300)
		return

	case7(FOLDER2RESU, EXTENSION)


else: raise Warning('Case unkwnon')
