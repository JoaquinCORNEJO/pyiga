from thesis.Elliptic.__init__ import *
from src.lib_mygeometry import create_uniform_knotvector
from src.lib_quadrules import weighted_quadrature
from src.lib_tensor_maths import fastdiagonalization
import time

# Set global variables
NDIM = 3
RUNSIMU = False
deg_list = np.arange(1, 6)
cut_list = np.arange(5, 9)

if RUNSIMU:

	time_list = np.zeros((len(cut_list)+1, len(deg_list)+1))
	time_list[1:, 0] = np.array([2**i for i in cut_list])
	time_list[0, 1:] = deg_list

	for TYPESIMU in ['heat', 'meca']:

		NBVARS = 1 if TYPESIMU == 'heat' else NDIM
		filename = f"preconditioner_{TYPESIMU}"

		for i, cuts in enumerate(cut_list):
			for j, degree in enumerate(deg_list):
				nbel = int(time_list[i+1, 0])
				nctrlpts = degree + nbel
				knotvector = create_uniform_knotvector(degree, nbel)

				quadrature = weighted_quadrature(degree, knotvector, quad_args={})
				quadrature.export_quadrature_rules()

				boundary = boundary_condition(nbctrlpts=np.array([nctrlpts]*NDIM), nbvars=NBVARS)
				boundary.add_constraint(location_list=[{'direction':','.join(str(i) for i in range(NDIM)),
														'face':','.join('both' for i in range(NDIM))}
														for _ in range(NBVARS)],
														constraint_type='dirichlet')

				fastdiag = fastdiagonalization()
				fastdiag.compute_space_eigendecomposition([quadrature]*NDIM, boundary.table_dirichlet)
				fastdiag.update_space_eigenvalues(scalar_coefs=np.ones(NDIM))
				fastdiag.add_free_controlpoints(boundary.select_nodes4solving()[0])

				start = time.process_time()
				if TYPESIMU == 'heat':
					array_in = np.random.random(nctrlpts**NDIM)
					fastdiag.apply_scalar_preconditioner(array_in)
				else:
					array_in = np.random.random((NDIM, nctrlpts**NDIM))
					fastdiag.apply_vectorial_preconditioner(array_in)
				finish = time.process_time()
				time_elapsed = finish - start

				print('For p = %s, nbel = %s, time: %.4f' %(degree, nbel, time_elapsed))
				time_list[i+1, j+1] = time_elapsed
				np.savetxt(f"{FOLDER2DATA}{filename}.dat", time_list)

from mpltools import annotation
cmap = mpl.colors.ListedColormap(COLORLIST[:len(deg_list)])
for i, TYPESIMU in enumerate(['heat', 'meca']):

	fig, ax = plt.subplots(figsize=(5.5, 4.5))
	filename = f"preconditioner_{TYPESIMU}"
	ax.grid(True, zorder=1)

	# Make dummie mappable
	c = np.arange(1, len(deg_list)+1, dtype=int)
	dummie_cax = ax.scatter(c, c, c=c, cmap=cmap)
	cbar = plt.colorbar(dummie_cax)
	ax.cla()

	file = np.loadtxt(f"{FOLDER2DATA}{filename}.dat")
	deg_list = file[0, 1:]; nbel_list = file[1:, 0]; time_elapsed = file[1:, 1:]
	for _, degree in enumerate(deg_list):
		im = ax.loglog(nbel_list**3, time_elapsed[:, _], **CONFIGLINE_WQ)

	slope = round(np.polyfit(np.log(nbel_list[1:]**3), np.log(time_elapsed[1:, 2]), 1)[0], 1)
	annotation.slope_marker((nbel_list[-2]**3,  time_elapsed[-2, 2]), slope,
					poly_kwargs={'facecolor': (0.73, 0.8, 1)}, ax=ax)

	cbar.set_label('Degree')
	tick_locs = 1+(np.arange(len(deg_list)) + 0.5)*(len(deg_list)-1)/len(deg_list)
	cbar.set_ticks(tick_locs)
	cbar.set_ticklabels(np.array(deg_list, dtype=int))

	ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
	ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

	if NDIM == 2:
		pass
	if NDIM == 3:
		ax.set_xticks([1e4, 32**3, 64**3, 128**3, 256**3, 1e8])
		ax.set_xticklabels([r'$10^3$', r'$32^3$', r'$64^3$', r'$128^3$', r'$256^3$', r'$10^8$'])

	ax.minorticks_off()
	ax.set_xlabel('Total number of elements')
	ax.set_ylabel('CPU time (s)')
	ax.set_ylim([1e-3, 1e2])
	fig.tight_layout()
	fig.savefig(f"{FOLDER2RESU}{filename}.pdf")

TYPESIMU = 'meca'
filename = f"preconditioner_{TYPESIMU}"
fig, ax = plt.subplots(figsize=(5.5,4.5))
ax.grid(True, zorder=1)
xx = np.array([2**i for i in cut_list])**3
yy = 4e-8*xx**(1+1/(NDIM+1))

# Make dummie mappable
c = np.arange(1,len(deg_list)+1, dtype=int)
dummie_cax = ax.scatter(c, c, c=c, cmap=cmap)
cbar = plt.colorbar(dummie_cax)
ax.cla()

file = np.loadtxt(f"{FOLDER2DATA}{filename}.dat")
deg_list = file[0, 1:]; nbel_list = file[1:, 0]; time_elapsed = file[1:, 1:]
for _, degree in enumerate(deg_list):
	color=COLORLIST[_]
	im = ax.loglog(nbel_list**3, time_elapsed[:, _], color=color, **CONFIGLINE_WQ)

slope = round(np.polyfit(np.log(nbel_list[1:]**3), np.log(time_elapsed[1:, 2]), 1)[0], 1)
annotation.slope_marker((nbel_list[-2]**3,  time_elapsed[-2, 2]), slope,
				poly_kwargs={'facecolor': (0.73, 0.8, 1)}, ax=ax)
ax.loglog(xx, yy, '--', c='tab:gray', label='$O(N^{1.25})$')

cbar.set_label('Degree')
tick_locs = 1+(np.arange(len(deg_list)) + 0.5)*(len(deg_list)-1)/len(deg_list)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(np.array(deg_list, dtype=int))

ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

xpos = 64**3
ypos = 1.286392219599997588e+01
ax.scatter(xpos, ypos, marker='o', color='k', s=80, zorder=2)
ax.text(
		xpos, 1.5*ypos,
		f'Time of matrix-free',
		fontsize=12,
		ha='center', va='bottom',
		bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='gray')
	)

if NDIM == 2:
	pass
if NDIM == 3:
	ax.set_xticks([1e4, 32**3, 64**3, 128**3, 256**3, 1e8])
	ax.set_xticklabels([r'$10^3$', r'$32^3$', r'$64^3$', r'$128^3$', r'$256^3$', r'$10^8$'])

ax.set_xlabel('Total number of elements')
ax.set_ylabel('CPU time (s) of applying $\mathsf{P}^{-1}$')
ax.set_ylim([1e-3, 1e2])
ax.legend()
fig.tight_layout()
fig.savefig(f"{FOLDER2RESU}{filename}_2.pdf")
