from thesis.SpaceTime.__init__ import *
from src.lib_mygeometry import create_uniform_knotvector
from src.lib_quadrules import weighted_quadrature
from src.lib_boundary import boundary_condition
from src.lib_tensor_maths import fastdiagonalization
import time

# Set global variables
NDIM = 3
RUNSIMU = False
deg_list = np.arange(1, 6)
cut_list = np.arange(4, 8)
filename = 'sptFD_time'

if RUNSIMU:

	time_list = np.zeros((len(cut_list)+1, len(deg_list)+1))
	time_list[1:, 0] = np.array([2**i for i in cut_list])
	time_list[0, 1:] = deg_list

	for i, cuts in enumerate(cut_list):
		for j, degree in enumerate(deg_list):
			nbel = int(time_list[i+1, 0])
			nctrlpts = degree + nbel
			knotvector = create_uniform_knotvector(degree, nbel)
			quadrature = weighted_quadrature(degree, knotvector, quad_args={})
			quadrature.export_quadrature_rules()

			boundary_sp = boundary_condition(nbctrlpts=np.array([nctrlpts]*NDIM), nbvars=1)
			boundary_sp.add_constraint(location_list=[{'direction':','.join(str(i) for i in range(NDIM)),
													'face':','.join('both' for i in range(NDIM))}],
													constraint_type='dirichlet')

			boundary_sptm = boundary_condition(nbctrlpts=np.array([nctrlpts]*(NDIM+1)), nbvars=1)
			boundary_sptm.add_constraint(location_list=[{'direction':','.join(str(i) for i in range(NDIM+1)),
													'face':','.join('both' if i != NDIM else 'left'
																	for i in range(NDIM+1))}],
													constraint_type='dirichlet')
			free_nodes = boundary_sptm.select_nodes4solving()[0]

			if cuts != cut_list[-1] or degree < 2:
				fastdiag = fastdiagonalization()
				fastdiag.compute_space_eigendecomposition([quadrature]*NDIM, boundary_sp.table_dirichlet)
				fastdiag.update_space_eigenvalues(scalar_coefs=np.ones(NDIM))
				fastdiag.compute_time_schurdecomposition(quadrature)
				fastdiag.add_free_controlpoints(free_nodes)

				start = time.process_time()
				array_in = np.random.random(nctrlpts**(NDIM+1))
				fastdiag.apply_spacetime_scalar_preconditioner(array_in)
				finish = time.process_time()
				time_elapsed = finish - start

			print('For p = %s, nbel = %s, time: %.4f' %(degree, nbel, time_elapsed))
			time_list[i+1, j+1] = time_elapsed
			np.savetxt(f"{FOLDER2DATA}{filename}.dat", time_list)

from mpltools import annotation
# Make dummie mappable
fig, ax = plt.subplots(figsize=(5.5, 5))
cmap = mpl.colors.ListedColormap(COLORLIST[:len(deg_list)])
c = np.arange(1,len(deg_list)+1, dtype=int)
dummie_cax = ax.scatter(c, c, c=c, cmap=cmap)
cbar = plt.colorbar(dummie_cax)
ax.cla()

xx = (2**cut_list)**4
yy = 10e-8*xx**(1.20)

# Load data
file = np.loadtxt(f"{FOLDER2DATA}{filename}.dat")
deg_list = file[0, 1:]; nbelList = file[1:, 0]; time_elapsed = file[1:, 1:]

for i, degree in enumerate(deg_list):
	ax.loglog(nbelList**4, time_elapsed[:, i], **CONFIGLINE_WQ)

ax.loglog(xx, yy, '--', c='tab:gray', label=r'$O(N^{1.2})$')
slope = round(np.polyfit(np.log(nbelList**4), np.log(time_elapsed[:, 2]), 1)[0], 1)
annotation.slope_marker((nbelList[-2]**4,  time_elapsed[-2, 2]), slope,
				poly_kwargs={'facecolor': (0.73, 0.8, 1)}, ax=ax)

cbar.set_label('Degree '+r'$p_s=p_t$')
tick_locs = 1+(np.arange(len(deg_list)) + 0.5)*(len(deg_list)-1)/len(deg_list)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(np.array(deg_list, dtype=int))

ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

if NDIM == 2:
	pass
if NDIM == 3:
	ax.set_xticks([1e4, 16**4, 32**4, 64**4, 128**4, 1e9])
	ax.set_xticklabels([r'$10^4$', r'$16^4$', r'$32^4$', r'$64^4$', r'$128^4$', r'$10^9$'])

ax.minorticks_off()
ax.legend()
ax.set_xlabel('Total number of elements')
ax.set_ylabel('CPU time (s) of applying $\mathsf{P}^{-1}$')
ax.set_ylim([1e-2, 1e3])
ax.set_xlim([1e4, 1e9])
fig.tight_layout()
fig.savefig(f"{FOLDER2RESU}{filename}.pdf")
