from thesis.Incremental.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch, vtk2png
from src.lib_material import J2plasticity3d
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_mechanical import mechanical_problem
import pickle

TRACTION = 200.0
YOUNG, POISSON = 2500, 0.25
NBSTEPS = 101
TIME_LIST = np.linspace(0, np.pi/2, NBSTEPS)

def surface_force(args:dict):
	position = args['position']
	x = position[0, :]
	for_tmp = np.zeros_like(position)
	for_tmp[1, :] = x**2-1/4
	force = np.zeros_like(position)
	force[1, :] = -TRACTION*(np.min(for_tmp, axis=0))**2
	return force

def simulate_el(degree, cuts, quad_args):
	geo_args = {'name':'SQ', 'degree': degree*np.ones(3, dtype=int), 'nbel': np.array([int(2**cuts) for _ in range(3)], dtype=int),
				'extra':{'XY':np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])}
			}
	material = J2plasticity3d({'elastic_modulus':YOUNG, 'elastic_limit':5, 'poisson_ratio':POISSON,
			'iso_hardening': {'name':'linear', 'Eiso':0.0},
			'kine_hardening':{'parameters':np.array([[500, 0]])}
			})
	geometry = mygeomdl(geo_args).export_geometry()
	patch = singlepatch(geometry, quad_args=quad_args)

	# Set Dirichlet boundaries
	boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nbvars=2)
	boundary.add_constraint(location_list=[{'direction':'x', 'face':'left'},
									{'direction':'y', 'face':'bottom'}],
							constraint_type='dirichlet')

	# Solve elastic problem
	problem = mechanical_problem(material, patch, boundary)
	time_list = np.linspace(0, 1.0, NBSTEPS)
	force_ref = problem.assemble_surface_force(surface_force, location={'direction':'y', 'face':'top'})
	external_force = np.tensordot(force_ref, time_list / time_list[-1], axes=0)
	displacement = np.zeros_like(external_force)
	internal_vars = problem.solve_elastoplasticity(displacement, external_force, save_plastic_vars=True)
	return problem, displacement, internal_vars

# Set global variables
deg_list = np.arange(1, 4)
cut_list = np.arange(1, 7)
selected_steps_list = np.arange(10, NBSTEPS, 4)
RUNSIMU = False

if RUNSIMU:
	degree, cuts = 2, 6
	quad_args = {'quadrule': 'wq', 'type': 2}
	problem, displacement, internal_vars = simulate_el(degree, cuts, quad_args)

	for k, step in enumerate(selected_steps_list):
		plastic_eq_quadpts = internal_vars[step-1].get('plastic_equivalent', None)
		zone_plastified = np.where(np.abs(plastic_eq_quadpts)<1e-8, 0.0, 1.0)
		problem.part.postprocessing_dual(name=f"out_{k}", folder=FOLDER2RESU,
										fields={'plastic_equivalent': plastic_eq_quadpts,
												'zone_plastified': zone_plastified})

	# np.save(f"{FOLDER2DATA}displacement_pl2d.npy", displacement)
	# with open(f"{FOLDER2DATA}refpart_pl2d.pkl", 'wb') as outp:
	# 	pickle.dump(problem.part, outp, pickle.HIGHEST_PROTOCOL)

	with open(f"{FOLDER2DATA}refpart_pl2d.pkl", 'rb') as inp:
		part_ref = pickle.load(inp)
	disp_ref = np.load(f"{FOLDER2DATA}displacement_pl2d.npy")

	for quadrule, quadtype in zip(['gs'], ['leg']):
		quad_args = {'quadrule':quadrule, 'type': quadtype}
		errorL2_list = np.ones((len(selected_steps_list), len(deg_list), len(cut_list)))
		errorH1_list = np.ones((len(selected_steps_list), len(deg_list), len(cut_list)))

		for i, degree in enumerate(deg_list):
			for j, cuts in enumerate(cut_list):
				problem, displacement = simulate_el(degree, cuts, quad_args)[:2]

				for k, step in enumerate(selected_steps_list):
					errorL2_list[k, i, j], _ = problem.norm_of_error(displacement[:, :, step],
															norm_args={'type':'L2',
															'part_ref':part_ref,
															'u_ref': disp_ref[:, :, step]})

					errorH1_list[k, i, j], _ = problem.norm_of_error(displacement[:, :, step],
															norm_args={'type':'H1',
															'part_ref':part_ref,
															'u_ref': disp_ref[:, :, step]})

				np.save(f"{FOLDER2DATA}abserror_pls2d_L2_{quadrule}_{str(quadtype)}.npy", errorL2_list)
				np.save(f"{FOLDER2DATA}abserror_pls2d_H1_{quadrule}_{str(quadtype)}.npy", errorH1_list)

# import imageio
# images = []
# for i in range(12):
# 	filepath = vtk2png(folder=FOLDER2RESU, filename=f'out_{i}', fieldname='zone_plastified', title='Plastic zone', n_colors=2, n_labels=2, camera_position='xy')
# 	images.append(imageio.imread(filepath))
# images = images[-1::-1] + images
# imageio.mimsave(f"{FOLDER2RESU}plastic.gif", images, duration=0.4)

from mpltools import annotation
nbel_list = 2**cut_list
FOLDER2RESU += '/pls2d/'
if not os.path.isdir(FOLDER2RESU): os.mkdir(FOLDER2RESU)

for k, step in enumerate(selected_steps_list):
	for error_name in ['H1']:
		fig, ax = plt.subplots(figsize=(5, 5))

		for quadrule, quadtype, plotopt in zip(['gs', 'wq'], ['leg', 2], [CONFIGLINE_IGA, CONFIGLINE_WQ]):
			error_list = np.load(f"{FOLDER2DATA}abserror_pls2d_{error_name}_{quadrule}_{str(quadtype)}.npy")

			for i, degree in enumerate(deg_list):
				color = COLORLIST[i]
				if quadrule == 'gs':
					ax.loglog(nbel_list, error_list[k, i, :], label=fr'Gauss quad. $p={degree}$', color=color, **plotopt)
					slope = round(np.polyfit(np.log(nbel_list[-3:]), np.log(error_list[k, i, -3:]), 1)[0], 1)
					annotation.slope_marker((nbel_list[-2],  error_list[k, i, -2]), slope,
									poly_kwargs={'facecolor': (0.73, 0.8, 1)}, ax=ax)
				else:
					ax.loglog(nbel_list, error_list[k, i, :], color=color, **plotopt)

		if error_name == 'H1':
			ax.set_ylabel(r'$H^1(\Omega)$ error')
			ax.set_ylim(bottom=1e-8, top=1e-1)
		if error_name == 'L2':
			ax.set_ylabel(r'$L^2(\Omega)$ error')
			ax.set_ylim(bottom=1e-9, top=1e-2)

		ax.set_xlabel('Number of elements by direction')
		ax.set_xlim(left=1, right=10**2)

		ax.semilogy([], [], color='k', **CONFIGLINE_WQ, label='Weighted quad.')

		ax.legend()
		fig.tight_layout()
		fig.savefig(f"{FOLDER2RESU}ConverPls2d_{error_name}_{k}.pdf")
		plt.close(fig)
