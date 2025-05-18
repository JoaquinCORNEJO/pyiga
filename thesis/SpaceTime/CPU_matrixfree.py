from thesis.SpaceTime.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import st_heat_transfer_problem
import time

# Set global variables
RUNSIMU = False
nbel = 16
deg_list = range(1, 10)

if RUNSIMU:

	timeMF_mass = np.zeros((len(deg_list), 2))
	timeMF_mass[:, 0] = deg_list
	timeMF_stiff = np.zeros((len(deg_list), 2))
	timeMF_stiff[:, 0] = deg_list
	timeMF_mass_ders = np.zeros((len(deg_list), 2))
	timeMF_mass_ders[:, 0] = deg_list
	timeMF_stiff_ders = np.zeros((len(deg_list), 2))
	timeMF_stiff_ders[:, 0] = deg_list

	for quadrule, quadtype in zip(['gs', 'wq'], ['leg', 2]):
		quad_args = {'quadrule':quadrule, 'type':quadtype}

		for i, degree in enumerate(deg_list):

			# Create model
			geometry = mygeomdl({'name':'VB', 'degree':degree, 'nbel':nbel}).export_geometry()
			space_patch = singlepatch(geometry, quad_args=quad_args)

			# Create time span
			time_interval = mygeomdl({'name':'line', 'degree':degree, 'nbel':nbel}).export_geometry()
			time_patch = singlepatch(time_interval, quad_args=quad_args)

			# Add material
			material = heat_transfer_mat()
			material.add_capacity(1, is_uniform=True)
			material.add_conductivity(np.array([[1., 0.5, 0], [0.5, 2., 0], [0, 0, 1]]), is_uniform=True, shape_tensor=3)
			material.add_ders_capacity(1, is_uniform=True)
			material.add_ders_conductivity(np.array([[1., 0.5, 0], [0.5, 2., 0], [0, 0, 1]]), is_uniform=True, shape_tensor=3)

			# Block boundaries
			boundary = boundary_condition(nbctrlpts=space_patch.nbctrlpts, nbvars=1)
			boundary.add_constraint(location_list=[{'direction':'x,y', 'face':'both,both'}], constraint_type='dirichlet')

			# Define space time problem
			problem = st_heat_transfer_problem(material, space_patch, time_patch, boundary)
			array_in = np.random.random(space_patch.nbctrlpts_total*time_patch.nbctrlpts_total)

			# ------------------
			# Compute MF product
			# ------------------
			if quadrule != 'gs' or degree < 7:
				print('******')
				start = time.process_time()
				problem.compute_mf_sptm_capacity(array_in)
				finish = time.process_time()
				print('Time Capacity:%.2e' %(finish-start))
				timeMF_mass[i, 1] = finish - start

				start = time.process_time()
				problem.compute_mf_sptm_conductivity(array_in)
				finish = time.process_time()
				print('Time Conductivity:%.2e' %(finish-start))
				timeMF_stiff[i, 1] = finish - start

				start = time.process_time()
				problem.compute_mf_sptm_ders_capacity(array_in)
				finish = time.process_time()
				print('Time Ders Capacity:%.2e' %(finish-start))
				timeMF_mass_ders[i, 1] = finish - start

				start = time.process_time()
				problem.compute_mf_sptm_ders_conductivity(array_in)
				finish = time.process_time()
				print('Time Ders Conductivity:%.2e' %(finish-start))
				timeMF_stiff_ders[i, 1] = finish - start

			np.savetxt(f"{FOLDER2DATA}sptMF_Mass_{quadrule}_{str(quadtype)}.dat", timeMF_mass)
			np.savetxt(f"{FOLDER2DATA}sptMF_Stiff_{quadrule}_{str(quadtype)}.dat", timeMF_stiff)
			np.savetxt(f"{FOLDER2DATA}sptMF_MassDers_{quadrule}_{str(quadtype)}.dat", timeMF_mass_ders)
			np.savetxt(f"{FOLDER2DATA}sptMF_StiffDers_{quadrule}_{str(quadtype)}.dat", timeMF_stiff_ders)

fig, ax = plt.subplots(figsize=(5, 5.5))
plotoptions = [CONFIGLINE_IGA, CONFIGLINE_WQ]
sufixList = ['gs_leg', 'wq_2']
label_list = ['Newton', 'Picard']

# Load data
for sufix, plotops in zip(sufixList, plotoptions):
	file_M1 = np.loadtxt(FOLDER2DATA+'sptMF_Mass_'+sufix+'.dat')
	file_K1 = np.loadtxt(FOLDER2DATA+'sptMF_Stiff_'+sufix+'.dat')
	file_M2 = np.loadtxt(FOLDER2DATA+'sptMF_MassDers_'+sufix+'.dat')
	file_K2 = np.loadtxt(FOLDER2DATA+'sptMF_StiffDers_'+sufix+'.dat')

	deg_list = file_M1[:, 0]
	time_list = [file_M1[:, 1]+file_K1[:, 1]+file_M2[:, 1]+file_K2[:, 1], file_M1[:, 1]+file_K1[:, 1]]
	quadrule = sufix.split('_')[0]

	for i, [time_elapsed, label] in enumerate(zip(time_list, label_list)):
		color = COLORLIST[i]
		if quadrule == 'gs':
			ax.semilogy(deg_list, time_elapsed, label=f'MF-GL {label}', color=color, **plotops)
		else:
			ax.semilogy(deg_list, time_elapsed, color=color, **plotops)

ax.semilogy([], [], color='k', **CONFIGLINE_WQ, label='MF-WQ')

ax.minorticks_off()
ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.2), loc='upper center')
ax.set_xlabel('Degree')
ax.set_ylabel('CPU time (s) of matrix-vector product')
ax.set_xlim([0, 10])
ax.set_ylim([1e-1, 1e3])
fig.tight_layout()
fig.savefig(f"{FOLDER2RESU}sptMF_time.pdf")
