from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem, st_heat_transfer_problem
from numpy import sin, cos, exp, pi
from time import process_time

def exact_temperature_sptm(args:dict):
	time = args['time']
	x = args['position'][0]
	u = np.zeros((len(x), len(time)))
	for i, t in enumerate(time):
		u[:, i] = sin(pi*x)*t*sin(t)*exp(-t)
	return np.ravel(u, order='F')

def power_density_inc(args:dict):
	t = args['time']
	x = args['position']
	f = exp(-t)*(-t*sin(t) + sin(t) + t*cos(t))*sin(pi*x) + pi**2*sin(pi*x)*t*sin(t)*exp(-t)
	return f

def power_density_sptm(args:dict):
	time = args['time']
	x = args['position'][0]
	f = np.zeros((len(x), len(time)))
	for i, t in enumerate(time):
		f[:, i] = exp(-t)*(-t*sin(t) + sin(t) + t*cos(t))*sin(pi*x) + pi**2*sin(pi*x)*t*sin(t)*exp(-t)
	return np.ravel(f, order='F')

# Create geometry
degree, nbel, length = 1, 128, 1.0 
geometry = mygeomdl({'name':'line', 'degree':degree, 'nbel':nbel, 'geo_parameters':{'L': length}}).export_geometry()
space_patch = singlepatch(geometry, quad_args={'quadrule': 'gs'})

# Create material
material = heat_transfer_mat()
material.add_conductivity(1.0, is_uniform=True, shape_tensor=1)
material.add_capacity(1.0, is_uniform=True)

# Create boundary condition
boundary = boundary_condition(nbctrlpts=space_patch.nbctrlpts, nbvars=1)
boundary.add_constraint(location_list=[{'direction':'x', 'face':'both'}], constraint_type='dirichlet')

# SPACE TIME METHOD
start = process_time()
time_interval = mygeomdl({'name':'line', 'degree':1, 'nbel':nbel, 'geo_parameters':{'L': 1.}}).export_geometry()
time_patch = singlepatch(time_interval, quad_args={'quadrule': 'gs'})
problem_sptm = st_heat_transfer_problem(material, space_patch, time_patch, boundary)
external_heat_source_sptm = problem_sptm.assemble_volumetric_force(power_density_sptm)

temperature_sptm = np.zeros_like(external_heat_source_sptm)
problem_sptm.solve_heat_transfer(temperature_sptm, external_heat_source_sptm, auto_inner_tolerance=False)
finish = process_time()
time_elapsed_sptm = finish - start
rel_error_sptm = problem_sptm.norm_of_error(temperature_sptm, norm_args={'type': 'L2', 'exact_function': exact_temperature_sptm})[-1]

# INCREMENTAL METHOD
start = process_time()
time_list = np.linspace(0, 1, nbel+1)
problem_inc = heat_transfer_problem(material, space_patch, boundary)
external_heat_source_inc = np.zeros((space_patch.nbctrlpts_total, len(time_list)))
for i, t in enumerate(time_list):
	external_heat_source_inc[:, i] = problem_inc.assemble_volumetric_force(power_density_inc, args={'time':t})

temperature_inc = np.zeros_like(external_heat_source_inc)
problem_inc.solve_heat_transfer(temperature_inc, external_heat_source_inc, time_list, alpha=0.5)
finish = process_time()
time_elapsed_inc = finish - start
rel_error_inc = problem_sptm.norm_of_error(temperature_inc.ravel('F'), norm_args={'type': 'L2', 'exact_function': exact_temperature_sptm})[-1]

print(f'For space-time method: error {rel_error_sptm:.2e} and time {time_elapsed_sptm:.2f}')
print(f'Forr incremental method: error {rel_error_inc:.2e} and time {time_elapsed_inc:.2f}')

# Post-processing
from src.lib_tensor_maths import bspline_operations
fig, ax = plt.subplots(figsize=(8, 4))
knots_interp = np.linspace(0, 1, 101)
POSITION, TIME = np.meshgrid(knots_interp*length, time_list)

diff_temp = bspline_operations.interpolate_meshgrid(quadrule_list=space_patch.quadrule_list, knots_list=[knots_interp], 
								u_ctrlpts=temperature_inc.T-np.reshape(temperature_sptm, newshape=(-1, len(time_list)), order='F').T,
								)	

im = ax.contourf(POSITION, TIME, abs(diff_temp), 5, cmap='viridis')
cbar = plt.colorbar(im)
cbar.set_label('Temperature (°C)')

ax.grid(False)
ax.set_ylabel('Time (s)')
ax.set_xlabel('Position (m)')
fig.tight_layout()
fig.savefig(RESULT_FOLDER + 'diff_space_time_incremental')