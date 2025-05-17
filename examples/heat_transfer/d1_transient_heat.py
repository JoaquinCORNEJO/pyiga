from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem

def power_density(args:dict):
	t = args['time']
	x = args['position']
	f = ((np.pi*np.cos((np.pi*t)/2)*np.sin(2*np.pi*x))/2
		+ 8*np.pi**2*np.sin((np.pi*t)/2)*np.sin(2*np.pi*x))
	return f

# Create geometry
degree, nbel, length = 6, 64, 1.0 
geometry = mygeomdl({'name':'line', 'degree': degree, 'nbel': nbel, 'geo_parameters':{'L': length}}).export_geometry()
patch = singlepatch(geometry, quad_args={'quadrule': 'wq'})

# Create material
material = heat_transfer_mat()
material.add_conductivity(1.0, is_uniform=True, shape_tensor=1)
material.add_capacity(1.0, is_uniform=True)

# Create boundary condition
boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nbvars=1)
boundary.add_constraint(location_list=[{'direction':'x', 'face':'both'}], constraint_type='dirichlet')

# Set transient heat transfer problem
problem = heat_transfer_problem(material, patch, boundary)

# Create external force	
time_list = np.linspace(0, 1, 65)
external_heat_source = np.zeros((patch.nbctrlpts_total, len(time_list)))
for i, t in enumerate(time_list):
	external_heat_source[:, i] = problem.assemble_volumetric_force(power_density, args={'time':t})

# Solve problem
temperature = np.zeros_like(external_heat_source)
# temperature[-1, :] = 10 # Add if we impose a constant temperature
problem.solve_heat_transfer(temperature, external_heat_source, time_list, alpha=1)

# Post-processing
from src.lib_tensor_maths import bspline_operations
fig, ax = plt.subplots(figsize=(8, 4))
knots_interp = np.linspace(0, 1, 101)
temperature_interp = bspline_operations.interpolate_meshgrid(quadrule_list=patch.quadrule_list, knots_list=[knots_interp], u_ctrlpts=temperature.T)
POSITION, TIME = np.meshgrid(knots_interp*length, time_list)
im = ax.contourf(POSITION, TIME, temperature_interp, 20, cmap='viridis')
cbar = plt.colorbar(im)
cbar.set_label('Temperature (°C)')

ax.grid(False)
ax.set_ylabel('Time (s)')
ax.set_xlabel('Position (m)')
fig.tight_layout()
fig.savefig(RESULT_FOLDER + 'transient_heat')