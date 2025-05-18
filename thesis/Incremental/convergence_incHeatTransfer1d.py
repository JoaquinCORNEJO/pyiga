from thesis.Incremental.__init__ import *
from src.lib_tensor_maths import bspline_operations
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import heat_transfer_problem
from numpy import sin, cos, pi, tanh

CST = 100

def conductivity_property(args:dict):
	temperature = args.get('temperature')
	conductivity = np.zeros(shape=(1, 1, *np.shape(temperature)))
	conductivity[0, 0, ...] = 3.0 + 2.0*tanh(temperature/50)
	return conductivity

def exact_temperature(args:dict):
	t = args['time']
	x = args['position']
	u = CST*sin(2*pi*x)*sin(pi/2*t)*(1+0.75*cos(3*pi/2*t))
	return u

def power_density(args:dict):
	t = args['time']
	x = args['position']
	u = CST*sin(2*pi*x)*sin(pi/2*t)*(1+0.75*cos(3*pi/2*t))
	f = (
		(CST*pi*cos((pi*t)/2)*sin(2*pi*x)*((3*cos((3*pi*t)/2))/4 + 1))/2
		- (9*CST*pi*sin((pi*t)/2)*sin((3*pi*t)/2)*sin(2*pi*x))/8
		+ 4*pi**2*u*(2*tanh(u/50) + 3)
		+ (4*CST**2*pi**2*cos(2*pi*x)**2*sin((pi*t)/2)**2*((3*cos((3*pi*t)/2))/4 + 1)**2*(tanh(u/50)**2 - 1))/25
		)
	return f

def simulate_ht(degree, cuts, nbel_time, quad_args, ivp='alpha', alpha=0.5):

	# Create geometry
	geometry = mygeomdl({'name':'line', 'degree':degree, 'nbel':int(2**cuts), 'geo_parameters':{'L':1.}}).export_geometry()
	patch = singlepatch(geometry, quad_args=quad_args)
	time_inc = np.linspace(0.0, 1.0, nbel_time+1)

	# Add material
	material = heat_transfer_mat()
	material.add_capacity(1.0, is_uniform=True)
	material.add_conductivity(conductivity_property, is_uniform=False, shape_tensor=1)

	# Block boundaries
	boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nbvars=1)
	boundary.add_constraint(location_list=[{'direction':'x', 'face':'both'}], constraint_type='dirichlet')

	# Transient model
	problem = heat_transfer_problem(material, patch, boundary)
	temperature = np.zeros((problem.part.nbctrlpts_total, len(time_inc)))

	if ivp == 'alpha':
		# Create external force
		external_heat_source = np.zeros_like(temperature)
		for i, t in enumerate(time_inc):
			external_heat_source[:, i] = problem.assemble_volumetric_force(power_density, args={'time':t})

		# Solve problem
		problem._tolerance_nonlinear = 1e-8
		problem.solve_heat_transfer(temperature, external_heat_source, time_inc, alpha=alpha)

	else:
		assert ivp in ['BDF1', 'BDF2', 'BDF3', 'BDF4'], "Invalid method"
		free_ctrlpts = problem.sp_free_ctrlpts[0]
		args = problem._verify_fun_args({"temperature": np.ones(problem.part.nbqp_total)})
		prop = problem.material.capacity(args)*problem.part.det_jac
		capacity = bspline_operations.assemble_scalar_u_v(problem.part.quadrule_list, prop, allow_lumping=False)
		inv_capacity = np.linalg.pinv(capacity.toarray()[np.ix_(free_ctrlpts, free_ctrlpts)])

		def fun(tm, y_input):
			y_extend = np.zeros(problem.part.nbctrlpts_total); y_extend[free_ctrlpts] = y_input
			y_interp = problem.interpolate_temperature(y_extend)
			args = problem._verify_fun_args({'time':tm, 'temperature':y_interp})
			b = problem.assemble_volumetric_force(power_density, args=args)[free_ctrlpts]
			problem.clear_properties()
			Au = problem.compute_mf_conductivity(y_extend, args=args)[free_ctrlpts]
			y_output = inv_capacity @ (b - Au)
			return y_output

		t_span = (time_inc[0], time_inc[-1])
		y0 = np.zeros(len(free_ctrlpts))
		yt = bdf(fun, t_span, y0, nbel_time, norder=int(ivp[-1]))[-1]
		temperature[free_ctrlpts, :] = np.copy(yt)

	return problem, time_inc, temperature

# Set global variables
PLOTRELATIVE = True
RUNSIMU = False

degree, cuts = 8, 7
quad_args = {'quadrule':'gs', 'type':'leg'}
IVP_method_list = ['BDF2', 'BDF3']
nbel_time_list = np.array([2**cuts for cuts in range(2, 9)])

if RUNSIMU:

	for IVP_method in IVP_method_list:

		abserror_list, relerror_list = np.ones(len(nbel_time_list)), np.ones(len(nbel_time_list))

		for i, nbel_time in enumerate(nbel_time_list):

			problem_inc, time_inc, temp_inc = simulate_ht(degree, cuts,
														nbel_time=nbel_time,
														quad_args=quad_args,
														ivp=IVP_method)

			abserror_list[i], relerror_list[i] = problem_inc.norm_of_error(
														u_ctrlpts=temp_inc[:, -1],
														norm_args={'type':'L2',
														'exact_function':exact_temperature,
														'exact_args':{'time':time_inc[-1]}})

			np.savetxt(f"{FOLDER2DATA}abserrorstag_inc_{IVP_method}.dat", abserror_list)
			np.savetxt(f"{FOLDER2DATA}relerrorstag_inc_{IVP_method}.dat", relerror_list)

from mpltools import annotation
IVP_method_list = ['BDF1', 'BDF2', 'BDF3', 'alpha']
fig, ax = plt.subplots(figsize=(5, 5))
for i, IVP_method in enumerate(IVP_method_list):
	label = 'Crank-Nicolson' if IVP_method == 'alpha' else f'BDF {IVP_method[-1]}'
	if PLOTRELATIVE: error_list = np.loadtxt(f"{FOLDER2DATA}relerrorstag_inc_{IVP_method}.dat")
	else: error_list = np.loadtxt(f"{FOLDER2DATA}abserrorstag_inc_{IVP_method}.dat")
	nbctrlpts = nbel_time_list+1
	if i < 3:
		ax.loglog(nbctrlpts, error_list, **CONFIGLINE_BDF,  label=label)
	elif i == 3:
		ax.loglog(nbctrlpts, error_list, **CONFIGLINE_INC, color='k', label=label)

	slope = np.polyfit(np.log10(nbctrlpts[3:]),np.log10(error_list[3:]), 1)[0]
	slope = round(slope, 1)
	annotation.slope_marker((nbctrlpts[-2], error_list[-2]), slope,
					poly_kwargs={'facecolor': (0.73, 0.8, 1)}, ax=ax)

if PLOTRELATIVE:
	ax.set_ylabel(r'Relative $L^2(\Omega)$ error at last time-step')
	ax.set_ylim(top=1e-1, bottom=1e-8)
else:
	ax.set_ylabel(r'$L^2(\Omega)$ error at last time-step')
	ax.set_ylim(top=1e0, bottom=1e-7)

ax.set_xlabel('Number of time-steps')
ax.set_xlim(left=2, right=600)
# ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.2), loc='upper center')
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(f"{FOLDER2RESU}stagnation_error_inc.pdf")
