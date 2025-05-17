from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import heat_transfer_mat
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_heat_transfer import st_heat_transfer_problem, heat_transfer_problem
from numpy import sin, cos, tanh, pi

ISLINEAR = False
CUTS_TIME = 6
CST = 50

def nonlinearfunc(args:dict):
	temperature = args.get('temperature')
	if ISLINEAR:
		Kprop1d = 2*np.ones(shape=np.shape(temperature))
	else:
		Kprop1d = 3.0 + 2.0*tanh(temperature/50)
	return np.atleast_2d(Kprop1d)

def nonlineardersfunc(args:dict):
	temperature = args.get('temperature')
	if ISLINEAR: Kprop = np.zeros(shape=np.shape(temperature))
	else: Kprop = (2.0/50)/(np.cosh(temperature/50))**2
	return Kprop

def conductivity_property(args:dict):
	temperature = args.get('temperature')
	Kprop1d = np.ravel(nonlinearfunc(args), order='F')
	reference = np.array([[1., 0.5],[0.5, 2.0]])
	Kprop2d = np.zeros((2, 2, len(temperature)))
	for i in range(2):
		for j in range(2):
			Kprop2d[i, j, :] = reference[i, j]*Kprop1d
	return Kprop2d

def conductivity_ders_property(args:dict):
	temperature = args.get('temperature')
	Kprop1d = np.ravel(nonlineardersfunc(args), order='F')
	reference = np.array([[1., 0.5],[0.5, 2.0]])
	Kprop2d = np.zeros((2, 2, len(temperature)))
	for i in range(2):
		for j in range(2):
			Kprop2d[i, j, :] = reference[i, j]*Kprop1d
	return Kprop2d

# Ring shape

def exactTemperatureRing_inc(args:dict):
	t = args['time']
	x = args['position'][0, :]
	y = args['position'][1, :]
	u = -CST*tanh(x**2+y**2-1.0)*sin(pi*(x**2+y**2-0.25**2))*sin(pi*x*y)*sin(pi/2*t)*(1+0.75*cos(3*pi/2*t))
	return u

def exactTemperatureRing_spt(args):
	time = args['time']
	position = args['position']
	nc_sp = np.size(position, axis=1); nc_tm = np.size(time); u = np.zeros((nc_sp, nc_tm))
	for i in range(nc_tm):
		t = time[i]
		u[:, i] = exactTemperatureRing_inc(args={'time':t, 'position':position})
	return np.ravel(u, order='F')

def powerDensityRing_inc(args:dict):
	t = args['time']
	x = args['position'][0, :]
	y = args['position'][1, :]

	u1 = pi*(x**2 + y**2 - 1/16); u2 = x**2 + y**2 - 1; u3 = (3*cos((3*pi*t)/2))/4 + 1; u4 = sin((pi*t)/2)
	if ISLINEAR:
		f = (2*CST*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
		- 12*CST*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
		+ 12*CST*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
		- (CST*pi*sin(pi*x*y)*sin(u1)*tanh(u2)*cos((pi*t)/2)*(u3))/2
		+ (9*CST*pi*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*sin((3*pi*t)/2))/8
		- 4*CST*x**2*pi*cos(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
		- 16*CST*x**2*pi*sin(pi*x*y)*cos(u1)*u4*(tanh(u2)**2 - 1)*(u3)
		- 4*CST*y**2*pi*cos(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
		- 32*CST*y**2*pi*sin(pi*x*y)*cos(u1)*u4*(tanh(u2)**2 - 1)*(u3)
		+ 16*CST*x**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(tanh(u2)**2 - 1)*(u3)
		+ 32*CST*y**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(tanh(u2)**2 - 1)*(u3)
		+ 4*CST*x**2*pi**2*cos(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
		+ 4*CST*y**2*pi**2*cos(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
		- 12*CST*x**2*pi**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
		- 18*CST*y**2*pi**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
		- 24*CST*x*y*pi*cos(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
		- 16*CST*x*y*pi*sin(pi*x*y)*cos(u1)*u4*(tanh(u2)**2 - 1)*(u3)
		+ 16*CST*x*y*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(tanh(u2)**2 - 1)*(u3)
		+ 24*CST*x*y*pi**2*cos(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
		- 10*CST*x*y*pi**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
	)
	else:

		f = ((2*tanh((CST*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50) - 3)*(2*CST*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
					- 2*CST*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
					+ 8*CST*x**2*pi*sin(pi*x*y)*cos(u1)*u4*(tanh(u2)**2 - 1)*(u3)
					- 8*CST*x**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(tanh(u2)**2 - 1)*(u3)
					+ 4*CST*x**2*pi**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
					+ CST*y**2*pi**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
					+ 4*CST*x*y*pi*cos(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
					- 4*CST*x*y*pi**2*cos(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3))
			- 2*(tanh((CST*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50) - 3/2)*(CST*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
					- 2*CST*x**2*pi*cos(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
					- 2*CST*y**2*pi*cos(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
					+ 2*CST*x**2*pi**2*cos(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
					+ 2*CST*y**2*pi**2*cos(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
					- 8*CST*x*y*pi*sin(pi*x*y)*cos(u1)*u4*(tanh(u2)**2 - 1)*(u3)
					+ 8*CST*x*y*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(tanh(u2)**2 - 1)*(u3)
					- 5*CST*x*y*pi**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))
			+ (4*tanh((CST*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50) - 6)*(2*CST*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
							- 2*CST*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
							+ 8*CST*y**2*pi*sin(pi*x*y)*cos(u1)*u4*(tanh(u2)**2 - 1)*(u3)
							- 8*CST*y**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(tanh(u2)**2 - 1)*(u3)
							+ CST*x**2*pi**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
							+ 4*CST*y**2*pi**2*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
							+ 4*CST*x*y*pi*cos(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
							- 4*CST*x*y*pi**2*cos(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3))
			+ 2*(tanh((CST*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50)**2 - 1)*(2*CST*x*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
							- 2*CST*x*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
							+ CST*y*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))*((CST*x*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3))/25
									- (CST*x*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3))/25
									+ (CST*y*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50)
			+ (tanh((CST*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50)**2 - 1)*(2*CST*x*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3)
					- 2*CST*x*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
					+ CST*y*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))*((CST*x*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50
							- (CST*y*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3))/25
							+ (CST*y*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3))/25)
			+ (tanh((CST*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50)**2 - 1)*((CST*x*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3))/25
					- (CST*x*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3))/25
					+ (CST*y*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50)*(CST*x*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
							- 2*CST*y*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
							+ 2*CST*y*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3))
			+ 4*(tanh((CST*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50)**2 - 1)*(CST*x*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3)
					- 2*CST*y*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3)
					+ 2*CST*y*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3))*((CST*x*pi*cos(pi*x*y)*sin(u1)*tanh(u2)*u4*(u3))/50
							- (CST*y*sin(pi*x*y)*sin(u1)*u4*(tanh(u2)**2 - 1)*(u3))/25
							+ (CST*y*pi*sin(pi*x*y)*cos(u1)*tanh(u2)*u4*(u3))/25)
			- (CST*pi*sin(pi*x*y)*sin(u1)*tanh(u2)*cos((pi*t)/2)*(u3))/2
			+ (9*CST*pi*sin(pi*x*y)*sin(u1)*tanh(u2)*u4*sin((3*pi*t)/2))/8
	)

	return f

def powerDensityRing_spt(args:dict):
	time = args['time']
	position = args['position']
	nc_sp = np.size(position, axis=1); nc_tm = np.size(time); f = np.zeros((nc_sp, nc_tm))
	for i in range(nc_tm):
		t = time[i]
		f[:, i] = powerDensityRing_inc(args={'time':t, 'position':position})
	return np.ravel(f, order='F')

def simulate_incremental(degree, cuts, powerdensity, nbel_time=None,
						quad_args=None, solve_system=True, alpha=0.5):

	# Create geometry
	if quad_args is None: quad_args = {'quadrule':'gs', 'type':'leg'}
	if nbel_time is None: nbel_time = int(2**CUTS_TIME)

	geometry = mygeomdl({'name':'QA', 'degree':degree, 'nbel':int(2**cuts)}).export_geometry()
	patch = singlepatch(geometry, quad_args=quad_args)
	time_inc = np.linspace(0, 1.0, nbel_time+1)

	# Add material
	material = heat_transfer_mat()
	material.add_capacity(1, is_uniform=True)
	material.add_conductivity(conductivity_property, is_uniform=False, shape_tensor=2)

	# Block boundaries
	boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nbvars=1)
	boundary.add_constraint(location_list=[{'direction':'x,y', 'face':'both,both'}], constraint_type='dirichlet')

	# Transient model
	problem_inc = heat_transfer_problem(material, patch, boundary)
	temperature_inc = np.zeros((patch.nbctrlpts_total, len(time_inc)))
	if not solve_system: return problem_inc, time_inc, temperature_inc

	# Add external force
	external_force = np.zeros((problem_inc.part.nbctrlpts_total, len(time_inc)))
	for i, t in enumerate(time_inc):
		external_force[:, i] = problem_inc.assemble_volumetric_force(powerdensity, args={'time':t})

	# Solve
	problem_inc._maxiters_nonlinear = 50
	problem_inc._tolerance_linear = 1e-12
	problem_inc._tolerance_nonlinear = 1e-12
	problem_inc.solve_heat_transfer(temperature_inc, external_force, time_list=time_inc, alpha=alpha)

	return problem_inc, time_inc, temperature_inc

def simulate_spacetime(degree, cuts, powerdensity, degree_time=None, nbel_time=None, quad_args=None,
					solve_system=True, use_newton=False, auto_inner_tolerance=True):

	# Create geometry
	if quad_args is None: quad_args = {'quadrule':'gs', 'type':'leg'}
	if degree_time is None: degree_time = degree
	if nbel_time is None: nbel_time = int(2**CUTS_TIME)

	geometry = mygeomdl({'name':'QA', 'degree':degree, 'nbel':int(2**cuts)}).export_geometry()
	space_patch = singlepatch(geometry, quad_args=quad_args)
	time_interval = mygeomdl({'name':'line', 'degree':degree_time, 'nbel':nbel_time}).export_geometry()
	time_patch = singlepatch(time_interval, quad_args=quad_args)

	# Add material
	material = heat_transfer_mat()
	material.add_capacity(1, is_uniform=True)
	material.add_conductivity(conductivity_property, is_uniform=False, shape_tensor=2)
	if use_newton: material.add_ders_capacity(0, is_uniform=True)
	if use_newton: material.add_ders_conductivity(conductivity_ders_property, is_uniform=False, shape_tensor=2)

	# Block boundaries
	boundary = boundary_condition(nbctrlpts=space_patch.nbctrlpts, nbvars=1)
	boundary.add_constraint(location_list=[{'direction':'x,y', 'face':'both,both'}], constraint_type='dirichlet')

	# Define space time problem
	problem_spt = st_heat_transfer_problem(material, space_patch, time_patch, boundary)

	# Add external force
	external_force = problem_spt.assemble_volumetric_force(powerdensity)
	temperature_spt = np.zeros_like(external_force)
	if not solve_system: return problem_spt, time_patch, temperature_spt

	problem_spt._maxiters_nonlinear = 50
	problem_spt._tolerance_linear = 1e-12
	problem_spt._tolerance_nonlinear = 1e-12
	problem_spt.solve_heat_transfer(temperature_spt, external_force, use_picard=(not use_newton),
								auto_inner_tolerance=auto_inner_tolerance)
	return problem_spt, time_patch, temperature_spt
