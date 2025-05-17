from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import J2plasticity3d
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_mechanical import mechanical_problem

def surface_force(args):
	position = args['position']
	prop = np.zeros_like(position)
	prop[1, :] = 10
	return prop

# Create model 
degree, nbel, length = 2, 32, 1.0 
geometry = mygeomdl({'name':'square', 'degree':degree, 'nbel':nbel}).export_geometry()
patch = singlepatch(geometry, {'quadrule': 'gs'})

# Add material 
material = J2plasticity3d({'elastic_modulus':1e2, 'poisson_ratio': 0.3})

# Set Dirichlet boundaries
boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nbvars=2)
boundary.add_constraint(location_list=[{'direction':'x', 'face':'both'}, 
								{'direction':'y', 'face':'bottom'}], 
						constraint_type='dirichlet')

# Elasticity problem
problem = mechanical_problem(material, patch, boundary)
external_force = problem.assemble_surface_force(surface_force, location={'direction':'y', 'face':'top'})

# Solve problem
displacement = np.zeros_like(external_force)
problem.solve_elastoplasticity(displacement, external_force)

# Post processing 
strain_3d = problem.interpolate_strain(displacement, convert_to_3d=True)
stress_3d = material.eval_elastic_stress(strain_3d)
vonmises = material.eval_von_mises_stress(stress_3d)
print('Von misses max:%.4e, min:%.4e' %(vonmises.max(), vonmises.min()))
print('Difference: %.4e' %(abs(vonmises.max()-vonmises.min())))
