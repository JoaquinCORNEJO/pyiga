# Source files

## 1. `__init__`

The `__init__` module is crucial for setting up the project's environment. It imports essential libraries such as `numpy` and `scipy` for optimized numerical computations, `matplotlib` for plotting and visualization, and `geomdl` for geometric modeling. Additionally, it customizes the default properties of `matplotlib` to enable LaTeX rendering, ensuring that mathematical expressions are displayed with high-quality typesetting.

<!--------------->

## 2. `lib_mygeometry`

The `lib_mygeometry` module contains the `mygeomdl` class, which is designed to simplify the creation of predefined geometries using the `geomdl` library. This class provides an easy-to-use interface for defining and parametrizing simple geometrical shapes such as squares, rings, and prisms, making it accessible for users to generate these geometries with minimal effort.

### 2.1. `mygeomdl` class

The `mygeomdl` class facilitates the creation and parametrization of geometries.

#### 2.1.1. Initialization

```python
def __init__(self, geo_args: dict):
```

- `geo_args`: A dictionary containing parameters for the geometry, such as `name`, `degree`, and `nbel`.

#### 2.1.2. Methods

- `export_geometry(self)`: Generates and returns the parametrized geometry based on the provided parameters.

### 2.2. Example

```python
# Initialize the mygeomdl class with parameters
geometry = mygeomdl({'name': 'quarter_annulus', 'degree': 3, 'nbel': 16, 'geo_parameters':{'Rin':1., 'Rex':2.}}).export_geometry()
```

In this example, the `mygeomdl` class is initialized with a dictionary of parameters, and the `export_geometry` method is called to create the geometry. The parameters include the name of the geometry, the degree of the shape, the number of elements (`nbel`) and extra arguments to change default values.

<!--------------->

## 3. `lib_part`

The `lib_part` module contains the `singlepatch` class, which reads a `geomdl` object (curve, surface, volume) and extracts important information for our own database. It also assigns a quadrature rule defined by the user. The quadrature rules are based on Gauss-Legendre and Weighted quadrature.

### 3.1. `singlepatch` class

The `singlepatch` class processes geometric objects and sets up quadrature rules.

#### 3.1.1. Initialization

```python
def __init__(self, obj, quad_args):
```

- `obj`: A `geomdl` object, which can be a curve, surface, or volume.
- `quad_args`: Arguments defining the quadrature rules.

#### 3.1.2. Methods

- `postprocessing_primal(self, fields={}, folder=None, sample_size=None, name='output')`: Exports the solution in VTK format, allowing for visualization using Paraview. This method takes several optional parameters:
  - `fields`: A dictionary of fields (at control points) to be exported.
  - `folder`: The folder where the output files will be saved.
  - `sample_size`: The size of the sample for the output.
  - `name`: The name of the output file.

- `postprocessing_dual(self, fields={}, folder=None, name='output')`: Exports the solution in VTK format, allowing for visualization using Paraview. This method takes several optional parameters:
  - `fields`: A dictionary of fields (evaluated at quadrature points) to be exported.
  - `folder`: The folder where the output files will be saved.
  - `name`: The name of the output file.

### 3.2. Example

```python
# Initialize the singlepatch class with a geomdl object and quadrature arguments
patch = singlepatch(geometry, quad_args={'quadrule': 'gs', 'type': 'leg'})
```

In this example, the `singlepatch` class is initialized with a `geomdl` geometry and quadrature arguments.

<!--------------->

## 4. `lib_boundary`

The `lib_boundary` module is responsible for managing boundary conditions in numerical simulations. It contains the `boundary_condition` class, which handles the connectivity operators to set up Dirichlet and Neumann conditions.

### 4.1. `boundary_condition` class

The `boundary_condition` class is designed to manage boundary conditions in numerical simulations. It provides methods to define and handle conditions on control points.

#### 4.1.1. Initialization

```python
def __init__(self, nbctrlpts=np.array([1, 1, 1]), nbvars=1):
```

- `nbctrlpts`: An array defining the number of control points in each direction.
- `nbvars`: The number of variables in the simulation.

#### 4.1.2. Methods

- `add_constraint(self, location: list, constraint_type: str)`: Adds a constraint (Dirichlet or Neumann) to the specified location.
- `select_nodes4solving(self)`: Selects the nodes that are free and those that have Dirichlet constraints.

### 4.2. Example

```python
# Initialize the boundary_condition class
bc = boundary_condition(nbctrlpts=np.array([4, 4, 4]), nbvars=2)

# Define a location for the constraint
location = [{'direction': 'x,y', 'face': 'left,both'}, {'direction': 'y', 'face': 'top'}]

# Add a Dirichlet constraint
bc.add_constraint(location, 'dirichlet')

# Select nodes for solving
free_nodes, constraint_nodes = bc.select_nodes4solving()
```

## 5. `lib_material`

The `lib_material` module is designed to handle material properties for different types of simulations. It includes classes for managing heat transfer properties and elastoplasticity behavior.

### 5.1. `heat_transfer_mat` class

The `heat_transfer_mat` class manages material properties related to heat transfer, such as thermal conductivity and heat capacity.

#### 5.1.1. Initialization

```python
def __init__(self):
```

#### 5.1.2. Methods

- `add_conductivity(self, value, is_uniform=True, shape_tensor=1)`: Adds thermal conductivity to the material. Parameters include:
  - `value`: The conductivity value or function.
  - `is_uniform`: A boolean indicating if the conductivity is uniform.
  - `shape_tensor`: The shape tensor for the conductivity.

- `add_capacity(self, value, is_uniform=True)`: Adds heat capacity to the material. Parameters include:
  - `value`: The capacity value or function.
  - `is_uniform`: A boolean indicating if the capacity is uniform.

### 5.2. Example

```python
# Initialize the heat_transfer_mat class
material = heat_transfer_mat()

# Add thermal conductivity
material.add_conductivity(1.0, is_uniform=True, shape_tensor=1)

# Add heat capacity
material.add_capacity(1.0, is_uniform=True)
```

### 5.3. `J2plasticity1d` and `J2plasticity3d` classes

The `J2plasticity1d` class handles elastoplasticity behavior in one-dimensional problems, including elastic modulus, elastic limit, and isotropic hardening; while `J2plasticity3d` handles elastoplasticity behavior in two-dimensional and three-dimensional geometries.

#### 5.3.1. Initialization

```python
def __init__(self, material_properties: dict):
```

- `material_properties`: A dictionary containing properties such as `elastic_modulus`, `elastic_limit`, `iso_hardening` and `kine_hardening`.

#### 5.3.2. Methods

There are no specific methods for the `J2plasticity1d` and `J2plasticity3d` classes. These classes are designed to handle elastoplasticity behavior based on the provided material properties during initialization.

### 5.4. Example

```python
# Initialize the J2plasticity3d class with material properties
material = J2plasticity3d({
        'elastic_modulus': 1.e3, 
        'poisson_ratio': 0.3 
        'elastic_limit': 1.e1, 
        'iso_hardening': {'name': 'linear', 'Eiso': 1.e2}
})
```
<!--------------->

## 6. `single_patch/lib_job_heat_transfer`

The `lib_job_heat_transfer` module handles heat transfer problems. It contains the `heat_transfer_problem` and `st_heat_transfer_problem` classes.

### 6.1. `heat_transfer_problem` class

The `heat_transfer_problem` class is designed to solve heat transfer problems using various methods.

#### 6.1.1. Initialization

```python
def __init__(self, material: heat_transfer_mat, patch: singlepatch, boundary: boundary_condition, solver_args={}):
```

- `material`: An instance of the `heat_transfer_mat` class.
- `patch`: An instance of the `singlepatch` class.
- `boundary`: An instance of the `boundary_condition` class.
- `solver_args`: Optional arguments for the solver.

#### 6.1.2. Methods

- `solve_heat_transfer(self, temperature_list, external_force_list, time_list=None, alpha=0.5)`: Solves heat transfer problems, including both steady and transient problems. Parameters include:
  - `temperature_list`: A list of temperatures (including initial and boundary conditions).
  - `external_force_list`: A list of external forces or heat sources.
  - `time_list`: An optional list of time steps for transient analysis.
  - `alpha`: A parameter for the time integration scheme, defaulting to 0.5 for the Crank-Nicolson method.

### 6.2. `st_heat_transfer_problem` class

The `st_heat_transfer_problem` class is designed to solve space-time heat transfer problems.

#### 6.2.1. Initialization

```python
def __init__(self, material: heat_transfer_mat, patch: singlepatch, time_patch: singlepatch, boundary: boundary_condition, solver_args={}):
```

- `material`: An instance of the `heat_transfer_mat` class.
- `patch`: An instance of the `singlepatch` class.
- `time_patch`: An instance of the `singlepatch` class for the time dimension.
- `boundary`: An instance of the `boundary_condition` class.
- `solver_args`: Optional arguments for the solver.

#### 6.2.2. Methods

- `solve_heat_transfer(self, temperature, external_force, use_picard=True, auto_inner_tolerance=True, auto_outer_tolerance=False, nonlinear_args={})`: Solves space-time heat transfer problems.
  - `temperature`: The temperature guessed in the space-time domain including initial and boundary conditions.
  - `external_force`: External force or heat source applied to the system.
  - `use_picard` (optional): If `True`, uses the Picard iteration method, otherwise uses Newton method.
  - `auto_inner_tolerance` (optional): If `True`, automatically determines the inner tolerance for the iterative linear solver.
  - `auto_outer_tolerance` (optional): If `True`, automatically determines the outer tolerance for the nonlinear solver.
  - `nonlinear_args` (optional): Additional arguments for inexact nonlinear solvers. Defaults to an empty dictionary.

### 6.3. Example

```python
# Initialize the heat_transfer_problem class
heat_problem = heat_transfer_problem(material, patch, boundary)

# Solve heat transfer problem
heat_problem.solve_heat_transfer(temperature_list, external_force_list, time_list)

# Initialize the st_heat_transfer_problem class
st_heat_problem = st_heat_transfer_problem(material, patch, time_patch, boundary)

# Solve space-time heat transfer problem
st_heat_problem.solve_heat_transfer(temperature, external_force)
```

<!--------------->

## 7. `single_patch/lib_job_mechanical`

The `lib_job_mechanical` module handles mechanical problems, including linear dynamics and elastoplasticity. It contains the `mechanical_problem` class, which extends the `space_problem` class.

### 7.1. `mechanical_problem` class

The `mechanical_problem` class is designed to solve various mechanical problems using different methods.

#### 7.1.1. Initialization

```python
def __init__(self, material, patch, boundary, solver_args={}, allow_lumping=False):
```

- `material`: An instance of the `plasticity` class.
- `patch`: An instance of the `singlepatch` class.
- `boundary`: An instance of the `boundary_condition` class.
- `solver_args`: Optional arguments for the solver.
- `allow_lumping`: A boolean indicating if mass lumping is allowed.

#### 7.1.2. Methods

- `solve_explicit_linear_dynamics(self, displacement_list, acceleration_list, external_force_list, time_list)`: Solves explicit linear dynamics problems.
- `solve_eigenvalue_problem(self, args={}, which='LM', k=None)`: Solves the eigenvalue problem.
- `solve_elastoplasticity(self, displacement_list: np.ndarray, external_force_list: np.ndarray)`: Solves elastoplasticity problems.

### 7.2. Example

```python
# Initialize the mechanical_problem class
mechanical = mechanical_problem(material, patch, boundary, allow_lumping=False)

# Solve explicit linear dynamics
mechanical.solve_explicit_linear_dynamics(displacement_list, acceleration_list, external_force_list, time_list)

# Solve eigenvalue problem
mechanical.solve_eigenvalue_problem()

# Solve elastoplasticity
mechanical.solve_elastoplasticity(displacement_list, external_force_list)
```

<!--------------->

## 8. Other modules

There are several internal modules that, while not directly referenced in the examples, provide essential functionalities that support the main operations of the project:

- **`lib_quadrules`**: Defines various quadrature rules used for numerical integration, including Gauss-Legendre, Gauss-Lobatto, and Weighted quadrature with 2 and 4 rules. These rules are crucial for accurate numerical computations.

- **`lib_solver`**: Contains algorithms for solving systems of equations, such as Conjugate Gradient (CG), Bi-Conjugate Gradient Stabilized (BiCG-STAB), and Generalized Minimal Residual (GMRES). These solvers are implemented with custom code to handle specific requirements like boundary conditions, preconditioners, and easy access to residuals, which are not fully addressed by standard libraries like `scipy`.

- **`single_patch/lib_job`**: Manages the execution of different simulation jobs, particularly in heat transfer and elastoplastic problems. This module orchestrates the workflow, ensuring that simulations run smoothly and efficiently.

- **`lib_tensor_maths`**: Provides mathematical operations for tensor calculations, leveraging `numpy` functions. It also defines preconditioners based on fast-diagonalization, enhancing the performance of numerical solvers.

These internal modules ensure the robustness and efficiency of the overall codebase, enabling complex simulations and analyses to be performed with high accuracy and reliability.
