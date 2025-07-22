import sympy as sp
from sympy import sin, cos, tanh, pi

# Variables
x, y, t, CST = sp.symbols("x y t CST")

# Solución manufacturada ejemplo
T = (
    CST
    * tanh(1.0 - x**2 - y**2)
    * sin(pi * (x**2 + y**2 - 0.25**2))
    * sin(2 * pi * x * y)
    * sin(2 * pi * t / 3)
)
# T = (
#     CST * sin(pi * x**2)
#     * sin(3*pi * y**2)
#     * cos(2 * pi * x * y)
#     * tanh(0.5 * pi * t)
#     * (1.0 + 0.75 * cos(1.5 * pi * t))
# )
# T = (
#     CST * sin(pi * y) * sin(pi * x)
#     * sin(pi * (y + 0.75 * x - 0.5) * (-y + 0.75 * x - 0.5))
#     * tanh(0.5 * pi * t)
#     * (1.0 + 0.75 * cos(1.5 * pi * t))
# )

# Parámetros numéricos
cp_val = 1
k_mat = sp.Matrix([[1.0, 0.5], [0.5, 2.0]])
# k_mat *= (3.0 + 2.0 * tanh(T / 50))

# Gradiente de T
grad_T = sp.Matrix([sp.diff(T, var) for var in (x, y)])

# Multiplicar K * grad_T
flux = k_mat @ grad_T

# Divergencia: derivar cada componente de flux respecto a x, y, z y sumar
div_flux = sum(sp.diff(flux[i], var) for i, var in enumerate((x, y)))

# Derivada temporal
dT_dt = sp.diff(T, t)

# Carga térmica Q
Q = cp_val * dT_dt - div_flux

print("Carga térmica Q(x,y,z,t):")
print(str(Q))


# import sympy as sp
# from sympy import sin, cos, pi, tanh

# # Variables
# x, t = sp.symbols('x t')

# # Solución manufacturada ejemplo
# T = sin(2 * pi * x) * sin(0.5 * pi * t) * (1.0 + 0.75 * cos(1.5 * pi * t))

# # Parámetros numéricos
# cp_val = 1
# k_mat = 3.0 + 2.0 * tanh(T / 50)

# # Gradiente de T
# grad_T = sp.diff(T, x)

# # Multiplicar K * grad_T
# flux = k_mat * grad_T

# # Divergencia: derivar cada componente de flux respecto a x, y, z y sumar
# div_flux = sp.diff(flux, x)

# # Derivada temporal
# dT_dt = sp.diff(T, t)

# # Carga térmica Q
# Q = cp_val * dT_dt - div_flux

# print("Carga térmica Q(x,y,z,t):")
# print(str(Q))
