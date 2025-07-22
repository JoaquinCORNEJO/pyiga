from src.lib_nonlinear_solver import *


def compute_residual(x, args):
    b = [2.0, 1.0]
    K = [[0.5 + x[1], -0.2], [-0.2, 1.0 + 2 * x[0]]]
    return b - K @ x, {"x": x}


def solve_linearization(res, args):
    x = args["x"]
    K = [[0.5 + x[1], -0.2], [-0.2, 1.0 + 2 * x[0]]]
    return np.linalg.solve(K, res)


# x_k = np.array([0.0, 0.0])
# for i in range(100):
#     r, _ = compute_residual(x_k, {})
#     inc = solve_linearization(r, {'x': x_k})
#     x_k += inc
# print(compute_residual(x_k, {})[0])
# print(x_k)

solver = nonlinsolver(allow_linesearch=True, allow_fixed_point_acceleration=True)
solver.solve([0.0, 0.0], compute_residual, solve_linearization, anderson=1)

# import numpy as np

# # Vector constante b
# b = np.array([2.0, 2.0])

# # F(x) = A(x) * x - b
# def F(x):
#     A = np.array([[x[0], 1.0],
#                   [1.0, x[1]]])
#     return A @ x - b

# # Jacobiana de F(x)
# def J(x):
#     # # A(x) = [[x0, 1], [1, x1]]
#     # # Necesitamos derivar A(x) @ x respecto a x
#     # J = np.zeros((2, 2))

#     # # Derivadas parciales (usamos cálculo manual)
#     # # F0 = x0^2 + x1 - 2
#     # J[0, 0] = 2 * x[0]       # dF0/dx0
#     # J[0, 1] = 1              # dF0/dx1

#     # # F1 = x0 + x1^2 - 2
#     # J[1, 0] = 1              # dF1/dx0
#     # J[1, 1] = 2 * x[1]       # dF1/dx1
#     J = np.array([[x[0], 1.0],
#                   [1.0, x[1]]])
#     return J

# # Método de Newton
# def newton(F, J, x0, tol=1e-6, max_iter=100):
#     x = x0
#     for i in range(max_iter):
#         Fx = F(x)
#         Jx = J(x)
#         delta = np.linalg.solve(Jx, -Fx)
#         x = x + delta
#         if np.linalg.norm(delta) < tol:
#             print(f"Convergió en {i+1} iteraciones.")
#             return x
#     print("No convergió.")
#     return x

# # Valor inicial
# x0 = np.array([0.0, 0.0])

# # Ejecutar el método
# sol = newton(F, J, x0)
# print("Solución:", sol)


# solver = nonlinsolver()
# x = solver.solve([0.0, 0.0], F, solve_linearization, anderson_iters=1)
# # # print(x)
