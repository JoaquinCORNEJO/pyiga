from src.__init__ import *
from scipy.optimize import fsolve

FOLDER2RESU = os.path.dirname(os.path.realpath(__file__)) + '/results/'
FOLDER2DATA = os.path.dirname(os.path.realpath(__file__)) + '/data/'
if not os.path.isdir(FOLDER2RESU): os.mkdir(FOLDER2RESU)
if not os.path.isdir(FOLDER2DATA): os.mkdir(FOLDER2DATA)

MARKERLIST = ["o", "v", "s", "X", "+", "p", "*"]
COLORLIST = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#C7C7C7",
    "#DBDB8D",
    "#9EDAE5",
]

CONFIGLINE_IGA = {"marker": "s", "linestyle": "-", "markersize": 10, "markerfacecolor":"w"}
CONFIGLINE_WQ = {"marker": "o", "linestyle": "--", "markersize": 6, "markerfacecolor":None}
CONFIGLINE_INC = {"marker": "d", "linestyle": "-.", "markersize": 6, "markerfacecolor":None}
CONFIGLINE_BDF = {"marker": "+", "linestyle": "-", "markersize": 6, "markerfacecolor":None}


def bdf(f, tspan, y0, nsteps, norder=1):
	"""
	Solves an ODE using the Backward Differentiation Formula (BDF) method.

	Parameters:
	f : The function defining the ODE system, f(t, y).
	tspan : A tuple containing the start and end times, (t0, tf).
	y0 : Initial conditions for the ODE system.
	nsteps : Number of time steps to use in the solution.
	norder : Order of the BDF method (default is 1).

	Returns:
	t : Array of time points.
	y : Array of solution values at each time point.
	"""

	def bdf_residual(f, dt, t_current, y_hist, y_current, norder):
		"Computes the residual for the BDF method of the specified order."
		assert len(y_hist) == norder, "Size problem."
		if norder == 1:
			[y1] = y_hist
			return y_current - y1 - dt*f(t_current, y_current)
		elif norder == 2:
			[y1, y2] = y_hist
			return 3*y_current - 4*y2 + y1 - 2*dt*f(t_current, y_current)
		elif norder == 3:
			[y1, y2, y3] = y_hist
			return 11*y_current - 18*y3 + 9*y2 - 2*y1 - 6*dt*f(t_current, y_current)
		elif norder == 4:
			[y1, y2, y3, y4] = y_hist
			return 25*y_current - 48*y4 + 36*y3 - 16*y2 + 3*y1 - 12*dt*f(t_current, y_current)
		else: raise ValueError("Order not supported")

	def create_y_list(i, y, norder):
		"Creates a list of previous solution values."
		return [y[:, i - k] for k in range(norder, 0, -1)]

	if not (1 <= norder <= 4): raise ValueError("Order not supported")

	# Initialize time and solution arrays
	m = len(y0)
	t = np.linspace(tspan[0], tspan[1], nsteps + 1)
	y = np.zeros((m, nsteps + 1))
	dt = (tspan[1] - tspan[0]) / nsteps

	# Set initial condition
	y[:, 0] = y0

	# Main loop to solve the ODE using the BDF method
	for i in range(1, nsteps + 1):

		# Determine the order to use based on the current step
		y_list = create_y_list(i, y, min(i, norder))

		# Predict the new solution value using the previous step
		t_current = t[i]
		y_guess = y[:, i - 1] + dt*f(t[i - 1], y[:, i - 1])

		# Solve the BDF residual equation to find the new solution value
		y[:, i] = fsolve(lambda y_current: bdf_residual(f, dt, t_current, y_list,
								y_current, min(i, norder)), y_guess, xtol=1e-10)

	return t, y
