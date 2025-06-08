# Python libraries
import os, sys, re
import numpy as np
from scipy import sparse as sp, linalg as sclin
import scipy.sparse.linalg as scsplin
from copy import deepcopy
import matplotlib as mpl
from matplotlib import pyplot as plt

# Dispensable functions:
try:
    from geomdl import helpers, BSpline
except:
    print(
        "geomdl module is not installed. For conveninece, we will use a homemade module"
    )
    import splines.helpers as helpers
    import splines.BSpline as BSpline
try:
    from pyevtk.hl import gridToVTK
except:
    print(
        "pyevtk module is not installed. Some functions in post-processing may be disabled"
    )
try:
    from pyevtk.vtk import VtkGroup
except:
    print(
        "pyevtk module is not installed. Some functions in post-processing may be disabled"
    )
try:
    import pyvista as pv
except:
    print(
        "pyvista module is not installed. Some functions in post-processing may be disabled"
    )

# Default properties
mpl.rcParams.update(
    {
        "figure.autolayout": True,
        "figure.figsize": (5.0, 4.0),
        "figure.dpi": 300,
        "axes.unicode_minus": True,
        "axes.grid": True,
    }
)

try:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
        }
    )

    # mpl.rcParams.update(
    #     {
    #         "text.usetex": True,
    #         "text.latex.preamble": r"""
    #     \usepackage{helvet}
    #     \usepackage{amsmath}
    # """,
    #     }
    # )

except Exception as e:
    print(f"LaTeX not available. Using default settings. ({e})")


# Select folder
RESULT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/../results/"
if not os.path.isdir(RESULT_FOLDER):
    os.mkdir(RESULT_FOLDER)

# Define size
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
