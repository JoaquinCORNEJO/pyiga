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
mpl.rcParams.update({"figure.autolayout": True})
mpl.rcParams["figure.figsize"] = (5.0, 4.0)
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["axes.unicode_minus"] = True
mpl.rcParams["axes.grid"] = True

try:
    mpl.rcParams["text.usetex"] = True
except Exception as e:
    print(f"Warning: LaTeX is not installed. Using default settings. ({e})")

try:
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.family"] = "STIXGeneral"
except Exception as e:
    print(
        f"Warning: STIX font family is not available. Using default font settings. ({e})"
    )

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
