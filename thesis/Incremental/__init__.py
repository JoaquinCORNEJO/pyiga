from src.__init__ import *
from scipy.optimize import fsolve

FOLDER2RESU = os.path.dirname(os.path.realpath(__file__)) + "/results/"
FOLDER2DATA = os.path.dirname(os.path.realpath(__file__)) + "/data/"
if not os.path.isdir(FOLDER2RESU):
    os.mkdir(FOLDER2RESU)
if not os.path.isdir(FOLDER2DATA):
    os.mkdir(FOLDER2DATA)

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

CONFIGLINE_IGA = {
    "marker": "s",
    "linestyle": "-",
    "markersize": 10,
    "markerfacecolor": "w",
}
CONFIGLINE_WQ = {
    "marker": "o",
    "linestyle": "--",
    "markersize": 6,
    "markerfacecolor": None,
}
CONFIGLINE_INC = {
    "marker": "d",
    "linestyle": "-.",
    "markersize": 6,
    "markerfacecolor": None,
}
CONFIGLINE_BDF = {
    "marker": "+",
    "linestyle": "-",
    "markersize": 6,
    "markerfacecolor": None,
}
