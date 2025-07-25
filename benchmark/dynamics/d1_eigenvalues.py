"""
.. Test of mecanical displacement 1D
.. Author: Fabio MADIE
.. Joaquin Cornejo added some corrections 28 nov. 2024
"""

from src.__init__ import *
from src.lib_mygeometry import mygeomdl
from src.lib_part import singlepatch
from src.lib_material import J2plasticity
from src.lib_boundary import boundary_condition
from src.single_patch.lib_job_mechanical import mechanical_problem

# Global constants
YOUNG, RHO, LENGTH = 210e9, 7800, 1.0


def simulate(degree):
    nbel = 128
    geometry = mygeomdl(
        {
            "name": "line",
            "degree": degree,
            "nbel": nbel,
            "geo_parameters": {"L": LENGTH},
        }
    ).export_geometry()
    patch = singlepatch(geometry, quad_args={"quadrule": "gs", "type": "leg"})
    boundary = boundary_condition(nbctrlpts=patch.nbctrlpts, nb_vars_per_ctrlpt=1)
    material = J2plasticity({"elastic_modulus": YOUNG}, is_unidimensional=True)
    material.add_density(RHO, is_uniform=True)
    problem = mechanical_problem(material, patch, boundary)
    frequency = np.sqrt(problem.solve_eigenvalue_problem(which="SM", k=nbel-2)[0][1:])
    return frequency


degree_labels = ["Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
fig, ax = plt.subplots()
for label, degree in zip(degree_labels, range(1, 6)):
    approx_freq = simulate(degree)
    freq_indices = np.arange(1, len(approx_freq) + 1)
    exact_freq = (freq_indices * np.pi / LENGTH) * np.sqrt(YOUNG / RHO)
    ax.plot(freq_indices / len(approx_freq), approx_freq / exact_freq, label=label)

ax.set_xlabel(r"$n/N$")
ax.set_ylabel(r"$\omega^{app}/\omega^{exact}$")
ax.legend()
ax.grid(False)
ax.set_xlim([0, 1])
ax.set_ylim([0.9, 1.5])
fig.tight_layout()
fig.savefig(f"{RESULT_FOLDER}eigspectrum")
