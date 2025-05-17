# IGA project using only python

by J. Cornejo Fuentes in collaboration with T. Elguedj, D. Dureisseix and A. Duval

## Overview

This project is an implementation of IsoGeometric Analysis (IGA), largely based on my developments during my PhD thesis. It is inspired by the YETI code from the LaMCoS laboratory, which is primarily written in Fortran.

The goal of this initiative is to make the code more accessible to those who only program in Python. By taking advantage of optimized libraries such as `numpy` and `scipy`, this project provides a fully Python-based IGA implementation without requiring Fortran or other compiled languages.

## Requirements

To run this project, you need to install the following Python packages:

```bash
pip install numpy scipy geomdl matplotlib pyevtk
```

Alternatively, you can install the packages using a `requirements.txt` file:

1. Create a `requirements.txt` file with the following content:

    ```text
    numpy
    scipy
    geomdl
    matplotlib
    pyevtk
    ```

2. Install the packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

It is recommended as well to install the latest version of ParaView and LaTeX.

## Installation and execution

No special installation is required beyond the package dependencies. The code can be executed simply by running:

```bash
python myfile.py
```

There are no optional parameters for execution. Some examples generate figures, while others produce .vts files that can be visualized in ParaView.

## File structure

The project is organized into the following directories and files:

```bash
📂 Project
 ┣ 📂 src
 ┃ ┣ 📂 single_patch
 ┃ ┣ 📂 multi_patch
 ┣ 📂 results
 ┣ 📂 examples
 ┃ ┣ 📂 heat_transfer
 ┃ ┣ 📂 elastoplasticity
 ┃ ┗ 📂 dynamics
 ┗ 📜 README.md

```

- `src/`: Contains the core functions and modules necessary for the IGA implementation.
  - `single_patch`: Contains IGA functions for a single patch parameterization.
  - `multi_patch`: TODO include functions for multi-patch parameterization.
- `results/`: Contains the figures and .vts files that are generated in some examples.
- `examples/`: Includes various example problems to demonstrate the capabilities of the project.
  - `heat_transfer/`: Contains examples related to heat transfer problems in 1D, 2D, and space-time.
  - `elastoplasticity/`: Contains examples related to elastoplasticity simulations.
  - `dynamics/`: Contains examples related to explicit dynamics simulations.
- `README.md`: The project documentation file you are currently reading.

## License

The code is freely available for modification and use. If used as a basis for publication, please cite my work (thesis, articles, or related publications).

## Contact

For any questions, feel free to reach out via gmail at [joaquin.cofu@gmail.com](mailto:joaquin.cofu@gmail.com).

## Bibliography (recommended)

In this repository, we develop IsoGeometric Analysis (IGA) with weighted quadrature (WQ), matrix-free (MF), and Fast Diagonalisation (FD) approaches. This code is mainly based on ideas and algorithms described in the following books and papers.

### Books

#### B-Splines and NURBS

- *The NURBS book* by L. Piegl
- *Isogeometric analysis: toward integration of CAD and FEA* by J. Cotrell, T.J.R. Hughes, and Y. Bazilevs

#### Finite Element method

- *The finite element method* by T.J.R. Hughes
- *Finite element simulation of heat transfer* by J.M. Bergheau and R. Fortunier

#### Plasticity Theory

- *Computational inelasticity* by J.C. Simo and T.J.R. Hughes
- *Introduction to nonlinear finite element analysis* by N.H. Kim

### Papers

#### Tensor Product and sum-factorization

- *Tensor decompositions and applications* by T. Kolda and B. Bader
- *Efficient matrix computation for tensor-product isogeometric analysis* by P. Antolin et al.

#### Iterative Solvers

- *Templates for the solution of linear systems: building blocks for iterative solvers* by R. Barrett et al.
- *Choosing the forcing terms in an inexact Newton method* by S. Eisenstat and H. Walker

#### Weighted Quadrature, matrix-free and fast diagonalisation

- *Fast formation of isogeometric Galerkin matrices by weighted quadrature* by F. Calabro et al.
- *Fast formation and assembly of finite element matrices with application to isogeometric linear elasticity* by R. Hiemstra et al.
- *Solving Poisson's equation using dataflow computing* by R. van Nieuwpoort (Master's thesis)
- *Matrix-free weighted quadrature for a computationally efficient isogeometric k-method* by G. Sangalli and M. Tani
- *Preconditioners for isogeometric analysis* by M. Montardini (PhD thesis)
- *Isogeometric preconditioners based on fast solvers for the Sylvester equation* by G. Sangalli and M. Tani

#### Space time theory

- *Space-time finite element methods for parabolic problems* by O. Steinbach
- *Space-time isogeometric Galerkin methods for parabolic problems and efficient solvers* by A. Bressan et al.

#### Others

- *A brief on tensor analysis* by J. Simmonds
- *B free* by J. Planas, I. Romero, and J.M. Sancho
- *Solution algorithms for nonlinear transient heat conduction analysis employing element-by-element iterative strategies* by J. Winget and T.J.R. Hughes
