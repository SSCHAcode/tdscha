
# Introduction

## What is TD-SCHA?

TD-SCHA is a theory to simulate quantum nuclear motion in materials with strong anharmonicity.  
TD-SCHA stands for Time-Dependent Self-Consistent Harmonic Approximation, and it is the dynamical extension of the SCHA theory, that can describe equilibrium properties of materials accounting for both quantum and dynamical nuclear fluctuations.  

The `tdscha` Python library allows performing dynamical linear response calculations on top of the equilibrium results (computed with the `python-sscha` package).

## Why would I need tdscha?

### A Tool for Advanced Material Simulations

**Enhanced Simulation Capabilities**  
TD-SCHA is an essential tool for researchers and professionals in material science, particularly for simulating transport or thermal properties of materials, phase diagrams, and phonon-related properties.

**Integration with python-sscha**  
Seamless integration with `python-sscha` allows for the inclusion of both thermal and quantum phonon fluctuations in *ab initio* simulations.

**Leveraging the SSCHA Method**

**Quantum and Thermal Fluctuations**  
The Stochastic Self-Consistent Harmonic Approximation (SSCHA) is a full-quantum method optimizing the nuclear wave-function or density matrix to minimize free energy, crucial for simulating highly anharmonic systems.

**Efficiency and Cost-Effectiveness**  
Despite its full quantum and thermal nature, the computational cost is comparable to classical molecular dynamics, enhanced by the algorithm's ability to exploit crystal symmetries.

**User-Friendly and Versatile**

**Python Library and Stand-alone Software**  
Available both as a Python library and stand-alone software, with input scripts sharing syntax with Quantum ESPRESSO.

**Broad Compatibility**  
Can couple with any *ab initio* engine and interacts through the Atomic Simulation Environment (ASE) with an interface for automatic job submission on remote clusters.

**Getting Started**

**Easy to Use**  
User-friendly with short, readable input files and comprehensive tutorials.  

**Download and Explore**  
Download and install `python-sscha`, and start exploring the tutorials to enhance your material simulation projects.

# How to install

You need to have installed `python-sscha` and `CellConstructor` to work with `tdscha`.  
Please read the installation guide for those packages before proceeding further.  
You may find all instructions on the [official website](http://www.sscha.eu).  

To install from PyPI, simply type:

```bash
pip install tdscha
```
You can alternatively clone the repository from GitHub and install the package with:

```
pip install .
```


# Quick start
Go to the [tutorials](https://sscha.eu/Tutorials/tutorial_05_ramanir/)

