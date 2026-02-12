# TD-SCHA Documentation

Time-Dependent Self-Consistent Harmonic Approximation (TD-SCHA) is a Python library for simulating quantum nuclear motion in materials with strong anharmonicity. It performs dynamical linear response calculations on top of equilibrium results from `python-sscha`. 

Part of the SSCHA ecosystem: `cellconstructor` → `python-sscha` → `tdscha`.

## Documentation Structure

1. **[Theory](theory.md)** - What is TD-SCHA, why linear response, and what quantities can be calculated with it. Theoretical background with main TDSCHA equations and the Lanczos algorithm.

2. **[Installation](installation.md)** - Complete installation guide following the official SSCHA.eu instructions, with emphasis on Julia for performance.

3. **[Quick Start](quickstart.md)** - Working example showing calculation and analysis via CLI commands.

4. **[In-Depth Usage](usage.md)** - Choosing perturbations, parallel execution, gamma-only trick, Wigner vs normal representation.

5. **[StaticHessian](static-hessian.md)** - Computing free energy Hessian of large systems via sparse linear algebra.

6. **[CLI Tools](cli.md)** - Command-line interface for analysis and visualization.

7. **[Examples](examples.md)** - Detailed examples and templates.

8. **API Reference** - Automatically generated documentation:
   - [DynamicalLanczos](api/dynamical_lanczos.md) - Core Lanczos algorithm
   - [StaticHessian](api/static_hessian.md) - Free energy Hessian calculations

## Key Features

- **Simulating the vibrational anharmonic spectra**: IR absorption, Raman scattering, Nuclear inelastic scattering, and phonon spectral functions.
- **Full quantum treatment of atomic nuclei** 
- **Parallel execution** with MPI
- **Symmetry-aware** calculations for efficiency

## Theoretical Foundation

TD-SCHA extends the Self-Consistent Harmonic Approximation (SCHA) to time-dependent perturbations, enabling computation of the response of nuclei to dynamical external fields.
This code explores the linear response regime, where the response is proportional to the perturbation, allowing for efficient calculations of susceptibilities and spectral functions.
If we prepare at time $t=0$ a system in equilibrium, and apply an external perturbation proportional to $\hat V_\text{ext}(t) = \hat B f_B(t)$ for $t > 0$, the expectation value of an observable $\hat A$ at time $t$ is given by:

$$
\langle \hat A(t) \rangle = \int_{0}^t dt'\chi_{AB}(t - t') f_B(t')
$$

or, in frequency space,

$$
\langle \hat A(\omega) \rangle = \chi_{AB}(\omega) f_B(\omega)
$$

The target of the TD-SCHA code is to compute the susceptibility $\chi_{AB}(\omega)$, which encodes the response of the system to the perturbation.
This quantity is related to many experimentally measurable properties, such as IR absorption spectra (where $\hat A$ is the dipole moment and $\hat B$ is the coupling between the electric fields and IR-active modes), Raman spectra (where $\hat A$ is the polarizability and $\hat B$ is the coupling of the electric field to Raman-active phonons), and phonon spectral functions (where $\hat A$ and $\hat B$ are the mass-rescaled atomic displacements).


The standard code computes the diagonal linear response, where $\hat A = \hat B$, for which a very efficient Lanczos algorithm has been developed.
You can prepare the system in the quantum/thermal equilibrium state by running a SSCHA calculation using the `python-sscha` package,
then use the `tdscha` code to compute the suscieptibility $\chi_{AA}(\omega)$ for the desired observable $\hat A$.


## Related Papers

1. **Monacelli et al., Physical Review B** 103, 104305 (2021) - Core TD-SCHA theory and Lanczos algorithm for linear response of anharmonic systems.
2. **Siciliano et al., Physical Review B** 107, 174307 (2023) - Wigner formulation of the TD-SCHA equations and application to phonon spectral functions - IR and Raman. 

## Getting Help

- Check the [examples](examples.md) for working templates
- See the [SSCHA website](http://www.sscha.eu) for tutorials
