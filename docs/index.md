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
   - [Tools](api/tools.md) - Krylov solvers and linear algebra utilities
   - [Perturbations](api/perturbations.md) - IR and Raman response functions
   - [Parallel](api/parallel.md) - MPI parallelization utilities

## Key Features

- **Dynamical linear response** beyond harmonic approximation
- **Lanczos algorithm** for efficient spectral function computation
- **Multiple perturbation types**: single phonon mode, IR, Raman (polarized/unpolarized)
- **Parallel execution**: MPI, Julia fast mode, C extensions
- **Symmetry-aware** calculations for efficiency
- **Static Hessian** computation via sparse linear algebra
- **Integration** with SSCHA equilibrium results

## Theoretical Foundation

TD-SCHA extends the Self-Consistent Harmonic Approximation (SCHA) to time-dependent perturbations, enabling computation of:
- Infrared (IR) absorption spectra
- Raman scattering spectra  
- Dynamical structure factor
- Phonon spectral functions with full anharmonicity

The method computes the dynamical susceptibility via a Lanczos algorithm, yielding the Green's function in continued fraction representation. This allows efficient calculation of spectral properties without artificial broadening.

## Related Papers

1. **Monacelli et al., Physical Review B** - Core TD-SCHA theory and equations (referenced as Eq. K4 in code)
2. **Raman intensity formulas** - J. Phys. Chem. DOI: 10.1021/jp5125266

## Getting Help

- Check the [examples](examples.md) for working templates
- Use the CLI tools for analysis and visualization
 - Refer to automatically generated API documentation for detailed method specifications
- See the [SSCHA website](http://www.sscha.eu) for tutorials