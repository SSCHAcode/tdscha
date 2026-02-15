# Theoretical Background

## What is TD-SCHA?

Time-Dependent Self-Consistent Harmonic Approximation (TD-SCHA) is a theory for simulating quantum nuclear motion in materials with strong anharmonicity. It extends the equilibrium Self-Consistent Harmonic Approximation (SCHA) to time-dependent perturbations, enabling computation of dynamical linear response properties.

TD-SCHA stands within the SSCHA (Stochastic Self-Consistent Harmonic Approximation) ecosystem, which provides the equilibrium statistical ensemble. TD-SCHA then computes the dynamical susceptibility on top of this ensemble, capturing both quantum and thermal fluctuations non-perturbatively.

## Why Linear Response?

Linear response theory connects small perturbations to measurable experimental signals. For quantum nuclei in materials, this enables calculation of:

- **Infrared (IR) absorption spectra** - Response to electromagnetic fields
- **Raman scattering spectra** - Response to polarizability fluctuations  
- **Dynamical structure factor** - Neutron scattering cross-sections
- **Phonon spectral functions** - Full anharmonic density of states

These observables are expressed through the dynamical susceptibility $\chi(\omega)$, which TD-SCHA computes via the Lanczos algorithm.

## Key Quantities Calculable with TD-SCHA

1. **One-phonon spectral function** - $\mathcal{S}(\omega) = -\frac{1}{\pi} \mathrm{Im} G(\omega)$
2. **IR absorption coefficient** - $\alpha(\omega) \propto \omega \mathrm{Im} \chi_{\mathrm{IR}}(\omega)$
3. **Raman intensity** - $I(\omega) \propto (n(\omega)+1) \mathrm{Im} \chi_{\mathrm{Raman}}(\omega)$
4. **Static susceptibility** - $\chi(0)$ for elastic constants
5. **Free energy Hessian** - Second derivative of free energy w.r.t. atomic displacements


