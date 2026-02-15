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

## Theoretical Framework

### Main TD-SCHA Equations

The core equation (Eq. K4 in Monacelli PRB appendix) governs the evolution of the density matrix under perturbations:

$$
\begin{bmatrix}
0   &  -X''  &  0 \\
-Z   &  -X    & 0 \\
-Z'  &  -X'   & 0
\end{bmatrix}
\begin{bmatrix}
R^{(1)} \\
\Upsilon^{(1)} - a'^{(1)} \\
\mathrm{Re}A^{(1)} - b'^{(1)}
\end{bmatrix}
= \begin{bmatrix}
\cdots
\end{bmatrix}
$$

Where:
- $R^{(1)}$ is the first-order response in displacement coordinates
- $\Upsilon^{(1)}$ and $A^{(1)}$ are auxiliary variables in the SCHA formalism
- $X, X', X''$ and $Z, Z'$ are operators containing anharmonic interactions

This matrix equation defines the linear operator $\mathcal{L}$ whose Green's function $G(\omega) = (-\mathcal{L} - \omega^2)^{-1}$ gives the dynamical response.

### Lanczos Algorithm for Green's Function

The Lanczos algorithm transforms $\mathcal{L}$ into a tridiagonal matrix:

$$
T_n = 
\begin{bmatrix}
a_0 & b_0 & 0 & \cdots & 0 \\
c_0 & a_1 & b_1 & \cdots & 0 \\
0 & c_1 & a_2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & a_{n-1}
\end{bmatrix}
$$

The Green's function element $G_{00}(\omega) = \langle v| (-\mathcal{L} - \omega^2)^{-1} |v\rangle$ for perturbation $|v\rangle$ is then expressed as a continued fraction:

$$
G_{00}(\omega) = \frac{1}{a_0 - \omega^2 - \frac{b_0 c_0}{a_1 - \omega^2 - \frac{b_1 c_1}{a_2 - \omega^2 - \cdots}}}
$$

This representation converges rapidly and allows inclusion of a "terminator" to approximate infinite continued fractions.

### Wigner vs Normal Representation

TD-SCHA implements two formalisms:

**Normal representation** (default):
- Inverts $(-\mathcal{L} - \omega^2)$
- Directly gives susceptibility $\chi(\omega)$

**Wigner representation**:
- Inverts $(\mathcal{L}_w + \omega^2)$  
- More efficient for certain calculations
- Required for two-phonon response

The choice affects sign conventions in the continued fraction but yields identical physical results.

### From Green's Function to Experimental Signals

The imaginary part of the Green's function gives the spectral function:

$$
\mathcal{S}(\omega) = -\frac{1}{\pi} \mathrm{Im} G(\omega)
$$

Experimental signals are then:

**IR absorption**:
$$
\alpha(\omega) \propto \omega \sum_{\alpha\beta} \epsilon_\alpha^\mathrm{in} \epsilon_\beta^\mathrm{out} \mathrm{Im} \chi_{\alpha\beta}^\mathrm{IR}(\omega)
$$
where $\chi^\mathrm{IR}$ uses effective charges as perturbation.

**Raman scattering** (unpolarized):
$$
I_\mathrm{unpol}(\omega) = \frac{45}{9}\alpha^2 + \frac{7}{2}(\beta_1^2 + \beta_2^2 + \beta_3^2) + 7\cdot 3(\beta_4^2 + \beta_5^2 + \beta_6^2)
$$
with $\alpha = \frac{1}{3}(xx + yy + zz)$ and $\beta_i$ components defined by Raman tensor contractions (see DOI: 10.1021/jp5125266).

**Dynamical structure factor**:
$$
S(\mathbf{q},\omega) \propto \sum_{\mu} |F_\mu(\mathbf{q})|^2 \mathcal{S}_\mu(\omega)
$$
where $F_\mu(\mathbf{q})$ are structure factors for mode $\mu$.

## Implementation Overview

The TD-SCHA implementation:

1. **Takes SSCHA ensemble** - Configurations, weights, forces
2. **Projects onto phonon modes** - Mass-weighted displacements/forces in polarization basis
3. **Builds linear operator $\mathcal{L}$** - Contains harmonic, cubic, quartic terms
4. **Runs Lanczos algorithm** - Generates $a_n$, $b_n$, $c_n$ coefficients
5. **Computes Green's function** - Via continued fraction with optional terminator
6. **Extracts spectra** - Imaginary part gives physical observables

The algorithm exploits crystal symmetries to reduce computational cost and supports parallel execution via MPI or Julia multithreading.

## References

1. **Monacelli et al., Physical Review B** - Core TD-SCHA theory, Eq. (K4) in appendix
2. **Raman intensity formulas** - J. Phys. Chem. **2015**, 119, 24, 7287–7296, DOI: 10.1021/jp5125266
3. **Bianco et al.** - Static Hessian algorithm for free energy calculations