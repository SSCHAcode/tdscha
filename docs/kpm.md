# Kernel Polynomial Method

The Kernel Polynomial Method (KPM) is an alternative algorithm to compute the dynamical linear response in TD-SCHA. It uses Chebyshev polynomials to express the spectral function, which provides a different trade-off compared to the standard Lanczos algorithm.

## Why KPM?

KPM never suffers from loss of orthogonality in the basis vectors. The Chebyshev polynomials are orthonormal by construction, so the computed spectral function does not develop spurious artifacts from numerical roundoff even for large numbers of steps. This makes KPM particularly reliable for systems where the Lanczos algorithm might accumulate numerical errors.

The price to pay is that KPM typically requires more steps than Lanczos to achieve the same spectral resolution, since the Chebyshev basis is generic rather than optimized for the specific spectral features of the problem.

## Spectral Bounds

KPM requires specifying the bounds of the eigenvalue spectrum of the Liouvillian operator. These bounds must enclose all eigenvalues, otherwise the method fails to converge to the correct spectral function. The bounds are specified through a `bound_factor` parameter that controls how much the KPM bounds extend beyond the estimated spectral width.

The relationship between frequency ω and Liouvillian eigenvalue λ depends on the Wigner formalism used. In Wigner mode, λ = -ω², so the eigenvalues are negative and bounded above by zero. The spectral width scales as (2ω_max)², where ω_max is the maximum phonon frequency in the system.

A `bound_factor = 1.2` means the KPM bounds extend 20% beyond the theoretical spectral width. The default value of 1.2 works well in practice: it is wide enough to safely enclose all spectral features, while being tight enough to maintain good resolution.

## Frequency Resolution

The number of steps N determines the frequency resolution through the Jackson kernel. The frequency resolution Δω scales as:

Δω ≈ π × rescale_a / (2 × ω_min × N)

where ω_min is the smallest phonon frequency at the perturbation q-point, and rescale_a is the half-width of the normalized eigenvalue interval. Larger bound factors increase rescale_a, which in turn requires more steps to achieve the same resolution.

After preparing a perturbation, you can estimate the required steps for a target precision using the `estimate_kpm_steps` method:

```python
kpm = QSpaceKPM.QSpaceKPM(ensemble)
kpm.init()
kpm.prepare_mode_q(0, 5)

# Estimate steps for 2 cm^-1 frequency precision
n_steps = kpm.estimate_kpm_steps(precision_cm=2.0, bound_factor=1.2)
kpm.run_KPM(n_steps)
```

## A Practical Example

```python
import tdscha.QSpaceKPM as QKPM

# Setup
kpm = QKPM.QSpaceKPM(ensemble)
kpm.init()
kpm.prepare_mode_q(iq=0, band_index=5)

# Estimate and run
n_steps = kpm.estimate_kpm_steps(1.0)  # 1 cm^-1 precision
kpm.run_KPM(n_steps)

# Compute the spectral function
w_cm = np.linspace(0, 200, 1000)
w_ry = w_cm / CC.Units.RY_TO_CM
spectral = kpm.get_spectral_function_KPM(w_ry, regularization="jackson")
```

Values of `bound_factor` larger than 1.5 can introduce visible baseline artifacts because the Jackson kernel produces broader peaks when the normalized eigenvalue range is larger. If you need larger bounds for safety, increase the number of steps to compensate.
