# Quick Start Guide

This guide walks through a complete TD-SCHA calculation from ensemble preparation to spectral analysis using CLI tools.

## Quick Test with Example Data

To quickly test, we provide a density matrix at equilibrium of the SnTe alloy on a 2x2x2 supercell.
In the following example, we run a dynamical linear response calculation to compute the IR spectrum, with the radiation field polarized alogn the x-axis.

We first load the SSCHA ensemble (which contains the equilibrium density matrix) and initialize the Lanczos algorithm:

```python
import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL

# Load the ensemble (here we use one provided by the tdscha package for testing)
ens = load_test_ensemble(n_configs = 10)

# Initialize the TD-SCHA calculation via the Lanczos algorithm
lanczos = DL.Lanczos(ens)
lanczos.init()

# Prepare the IR perturbation with polarization along x
lanczos.prepare_ir(pol_vec=[1, 0, 0])

# Run the linear-response calculation at finite temperature (this is specified in the ensemble) 
# For 10 Lanczos steps. Usually 100-200 steps are needed for convergence, but this is just a quick test.
lanczos.run_FT(10)

# Save the results
lanczos.save_status("ir_spectrum_x.npz")
```

Then you can plot the spectrum using the CLI tool:

```bash
tdscha-plot-data ir_spectrum_x.npz 0 1000 2
```

Here the parameters specify the frequency range (0-1000 cm⁻¹) and the smearing (2 cm⁻¹) for the plot.


```pycon
### Basic Lanczos Workflow

Here's a complete example showing mode perturbation and a few Lanczos steps:

```pycon
# doctest: +ELLIPSIS
>>> from tdscha.testing.test_data import load_test_ensemble
>>> import tdscha.DynamicalLanczos as DL
>>> import numpy as np
>>> ens = load_test_ensemble(n_configs=5)  # Small subset for speed
>>> lanczos = DL.Lanczos(ens)
Generating Real space force constant matrix...
Time to generate the real space force constant matrix: ... s
TODO: the last time could be speedup with the FFT algorithm.
>>> lanczos.verbose = False
>>> lanczos.init(use_symmetries=True)
Time to get the symmetries [...] from spglib: ... s
Time to convert symmetries in the polarizaion space: ... s
Time to create the block_id array: ... s
>>> # Prepare perturbation for mode 10
>>> lanczos.prepare_mode(10)

...
>>> # Run 2 Lanczos steps
>>> lanczos.run_FT(2, debug=False)
...
>>> print(f"Lanczos coefficients after 2 steps: a={lanczos.a_coeffs[:2]}")
Lanczos coefficients after 2 steps: a=[np.float64(...), np.float64(...)]
```

## Prerequisites

Ensure you have:
1. A converged SSCHA ensemble (configurations, forces, weights)
2. The final SSCHA dynamical matrix
3. TD-SCHA installed (see [Installation](installation.md))

## Example Workflow

### Step 1: Prepare the Ensemble

First, load your SSCHA results:

```python
import cellconstructor as CC
import sscha.Ensemble
import numpy as np

# Load dynamical matrices
dyn_initial = CC.Phonons.Phonons("dyn_start_", nqirr=1)
dyn_final = CC.Phonons.Phonons("dyn_final_", nqirr=1)

# Create ensemble
temperature = 100  # K
ens = sscha.Ensemble.Ensemble(dyn_initial, temperature)

# Load configurations (adjust paths as needed)
ens.load("ensemble_data/", population_id=1, n_configs=10000)

# Update weights to final dynamical matrix
ens.update_weights(dyn_final, temperature)
```

### Step 2: Run TD-SCHA Calculation

Choose a perturbation type and run Lanczos:

#### Option A: Single Phonon Mode

```python
import tdscha.DynamicalLanczos as DL

# Initialize Lanczos
lanczos = DL.Lanczos(ens)
lanczos.init(use_symmetries=True)

# Prepare perturbation along mode 10 (adjust as needed)
mode_id = 10
print(f"Mode {mode_id} frequency: {lanczos.w[mode_id] * CC.Units.RY_TO_CM:.1f} cm⁻¹")
lanczos.prepare_mode(mode_id)

# Run Lanczos (20 steps, save each 5)
lanczos.run_FT(20, save_dir="tdscha_output", 
               prefix="lanczos_m10", save_each=5)

# Save final status
lanczos.save_status("tdscha_output/lanczos_final.npz")
```

#### Option B: IR Absorption

```python
# Ensure dyn_final has effective charges
lanczos.prepare_ir(pol_vec=np.array([1, 0, 0]))  # x-polarization
lanczos.run_FT(20, save_dir="tdscha_ir", prefix="lanczos_ir")
```

#### Option C: Raman Scattering

```python
# Ensure dyn_final has Raman tensor
lanczos.prepare_raman(pol_vec_in=[1,0,0], pol_vec_out=[1,0,0])
lanczos.run_FT(20, save_dir="tdscha_raman", prefix="lanczos_raman")
```

### Step 3: Analyze Results with CLI

TD-SCHA provides four CLI tools for analysis:

#### 1. Convert Output to ABC Format

If you ran the C++ executable `tdscha-lanczos.x`, convert its output:

```bash
tdscha-output2abc lanczos.stdout lanczos.abc
```

#### 2. Plot Spectrum

Plot the spectral function from .abc or .npz files:

```bash
# Basic plot (0-5000 cm⁻¹, 5 cm⁻¹ smearing)
tdscha-plot-data lanczos_final.npz

# Custom frequency range and smearing
tdscha-plot-data lanczos.abc 0 1000 2
```

#### 3. Analyze Convergence

Check how the spectral function converges with Lanczos steps:

```bash
tdscha-convergence-analysis lanczos_final.npz 5
```

This generates three plots:
- Static frequency vs Lanczos steps
- Spectral function evolution (without terminator)
- Spectral function evolution (with terminator)
- Final converged spectrum

#### 4. Analyze Hessian Convergence (StaticHessian)

For StaticHessian calculations:

```bash
tdscha-hessian-convergence hessian_steps/ hessian_calculation
```

## Complete Example Script

Here's a complete script combining Python calculation and CLI analysis:

```python
# run_calculation.py
import cellconstructor as CC
import sscha.Ensemble
import tdscha.DynamicalLanczos as DL
import subprocess
import sys

# 1. Load ensemble
dyn = CC.Phonons.Phonons("dyn_prefix_", 1)
final_dyn = CC.Phonons.Phonons("dyn_final_", 1)
ens = sscha.Ensemble.Ensemble(dyn, 100)
ens.load("ensemble/", 1, 10000)
ens.update_weights(final_dyn, 100)

# 2. Run TD-SCHA
lanc = DL.Lanczos(ens)
lanc.init()
lanc.prepare_mode(10)
lanc.run_FT(50, save_dir="output", prefix="calc", save_each=10)
lanc.save_status("output/final.npz")

# 3. Run CLI analysis
print("\n=== Running CLI Analysis ===")
subprocess.run(["tdscha-convergence-analysis", "output/final.npz", "5"])
subprocess.run(["tdscha-plot-data", "output/final.npz", "0", "500", "2"])
```

Run with MPI for parallel execution:

```bash
mpirun -np 4 python run_calculation.py
```

## Understanding the Output

### Lanczos Coefficients
The calculation produces three arrays:
- `a_coeffs` - Diagonal elements of tridiagonal matrix
- `b_coeffs`, `c_coeffs` - Off-diagonal elements

These define the continued fraction for the Green's function.

### Spectral Function
The spectral function $S(\omega) = -\frac{1}{\pi}\mathrm{Im}G(\omega)$ contains:
- **Peak positions** - Renormalized phonon frequencies
- **Peak widths** - Phonon lifetimes from anharmonicity
- **Spectral weight** - Mode intensities

### Convergence Metrics
- **Static limit** - $\omega_{\mathrm{static}} = \sqrt{1/G(0)}$ should converge
- **Terminator effect** - Spectral function should stabilize with Lanczos steps

## Quick CLI Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `tdscha-convergence-analysis` | Analyze Lanczos convergence | `tdscha-convergence-analysis file.npz 5` |
| `tdscha-plot-data` | Plot spectrum | `tdscha-plot-data file.abc 0 1000 2` |
| `tdscha-output2abc` | Convert stdout to .abc | `tdscha-output2abc stdout.txt output.abc` |
| `tdscha-hessian-convergence` | Plot Hessian convergence | `tdscha-hessian-convergence dir/ prefix` |

## Next Steps

- Explore [In-Depth Usage](usage.md) for advanced features
- Learn about [StaticHessian](static-hessian.md) for free energy calculations
- Check [Examples](examples.md) for complete workflows
- Refer to [API Documentation](api/dynamical_lanczos.md) for detailed method specifications
