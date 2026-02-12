# Examples and Templates

This section provides complete working examples for common TD-SCHA calculations, based on templates from the `Examples/` directory.

## Template Overview

The `Examples/Templates/` directory contains:

| File | Purpose |
|------|---------|
| `run_local.py` | Standard local calculation with mode/IR/Raman perturbations |
| `prepare_input_for_cluster.py` | Prepare input files for C++ executable on clusters |
| `restart_run.py` | Restart calculations from saved status |
| `get_spectral_function.py` | Extract spectral functions from results |

## Example 1: Basic Mode Perturbation

**File**: `Examples/Templates/run_local.py` (simplified)

```python
import cellconstructor as CC
import sscha.Ensemble
import tdscha.DynamicalLanczos as DL
import numpy as np

# ========== CONFIGURATION ==========
ORIGINAL_DYN = "dyn_start_"      # Initial dynamical matrix
FINAL_DYN = "dyn_final_"         # Final SSCHA dynamical matrix  
NQIRR = 1                        # Number of irreducible q-points
TEMPERATURE = 100                # K
ENSEMBLE_DIR = "ensemble/"       # Ensemble directory
N_CONFIGS = 10000                # Number of configurations
MODE_ID = 10                     # Mode to perturb (0-indexed)
LANCZOS_STEPS = 50               # Number of Lanczos steps
SAVE_DIR = "output"              # Output directory
# ===================================

# 1. Load dynamical matrices
dyn = CC.Phonons.Phonons(ORIGINAL_DYN, NQIRR)
final_dyn = CC.Phonons.Phonons(FINAL_DYN, NQIRR)

# 2. Load ensemble
ens = sscha.Ensemble.Ensemble(dyn, TEMPERATURE)
ens.load(ENSEMBLE_DIR, population_id=1, n_configs=N_CONFIGS)
ens.update_weights(final_dyn, TEMPERATURE)

# 3. Initialize Lanczos
lanczos = DL.Lanczos(ens)
lanczos.init(use_symmetries=True)

# 4. Prepare perturbation along mode 10
print(f"Mode {MODE_ID} frequency: {lanczos.w[MODE_ID] * CC.Units.RY_TO_CM:.1f} cm⁻¹")
lanczos.prepare_mode(MODE_ID)

# 5. Run calculation
lanczos.run_FT(LANCZOS_STEPS, save_dir=SAVE_DIR, 
               prefix="lanczos_m10", save_each=10)

# 6. Save final status
lanczos.save_status(f"{SAVE_DIR}/final.npz")
print("Calculation complete!")
```

**Quick test with example data**:

# doctest: +ELLIPSIS
>>> from tdscha.testing.test_data import load_test_ensemble
>>> import tdscha.DynamicalLanczos as DL
>>> ens = load_test_ensemble(n_configs=5)
>>> lanczos = DL.Lanczos(ens)
Generating Real space force constant matrix...
Time to generate the real space force constant matrix: ... s
TODO: the last time could be speedup with the FFT algorithm.
>>> lanczos.init(use_symmetries=True)
Time to get the symmetries [...] from spglib: ... s
Time to convert symmetries in the polarizaion space: ... s
Time to create the block_id array: ... s
>>> # Prepare perturbation for mode 5
>>> lanczos.prepare_mode(5)

...
>>> print(f"Prepared perturbation for mode 5")
Prepared perturbation for mode 5
>>> # Run a few Lanczos steps
>>> import sys
>>> from io import StringIO
>>> old_stdout = sys.stdout
>>> sys.stdout = StringIO()
>>> lanczos.run_FT(2, debug=False)
>>> sys.stdout = old_stdout
>>> print(f"Ran 2 Lanczos steps, coefficients: a={lanczos.a_coeffs[:2]}")
Ran 2 Lanczos steps, coefficients: a=...

## Example 2: IR Absorption Spectrum

Extend the previous example for IR calculations:

```python
# After loading ensemble and initializing Lanczos...

# Ensure final_dyn has effective charges
if final_dyn.effective_charges is None:
    raise ValueError("Dynamical matrix must have effective charges for IR")

# Prepare IR perturbation (x-polarized light)
pol_vec = np.array([1, 0, 0])  # x-polarization
lanczos.prepare_ir(pol_vec=pol_vec)

# Run calculation
lanczos.run_FT(100, save_dir="ir_output", prefix="lanczos_ir_x")

# Compute spectrum
w_array = np.linspace(0, 1500/CC.Units.RY_TO_CM, 1500)  # 0-1500 cm⁻¹
gf = lanczos.get_green_function_continued_fraction(w_array, smearing=2/CC.Units.RY_TO_CM)
spectrum = -np.imag(gf)

# Plot (or use tdscha-plot-data)
import matplotlib.pyplot as plt
plt.plot(w_array * CC.Units.RY_TO_CM, spectrum)
plt.xlabel("Frequency [cm⁻¹]")
plt.ylabel("IR absorption [a.u.]")
plt.title("x-polarized IR spectrum")
plt.show()
```

## Example 3: Raman Scattering (Unpolarized)

Compute unpolarized Raman spectrum:

```python
# After loading ensemble and initializing Lanczos...

# Ensure final_dyn has Raman tensor
if final_dyn.raman_tensor is None:
    raise ValueError("Dynamical matrix must have Raman tensor")

# Compute all 7 components of unpolarized Raman
prefactors = [45/9, 7/2, 7/2, 7/2, 7*3, 7*3, 7*3]
w_array = np.linspace(0, 1000/CC.Units.RY_TO_CM, 1000)
total_spectrum = np.zeros_like(w_array)

for i in range(7):
    lanczos.reset()  # Clear previous perturbation
    lanczos.prepare_unpolarized_raman(index=i)
    lanczos.run_FT(80, save_dir=f"raman_{i}", prefix=f"comp_{i}")
    
    gf = lanczos.get_green_function_continued_fraction(w_array, smearing=5/CC.Units.RY_TO_CM)
    spectrum = -np.imag(gf) * prefactors[i]
    total_spectrum += spectrum
    
    # Save component
    np.savetxt(f"raman_component_{i}.dat", 
               np.column_stack([w_array * CC.Units.RY_TO_CM, spectrum]))

# Save total spectrum
np.savetxt("raman_unpolarized_total.dat",
           np.column_stack([w_array * CC.Units.RY_TO_CM, total_spectrum]))

print("Total unpolarized Raman intensity computed")
```

## Example 4: StaticHessian Calculation

Compute free energy Hessian for stability analysis:

```python
import tdscha.StaticHessian as SH

# Load ensemble (same as Example 1)
ens = sscha.Ensemble.Ensemble(dyn, TEMPERATURE)
ens.load(ENSEMBLE_DIR, 1, N_CONFIGS)
ens.update_weights(final_dyn, TEMPERATURE)

# Initialize StaticHessian
hessian = SH.StaticHessian(ensemble=ens, verbose=True)

# Run calculation (200 steps, preconditioned CG)
hessian.run(n_steps=200, save_dir="hessian_out", 
            threshold=1e-7, algorithm="cg-prec")

# Retrieve Hessian as Phonons object
hessian_phonons = hessian.retrieve_hessian()

# Diagonalize to get frequencies
w, pols = hessian_phonons.DiagonalizeSupercell()

# Remove translations
masses = hessian_phonons.structure.get_masses_array()
trans = CC.Methods.get_translations(pols, masses)
w_nontrans = w[~trans]

print(f"Hessian frequencies ({len(w_nontrans)} modes):")
print(w_nontrans * CC.Units.RY_TO_CM)
print(f"Minimum frequency: {np.min(w_nontrans) * CC.Units.RY_TO_CM:.1f} cm⁻¹")

# Check stability (negative frequencies indicate instability)
if np.any(w_nontrans < 0):
    print("WARNING: System is unstable (imaginary frequencies)")
    n_unstable = np.sum(w_nontrans < 0)
    print(f"  {n_unstable} unstable modes")
```

**Quick test with example data**:

# doctest: +ELLIPSIS
>>> from tdscha.testing.test_data import load_test_ensemble
>>> import tdscha.StaticHessian as SH
>>> ens = load_test_ensemble(n_configs=5)
>>> hessian = SH.StaticHessian(ens); print(f"Initialized StaticHessian with {hessian.lanczos.n_modes} modes")
Generating Real space force constant matrix...
Time to generate the real space force constant matrix: ... s
TODO: the last time could be speedup with the FFT algorithm.
Generating Real space force constant matrix...
Time to generate the real space force constant matrix: ... s
TODO: the last time could be speedup with the FFT algorithm.
Time to get the symmetries [...] from spglib: ... s
Time to convert symmetries in the polarizaion space: ... s
Time to create the block_id array: ... s
Initialized StaticHessian with ... modes
>>> # Note: Running the full minimization is time-consuming for doctests
>>> # hessian.run(n_steps=2) would be skipped in quick validation

## Example 5: Cluster Calculation with C++ Executable

For large systems, use the C++ executable on HPC clusters:

### Step 1: Prepare Input Files (Python)

```python
# prepare_input_for_cluster.py
import tdscha.DynamicalLanczos as DL

# Initialize Lanczos as usual
lanczos = DL.Lanczos(ens)
lanczos.init()
lanczos.prepare_mode(10)

# Prepare input files for C++ executable
lanczos.prepare_input_files(
    root_name="tdscha_calc",
    n_steps=200,
    directory="cluster_input",
    run_symm=False  # Use symmetric Lanczos
)

print("Input files prepared in 'cluster_input/'")
print("Copy to cluster and run: ./tdscha-lanczos.x > output.txt")
```

### Step 2: Run on Cluster (Bash)

```bash
#!/bin/bash
#SBATCH --job-name=tdscha
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00

# Load modules
module load intel mpich

# Copy input files
cp -r cluster_input/* $SLURM_SUBMIT_DIR/

# Run C++ executable
mpirun -np 64 ./tdscha-lanczos.x > tdscha_calc.stdout

# Convert output
tdscha-output2abc tdscha_calc.stdout tdscha_calc.abc

# Basic analysis
tdscha-plot-data tdscha_calc.abc 0 1000 5
tdscha-convergence-analysis tdscha_calc.abc 5
```

### Step 3: Analyze Results (Python)

```python
# analyze_cluster.py
import tdscha.DynamicalLanczos as DL

# Load results from .abc file
lanczos = DL.Lanczos()
lanczos.load_abc("tdscha_calc.abc")

# Compute spectrum
w_array = np.linspace(0, 1000/CC.Units.RY_TO_CM, 1000)
gf = lanczos.get_green_function_continued_fraction(w_array, smearing=5/CC.Units.RY_TO_CM)
spectrum = -np.imag(gf)

# Save for plotting
np.savetxt("spectrum.dat", np.column_stack([w_array * CC.Units.RY_TO_CM, spectrum]))
```

## Example 6: Convergence Analysis

Analyze Lanczos convergence for optimal step count:

```python
# convergence_analysis.py
import tdscha.DynamicalLanczos as DL
import numpy as np

# Load final Lanczos state
lanczos = DL.Lanczos()
lanczos.load_status("output/final.npz")

# Analyze convergence of static frequency
n_steps = len(lanczos.a_coeffs) - 1
static_freqs = np.zeros(n_steps)

for i in range(1, n_steps + 1):
    # Truncate coefficients to i steps
    a_partial = lanczos.a_coeffs[:i]
    b_partial = lanczos.b_coeffs[:i-1] if i > 1 else []
    c_partial = lanczos.c_coeffs[:i-1] if i > 1 else []
    
    # Temporary Lanczos object with truncated coefficients
    temp_lanc = DL.Lanczos()
    temp_lanc.a_coeffs = a_partial
    temp_lanc.b_coeffs = b_partial
    temp_lanc.c_coeffs = c_partial
    temp_lanc.use_wigner = lanczos.use_wigner
    temp_lanc.shift_value = lanczos.shift_value
    
    # Compute static Green's function
    gf0 = temp_lanc.get_green_function_continued_fraction(np.array([0]), use_terminator=False)[0]
    static_freq = np.sign(np.real(gf0)) * np.sqrt(np.abs(1/np.real(gf0)))
    static_freqs[i-1] = static_freq * CC.Units.RY_TO_CM

# Plot convergence
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(range(1, n_steps+1), static_freqs, 'o-')
plt.xlabel("Lanczos steps")
plt.ylabel("Static frequency [cm⁻¹]")
plt.title("Convergence of static frequency")

# Or use CLI tool
print("Run: tdscha-convergence-analysis output/final.npz 5")
```

## Example 7: Comparing Normal vs Wigner Representation

```python
# comparison_wigner.py
import tdscha.DynamicalLanczos as DL

results = {}
for use_wigner in [False, True]:
    print(f"\n=== {'Wigner' if use_wigner else 'Normal'} representation ===")
    
    lanczos = DL.Lanczos(ensemble, use_wigner=use_wigner)
    lanczos.init()
    lanczos.prepare_mode(10)
    
    # Run with same parameters
    lanczos.run_FT(80, save_dir=f"output_{'wigner' if use_wigner else 'normal'}")
    
    # Compute spectrum
    w_array = np.linspace(0, 500/CC.Units.RY_TO_CM, 500)
    gf = lanczos.get_green_function_continued_fraction(w_array, smearing=5/CC.Units.RY_TO_CM)
    
    results['wigner' if use_wigner else 'normal'] = {
        'frequencies': w_array * CC.Units.RY_TO_CM,
        'spectrum': -np.imag(gf),
        'a_coeffs': lanczos.a_coeffs,
        'convergence': len(lanczos.a_coeffs)
    }

# Compare
plt.figure()
for label, data in results.items():
    plt.plot(data['frequencies'], data['spectrum'], label=label, alpha=0.7)
plt.xlabel("Frequency [cm⁻¹]")
plt.ylabel("Spectrum [a.u.]")
plt.legend()
plt.title("Normal vs Wigner representation")
plt.show()
```

## Example 8: Restarting Calculations

```python
# restart_run.py
import tdscha.DynamicalLanczos as DL

# Method 1: Continue from saved status
lanczos = DL.Lanczos()
lanczos.load_status("output/tdscha_lanczos_STEP50")  # Load checkpoint at step 50
lanczos.run_FT(100, save_dir="output_restart", prefix="restart")  # Continue to step 150

# Method 2: Start new calculation with same ensemble
lanczos2 = DL.Lanczos(ensemble)  # Same ensemble as before
lanczos2.init()
lanczos2.prepare_mode(10)
lanczos2.run_FT(150, save_dir="output_full")  # Start from scratch

# Compare results
print(f"Restarted: {len(lanczos.a_coeffs)} coefficients")
print(f"Fresh: {len(lanczos2.a_coeffs)} coefficients")
```

## Example 9: Custom Perturbation Vector

```python
# custom_perturbation.py
import numpy as np

# Create custom perturbation (e.g., specific atomic displacement)
nat_sc = ensemble.nat  # Atoms in supercell
custom_vec = np.zeros(3 * nat_sc)

# Displace atom 0 in x-direction
custom_vec[0] = 1.0  # x of atom 0
# custom_vec[1] = 0.0  # y of atom 0 (default)
# custom_vec[2] = 0.0  # z of atom 0 (default)

# Prepare perturbation (masses_exp=1 for displacements, -1 for forces)
lanczos.prepare_perturbation(custom_vec, masses_exp=1)

# Run calculation
lanczos.run_FT(100, save_dir="custom_pert")

# Note: spectrum will be response to this specific displacement pattern
```

## Example 10: Full Workflow with CLI Analysis

Complete workflow combining Python and CLI:

```bash
#!/bin/bash
# full_workflow.sh

# 1. Run Python calculation
python run_calculation.py

# 2. Convert to .abc if using C++ executable
tdscha-output2abc lanczos.stdout lanczos.abc

# 3. Analyze convergence
tdscha-convergence-analysis lanczos_final.npz 5
mv convergence.png convergence_analysis.png

# 4. Plot spectra at different resolutions
tdscha-plot-data lanczos_final.npz 0 200 0.5
mv spectrum.png spectrum_highres.png

tdscha-plot-data lanczos_final.npz 0 1000 5
mv spectrum.png spectrum_lowres.png

# 5. For Hessian calculations
tdscha-hessian-convergence hessian_steps/ hessian_calculation
mv hessian_convergence.png hessian_analysis.png

echo "Workflow complete! Check *.png files for results."
```

## Directory Structure for Examples

```
project/
├── dyn_start_*           # Initial dynamical matrix files
├── dyn_final_*          # Final SSCHA dynamical matrix
├── ensemble/            # SSCHA ensemble
│   ├── ensemble_pop1_x.dat
│   ├── ensemble_pop1_f.dat
│   └── ensemble_pop1_rho.dat
├── run_local.py         # Main calculation script
├── output/              # Lanczos output
│   ├── tdscha_lanczos_STEP*.npz
│   └── final.npz
├── spectrum.dat         # Final spectrum
└── convergence.png      # Convergence analysis
```

## Common Parameters and Defaults

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `LANCZOS_STEPS` | 50-200 | Steps for convergence |
| `SAVE_EACH` | 5-10 | Checkpoint frequency |
| `SMEARING` | 1-10 cm⁻¹ | Lorentzian broadening |
| `THRESHOLD` | 1e-6 to 1e-8 | Convergence threshold |
| `MODE` | Auto (Julia > C > Python) | Computation mode |

## Troubleshooting Examples

See the `Examples/Comparison/` directory for:
- `normal.py` vs `wigner.py` - Representation comparison
- `convergence.py` - Step convergence analysis
- `comparison.py` - Method comparison utilities

And `Examples/example_IR_Raman_2p/` for:
- `IR_UNPOL/` - Unpolarized IR with 1ph/2ph processes
- `RAMAN_UNPOL/` - Unpolarized Raman calculations
- Complete README files with instructions