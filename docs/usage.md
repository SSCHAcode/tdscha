# In-Depth Usage Guide

## Choosing Perturbation Types

TD-SCHA supports three main perturbation types, each with specific use cases:

### 1. Single Phonon Mode

Perturb along a specific phonon eigenmode of the SSCHA dynamical matrix:

```python
lanczos.prepare_mode(mode_index)
```

**When to use**: Studying specific phonon modes, testing convergence, debugging.

**Parameters**:
- `mode_index`: Integer (0 to n_modes-1, excluding translations)
- Mode frequencies available in `lanczos.w` array

**Example**:
```python
# Find low-frequency modes
low_freq_indices = np.argsort(lanczos.w)[:5]
for idx in low_freq_indices:
    freq_cm = lanczos.w[idx] * CC.Units.RY_TO_CM
    print(f"Mode {idx}: {freq_cm:.1f} cm⁻¹")

# Study softest mode
lanczos.prepare_mode(low_freq_indices[0])
```

**Example with test data**:

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

### 2. Infrared (IR) Absorption

Compute IR spectra using effective charges:

```python
lanczos.prepare_ir(pol_vec=np.array([1,0,0]), 
                   effective_charges=None)
```

**When to use**: IR absorption spectra, dielectric response.

**Parameters**:
- `pol_vec`: Light polarization vector (cartesian)
- `effective_charges`: Override default from dynamical matrix

**Example**:
```python
# x-polarized IR
lanczos.prepare_ir(pol_vec=[1,0,0])

# y-polarized IR  
lanczos.prepare_ir(pol_vec=[0,1,0])

# Circular polarization (requires two calculations)
lanczos.prepare_ir(pol_vec=[1,0,0])
# Then combine with pol_vec=[0,1,0] results
```

### 3. Raman Scattering

Compute Raman spectra using Raman tensor:

#### Polarized Raman
```python
lanczos.prepare_raman(pol_vec_in=[1,0,0], 
                      pol_vec_out=[1,0,0],
                      mixed=False)
```

#### Unpolarized Raman Components
```python
# Compute individual components (0-6)
for i in range(7):
    lanczos.prepare_unpolarized_raman(index=i)
    lanczos.run_FT(50, save_dir=f"raman_{i}")
    # Weight by prefactors later
```

**Raman tensor components**:
- `0`: $\alpha^2 = (xx + yy + zz)^2/9$
- `1`: $\beta_1^2 = (xx - yy)^2/2$
- `2`: $\beta_2^2 = (xx - zz)^2/2$
- `3`: $\beta_3^2 = (yy - zz)^2/2$
- `4`: $\beta_4^2 = 3xy^2$
- `5`: $\beta_5^2 = 3xz^2$
- `6`: $\beta_6^2 = 3yz^2$

**Total unpolarized intensity**:
$$
I_{\text{unpol}} = 45\alpha^2 + 7(\beta_1^2 + \beta_2^2 + \beta_3^2 + \beta_4^2 + \beta_5^2 + \beta_6^2)
$$

## Running Unpolarized Raman

### Complete Workflow

```python
import numpy as np
import tdscha.DynamicalLanczos as DL

# Initialize Lanczos
lanczos = DL.Lanczos(ensemble)
lanczos.init()

# Prefactors for unpolarized Raman
prefactors = [45/9, 7/2, 7/2, 7/2, 7*3, 7*3, 7*3]

# Arrays to store spectra
w_array = np.linspace(0, 1000/CC.Units.RY_TO_CM, 1000)
spectra = np.zeros((7, len(w_array)))

# Compute each component
for i in range(7):
    lanczos.reset()  # Clear previous perturbation
    lanczos.prepare_unpolarized_raman(index=i)
    lanczos.run_FT(50)
    
    gf = lanczos.get_green_function_continued_fraction(w_array)
    spectra[i] = -np.imag(gf) * prefactors[i]

# Sum components
total_spectrum = np.sum(spectra, axis=0)
```

### Fluctuation-Corrected Raman

For Raman tensor fluctuations (beyond harmonic approximation):

```python
lanczos.prepare_unpolarized_raman_FT(
    index=0,
    eq_raman_tns=equilibrium_raman_tensor,
    use_symm=True,
    ens_av_raman=ensemble_average_raman,
    raman_tns_ens=raman_tensor_ensemble,
    add_2ph=True
)
```

## Parallel Execution Modes

TD-SCHA supports four computation modes:

### Mode Selection

```python
# Auto-select fastest available
lanczos = DL.Lanczos(ensemble)  # Default: Julia if available, else C serial

# Manual selection
from tdscha.DynamicalLanczos import (
    MODE_SLOW_SERIAL,    # 0: Pure Python (testing)
    MODE_FAST_SERIAL,    # 1: C extension  
    MODE_FAST_MPI,       # 2: C with MPI
    MODE_FAST_JULIA      # 3: Julia (fastest)
)

lanczos = DL.Lanczos(ensemble, mode=MODE_FAST_JULIA)
```

**Testing mode selection with example data**:

# doctest: +ELLIPSIS
>>> from tdscha.testing.test_data import load_test_ensemble
>>> import tdscha.DynamicalLanczos as DL
>>> ens = load_test_ensemble(n_configs=5)
>>> # Check if Julia is available
>>> if DL.is_julia_enabled():
...     print("Julia mode available")
... else:
...     print("Julia not available, using C mode")
Julia mode available...

### MPI Parallelization

```bash
# Run with MPI
mpirun -np 16 python script.py

# Hybrid MPI+OpenMP
export OMP_NUM_THREADS=4
mpirun -np 4 --bind-to socket python script.py
```

**MPI-specific considerations**:
- Only rank 0 prints output (via `pprint()`)
- Collective operations synchronized with `Parallel.barrier()`
- Load balancing automatic via symmetry decomposition

### Julia Fast Mode

Requirements:
1. Julia installed and in PATH
2. `julia` Python package installed
3. Required Julia packages: `SparseArrays`, `InteractiveUtils`

```python
# Check if Julia available
if DL.is_julia_enabled():
    lanczos = DL.Lanczos(ensemble, mode=MODE_FAST_JULIA)
else:
    print("Julia not available, falling back to C")
```

**Performance**: 2-10x speedup over C implementation for large systems.

## Gamma-Only Trick

For $\Gamma$-point-only calculations, use point-group symmetries only and project translations:

 ```python
lanczos.gamma_only = True
lanczos.init(use_symmetries=True)
 ```

 **Example with test data**:

 # doctest: +ELLIPSIS
 >>> from tdscha.testing.test_data import load_test_ensemble
 >>> import tdscha.DynamicalLanczos as DL
 >>> ens = load_test_ensemble(n_configs=5)
 >>> lanczos = DL.Lanczos(ens)
Generating Real space force constant matrix...
Time to generate the real space force constant matrix: ... s
TODO: the last time could be speedup with the FFT algorithm.
 >>> lanczos.gamma_only = True
 >>> lanczos.init(use_symmetries=True)
Time to get the symmetries [...] from spglib: ... s
Time to convert symmetries in the polarizaion space: ... s
Time to create the block_id array: ... s
  >>> print(f"Initialized with gamma_only={lanczos.gamma_only}")
 Initialized with gamma_only=True

 **How it works**:
1. Separates point-group from translational symmetries
2. Builds translation operators in mode space: $T_R^{\text{mode}} = P^T P_R P$
3. Applies translation projector: $P = \frac{1}{N_{\text{cells}}} \sum_R T_R^{\text{mode}}$

**Benefits**:
- Reduces symmetry operations from $N_{\text{total}}$ to $N_{\text{point-group}}$
- Speedup: $N_{\text{total}} / N_{\text{point-group}}$ (typically 4-48x)

**When to use**: Large supercells, $\Gamma$-point properties only.

## Wigner vs Normal Representation

### Normal Representation (Default)

```python
lanczos.use_wigner = False  # Default
```

**Equations**: Inverts $(-\mathcal{L} - \omega^2)$

**Pros**:
- Direct physical interpretation
- Standard linear response formalism
- Compatible with all perturbation types

**Cons**:
- Slower convergence for some systems

### Wigner Representation

```python
lanczos.use_wigner = True
```

**Equations**: Inverts $(\mathcal{L}_w + \omega^2)$

**Pros**:
- Faster convergence for low-temperature systems
- Required for two-phonon response
- More stable for certain anharmonic potentials

**Cons**:
- Different sign conventions
- Limited to specific perturbation types

### Comparison Example

```python
# Compare representations
for use_wigner in [False, True]:
    lanczos = DL.Lanczos(ensemble, use_wigner=use_wigner)
    lanczos.init()
    lanczos.prepare_mode(10)
    lanczos.run_FT(30)
    
    gf = lanczos.get_green_function_continued_fraction(w_array)
    spectrum = -np.imag(gf)
     # Note: peak positions identical, convergence rates differ
 ```

 **Quick test with example data**:

 # doctest: +ELLIPSIS
 >>> from tdscha.testing.test_data import load_test_ensemble
 >>> import tdscha.DynamicalLanczos as DL
 >>> ens = load_test_ensemble(n_configs=5)
>>> # Test normal representation
>>> lanczos_normal = DL.Lanczos(ens, use_wigner=False)
Generating Real space force constant matrix...
Time to generate the real space force constant matrix: ... s
TODO: the last time could be speedup with the FFT algorithm.
>>> lanczos_normal.init()
Time to get the symmetries [...] from spglib: ... s
Time to convert symmetries in the polarizaion space: ... s
Time to create the block_id array: ... s
>>> # Test Wigner representation
>>> lanczos_wigner = DL.Lanczos(ens, use_wigner=True)
Generating Real space force constant matrix...
Time to generate the real space force constant matrix: ... s
TODO: the last time could be speedup with the FFT algorithm.
>>> lanczos_wigner.init()
Time to get the symmetries [...] from spglib: ... s
Time to convert symmetries in the polarizaion space: ... s
Time to create the block_id array: ... s
 >>> print(f"Normal representation: {len(lanczos_normal.w)} modes")
 Normal representation: ... modes
 >>> print(f"Wigner representation: {len(lanczos_wigner.w)} modes")
 Wigner representation: ... modes

 ## Convergence Parameters

### Lanczos Steps

```python
lanczos.run_FT(n_iter=100,           # Total steps
               save_each=10,         # Save checkpoint frequency
               save_dir="output")    # Directory for checkpoints
```

 **Guidelines**:
 - Start with 20-50 steps for testing
 - 100-200 steps for production
 - Check convergence with `tdscha-convergence-analysis`

 **Example with test data**:

 # doctest: +ELLIPSIS
>>> from tdscha.testing.test_data import load_test_ensemble
>>> import tdscha.DynamicalLanczos as DL
>>> ens = load_test_ensemble(n_configs=5)
>>> lanczos = DL.Lanczos(ens)
Generating Real space force constant matrix...
Time to generate the real space force constant matrix: ... s
TODO: the last time could be speedup with the FFT algorithm.
>>> lanczos.init()
Time to get the symmetries [...] from spglib: ... s
Time to convert symmetries in the polarizaion space: ... s
Time to create the block_id array: ... s
>>> lanczos.prepare_mode(5)

...
>>> # Run just 2 steps for quick test
>>> import sys
>>> from io import StringIO
>>> old_stdout = sys.stdout
>>> sys.stdout = StringIO()
>>> lanczos.run_FT(2, debug=False)  # doctest: +SKIP
>>> sys.stdout = old_stdout
>>> print(f"Ran 2 Lanczos steps, coefficients: a={lanczos.a_coeffs[:2]}")  # doctest: +SKIP
Ran 2 Lanczos steps, coefficients: a=...

 ### Restarting Calculations

```python
# Save checkpoint
lanczos.save_status("checkpoint.npz")

# Later, restart
lanczos2 = DL.Lanczos()
lanczos2.load_status("checkpoint.npz")
lanczos2.run_FT(50)  # Continue from saved state
```

 **Note**: For doctest purposes, we skip file I/O, but the pattern is shown above.

 ### Smearing and Terminator

```python
# Compute Green's function with smearing
gf = lanczos.get_green_function_continued_fraction(
    w_array,
    smearing=5/CC.Units.RY_TO_CM,  # 5 cm⁻¹ broadening
    use_terminator=True,           # Approximate infinite fraction
    last_average=3                 # Average last N coefficients for terminator
)
```

## Memory Management

### Large System Considerations

```python
# Exclude irrelevant modes (e.g., high-frequency)
select_modes = lanczos.w < 500/CC.Units.RY_TO_CM
lanczos = DL.Lanczos(ensemble, select_modes=select_modes)

# Estimate memory usage
n_modes = lanczos.n_modes
memory_gb = (n_modes**2 * 8 * 3) / 1024**3  # ~3 arrays needed
print(f"Estimated memory: {memory_gb:.1f} GB")
```

### Disk-Based Calculations

For extremely large systems, use checkpointing:

```python
lanczos.run_FT(100, save_dir="large_calc", save_each=5)
# Can restart if memory issues occur
```

## Advanced Features

### Custom Perturbations

```python
# Define custom vector in Cartesian coordinates
custom_vector = np.random.randn(3 * nat_sc)
lanczos.prepare_perturbation(custom_vector, masses_exp=1)
```

### Two-Phonon Response

```python
# Available only in Wigner representation
lanczos.use_wigner = True
lanczos.prepare_two_phonon_response(mode1, mode2)
```

### Interpolation to Denser q-Meshes

```python
# Interpolate to 4x4x4 q-mesh
q_mesh = [4, 4, 4]
interp_lanczos = lanczos.interpolate(q_mesh)
```

## Troubleshooting

### Common Issues

1. **Slow convergence**: Try Wigner representation or increase steps
2. **Memory errors**: Use `select_modes` or MPI parallelization
3. **Symmetry errors**: Check spglib installation or use `no_sym=True`
4. **Julia errors**: Verify Julia installation and package dependencies

### Performance Optimization

- Use `gamma_only=True` for Γ-point calculations
- Choose appropriate `mode` for your hardware
- Balance MPI processes vs OpenMP threads
- Monitor convergence to avoid unnecessary steps

## Next Steps

- Explore [StaticHessian](static-hessian.md) for free energy calculations
- Check [Examples](examples.md) for complete workflows
- Refer to [API Documentation](api/dynamical_lanczos.md) for method details