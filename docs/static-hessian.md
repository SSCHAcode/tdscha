# StaticHessian: Free Energy Hessian Computation

The `StaticHessian` class computes the free energy Hessian matrix (second derivative of free energy with respect to atomic displacements) for large systems using sparse linear algebra. It exploits Lanczos-based inversion to include fourth-order anharmonic contributions efficiently.

## Purpose and Use Cases

The free energy Hessian provides:

1. **Thermodynamic stability** - Eigenvalues determine stability at given temperature
2. **Anharmonic phonons** - Renormalized frequencies beyond harmonic approximation
3. **Elastic constants** - Second derivative w.r.t. strain
4. **Phase transitions** - Soft modes and instability analysis

**When to use StaticHessian**:
- Systems too large for full dynamical Lanczos
- Need only static ($\omega=0$) response
- Require full Hessian matrix (not just diagonal)
- Including fourth-order contributions is essential

## Theory

The free energy Hessian $H_{ij} = \frac{\partial^2 F}{\partial u_i \partial u_j}$ satisfies:

$$
H = \Phi^{(2)} - \Phi^{(4)} : G
$$

where:
- $\Phi^{(2)}$ is the harmonic force constant matrix
- $\Phi^{(4)}$ is the fourth-order force constant tensor
- $G$ is the static limit of the Green's function
- $:$ denotes tensor contraction

The equation is solved via the linear system:

$$
L \cdot x = b
$$

with:
- $L$: Linear operator containing $\Phi^{(2)}$ and $\Phi^{(4)}$
- $x$: Vector representation of $G$ and auxiliary tensors
- $b$: Right-hand side from harmonic problem

## Basic Usage

### Initialization

```python
import tdscha.StaticHessian
import sscha.Ensemble

# From ensemble
hessian = tdscha.StaticHessian.StaticHessian(ensemble)

# Or initialize later
hessian = tdscha.StaticHessian.StaticHessian()
hessian.init(ensemble, verbose=True)
```

**Example with test data**:

# doctest: +ELLIPSIS
>>> from tdscha.testing.test_data import load_test_ensemble
>>> import tdscha.StaticHessian
>>> ens = load_test_ensemble(n_configs=5)
>>> hessian = tdscha.StaticHessian.StaticHessian(ens); print(f"Initialized StaticHessian with {hessian.lanczos.n_modes} modes")
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

**Parameters**:
- `ensemble`: SSCHA ensemble object
- `verbose`: Print memory usage and progress
- `lanczos_input`: Dictionary passed to embedded `Lanczos` object

### Running the Calculation

```python
hessian.run(n_steps=200,
            save_dir="hessian_steps",
            threshold=1e-6,
            algorithm="cg-prec",
            extra_options={})
```

**Parameters**:
- `n_steps`: Maximum minimization steps
- `save_dir`: Directory for saving convergence steps
- `threshold`: Convergence threshold for residual
- `algorithm`: Minimization algorithm (see below)
- `extra_options`: Algorithm-specific options

### Retrieving Results

```python
# Get Hessian as Phonons object (with q-points)
hessian_phonons = hessian.retrieve_hessian()

# Get raw matrix in supercell (no q-points)
hessian_matrix = hessian.retrieve_hessian(noq=True)

# Extract frequencies
w, pols = hessian_phonons.DiagonalizeSupercell()
```

## Algorithms

### Preconditioned Conjugate Gradient (`"cg-prec"`)

**Default and recommended**. Uses $L^{1/2}$ as preconditioner:

```python
hessian.run(algorithm="cg-prec", threshold=1e-8)
```

**Pros**: Fast convergence, minimal memory
**Cons**: Requires preconditioner application

### Standard Conjugate Gradient (`"cg"`)

```python
hessian.run(algorithm="cg", threshold=1e-8)
```

**Pros**: Simpler, no preconditioner needed  
**Cons**: Slower convergence for ill-conditioned systems

### Full Orthogonalization Method (`"fom"`)

Requires specifying Krylov subspace dimension:

```python
hessian.run(algorithm="fom", 
            extra_options={"Krylov_dimension": 50})
```

**Pros**: Better convergence for difficult systems
**Cons**: Higher memory usage

## Convergence Monitoring

### Step-by-Step Saving

```python
hessian.run(n_steps=100, save_dir="convergence")
```

Files saved in `save_dir`:
- `hessian_calculation_stepXXXXX.dat` - Vector at step X
- `hessian_calculation.json` - Final status (JSON)
- `hessian_calculation` - Final status (NumPy)

### Plotting Convergence

Use the CLI tool:

```bash
tdscha-hessian-convergence convergence/ hessian_calculation
```

This plots eigenvalue convergence vs minimization steps.

### Manual Convergence Check

```python
import numpy as np

# Load step files
steps = []
for i in range(n_steps):
    hessian.vector = np.loadtxt(f"convergence/step_{i:05d}.dat")
    H = hessian.retrieve_hessian(noq=True)
    w2 = np.linalg.eigvalsh(H)
    steps.append(np.sqrt(np.abs(w2)) * np.sign(w2))
```

## Advanced Features

### No Mode-Mixing Approximation

Neglects mode-mixing (bubble diagrams) for faster computation:

```python
hessian_nomix = hessian.run_no_mode_mixing(
    nsteps=100,
    save_dir="no_mixing",
    restart_from_file=False
)
```

**When to use**: Quick estimates, large systems where mode-mixing is weak
**Limitation**: Neglects phonon-phonon scattering diagrams

### Custom Initial Guess

```python
# Start from harmonic approximation
hessian.preconitioned = False  # Use Φ^(2) as initial guess
hessian.init(ensemble)

# Or start from previous calculation
hessian.load_status("previous_calc.json")
hessian.run(n_steps=50)  # Continue from loaded state
```

### Memory Optimization

The algorithm stores vectors of size:
$$
N_\text{vec} = N_\text{modes} + \frac{N_\text{modes}(N_\text{modes}+1)}{2} + \frac{N_\text{modes}(N_\text{modes}^2+3N_\text{modes}+2)}{6}
$$

**Memory estimate**:
```python
n_modes = hessian.lanczos.n_modes
n_g = (n_modes * (n_modes + 1)) // 2
n_w = (n_modes * (n_modes**2 + 3*n_modes + 2)) // 6
memory_gb = (n_g + n_w) * 8 * 3 / 1024**3  # 3 working arrays
```

## Complete Example

```python
import cellconstructor as CC
import sscha.Ensemble
import tdscha.StaticHessian
import numpy as np

# 1. Load ensemble
dyn = CC.Phonons.Phonons("dyn_", 1)
ens = sscha.Ensemble.Ensemble(dyn, 300)
ens.load("ensemble/", 1, 5000)

# 2. Initialize StaticHessian
hessian = tdscha.StaticHessian.StaticHessian(
    ensemble=ens,
    verbose=True,
    lanczos_input={"use_wigner": False}  # Options for embedded Lanczos
)

# 3. Run minimization
print("Memory estimate:", hessian.vector.nbytes * 3 / 1024**3, "GB")
hessian.run(n_steps=200, save_dir="hessian_out", threshold=1e-7)

# 4. Analyze results
hessian_phonons = hessian.retrieve_hessian()
w, pols = hessian_phonons.DiagonalizeSupercell()

# Remove translations
masses = hessian_phonons.structure.get_masses_array()
trans = CC.Methods.get_translations(pols, masses)
w_nontrans = w[~trans]

print(f"Hessian frequencies: {w_nontrans * CC.Units.RY_TO_CM}")
print(f"Min frequency: {np.min(w_nontrans) * CC.Units.RY_TO_CM:.1f} cm⁻¹")

# 5. Check convergence
if hessian.verbose:
    print("Calculation converged!")
```

## Parallel Execution

StaticHessian uses the embedded Lanczos object for parallelization:

```python
# MPI parallelization
hessian = tdscha.StaticHessian.StaticHessian(
    ensemble=ens,
    lanczos_input={"mode": DL.MODE_FAST_MPI}
)

# Julia fast mode  
hessian = tdscha.StaticHessian.StaticHessian(
    ensemble=ens,
    lanczos_input={"mode": DL.MODE_FAST_JULIA}
)
```

Run with MPI:
```bash
mpirun -np 4 python hessian_calculation.py
```

## Comparison with Dynamical Lanczos

| Aspect | StaticHessian | DynamicalLanczos |
|--------|---------------|------------------|
| **Output** | Static ($\omega=0$) Hessian | Full $\chi(\omega)$ |
| **Memory** | Lower (sparse algebra) | Higher (Lanczos basis) |
| **Speed** | Faster for Hessian only | Slower (computes all $\omega$) |
| **Use case** | Stability analysis, elastic constants | Spectra, dynamical properties |

## Troubleshooting

### Slow Convergence
- Increase `n_steps` (100-500 typical)
- Try `algorithm="fom"` with larger Krylov dimension
- Check if system is near instability (very small eigenvalues)

### Memory Issues
- Use `select_modes` in `lanczos_input` to exclude high-frequency modes
- Consider `run_no_mode_mixing` for large systems
- Use MPI parallelization to distribute memory

### Numerical Instability
- Ensure ensemble is well-converged
- Check weights are positive (no numerical issues)
- Try different `algorithm` or preconditioning

## Related Methods

### Bianco Algorithm

The implementation follows the "Bianco algorithm" for static Hessian computation, which:
1. Reformulates Hessian as linear system
2. Uses Krylov subspace methods for inversion
3. Exploits sparsity of anharmonic interactions

### Connection to SSCHA

The Hessian is the second derivative of SSCHA free energy:
$$
H_{ij} = \frac{\partial^2 F_\text{SCHA}}{\partial R_i \partial R_j}
$$
where $F_\text{SCHA}$ is minimized by the SSCHA self-consistent procedure.

## References

1. **Bianco et al.** - Static Hessian algorithm for anharmonic systems
2. **Monacelli et al., PRB** - Free energy derivatives in SCHA
3. SSCHA documentation for equilibrium properties