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

If you are unsure, just load the lanczos object in the python REPL and check the `lanczos.w` array to see the frequencies of the modes. The modes are ordered from lowest to highest frequency, so `mode_index=0` corresponds to the softest mode (excluding translations).
You can convert them in cm-1 by multiplying by `CC.Units.RY_TO_CM`.

```python
# Find low-frequency modes
for idx, w in lanczos.w:
    freq_cm = w * CC.Units.RY_TO_CM
    print(f"Mode {idx}: {freq_cm:.1f} cm⁻¹")
```

Note that the first 3 acoustic modes are excluded from the mode index counting, so `mode_index=0` corresponds to the first non zero mode across all the Brilluin zone (only commensurate q points).


### 2. Infrared (IR) Absorption

Compute IR spectra using effective charges:

```python
lanczos.prepare_ir(pol_vec=np.array([1,0,0]))
```

Note that you need effective charges available to compute the IR response. These should be provided in the final dynamical matrix of the ensemble used to initialize the Lanczos object. If they are not available, you can provide them directly via the `effective_charges` parameter of the `prepare_ir` method.

**When to use**: IR absorption spectra, dielectric response.

**Parameters**:
- `pol_vec`: Light polarization vector (cartesian)
- `effective_charges`: Override default from dynamical matrix


### 3. Raman Scattering

Compute Raman spectra using Raman tensor. You can both compute the polarized Raman spectrum for specific incoming/outgoing polarizations, or the unpolarized Raman spectrum.

#### Polarized Raman
```python
lanczos.prepare_raman(pol_vec_in=[1,0,0], 
                      pol_vec_out=[1,0,0])
```

#### Unpolarized Raman Components
Unpolarized Raman can be obtained by averaging across all possible polarization direction. However, a very good approximation to the 
unpolarized Raman spectra can be obtained by computing only 7 specific components of the Raman tensor, which are weighted by specific prefactors to obtain the total unpolarized Raman spectrum.
The `tdscha` code offers a convenient method to initialize and run all of them.

```python
for i in range(7):
    lanczos.reset()
    lanczos.prepare_raman(unpolarized = i)
    lanczos.run_FT(50)
    lanczos.save_status(f"raman_unpolarized_{i}.npz")
```


Then you can plot the unpolarized Raman spectrum by summing the contributions of the 7 components. This is done in the following way:

```python
import tdscha, tdscha.DynamicalLanczos as DL
import cellconstructor as CC, cellconstructor.Units
import numpy as np
import matplotlib.pyplot as plt

# Prepare the frequency range in which you want to plot the spectrum (in cm-1)
w = np.linspace(0, 1000, 500)  # Frequency range in cm⁻¹
w_ry = w/CC.Units.RY_TO_CM # Convert in Ry (the internal unit of tdscha)
smearing = 2/CC.Units.RY_TO_CM  # Smearing in cm⁻¹

raman_signal = np.zeros_like(w)

# Load the 7 unpolarized Raman components and sum them.
for i in range(7):
    lanczos = DL.Lanczos()
    lanczos.load_status(f"raman_unpolarized_{i}.npz")

    # Get the dynamical green function from the linear response calculation
    gf = lanczos.get_green_function_continued_fraction(w, smearing=smearing)

    # The response is proportional to the imaginary part of the Green's function. 
    # The '-' sign selects the retarded response, which is the one relevant for Raman scattering.
    raman_signal += -np.imag(gf)


# Then, we can just plot the data
plt.plot(w, raman_signal)
plt.xlabel("Frequency (cm-1)")
plt.ylabel("Unpolarized Raman Intensity (arb. units)")
plt.show()
```

## Parallel Execution Modes

TD-SCHA supports four computation modes:

### Mode Selection

```python
# Auto-select fastest available
lanczos = DL.Lanczos(ensemble)  # Default: Julia if available, else C serial

# Manual selection
from tdscha.DynamicalLanczos import (
    MODE_SLOW_SERIAL,    # 0: Pure Python (testing) - extremely slow
    MODE_FAST_SERIAL,    # 1: C extension  
    MODE_FAST_MPI,       # 2: C with MPI
    MODE_FAST_JULIA      # 3: Julia (fastest) and parallel
)

lanczos = DL.Lanczos(ensemble, mode=MODE_FAST_JULIA)
```
### MPI Parallelization

```bash
# Run with MPI
mpirun -np 16 python script.py
```

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

For $\Gamma$-point-only perturbation, we have a special optimization that exploits the fact that the response is also at $\Gamma$. 
This allows us to separate point-group from translational symmetries. This speedup is proportional to the number of unit cells in the supercell.
The flag should be specified **before** calling `init()`, as it changes the way symmetries are initialized.

 ```python
lanczos.gamma_only = True
lanczos.init()
 ```

## Wigner vs Normal Representation

The original implementation of the TD-SCHA work a bi-conjugate Lanczos algorithm, as the Liouvillian was not Hermitian [Monacelli, Mauri, Physical Review B 103, 104305 (2021)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.104305).
Later, [Siciliano, Monacelli, Caldarelli, Mauri, Physical Review B 107, 174307 (2023)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.174307) provided a reformulation of the TD-SCHA in the Wigner representation, which allows to work with a Hermitian Liouvillian and thus to use the standard Lanczos algorithm.
The standard Lanczos algorithm requires just one application of the Liouvillian per step, while the bi-conjugate Lanczos requires two applications, thus the Wigner representation 
is twice as fast as the normal representation.

### Normal Representation (Default, probably changed in the future)

```python
lanczos.use_wigner = False  # Default
```

### Wigner Representation

```python
lanczos.use_wigner = True
```

## Save status during the calculation and restart

Sometimes the TDSCHA run can require a lot of time.
To avoid losing the data after a timeout, you can set the `save_each` parameter of the `run_FT` method, which will save a checkpoint every `save_each` steps in the specified directory.

```python
lanczos.run_FT(n_iter=100,           # Total steps
               save_each=10,         # Save checkpoint frequency
               save_dir="output",
               prefix="checkpoint"
               )    # Directory for checkpoints
```

The checkpoint files are saved in NumPy `.npz` format and contain all necessary data to restart the calculation from that point.
The files are named as `save_dir/prefix_step.npz`, where `step` is the current Lanczos step.

You can load a checkpoint and restart the calculation by running the following code:

```python
import tdscha.DynamicalLanczos as DL

lanczos = DL.Lanczos()
lanczos.load_status("output/checkpoint_50.npz")  # Load checkpoint from step 50
lanczos.run_FT(50, save_each=10, save_dir="output", prefix="checkpoint")  # Continue for 50 more steps
lanczos.save_status("final_result.npz")  # Save final result
```


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


## Q-Space Lanczos

The `QSpaceLanczos` module provides a reformulation of the Lanczos algorithm in q-space (Bloch basis), exploiting momentum conservation from Bloch's theorem. This can give a significant speedup over the standard real-space calculation, proportional to the number of unit cells in the supercell ($N_\text{cell}$).

### How It Works

In the standard real-space Lanczos, the two-phonon sector has dimension $\sim n_\text{modes}^2 / 2$, where $n_\text{modes} = 3 N_\text{atoms,sc}$ is the number of supercell modes. For a 4x4x4 supercell of a 2-atom unit cell, this means $\sim 460{,}000$ entries.

By working in q-space, momentum conservation ($\mathbf{q}_1 + \mathbf{q}_2 = \mathbf{q}_\text{pert} + \mathbf{G}$) makes the two-phonon sector **block-diagonal**: only pairs of q-points satisfying the conservation law contribute. The total two-phonon size drops from $\sim n_\text{modes}^2/2$ to $\sim N_\text{cell} \cdot n_\text{bands}^2/2$ (where $n_\text{bands} = 3 N_\text{atoms,uc}$). Moreover, translational symmetries are handled analytically via the Fourier transform, so only point-group symmetries need to be applied explicitly.

### Requirements

- **Julia** must be installed and configured (see [Installation](installation.md))
- **spglib** for symmetry detection
- The Q-space Lanczos always uses the **Wigner representation** and **Julia backend**

### Basic Usage

The workflow mirrors the standard Lanczos but uses `QSpaceLanczos` instead of `Lanczos`, and specifies perturbations by q-point index and band index rather than supercell mode index.

```python
import cellconstructor as CC
import cellconstructor.Phonons
import sscha, sscha.Ensemble
import tdscha.QSpaceLanczos as QL

# Load ensemble (same as standard workflow)
dyn = CC.Phonons.Phonons("dyn_final_", nqirr=3)
ens = sscha.Ensemble.Ensemble(dyn, 300)
ens.load_bin("ensemble_dir", 1)

# Create the Q-space Lanczos object
qlanc = QL.QSpaceLanczos(ens)
qlanc.init(use_symmetries=True)

# Perturb a specific phonon mode at q-point 0 (Gamma), band 3
qlanc.prepare_mode_q(iq=0, band_index=3)

# Run the Lanczos iteration
qlanc.run_FT(100)

# Extract the Green function (same interface as standard Lanczos)
import numpy as np
w = np.linspace(0, 500, 1000) / CC.Units.RY_TO_CM  # frequency grid in Ry
smearing = 5 / CC.Units.RY_TO_CM
gf = qlanc.get_green_function_continued_fraction(w, smearing=smearing)
spectral = -np.imag(gf)
```

### Choosing the Perturbation

#### Single mode at a q-point

To perturb a single phonon band at a specific q-point:

```python
qlanc.prepare_mode_q(iq, band_index)
```

- `iq`: index of the q-point in `qlanc.q_points` (0 = $\Gamma$)
- `band_index`: band index (0-based, excluding acoustic modes is your responsibility)

You can inspect the available q-points and frequencies:

```python
# Print q-points (Cartesian, 2pi/alat units)
for i, q in enumerate(qlanc.q_points):
    print(f"iq={i}: q = {q}")

# Print band frequencies at a given q-point (in cm-1)
iq = 0
for nu in range(qlanc.n_bands):
    freq = qlanc.w_q[nu, iq] * CC.Units.RY_TO_CM
    print(f"  band {nu}: {freq:.2f} cm-1")
```

#### Custom perturbation vector

To perturb with an arbitrary displacement pattern at a q-point (e.g., for IR-like perturbations):

```python
# vector is a real-space displacement pattern (3*n_atoms_uc,) in Cartesian coords
qlanc.prepare_perturbation_q(iq, vector)
```

This projects the vector onto the q-space eigenmodes at `iq`.

### Looping Over Multiple Modes

To compute the full spectral function, you need to run one Lanczos calculation per mode.
You can reset the state and prepare a new perturbation between runs:

```python
import cellconstructor as CC
import cellconstructor.Phonons
import sscha, sscha.Ensemble
import tdscha.QSpaceLanczos as QL

dyn = CC.Phonons.Phonons("dyn_final_", nqirr=3)
ens = sscha.Ensemble.Ensemble(dyn, 300)
ens.load_bin("ensemble_dir", 1)

qlanc = QL.QSpaceLanczos(ens)
qlanc.init(use_symmetries=True)

# Loop over modes at Gamma
iq = 0
for band in range(qlanc.n_bands):
    # Skip acoustic modes (zero frequency)
    if qlanc.w_q[band, iq] < 1e-6:
        continue

    qlanc.prepare_mode_q(iq, band)
    qlanc.run_FT(100)
    qlanc.save_status(f"qspace_iq{iq}_band{band}.npz")
```

### Saving and Loading

Checkpoint saving and loading works the same as the standard Lanczos:

```python
# Save during calculation
qlanc.run_FT(100, save_each=10, save_dir="output", prefix="qlanc")

# Save final result
qlanc.save_status("qspace_result.npz")
```

!!! note "Restart limitations"
    Restarting from a saved checkpoint currently requires re-creating the `QSpaceLanczos` object from the same ensemble, loading the status, and then continuing. The q-space internal state (pair maps, symmetries) is reconstructed from the ensemble on `init()`.

### When to Use Q-Space vs Real-Space Lanczos

| Feature | Real-space (`Lanczos`) | Q-space (`QSpaceLanczos`) |
|---------|----------------------|--------------------------|
| Backend | C, MPI, or Julia | Julia only |
| Representation | Real or Wigner | Wigner only |
| Perturbation | Supercell mode index | (q-point, band) index |
| Symmetry handling | Full space group | Point group (translations analytic) |
| Two-phonon size | $\sim n_\text{modes}^2/2$ | $\sim N_\text{cell} \cdot n_\text{bands}^2/2$ |
| Best for | Small supercells, $\Gamma$-only | Large supercells, finite-q perturbations |

The Q-space approach is most advantageous when:

- The supercell is large (large $N_\text{cell}$), since the speedup scales as $\sim N_\text{cell}$
- You want to study phonon spectral functions at specific q-points
- Julia is available and configured

For $\Gamma$-point-only calculations on small supercells, the standard Lanczos with `gamma_only=True` may be comparable or faster due to lower overhead.


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

### Performance Optimization

- Use `gamma_only=True` for Γ-point calculations
- Choose appropriate `mode` for your hardware
- Monitor convergence to avoid unnecessary steps

## Next Steps

- Explore [StaticHessian](static-hessian.md) for free energy calculations
- Check [Examples](examples.md) for complete workflows
- Refer to [API Documentation](api/dynamical_lanczos.md) for method details
