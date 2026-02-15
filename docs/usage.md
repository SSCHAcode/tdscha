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
