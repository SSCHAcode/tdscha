# Command-Line Interface (CLI) Tools

TD-SCHA provides four CLI tools for analysis and visualization of calculation results.

## Installation and Availability

The CLI tools are installed as entry points when installing the package:

```bash
# Verify installation
which tdscha-convergence-analysis
which tdscha-plot-data
which tdscha-output2abc
which tdscha-hessian-convergence
```

If not found, ensure TD-SCHA was installed with `pip install .` (not `pip install -e .`).

## Tool Overview

| Tool | Purpose | Input Formats | Output |
|------|---------|---------------|--------|
| `tdscha-convergence-analysis` | Analyze Lanczos convergence | `.npz`, `.abc` | Convergence plots |
| `tdscha-plot-data` | Plot spectra | `.npz`, `.abc` | Spectrum plot |
| `tdscha-output2abc` | Convert stdout to .abc | Text stdout | `.abc` file |
| `tdscha-hessian-convergence` | Plot Hessian convergence | Step files + JSON | Convergence plot |

## tdscha-convergence-analysis

Analyzes how spectral functions converge with Lanczos steps.

### Usage

```bash
tdscha-convergence-analysis lanczos_file [smearing]
```

**Arguments**:
- `lanczos_file`: `.npz` or `.abc` file from Lanczos calculation
- `smearing`: Optional smearing in cm⁻¹ (default: 1 cm⁻¹)

### Examples

```bash
# Basic analysis with default smearing
tdscha-convergence-analysis lanczos_final.npz

# Custom smearing (5 cm⁻¹)
tdscha-convergence-analysis lanczos_final.npz 5

# From .abc file (after converting the output of the calculation)
tdscha-convergence-analysis lanczos.abc 2
```

### Output

Generates four matplotlib figures:

1. **Static frequency convergence** - $\omega_\text{static} = \sqrt{1/G(0)}$ vs Lanczos steps
2. **Spectral function evolution (no terminator)** - 2D plot of $-ImG(\omega)$ vs steps
3. **Spectral function evolution (with terminator)** - Same but with continued fraction terminator
4. **Final converged spectrum** - All steps overlaid, final in red

### Interpretation

- **Converged static frequency** indicates Lanczos steps sufficient
- **Stable spectral function** across steps shows convergence
- **Terminator effect** should be small for converged calculations

## tdscha-plot-data

Plots spectral functions from Lanczos results.

### Usage

```bash
tdscha-plot-data file [w_start w_end [smearing]]
```

**Arguments**:
- `file`: `.npz` or `.abc` file
- `w_start`, `w_end`: Frequency range in cm⁻¹ (default: 0-5000)
- `smearing`: Smearing in cm⁻¹ (default: 5)

### Examples

```bash
# Basic plot (0-5000 cm⁻¹, 5 cm⁻¹ smearing)
tdscha-plot-data lanczos.npz

# Custom range (0-1000 cm⁻¹)
tdscha-plot-data lanczos.abc 0 1000

# Custom range and smearing (0-500 cm⁻¹, 2 cm⁻¹)
tdscha-plot-data lanczos.npz 0 500 2
```

### Output

Single figure showing:
- Spectrum $-ImG(\omega)$ vs frequency (cm⁻¹)
- X-axis: Frequency [cm⁻¹]
- Y-axis: Spectrum [a.u.]
- Title includes number of Lanczos poles

## tdscha-output2abc

Converts stdout of the run to `.abc` format for plotting the results.

### Usage

```bash
tdscha-output2abc output_file abc_file
```

**Arguments**:
- `output_file`: Text file containing stdout of `tdscha-lanczos.x`
- `abc_file`: Output `.abc` file name

### File Format

The `.abc` file contains three columns:
```
a0 b0 c0
a1 b1 c1
a2 b2 c2
...
```

Where:
- `a_n`: Diagonal Lanczos coefficients
- `b_n`, `c_n`: Off-diagonal coefficients
- Each row corresponds to Lanczos step `n`

### Parsing Details

The tool searches for lines containing "LANCZOS ALGORITHM" then extracts coefficients matching patterns:
- `a_0 = value`
- `b_0 = value`  
- `c_0 = value`

## tdscha-hessian-convergence

Plots convergence of StaticHessian minimization.

### Usage

```bash
tdscha-hessian-convergence directory prefix [initial_status] [dynamical_matrix]
```

**Arguments**:
- `directory`: Directory containing step files (`.dat`)
- `prefix`: File prefix (e.g., `hessian_calculation` for `hessian_calculation_step00001.dat`)
- `initial_status`: Optional `.json` or `.npz` file to initialize system
- `dynamical_matrix`: Optional reference dynamical matrix for comparison

### Examples

```bash
# Basic usage
tdscha-hessian-convergence hessian_steps/ hessian_calculation

# With initialization file
tdscha-hessian-convergence hessian_steps/ hessian_calculation init.json

# With reference dynamical matrix
tdscha-hessian-convergence hessian_steps/ hessian_calculation init.json dyn_prefix_
```

### File Structure Requirements

The directory must contain step files named:
```
{prefix}_step00001.dat
{prefix}_step00002.dat
...
{prefix}_step00100.dat
```

And optionally a converged file:
```
{prefix}  # Final converged vector (no _step suffix)
```

### Output

Single figure showing:
- All Hessian eigenvalues (converted to frequencies) vs minimization steps
- Optional: Reference harmonic frequencies as horizontal dashed lines
- X-axis: Optimization steps
- Y-axis: Hessian frequencies [cm⁻¹]

### Interpretation

- **Converging lines** indicate minimization progress
- **Flat lines** show converged eigenvalues
- **Reference lines** help identify mode correspondence

### C++ Executable Workflow

We also provide a C++ executable for the Lanczos algorithm, which can be used in place of the Python implementation for large systems or high-performance needs. The workflow is similar, but requires an additional conversion step for the initialization.
Note that the C++ code is less efficient than the Julia version available with the Python library, however it is much easier to install on clusters without the need of properly configured Julia and Python environments.
You can use the python library locally to prepare all the input files used by the C++ code, and then run the C++ executable on the cluster, and finally use the CLI tools to analyze the results.

```bash
# 1. Prepare input files (from Python)
python prepare_input_for_cluster.py

# 2. Run C++ executable
mpirun -np 16 ./tdscha-lanczos.x > lanczos.stdout

# 3. Convert and analyze
tdscha-output2abc lanczos.stdout lanczos.abc
tdscha-convergence-analysis lanczos.abc 5
tdscha-plot-data lanczos.abc 0 500 2
```

The input preparation script should generate the necessary files for the C++ executable, including the initial Lanczos vector and any required parameters.
A premade script to prepare the input for the cluster is available in the repository under `Examples/Templates/prepare_input_for_cluster.py`
The workflow is very similar to a standard python one, but instead of running the Lanczos algorithm, we save the prepared Lanczos object to a file, in the following way


```python
import tdscha, tdscha.DynamicalLanczos as DL
import sscha, sscha.Ensemble
import cellconstructor as CC, cellconstructor.Phonons

# Load the ensemble from a previous SSCHA calculation
start_dyn = CC.Phonons.Phonons("dyn_", nqirr=3)
ensemble = sscha.Ensemble.Ensemble(start_dyn, 300) # Temperature in K
ensemble.load_dir("ensemble", 1) # Load from directory, sscha iteration 1
final_dyn = CC.Phonons.Phonons("final_dyn_", nqirr=3) # load the converged dynamical matrix
ensemble.update_weights(final_dyn, 300) # Update the ensemble


# Initialize the Lanczos object
lanczos = DL.DynamicalLanczos(ensemble)
lanczos.init()

# Setup the perturbation - in this case IR spectrum polarized along the x axis
lanczos.prepare_ir([1.0, 0.0, 0.0])

# Save the Lanczos object to a file for the C++ executable
lanczos.prepare_input_files(n_steps = 100,                # Number of Lanczos steps
                            directory = "lanczos_input",  # Directory in which to save the input files
                            root_name = "lanzos_ir_xpol") # Prefix

```

The script will create a directory called `lanczos_input` containing the necessary input files for the C++ executable, which can then be transferred to the cluster and used to run the Lanczos algorithm.
The C executable must be run inside the directory containing the input files, passing as only argument the `root_name` used in the preparation step, in the following way

```bash
cd lanczos_input
mpirun -np 16 /path/to/tdscha-lanczos.x lanzos_ir_xpol > lanczos.stdout
```

The parameters for the Lanczos run can also by edited manually, as they are written in a clear text json format called `root_name.json` (where `root_name` is the prefix used in the preparation step).


## Advanced Usage

### Custom Frequency Ranges

For high-resolution plots:

```bash
# High resolution: 0-200 cm⁻¹, 0.5 cm⁻¹ smearing
tdscha-plot-data lanczos.npz 0 200 0.5
```
## See Also

- [Quick Start Guide](quickstart.md) for complete workflows
- [Examples](examples.md) for practical usage
- [API Documentation](api/dynamical_lanczos.md) for Python interface
