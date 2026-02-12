# Command-Line Interface (CLI) Tools

TD-SCHA provides four CLI tools for analysis and visualization of calculation results. These tools work with output files from both Python and C++ executables.

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

# From .abc file (C++ executable output)
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

### Notes

- Uses continued fraction **without terminator** for plotting
- For terminator-included spectra, use convergence analysis tool
- Spectrum is normalized by perturbation modulus

## tdscha-output2abc

Converts stdout from C++ executable `tdscha-lanczos.x` to `.abc` format.

### Usage

```bash
tdscha-output2abc output_file abc_file
```

**Arguments**:
- `output_file`: Text file containing stdout of `tdscha-lanczos.x`
- `abc_file`: Output `.abc` file name

### Example

```bash
# Run C++ executable
./tdscha-lanczos.x > lanczos.stdout

# Convert to .abc
tdscha-output2abc lanczos.stdout lanczos.abc

# Now use .abc with other tools
tdscha-plot-data lanczos.abc
```

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

## Integration with Workflows

### Python Script Integration

```python
import subprocess
import sys

# After Lanczos calculation
lanczos.save_status("output.npz")

# Run CLI analysis
subprocess.run(["tdscha-convergence-analysis", "output.npz", "5"])
subprocess.run(["tdscha-plot-data", "output.npz", "0", "1000", "2"])
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple calculations
for file in results/*.npz; do
    basename=$(basename $file .npz)
    tdscha-convergence-analysis "$file" 5
    mv convergence.png "${basename}_convergence.png"
    tdscha-plot-data "$file" 0 1000 2
    mv spectrum.png "${basename}_spectrum.png"
done
```

### C++ Executable Workflow

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

## Advanced Usage

### Custom Frequency Ranges

For high-resolution plots:

```bash
# High resolution: 0-200 cm⁻¹, 0.5 cm⁻¹ smearing
tdscha-plot-data lanczos.npz 0 200 0.5
```

### Comparing Multiple Calculations

```bash
# Generate .abc files for comparison
tdscha-output2abc calc1.stdout calc1.abc
tdscha-output2abc calc2.stdout calc2.abc

# Plot together (requires custom script)
python plot_comparison.py calc1.abc calc2.abc
```

### Scripting with Python

```python
from tdscha import cli
import sys

# Mock command-line arguments
sys.argv = ["tdscha-plot-data", "lanczos.npz", "0", "500", "2"]
cli.plot()  # Calls the plot function directly
```

## Troubleshooting

### "File not found" Errors
- Ensure file paths are correct
- `.npz` files must contain Lanczos object data
- `.abc` files must have 3 columns of numbers

### Plotting Issues
- Install matplotlib: `pip install matplotlib`
- Set backend for headless systems: `export MPLBACKEND=Agg`
- For terminator plots, use convergence analysis tool

### Conversion Errors
- Ensure stdout contains Lanczos coefficient lines
- Check C++ executable compiled correctly
- Verify no extraneous output in stdout

### Hessian Convergence Issues
- Step files must follow naming convention
- Initial status file must be `.json` or `.npz`
- Reference dynamical matrix must have correct prefix

## Environment Variables

- `MPLBACKEND`: Matplotlib backend (e.g., `Agg` for headless)
- `OMP_NUM_THREADS`: OpenMP threads for C++ executable
- `MPIEXEC`: MPI executable path for C++ version

## See Also

- [Quick Start Guide](quickstart.md) for complete workflows
- [Examples](examples.md) for practical usage
- [API Documentation](api/dynamical_lanczos.md) for Python interface