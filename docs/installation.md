# Installation Guide

This guide provides installation instructions for TD-SCHA (Time-Dependent Self-Consistent Harmonic Approximation), following the official [SSCHA installation guide](https://sscha.eu/download/). TD-SCHA requires `cellconstructor` and `python-sscha` as prerequisites.

## 1. Easy Installation through Anaconda/Mamba (Recommended)

The easiest way to install TD-SCHA with all dependencies is using Anaconda or Mamba.

### Option A: Anaconda

```bash
# Create a new environment with all required libraries
conda create -n sscha -c conda-forge python gfortran libblas lapack openmpi julia openmpi-mpicc pip numpy scipy spglib pkgconfig
conda activate sscha

# Install additional Python dependencies
pip install meson meson-python ninja
pip install ase julia mpi4py

# Install the SSCHA ecosystem
pip install --no-build-isolation cellconstructor python-sscha tdscha
```

### Option B: Micromamba (Lightweight Alternative)

If Anaconda is too large, use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

```bash
# Create environment
micromamba create -n sscha -c conda-forge python gfortran libblas lapack openmpi julia openmpi-mpicc pip numpy scipy spglib pkgconfig
micromamba activate sscha

# Install dependencies
pip install meson meson-python ninja
pip install ase julia mpi4py
pip install --no-build-isolation  cellconstructor python-sscha tdscha
```

### Setting Up Julia for Maximum Performance

**Critical Step**: TD-SCHA achieves 2-10× speedup with Julia enabled. Configure it properly:

```bash
# Install Julia Python bindings
python -c 'import julia; julia.install()'
```

**Note**: In some micromamba installations, you may need to specify the conda executable location:

```bash
export CONDA_JL_CONDA_EXE=$HOME/.local/bin/micromamba
echo "export CONDA_JL_CONDA_EXE=$HOME/.local/bin/micromamba" >> $HOME/.bashrc
```

To configure Julia PyCall to work with conda, open a Julia shell and install required packages:

```julia
# In Julia REPL, type ']' to enter package manager
pkg> add SparseArrays LinearAlgebra InteractiveUtils PyCall
```

## 2. Installing Without Package Managers

If you cannot use Anaconda/Mamba, install system dependencies manually.

### System Requirements

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install libblas-dev liblapack-dev liblapacke-dev gfortran openmpi-bin
```

**CentOS/RHEL/Fedora**:
```bash
sudo dnf install blas-devel lapack-devel gcc-gfortran openmpi-devel
```

**macOS with Homebrew**:
```bash
brew install openblas gcc open-mpi
export LDFLAGS="-L/usr/local/opt/openblas/lib"
export CPPFLAGS="-I/usr/local/opt/openblas/include"
```

### Python Dependencies

```bash
pip install meson meson-python ninja
pip install ase spglib mpi4py julia
```

## 3. Installing TD-SCHA

### From PyPI (Simplest)

```bash
pip install tdscha
```

This installs the latest release with runtime dependencies.

### From Source (Development Version)

```bash
git clone https://github.com/SSCHAcode/tdscha.git
cd tdscha
pip install --no-build-isolation .
```

The `--no-build-isolation` flag is required for C extensions to find system libraries.

### Development Installation

For contributions or development:

```bash
git clone https://github.com/SSCHAcode/tdscha.git
cd tdscha
pip install -e .[dev]
```

## 4. MPI Parallelization for Production Runs

MPI parallelization is optional but recommended for large systems:

```bash
# Install mpi4py (if not already installed)
pip install mpi4py

# Run TD-SCHA with MPI
mpirun -np 4 python your_script.py
```

**Note**: For maximum performance, combine MPI with Julia speedup.

## 5. Build System Details

TD-SCHA uses **Meson + meson-python + Ninja** as its build system:

```bash
# Install build tools
pip install meson>=1.1.0 meson-python>=0.13.0 ninja>=1.10

# Optional: Build with Intel MKL
pip install --no-build-isolation . -Duse_mkl=true

# Optional: Specify BLAS/LAPACK implementation
pip install --no-build-isolation . -Dblas=openblas -Dlapack=openblas
```

### C Extensions

The package includes C extensions compiled from:
- `CModules/odd_corr_module.c` - Odd correlation functions
- `CModules/LanczosFunctions.c` - Core Lanczos implementation (~101KB)

These are compiled into the `sscha_HP_odd` module with OpenMP support. MPI support is enabled via the `-D_MPI` flag.

## 6. Verification and Testing

### Quick Verification

```python
import tdscha
import tdscha.DynamicalLanczos as DL

# Check Julia availability (critical for performance)
print("Julia enabled:", DL.is_julia_enabled())
print("Recommended: Julia provides 2-10× speedup")

# Test module imports
import tdscha.StaticHessian
import tdscha.Tools
import tdscha.Perturbations
print("All TD-SCHA modules imported successfully")
```
### Running Tests

```bash
# Run standard tests (excludes heavy tests)
pytest -v

# Exclude release-tagged heavy tests (as CI does)
pytest -v -m "not release"

# Test specific components
pytest tests/test_lanczos_fast/test_lanczos_fast_rt.py -v
pytest tests/test_julia/test_julia.py -v
```

## 8. Troubleshooting Common Issues

### C Extension Compilation Failures

**Error**: `ImportError: cannot import name 'sscha_HP_odd'`

**Solution**:
1. Ensure BLAS/LAPACK development libraries are installed:
   ```bash
   # Ubuntu/Debian
   sudo apt install libblas-dev liblapack-dev
   ```
2. Check compiler logs during installation
3. Use `--no-build-isolation` flag:
   ```bash
   pip install --no-build-isolation tdscha
   ```

### Julia Not Found or Not Working

**Error**: `Julia not found` or slow performance despite Julia installation

**Solution**:
1. Verify Julia is in PATH:
   ```bash
   which julia
   julia --version
   ```
2. Set `JULIA_BINDIR` if needed:
   ```bash
   export JULIA_BINDIR=/path/to/julia/bin
   ```
3. Reinstall Julia Python bindings:
   ```bash
   pip install --force-reinstall julia
   python -c "import julia; julia.install()"
   ```

### MPI Configuration Issues

**Error**: MPI jobs fail or hang

**Solution**:
1. Test mpi4py installation:
   ```bash
   python -c "from mpi4py import MPI; print('MPI version:', MPI.Get_version())"
   ```
2. Ensure MPI implementation matches mpi4py (MPICH vs OpenMPI)
3. For cluster deployments, load correct MPI module before installation

### CHOLMOD Version Incompatibility (Julia)

**Warning**: `CHOLMOD version incompatibility`

**Solution**:
```bash
# Remove conda-installed julia if present
conda remove julia

# Use system Julia or install via Juliaup
curl -fsSL https://install.julialang.org | sh
```

## 9. Next Steps

After successful installation:

1. **Run the [Quick Start](quickstart.md)** guide for your first calculation
2. **Check [In-Depth Usage](usage.md)** for advanced features

## 10. Getting Help

- **Documentation**: This guide and other pages in [tdscha docs](https://sscha.eu/tdscha)
- **GitHub Issues**: [SSCHAcode/tdscha/issues](https://github.com/SSCHAcode/tdscha/issues)
- **SSCHA Website**: [sscha.eu](https://sscha.eu) with tutorials and FAQs
- **Community**: Check the SSCHA website for community forums and contact information
