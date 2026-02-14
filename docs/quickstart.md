# Quick Start Guide

This guide walks through a complete TD-SCHA calculation from ensemble preparation to spectral analysis using CLI tools.

## The IR spectrum of SnTe

In the following example, we run a dynamical linear response calculation to compute the IR spectrum, with the radiation field polarized alogn the x-axis,
in the thermoelectric material SnTe.

In the real case, you should first run a standard SSCHA calculation. In particular, you need the following files:

1. An ensemble of configurations, with computed energies and forces (loaded and saved via the `sscha.Ensemble` module)
2. The initial dynamical matrices (e.g., `dyn_start_1`, `dyn_start_2`, etc.) used to generate the ensemble
3. The final dynamical matrix of a converged SSCHA run with the previous ensemble (e.g., `dyn_final_1`, `dyn_final_2`, etc.)


The workflow consist in reading the ensemble, updating it to reflect the final dynamical matrix, and then running the dynamical linear-response calculation.

```python
# Import the libraries
import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL

# Load the ensemble of SnTe and the relative dynamical matrix (3 irreducible q-points)
dyn_start = CC.Phonons.Phonons("dyn_start_", nqirr=3)
ens = sscha.Ensemble.Ensemble(dyn_start, 300)  # Temperature 300 K
ens.load_bin("ensemble_dir", 1) # Load ensemble from binary files (adjust path and population_id as needed)
final_dyn = CC.Phonons.Phonons("dyn_final_", nqirr=3)

# Update the SSCHA ensemble weights to reflect the final dynamical matrix 
ens.update_weights(final_dyn, 300) # 300 K

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

## Spectral function of a Single Phonon Mode

In the following, we modify the previous example to compute
the spectral function of a single phonon mode (e.g., mode 10) instead of the IR spectrum.
The overall spectral function can be computed by summing the contributions of all modes.

In this example we select the mode 10. The phonon modes are numbered from lowest energy to highest energy of the SSCHA auxiliary phonons, including all the modes commensurate with the provided supercell.

```python
# Import the libraries
import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL

# Load the ensemble of SnTe and the relative dynamical matrix (3 irreducible q-points)
dyn_start = CC.Phonons.Phonons("dyn_start_", nqirr=3)
ens = sscha.Ensemble.Ensemble(dyn_start, 300)  # Temperature 300 K
ens.load_bin("ensemble_dir", 1) # Load ensemble from binary files (adjust path and population_id as needed)
final_dyn = CC.Phonons.Phonons("dyn_final_", nqirr=3)

# Update the SSCHA ensemble weights to reflect the final dynamical matrix 
ens.update_weights(final_dyn, 300) # 300 K

# Initialize the TD-SCHA calculation via the Lanczos algorithm
lanczos = DL.Lanczos(ens)
lanczos.init()

# ** HERE THE DIFFERENCE ** - prepare the perturbation on the 10th phonon mode
lanczos.prepare_mode(10)

# Run the linear-response calculation at finite temperature (this is specified in the ensemble) 
# For 50 Lanczos steps. Usually 100-200 steps are needed for convergence, but this is just a quick test.
lanczos.run_FT(50)

# Save the results
lanczos.save_status("spectral_function_mode_10.npz")
```

Also in this case, you can plot the spectrum using the CLI tool:

```bash
tdscha-plot-data spectral_function_mode_10.npz 0 1000 2
```

However, you can also use this tool to analyze the value of the static limit $\omega_{\mathrm{static}} = \sqrt{1/G(0)}$, where $G(\omega)$ is the Green-function of the mode. 
This is the frequency corresponding to the free energy Hessian, and it is the best way to compute the free energy Hessian accounting for the complete anhamronic renormalization.

To extract this value, you can run:
```bash
tdscha-convergence-analysis spectral_function_mode_10.npz 
```

The script will generate four plots, one of which shows the convergence of $\omega_{\mathrm{static}}$ with the number of Lanczos steps.


##  Raman Scattering

Raman scattering can also be computed by TD-SCHA. The only difference is that you need to prepare the Raman tensor before running the calculation.
The final dynamical matrix must have the Raman Tensor attached to it. Then, the Raman perturbation can be run with

```python
# Ensure dyn_final has Raman tensor
lanczos.prepare_raman(pol_vec_in=[1,0,0], pol_vec_out=[1,0,0])
lanczos.run_FT(50)
lanczos.save_status("raman_spectrum_xx.npz")
```

Since the Raman is a scattering process, we have to specify the polarization vector of the incoming
radiation (`pol_vec_in`) and the outgoing radiation (`pol_vec_out`). In this example, we compute the Raman spectrum for incoming and outgoing radiation polarized along the x-axis. You can change these vectors to compute different polarization configurations.

Unpolarized Raman can be computed efficiently running a special combination of perturbations, as described in the [In-Depth Usage](usage.md) section.

### Step 3: Analyze Results with CLI

TD-SCHA provides four CLI tools for analysis. If your calculation was interrupted, you can recover the data from the standard output.
We provide a tool to convert the standard output of a Lanczos calculation into .abc format, which can then be plotted or analyzed:

```bash
tdscha-output2abc lanczos.stdout lanczos.abc
```

The `lanczos.abc` cannot be used to restart the calculation, but you can replace it with a .npz in all the analysis script.

#### Quick CLI Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `tdscha-convergence-analysis` | Analyze Lanczos convergence | `tdscha-convergence-analysis file.npz 5` |
| `tdscha-plot-data` | Plot spectrum | `tdscha-plot-data file.abc 0 1000 2` |
| `tdscha-output2abc` | Convert stdout to .abc | `tdscha-output2abc stdout.txt output.abc` |
| `tdscha-hessian-convergence` | Plot Hessian convergence | `tdscha-hessian-convergence dir/ prefix` |


## Parallel execution 

All the previous examples can be run in parallel using MPI.
You just need `mpi4py` installed and properly configured. Then, you can run the same script with `mpirun`:

```bash
mpirun -np 4 python run_calculation.py
```


## Analyze Convergence

Check how the spectral function converges with Lanczos steps:

```bash
tdscha-convergence-analysis lanczos_final.npz 5
```

This generates three plots:
- Static frequency vs Lanczos steps
- Spectral function evolution (without terminator)
- Spectral function evolution (with terminator)
- Final converged spectrum

From these plots, you can assess the convergence of the calculation.

## Next Steps

- Explore [In-Depth Usage](usage.md) for advanced features
- Learn about [StaticHessian](static-hessian.md) for free energy calculations
- Check [Examples](examples.md) for complete workflows
- Refer to [API Documentation](api/dynamical_lanczos.md) for detailed method specifications
