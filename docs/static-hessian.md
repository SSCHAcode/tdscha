# StaticHessian: Free Energy Hessian Computation

The `StaticHessian` class computes the free energy Hessian matrix (second derivative of free energy with respect to atomic displacements) for large systems using sparse linear algebra. It exploits Lanczos-based inversion to include fourth-order anharmonic contributions efficiently.

The result is, in principle, equivalent to what you obtain from the `sscha` package `get_free_energy_hessian`, but by exploiting the sparsity of the anharmonic interactions, it can be applied to much larger systems (hundreds of modes) that are out of reach for direct diagonalization employed in `get_free_energy_hessian`, especially when the fourth-order contributions are significant.

The basic idea is to compute the full free energy Hessian matrix, defined by [R Bianco et al, Physical Review B 96, 014111 (2017)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.014111), in equation (21).

$$
\frac{\partial^2 F}{\partial R \partial R} = \Phi + \stackrel{(3)}{\Phi} \cdot \Lambda \left[1 - \stackrel{(4)}{\Phi}\Lambda\right]^{-1} \cdot \stackrel{(3)}{\Phi} 
$$

where $\Phi$ is the SSCHA auxiliary force constant matrix, $\stackrel{(3)}{\Phi}$ and $\stackrel{(4)}{\Phi}$ are the third and fourth order force constant tensors, and $\Lambda$ is proportional to the static limit of the two phonons free propagator (of the SSCHA auxiliary phonons).
The inversion in the square brakets is extremely memory demanding, as both $\stackref{(4)}{\Phi}$ and $\Lambda$ are dense tensors in the $N_\text{modes}^2\times N_\text{modes}^2$ space.
For this reason, the direct approach scales as $N_\text{modes}^4$ in memory and $N_\text{modes}^6$ in time, which becomes easily prohibitive for systems with more than 50 atoms in the supercell.

The `tdscha` package offers an alternative approach, by reformulating the free energy Hessian as the inverse of the static limit of the dynamical Green's function, as proved by [L Monacelli, Physical Review B 112, 014109 (2025)](https://journals.aps.org/prb/abstract/10.1103/8611-5k5v).

$$
\frac{\partial^2 F}{\partial R_a \partial R_b} = \lim_{\omega \to 0} \chi_{R_aR_b}(\omega)^{-1}
$$

The purpose of this module is to compute the free energy Hessian by applying Lanczos-based Krylov subspace methods to compute the inverse of $\chi(\omega=0)$.

## Purpose and Use Cases

The free energy Hessian provides:

1. **Thermodynamic stability** - Eigenvalues determine stability at given temperature
2. **Anharmonic phonons** - Renormalized frequencies beyond harmonic approximation
4. **Phase transitions** - Soft modes and instability analysis

**When to use StaticHessian**:
- Systems too large for the direct matrix inversion in `get_free_energy_hessian`
- Including fourth-order contributions is essential

## Basic Usage

### Initialization and Running

```python
import tdscha.StaticHessian
import sscha.Ensemble

# Load the ensemble from a previous SSCHA calculation
start_dyn = CC.Phonons.Phonons("dyn_", nqirr=3)
ensemble = sscha.Ensemble.Ensemble(start_dyn, 300) # Temperature in K
ensemble.load_dir("ensemble", 1) # Load from directory, sscha iteration 1
final_dyn = CC.Phonons.Phonons("final_dyn_", nqirr=3) # load the converged dynamical matrix
ensemble.update_weights(final_dyn, 300) # Update the ensemble


# Initialize the Hessian calculation
hessian = tdscha.StaticHessian.StaticHessian(ensemble)

# Un the Hessianminimization
hessian.run_no_mode_mixing(n_steps=200,
            save_dir="hessian_steps",
            restart_from_file=False)
```

Compute the Hessian matrix assuming that the polarization vectors do not change from the equilibrium SSCHA result (where instead polarization vectors are relaxed). This is usually a very good approximation.

The flag `restart_from_file` allows to restart from a previous calculation, by loading the last saved vector in `save_dir`. If `False`, the calculation starts from the initial guess (which is the SSCHA dynamical matrix).

This method calls the `DynamicalLanczos` on every phonon mode, to retrive the static limit of the dynamical Green's function. You can pass any keyword argument accepted by the constructor of `DynamicalLanczos` via the `lanczos_input` dictionary argument of the `StaticHessian` constructor.


### Retrieving Results

```python
# Get Hessian as Phonons object (with q-points)
hessian_phonons = hessian.retrieve_hessian()

# Extract frequencies
w, pols = hessian_phonons.DiagonalizeSupercell()

# Save in quantum espresso format
hessian_phonons.save_qe("hessian_dynmat_")
```

## Parallel Execution

StaticHessian uses the embedded Lanczos object for parallelization, therefore you can just submit the calculation with MPI, and the Lanczos will automatically distribute the calculation across processors.

```bash
mpirun -np 4 python hessian_calculation.py
```


