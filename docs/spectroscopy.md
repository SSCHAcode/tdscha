# Raman and IR spectroscopy

`tdscha.Spectroscopy.Spectroscopy` is the stable API for one-phonon Raman
and IR calculations. It owns request planning, symmetry reduction, execution,
restart files, and spectrum assembly. The lower-level
`DynamicalLanczos.prepare_raman` and `prepare_ir` methods remain available so
existing scripts keep working, but new calculations should use this class.

## Where the ensemble comes from

The driver takes the **location** of the ensemble, not a loaded one:

```python
from tdscha.Spectroscopy import EnsembleSource

source = EnsembleSource(
    data_dir="ensemble_data",
    population=3,
    dyn="dyn_gen_pop3_", nqirr=8,   # the generating dynamical matrix
    T=250.0,
    final_dyn="dyn_end_", final_nqirr=8,   # the converged solution
    final_T=250.0,
    n_configs=None,                 # None reads the whole population
)
```

Under `mpirun` the `qspace` and `atom_fourier` backends then read the
configurations **once, on the master**, and scatter them: every rank keeps
only `N / n_procs` of the Bloch-transformed displacements and forces. A
160 000-configuration ensemble therefore never exists more than once, which
is what makes it runnable at all. Nothing else is distributed — the
dynamical matrices are small and every rank holds them, because the run
plan, the symmetry analysis, and the spectral assembly all need them.

`final_dyn` is the reference of the whole calculation: the ensemble is
reweighted onto it, and its Raman tensor, Born effective charges, and
dielectric tensor define the optical vertices. Production runs should always
set it.

A loaded `sscha.Ensemble.Ensemble` is still accepted and behaves as before —
replicated on every rank (a warning says so when a q-space backend gets one
under `mpirun`). That is the right thing for small systems and it is what
`backend="real"` needs, since the real-space Lanczos parallelizes by splitting
a replicated ensemble across ranks. Passing an `EnsembleSource` to
`backend="real"` therefore loads the ensemble on every rank, by design.

## Polarized and unpolarized requests

```python
import numpy as np
from tdscha.Spectroscopy import Spectroscopy

spectra = Spectroscopy(
    source,
    backend="qspace",              # real, qspace, or atom_fourier
    workdir="optical_spectroscopy",
    ignore_v3=False,
    ignore_v4=False,
    lo_to_split=None,               # None, "random", or a 3-vector
)

# Raman: incoming and outgoing optical polarization vectors.
spectra.add_raman_polarized([1, 0, 0], [0, 1, 0], name="raman_xy")
spectra.add_raman_unpolarized(name="raman_powder")

# IR: electric-field direction or Cartesian powder average.
spectra.add_ir_polarized([1, 0, 0], name="ir_x")
spectra.add_ir_unpolarized(name="ir_powder")

print(spectra.plan_calculations())
spectra.run(200, save_each=10)
```

`Spectroscopy.from_ensemble_path` is the shorthand that builds the
`EnsembleSource` for you:

```python
spectra = Spectroscopy.from_ensemble_path(
    "ensemble_data", 3, "dyn_gen_pop3_", 250.0, nqirr=8,
    final_dyn="dyn_end_", final_nqirr=8,
    backend="qspace", workdir="optical_spectroscopy",
)
```

The polarization vectors are normalized by the API. Equilibrium Raman
derivatives, Born effective charges, and the electronic dielectric tensor are
read from the reference dynamical matrix (`spectra.reference_dyn`, i.e. the
ensemble's `current_dyn`: `final_dyn` when the ensemble is reweighted).
`effective_charges=` can be passed to either IR request when an explicit
override is needed.

For `atom_fourier`, the interpolation mesh is a backend option:

```python
spectra = Spectroscopy(
    source, backend="atom_fourier", workdir="raman_interpolated",
    backend_options={"fine_mesh": (8, 6, 8)},
)
```

`ignore_v3` and `ignore_v4` are explicit backend-independent physics
switches. The former `backend_options={"ignore_v3": ..., "ignore_v4": ...}`
spelling remains accepted for existing scripts. For `backend="atom_fourier"`,
`lo_to_split` also fixes the nonanalytic Gamma direction while the tensorial
dipole correction is retained on the interpolated mesh. When
`ignore_effective_charges=True`, that harmonic interpolation instead omits the
entire dipolar correction, including its Gamma LO--TO limit. The original
dynamical matrix is not modified, so its effective charges remain available
for constructing IR perturbations.

For data that are already contracted, use `add_raman_vector` or
`add_ir_vector`. An explicit IR vector needs `electronic_projection=` when
constructing a dielectric function because its electric-field direction is
not recoverable from the vector alone.

## Symmetry reduction

`use_symmetries=True` is the default. The planner:

1. excludes crystal rotations that do not preserve the finite supercell mesh;
2. groups symmetry-equivalent requested perturbations, so cubic `x`, `y`, and
   `z` IR requests require one Lanczos run;
3. finds the sign-aware stabilizer of each independent perturbation;
4. evaluates anharmonic ensemble averages only on right-coset
   representatives and applies the stabilizer projector to each returned
   force and second-derivative contribution.

An exactly zero contracted optical vertex is a valid symmetry-forbidden
component. It is recorded in the request reconstruction map as an exact zero
and does not start an undefined zero-norm Lanczos recursion.

The last step is implemented in the shared Lanczos operator path and is used
by the real-space, q-space, and atom-Fourier backends. Set
`use_symmetries=False` for a direct unsymmetrized reference calculation.

For a manually prepared finite-q `QSpaceLanczos` calculation,
`configure_qspace_perturbation_symmetry()` detects the little group in the
actual complex Bloch-mode representation. It supports complex characters at
non-time-reversal-invariant q. The optical workflow uses the Gamma-specific
`configure_spectroscopy_symmetry()` automatically.

## Restart and load-only analysis

`run(n_steps, ...)` interprets `n_steps` as the total target, including an
existing checkpoint. A compatible calculation resumes automatically.

```python
# Continue an interrupted run from 100 to 200 coefficients.
spectra.run(200, save_each=10, resume=True)

# Analyze on a machine that does not hold the original ensemble.
loaded = Spectroscopy.load("optical_spectroscopy")
```

Each independent perturbation stores native restart state, a backend-neutral
result, and (for Lanczos backends) a portable `.abc` file. Manifest
fingerprints reject restarts made with different ensembles, requests, backend
options, or run options. The manifest records both a fingerprint of the
reference dynamical matrix and temperature, and the identity of the ensemble
on disk (`data_dir`, population, `n_configs`, whether it was reweighted), so a
restart pointed at a different population is refused rather than silently
mixed.

The engine is built once per `run()` and reused for every independent
perturbation: preparing a perturbation resets the whole Lanczos state, and
reading a production ensemble is minutes of I/O that must not be repeated per
run. It is built lazily, so re-running a finished calculation for analysis
reloads nothing.

## Raman spectra

Frequencies use TD-SCHA internal Rydberg units.

```python
omega = np.linspace(1e-5, 0.01, 1000)
analysis = dict(smearing=2e-5, use_terminator=False)

response = loaded.raman_spectrum(
    "raman_powder", omega, kind="response", **analysis)
stokes = loaded.raman_spectrum(
    "raman_powder", omega, kind="stokes", **analysis)
anti_stokes = loaded.raman_spectrum(
    "raman_powder", omega, kind="anti_stokes", **analysis)
```

The default unpolarized convention is the normalized seven-component Placzek
invariant with weights `[45, 7, 7, 7, 7, 7, 7]`. The legacy raw convention is
available as `convention="raw"`; both conventions assemble the same total.
Supplying `laser_frequency=` additionally applies the scattered-frequency
fourth-power factor.

## IR susceptibility and dielectric function

```python
chi_ionic = loaded.ir_susceptibility("ir_x", omega, **analysis)
epsilon_x = loaded.dielectric_function(
    "ir_x", omega,
    **analysis,
)
epsilon_powder = loaded.dielectric_function(
    "ir_powder", omega,
    **analysis,
)
```

The default projected dielectric result is
`epsilon_infinity + (4*pi/Omega) * chi_ionic`, with `Omega` the **supercell**
volume converted from Angstrom cubed to Bohr cubed.  `ir_susceptibility`
includes the factor of two that converts the raw Rydberg-convention Lanczos
Green function to Hartree atomic units.  The Lanczos perturbation carries
`sqrt(n_cell)` (`prepare_ir`), so `chi_ionic` already includes the `n_cell`
factor and `Omega` must be the supercell volume
`n_cell * V_unit_cell`; together the total prefactor is `8*pi/V_supercell`,
matching the CellConstructor non-analytic LO-TO term (the 8 is the Rydberg
`e^2 = 2`).  `ionic_prefactor=` can override the remaining `4*pi/Omega`
convention explicitly.
The full 3x3 `reference_dyn.dielectric_tensor` is stored in the
manifest and inferred during both live and load-only analysis. Polarized IR
uses `e.T @ epsilon_infinity @ e`; an unpolarized request uses
`trace(epsilon_infinity)/3`. Passing `epsilon_infinity=` remains an explicit
tensor override.

## Two-phonon optical vertices

The former configuration-dependent two-phonon Raman entry points are kept as
compatibility names but deliberately raise `NotImplementedError`. Their old
source is archived privately in `_TwoPhononRamanLegacy.py` and is neither
installed nor imported. It must not be used for production spectra until its
observable definition, units, and symmetry properties are independently
validated.
