# Raman and IR spectroscopy

`tdscha.Spectroscopy.Spectroscopy` is the stable API for one-phonon Raman
and IR calculations. It owns request planning, symmetry reduction, execution,
restart files, and spectrum assembly. The lower-level
`DynamicalLanczos.prepare_raman` and `prepare_ir` methods remain available so
existing scripts keep working, but new calculations should use this class.

## Polarized and unpolarized requests

```python
import numpy as np
from tdscha.Spectroscopy import Spectroscopy

spectra = Spectroscopy(
    ensemble,
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

The polarization vectors are normalized by the API. Equilibrium Raman
derivatives, Born effective charges, and the electronic dielectric tensor are
read from `ensemble.current_dyn`. `effective_charges=` can be passed to either
IR request when an explicit override is needed.

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
options, or run options.

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
The full 3x3 `ensemble.current_dyn.dielectric_tensor` is stored in the
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
