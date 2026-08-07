# Stable, symmetry-aware Raman and IR API

## Decision

Introduce a new high-level `tdscha.Spectroscopy` class, analogous in role to
`StaticHessian` and `QSpaceHessian`.  It owns optical perturbations, symmetry
reduction, independent Lanczos runs, checkpoints, loading, and spectrum
assembly.  `DynamicalLanczos.Lanczos` and its q-space derivatives remain
numerical engines; they will no longer be the recommended user-facing Raman
or IR API.

Only equilibrium one-phonon Raman and IR perturbations will be supported in
the stable API initially.  The configuration-dependent/two-phonon Raman code
is not trustworthy enough to expose.  It will be copied to a private legacy
module as a source backup, excluded from imports, and every old public entry
point that could invoke it will raise `NotImplementedError` with a direct
explanation.  Git history remains the authoritative history of the removed
implementation.

This supersedes the earlier plan to add more selectors directly to
`prepare_raman`.  Existing correct scripts must keep working through small
compatibility wrappers, but new scripts should use `Spectroscopy`.

## Physical conventions retained

For a symmetric, non-resonant Raman tensor, the unpolarized response is

\[
I = 45\,\bar\alpha^2 + 7\,\gamma^2,
\qquad
\bar\alpha = (R_{xx}+R_{yy}+R_{zz})/3,
\]

\[
\gamma^2 = \frac{1}{2}\left[(R_{xx}-R_{yy})^2
+(R_{xx}-R_{zz})^2+(R_{yy}-R_{zz})^2\right]
+3(R_{xy}^2+R_{xz}^2+R_{yz}^2).
\]

The canonical component convention remains the normalized convention already
used by the CsSnI3 production scripts:

| Component | Perturbation | Response weight |
|---:|---|---:|
| 0 | `(xx + yy + zz) / 3` | 45 |
| 1 | `(xx - yy) / sqrt(2)` | 7 |
| 2 | `(xx - zz) / sqrt(2)` | 7 |
| 3 | `(yy - zz) / sqrt(2)` | 7 |
| 4 | `sqrt(3) xy` | 7 |
| 5 | `sqrt(3) xz` | 7 |
| 6 | `sqrt(3) yz` | 7 |

The legacy raw convention is retained only by the old
`prepare_unpolarized_raman(index=...)` wrapper with weights
`[5, 7/2, 7/2, 7/2, 21, 21, 21]`.

The stable Raman result API distinguishes:

- `response`: the Bose-free retarded response, proportional to `-Im G`;
- `stokes`: response multiplied by `n_B(omega, T) + 1`;
- `anti_stokes`: response multiplied by `n_B(omega, T)`;
- optional experimental photon-frequency prefactors, only when the incident
  laser frequency and the desired unit convention are explicitly supplied.

The stable IR API returns the ionic susceptibility first.  A dielectric result
is constructed as `epsilon(omega) = epsilon_infinity + epsilon_ionic(omega)`.
The implementation must not hard-code the volume, `4 pi`, charge, or frequency
conversion until the existing internal units have been audited and checked
against a harmonic reference.  Absorption and optical conductivity are
derived from that complex dielectric function with an explicit unit system.

## Public API

The intended workflow is:

```python
import tdscha.Spectroscopy as SP

job = SP.Spectroscopy(
    ensemble,
    backend="qspace",          # "real", "qspace", or "atom_fourier"
    workdir="spectroscopy",
    ignore_v3=False,
    ignore_v4=False,
    lo_to_split=None,
)

job.add_raman_polarized(
    incoming=[1, 0, 0],
    outgoing=[0, 1, 0],
    name="raman_xy",
)
job.add_raman_unpolarized(name="raman_unpolarized")
job.add_ir_polarized(direction=[1, 0, 0], name="ir_x")
job.add_ir_unpolarized(name="ir_unpolarized")

job.run(n_steps=500, save_each=10, resume=True)

raman = job.raman_spectrum(
    "raman_unpolarized", frequencies,
    kind="stokes", smearing=5.0,
)
epsilon = job.dielectric_function(
    "ir_unpolarized", frequencies,
    smearing=5.0,
)
```

Additional stable perturbation constructors are:

```python
job.add_raman_tensor(tensor, name=...)
job.add_ir_polarized(direction, effective_charges=..., name=...)
job.add_cartesian_perturbation(vector, observable=..., name=...)
```

`add_raman_tensor` accepts a symmetric 3x3 coefficient tensor `C` and prepares
`sum_ab C[a,b] d alpha_ab / dR`; it is the unambiguous replacement for
`mixed=True` and the various `pol_in_2`/`pol_out_2` arguments.  A polarized
Raman request is converted to this representation internally.  Antisymmetric
or resonant Raman tensors are rejected in the first release because their
rotational invariants differ.

The electronic background defaults to the full 3x3
`ensemble.current_dyn.dielectric_tensor`, which is persisted for load-only
analysis.  Polarized IR projects it as `e.T @ epsilon @ e`; unpolarized IR
uses `trace(epsilon)/3`.  `epsilon_infinity=` remains an explicit tensor
override.  The public constructor exposes `ignore_v3`, `ignore_v4`, and the
common `lo_to_split` convention (`None`, `"random"`, or a direction).
Atom-Fourier pins commensurate modes to the parent LO--TO basis and retains
the tensorial long-range correction between coarse points.  If
`ignore_effective_charges=True`, only the harmonic interpolation drops the
dipolar tail and LO--TO limit; the original dynamical matrix keeps its
effective charges so IR perturbations remain available.

Each `add_*` method returns an immutable perturbation identifier.  Names must
be unique and filesystem-safe.  The driver expands aggregate requests such as
unpolarized Raman/IR into symmetry orbits of elementary perturbations, but the
manifest records both the requested observable and every actual Lanczos run.

## Internal data model

Create `Modules/Spectroscopy.py` with immutable specifications:

- `PerturbationKind`: `RAMAN`, `IR`, `CARTESIAN`;
- `RamanTensorPerturbation`: symmetric 3x3 optical coefficient tensor;
- `IRPolarizationPerturbation`: normalized electric-field direction;
- `CartesianPerturbation`: explicit unit-cell Cartesian vector;
- `PerturbationOrbit`: representative, equivalent members, group mappings,
  stabilizer, coset representatives, and reconstruction weights;
- `SpectroscopyResult`: loaded coefficients plus immutable manifest metadata;
- `Spectroscopy`: orchestration, execution, restart, and analysis.

The definitions of the seven normalized Raman components and their weights
live once in this module.  Lanczos compatibility wrappers import or delegate
to the same definitions; no second conditional implementation remains.

The driver creates one fresh backend object per inequivalent perturbation.
Lanczos objects are not reused between channels because their recursion and
restart state are mutable.  Backend creation is isolated behind a small
factory so tests can use a fake engine and real/q-space/interpolation can
share orchestration.

## Symmetry design

### Observable action

Use the crystallographic point group of the unit cell.  For Cartesian
rotation `R`:

- IR directions transform as `e -> R e`;
- Raman coefficient tensors transform as `C -> R C R^T`;
- explicit atomic Cartesian vectors transform with the atom permutation and
  Cartesian rotation used by the selected backend.

Two requested perturbations are symmetry-equivalent only when a stored group
operation maps their complete prepared vectors within a numerical tolerance.
Comparing only labels (`x`, `y`, `z`) is forbidden.  The map may include a
known scalar phase/sign; reconstruction records that character explicitly.
For diagonal scalar response functions, sign-related vectors have the same
response, but cross responses must retain the sign/phase.

This reduces cubic unpolarized IR from three runs to one.  For unpolarized
Raman it normally leaves one trace representative, one diagonal-deviatoric
representative, and one off-diagonal representative, rather than seven runs.
Lower-symmetry systems automatically retain more representatives.

### Stabilizer and cosets inside `apply_anharmonic_FT`

Let `G` be the symmetry group used to symmetrize the ensemble and `H` the
stabilizer (little group) of the prepared perturbation/Krylov representation.
The existing expensive average over all `(configuration, g in G)` replicas
can be decomposed into:

1. rotate/symmetrize configurations only with representatives of the relevant
   cosets of `H` in `G`;
2. apply the subgroup projector to each returned perturbed average:
   `f_pert` transforms as a vector and `d2v_pert` as a rank-two operator;
3. restore exactly the same normalization as the full `G` average.

For an invariant perturbation the projector is the ordinary subgroup average.
For a one-dimensional sign/phase representation it must be the
character-weighted projector

\[
P_H^{(\chi)} = |H|^{-1}\sum_{h\in H}\chi(h)^* D(h).
\]

Higher-dimensional irreducible subspaces cannot use a scalar character
shortcut.  The first implementation falls back to the full symmetry average
unless it can prove that the requested reduction is exact.  It must never
silently approximate.

Group multiplication order matters when choosing left versus right cosets.
The implementation will determine it from the actual representation matrices
and verify the decomposition numerically.  The optimization remains disabled
by default until full-average parity tests pass for every backend.

### Lanczos backend hook

Add a private reduction object to the Lanczos engines and a narrow API:

```python
lanczos.configure_perturbation_symmetry(reduction_or_none)
```

The reduction supplies coset symmetry indices and subgroup representation
operators.  It is configured after preparing `psi` and before the recursion.
It is cleared whenever `psi`, q point, mesh, or symmetrization data changes.

Refactor the expensive part of `apply_anharmonic_FT` into two backend hooks:

```python
f_pert, d2v_pert = self._compute_perturbed_averages(symmetry_indices=...)
f_pert, d2v_pert = self._project_perturbed_averages(
    f_pert, d2v_pert, reduction=...)
```

- `DynamicalLanczos` passes only coset representatives to the Julia/C kernel,
  then projects the returned mode-space vector and matrix under `H`.
- `QSpaceLanczos` retains Python copies of its sparse q-space representation
  matrices (currently only cached in Julia), passes the selected coset subset
  to the q-space kernel, and projects the force block and every momentum-pair
  `d2v` block consistently.
- `QSpaceAtomFourierLanczos` uses the coarse-ensemble symmetry
  representation for the external Gamma perturbation and its existing
  fine/coarse maps for the two-phonon blocks.  Identity-mesh and fine-mesh
  parity tests are mandatory.

The existing `gamma_only` optimization becomes the translation-only special
case of this mechanism.  It is not extended independently.  Until the generic
path is validated, the old flag stays functional and unchanged.

## Checkpoint and restart format

Each work directory contains:

```text
spectroscopy/
  manifest.json
  runs/
    <stable-perturbation-id>/
      status.npz
      lanczos.abc
      metadata.json
```

The manifest is schema-versioned and written atomically.  It stores:

- package/schema version and backend type;
- temperature, supercell, structure, masses, mode-selection flags, and a
  stable fingerprint of the dynamical matrix and optical tensors;
- user request, canonical perturbation specification, prepared vector hash,
  modulus, symmetry orbit, stabilizer, cosets, and reconstruction map;
- Lanczos options and completed iteration count;
- state: `pending`, `running`, `complete`, or `failed` plus failure message.

`run(resume=True)` validates the manifest against the current ensemble,
loads every compatible incomplete run, and continues it.  It refuses
incompatible files with a field-by-field error.  Completed symmetry-equivalent
runs are reconstructed in memory and are never recomputed.

`Spectroscopy.load(workdir, ensemble=None)` loads all run files and can produce
spectra without an ensemble when every quantity required for analysis is in
the manifest.  Restarting recursion still requires compatible backend state.
Legacy `.abc`/`.npz` files may be imported only with explicit observable,
component, convention, and backend metadata; the loader never guesses.

Writes use a temporary file in the same directory followed by `os.replace`.
Only the MPI master writes manifests/checkpoints and all ranks synchronize
around state transitions.

## Backward compatibility and cleanup

Keep these equilibrium calls numerically unchanged during the transition:

- `prepare_raman(pol_vec_in=..., pol_vec_out=...)`;
- `prepare_raman(unpolarized=i)` with normalized components;
- `prepare_unpolarized_raman(index=i)` with raw components;
- `get_prefactors_unpolarized_raman(i)`;
- `prepare_ir(effective_charges=..., pol_vec=...)`.

They become thin wrappers around shared vector builders and the existing
backend-specific Gamma-vector preparation hook.  Bare `prepare_raman()` stays
polarized XX for compatibility, but documentation points to `Spectroscopy`.

Remove duplicated equilibrium public implementations from `QSpaceLanczos`.
The base class builds unit-cell vectors and calls
`_prepare_gamma_cartesian_perturbation`; real space tiles the vector and
q-space applies the existing `sqrt(N_cell)` Gamma normalization.

Quarantine these broken/ambiguous advanced entry points:

- `prepare_unpolarized_raman_FT`;
- `prepare_anharmonic_raman_FT`;
- `prepare_anharmonic_raman_FT_2ph`;
- any alias that can add an explicit two-phonon Raman perturbation.

Their implementations are copied to private
`Modules/_TwoPhononSpectroscopyLegacy.py` and are not exported.  Public methods
raise `NotImplementedError("two-phonon Raman is disabled: the previous
implementation is unvalidated")`.  The private backup is deliberately not
callable.  The same audit is applied to position-dependent effective-charge
IR methods; a method is retained only if characterization tests establish its
physics and backend behavior.  Otherwise it is quarantined with the same
policy rather than exposed through `Spectroscopy`.

Pre-hotfix saved Raman data keeps the earlier migration rules: real-space
normalized channels 0--3 require reruns; old q-space coefficients contain the
right direction but require a verified modulus correction; raw-convention
files are unaffected.

## Implementation phases

### Phase 1: API and symmetry algebra foundation

1. Add characterization tests for the current equilibrium polarized Raman,
   normalized/raw unpolarized Raman, and polarized IR vectors in real and
   q-space.
2. Add `Spectroscopy.py` immutable perturbation definitions, input validation,
   normalized Raman component table, vector construction, and result weights.
3. Implement point-group actions, vector-based equivalence detection,
   stabilizers, group multiplication, cosets, and reconstruction maps as pure
   NumPy code with cubic, tetragonal, and no-symmetry tests.
4. Add a fake backend and test orchestration expansion: cubic IR needs one
   run; cubic unpolarized Raman needs three; triclinic cases do not overreduce.
5. Export `Spectroscopy` and document the provisional API.

### Phase 2: equilibrium backend consolidation

1. Add `_prepare_gamma_cartesian_perturbation` to the real/q-space engines.
2. Move all vector definitions to the shared module and make old methods thin
   wrappers.
3. Delete duplicated q-space public Raman/IR preparation code.
4. Verify exact `psi` and `perturbation_modulus` parity for real, q-space,
   and atom-Fourier interpolation.

### Phase 3: restartable multi-perturbation execution

1. Implement backend factory, one-engine-per-representative execution, and
   explicit run options.
2. Add atomic schema-versioned manifests and per-run checkpoints.
3. Implement strict restart validation and load-only analysis.
4. Test interruption after arbitrary iterations and after arbitrary
   perturbations, including MPI master-only writes.

### Phase 4: subgroup/coset acceleration

1. Introduce a backend-neutral reduction object and full-average reference
   tests.
2. Refactor real-space perturbed-average computation into compute/project
   hooks; select coset symmetry matrices in Julia; apply invariant and
   character-weighted subgroup projectors.
3. Implement the q-space version, retaining representation matrices and
   correctly transforming all momentum-pair blocks.
4. Validate each Lanczos application and final coefficients against the full
   group for IR and Raman representatives in multiple space groups.
5. Enable automatically only for proven one-dimensional stabilizer actions;
   record the achieved reduction and fallback reason in the manifest.

### Phase 5: spectrum assembly

1. Combine representative Green functions using orbit multiplicities and the
   canonical Raman invariant weights.
2. Add response, Stokes, and anti-Stokes Raman outputs and detailed-balance
   tests.
3. Implement ionic IR susceptibility, then dielectric function with the
   electronic tensor, after a unit audit and harmonic-reference validation.
4. Add absorption/optical-conductivity helpers with explicit units.
5. Reproduce a corrected CsSnI3 Raman calculation and a harmonic IR reference.

### Phase 6: quarantine and documentation migration

1. Copy the unvalidated two-phonon source to the private backup module.
2. Replace every public two-phonon entry point with `NotImplementedError` and
   test that derived classes fail identically.
3. Audit/quarantine unvalidated position-dependent IR code.
4. Update examples and CsSnI3 scripts to the new driver while retaining tests
   that their old equilibrium calls still work.

## Correctness gates

No symmetry speedup is accepted merely because spectra look similar.  Each
backend must satisfy all of the following against the unreduced calculation:

- prepared Cartesian vector, `psi`, and perturbation modulus;
- a single application of harmonic and anharmonic `L` separately;
- returned `f_pert` and every `d2v_pert` block before final assembly;
- Lanczos coefficients and Green function over a frequency grid;
- rotation invariance of the seven-channel Raman sum;
- detailed balance between Stokes and anti-Stokes;
- cubic orbit reduction and low-symmetry non-reduction;
- restart equivalence at bitwise-identical options, or documented numerical
  tolerance where MPI reductions change summation order.

If the scalar stabilizer/character assumptions do not hold, the calculation
falls back to the full symmetry average and records why.

## Code-quality gates

- Every physical formula and Cartesian perturbation definition has one source
  of truth.  Compatibility methods delegate to it and contain no copied
  branches or weight tables.
- `Spectroscopy` owns workflow state; Lanczos owns operator application.  The
  driver does not reproduce Lanczos recursion, continued fractions, MPI
  distribution, or backend normalization.
- A backend override is allowed only for a different data representation
  (real-space tiling, q-space projection, or fine/coarse interpolation).  All
  validation, observable definitions, orbit analysis, manifest handling, and
  spectrum assembly remain backend-neutral.
- Symmetry matrices and group metadata are built once and passed through small
  immutable objects.  Real/q-space adapters consume them rather than running
  independent spglib analyses with separate conventions.
- Private helpers are short and composable; no new public method is added to
  `DynamicalLanczos` unless it is a necessary numerical-engine capability.
- Tests use shared fixtures and parametrization across backends instead of
  copying complete real-space and q-space test bodies.
- Dead compatibility code is removed after delegation.  The quarantined
  two-phonon source is a non-imported archival snapshot, not a second live
  implementation.
- The branch must pass formatting/static checks used by the project and leave
  no newly introduced mutable NumPy defaults, assertion-based user input
  validation, or broad exception handlers.

## Original first branch milestone

The first implementation milestone on `feature/symmetry-spectroscopy` is
deliberately non-invasive:

1. add the immutable perturbation and Raman-component definitions;
2. add pure symmetry-orbit/stabilizer/coset analysis with tests;
3. add a skeletal `Spectroscopy` request registry and manifest schema;
4. leave existing Lanczos preparation and `apply_anharmonic_FT` behavior
   unchanged until the characterization suite is in place.

This gives a reviewable API and proves the group algebra before touching the
performance-critical real/q-space kernels.

### Implementation status on `feature/symmetry-spectroscopy`

Implemented:

- immutable optical perturbation definitions, one canonical Raman-component
  table, and shared Raman/IR Cartesian-vector builders;
- the public `Spectroscopy` driver for polarized, unpolarized, tensor, and
  explicit-vector Raman/IR requests;
- spglib point-group construction with atom permutations and filtering of
  operations that do not preserve an anisotropic finite supercell mesh;
- global request deduplication, sign-aware perturbation orbits, stabilizers,
  and right-coset planning (cubic IR axes reduce to one Lanczos run; the seven
  Raman invariants reduce to three symmetry orbits);
- explicit treatment of symmetry-forbidden Raman channels: a zero contracted
  optical vertex is recorded as a zero contribution and does not start an
  undefined zero-norm Lanczos recursion;
- a single real/q-space Gamma preparation hook used by all compatibility and
  high-level paths; the duplicate QSpace Raman/IR API was removed;
- real-space, q-space, distributed q-space, and atom-Fourier execution
  through one workflow layer without copying Lanczos recursions or continued
  fractions;
- stabilizer/coset reduction inside the shared anharmonic Julia kernels: one
  ensemble transformation is evaluated per right coset and the returned force
  and second-derivative blocks are character-projected over the stabilizer;
- exact mapping between unit-cell Cartesian rotations and each backend's
  native symmetry ordering, including CellConstructor's fractional
  real-space convention;
- atomic native status, backend-neutral results, portable `.abc` files,
  strict manifest fingerprints, incremental resume, early-recursion
  convergence, and load-only analysis;
- weighted response, Stokes and anti-Stokes Raman spectra, ionic IR
  susceptibility, and projected dielectric functions including the electronic
  tensor and explicit volume/unit conversion;
- inert private backup of the former configuration-dependent two-phonon Raman
  source, with every public entry point raising `NotImplementedError`;
- dedicated user/API documentation and tests for legacy delegation, symmetry
  algebra, restart/load, every backend, spectrum assembly, and disabled code.
- complete removal of the obsolete KPM backend from source, installation,
  command-line tools, documentation, and tests;
- a standalone `spectroscopy_report/` LaTeX report, independent of the
  interpolation report, with reproducible benchmark data and vector figures.

Validation completed:

- the reduced real-space and q-space anharmonic applications agree with their
  full 48-operation group averages to numerical precision while evaluating
  only three ensemble representatives for the cubic IR fixture;
- a cubic 3x3x3 q-space test compares the full-replica Julia path with
  `configure_qspace_perturbation_symmetry()` at all 27 q points (Gamma plus
  26 non-TRI points), both before and after a first application has generated
  a nonzero two-phonon sector;
- the isolated-kernel speedup grows from about 1.9x/1.6x at 10 configurations
  to 12.7x/11.4x at 640 configurations (real/q-space), demonstrating that the
  fixed stabilizer projector is amortized toward the ideal 48/3 ratio;
- a 256-step controlled cubic benchmark reduces unpolarized Raman from seven
  requested components (three nonzero legacy calculations) to one independent
  calculation and unpolarized IR from three calculations to one; the legacy
  and reconstructed spectra agree within about `3.2e-14` relative error;
- the final focused spectroscopy/Raman suite passes 47 tests;
- the complete non-heavy repository suite passes 138 tests with 8 expected
  skips, covering the unchanged Lanczos, q-space, Hessian,
  interpolation/atom-Fourier, and restart paths.

Intentionally outside this one-phonon API:

- configuration-dependent/two-phonon Raman and IR vertices remain unsupported
  until their observable definitions, units, and symmetry transformations are
  independently derived and validated;
- absorption-coefficient and optical-conductivity unit conventions are not
  guessed by this layer; users can obtain the complex dielectric response and
  apply the convention appropriate to their electromagnetic unit system.
