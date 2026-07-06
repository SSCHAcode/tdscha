# Atomic D3 Centering: Permutation Symmetry + ASR Implementation Plan

Status: started after commit `a6abf0f0` on branch `qspace_interpolation`.

## D4 follow-up note (2026-07-05)

The D3 atomic permutation/ASR construction below is working, but the D4
extension is not solved by reusing the same force-star geometry.  SnTe/Fm-3m
benchmarks show `d4_center="leg"` has the wrong sign for the D4 spectral
moment shift even though its channel split and commensurate identity tests
pass.  A safety patch now maps `d4_center=True` to the legacy
`"reference"` mode; `"leg"` remains only an explicit diagnostic option.
The reference mode is not a stored-Phi4 oracle: it remains tensor-free and
uses the channel-0 atom/origin stochastic passes, applying the same
atom-resolved window to all four D4 legs relative to an external pinned atom.
On the clean Fm-3m SnTe smoke benchmark (`N=300`, 25 Lanczos steps, p4 scale
3) it captures only half of the direct D4 first-moment shift (-3.63 cm^-1 vs
-7.25 cm^-1, L1=0.391), so it is a safer comparison/guardrail rather than a
solved D4 centering.

Root-cause status: a SnTe L=2 geometry scan comparing the force-star D4 image
assignment to a true four-leg perimeter target gives TV mean 0.614, max 1.0,
nonzero in 85.3% of classes.  The next D4 implementation should therefore
target a genuine four-leg kernel, likely with a dedicated four-slot D4 kernel
or a low-rank factorization of the four-leg perimeter/ASR target.  Benchmarks
must be run with the branch overlay (`PYTHONPATH=/tmp/tdscha_pyshim` or
equivalent), because bare `micromamba run -n sscha` imports the installed
site-package copy.

## New D4 implementation plan (restart from the invariants)

Goal: interpolate the fourth-order two-phonon-to-two-phonon operation using
only the stochastic displacement/force fields of the small supercell, never
storing Phi4, while preserving:

1. the current four-leg permutation invariance of the D4 estimator;
2. exact translational ASR on every D4 leg;
3. exact identity at commensurate coarse-grid points;
4. linear scaling in the fine interpolation mesh size.

The failed lesson is that pinning only the force leg is not enough.  The D4
estimator has four logical vertex legs, even though the current Julia kernel
reuses only the two pair slots `w` and `v`.  A correct interpolation must
assign images to the whole four-leg cluster, not to a force-star product.

### Target kernel

Define four logical D4 slots:

```text
Ew, Ev  external/output two-phonon legs
Iw, Iv  internal/input two-phonon legs contracted with alpha1
```

For a primitive atom tuple and coarse-cell difference tuple, build a
geometry-only atom-resolved target

```text
T4(Ew, Ev, Iw, Iv; cell images)
```

by choosing the periodic images of the four atoms that minimize a symmetric
four-leg cluster cost.  The default cost should be the sum of squared pair
distances over the complete graph, with exact ties split equally.  This is
more stable than choosing a single pinned leg, is symmetric under all four
leg permutations, and handles SnTe-like basis ties because atoms, not only
cell indices, enter the distance test.

The raw target is then projected once onto the linear constraint subspace:

- **partition/commensurate identity:** for every folded atom/cell class, the
  sum over periodic images is exactly one, so at coarse-grid q the estimator
  collapses to the current plain D4 pass configuration by configuration;
- **ASR:** summing the target over the folded class of any one leg gives a
  leg-independent constant component only, so an acoustic translation on any
  leg contracts to zero;
- **permutation:** average/project over the 24 permutations of
  `(Ew, Ev, Iw, Iv)` after applying the corresponding momentum-role relabeling.

This projection is pure geometry and linear algebra.  Phi4 values are never
used.  A diagnostic should report the distance from the raw minimal-cluster
target, the ASR residual, the partition residual, and the distance to the old
plain/reference/leg targets.  The existing SnTe L=2 all-class scan becomes the
first regression: the new target must match the true four-leg perimeter target
up to projection residual, while the current force-star target remains the
negative control.

### Runtime representation

Do not evaluate the projected kernel as a dense four-index object at runtime.
Factor it into a small sum of separable four-slot windows:

```text
T4_asr ~= sum_r c_r
          A_r(Ew) B_r(Ev) C_r(Iw) D_r(Iv)
```

with each one-leg factor carrying atom-resolved folded class sums compatible
with the ASR/partition constraints.  The fit can start from SVD/CP
decomposition of matrix unfoldings and finish with constrained ALS.  The
constraints are enforced on the factors, not repaired per configuration, so
runtime does not need tensor ASR and does not bias the stochastic estimator.

The rank is a controlled implementation parameter.  For L=2 SnTe the target is
small enough to fit and inspect exactly; production should start with ranks
that reproduce the projected target to below the stochastic noise floor, then
increase rank only if the SnTe first-shell D4 shift does not converge.  Store
only factor tables, not D4 force constants.

Equivalence proof added to the LaTeX report: the factorization is of the
image-assignment kernel `K_R`, not of Phi4.  Multilinearity of the stochastic
four-field estimator gives the q-space D4 contraction with kernel `K_R`;
Gaussian integration by parts identifies that average with the Phi4 vertex in
the infinite-sampling, SSCHA-stationary limit.  The partition constraint on
`K_R` proves exact equality to the current plain D4 operator at commensurate
coarse q configuration by configuration, before any ensemble average.  Fitting
can be matrix-free: ALS/randomized CP needs only contractions of the
geometry target with three one-leg vectors, i.e. a force-like target action,
computed from the four-leg image oracle and ASR/partition projectors without
storing either Phi4 or a dense T4.

### Julia/Python execution path

Add a new explicit mode, e.g. `d4_center="factor4"` or
`window_design="atomic4_asr"`, and keep `reference`/`leg` only as diagnostics.
The runtime path is:

1. build one field set per distinct one-leg factor and per logical D4 slot
   role, using the existing NUDFT machinery with atom-resolved Bloch phases;
2. extend the Julia D4 kernel to accept four logical D4 slots
   `(Ew, Ev, Iw, Iv)` instead of reusing `w/v` for both external and internal
   legs;
3. for each low-rank factor `r`, call the D4-only kernel with the four factor
   field sets and coefficient `c_r`; sum the results.

The pair list is still the existing O(N_f) list of two-phonon blocks, and each
rank term performs one batched D4 pass over that list.  Therefore the runtime
scales as

```text
O(R4 * N_configs * N_syms * N_f * nb^2)
```

with memory `O(R4 * N_f * N_local * nb)` if fields are cached, or lower if
rank/slot fields are streamed in chunks.  `R4` is independent of the fine mesh;
hence the implementation remains linear in interpolation mesh size.

### Validation ladder

1. **Pure geometry tests:** permutation, partition identity, ASR rows, and
   SnTe L=2 four-leg perimeter agreement.  The old force-star target must fail
   the same SnTe diagnostic.
2. **Synthetic stochastic identity:** on a commensurate fine mesh, the new D4
   mode equals the plain D4 operator to numerical precision for arbitrary
   random fields and alpha vectors.
3. **Hermiticity/adjointness:** Lanczos `b` and `c` coefficients match within
   stochastic noise; individual rank terms may be non-symmetric, but the
   projected/factorized sum must be symmetric.
4. **ASR probe:** D4 acoustic-leg row norms vanish at Gamma and scale with the
   correct small-q power without relying on the old q-space Delta projector as
   the only protection.
5. **Toy exactness:** a small model where direct fine-supercell D4 is cheap
   must show convergence of the D4 shift with factor rank.
6. **SnTe decisive benchmark:** Fm-3m first off-grid shell, p4x2/p3x1.6,
   compare direct 4x4x4 vs interpolation.  Passing means the D4 first-moment
   shift and static `w_eff` move in the correct direction for both band 5 and
   band 0, with reference/leg kept as negative controls.

Implementation should stop at the geometry/factorization gate if the projected
target cannot be represented at low rank.  That failure would be useful: it
would prove that a fast tensor-free D4 interpolation needs either larger
support/rank or a different four-leg target, before any expensive SnTe spectra
are run.

## Report rewrite note (2026-07-05)

`report/interpolation/main.tex` has been rewritten to be more pedagogical:
notation and assumptions are defined up front; the off-grid Bloch transform
is explained as an image-assignment kernel; the ASR section now follows the
logic "what breaks, what constraints fix it, why one-period windows cannot
work, why doubled-support windows can"; the SnTe atomic section is presented
as construction plus validation; and the D4 section clearly separates the
validated chain result from the unresolved SnTe four-leg centering problem.

## Objective

Upgrade `window_design="atomic"` from the current successful SnTe heuristic
into a formally controlled tensor-free D3 centering mode that:

1. respects the same D3 permutation symmetrization already used by the Julia
   stochastic estimator;
2. satisfies the acoustic sum rule at the atom-resolved kernel level, so that
   acoustic-leg vertices vanish near Gamma without relying on tensor
   `Apply_ASR`;
3. preserves the SnTe 2x2x2 -> 4x4x4 result: dynamic Gamma TO peak near
   37.6-38 cm^-1, not the broken 52 cm^-1 plain-window result;
4. never constructs, stores, or fits to the actual Phi3 tensor.

## Current Atomic Mode

Implemented in `Modules/QSpaceInterpolation.py`:

- `window_design="atomic"` builds one D3 pass for each primitive atom `a`
  and each coarse origin `o`.
- The `z` slot is pinned to `(a, o)`.
- The pair slots `w` and `v` use identical atom-resolved minimal-image
  windows relative to the pinned atom `a`.
- Extended images are folded back onto the periodic supercell with the
  correct Bloch phase.
- Full origin sum restores the translation sum, so there is no `1/Nc`
  origin-averaging prefactor.

This works on SnTe:

- N=1000, 30 steps, D3-only Gamma TO: peak 37.7 cm^-1.
- Direct 4x4x4: 37.56 cm^-1.
- Tensor-D3 oracle: 37.64 cm^-1.
- Plain-window broken result: 52.28 cm^-1.

But it is not yet the final construction:

- It treats the `z` slot specially for all D3 terms.
- It does not explicitly match the Julia 1/3 + 2/3 permutation split in
  the geometry of the interpolation kernel.
- It does not impose atom-resolved ASR constraints.

## D3 Permutation Structure in Julia

The Julia slot kernel already exploits full D3 permutation symmetry.
For one slot triple `(z, w, v)`:

### R -> d2v map

The code computes:

- `weight_Rf = <R, y_z> / 3`, then contributes
  `-weight_Rf * x_w * x_v`.
  This is the force leg on the `z` slot. It represents one of three D3
  permutations.

- `weight_R = <R, x_z> / 3`, then contributes
  `-weight_R * (x_w * y_v + y_w * x_v)`.
  These are the force leg on `v` and the force leg on `w`. Together they
  represent the other two D3 permutations.

### alpha -> f map

The same split appears in adjoint form:

- `w1 * y_z` is the force leg on `z`, one third.
- `w2 * x_z`, where `buf_f_weight` contracts `y_w` and `y_v`, is the force
  leg on the pair slots, two thirds.

Therefore a permutation-correct interpolation must not use the same
`z`-pinned geometry for all three terms. Each force-leg channel needs the
geometry appropriate to the slot that carries the force.

## Permutation-Correct Atomic Kernel

For every D3 term, define three geometry channels:

1. `force_z`:
   - pin slot `z`;
   - choose slot `w` images relative to `z`;
   - choose slot `v` images relative to `z`.

2. `force_w`:
   - pin slot `w`;
   - choose slot `z` images relative to `w`;
   - choose slot `v` images relative to `w`.

3. `force_v`:
   - pin slot `v`;
   - choose slot `z` images relative to `v`;
   - choose slot `w` images relative to `v`.

Each channel has the same `1/3` statistical prefactor as the Julia
permutation split. The current code effectively implements only the
`force_z` geometry and feeds it into all three Julia contributions.

## Efficient Implementation Strategy for Permutations

Avoid rewriting the whole Julia kernel. Add channel-selection booleans to
the slot kernel:

- `compute_d3_z`: include only force-on-z D3 contributions:
  - `weight_Rf` in the R -> d2v map;
  - `w1 * y_z` in the alpha -> f map.

- `compute_d3_w`: include only force-on-w D3 contributions:
  - the `y_w * x_v` part of the R -> d2v cross term;
  - the `bu_w * conj(y_w)` part of `buf_f_weight` in alpha -> f.

- `compute_d3_v`: include only force-on-v D3 contributions:
  - the `x_w * y_v` part of the R -> d2v cross term;
  - the `bu_v * conj(y_v)` part of `buf_f_weight`.

The public wrapper can keep the existing `compute_d3` API for the old modes
by translating `compute_d3=True` into all three booleans true. A new wrapper
or optional arguments can expose the three booleans to Python for atomic
mode.

Python-side atomic pass construction:

- Build slot-pinned window maps for all three channels.
- For each pinned slot and each pinned atom/origin, build a slot triple
  `(z, w, v)` where the pinned slot is a delta and the other two slots are
  atom-resolved minimal-image windows relative to the pinned atom.
- Call the slot kernel with only the corresponding D3 channel enabled.

Cost:

- Current atomic SnTe cost: `nat * Nc = 16` D3 pass calls.
- Permutation-correct cost if done literally: `3 * nat * Nc = 48` D3 pass
  calls.
- This is acceptable for the underresolved small cells where atomic mode is
  needed. Later optimization can fuse the three channel calls into one Julia
  pass group.

## Acoustic Sum Rule Problem

ASR requires that the D3 vertex vanish when any leg is an acoustic
translation at q -> 0. For a centered tensor this is imposed by
`Apply_ASR`. For a stochastic interpolation kernel, the equivalent
condition is an image-sum constraint:

- summing over the atom/cell class of any leg must produce a slot-independent
  constant component only;
- that constant component then contracts with the physical force/displacement
  sum rule and vanishes.

The current atomic windows do not guarantee this. A pinned delta window has
nonuniform class sums, and per-q complex atom weights bypass the field-level
ASR projection in `_build_field_set`. Applying that field projection blindly
is not acceptable, because previous tests showed weighted zero-mode
projection modifies the designed kernel and can bias the result.

## ASR-Compatible Atomic Design

The correct approach is a kernel-level correction, not a field-level
projection.

### Step 1: Define the raw geometry target

For each permutation channel `c in {z,w,v}`, construct a geometry-only
kernel target from atomic minimal images:

```text
K_raw_c(atom_z, atom_w, atom_v; delta_zw, delta_zv)
```

or the equivalent slot coordinate representation for the pinned slot.

No Phi3 values enter this construction.

### Step 2: Define atom-resolved ASR constraints

For each channel target, impose three leg constraints:

- ASR on z leg: image/class sums over the z atom and z cell are independent
  of the z class, for fixed relative geometry of the other two legs.
- ASR on w leg: same for w.
- ASR on v leg: same for v.

Because the implementation works in slot windows, these constraints can be
applied to the effective kernel tensor, not directly to per-field data.

The constraints are linear in the kernel entries. This mirrors the existing
`asr_projected_target_1d` construction, but now the grid is:

```text
(slot atom, extended cell) x (slot atom, extended cell)
```

for two relative coordinates, with the pinned slot fixed per channel.

### Step 3: Project raw target to ASR subspace

Compute:

```text
K_asr = argmin ||K - K_raw||_W
        subject to atom-resolved ASR rows and partition rows
```

where `W` can optionally be a distance-decay metric to concentrate
fidelity at short atom-resolved distances. Start with unweighted projection.

Important: this projection still uses only geometry and linear constraints,
not Phi3.

### Step 4: Factor ASR-projected target into window passes

There are two possible levels:

#### Phase A: direct kernel-oracle implementation for design validation

Implement pure-Python/Numpy tools that build and project the atomic kernel,
then compare the projected target against:

- raw atomic target;
- perimeter oracle;
- tensor-D3 oracle blocks from `kernel_study.py`.

This does not run in production; it validates that ASR projection does not
destroy SnTe centering quality.

#### Phase B: runtime factorization

Represent `K_asr` as a low-rank sum of pinned slot-window products:

```text
K_asr ≈ sum_r z_r(slot_z) w_r(slot_w) v_r(slot_v)
```

with hard uniform class-sum constraints on each slot window. This is the
atom-resolved analogue of the existing doubled-support ASR design.

Use ALS/SVD:

- initialize from the raw atomic pinned windows;
- enforce uniform folded class sums in each slot parameterization;
- enforce partition with a KKT row or final normalization;
- stop when kernel error is below a practical threshold and SnTe oracle
  error remains close to raw atomic.

This gives a production ASR-exact atomic mode.

## Practical Incremental Plan

### Milestone 1: permutation-correct atomic mode

Implement first because it is clear and directly tied to the current Julia
estimator:

1. Extend Julia slot kernels with channel booleans.
2. Add a Python `_call_slots_d3_channels(...)`.
3. Build slot-pinned atomic windows for `force_z`, `force_w`, `force_v`.
4. Add `window_design="atomic_perm"` or an option
   `atomic_permutation=True`; keep current `atomic` for comparison until
   validated.
5. Tests:
   - existing atomic geometry tests still pass;
   - Hermiticity `|b-c|` on toy/SnTe smoke;
   - SnTe short benchmark still peaks near 38 cm^-1;
   - compare current atomic vs permutation-correct atomic on the kernel
     oracle.

### Milestone 2: ASR diagnostics for atomic modes

Before changing runtime ASR:

1. Add a diagnostic that measures an acoustic-leg D3 vertex as q -> 0 for:
   - plain;
   - asr;
   - current atomic;
   - permutation-correct atomic.
2. Use the existing toy `test_acoustic_vertex_decay` pattern, but with
   atom-resolved windows and a basis-offset test case.
3. Run a small SnTe near-Gamma diagnostic if practical.

Expected:

- current atomic may show a plateau/leak;
- permutation-correct atomic may improve but is not guaranteed ASR-clean.

### Milestone 3: ASR projection prototype

Implement pure-Numpy atom-resolved kernel projection outside the runtime path:

1. Build raw atomic/permutation-symmetric kernel.
2. Build linear ASR + partition rows.
3. Project the target.
4. Measure:
   - ASR residual;
   - distance to raw atomic;
   - distance to perimeter oracle;
   - SnTe D3 block error using Phi3 as oracle only.

Decision gate:

- if projection keeps SnTe block error close to raw atomic, proceed to
  runtime factorization;
- if projection destroys centering, introduce distance-weighted projection
  or enlarge support/rank.

### Milestone 4: ASR-exact runtime factorization

1. Add constrained atom-window parameterization with uniform folded class
   sums.
2. Fit low-rank slot-window passes to projected target.
3. Add `window_design="atomic_asr"` or make it the default after validation.
4. Tests:
   - commensurate identity;
   - acoustic-leg decay;
   - Hermiticity;
   - SnTe 2x2x2 -> 4x4x4 peak and L1;
   - toy regression.

## Immediate Next Action

Start with Milestone 1: make atomic mode permutation-channel correct with
minimal disturbance to existing modes. This is the smallest change that makes
the geometry consistent with the stochastic D3 permutation symmetry already
present in Julia.

## Progress Log

### 2026-07-04: Milestone 1 implementation started

Changes made:

- Generalized `get_atomic_window_passes(..., pinned_slot=0)` so the pinned
  delta window can live on `z`, `w`, or `v`.
- Changed `window_design="atomic"` setup to build three channel families:
  `force_z`, `force_w`, and `force_v`, each with all pinned atoms and all
  coarse origins.
- Changed atomic pass metadata from `(iz, iw, iv)` to
  `(channel, iz, iw, iv)`.
- Added optional D3 force-channel booleans to the Julia slot kernels:
  `d3_force_z`, `d3_force_w`, `d3_force_v`.
- Preserved old Julia API compatibility by keeping the existing positional
  `batched` argument before the new channel booleans.
- Updated Python `_call_slots` to pass `batched=true` followed by the channel
  mask.

Expected effect:

- Old `plain`, `minimal_image`, and `asr` paths still call all three D3
  channels and should remain unchanged.
- `atomic` now uses slot-pinned geometry consistent with the Julia D3
  permutation split.
- SnTe atomic pass count increases from 16 to 48 for the 2-atom, 8-cell
  coarse mesh.

Next checks:

- Python syntax check. DONE.
- Reinstall into the `sscha` micromamba environment. DONE.
- Run `test_atomic_windows.py`. DONE.
- Run scalar-vs-batched slot-kernel equivalence. DONE.
- Run a short SnTe atomic smoke benchmark to confirm construction and peak
  direction. DONE.

### 2026-07-04: Milestone 1 validation

Focused tests after the permutation-channel implementation:

- `python -m py_compile Modules/QSpaceInterpolation.py
  tests/test_interpolation/test_atomic_windows.py`: PASS.
- `python -m pip install .` in the `sscha` micromamba environment: PASS.
- `pytest -q tests/test_interpolation/test_atomic_windows.py
  tests/test_interpolation/test_windows.py::test_batched_equals_scalar_kernel`:
  PASS, 5 tests.

SnTe short control, D3-only, 2x2x2 -> 4x4x4, Gamma TO, N=100, 5 Lanczos
steps:

- `window_design="atomic"` now reports 48 passes, 32 field sets, 36 pair
  blocks. This is the expected factor of three over the earlier z-pinned
  atomic construction: 3 force channels x 2 primitive pinned atoms x 8
  coarse-cell origins.
- Runtime was 19 s for 5 steps on this run.
- Static `g=+3.6817e+07` (unstable, consistent direction for the direct/tensor
  answer).
- Dynamic peak with 1.5 cm-1 smearing was 42.2 cm-1. This short N=100/5 run is
  not the quality benchmark, but it confirms construction and normalization
  are not reverting to the broken 52 cm-1 plain result.
- Curve written in the SnTe work directory as
  `atomic_perm_asr_N100_s5.dat`.

SnTe medium validation, same settings but N=300 and 20 Lanczos steps:

- `window_design="atomic"`: 48 passes, 32 field sets, 36 pair blocks.
- Runtime was 213 s for 20 steps.
- Static `g=+7.8716e+06`, i.e. unstable like the direct/tensor reference.
- Dynamic peak with 1.5 cm-1 smearing was 38.5 cm-1.
- Curve written as `atomic_perm_asr_N300_s20.dat`.
- This is the relevant validation of the permutation-correct atom-resolved
  mode: it preserves the earlier tensor-free SnTe fix while implementing the
  D3 permutation split correctly.

Interpretation:

- The Julia D3 permutation split is now represented explicitly. The
  `force_z` contribution uses a z-pinned atomic window, `force_w` uses a
  w-pinned atomic window, and `force_v` uses a v-pinned atomic window.
- Old non-atomic modes remain algebraically unchanged because they call the
  slot kernel with all three D3 channel flags enabled.
- This is the current production candidate for tensor-free SnTe centering
  while the ASR-specific diagnostics are developed.

### 2026-07-04: ASR factorization prototype result

Implemented an experimental `get_atomic_asr_window_passes(...)` and
`window_design="atomic_asr"`:

- For each force channel, it replaces the pinned-atom delta by a uniform
  atom/cell window.
- The non-pinned pair kernel is the pinned-atom average of the raw atomic
  pair kernel and is factorized by SVD.
- This gives a very small pass count for SnTe: 6 SVD/channel passes and 6
  distinct field sets.
- The geometry unit test verifies that the SVD factors reconstruct the
  intended atom-averaged pair kernel.

Runtime dispatch bug found and fixed:

- The first smoke test had built `atomic_asr` passes but fell through the
  plain execution branch because `_call_julia_qspace` did not include
  `atomic_asr` in the windowed-mode dispatch tuple.
- After adding `atomic_asr` to that tuple, the mode is genuinely executed.

Corrected SnTe short smoke, D3-only, 2x2x2 -> 4x4x4, Gamma TO, N=100, 5
Lanczos steps:

- `window_design="atomic_asr"`: 6 passes, 6 field sets, runtime 5 s.
- Static `g=-4.7175e+06`, effective frequency 50.52 cm-1.
- Dynamic peak with 1.5 cm-1 smearing: 52.0 cm-1.
- Curve written as `atomic_asr_perm_asr_N100_s5.dat`.

Conclusion:

- This ASR prototype is not acceptable for production. It is formally
  attractive but too destructive: by averaging out the pinned atom it removes
  the atom-conditioned basis-offset information that fixed the L=2 SnTe
  image tie.
- Keep it only as a diagnostic/prototype until an ASR correction is built as
  a small constrained correction on top of raw atomic centering, not as a
  replacement of the raw atomic kernel.

### 2026-07-04: Revised ASR implementation direction

The permutation-correct raw atomic mode already has the important discrete
partition property at Gamma:

- for each folded atom/cell class, every non-pinned atomic leg map has unit
  total weight over all selected extended images;
- after summing pinned atoms and full coarse-cell origins, the pinned slot
  also has unit class sum;
- therefore any exactly ASR-satisfying folded coarse D3 tensor remains ASR
  satisfying at the strict q=0 acoustic point, because contraction with a
  uniform acoustic displacement only sees these class sums.

What is still not guaranteed is the small-q behavior of acoustic vertices:
different image choices have different first moments, so a near-Gamma
acoustic leg can pick up an O(q) coefficient that differs from the
perimeter-centered tensor. This matters for near-Gamma interpolation even if
the exact Gamma sum is zero.

The next ASR work should therefore be diagnostic and corrective:

1. Add a geometry test for the class-sum partition of permutation-correct
   atomic windows on all three slots. This is the tensor-free ASR-at-Gamma
   invariant. DONE in
   `tests/test_interpolation/test_atomic_windows.py`.
2. Add a channel-split regression proving that, for identical fields, the sum
   of `force_z + force_w + force_v` slot calls equals the old all-channel D3
   call. This protects the 1/3 + 2/3 Julia permutation symmetry. DONE in
   `tests/test_interpolation/test_windows.py`.
3. Build an acoustic-vertex diagnostic on the toy chain and then SnTe:
   compare plain, z-pinned atomic, permutation-correct atomic, and tensor-D3
   as q approaches Gamma.
4. If a near-Gamma acoustic leak is observed, fit a low-rank correction
   `Delta = K_ASR_projected - K_atomic_raw` and add it as extra window passes.
   The raw atomic passes must remain dominant; the failed `atomic_asr`
   experiment shows that replacing them by an averaged kernel destroys the
   SnTe result.

### 2026-07-04: Spectral/ForceTensor ASR comparison

Inspection of `cellconstructor.ForceTensor.Tensor3.Apply_ASR` and the
underlying `third_order_ASR.f90` clarifies why the first atomic ASR prototype
was the wrong analogue.

What `Tensor3.Center(Far=...) + Apply_ASR()` does:

- `Center(Far=...)` first expands the real-space support. The ASR routine
  expects every permutation of every triplet to already exist in the support;
  otherwise it stops with the instruction to repeat centering with a larger
  `Far`.
- `Apply_ASR` imposes the sum rule on the third index for fixed first two
  legs, then averages over the six permutations, and iterates to convergence.
- For each fixed `(first leg, second leg, Cartesian component of third leg)`,
  the ASR correction subtracts only the violated sum over all third-leg atoms
  and third-leg image blocks in the current support.
- With `power=0`, that correction is distributed uniformly over the available
  support. With `power>0`, it is distributed proportionally to
  `abs(Phi3)**power`.
- Therefore tensor ASR is a small additive correction on top of the centered
  tensor. It does not average away which atom/image was selected by
  centering.

Implication for the atom-resolved tensor-free scheme:

- The failed `atomic_asr` prototype was not analogous to `Apply_ASR`: it
  replaced the raw pinned-atom geometry by a uniform pinned slot and an
  atom-averaged pair kernel. That removes the basis-offset information that
  solves the SnTe `L=2` tie.
- A safe ASR mode should preserve the raw permutation-correct atomic passes
  and add a correction, not replace them.

Safe real-space/kernel design:

1. Build the raw permutation-correct atomic kernel on an enlarged support
   (`window_far` at least as large as the centering `Far`, with an option to
   increase it if permutation/ASR constraints need more images).
2. For each channel, form linear ASR residuals on the effective kernel:
   fixed two legs, sum over atom/image classes of the acoustic leg.
3. Solve only for an additive correction `Delta`, minimizing
   `||Delta||_W` subject to removing those residuals and preserving the
   folded partition/commensurate identity.
4. Use a Spectral-like metric:
   - default `power=0`: minimum uniform correction over the enlarged support;
   - optional geometry weights replacing unavailable `abs(Phi3)**power`, e.g.
     perimeter or leg-distance decay, to keep corrections local;
   - optional oracle-only diagnostic with actual `Phi3` to compare against
     `Apply_ASR`, but not in production.
5. Apply the same channel/permutation alternation as the stochastic D3 split:
   correct one leg, permute/channel-average, iterate until ASR residual and
   correction norm are small.
6. Factor only `Delta` into extra low-rank window passes and keep the raw
   48 atomic passes unchanged. This makes the correction visibly bounded in
   benchmarks and avoids the destructive replacement seen in `atomic_asr`.

Safe q-space alternative / complement:

- `Tensor3.Interpolate(asr=True)` applies an acoustic projector directly at
  the interpolated q point:
  `Phi3 -= Phi3 projected on leg * f_q`, with a sinc-like `f_q` determined
  by the centered support size.
- A tensor-free analogue can be applied after each windowed NUDFT, before
  contraction with phonon polarizations: project each Cartesian slot field
  away from the uniform acoustic subspace with the same `f_q` factor.
- This does not alter atomic image assignment and is identity-like away from
  Gamma. It is therefore a safer near-Gamma stabilizer than replacing the
  kernel, but it should be benchmarked because Spectral's bubble usually
  relies on real-space `Apply_ASR` and calls `Interpolate(..., asr=False)`.

Current recommendation:

- Do not promote the existing `atomic_asr`.
- First implement diagnostics for the acoustic vertex of raw
  permutation-correct atomic near Gamma.
- If a leak is measurable, prototype the additive `Delta` correction with
  enlarged support and a cap on `||Delta||/||K_raw||`; validate against the
  SnTe tensor oracle before adding runtime factorization.

### 2026-07-04: Production validation of atomic_delta

Ran a comprehensive production benchmark on SnTe 2x2x2 -> 4x4x4 comparing
`atomic_delta` against `atomic` (raw), `plain` (broken baseline), and
tensor-D3 (oracle reference). Quality: N=1000, 30 Lanczos steps, D3-only,
Gamma TO, smearing 1.5 cm-1.

Results summary:

```
mode                  static g      w_eff     peak(cm-1)  passes  fsets
----                  --------      -----     ----------  ------  -----
plain (broken)        -4.9067e+06   49.54     51.83       0       0
atomic (raw perm)     +5.1350e+06   nan       37.52       48      32
atomic_delta (ASR)    +5.0217e+06   nan       37.52       48      32
tensor-D3 (oracle)    +5.4156e+06   nan       37.65       0       0

Reference: direct 4x4x4 = 37.56 cm-1
```

Spectral quality:

- `atomic_delta` reproduces the direct 4x4x4 TO peak (37.52 vs 37.56 cm-1)
  and the tensor-D3 oracle (37.65 cm-1) within sampling noise.
- The raw `atomic` mode gives the same peak (37.52 cm-1), confirming that
  the q-space ASR projector does not degrade centering quality.
- The `plain` mode remains at the broken 51.83 cm-1 peak, confirming it
  cannot center at L=2.

ASR validation (acoustic-band D3 vertex norms at near-Gamma |q| = 0.066):

```
mode                  |D3_ac| at |q|=0    |D3_ac| at |q|=0.066
----                  ----------------    --------------------
atomic (raw)          1e-29               5e-13
atomic_delta (ASR)    1e-35               3.4e-17
```

The ASR projector reduces the acoustic-mode D3 coupling by a factor of
~14,000 at near-Gamma compared to raw atomic. At exact Gamma (q=0) both
modes have machine-zero acoustic vertices (the hard partition / class-sum
property of the permutation-correct atomic windows already enforces
Gamma-point ASR). The dramatic improvement at finite q is the novel
contribution of the q-space Delta projector.

Key conclusions:

1. **ASR is correctly implemented.** The `atomic_delta` mode satisfies the
   acoustic sum rule at both Gamma (machine zero) and near-Gamma (14,000x
   smaller leak than raw atomic). The projector subtracts only the uniform
   acoustic Cartesian component from the windowed fields, with a sinc-like
   support factor that suppresses the correction away from Gamma.

2. **Q-space interpolation works as expected.** The 2x2x2 -> 4x4x4
   interpolation with `atomic_delta` reproduces the direct 4x4x4 benchmark
   within sampling noise (37.52 vs 37.56 cm-1). The 48-pass
   permutation-correct atom-resolved centering resolves the L=2 basis ties
   that the plain/ASR/minimal-image window schemes cannot handle.

3. **The additive Delta design is safe.** Unlike the failed `atomic_asr`
   prototype (which replaced the pinned-atom kernel and reverted to the
   52 cm-1 plain peak), the `atomic_delta` projector preserves the raw
   atomic centering passes and adds only a conservative correction on
   the already-windowed fields. The product structure is K_raw + Delta,
   not a replacement.

4. **Production readiness.** `atomic_delta` is validated at production
   quality for SnTe and is the recommended tensor-free centering mode for
   underresolved small supercells (L=2) where tensor-D3 is unavailable.

Benchmark scripts and data: `bench_atomic_delta.py` and
`atomic_delta_perm_asr_N1000_s30.dat` (plus ASR diagnostic companion
`atomic_delta_perm_asr_N1000_s30_asr.dat`) in
`SnTe_FF/Spectral/TDSCHA_Interpolate/`.

### 2026-07-04: Delta-ASR mathematical plan

This section defines the implementation target before any code is written.
The goal is to imitate the safe part of `Tensor3.Apply_ASR`: preserve the
centered atomic kernel and add the smallest correction that enforces the
sum rule on the chosen support.

#### Notation

Let the three D3 estimator slots be

```text
s = 0, 1, 2  <=>  z, w, v
```

and let `c in {0,1,2}` denote the force channel. In the
permutation-correct atomic mode, channel `c` pins slot `c`; the other two
slots are centered by atom-resolved minimal images relative to the pinned
atom.

Define an extended atom/cell index

```text
i_s = (a_s, R_s),     a_s in primitive atoms, R_s in Z^3
```

and a folded class

```text
[i_s] = (a_s, R_s mod L).
```

For a fixed channel `c`, the raw atom-resolved kernel is represented by the
current window passes as a sparse CP sum

```text
K_raw^c(i_0,i_1,i_2)
  = sum_{p in raw(c)} z_p(i_0) w_p(i_1) v_p(i_2).
```

For the current production `atomic` mode on SnTe:

```text
# raw passes = 3 * nat * Nc = 48
```

where `nat=2` and `Nc=2*2*2=8`.

For wavevectors satisfying momentum conservation

```text
q_0 + q_1 + q_2 = G,
```

the geometry part of the interpolated D3 vertex is

```text
V^c(q_0,q_1,q_2)
  = sum_{i0,i1,i2} K^c(i0,i1,i2)
      exp[-2 pi i (q_0.R_0 + q_1.R_1 + q_2.R_2)].
```

All physical stochastic averages and phonon polarizations multiply this
geometry kernel later. The Delta plan only modifies `K`.

#### ASR constraints as linear image-sum equations

For an acoustic displacement on leg `ell`, the D3 tensor must vanish after
summing over all atoms and all images of that leg while the other two legs
are fixed. In kernel language, the interpolator must preserve the folded
sum seen by an ASR-satisfying coarse tensor.

Let `J_ell` be a fixed pair of indices for the two legs other than `ell`.
For example, for `ell=2`,

```text
J_2 = (i_0, i_1).
```

Let `Omega_ell(J_ell)` be the allowed correction support for the acoustic
leg, i.e. all atom/image entries `(a_ell,R_ell)` included by the chosen
`Far` and by the folded class policy. The ASR row is

```text
A_{ell,J_ell} K
  = sum_{i_ell in Omega_ell(J_ell)} K(i_0,i_1,i_2).
```

The desired value is the class-sum-compatible value

```text
d_{ell,J_ell}.
```

In the strict Gamma case this is the same folded sum as the commensurate
plain estimator. For the additive correction, define the residual

```text
r_{ell,J_ell} = d_{ell,J_ell} - A_{ell,J_ell} K_raw.
```

The correction must satisfy

```text
A Delta = r.
```

It must also preserve the commensurate identity/partition:

```text
B Delta = 0.
```

Here `B` contains folded-class partition rows. At commensurate q, all
extended images in the same folded class have phase one, so `B Delta=0`
guarantees that adding Delta does not change the exact coarse-grid
stochastic estimator.

The combined linear system is therefore

```text
G Delta = b,

G = [A]
    [B],

b = [r]
    [0].
```

This is the precise tensor-free analogue of `Apply_ASR`: only the ASR
residual is removed; the centered raw kernel is not replaced.

#### Minimum-change Delta

Choose a positive diagonal metric `H` on the correction variables. The
metric defines what "smallest correction" means:

```text
min_Delta  1/2 Delta^T H Delta
subject to G Delta = b.
```

The KKT equations are

```text
H Delta + G^T lambda = 0
G Delta              = b
```

and the Schur complement solution is

```text
Delta = H^{-1} G^T (G H^{-1} G^T)^{-1} b.
```

This is the form to implement for the diagnostic prototype because it gives
the correction norm and constraint residual directly.

If only one ASR leg is corrected at a time and `B` is temporarily omitted,
the rows decouple. For a fixed pair `J_ell`, write the allowed entries as
`e=(J_ell,i_ell)` and define the mobility

```text
mu_e = 1 / H_e.
```

Then the minimum-change correction is

```text
Delta_e = r_{ell,J_ell} * mu_e / sum_{e' in Omega_ell(J_ell)} mu_{e'}.
```

This reproduces the structure of `third_order_ASR.f90`:

- `power=0` corresponds to uniform mobility `mu_e = 1`;
- `power>0` in tensor space corresponds to
  `mu_e proportional to abs(Phi3_e)^power`.

In production we cannot use `abs(Phi3)`. The tensor-free replacement should
be a geometry mobility, for example

```text
mu_e = exp[-alpha * perimeter(e)]
```

or

```text
mu_e = rho ** spread(e),    0 < rho <= 1,
```

where `spread` is the maximum pair distance or triplet perimeter measured in
supercell-lattice units. This pushes the correction onto short, physically
plausible images, analogous to `Apply_ASR(power)` preserving large tensor
entries more than small long-range ones.

#### Permutation/channel iteration

`Tensor3.Apply_ASR` alternates:

```text
ASR on one index  ->  full permutation symmetrization.
```

The slot implementation should mirror this in channel space. Let `P` be
the linear operator that maps a kernel to the average over the six slot
permutations, re-gauged so the selected channel slot is the pinned slot.
Let `Q_ell` be the minimum-change ASR correction operator for leg `ell`.

One Delta iteration is

```text
K^{n+1/2} = K^n + Q_2(K^n)
K^{n+1}   = P K^{n+1/2}.
```

Because the runtime D3 estimator already splits the force channels, the
implementation can store this as three channel kernels:

```text
K^n = {K_z^n, K_w^n, K_v^n}.
```

The permutation operator is then a deterministic remapping between channel
supports, followed by averaging. The iteration stops when

```text
||A K^n - d|| / ||K_raw|| < eps_asr
```

and

```text
||K^n - K^{n-1}|| / ||K_raw|| < eps_perm.
```

The production correction is

```text
Delta = K^final - K_raw.
```

Important: this preserves the raw atomic mode in the final estimator:

```text
K_final = K_raw + Delta.
```

The failed `atomic_asr` prototype effectively used only a new projected
kernel and discarded `K_raw`; that is what must not be repeated.

#### Factorizing Delta into runtime passes

The raw kernel is already factorized as window passes. Delta must be added
as extra passes.

For a fixed channel, unfold Delta along one slot:

```text
Delta(i_c, i_a, i_b)
  ->  D[i_c, (i_a,i_b)].
```

If the pinned slot correction is uniform or low-dimensional, this unfolding
often has low numerical rank. Compute a truncated SVD:

```text
D ~= sum_{r=1}^{R_delta} sigma_r u_r(i_c) v_r(i_a,i_b).
```

Then factor each pair part `v_r(i_a,i_b)` again:

```text
v_r(i_a,i_b) ~= sum_t sigma_{rt} a_{rt}(i_a) b_{rt}(i_b).
```

Each term becomes one extra slot pass:

```text
z(i_0) w(i_1) v(i_2)
```

with a signed/complex real-space weight allowed. The total extra pass count
is

```text
P_delta = sum_channels sum_r rank_pair(r).
```

Acceptance gates:

```text
||Delta - Delta_factored|| / ||K_raw||       < eps_fact
||A(K_raw + Delta_factored) - d|| / ||K_raw|| < eps_asr
||Delta_factored|| / ||K_raw||               < eta_max
```

Start conservatively:

```text
eps_fact = 1e-8 for geometry tests
eps_asr  = 1e-10 for exact small tests, looser for large supports
eta_max  = 0.1 initially
```

If `eta_max` is exceeded, the correction is too invasive and the support or
metric must be changed before physics tests.

#### Design-time scaling

Definitions:

```text
A      = number of primitive atoms
Nc     = number of coarse supercell cells = prod(L)
F      = atomic image search radius (`window_far`)
M_F    = number of candidate replica shifts = (2F + 1)^3
S      = number of retained extended atom/image entries per slot
Nq     = number of fine q points = prod(fine_mesh)
Nb     = number of phonon bands = 3A
Ncfg   = number of stochastic configurations
Nsym   = number of q-space symmetries
Npair  = number of unique fine pairs for one perturbation q, about Nq/2
P_raw  = raw atomic pass count = 3 A Nc
P_delta = extra factored Delta pass count
Fsets  = number of distinct field sets after deduplication
```

Raw atomic window construction:

```text
O(3 * A^2 * Nc * M_F)
```

distance evaluations, independent of `Nq`, `Ncfg`, and `Nsym`. For SnTe
with `A=2`, `Nc=8`, `F=3`, this is tiny.

Dense diagnostic projection:

If the full channel kernel is materialized, variables scale as

```text
O(S^3) per channel
```

and a dense KKT solve is not acceptable except for very small supports.
This is only useful for toy diagnostics.

Sparse/row-local Delta projection:

The one-leg correction formula above avoids a dense KKT. It loops over ASR
rows and distributes residuals over the acoustic leg support:

```text
O(number_of_ASR_rows * average_support_per_row).
```

For a support represented by folded pair classes this is roughly

```text
O(3 channels * 3 legs * A^3 * Nc^2 * S_leg)
```

where `S_leg` is the number of extended images available for the corrected
leg in a fixed folded class. This is still independent of `Nq` and `Ncfg`.

Permutation remapping:

Each iteration needs to remap support entries under the six slot
permutations:

```text
O(number_of_nonzero_kernel_entries)
```

with sparse storage. The number of iterations should be small if the
initial atomic kernel is already close to ASR; `Apply_ASR` often converges
in one or a few iterations after centering.

Factorization:

For each channel, a dense SVD of an unfolding with dimensions

```text
S_pinned x S_pair
```

costs

```text
O(min(S_pinned^2 S_pair, S_pinned S_pair^2)).
```

This should be used only after sparsifying/restricting Delta. A more scalable
implementation should use randomized SVD or exploit the row-local correction
structure, where each one-leg ASR correction is already close to
`pair_residual(i,j) * leg_distribution(k)` and can be factorized by SVD of
the pair residual matrix.

Design-time memory:

```text
O(nnz(K_raw) + nnz(Delta))
```

for the sparse algorithm, plus temporary SVD work arrays for the selected
unfolding. This is independent of `Ncfg`.

#### Runtime scaling after factorization

Building each distinct field set stores two complex arrays:

```text
X, Y shape = (Nq, Ncfg, Nb).
```

Memory per field set is approximately

```text
M_field = 2 * 16 bytes * Nq * Ncfg * Nb.
```

For SnTe (`Nq=64`, `Nb=6`):

```text
M_field ~= 0.0117 MB * Ncfg.
```

So:

```text
Ncfg=300  -> 3.5 MB per field set
Ncfg=1000 -> 11.7 MB per field set
```

The current permutation-correct atomic SnTe mode has 32 distinct field sets,
so the field-set memory is roughly:

```text
Ncfg=300  -> 112 MB
Ncfg=1000 -> 375 MB
```

Delta increases this linearly with the number of new distinct windows. This
is why `P_delta` and `Fsets_delta` must be capped.

The Julia D3 slot-kernel cost per pass and Lanczos application is roughly

```text
O(Ncfg * Nsym * (Nq * Nb + Npair * Nb^2)).
```

Since `Npair ~ Nq/2`, the dominant scaling is

```text
O(Ncfg * Nsym * Nq * Nb^2)
```

per pass. Therefore the total D3 runtime scales as

```text
O((P_raw + P_delta) * Ncfg * Nsym * Nq * Nb^2).
```

D4 remains one plain pass:

```text
O(Ncfg * Nsym * Nq * Nb^2)
```

with a larger constant from four-field contractions, but it is not multiplied
by the atomic centering pass count.

Fine-mesh scaling:

- The Delta design itself is independent of `Nq`.
- Field construction scales as `O(Fsets * Nq * Ncfg * A * Nc * Nb)`.
- Each Lanczos step scales linearly with `Nq` through `Npair`.
- Near-Gamma accuracy improves only if the kernel has the correct ASR
  behavior; increasing `Nq` without ASR can amplify the acoustic leak.

Coarse-mesh scaling:

- `P_raw = 3 A Nc`, so raw atomic runtime grows linearly with the coarse
  supercell size.
- The design support and Delta projection grow faster if `Far` is increased,
  because candidate images scale as `(2F+1)^3`.
- Atomic Delta should only be enabled for underresolved small cells such as
  `L=2`, where the pass count is still manageable and ordinary cell-index
  windows cannot resolve basis ties.

Configuration scaling:

- Design and factorization are independent of `Ncfg`.
- Runtime and field-set memory are linear in `Ncfg`.
- Physics noise falls statistically with `Ncfg`, so any Delta benchmark must
  compare at fixed `Ncfg` to raw atomic and tensor-D3 oracle curves.

#### Implementation checkpoints before runtime use

1. Geometry-only diagnostic:
   - build sparse `K_raw`;
   - compute ASR residuals;
   - compute row-local `Delta`;
   - verify `B Delta = 0`;
   - verify `||Delta||/||K_raw||` is small.
2. Permutation iteration:
   - apply channel permutation averaging;
   - verify channel-split symmetry and ASR residual convergence.
3. Oracle-only SnTe kernel test:
   - contract periodic Phi3 only as a diagnostic;
   - compare raw atomic, raw+Delta, and Tensor3.Center+Apply_ASR.
4. Factorization test:
   - reconstruct Delta from passes;
   - check factorization error and ASR residual.
5. Runtime smoke:
   - short SnTe D3-only run near Gamma;
   - require no return to the 52 cm-1 plain failure.
6. Runtime benchmark:
   - N=300/20 and then N=1000/30 if the short run passes;
   - compare peak, static sign, L1 vs direct and tensor-D3.

#### Quick SnTe probe: replacement vs small correction

Before implementing production Delta, a quick script was run in `/tmp`
(`atomic_delta_probe_quiet.py`) to quantify the known failure mode.

Setup:

- SnTe force-field benchmark directory.
- Coarse cell `L=(2,2,2)`, `Far=3`.
- Fine mesh `4x4x4`, interpolated Gamma-pair blocks `q,-q`.
- `Phi3` was used only as an oracle to measure what
  `Tensor3.Center(Far=3)+Apply_ASR()` changes; it is not part of the
  proposed production algorithm.

Results:

```text
atomic_asr geometry replacement vs raw atomic:
  mean relative change = 0.534888
  max  relative change = 0.707107

Tensor3 Apply_ASR relative real-space L2 change:
  1.668434e-03

Tensor3 Apply_ASR relative d3(q,-q) block change:
  mean = 4.750860e-03
  max  = 6.761252e-03
```

Interpretation:

- The failed `atomic_asr` prototype is not a small ASR correction. It changes
  the atom-resolved geometry kernel by roughly 50%, which explains why the
  SnTe spectrum falls back to the plain-like 52 cm-1 result.
- The actual tensor ASR used by `ForceTensor` is small after centering: about
  0.17% in real-space L2 and below 0.7% on interpolated SnTe D3 blocks.
- Therefore the Delta plan is viable only if it remains in this small
  correction regime.

Hard acceptance gates for the first Delta prototype:

```text
mean_geometry_change(raw+Delta, raw) << 0.53
max_geometry_change(raw+Delta, raw)  << 0.70
oracle block change should be O(1e-3 to 1e-2), not O(1e-1)
SnTe short spectrum must not return to the 52 cm-1 peak
```

Initial numerical thresholds:

```text
||Delta||_2 / ||K_raw||_2 <= 1e-2    target
||Delta||_2 / ||K_raw||_2 <= 5e-2    absolute stop/go ceiling
```

If the ASR constraints require a larger correction, the support/metric is
wrong and the runtime implementation should not proceed.

### 2026-07-04: First safe runtime implementation (`atomic_delta`)

Implemented the conservative q-space Delta branch before the real-space
low-rank Delta factorization. This is intentionally not the old
`atomic_asr` replacement.

Definition:

```text
window_design="atomic_delta"
```

uses the same raw permutation-correct atomic window passes as
`window_design="atomic"`:

```text
P_raw = 3 * nat * Nc
```

For SnTe this is still 48 passes and 32 distinct field sets. The difference
is in `_build_field_set`: after the windowed NUDFT and before contraction
with phonon polarizations, the code subtracts only the uniform acoustic
Cartesian component from each slot field:

```text
U_delta(q) = U_raw(q) - f_N(q) P_ac U_raw(q)
F_delta(q) = F_raw(q) - f_N(q) P_ac F_raw(q).
```

Here

```text
P_ac = sum_{alpha=x,y,z} |t_alpha><t_alpha| / <t_alpha|t_alpha>,
t_alpha(a,beta) = sqrt(M_a) delta_{alpha,beta},
```

and the Tensor3-style support factor is

```text
f_N(q) = product_i sin(N_i pi a_i.q) / [N_i sin(pi a_i.q)],
N_i = 2 * Far * L_i + 1.
```

At Gamma, `f_N(q)=1` and the uniform acoustic component is removed. Away
from Gamma, the sinc factor rapidly suppresses the correction. The atomic
image assignment and pinned-atom windows are unchanged.

This is an implicit additive Delta:

```text
product(projected fields)
  = product(raw fields) + terms containing (-f_N P_ac raw field).
```

Therefore it has the key safety property missing from `atomic_asr`: it
does not average or replace the atom-resolved kernel. It only subtracts the
measured acoustic component of the already-windowed fields.

Short SnTe smoke, D3-only, 2x2x2 -> 4x4x4, Gamma TO, N=100, 5 Lanczos
steps:

```text
window_design="atomic_delta"
passes = 48
field sets = 32
runtime = 19 s
static g = +3.2231e+07
dynamic peak = 42.3 cm-1
output = atomic_delta_perm_asr_N100_s5.dat
```

This passes the first safety gate: it does not return to the `atomic_asr`
/ plain-like 52 cm-1 failure. It tracks the raw permutation-correct atomic
short smoke (`42.2 cm-1`, `g=+3.6817e+07`) while adding the acoustic
projector.

Limitations:

- This is the q-space Delta projector path, not yet the real-space
  minimum-change `G Delta=b` factorization.
- It should be viewed as a conservative near-Gamma stabilizer. The next
  required check is the N=300/20 SnTe run and then an explicit acoustic
  vertex diagnostic.

#### Why Delta is mathematically safer than the failed `atomic_asr`

This plan is not based on certainty that every possible atom-resolved support
will admit a tiny ASR correction. The mathematical claim is narrower and
stronger:

```text
Delta-ASR changes only the component of K_raw that violates the chosen ASR
constraints, while atomic_asr changed unconstrained components too.
```

Let `C` be the affine ASR + partition constraint set:

```text
C = { K : G K = d }.
```

The Delta construction is the metric projection of the raw atomic kernel
onto this set:

```text
K_Delta = argmin_{K in C} ||K - K_raw||_H
Delta   = K_Delta - K_raw.
```

Equivalently,

```text
Delta = H^{-1} G^T (G H^{-1} G^T)^{-1} (d - G K_raw).
```

Therefore:

1. **Idempotence on good kernels.** If the raw kernel already satisfies the
   constraints (`G K_raw = d`), then `Delta = 0`. The correction cannot
   change an already ASR-compatible centered kernel.

2. **Only residual-driven changes.** The correction depends only on
   `r = d - G K_raw`, the ASR residual. Any component of `K_raw` in the null
   space of `G` is left untouched by construction.

3. **Minimum-change theorem.** Among all kernels satisfying the constraints,
   `K_Delta` is the closest one to `K_raw` in the chosen metric. No other
   exact-ASR kernel on the same support changes the raw atomic centering less
   under that metric.

4. **A priori bound.** In the metric norm,

   ```text
   ||Delta||_H <= ||G_H^+|| * ||d - G K_raw||,
   ```

   where `G_H^+ = H^{-1}G^T(GH^{-1}G^T)^{-1}` is the weighted right inverse.
   Thus a small residual and a well-conditioned enlarged support imply a
   small correction. If the residual is not small, or the support makes
   `G_H^+` large, the diagnostic will expose it before runtime use.

The failed `atomic_asr` construction does not have these properties. It
applies a replacement/averaging operator `M`:

```text
K_atomic_asr = M K_raw,
```

where `M` makes the pinned slot uniform and factorizes an atom-averaged pair
kernel. This operator is not the metric projection onto the ASR constraint
set and is not identity on `C`. In other words, even if a component of
`K_raw` is already ASR-compatible and physically important for centering,
`M` can still remove it.

This is exactly the SnTe failure:

- the atom/basis-conditioned pinned information is mostly an ASR-compatible
  centering component;
- `atomic_asr` averages it away anyway;
- the geometry kernel changes by `~0.53` relative norm and the spectrum
  returns to the plain-like `~52 cm-1` peak.

The Delta construction is designed to avoid that failure mode:

```text
K_final = K_raw + small residual-driven Delta
```

not

```text
K_final = replacement averaged kernel.
```

The evidence in favor of the approach is therefore:

- **Structural:** it has the same minimum-change/additive character as
  `Tensor3.Apply_ASR`.
- **Projection-theoretic:** it is exactly identity if raw atomic is already
  ASR-clean on the chosen support.
- **Empirical oracle:** on SnTe, the actual tensor ASR after centering is
  small (`1.7e-3` real-space L2 and `~5e-3` on interpolated blocks), while
  `atomic_asr` is large (`~0.53` geometry change). This strongly suggests
  that the correct ASR operation is small and additive, not a replacement.

What is not guaranteed:

- If the atom-resolved raw kernel has a large ASR residual on the chosen
  support, Delta will necessarily be large. No exact-ASR method can avoid
  changing the kernel in that case.
- If the support is too small or badly conditioned, the minimum-change
  correction may be amplified. The remedy is to enlarge `Far`, change the
  geometry metric, or abandon exact ASR for that support.
- Therefore the first implementation must be diagnostic, with hard stop/go
  gates on `||Delta||/||K_raw||`, ASR residual, partition preservation, and
  SnTe tensor-oracle block error.

This is why the Delta plan is more appropriate than `atomic_asr`, but also
why it must be validated before promotion to a runtime mode.

---

# Fourth-order (D4) interpolation: extending atomic_delta centering

Investigation started 2026-07-04 (branch `qspace_interpolation`). Companion
running log + scripts:
`.../SnTe_FF/Spectral/TDSCHA_Interpolate/D4_interpolation_plan.md`.

## Objective

The D3 vertex now has permutation-correct atom-resolved centering
(`window_design="atomic_delta"`). **D4 is still always interpolated on the
single PLAIN pass.** Investigate whether atomic_delta-style centering can and
should be extended to the 4-leg D4 vertex (respecting its permutation
symmetry), and benchmark it in a regime where D4 is *fundamental*.

## The D4 estimator: leg and permutation structure

From `get_perturb_averages_qspace_slots_kernel` (`Modules/tdscha_qspace.jl`,
D4 block). The fourth order contributes to the two-phonon output block
`d2v(q1,q2)` (q1+q2 = q_pert) a sum of products of **four real-space Bloch
fields = three displacements x + one force y** (the SSCHA ⟨uuuf⟩ estimator),
in a factorised (internal scalar) × (external bilinear) form:

- **Term A** (`total_wD4 = -total_sum·ρ/8·s4`):
  internal `total_sum = Σ_pairs conj(x_w@qa)·α1·conj(x_v@qb)` (2 displacement
  legs); external `(f_Y x_w@q1)(y_v@q2) + (y_w@q1)(f_Y x_v@q2)` (1 disp + 1
  force).
- **Term B** (`total_wb = -buf_f_weight·ρ/4·s4`):
  internal `buffer_u = α1·conj(x)`, `buf_f_weight = Σ buffer_u·f_psi·conj(y)`
  (1 disp + 1 force); external `(f_Y x_w@q1)(f_Y x_v@q2)` (2 disp).

Both internal (qa,qb) and external (q1,q2) pairs sum to q_pert; the vertex is
Φ⁴(q1,q2,−qa,−qb), fully symmetric in its 4 legs. The single force leg
rotates over the four vertex positions through the two terms plus the r1↔r2
symmetrisation — the D4 analogue of the D3 1/3+2/3 permutation split.

## The structural constraint (central design problem)

The slot kernel exposes three field slots (z = q_pert leg, w, v = pair legs)
but **D4 uses only slots w and v** — the z slot is D3-only — and it *reuses*
w,v for both the internal and the external leg pairs. So the four D4 legs are
served by only **two** window families. A general atomic centering ("pin one
atom, minimal-image the other three relative to it") is therefore **not
expressible in the current 3-slot kernel for D4**. Options:

1. **Reuse the D3 atomic passes for D4 (2-window centering).** Run the D4
   term (`compute_d4=True`) through the existing atomic windowed passes (w,v
   carry the atomic minimal-image windows; z is the pinned delta, irrelevant
   to D4) instead of the single plain pass, with the partition-of-unity
   normalisation that already makes the windowed passes sum to the plain
   estimator in the commensurate limit. Centers the two independent leg-pairs;
   the third relative vector's centering is only partially captured. Cheapest;
   no Julia change. **First thing to try if the baseline fails.**
2. **4-slot kernel.** Add a fourth field slot so the internal and external
   pairs can carry independent atomic windows (pin one leg, window three).
   Full 4-leg centering + permutation, but ~2× the field sets and a Julia
   kernel extension. Only if (1) is insufficient.
3. **Tensor-D4 hybrid** (analogue of `d3_mode="tensor"`): deterministic D4
   vertex from a centered Φ⁴ tensor. Exact centering but needs the FC4 tensor
   and is O(nb⁴·n_pairs) — the "oracle", not the fast production path.

Ordering by cost/complexity: (1) → (2) → (3). **Measure first**: only build
machinery if the plain-D4 baseline actually fails to reproduce the direct
fine-mesh D4 renormalisation.

## Benchmark: SnTe R3m at ~180 K (the D4-fundamental case)

Paper (Monacelli et al., JPCM 33, 363001 (2021), Fig. 3): in the R3m phase
the bubble (D3-only) Hessian is inaccurate and the full Hessian **with D4** is
required. At 180 K the third order softens the A1 order-parameter ([111]
Sn–Te) mode enormously; D4 corrects it. Coarse-cell (2×2×2, N=50000) free
energy Hessian, verified:

| A1 mode | aux (SSCHA) | D3-only bubble | D3+D4 full |
|---|---|---|---|
| ω (cm⁻¹) | 76.2 | 25.3 | **39.1** |

D3: −51 cm⁻¹; **D4: +13.9 cm⁻¹** — a large, stable D4 renormalisation. The
2×2×2→4×4×4 interpolation must reproduce this fine-mesh D4 effect (static +
Γ spectral). Ladder: (R) direct 4×4×4, (B) interp atomic_delta + plain D4,
(C) interp + D4 centering if B fails. Scripts and running results in the
companion `D4_interpolation_plan.md`.

## 2026-07-05: Verification of the current code (permutation + ASR status)

Full audit of `QSpaceInterpolation.py` + `tdscha_qspace.jl` after the R3m
campaign. Unit tests: `test_atomic_windows.py` + `test_windows.py` +
`test_d4_interp.py` all green (21 tests).

### D3 `atomic_delta` — verified properties

1. **Permutation symmetry: exact by construction.** The three force channels
   (`d3_force_z/w/v`) pin the slot that carries the force leg and window the
   other two legs relative to the pinned atom, matching the Julia 1/3 + 2/3
   permutation split term by term. Channel-split regression
   (`test_windows.py`) proves `force_z + force_w + force_v` == the
   all-channel D3 call on identical fields.
2. **ASR:** hard partition (unit folded class sums, geometry test in
   `test_atomic_windows.py`) gives machine-zero acoustic vertices at exact
   Gamma; the q-space Delta projector (sinc-weighted acoustic subtraction in
   `_build_field_set`) suppresses the near-Gamma leak by ~1.4e4 (SnTe
   production benchmark 2026-07-04).

### Current `d4_center` (Option 1) — verified properties

- In `get_atomic_window_passes(pinned_slot=0)` the w and v maps are the
  *same dict*, so after field-set dedup `iw == iv` in every channel-0 pass:
  **all four D4 legs (internal w,v + external w,v) carry the identical
  atom-resolved window** `W_{a,o}`. The per-pass effective D4 kernel is
  `W⊗W⊗W⊗W`, manifestly symmetric under any 4-leg permutation, and the
  (a,o) average preserves that. Permutation symmetry: exact.
- **ASR:** the delta projector in `_build_field_set` acts on *every* field
  set (it is applied after the NUDFT, independent of the window), so the
  D4 fields inherit the same acoustic subtraction as D3. Note that even
  with `d4_center=False` the PLAIN fields (`self.X_q/Y_q`, built at line
  ~1119 through `_build_field_set(None)`) get the projector when
  `window_design="atomic_delta"` — uncentered D4 is already ASR-protected
  in that mode; centering and ASR are independent switches.
- **Commensurate identity:** each pass collapses to the plain estimator at
  coarse q (unit class sums), so the `1/len(ch0)` average equals plain-D4
  exactly (verified 1e-15 on R3m identity test).
- **What is NOT correct about Option 1: the centering reference.** The
  pinned atom `(a,o)` is not one of the four D4 legs; all four legs are
  imaged relative to an *external* reference. At L=2 every leg suffers the
  tie dilution (split weights), and the products multiply the dilution over
  four legs. This is the structural explanation consistent with the R3m
  observation: the centered D4 block differs from plain by 78% while the
  physical observable does not improve.

### Option 2 design (pin-one-leg D4 centering), from the Julia kernel audit

The slot kernel's D4 force leg occupies four positions:

```text
force on E1 (external w):  term A, the  y_w(q1) * f_Y x_v(q2)  part
force on E2 (external v):  term A, the  f_Y x_w(q1) * y_v(q2)  part
force on I1 (internal w):  term B via buf_f_weight_w = (alpha1 conj(x_v)) f_psi conj(y_w)
force on I2 (internal v):  term B via buf_f_weight_v = (alpha1^T conj(x_w)) f_psi conj(y_v)
```

A permutation-correct atomic D4 needs four channels, each pinning the
force-leg atom and windowing the other three legs relative to it. Since the
kernel reuses the w/v field arrays for the internal legs, this requires:

1. Julia: four extra (optional) field arrays `Xw_int, Yw_int, Xv_int,
   Yv_int` defaulting to the external ones, plus four D4 channel booleans
   `d4_force_e1/e2/i1/i2` gating the four force-position terms.
2. Python: per channel and pinned (a,o), pass the delta field set on the
   pinned leg and the atomic window `W_{a,o}` on the other three; sum over
   (a,o) *without* the 1/(nat*Nc) prefactor (the delta reconstructs the leg
   sum, exactly like the D3 channels). Commensurate identity then holds per
   configuration by the same argument as D3.
3. No new field sets: the delta sets (pinned atom x origin) and the atomic
   window sets already exist among the D3 field sets (32 for SnTe).
   Cost: 4*nat*Nc = 64 D4-only kernel calls per Lanczos application.

## 2026-07-05: Pin-one-leg D4 centering IMPLEMENTED (`d4_center="leg"`)

Key simplification discovered during implementation: **no internal field
arrays are needed in Julia.** In each force channel the pinned delta touches
only the FORCE (Y) field of one slot, while the displacement (X) field of
that slot belongs to other legs of the same channel and stays windowed:

- W-force group (channels ew+iw): slot w X is internal-displacement (E-w
  channel) or external-displacement (I-w channel) -> always the atomic
  window W_{a,o}; slot w Y is the external force (ew) or internal force
  (iw) -> always the pinned delta. Slot v: X = W, Y unused.
- V-force group (channels ev+iv): mirror image (Yv = delta, rest W).

So the centering is realized purely by (a) four D4 force-channel booleans
`d4_force_ew/ev/iw/iv` in the Julia kernels (scalar + batched + entry
point, defaults true = old behavior), and (b) Python-side field mixing
`(X from window set, Y from delta set)`. Two kernel calls per pinned
(atom, origin) -> 2*nat*Nc = 32 D4 calls for SnTe (not 64).

Normalization: each gated term is LINEAR in the delta-carrying Y field, so
summing pinned atoms and full coarse origins reconstructs the force-leg
atom sum with NO 1/(nat*Nc) prefactor (exactly like the D3 channels), and
at commensurate q the W legs collapse to plain per configuration -> the
sum equals plain D4 exactly.

API after the safety patch: `d4_center` takes False | True/"reference"
(legacy external-reference centering) | "leg" (explicit pin-one-leg
diagnostic). The initial `True -> "leg"` alias was removed after the SnTe
wrong-sign benchmark.

Hermiticity note: a single channel's kernel delta(force leg) x W x W x W is
not permutation symmetric, but the four-channel sum
sum_legs delta(leg) W W W is fully symmetric under leg exchange -- same
structure as the D3 three-channel sum. Verified through the Lanczos b/c
coefficients in the benchmark runs.

Tests (`tests/test_interpolation/test_d4_center.py`, 3 green):

1. `test_d4_channel_split_equals_full`: (ew+iw) + (ev+iv) gated calls ==
   all-channel D4 call on identical plain fields (1e-12, protects the
   4-way permutation split in both Julia kernels).
2. `test_d4_center_commensurate_identity[leg|reference]`: interp onto the
   coarse mesh reproduces the plain-pass D4 operator to 1e-10.

## 2026-07-05: Clean cubic benchmark chosen (Fm-3m T=280, quartic x3)

Probes (`fm3m_probe.py`, N=10000, coarse 2x2x2 free-energy Hessian):

```text
config                interp 4x4x4      Gamma TO bubble -> full   D4 shift
Fm-3m T=280 (as-is)   clean (min 40.5)  21.68 -> 24.36            +2.7
Fm-3m T=250 (as-is)   clean (min 39.9)  18.19 -> 21.14            +3.0 (dyn slightly sym-broken)
Fm-3m T=280 p4 x3     clean (min 40.5)  21.68 -> 39.76            +18.1
```

The p4-scaled model (p4=-0.066, p4x=-0.042, p3 unchanged) makes D4 as
fundamental as in the R3m case (+13.9) while keeping the perfectly cubic
Fm-3m cell: no symmetrization bias, no imaginary interpolated modes, no
masking confound. Scaling the quartic couplings is legitimate for the
interp-vs-direct benchmark because both sides share the identical model
and ensemble protocol; the direct fine-mesh Lanczos is the exact target
regardless of SSCHA self-consistency.

## 2026-07-05: D4 ASR diagnostic (fm3m_d4_asr.py, N=300)

With R1=0 the d2v output is PURE D4 (the D3 d2v weights are R1
contractions), so acoustic-band row norms of the d2v blocks measure the
acoustic-leg D4 vertex directly. Results on the interpolated 2x2x2->4x4x4
operator (max acoustic row norm; optical rows ~6e-8 for scale):

```text
design         d4_center   |D4_ac| at first shell |q|=0.066
atomic         plain       4.7e-08
atomic         reference   1.8e-08
atomic         leg         3.5e-08
atomic_delta   (same three, within 5%)
```

Interpretation (important, do not over-claim):

1. At EXACT Gamma the acoustic modes are masked (valid_modes_q), and the
   plain estimator additionally satisfies per-configuration Newton/COM sum
   rules while the atomic windows have unit folded class sums -- exact-
   Gamma D4 ASR holds structurally in every mode. (The diagnostic's "q=0"
   row picks the lowest VALID mode = the TO, so it is not an ASR probe.)
2. At the first off-grid shell the acoustic D4 coupling is PHYSICAL O(q),
   and the plain estimator (no windows, exact per-config sum rules) gives
   4.7e-8 there. Both centered modes stay at or below the plain value:
   the leg centering introduces NO spurious acoustic leak.
3. The atomic_delta q-space projector is inactive at the first shell by
   design: its sinc factor is (-1/13)^3 ~ -4.6e-4 there (N_i = 2*Far*L+1 =
   13). It only acts asymptotically close to Gamma, where for D4 the mode
   mask already removes the exact zero-mode. Hence identical numbers for
   atomic vs atomic_delta.

## 2026-07-05: First Fm-3m smoke (N=300/25) + CORRECTION (re-minimization)

Smoke results with the (NOT re-minimized, see below) dyn_T_280 aux,
Gamma TO, spectral first moment <w> and static g:

```text
direct  D3-only: g=+5.9e6 (UNSTABLE)  <w>=33.37
direct  D3+D4:   g=-3.2e7 (STABLE, w_eff=38.9)  <w>=26.12   D4 shift -7.25
interp  D3-only: g=+4.3e6 (unstable, right side) <w>=33.55  (0.18 off direct)
interp  D3+D4:
  plain:      g=+7.6e6 (WRONG: still unstable)  shift -1.65  captures  23%  L1=0.50
  reference:  g=+1.2e7 (WRONG: still unstable)  shift -3.63  captures  50%  L1=0.39
  leg:        g=-1.2e7 (stable but w_eff=64.6, OVERSHOOTS direct 38.9)
              shift +2.05  captures -28% (WRONG SIGN in <w>)  L1=0.60
```

Two headline facts:

1. **Plain-D4 fails qualitatively on the clean cubic cell**: it misses the
   D4 stabilization of the TO mode entirely (static stays unstable). The
   R3m under-capture is confirmed as a real, system-independent D4-vertex
   aliasing effect at L=2. D4 centering is genuinely needed.
2. **The first leg-mode result overshoots** (static lands at 64.6 vs 38.9;
   <w> moves the wrong way). Since the commensurate identity and channel
   split are exact, the problem is off-grid only: either an implementation
   subtlety (adjoint pairing / orientation of the gated channels), a
   centering-geometry artifact, or N=300 noise. Under investigation with a
   toy-chain L=2 replica (high statistics, seeds) before trusting any
   production run.

**CORRECTION (Lorenzo, 2026-07-05): the p4x3 benchmark must be run at the
SSCHA minimum of the SCALED model.** The smoke above used dyn_T_280
(converged for the ORIGINAL p4) as generating/aux dyn with x3 forces:
importance sampling is still exact (ensemble generated from the aux
itself) and centroid forces vanish by cubic symmetry, but Phi != <d2V> --
the aux propagators are not those of the scaled model's SSCHA minimum, so
the measured "D4 shift" is not the true D4 effect at the minimum.
Re-minimized with `fm3m_minimize_p4x3.py` (2x2x2, T=280, N=10000/pop,
converged): **aux Gamma TO 53.98 -> 86.34 cm-1** (the stronger quartic
tadpole is resummed into the aux). All production benchmarks now use
`Spectral/dyn_T280_p4x3_` and the coarse-cell Hessian probe is being
re-run at the minimum.

## 2026-07-05: Toy-chain L=2 replica exonerates the leg mode

Diatomic chain (generic basis offsets = the L=2 tie regime), coarse
(1,1,2) -> fine (1,1,6), g3=0.1 g4=1.0, N=4000, two seeds, three probes.
D4 contribution to the renormalization (full - D3only, Ry units x1e-4):

```text
probe        direct   plain   reference   leg
seed 11
nz=1 b=4     +1.9     +1.0    +1.0        +1.2
nz=2 b=3     +0.8     +0.4    +0.4        +0.5
nz=2 b=5     +1.3     +0.6    +0.6        +0.8
seed 42
nz=1 b=4     +1.7     +1.1    +1.1        +1.3
nz=2 b=3     +0.8     +0.5    +0.4        +0.6
nz=2 b=5     +1.1     +0.7    +0.7        +0.8
```

Conclusions:

1. The leg mode is CORRECT-behaving: right sign, reproducible across
   seeds, and consistently captures MORE of the direct D4 contribution
   (~60-70%) than plain or reference (~50%). No wrong-sign anomaly.
2. The Fm-3m N=300 leg overshoot was therefore noise amplified by the
   near-critical, non-minimized regime -- production needs the
   re-minimized dyn and larger N.
3. Side observation (separate issue): at L=2 the chain's D3-only interp
   under-captures the cubic renormalization by ~30% (interp -2.0e-4 vs
   direct -2.8e-4 at nz=1) even with atomic_delta -- the chain L=2 case
   is harder than SnTe for D3 centering too. Not pursued now.

## 2026-07-05: Resummation lesson -- benchmark redesigned around T, not p4

Probe at the re-minimized minimum of the p4x3 model: bubble 77.02 -> full
78.09, **D4 shift only +1.07 cm-1** (was +18.1 in the unconverged setup;
original model at its own minimum: +2.7). Scaling the quartic up is
COUNTERPRODUCTIVE: at the SSCHA minimum the quartic tadpole is resummed
into the aux (Gamma TO 54 -> 86 cm-1), the hardened aux suppresses the
fluctuations, and the residual D4 in the response SHRINKS. This is the
quantitative face of the paper's statement that the bubble is accurate in
the high-symmetry phase.

First idea: TEMPERATURE (D4 Hessian correction ~ <u^2>^2 ~ T^2 vs bubble
~ T). T=400 re-minimized: aux TO 63.6, bubble 31.2 -> full 34.2, D4 shift
+3.05, interp clean. Production run launched there (N=2000).

**Lorenzo's better recipe (2026-07-05): scale D3 (and D4 together), not
T.** Raising T hardens the aux through the loop (the loop diagram
dominates). The D4 diagrams that matter in the response enter through the
D3-opened two-phonon channel and scale as d3^2*d4 -- exactly like the
bubble's d3^2 -- so raising d3 amplifies the D4 term over BOTH the bubble
(relative headroom via d4) and the loop (which has no d3). Crucially, in
the cubic phase the SSCHA minimum is EXACTLY p3-independent (parity:
<d2V> gets d3*<u>=0; the stochastic gradient's p3 part is an odd Gaussian
moment; centroid forces vanish on the rocksalt inversion centers), so
**only p4 changes require re-minimization; p3 can be scanned freely on a
fixed converged dyn.** The constraint is bubble stability: at T=280
original couplings the headroom is only p3 x1.09 (Sigma_bubble = 54^2 -
21.7^2 cm^-2 nearly exhausts aux^2), hence p4 must be raised WITH p3: the
d4 loop hardens the aux to buy back the stability the stronger bubble
spends. Plan: minimize p4x2 at T=280 (dyn_T280_p4x2.0_), scan p3 x
{1.25, 1.5, 1.75} with the Hessian probe (no re-minimization needed),
pick the combo with the full TO comfortably stable on coarse+fine mesh
and the largest D4 share, then run the production benchmark there.

### p3 scan at the p4x2 minimum (T=280, aux TO 73.79)

```text
p3scale   coarse bubble   coarse full   D4 shift
1.3       47.56           50.53         +3.0
1.5       34.76           39.95         +5.2
1.6       24.99           ~33           ~+8
1.7        1.71           22.39         +20.7
```

Sigma_bubble = 1883*s^2 cm^-2 and Sigma_D4 = 172*s^2 exactly (verified at
all scan points) -- Lorenzo's d3^2 scaling is quantitative.

### Fine-mesh reality: Gamma is unusable, first shell is the flagship

Direct 4x4x4 smokes (N=500): at Gamma the static response is UNSTABLE for
both D3-only and D3+D4 at every p3scale >= 1.5 (the dense two-phonon mesh
over-softens by a factor Sigma_fine/Sigma_coarse ~ 1.3-1.4; the <w> D4
shift there is < 2 cm-1 -- weak observable). Off Gamma at p3x1.6:

```text
|q|=0.0660 (FIRST SHELL of 4x4x4 = genuinely INTERPOLATED q):
  band 5: D3only w_eff=35.2 STABLE -> D3+D4 w_eff=53.1 STABLE
          (D4 shift +17.9 cm-1 on a stable static!)  <w>: 55.6 -> 64.7
  band 0: D3only marginal (w_eff 113) -> D3+D4 35.6 (stabilization flip)
|q|=0.0762 ((1/2,0,0)-type, COMMENSURATE -> useless for centering test):
  band 5: D3only UNSTABLE -> D3+D4 w_eff=25.8 (sign flip, on-grid only)
```

**PRODUCTION BENCHMARK (launched, pid 374595):** T=280,
dyn_T280_p4x2.0_, p4x2, p3x1.6, probes shell1:band5 + shell1:band0,
N=2000, NSTEP=30, direct vs interp atomic_delta with d4 in
{plain, reference, leg}. Observables: w_eff (static), <w>, L1 lineshape.
Log: fm3m_bench_T280_p4x2_p3x1.6_N2000.log. The T=400 N=2000 Gamma run
(weak D4, both statics unstable) continues as a null-test data point.
