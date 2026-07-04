# Atomic D3 Centering: Permutation Symmetry + ASR Implementation Plan

Status: started after commit `a6abf0f0` on branch `qspace_interpolation`.

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
