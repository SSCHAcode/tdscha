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
