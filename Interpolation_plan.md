# Interpolation plan for the q-space TDSCHA Lanczos

**Implementation status (branch `qspace_interpolation`).**
- DONE (M2+): `Modules/QSpaceInterpolation.py` (`QSpaceLanczosInterp`), plain-window
  NUDFT fields with ASR zero-mode projection, interpolated dyn (Tensor2 centering +
  Apply_ASR, TRI gauge), O(N_f) pair map/hash lookup, `scale3`/`scale4` in
  `tdscha_qspace.jl`, **field pre-filtering** (section 5.6 — found to be essential,
  not optional: without it the toy-chain renormalization is systematically ~2x too
  small because the raw correlations carry the phonon-propagator range).
  Tests in `tests/test_interpolation/` (V0 machine-precision identity, unit tests,
  physics validation vs a direct fine-supercell ensemble incl. non-TRI q,
  scale-factor necessity test, ASR field-invariance test).
- DONE (M3): designed multitaper windows (`window_design="minimal_image"`):
  slot-resolved ALS designs (numerically exact minimal-image kernels for L=3, K=3
  and L=4, K=2 with hard partition-of-unity constraints), slot-resolved Julia
  kernel with w<->v orientation averaging, origin averaging by data permutation.
  Measured (chain toy, Lc=3 -> Lf=6, N=4000): noise floor 0.103, plain 0.299,
  windows 0.158, windows + 3 origins 0.122 — the designed windows reach the
  statistical noise floor. IMPORTANT deviations from the original plan found
  during implementation: (a) the window-weighted per-config ASR projection is a
  rank-one modification of the effective window and BIASES the designed kernels
  (~30% on the toy) — it is applied only to (near-)uniform windows; design-level
  ASR remains future work; (b) origin shifts must permute the DATA, not
  cyclically shift the window (support wrap destroys the kernel).
- DONE (M4): batched BLAS-3 slot kernel (sparse x dense rotations + GEMM
  accumulations, chunked over configs at fixed symmetry), equivalent to the
  scalar kernel at 1e-20; near-linear wall-time scaling measured N_f = 8..128.
- DONE: report/interpolation/main.tex (full math, derivation appendices,
  benchmarks, example application vs the standard Spectral.py d3 bubble: peak
  agreement within the energy-grid step at interpolated q).
- DONE: design-level ASR for non-plain windows (section 5.7,
  `window_design="asr"`): doubled-support (2L) windows with exactly uniform class
  sums = one-shot centering + ASR + symmetrization construction (no iterative
  Apply_ASR analogue needed); explains the §5.5 field-projection bias as a
  one-period no-go theorem. Pairwise-toy precision kept (asr+3origins 0.150 vs
  minimal-image+3origins 0.122, floor 0.103, plain 0.299) with the vertex ASR
  exact at every q̃ (deterministic 1e-15; stochastic plateau removed). Leak
  geometry, pairwise immunity and the WS-resolution rule in §5.7(f); report
  section + figures in report/interpolation (asr_kernel.pdf, asr_leak.pdf);
  8 new tests in tests/test_interpolation/test_asr_windows.py.
- DONE (2026-07-04): **hybrid tensor-D3 mode** (`d3_mode="tensor"`,
  section 5.8) — fix for the real-material failure found on SnTe
  (2×2×2 → 4×4×4 interpolation stuck at ~52 cm⁻¹ vs the correct ~38): at L=2
  no window design can represent the atomic-basis, perimeter-minimizing
  centering of `Tensor3.Center(Far=3)`, yet the SAME 2×2×2 stochastic d3,
  centered that way, reproduces the direct 4×4×4 result through the old
  Spectral bubble. Since every D3 term of the Lanczos has one leg pinned at
  q_pert and the estimator is linear in the D3 correlations, the D3 channel
  is replaced by a deterministic contraction of the centered Tensor3
  interpolated at the fine pairs — by construction identical to Spectral's
  vertex interpolation. VALIDATED: SnTe interp 2×2×2→4×4×4 dynamic TO peak
  37.6 cm⁻¹ = direct 4×4×4 (lineshape L1 0.054; was 52.3 broken); identity
  static = free-energy-Hessian reference 21.28 exactly; 5 new toy-chain
  tests green (`tests/test_interpolation/test_d3_tensor_mode.py`). Full
  numbers in §5.8(f) and in the SnTe issue file
  (`.../SnTe_FF/Spectral/TDSCHA_Interpolate/issues.md`).
- DONE (2026-07-04): **tensor-free atomic centering**
  (`window_design="atomic"`, section 5.9) — production-compatible fix for
  the same SnTe 2×2×2→4×4×4 failure without computing/storing Φ³. The
  q_pert leg is pinned to one primitive atom and all coarse origins; the
  pair legs use geometry-only atomic-basis minimal images with extended
  images folded back by Bloch phases. SnTe Γ TO D3-only benchmark:
  N=1000, 30 steps peak 37.7 cm⁻¹ (direct 37.56, tensor-D3 37.64,
  broken plain 52.28), normalized L1 vs direct 0.0865 and vs tensor-D3
  0.0466. Added pure geometry tests in
  `tests/test_interpolation/test_atomic_windows.py`.
- TODO: LO-TO, off-mesh q_pert, IR/Raman sqrt(N_f) prefactor, distributed-mode
  support, KPM variant (M5, skipped by decision).

**Goal.** Run the full q-space TDSCHA Lanczos (`Modules/QSpaceLanczos.py` +
`Modules/tdscha_qspace.jl`) on a **fine** q-mesh not commensurate with the supercell of
the stochastic ensemble, while the ensemble stays on the **coarse** commensurate mesh.
The two-phonon sector of the Lanczos must live on the fine mesh, so that a phonon at
`q_pert` can decay into pairs of phonons at *interpolated* q-points (dense two-phonon
continuum, converged linewidths). Constraints:

1. **No high-order tensors.** Never compute or store the 3rd/4th order force-constant
   tensors (unlike `cellconstructor.Spectral` / `ForceTensor`, which build and center
   Φ⁽³⁾ before Fourier-interpolating it).
2. **Fast, linear scaling** in the fine-mesh size `N_f` (per Lanczos iteration and in
   memory).
3. Reuse the existing stochastic machinery (ensemble, weights, symmetries, Hermitian
   Lanczos, continued fraction) with minimal structural change.

---

## 1. Background: what the q-space Lanczos actually computes

State vector (`QSpaceLanczos._compute_block_layout`,
`Modules/QSpaceLanczos.py:310`):

```
psi = [ R(q_pert)_ν  |  a'(q1,q2)_{ν1ν2}  |  b'(q1,q2)_{ν1ν2} ]
```

with one `(n_bands × n_bands)` block per unique pair `(q1, q2)` satisfying
`q1 + q2 = q_pert + G` (`build_q_pair_map`, `Modules/QSpaceLanczos.py:273`). The
diagonal pairs `q1 = q2` use upper-triangle storage (blocks are complex *symmetric* in
the bilinear convention). The L operator splits as:

- **Harmonic** (`apply_L1_FT`, `Modules/QSpaceLanczos.py:514`): diagonal,
  `-(ω_{q_pert,ν})²` on R, `-(ω1∓ω2)²` on a'/b'. Needs only ω(q), e(q).
- **Anharmonic** (`apply_anharmonic_FT` → Julia
  `get_perturb_averages_qspace_fused`, `Modules/tdscha_qspace.jl:466`): a **stochastic
  estimator**. For each `(configuration, point-group symmetry)` it contracts the
  per-configuration Bloch fields

  ```
  x(q)_ν = Σ_a conj(e_ν^a(q)) √m_a ũ_a(q),     ũ_a(q) = N_c^{-1/2} Σ_R e^{-i q·R} u_a(R)
  y(q)_ν = Σ_a conj(e_ν^a(q)) m_a^{-1/2} δf̃_a(q),   δf = f − f_SSCHA − ⟨f⟩
  ```

  (convention of `vector_r2q!` in python-sscha `fourier_gradient.jl`: phase
  `e^{-2πi q·R}` on the **cell origin** `R` of each supercell atom, normalization
  `1/√N_c` where `N_c` = number of coarse cells = number of coarse q-points)
  into weights and rank-1 outer products:

  | Julia quantity | fields involved | vertex order |
  |---|---|---|
  | `weight_R`  = Σ f_Y x*(q_pert)·R1        | x*  → ×(x(q1) y(q2) dyad) | 3-field, **D3** |
  | `weight_Rf` = Σ R1·y*(q_pert)            | y*  → ×(x(q1) x(q2) dyad) | 3-field, **D3** |
  | `w1` = −total_sum/2 (α1 : x*x*) → f ∝ y(q_pert) | x*x*y | 3-field, **D3** |
  | `w2` = −buf_f_weight (α1 : x* f_ψ y*) → f ∝ x(q_pert) | x*y*x | 3-field, **D3** |
  | `total_wD4` = −(α1 : x*x*)/8 → ×(x y dyad) | x*x*xy | 4-field, **D4** |
  | `total_wb`  = −(α1 : x* f_ψ y*)/4 → ×(x x dyad) | x*y*xx | 4-field, **D4** |

  These are **exact Gaussian integration-by-parts extractors**: in the infinite-sampling
  limit, the 3-field averages equal contractions of ⟨∂³V⟩ and the 4-field averages
  equal contractions of ⟨∂⁴V⟩ (disconnected Wick pieces vanish because ⟨x⟩ = ⟨y⟩ = 0
  and ⟨x y⟩ ∝ ⟨∂²V⟩ − Φ_SSCHA = 0 at SSCHA stationarity — see §6.3).

**The crucial structural fact:** the estimator needs only the per-configuration Bloch
fields `x(q), y(q)` at the q-points where the psi blocks live, plus ω(q), e(q) there.
The tensors Φ⁽³⁾, Φ⁽⁴⁾ never appear. Therefore interpolation = providing *consistent*
`x(q̃), y(q̃), ω(q̃), e(q̃)` at arbitrary q̃, plus a normalization correction (§6).

---

## 2. Why the naive off-grid Bloch sum is NOT a valid interpolation

The per-configuration data `u(R)` is periodic with the supercell: it contains
information only at the `N_c` commensurate q-points. Three equivalent statements of
the problem:

- Fourier-transforming the *infinite* periodic repetition of `u(R)` at an
  incommensurate q̃ gives **exactly zero** (delta comb on the commensurate points).
- Evaluating the *finite* sum `Σ_{R∈D} e^{-iq̃·R} u(R)` over one fundamental domain D
  is implicitly a **rectangular window** (zero-padding with the *wrong* support): it
  does not vanish, but it is a windowed estimate whose quality depends entirely on
  which periodic image of each correlation component gets which phase.
- In tensor language: Fourier interpolation of Φ⁽ⁿ⁾ is only meaningful after
  **centering** — re-assigning each matrix element to the periodic image(s) with the
  minimal inter-atomic distances, zero-padding everything beyond half the supercell.
  This is what `ForceTensor.Center()` and `Spectral.py` do, and it is exactly the step
  that requires storing the high-order tensor.

The plan must therefore (a) show precisely what interpolation the finite Bloch sum
implements (§4), and (b) construct the analogue of the centering *without tensors*
(§5). The key enabler is that the estimator is **multilinear in the per-configuration
fields**, so windowing the *configurations* induces designable windows on the
*correlations*.

### 2.1 What the ensemble average of off-grid products yields

Take three off-grid fields with the exact constraint `q̃1 + q̃2 = q̃3` (the pair map
guarantees this on the fine mesh; `e^{iG·R} = 1` for primitive-lattice G, so
conservation is *exact*, not approximate). Using ensemble translation invariance
(⟨u(R1) f(R2) u(R3)⟩ = c₃(R1−R3, R2−R3), periodic in each argument mod supercell):

```
⟨ x̃(q̃1) ỹ(q̃2) x̃*(q̃3) ⟩  =  N_c^{-1/2} Σ_{δ1,δ2}  K₃(δ1, δ2) · c₃^∞(δ1, δ2) · e^{-i q̃1·δ1 - i q̃2·δ2}
```

where the sum runs over **all periodic images** of the physical (short-ranged)
correlation c₃^∞ and `K₃(δ1,δ2)` is an **effective interpolation kernel** determined
purely by how the finite sums are windowed. The whole interpolation quality is the
question: *how close is K₃ to the minimal-image indicator?*

---

## 3. Core idea: windowed per-configuration Bloch fields ("stochastic centering")

Define, per configuration I and per **window pass** r:

```
x̃_r(q̃; I)_ν = Σ_a conj(e_ν^a(q̃)) √m_a · N_c^{-1/2} Σ_R  W_r(R − R0_I) e^{-i q̃·R} u_a(R; I)
```

- `W_r(n)` is a separable window over the fractional cell index,
  `W_r(n) = w_r^{(1)}(n_1) w_r^{(2)}(n_2) w_r^{(3)}(n_3)`, real-valued, supported on at
  most one period per dimension (**this is the zero-padding**: data outside the window
  contributes 0, no wrapped phases).
- `R0_I` is a random origin per configuration (frozen at initialization so L is a
  fixed operator during the Lanczos run); the data is accessed periodically
  (`u(R mod supercell)`), the window is not. Random origins restore translation
  invariance of the *kernel* at no extra cost (deterministic full-origin averaging is
  an optional ×N_c exact mode).
- The anharmonic estimator becomes a small sum of passes,
  `L_anh = Σ_r c_r · E[x̃_r, ỹ_r]` with real design coefficients `c_r`, where `E[·]`
  is the *unchanged* fused kernel evaluated on the windowed field set of pass r
  (different fields slots may carry different windows within one pass, see §5.3).

Because every term of the fused kernel is a product of three or four fields, the
ensemble average of pass r applies the kernel

```
K₃^{(r)}(δ1, δ2) = Σ_s  z_r(s) · w_r(s + δ1) · v_r(s + δ2)        (per dimension)
```

(z = window on the q_pert-slot field, w, v = windows on the two pair-slot fields;
sums over absolute positions; after origin averaging the kernel depends only on
differences). **The kernel is a designable object.** The interpolation of the D3
correlations delivered by the estimator is `Σ_r c_r K₃^{(r)}`, and similarly a
4-index `K₄` for the D4 terms.

This is precisely a *multitaper spectral estimation* scheme: the correlation
functions are Fourier-interpolated with a window designed in real space, but the
windows are applied to configurations (vectors, O(N_c·n_bands) each), never to
correlation tensors.

---

## 4. The plain estimator (single rectangular window) — baseline and its bias

With a single pass and `W = 1` on the full fundamental domain (this is *identical* to
evaluating today's `vector_r2q` at off-grid q̃), the per-dimension kernel is the
triple correlation of rect(L):

```
K₃(δ1, δ2) = (1/L) · max(0, L − spread(0, δ1, δ2)),   spread = max−min
```

**Properties (verified numerically, L = 4):**

- **Partition of unity:** `Σ_images K₃(δ + L·n) = 1` for every difference class ⇒ at
  commensurate q̃ the estimator reduces *exactly* to the current coarse one (the phases
  of all images coincide). The scheme is exact on-grid by construction.
- **Tent-shaped leakage off-grid:** the weight of a physical correlation component at
  difference δ is split between its images ∝ (1 − spread/L). Example (L = 4): class
  (δ1,δ2) = (1,0) puts weight 3/4 on the compact image and **1/4 on the image (−3,0)**,
  which enters with the wrong phase at incommensurate q̃. Worst-case bias on
  nearest-neighbor D3 components is tens of percent at mid-grid q̃ — *this is the
  failure mode the naive approach must be corrected for* (it shows up as a sagging
  interpolation between exact on-grid values).

The plain estimator is therefore: exact at commensurate q̃, O(range/L)-biased between
them, zero extra cost. It is the correct *baseline and feasible point* for the window
design, not the production scheme.

---

## 5. Window design: emulating minimal-image centering without tensors

### 5.1 Pair kernels: exact minimal-image with 2 windows (analytic result)

Let `r_m(δ) = max(0, m−|δ|)` be the autocorrelation of rect(m). Then, per dimension
with even L:

```
½ [ r_{L/2+1}(δ) − r_{L/2−1}(δ) ]  =  1   for |δ| ≤ L/2 − 1
                                      ½   for |δ| = L/2      (tie split, WS boundary)
                                      0   beyond
```

i.e. **two rectangular windows with coefficients ±½ reproduce the minimal-image
(zero-padded) pair kernel exactly**, including the Wigner-Seitz tie-splitting, and
satisfy partition of unity (verified numerically). For odd L, `r_{(L+1)/2} − r_{(L−1)/2}`
gives the exact indicator with no tie. This proves the concept: sharp centering *is*
reachable by separable per-configuration windowing.

### 5.2 Triple kernels (D3): low-rank designs (numerical results, L = 4)

The D3 kernel needs the 2D target `M(δ1,δ2)` = indicator of the minimal-spread image
(ties split), which matches the `ForceTensor` d3 centering (per-dimension
parallelepiped version). Results of the design study (constrained least squares /
ALS over window shapes, partition of unity enforced):

| scheme | passes/dim | RMS error vs minimal-image target |
|---|---|---|
| plain rect(L) | 1 | 0.247 |
| same-window rect triples, any combination | any | 0.247 (provably cannot improve) |
| **mixed rect windows + shifts, full basis** | ~10³ | **exactly 0** |
| **general real windows, symmetric ALS** | **2** | **3·10⁻⁴** |
| general real windows, symmetric ALS | 4 | numerically exact |

Interpretation: the exact minimal-image triple kernel is representable, and its
**CP rank is small** — 2 passes per dimension already give a three-orders-of-magnitude
improvement, 4 passes are exact (L = 4). In 3D the windows are per-dimension products,
so the pass count multiplies: **K per dimension ⇒ K³ field sets** (K=2 → 8 passes;
each pass costs one kernel application ⇒ constant-factor overhead, scaling unchanged).

Design procedure (offline, once per supercell shape, milliseconds):
1. Per dimension α with length L_α, set the target `M(δ1,δ2)` (minimal-spread with tie
   splits).
2. Fit `Σ_r c_r z_r(s) w_r(s+δ1) v_r(s+δ2)` by ALS/LSQ with linear constraints:
   - partition of unity over image classes (**hard constraint** ⇒ exact commensurate
     limit is preserved for any design);
   - symmetry `w ↔ v` (transpose symmetry of the pair blocks);
   - windows for symmetry-equivalent axes constrained equal (PG compatibility, §8).
3. Store `{c_r, z_r, w_r, v_r}` per dimension; 3D windows/coefficients are products.

### 5.3 Slot assignment and Hermiticity

Within one pass the three field slots are: the `q_pert` slot (fields entering
`weight_R`, `weight_Rf`, and the f_pert outputs) gets window `z_r`; the two pair
slots (fields entering dyads `r1⊗r2` and the α1 contractions) get `w_r`, `v_r`
(symmetrized). Hermiticity of L under the masked inner product holds because:

- the map `R1 → d2v` and the map `α1 → f_pert` are mutual adjoints **provided the same
  window-triple set (with the same c_r) is used for both** — the fused kernel already
  computes them from identical contractions, so this is automatic if the pass
  structure wraps the whole fused kernel;
- real `c_r` and the `w↔v` symmetrization preserve the complex-symmetric block
  structure and the a'/b' sector relations.

Monitor `|b_i − c_i|` of the Lanczos as the runtime Hermiticity check (already
printed by `run_FT`).

### 5.4 Quadruple kernels (D4)

Same machinery with the 3D difference target (4 fields ⇒ 3 independent differences).
Default plan: **plain window for the D4 terms** (D4 correlations are typically much
shorter-ranged than D3, and the D4 terms carry the smaller `N_c/N_f` weight, §6),
designed windows as an optional refinement if validation shows a need. The pass
structure allows different window sets for D3-type and D4-type terms at the cost of
one extra kernel pass group.

### 5.5 Acoustic sum rules (ASR): centering breaks them, and they MUST be restored

This is the direct analogue of what `cellconstructor.ForceTensor` does for tensors:
`Tensor3.Center()` is always followed by `Tensor3.Apply_ASR()`
(`ForceTensor.py:1468` and `ForceTensor.py:1693`), which *iteratively* re-imposes
`Σ_{third index} Φ⁽³⁾ = 0` alternated with permutation-symmetry re-imposition until a
fixed point. The reason it exists is that **centering itself destroys the sum rule**:
re-assigning matrix elements to minimal images and zero-padding the rest breaks the
exact telescoping of the translational-invariance identity that the periodic tensor
satisfies. Our stochastic centering (§5) breaks it in exactly the same way, and it
must be restored explicitly — an uncorrected violation does not degrade the result
gracefully, it destroys it.

**Why a violation is catastrophic here.** With ASR, a D3 vertex with one leg on an
acoustic branch vanishes as `O(q̃)` for q̃ → 0 (in Cartesian convention;
`O(√ω)` for the mode-normalized matrix element). This vanishing is what tames the
thermodynamic factors attached to that leg, which all diverge at small ω:
`f_psi = (1+2n)/(2ω) ~ T/ω²`, `χ± ~ 1/(ω1 ω2)`, `n(ω) ~ T/ω`. If the interpolated
vertex tends instead to a spurious constant, the two-phonon sums acquire
contributions diverging as inverse powers of `ω_ac(q̃) ≈ c|q̃|` — and the fine mesh
densely samples precisely the Γ-neighborhood where this blows up, which is the whole
point of interpolating. A per-mille ASR leak in the vertex is amplified into O(1) or
divergent artifacts in the spectral function. Note the coarse code never faces this:
the only exactly-acoustic modes are at Γ and are masked (`valid_modes_q`); the fine
q̃ near Γ **cannot** be masked without throwing away the acoustic decay channels we
are after.

**Microscopic origin of the violation in this scheme.** The *unwindowed* estimator
inherits the ASR exactly, configuration by configuration:

- `Σ_{R,a} δf_a(R) = 0` per configuration (Newton's third law for the DFT forces,
  the D2 ASR of Φ_SSCHA for `f_sscha`, and the ⟨f⟩ subtraction is a constant that
  drops out at any q ≠ 0);
- the ensemble displacements carry no net mass-weighted translation.

Hence `y(Γ)·t = x(Γ)·t = 0` identically (t = translation pattern), with **zero
statistical noise** — the coarse vertex vanishes on translation legs exactly.
Windowing destroys precisely this mechanism: forces balance only over the *whole*
supercell, so the window-weighted sums `Σ_R W_r(R−R0) δf(R)` do not vanish per
configuration; likewise the windowed center of mass. Equivalently, at the kernel
level: a centered kernel K₃ (decaying in its difference arguments) cannot have
δ-sums independent of the other argument, which is what inheriting the sum rule
would require — the same structural obstruction that forces `ForceTensor` to
re-impose ASR after centering. The violation is therefore **structural, not a small
numerical artifact**, and it appears at O(window bias) on every fine q̃ near Γ.

**Restoration: windowed zero-mode projection (the tensor-free `Apply_ASR`).**
Per (configuration, pass, origin), *before* the NUDFT, project the fields onto the
orthogonal complement of the uniform translations **with the window weight**, per
Cartesian component:

```
u_a(R)  ← u_a(R)  − d,           d   = Σ_{R,b} W_r(R−R0) m_b u_b(R) / (Σ_R W_r(R−R0) Σ_b m_b)
δf_a(R) ← δf_a(R) − (m_a/M̄) g,   g   = Σ_{R,b} W_r(R−R0) δf_b(R),   M̄ = Σ_R W_r Σ_b m_b (normalized)
```

i.e. the displacements lose their (window-weighted) mass-weighted center of mass and
the forces lose the net force redistributed **proportionally to the masses** (a rigid
counter-acceleration). These are exactly the projections onto the translation modes
`t ∝ √m ê` in the mass-scaled representation: the mass-scaled displacement field
contracts translations as `x̃(0)·t ∝ Σ W Σ_a m_a u_a`, the mass-scaled force field
(`y = f/√m`) as `ỹ(0)·t ∝ Σ W Σ_a δf_a`, and both subtracted patterns are ∝ `t`
itself, so **optical components are untouched** (a plain per-atom mean subtraction of
the force would leak into optical modes for unequal masses). After the
projection, **every field slot annihilates the q̃ = 0 translations exactly per
configuration**, so every leg of the effective D3 *and* D4 kernels vanishes on
acoustic branches as q̃ → 0 — by construction, with zero statistical noise, exactly
as in the unwindowed coarse estimator. Cost: O(N_c) per (config, pass) — negligible.
Like `Apply_ASR(power=...)`, which chooses *how* to spread the correction over the
tensor, the projection spreads the correction over the fields; the perturbation of
the finite-q̃ physics is confined to a Γ-neighborhood of width ~2π/(window size)
(the subtracted term enters other q̃ only through Ŵ(q̃), the window's Fourier
transform) and is of the same order as the interpolation resolution.

**Interaction with the kernel design (exactness bookkeeping).**

- The projection is a rank-one (per dimension, per slot) linear modification of the
  effective window, `W → W − W⊗(weighted average)`. The partition-of-unity /
  on-grid-exactness constraints of §5.2 must therefore be evaluated on the
  **projected** kernels: fold the projection into the ALS/LSQ design loop, so both
  the exact commensurate limit and the exact q̃→0 ASR hold simultaneously by
  construction.
- For the plain full-period window the projection is a no-op (`g = d = 0` exactly,
  by the per-configuration sum rules above), so the baseline scheme and the
  V0 bit-exactness test are untouched.
- Hermiticity is preserved: the projection is applied to the data once, upstream of
  the estimator; L remains a fixed Hermitian operator built from projected fields.

**Consistency on the harmonic side.** The interpolated SSCHA dyn must itself have
the D2 ASR enforced (standard `Tensor2` centering + ASR / `CustomASR`) so that
`ω_ac(q̃) → 0` cleanly along acoustic branches: the χ/f_psi factors and the vertex
must vanish *at the same points*. Keep the `w_min` guard as a safety net and abort
loudly on interpolated ω² < 0 off Γ (§7.4).

**Not enforced (same scope as ForceTensor):** rotational / Born–Huang invariances
and higher-order sum rules. The D4 double-acoustic limits are covered by the same
field projection (every leg annihilates translations); no additional constraint is
imposed beyond that.

### 5.6 Field pre-filtering: interpolate Φ-ranged objects, not propagator-dressed ones

**This refinement proved essential in validation, not optional.** The Gaussian
integration-by-parts identities mean the raw 3-field correlation is

```
⟨ x(q1) x(q2) y(q3) ⟩  =  f_psi(q1) f_psi(q2) · [Φ⁽³⁾ contraction],   f_psi = (1+2n)/2ω
```

every **displacement leg carries a phonon-propagator dressing** `f_psi`. In real
space the interpolated object therefore decays with the *propagator* range (long,
algebraic tails at low T from the acoustic `1/ω ~ 1/|q|` nonanalyticity), not with
the short range of Φ⁽³⁾. On the bond-anharmonic chain toy this produced a
systematic ~2x deficit of the interpolated renormalization that shrank only slowly
with the coarse supercell size.

The fix exploits that the fused kernel multiplies every displacement leg by
`f_Y = 2ω/(1+2n) = 1/f_psi` anyway (the `r1 = f_Y·x` fields, `weight_R`, w2):

1. apply `f_Y` **on the coarse grid** per configuration (exact there — it strips the
   dressing mode by mode), rebuild the filtered real-space configuration, and NUDFT
   *that*;
2. in the kernel set the `f_Y` and `f_psi` tables to the validity mask
   (`prefiltered=true` flag in `get_perturb_averages_qspace`);
3. fold the exact **fine-side** `f_psi(q̃1)⊗f_psi(q̃2)` into the α₁ blocks (Python,
   `QSpaceLanczosInterp._call_julia_qspace`), because the α₁-contracted legs use the
   filtered fields too.

On-grid this is an exact identity (V0 stays machine-precision). Off-grid the
interpolated correlations now decay with the range of the **anharmonic force
constants themselves** — the same locality class as Spectral.py's tensor
interpolation. The polarization-vector projections and all χ/f factors are evaluated
analytically at the fine q̃, so no band-structure information is interpolated
stochastically. Measured effect on the chain toy: systematic deficit eliminated
(ratios 0.4–0.7 → ~1.0 within noise + tent bias).

A per-configuration ASR note: the filtered displacement field needs no zero-mode
projection (`f_Y → 0` kills the Γ acoustic components exactly); the force residuals
keep the §5.5 projection.

### 5.7 Design-level ASR: one projection for centering + ASR + symmetrization

**Status: WORKING NOTES → to be validated numerically, then promoted.** This section
records the strategy for enforcing the ASR under *non-plain* (sign-oscillating
designed) windows, where the §5.5 field projection was found to bias the kernel by
~30% and had to be disabled. The goal, following the observation that in
`ForceTensor` the `Apply_ASR` projection *spoils the permutation symmetrization* and
the two must be alternated iteratively to a fixed point, is a **single, one-shot
construction** in which centering (the kernel target), the acoustic sum rule, the
commensurate-limit exactness, and the symmetrization all hold **simultaneously and
by construction** — accepting, if needed, a slightly larger window support.

**(a) What the ASR means at the kernel level.** The expectation of the windowed
estimator is the contraction of the (coarse-periodic) tensor with the pair kernel:

```
Φ_eff(δ1, δ2)  =  (1/L) S(δ1, δ2) · Φ_per(δ1 mod L, δ2 mod L),
S(δ1, δ2)      =  Σ_s z(s) w(s+δ1) v(s+δ2)          (per dimension)
```

with `(δ1, δ2)` on the extended difference grid (support set by the window
lengths). The periodic tensor inherits the exact ASR of the true Φ⁽³⁾ under mesh
folding, in the form of *period-sum rules*: `Σ_{d over one period} Φ_per(d, ·) = 0`
(v leg), same for the w leg, and `Σ_d Φ_per(d, d+Δ) = 0` for every fixed Δ (z leg,
i.e. the sum over the first index at fixed positions of the other two, which in
difference coordinates runs along diagonals). Therefore Φ_eff satisfies the ASR
**for every Φ_per allowed by the exact sum rules** iff the kernel's *image sums are
constant on each class*:

```
(ASR-v)  Σ_b S(δ1, d2 + bL)                 independent of d2   for every fixed δ1
(ASR-w)  Σ_a S(d1 + aL, δ2)                 independent of d1   for every fixed δ2
(ASR-z)  Σ_b S(d + bL, d + Δ + bL)          independent of d    for every fixed Δ
```

These are **linear constraints on S** — exactly like the partition-of-unity. In
fact they *strengthen* it: (ASR-v) + (ASR-w) imply that the total class sum
`Σ_{a,b} S(d1+aL, d2+bL)` is a single constant, so the commensurate-limit
constraint reduces to one normalization row (`= L`). This is the structural reason
a "single projection" exists here while `ForceTensor` must iterate: in tensor space
the ASR and symmetrization projectors act on a huge object and do not commute; in
window-design space **all the constraints are simultaneous linear conditions on a
tiny object** (the kernel, `O(L²)` numbers, parameterized by `3K` windows), and the
permutation symmetrization (w↔v orientation averaging + the z-leg estimator
structure) is already built into the kernel definition `S_sym`, not applied after
the fact. One constrained least-squares fit replaces the alternating projections.

**(b) A sufficient window condition, and the one-period no-go.** A window `w` has
*uniform class sums* if `W̄(d) = Σ_b w(d + bL)` is independent of `d`. Then

```
Σ_b S(δ1, d2+bL) = Σ_s z(s) w(s+δ1) V̄(s+d2) = V̄ · Σ_s z(s) w(s+δ1)   (d2-independent)
```

so **uniform class sums on the v (w) window imply (ASR-v) ((ASR-w)) identically**,
for *any* other two windows. This immediately explains the §5.5 failure as a no-go
theorem: for windows supported on **one period** (length L, the current design),
the class sum is the window itself — uniform class sums ⇔ the plain window. An
oscillating one-period window *cannot* inherit the ASR at the kernel level, and no
per-configuration projection can fix it without modifying the kernel (the observed
rank-one bias). The §5.5 `is_uniform` gate was the symptom of this theorem.

**(c) The resolution: doubled support.** Let the windows live on **two periods**
(support 2L, applied to the periodically-continued configuration with the true
non-periodic Bloch phases — i.e. genuine zero-padding, the same move that makes
tensor centering possible). Parameterize each window *exactly* on the constraint
manifold:

```
w(d)     = h(d),            d = 0 … L−1
w(d + L) = c_w − h(d)                       (h ∈ R^L and c_w free)
```

so `W̄(d) = c_w` uniformly, **by construction, not by penalty**. The feasible set
now contains oscillating windows (`h = c_w/2 ± osc`) *and* the plain window
(`h = 1, c_w = 1`: second period zero), so the design can only improve on the tent
kernel. The kernel support grows to `|δ| ≤ 2L−1`; the minimal-image target is
unchanged (supported in the Wigner–Seitz cell), the fit must also zero the kernel
tail — this is the price ("increasing a bit the window").

**(d) What comes for free.** With every slot on the constraint manifold:

1. *(ASR-w), (ASR-v)* hold identically (point (b)). *(ASR-z)* is **not automatic**
   in general — check numerically; if violated, either add its (linear) rows to the
   per-slot LSQ, or restrict the z slot to plain-window class sums (for z uniform
   over one period, `Σ_b Σ_{s∈period} w·v` telescopes to the full cross-correlation
   `C_wv(Δ)`, which is manifestly d-independent).
2. *Commensurate exactness* reduces to the single normalization row (point (a)).
3. *Per-configuration acoustic zeros, with zero statistical noise.* At q̃ = 0 the
   effective transform weight of the doubled window on the L-periodic data is the
   class sum: `Σ_{s∈2L} w(s) x(s mod L) = Σ_{s∈L} W̄(s) x(s) = c_w Σ_s x(s) = 0`
   **exactly per configuration** by Newton's third law / zero COM — the same
   mechanism as the unwindowed coarse estimator, restored. Near Γ the leakage
   vanishes as O(q̃) with a finite (noisy but tamed) slope. **No field projection is
   needed at all** — the §5.5 machinery becomes a no-op for this design.
4. *Symmetrization* is structural (w↔v averaging in `S_sym`, Hermiticity by the
   orientation-averaged estimator) and commutes with everything above because it is
   part of the parameterization, not a post-projection.

**(e) Implementation notes.** On the L-periodic data domain a 2L-support window is
a **q̃-dependent complex effective weight** per atom and dimension:

```
ŵ_q(s) = w(s) + w(s+L) e^{−2πi q_frac L}        (per dimension, s = 0 … L−1)
```

(the second period contributes the same data with an extra Bloch phase across the
supercell). `_build_field_set` therefore needs per-q complex weights
`(n_q, nat_sc)` instead of one real vector; the NUDFT already loops over q. Field
dedup keys on the 1D window tuples. At commensurate q the phase is 1 and ŵ reduces
to the class sum — plain behavior, V0 exactness untouched. D4 stays on the plain
pass (plain = feasible point of the manifold; its ASR is already exact).

**(f) Open questions and numerical findings (kept updated).**

- *(ASR-z) automatic?* **YES** — settled numerically: every constrained fit
  satisfies the diagonal image-sum constancy at 1e-15 with no extra rows. All
  three leg sum rules are structural on the uniform-class-sum manifold.
- *Fit quality.* The constrained ALS hits a hard floor **independent of K**
  (L=3: RMS 0.147; L=4: 0.26) because **the minimal-image target itself violates
  the ASR constraints** — the kernel-space restatement of "centering destroys the
  sum rule" (§5.5), i.e. the same reason `Apply_ASR` must modify the centered
  tensor. The correct target is the **projection of L·M onto the ASR+partition
  affine subspace**, computed once in closed form (KKT / lstsq on the constraint
  Gram matrix). Measured projection distances (RMS per grid point):
  L=3: 0.144 (S=2L), 0.072 (S=3L), 0.045 (S=4L); L=4: 0.248 / 0.127 / 0.081;
  L=6: 0.374 / 0.193 / 0.123. The windows **realize the projection almost
  exactly** (0.1475 vs 0.1440 at L=3, S=2L; K=2 already suffices): the
  multilinear window parameterization costs essentially nothing. For scale: the
  plain (tent) kernel has RMS 0.315 at L=3 with exact ASR — the constrained
  design halves the centering error at equal (exact) ASR, and support ×3 halves
  it again. The residual is the *irreducible price of exact ASR at compact
  support*, directly analogous to the modification `Apply_ASR` imposes on a
  centered tensor.
- *Weighted projection (optional refinement).* The unweighted projection spreads
  the ASR correction uniformly over the difference grid; weighting the fit
  residual by an estimate of the tensor decay (e.g. ρ^spread) concentrates
  kernel fidelity where Φ⁽³⁾ is large — the exact analogue of the `power`
  parameter of `Tensor3.Apply_ASR`. ASR and partition stay exact (structural /
  hard-constrained); only the *distribution* of the irreducible residual moves.
- *Where the leak actually bites (deterministic, settled):* contracting the exact
  kernels with an ASR-satisfying three-body test tensor shows the minimal-image
  leak **vanishes when both pair legs go to Γ together** (q_pert = 0: phases
  cancel; also at every commensurate q̃1) but is **O(1) at fixed finite q̃1**
  (|T| up to 8 on a tensor of scale 6) — the dangerous channel is a finite-q
  phonon emitting a near-Γ acoustic phonon (two-phonon continuum edge). The
  plain kernel is exactly zero everywhere; the asr design is 1e-15 everywhere.
  Consequence: probes/tests must put q_pert at finite q (zone boundary), not Γ.
- *Stochastic confirmation (zb probe, Lc=3→Lf=48, N=4000, g3=0.02, g3b=0.4,
  q_pert=1/2):* acoustic-leg vertex row at q̃2=1/48: plain 0.94e-6 (decays ∝ q̃2),
  minimal-image 7.2e-6 (plateau, 7.6× plain), asr 1.6e-6 (decays, tracks plain).
  At the commensurate probe q̃2=1/3 all designs agree with plain to 1e-22 (field-
  level commensurate collapse of the asr design verified).
- *Response-level, SETTLED — two independent error channels:*
  (1) **Lc=3→Lf=6 with g3b=0.08 (spread-2 tensor exactly ON the L=3
  Wigner–Seitz tie boundary = Nyquist-ambiguous)**: aggregate renorm error —
  noise floor 48.1, plain 50.6 (0 failures), mimg3 369.6 (6 spurious
  non-negative-definite static responses), asr3 313.5 / asr_decay3 288.9
  (8 failures). Under-resolution kills ALL oscillating designs (the tie
  weight redistribution is O(1) with wrong phases in the response); plain's
  tent is benign there. This is a CENTERING-RESOLUTION failure, not an ASR
  one — same locality requirement as ForceTensor centering.
  (2) **Lc=5→Lf=10 control (tensor WS-interior, resolvable)**: near-Γ probes
  (renorm cm⁻¹, direct / plain / mimg / asr): q̃=1/10 ac: −56 / FAIL(g>0) /
  −77 / −92; q̃=1/10 opt: −28 / −35 / −31 / −34; q̃=3/10 ac: −413 / −234 /
  −396 / −407; q̃=3/10 opt: −793 / −341 / −733 / −752. Here PLAIN is the
  broken one (tent bias halves the big renorms and flips a near-soft mode);
  mimg and asr both track the direct reference, asr best on the strongly
  renormalized modes and with exact ASR (mimg's structural vertex leak grows
  in weight as the fine mesh densifies near Γ; at Lf=2Lc it is not yet fatal).
  **Practical guidance:** designed windows (mimg AND asr) require the coarse
  cell to RESOLVE the third-order range (spread < WS radius); under-resolution
  → use plain. When resolvable → use asr (centering ≈ mimg, ASR exact, no
  Γ-channel time bombs).
- **Pairwise potentials are structurally immune to the windowed ASR leak**
  (measured, then proven for L=3): probing the D3 vertex on the acoustic leg at
  q̃ = n/24 (Lc=3, N=4000) showed NO difference plain vs minimal-image — all
  decay ~ O(q̃). Reason: the minimal-image kernel's image-sum non-constancy is
  concentrated on rows crossing the WS boundary (|δ1| = 2 for L=3), and there
  the only nonzero weight, C(2, d2=1) = 3/2 from the (2,1)/(−1,1) tie split,
  multiplies Φ_per(−1, 1) — an entry whose three legs span three DISTINCT
  cells. A two-body (bond) potential only populates entries whose legs sit on
  two sites, so exactly those entries vanish and the leak has nothing to
  contract with. This also explains why the M3 benchmarks never showed ASR
  artifacts. Consequence for testing: the toy needs a genuine THREE-BODY cubic
  term (added: `g3b · s1²·s2` over A-atom triplets spanning 3 cells,
  `_toy_chain.make_ensemble(..., g3b=...)`) to expose the leak. Real crystals
  generically have three-site Φ⁽³⁾ entries, so the leak is real there.
- *Variance cost of the doubled support:* small — pairwise-toy aggregate 0.150
  (asr+3origins) vs 0.122 (minimal-image+3origins) at floor 0.103; part of the
  gap is the 0.028 realization residual, part the extra kernel support. The
  decay-weighted variant (ρ=0.4) measures WORSE on the short-ranged toy (0.196):
  the larger far-range kernel entries cost variance — keep ρ=1 unless the
  physical tensor range genuinely demands the reweighting.

### 5.8 Hybrid tensor-D3 mode: when the window formalism cannot center (L=2, real materials)

**Status: DONE, VALIDATED (2026-07-04), triggered by the SnTe failure.** Interpolating
the SnTe force-field model from the 2×2×2 tutorial ensemble to 4×4×4 leaves the
Γ TO peak at ~52 cm⁻¹ instead of the correct ~38 (direct 4×4×4 ensemble AND the
old `Spectral` d3 bubble agree on ~38). Crucially, the old bubble is built from
the **same 2×2×2 ensemble** (stochastic d3 of `get_free_energy_hessian`,
N=10000) followed by `Tensor3.Center(Far=3) + Apply_ASR` — so the coarse data
contains the physics and the failure is purely our interpolation kernel.

**(a) Three structural gaps of the window kernels w.r.t. `Tensor3.Center`.**

1. *One-period no-go at L=2* (§5.7b): every nonzero cell difference is a WS
   tie, so plain = minimal-image = the only design; each δ=1 class is split
   50/50 between the ±1 images with wrong relative phases at off-grid q̃.
2. *Cell-index vs atomic-basis metric*: the kernel `K₃(δ1, δ2)` acts on CELL
   difference classes; `Center` assigns weight by actual interatomic
   Cartesian distances including the basis offsets τ_b − τ_a. It is exactly
   the basis offset that breaks the L=2 ties (and why `Center` needs
   Far up to 3 on a 2×2×2 fcc supercell: the distance-minimal replica can
   sit outside the first supercell parallelepiped).
3. *Separability vs perimeter criterion*: `Center` minimizes the triplet
   perimeter |r_ab| + |r_ac| + |r_bc| over replica pairs — a coupled,
   non-separable criterion in the (non-orthogonal) supercell fractional
   coordinates. Per-dimension window products cannot represent it; even the
   pair-leg version would need per-atom, multi-period windows.

**(b) The exact reorganization that makes a tensor fix legitimate.** Every D3
term of the q-space Lanczos has one leg pinned at `q_pert` (`weight_R`,
`weight_Rf`, `w1`, `w2` in the §1 table), and the estimator is LINEAR in the
3-field correlations. Its infinite-N expectation is therefore

```
(D3 action at fine q̃)  =  Σ_{coarse data}  (kernel)  ×  (coarse d3 correlations)
```

for ANY windowed scheme — the windows only choose the kernel. Averaging the
per-configuration data first (= the stochastic d3 of the free-energy Hessian)
and interpolating with the centered-tensor kernel of `Spectral` is then an
exact substitution of a better kernel, not an approximation with new inputs.
The D3 channel needs only the N_f mode-space blocks with one leg at q_pert:

```
D3̃[ν, ν1, ν2](q̃1, q̃2) = N_f^{-1/2} Σ_{abc} e_ν^a(q_pert) conj(e_{ν1}^b(q̃1)) conj(e_{ν2}^c(q̃2))
                          (m_a m_b m_c)^{-1/2}  Φ̂³_{abc}(−q̃1, −q̃2)
```

(leg a at cell 0 carries the implied momentum q̃1 + q̃2 = q_pert; legs b, c
pair with the displacement fields x(q̃1), x(q̃2), whence the conjugated
polarization vectors — verified by the Wick derivation of the estimator
expectation, see the report appendix.)

with `Φ̂³(q2, q3) = Σ_{R2 R3} Φ³(0, R2, R3) e^{+2πi (q2·R2 + q3·R3)}`
= `Tensor3.Interpolate(−q2, −q3)` (cellconstructor phase convention
`e^{−2πi q·r}`), i.e. `D3̃` uses `Tensor3.Interpolate(q̃1, q̃2)`.
Storage O(N_f · nb³) — no large tensor ever lives on the fine mesh.

**(c) Deterministic replacement of the stochastic D3 terms** (derived from the
Gaussian IBP expectation of the fused kernel, conventions of `vector_r2q`
(phase e^{−2πi q·R}, 1/√N_c), TRI gauge e(−q) = conj(e(q)), force Taylor
f = −½ Φ³ u u; the three Wick orderings weight_R (2 orientations) + weight_Rf
(1) sum to the full contraction, likewise w1 (1) + w2 (2)):

```
d2v(q̃1, q̃2)_{ν1 ν2}  =  + Σ_ν  R1[ν] · D3̃[ν, ν1, ν2]
f_pert[ν]            =  + ½ Σ_{ordered pairs (q̃1 q̃2)} Σ_{ν1 ν2}
                          conj(D3̃[ν, ν1, ν2]) f_ψ(q̃1 ν1) f_ψ(q̃2 ν2) α1_{ν1 ν2}
```

("ordered pairs" = unique pairs with off-diagonal counted twice via the
transpose-symmetric α1 blocks, matching the factor-2 of `total_sum`). The two
maps are mutual adjoints by construction (same D3̃), so L stays Hermitian
exactly; there is no n_syms average (the centered tensor is already
symmetric) and no scale3 (the N_f^{-1/2} normalization is direct, replacing
N_c^{-1/2}·sqrt(N_c/N_f)). The fine-side f_ψ appear explicitly — the tensor
path bypasses the prefilter folding (`_fold_alpha1`), which remains for the
stochastic D4 pass. Mode-validity masks are applied to the blocks. D4 stays
on the stochastic plain-window pass with scale4 (shorter range, smaller
weight, §5.4); ignore_v3 keeps the existing semantics (R1 zeroed upstream ⇒
d2v_D3 = 0, f_pert-from-α1 still active, as in the stochastic path).

**(d) API.**

```
lanczos = QSpaceLanczosInterp(ens, fine_mesh=(4,4,4),
                              d3_mode="tensor", d3_tensor=t3)   # centered CC Tensor3
```

`d3_tensor` must be a `cellconstructor.ForceTensor.Tensor3` already
`Center()`ed (+ `Apply_ASR()`), in Ry/Bohr³ consistent with the dyn — the
exact object the Spectral bubble consumes; typically from the stochastic d3
of `get_free_energy_hessian(..., verbose=True)` on the same ensemble.
`window_design` is forced to "plain" in this mode (windows only ever served
the D3 channel). Blocks are (re)built at `build_q_pair_map` time (they
depend on q_pert), cost N_f · nb³ contractions via one zgemm chain.

**(e) Scope note.** This mode reintroduces a stored Φ³ (size N_c²·(3nat)³ —
the same object Spectral already requires, affordable whenever the coarse
ensemble is), trading requirement 1 for correctness whenever the coarse
supercell cannot resolve the D3 range in the window sense (WS-resolution
rule, §5.7f) or the L=2/atomic-basis/perimeter gaps of point (a) apply.
The tensor-free windowed schemes remain the default for L ≥ 3 resolvable
cases and for D4 always.

**(f) Validation — ALL PASSED (2026-07-04).**

- *Toy chain, operator level* (`tests/test_interpolation/test_d3_tensor_mode.py`,
  5 tests green): deterministic d2v+f_pert vs the stochastic estimator on the
  identity mesh at non-TRI q_pert agree with pure-noise residuals — relative
  L2 0.134 / 0.101 / 0.047 at N = 2k/8k/32k (1/√N), LSQ scalar fit → 0.999.
  Hermiticity |b−c| < 1e-8 on a genuinely interpolated mesh (the two maps are
  exact mutual adjoints by construction). Physics interp 3→6 with the exact
  Φ³ tensor: aggregate renorm error vs an independent direct fine ensemble
  at the stochastic noise floor (< 0.20; plain-window error is ~0.30).
- *SnTe (the trigger case, D3-only, no LO-TO; scripts in the issue dir):*
  identity 2×2×2 static = **21.28 cm⁻¹ = the free-energy-Hessian static
  reference exactly** (the static Lanczos with tensor d3 IS the Hessian
  bubble); interp 2×2×2→4×4×4 static g > 0 exactly like the direct 4×4×4
  (the plain window had produced a spurious stable 50.3); interp
  2×2×2→4×4×4 **dynamic TO peak 37.6 cm⁻¹ = direct 4×4×4** (bubble 38.5,
  broken plain interp 52.3), normalized-lineshape L1(interp, direct) =
  0.054 against a direct-vs-bubble method baseline of 0.322.

### 5.9 Tensor-free "atomic" windows: stochastic centering that works at L=2

**Status: DONE, VALIDATED on SnTe (2026-07-04).** Requirement from Lorenzo:
the production scheme must be purely stochastic — computing/storing Φ³ is too
heavy for large supercells — and must reach Center(Far)-quality interpolation
*even on the SnTe 2×2×2 case*; the §5.8 tensor mode remains as the validated
reference/oracle. The centering recipe may differ from ForceTensor's
(perimeter) as long as the quality matches.

**(a) What the §5.8 analysis says a working window scheme MUST have.**

1. *Atom resolution*: the kernel must weight images per interatomic offset
   τ_b − τ_a, not per cell-difference class (this is what breaks the L=2
   ties). ⇒ per-atom windows w_b(s): the NUDFT weight becomes a
   (n_q, nat_sc) array — infrastructure ALREADY EXISTS (`_build_field_set`
   accepts per-q complex per-atom weights, used by the "asr" design).
2. *Multi-period support*: distinct images of a class must get distinct
   weights ⇒ support ≥ 2L per dimension (the §5.7 doubled-support machinery,
   with the e^{−2πi q·L a} second-period Bloch phases).
3. *No per-dimension separability assumption*: for non-orthogonal (fcc)
   supercells the distance-minimal image is not separable in fractional
   coordinates ⇒ design directly in 3D (window = arbitrary function of the
   cell index vector + atom, fitted as a whole; the small L makes this
   affordable: nat·(2L)³ values per slot per pass).

**(b) Candidate kernel targets (geometry-only — the design must never see
the tensor, else it is cheating).**

- *Leg-wise atomic minimal-image ("atomic")*: image of pair leg b at cell
  difference δ⃗1 assigned by minimizing |τ_b − τ_a + (δ⃗1 + n⃗L)·cell| over
  replicas n⃗ (ties split), independently for the two pair legs, both
  relative to the z-slot atom a. Product structure over legs ⇒ compatible
  with the shared-s window kernel K^{abc}(δ1,δ2) = Σ_s z_a(s) w_b(s+δ1)
  v_c(s+δ2) (a CP decomposition in a ⊗ (b,δ1) ⊗ (c,δ2); §5.2 measured small
  CP ranks for the cell-index analogue).
- *Perimeter (ForceTensor's criterion)*: jointly minimizes
  |r_ab| + |r_ac| + |r_bc| — NOT leg-factorizable; reachable only via
  higher-rank CP fits or a pinned-z-slot estimator variant (kernel
  w_b(δ1)·v_c(δ2) exact per origin, full origin average costs ×N_c — cheap
  precisely in the small-N_c regime where all of this matters).

**(c) Decision experiment (before building anything).** The expectation of
ANY window scheme is a deterministic contraction of the periodic tensor with
its kernel. Use the SnTe Φ³ as a TEST ORACLE ONLY: contract Φ_per with
(plain | leg-wise atomic | perimeter) kernels, compare the D3 vertex blocks
at the 4×4×4 fine pairs against Tensor3.Interpolate(Far=3). If leg-wise
atomic ≈ perimeter on SnTe ⇒ the shared-s per-atom window design suffices;
otherwise fall back to the pinned-slot variant.

**Result (2026-07-04, `kernel_study.py`):** plain is unusable on the
interpolated Γ-pair blocks (mean relative Frobenius error 1.048, max
1.582); leg-wise atomic minimal images are already very close to the
perimeter reference (mean 0.008, max 0.011); the geometry-only perimeter
target matches `Tensor3.Center(Far=3)` on interpolated pairs at numerical
zero. The diagnostic's commensurate rows show a reference-normalization/gauge
mismatch and are not the off-grid quality metric. Decision: implement the
pinned-z **leg-wise atomic** kernel first, because it is already at the
sub-percent/percent level on the failed SnTe blocks, has exact unit class
sums, and is much cheaper than a high-rank perimeter CP factorization.

**Smoke test (2026-07-04):** `window_design="atomic"` on SnTe
2×2×2→4×4×4 with N=20, Γ TO, D3-only constructs 16 D3 passes
(2 pinned atoms × 8 full origins), 32 distinct field sets and 36 pair
blocks; two Lanczos steps run without construction or normalization errors.

**(d) Implemented estimator.** `window_design="atomic"` uses one pass per
pinned primitive atom `a` and all coarse origins `o`. The z-slot window is
`δ_{atom,a} δ_{cell,o}`; the two pair-leg windows assign each atom/class
`(b,δ)` to the geometry-only minimal images of
`τ_b-τ_a+(δ+nL)A`, splitting exact distance ties. Extended images are folded
back onto the periodic supercell with the correct Bloch phase
`exp[-2πi q·(nL A)]`, using the existing per-q atom-weight path of
`_build_field_set`. The full-origin sum restores the translation sum of the
plain estimator, so no `1/N_c` origin-averaging prefactor is used. D4 stays
plain. API:

```
lanc = QSpaceLanczosInterp(ens, fine_mesh=(4,4,4),
                           window_design="atomic", window_far=3)
```

`window_far` is the geometry replica search range, analogous to
`Tensor3.Center(Far=...)` but never using Φ³.

**(e) SnTe validation.** D3-only, no LO-TO, Γ TO, 2×2×2 ensemble interpolated
to 4×4×4:

- N=300, 20 Lanczos steps: static `g=+7.20e6` (unstable, like direct/tensor),
  dynamic peak 38.2 cm⁻¹; normalized L1 vs direct 4×4×4 = 0.293 (short/noisy),
  vs tensor-D3 = 0.254.
- N=1000, 30 Lanczos steps: static `g=+5.39e6` (unstable), dynamic peak
  **37.7 cm⁻¹**; normalized L1 vs direct 4×4×4 = **0.0865**, vs tensor-D3 =
  **0.0466**. References: direct 4×4×4 peak 37.56, tensor-D3 37.64, old bubble
  38.46, broken plain 52.28 (plain L1 vs direct 1.7507). This matches the
  tensor-FC3 quality while never constructing or storing Φ³.

**(f) Tests.** `tests/test_interpolation/test_atomic_windows.py` checks:
atomic-basis tie breaking at L=2, unit class sums of the atom/class windows,
and the extended-cell Bloch phase used to fold images back to the periodic
supercell.

**(g) Expected side benefit.** Atomic-basis ties are exactly the §5.7(f)
error channel 1 (tensor spread AT the WS boundary kills all cell-index
designs): atom-resolved windows should shift that resolution limit too.

---

## 6. Vertex renormalization: coarse ensemble → fine mesh

This is the one place where a factor must be *inserted* by hand. Mode-space vertices
on an N-cell mesh scale as

```
d₃(q1ν1, q2ν2, q3ν3) = N^{-1/2} δ_{q1+q2+q3,G} · Φ̄₃(...)      (Φ̄₃ intensive)
d₄(...)              = N^{-1}   δ_{Σq,G}       · Φ̄₄(...)
```

The coarse-ensemble estimator delivers the values with `N = N_c`; the fine-mesh
Lanczos (which is exactly the Lanczos of a hypothetical N_f-cell supercell) requires
`N = N_f`. Therefore multiply:

```
scale3 = sqrt(N_c / N_f)   on all D3-type terms:  weight_R, weight_Rf, w1, w2
scale4 =      N_c / N_f    on all D4-type terms:  total_wD4, total_wb
```

(N_c, N_f = numbers of q-points in the coarse and fine uniform meshes.)

### 6.1 Consistency check (bubble self-energy)

Σ(q_pert) ~ Σ_{pairs} |d₃|²·J: coarse = N_c pairs × N_c⁻¹|Φ̄₃|² = intensive; fine with
scale3: N_f pairs × (N_c⁻¹|Φ̄₃|² · N_c/N_f) = intensive and identical in the
exact-interpolation limit. ✓ The two-phonon sum becomes a Riemann sum over the fine
mesh — exactly the desired dense decay channels.

### 6.2 Where it enters the code

Two `Float64` arguments added to `get_perturb_averages_qspace`
(`Modules/tdscha_qspace.jl:650`), multiplying the six weights listed in §1's table.
`scale3 = scale4 = 1` reproduces today's behavior bit-for-bit.

### 6.3 Caveat: SSCHA stationarity

The uniform per-vertex-order scaling is exact **only** because the disconnected Wick
pieces of the 4-field averages vanish at SSCHA self-consistency (they are
proportional to ⟨u ⊗ δf⟩ ∝ Υ⁻¹(⟨∂²V⟩ − Φ_SSCHA) = 0). For a not-fully-converged
final dyn or a heavily reweighted ensemble, a residual O(⟨∂²V⟩−Φ) piece gets scaled
by N_c/N_f instead of 1. This is benign (the factor *suppresses* the spurious piece)
but must be documented: **use the converged SSCHA dyn via `update_weights` before
interpolating** (already the recommended workflow, cf. `load_distributed_tdscha`).

---

## 7. Harmonic sector on the fine mesh

1. **ω(q̃), e(q̃):** Fourier-interpolate the SSCHA dynamical matrix (2nd-order tensor —
   allowed and cheap) with standard WS centering; use
   `CC.Phonons.Interpolate` / `ForceTensor.Tensor2` for the centering quality, with
   the **nonanalytic dipole term from effective charges** (LO-TO) handled as usual.
   On-grid q̃ reproduce the coarse ω, e exactly ⇒ back-compatibility of `apply_L1_FT`.
2. **Time-reversal gauge:** enforce `e(−q̃) = conj(e(q̃))` constructively (diagonalize
   on an irreducible half-mesh, generate the other half by conjugation). This is what
   keeps the pair blocks complex-symmetric (the convention that was the source of the
   fixed non-TRI bug — see memory note on the conjugation convention in
   tdscha_qspace.jl; a regression test at non-TRI q̃ is mandatory).
3. **Gauge freedom in degenerate subspaces** at each q̃ is harmless: every quantity
   contracts e(q̃) consistently within the same run (the estimator, the psi blocks and
   the symmetry matrices are all built from the same set), so L is gauge-covariant.
4. **Acoustic / small-ω handling:** `valid_modes_q` generalizes: mask exact
   translations at Γ only; apply a `w_min` guard for interpolation artifacts (fine q̃
   near Γ have genuinely small ω with huge `f_psi = (1+2n)/2ω` — physical, but
   interpolation noise there must not produce negative/NaN frequencies; abort with a
   clear error if the interpolated dyn has ω² < 0 off Γ, as the theory requires a
   positive-definite SSCHA dyn).

---

## 8. Symmetries on the fine mesh

`prepare_symmetrization` / `_build_qspace_symmetries`
(`Modules/QSpaceLanczos.py:1447,1527`) generalize almost verbatim:

- A uniform Γ-centered fine mesh is closed under the point group and under
  `q ↦ q_pert − q` when `q_pert` is on the mesh; the same sparse block construction
  `D(iq′←iq) = e†(Sq) P_uc(q′) e(q)` applies, with nnz = O(N_f · n_bands²) — linear.
- Replace the O(n_q²) linear searches (`find_q_index` loops in
  `build_q_pair_map` and `_build_qspace_symmetries`) with **O(1) hash lookups** on
  rounded fractional mesh indices; otherwise setup becomes the bottleneck at large
  N_f.
- **Windows vs point group:** applying S to a windowed configuration is equivalent to
  using the S-transformed window (rotated axes assignment, mapped origin). If the
  per-dimension window designs are constrained equal on symmetry-equivalent axes
  (§5.2), the design family is closed under the PG and the symmetrized kernel remains
  the target kernel. Random origins are per (config, pass) and frozen; symmetry
  averaging then simply enlarges the origin sampling — no bias.
- **Off-mesh `q_pert` (spectral functions along a path):** allowed by the same
  machinery with q-list = {q_pert} ∪ mesh ∪ (q_pert − mesh) and symmetries restricted
  to the little co-group of q_pert (the extended list is closed only under it). This
  is a headline capability: σ(q, ω) maps at arbitrary q from one coarse ensemble.
  Phase 2 of the implementation.

---

## 9. Implementation plan

### 9.1 Python: `QSpaceLanczosInterp(QSpaceLanczos)` (new module or flag)

```
lanczos = QSpaceLanczosInterp(ensemble, fine_mesh=(8,8,8),
                              window_design="minimal_image",  # or "plain"
                              n_window_passes=2)              # per dimension
```

Setup steps (all linear in N_f):

1. Build fine q-list (hash-indexed); require q_pert ∈ fine mesh (phase 1).
2. Interpolate SSCHA dyn → `w_q[nb, N_f]`, `pols_q[3nat, nb, N_f]` with TRI gauge and
   LO-TO (§7). Reuse `self.dyn.Interpolate` + a small gauge-fixing helper.
3. Load per-dimension window designs (precomputed table per L; fall back to on-the-fly
   ALS fit, §5.2, with the ASR projection folded in, §5.5); draw and freeze random
   origins.
4. **ASR zero-mode projection** per (config, pass, origin): subtract the
   window-weighted uniform translation from `u` (mass metric) and from `δf` (plain
   sum), §5.5. No-op for the plain full-period window.
5. **Windowed NUDFT:** for each pass r, compute
   `X_q_r[N_f, N, nb]`, `Y_q_r[N_f, N, nb]`. Implementation: one zgemm
   `(N_f × N_c phase-window matrix) · (N_c × N·3nat_uc data)` + per-q̃ zgemm band
   projection. Cost O(K³·N·N_c·N_f·nb) once; memory O(K³·N_f·N·nb·16B) per array —
   chunk over q̃ and/or reuse the existing MPI config distribution
   (`load_distributed_tdscha` already slices over configs, which divides this memory
   by n_ranks).
6. Pair map / block layout / mask: index arithmetic on mesh indices, O(N_f).
7. Fine-mesh symmetry sparse matrices (§8).

Runtime: `apply_anharmonic_FT` loops over passes, calls the Julia kernel per pass
with that pass's field set and (scale3·c_r, scale4·c_r'), and sums. Everything else
(`run_FT`, mask, continued fraction, save/restart) is untouched.

### 9.2 Julia kernel changes (`tdscha_qspace.jl`)

1. Add `scale3::Float64, scale4::Float64` to `get_perturb_averages_qspace(_fused)`;
   multiply the six weights (§6.2). Backward compatible defaults = 1.
2. Accept the field arrays per pass (no structural change — the kernel is already
   agnostic to how X_q/Y_q were produced and to n_q).
3. **BLAS-3 batching (the main speed lever):** today's inner loops are scalar rank-1
   updates. Restructure: over a chunk of (config, sym) of size B,
   - d2v accumulation per pair = 3 zgemms `(B × nb)ᴴ(B × nb)` (one per outer-product
     type, weights folded into the left factors);
   - `buffer_u` = batched zgemm `alpha1[p] · conj(Xchunk)`;
   - weights = batched dot products (zgemv).
   Expected ≳10× over the scalar loops; keeps memory bounded by B.
4. Optional later: GPU port of the batched form (embarrassingly parallel over pairs).

### 9.3 Cost & scaling summary

Per Lanczos iteration: `O(K³ · N · n_syms · N_f · nb²)` flops (BLAS-3), i.e. **linear
in N_f** with constant factor K³ (=8 for the recommended design). Memory:
`O(K³ · N_f · N_local · nb)`. Setup: one-time `O(K³ · N · N_c · N_f · nb)` zgemm.
Example (2-atom cell, nb=6, N=1000 configs, n_syms=48, fine 16³=4096, K³=8):
~5·10¹³ flop per L-application ⇒ seconds-to-minutes on one node with the batched
kernel + MPI; plain-window quick mode (K³=1) is 8× cheaper.

---

## 10. Validation plan

- **V0 (exactness on-grid):** fine mesh = coarse mesh, plain window ⇒ a/b/c
  coefficients must match `QSpaceLanczos` to machine precision (same code path test).
  Then fine mesh = coarse mesh with the designed windows ⇒ must still match in the
  infinite-sampling limit (partition of unity); verify on a large synthetic ensemble.
- **V1 (kernel-level unit tests):** assemble K₃/K₄ numerically from the windowed
  estimator on random synthetic data and compare against the design targets;
  test partition of unity and the commensurate limit per dimension.
- **V2 (physics, the decisive test):** toy anharmonic model (cheap forces; reuse
  the test infrastructure in `tests/test_qspace/`): generate ensembles on 2×2×2 *and*
  4×4×4; compare interp(2×2×2 → 4×4×4) against direct 4×4×4 at matching q_pert. Judge
  agreement on the **anharmonic renormalization (Lanczos − SSCHA), not absolute
  frequencies** (established error metric for these comparisons). Repeat at a non-TRI
  q̃ (regression for the conjugation convention).
- **V3 (vs Spectral.py):** static/bubble limit against the d3-interpolated dynamical
  bubble from `cellconstructor.Spectral` on the same ensemble — direct check of the
  stochastic centering vs tensor centering.
- **V4 (Hermiticity/positivity):** |b−c| per iteration; spectral function
  non-negativity within noise (negative design coefficients c_r can in principle
  produce tiny negative weight of order the interpolation error — monitor).
- **V5 (scaling):** time and memory per L-application vs N_f = 4³…16³ (assert linear).
- **V6 (ASR):**
  - unit test: a synthetic "configuration" equal to a rigid translation (u = const,
    δf = 0 and vice versa a constant force field) must give exactly zero projected
    windowed fields for every pass;
  - runtime diagnostic: magnitude of the d2v blocks on acoustic branches vs |q̃| must
    extrapolate to zero linearly as q̃ → 0 — a plateau signals an ASR leak (emit a
    warning);
  - physics test: acoustic linewidths vanish with the correct power as q̃ → 0 and the
    spectral function near Γ is free of 1/ω artifacts; repeat with the projection
    deliberately disabled to demonstrate the failure mode (documentation of why the
    projection is mandatory).

---

## 11. Error budget and limitations

1. **Locality (systematic):** all schemes assume D3/D4 correlations decay within half
   the coarse supercell — identical to the assumption behind tensor interpolation.
   Designed windows reach the same asymptotic accuracy class as `ForceTensor`
   centering; the plain window degrades to O(range/L) between grid points.
2. **Long-range (polar) D3 tails:** dipole-induced long-range parts of Φ⁽³⁾ are *not*
   separated analytically (Spectral.py has the same limitation for d3; only the
   D2/LO-TO part is treated via effective charges). Out of scope; note for future work
   (subtract a model long-range vertex from y before windowing, add back analytically).
3. **Statistics:** the fine-mesh operator contains no new microscopic information —
   it is a smooth (deterministic) function of the coarse data. Noise at nearby fine q̃
   is strongly correlated over the coarse-grid spacing ⇒ spectra are smooth, but the
   genuine q-resolution of the *anharmonic* content is set by the coarse supercell
   (the *harmonic* dispersion and two-phonon DOS are exact at the interpolated-dyn
   level — this is where the physical gain comes from). Windowing slightly increases
   per-pass variance (data weighting); the K³ passes and symmetry augmentation
   compensate.
4. **ω-covariance mismatch:** `f_Y, f_psi` use interpolated-dyn ω(q̃) while the field
   covariance is the interpolated correlation — two interpolations of consistent
   on-grid data; mismatch is first order in the interpolation error, same order as
   item 1.
5. **Stationarity assumption** for the D4 rescaling (§6.3).
6. **Frozen randomness:** origins (and any pass subsampling) must be drawn once at
   init — L must be one fixed linear operator across all Lanczos iterations.
7. **ASR enforcement moves weight near Γ:** the zero-mode projection (§5.5) perturbs
   the fields within a Γ-neighborhood of width ~2π/(window size), i.e. the acoustic
   vertex at small finite q̃ is regularized at the price of an O(interpolation
   resolution) redistribution — the same trade-off as `Apply_ASR`'s correction
   spreading on tensors. Only translational sum rules are enforced (no rotational /
   Born–Huang), matching the scope of `ForceTensor`.

---

## 12. Alternatives considered and rejected

- **Build + center + interpolate Φ⁽³⁾/Φ⁽⁴⁾ (Spectral.py):** excluded by requirement
  (memory O((3nat·N_c)³), and no nonperturbative D4/vertex-mixing in the bubble).
- **Interpolate the self-energy / Lanczos coefficients across q:** cheap, but decay
  channels stay on the coarse two-phonon grid — misses precisely the physics asked for.
- **Ensemble "unfolding" (tiling configs to a larger supercell):** tiled data is
  periodic with the original cell ⇒ zero new Fourier components (this is the "you get
  zero" statement); adding synthetic incommensurate modes requires forces we don't
  have.
- **Pinned-field (delta-window) estimator:** makes the D3 kernel exactly separable
  with one pass, but uses one lattice site per config for one slot ⇒ variance ×N_c,
  or cost ×N_c to average origins. The low-rank window design (§5.2) achieves the
  same kernel at ×8.

---

## 13. Milestones

1. **M1 — kernel design toolbox (pure Python/NumPy, no Julia):** per-dimension window
   fitter (ALS + partition constraints, evaluated on ASR-projected kernels, §5.5),
   kernel visualization, unit tests (V1, V6 unit part).
   *Deliverable: designs for L = 2…8 stored as data files.*
2. **M2 — plain-window prototype:** fine-mesh setup (interpolated dyn with D2 ASR,
   TRI gauge, hash maps, pair map), NUDFT fields, `scale3/scale4` in Julia, no
   windows (K=1). Validate V0 + V2-lite (coarse-commensurate fine mesh) + the V6
   acoustic-vertex diagnostic (the plain window already needs the interpolated-dyn
   ASR and the small-ω guards).
3. **M3 — designed windows:** pass loop, slot assignment, ASR zero-mode projection,
   Hermiticity checks. Full V2, V3, V6 validation.
4. **M4 — performance:** BLAS-3 batched Julia kernel, MPI-distributed fields, V5
   scaling benchmarks.
5. **M5 — extensions:** off-mesh q_pert along paths (little-group symmetrization),
   optional D4 window design, KPM variant (`QSpaceKPM` uses the same estimator —
   the substitution carries over unchanged).
