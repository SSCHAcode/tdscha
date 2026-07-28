# Trilinear q-space interpolation for the TDSCHA Lanczos — implementation plan

**Status: IMPLEMENTED; SOURCE-MESH PHYSICS AUDIT IN PROGRESS** (`Modules/QSpaceTrilinear.py`, class
`QSpaceTrilinearLanczos`; tests in `tests/test_trilinear/`). Fourier-Upsilon
variants were tested, found uniformly worse on SnTe, and removed. The class
uses only the paper's coarse corner-Upsilon prescription. The optional
atom-position Bloch gauge remains as a diagnostic. Retained paths pass
machine-precision identity, exact fold transpose symmetry
`A(Q−k) = A(k)ᵀ`, and full-operator Hermiticity at TRI and non-TRI Q.

**M4–M6 SnTe benchmark provenance correction (2026-07-17; full record in
`benchmark.md`).** Γ TO, D3-only, T=280: the interpolation identities and
Hermiticity tests remain valid, and the Upsilon variants were compared on a
matched control. However, the saved 4³ auxiliary matrix used in the physics
comparison was itself interpolated from the converged 2³ SSCHA matrix; its
ensemble was not an independent converged 4³ SSCHA ensemble. The curve called
“full 8³” likewise used the 2³ auxiliary matrix and 2³ FC3 on an 8³
integration grid. Therefore the previously quoted 4³→8³ and full-8³ peaks
are withdrawn as evidence for source-mesh convergence. A fresh 4³ SSCHA
relaxation with N=4000 configurations per population is converged: its Γ
auxiliary TO mode is 47.463 cm⁻¹, and its native D3-only TDSCHA Lanczos
spectrum peaks at **34.79 cm⁻¹** (`ignore_v4=True`). The existing centered
2³→4³ oracle peaks at 37.65 cm⁻¹, whereas trilinear peaks at 46.11 cm⁻¹ and
the original 2³ result at 44.03 cm⁻¹. Thus the trilinear hardening is real,
not an artifact of the former reference, while atom-resolved centering is
much closer. The 8³ audit is deferred until an independently converged 8³
reference is available. Trilinear-in-q ≡ *cell-basis*
Bartlett centering: increasing L resolves progressively more real-space
range, but at small L it cannot represent SnTe's atomic-basis Far=3 FC3
assignment. The toy chain
(bond-ranged vertex) stays good at 16% aggregate. Consequence: for
SnTe-like materials at these meshes this class still needs the atomic-basis
D3 fix (hybrid tensor-D3 channel + trilinear D4).

**Cause of the apparent wrong-sign mesh correction.** The D3 sign and
supercell normalization are correct: on the 2³→4³ audit the coarse map
matches the native map times `sqrt(8/64)` to 1.83e-14, and the fitted
commensurate vertex amplitude is +0.9979. The failure is at the new q
points. After matching the commensurate amplitude, off-grid vertex blocks
differ from centered FC3 by 113%, while their thermally weighted squared
norm is only 29.2% of the centered oracle (35.45% over all pairs).
Trilinear interpolation is a non-unitary prolongation: it averages complex
vertex amplitudes before the bubble forms `|V3|²`, strongly attenuating
short-wavelength/atom-image components. Hence D3 still softens the mode,
but only 53.985→46.11 cm⁻¹ instead of the native-4³
47.463→34.79 cm⁻¹. The interpolation-induced change relative to native 2³
is consequently an apparent hardening (44.03→46.11 cm⁻¹). Because the
atomic-position phase is unitary, it cannot restore the lost singular
values. A scalar norm correction would not repair the 113% off-grid shape
error; atom-triplet centering or centered-Tensor3 Fourier padding is needed.

The proposed Fourier-Upsilon change did **not** repair this. On the identical seeded
2×2×2→4×4×4 run, directly Fourier-interpolating the Cartesian Upsilon
matrix moves the peak from 46.11 to 49.88 cm⁻¹; constructing Upsilon from
the Fourier-interpolated harmonic dynamical matrix gives 51.25 cm⁻¹. A
full-Bloch atomic phase `exp(−2π i q·tau_a)` on the complete rank-2
fold/unfold map changes the original result only to 45.98 cm⁻¹, while the
phase plus direct Fourier Upsilon gives 49.94 cm⁻¹. Thus neither hypothesis
approaches the converged 4³ TDSCHA reference at 34.79 cm⁻¹. The former
37.56 cm⁻¹ value is only the matched control around an auxiliary matrix
interpolated from 2³. The Fourier-Upsilon code was therefore removed. The
phase is a gauge transport; it cannot reconstruct the
atom-resolved, long-ranged FC3 information missing from cell-corner
interpolation.

**Implementation finding (deviation from §3.4-3.5 as first drafted).**
The parent's `get_alpha1_beta1_wigner_q` folds the vertex filter
`w1·w2/X = 2·f_Y(q1)·f_Y(q2)` into the input blocks *at the pair's own
q-points*. Doing that at the FINE pairs breaks the exact per-configuration
Hermiticity of L (the kernel's output dyadics carry `f_Y` at the coarse
corners; commensurately pair == corner and the element is symmetric, ~1e-3
asymmetry once interpolated). The fix — which is exactly the paper's
`Υ(k_ε)` prescription in Eq. (w:2:q:fine): coarse Υ on the interpolated
Υu legs — is to fold the *bare* χ-dressed blocks (`_get_alpha1_bare`,
factor 2 in place of `w1w2/X`) and apply the **coarse** `f_Y` filter on
both legs of the folded kernel (`_fold_alpha_to_coarse`). In the
commensurate limit `f_Y(k1) f_Y(k2) · bare == (w1w2/X) · dressed`
identically. Verified: parent commensurate L is Hermitian to 0.0 on random
vectors; the trilinear L matches after the fix. The atomic phases are paired
with their inverse-conjugate phases on the return path, so that optional
gauge also remains Hermitian.

Source of the scheme: Overleaf project *"Interpolation of high-rank phonon
interactions within the TD-SCHA"* (`6a4ba34cdbe5288bb39378e3`, `main.tex`).
This is a **new interpolation strategy**, independent from the windowed/NUDFT
scheme of `Interpolation_plan.md` / `QSpaceInterpolation.QSpaceLanczosInterp`.
It will live in a new module with a new class inheriting from
`QSpaceLanczos`, and all q-mixing operations are performed in **primitive-cell
Cartesian coordinates** (never interpolating across q in the polarization
basis).

---

## 1. Equation audit (paper vs. validated code conventions)

The commensurate q-space kernel (`Modules/tdscha_qspace.jl` +
`Modules/QSpaceLanczos.py`) is validated against the real-space TDSCHA and is
the reference for every convention below. Findings (all annotated in red on
the Overleaf):

| # | Equation | Finding |
|---|----------|---------|
| 1 | `f_a^(i)` definition | Sign error: the residual is `f = f_BO − f_SCHA = −dV/dR_a **+** Σ_b Φ_ab u_b` (code: `Y = forces − sscha_forces`, `f_SCHA = −Φu`). |
| 2 | Eq. (w:2) | Index typo (`∂lnρ/∂Υ_hk` → `∂Υ_ab`). Sign: with `w = δlnρ` there is **no** overall minus: `w(Υ¹) = −½ Σ Υ¹_ab (u_a u_b − Ψ_ab)`. |
| 3 | (w:2) vs (phi3:rav) vs (phi4:avg) | Internal inconsistency: the minus in (w:2) matches the `+` of (phi4:avg)'s second term but **contradicts** the `−` of (phi3:rav). Fix set: drop the minus in (w:2), flip `+`→`−` in (phi4:avg). All averages also need `1/Σρ_i`. |
| 4 | Eq. (phi3:q) | Missing minus (`Φ³ ∝ −⟨[Υu][Υu]f⟩`), missing `1/Σρ_i`; rank-2/3 Fourier normalization convention should be stated (it fixes the `1/N_q` in (w:2:q)). |
| 5 | Eq. (w:2:q) | (i) LHS `w(Ψ¹)` vs RHS `Υ¹` symbol mismatch; (ii) missing `−½` (q′ runs over the full BZ ⇒ each unordered pair counted twice); (iii) **the u fields must be conjugated**: `conj(u(q′)) conj(u(q−q′))`. As written the weight carries momentum `+q` and couples to `f(−q)`, i.e. (force:q) computes the `−q` component while labeled `+q`. The conjugated (bilinear) convention is what the code uses; the wrong choice silently kills the anharmonicity at **non-TRI q** (this exact bug already happened once in `tdscha_qspace.jl`). |
| 6 | Eq. (w:1:q) | `𝓡¹(−q)` → `𝓡¹(+q)`: the operator must be complex-**linear** in the stored perturbation (code: `weight_Rf = Σ R1[ν]·conj(y_pert[ν])`). Equilibrium fields correctly conjugated. |
| 7 | Eq. (w:2:q:fine) | Prefactor: must be `1/N_f = (1/N_q)·(N_q/N_f)`, not `N_q/N_f` alone. Limit check: `N_f = N_q` must reduce to (w:2:q) ⇒ `1/N_f` ✓, `N_q/N_f → 1` ✗. (The prose is right; the typeset formula dropped the `1/N_q`.) |
| 8 | Eq. (d2vdr2:q) | This is a **pointwise** output at the fine pair (no q′ sum) ⇒ carries **no** mesh factor. As typeset, the `R¹ → 2ph → R¹` round trip gets `(N_q/N_f)²` instead of the physical `N_q/N_f`. Hermitian-symmetric split used in the implementation: `√(N_q/N_f)` on each D3 channel, `N_q/N_f` on D4 — exactly the existing `scale3`/`scale4` of `tdscha_qspace.jl`. |
| 9 | rank-3 interp eq | Duplicate label `eq:rank2:interp`. Also noted: single-leg interpolation automatically preserves q₁↔q₂ permutation symmetry (corner-mirror property, §2.4); the permutation with the Q leg is enforced by the estimator's force-channel symmetrization. |

**Fixed convention set for the implementation** (identical to the working code):

- FT: `v(q) = (1/√N_q) Σ_R v(R) e^{−2πi q·R}` — matches `sscha`'s
  `vector_r2q!` exactly (paper Eq. (q) is correct).
- Bilinear pair blocks at `q₁+q₂ = Q+G`; kernel contractions use
  `conj(x(q))`; dyadic outputs use unconjugated components; reverse pair
  block = transpose (no conjugation).
- Weights: `w(𝓡¹) = Σ R¹_a(Q)·conj([Υu]_a(Q))`,
  `w(Υ¹) = −½ Σ_{q′∈BZ} α(q′,Q−q′)·conj(x(q′))conj(x(Q−q′))` (the −½ and
  channel prefactors are already inside the Julia kernel).
- **Dyn-matrix Fourier convention (the user's warning):** CellConstructor
  builds `D(q) = (1/N_q) Σ FC(R_i,R_j) e^{−2πi q·(R_i−R_j)}` — *opposite
  sign* to the usual `+iqR` convention, and `ForceTensor.Tensor2.Interpolate`
  uses yet another sign, so the fine dyn must be obtained as
  `t2.Interpolate(−q)` (already handled + regression-tested in
  `QSpaceInterpolation.interpolate_dyn_fine`, which we reuse).

---

## 2. The scheme

### 2.1 Setup

- Perturbation momentum **Q constrained to the coarse mesh** (as in the
  paper). Fine mesh `N_f = (m₁N₁, m₂N₂, m₃N₃)` — integer multiple of the
  coarse mesh, Γ-centered, so coarse ⊂ fine (asserted at init).
- ψ layout is the parent's, but the two-phonon sector runs over **fine
  pairs** `(q′, Q−q′)`, `q′ ∈` fine mesh (unique pairs, upper-triangle for
  diagonal): `[R¹(Q) | a′(fine pairs) | b′(fine pairs)]`.
- Harmonic sector: `ω(q′), e(q′)` from **standard Fourier interpolation of
  the SSCHA dynamical matrix** (`interpolate_dyn_fine`: Tensor2 centering +
  ASR + TRI gauge, `Interpolate(−q)` sign fix, coarse points copied — see
  §3.2). Parent's `apply_L1_FT`, `get_chi_*`, `get_alpha1_beta1_wigner_q`,
  `mask_dot_wigner`, `run_FT` all work unchanged on the fine arrays.

### 2.2 The interpolated anharmonic estimator

Per the (corrected) paper: for each fine pair, every per-configuration field
is evaluated at the trilinear **corners** of `q′` on the coarse mesh:

```
w_i(Υ¹)  = −(1/2N_f)·Σ_{q′∈fine BZ} Σ_ε I_ε(q′) α_ab(q′,Q−q′)·conj(g_a(k_ε)) conj(g_b(Q−k_ε))
d2v(q′,Q−q′) = Σ_ε I_ε(q′) · D2V_cart(k_ε, Q−k_ε)          (pointwise; g = Υu, all Cartesian)
```

`k_ε(q′)` are the 8 coarse corners, `I_ε(q′)` the trilinear weights (paper
Eqs. coarse-box/coarse-corner/trilinear-weight — verified correct). Since Q
is coarse, the partner leg `Q − k_ε(q′)` is **also a coarse point**: momentum
conservation holds term by term, and the coarse Υ enters through the coarse
`f_Y` filter (the paper's requirement that Υ be taken on the coarse mesh is
automatic), while the fine Ψ/ω enter only the per-fine-pair χ± factors
("Ψ Fourier-interpolated at fine q" in the paper).

### 2.3 Key algorithmic result: exact folding ⇒ coarse-cost kernel

The per-configuration weights do not depend on q′, and every contraction of
the fine kernel `α(q′,Q−q′)` with ensemble data only ever sees the fields
**at the corners**. Therefore the whole fine sum folds exactly onto the
coarse mesh:

```
A_cart(k) = Σ_{q′∈fine BZ} I_{ε:k_ε(q′)=k}(q′) · α_cart(q′, Q−q′)     (fold, exact)
w_i(Υ¹)   = Σ_{k∈coarse BZ} A_ab(k)·conj(g_a(k))·conj(g_b(Q−k))       (coarse contraction)
```

and symmetrically the fine `d2v` outputs are the trilinear interpolation of
the coarse `d2v` blocks computed with those same weights. Consequences:

- The config×symmetry loop runs on the **unmodified coarse Julia kernel**
  `get_perturb_averages_qspace` with the folded `alpha1` blocks — **no new
  Julia code**, no `8·N_f/N_q` cost blow-up (the paper's cost estimate is a
  large overestimate); the O(N_f) fold/unfold is trivial numpy work.
- Hermiticity is preserved by construction: the unfold is the exact adjoint
  of the fold (same `I_ε` weights on both sides), and the mesh factors are
  the Hermitian-symmetric `scale3 = √(N_c/N_f)`, `scale4 = N_c/N_f` already
  supported by the kernel (audit item 8).

### 2.4 Permutation symmetry (requirement from the Julia code)

- **q₁ ↔ q₂ (pair legs):** for coarse Q, the corner set of `Q−q′` is the
  mirror of the corner set of `q′` with *identical* weights
  (`j′ = N_c·x_Q − j − 1 mod N_c`, `t′ = 1−t`). Folding the full fine BZ
  (each stored unique block plus its transpose for the reverse orientation)
  therefore yields `A(Q−k) = A(k)ᵀ` **exactly** — the transpose symmetry the
  Julia kernel assumes for its `×2` off-diagonal multiplicity. This will be
  asserted numerically.
- **Q leg permutations:** handled, as in the commensurate code, by the
  estimator's three D3 force channels (`weight_R` cross terms + `weight_Rf`
  diagonal term) and four D4 channels inside the fused kernel — reused
  unchanged.
- A dedicated test extracts the effective interpolated Φ³ with unit-vector
  probes and checks invariance under leg exchange (§6).

### 2.5 Why the atom-position phase is not atom-resolved centering

The optional `atomic_phase=True` uses the full-Bloch convention

```
u_full,a(q) = exp(-2π i q·tau_a) u_cell,a(q)
```

for both legs of every rank-2 block. Fold transport from fine q to coarse k
uses `P(k)† P(q)` on each leg; unfold uses its inverse adjoint. The centered
BZ representative of q is essential because this gauge is not periodic
under `q -> q+G` for a non-Bravais basis. These paired phase maps retain
`A(Q-k)=A(k)^T` and Hermiticity. Numerically they barely affect SnTe, which
shows that the error is not a missing Fourier convention.

This phase is only a unitary change of Bloch basis. It inserts the intracell
offset `tau_a`, but does not choose a periodic image
`delta + L m` for an aliased real-space interaction. Consequently every
atom pair still receives the same scalar trilinear/Bartlett corner weight.
For a 2×2×2 mesh, the cell difference `delta=1` remains an equal `+1/-1`
image tie. Multiplying by `exp(-2π i q·tau_a)` cannot decide which image
makes a particular Sn–Te–Sn triplet shortest.

There is a second obstruction: the required image is a property of the
complete three-atom cluster, not of either pair separately. For folded class
`(a,b,c; delta_b,delta_c)`, exact Tensor3-style centering chooses

```
M_abc = argmin_(m_b,m_c) [ |x_a-x_b| + |x_a-x_c| + |x_b-x_c| ]
x_a = tau_a
x_b = tau_b + (delta_b + L m_b) A
x_c = tau_c + (delta_c + L m_c) A .
```

The corresponding interpolation kernel is atom-triplet dependent,

```
K_abc(q1,q2) = Σ_(m_b,m_c in M_abc) w_m
  exp{-2πi[q1·(delta_b+L m_b) + q2·(delta_c+L m_c)]} .
```

The current trilinear fold cannot express this kernel: it folds the two pair
legs before entering the estimator and uses one weight for the whole
Cartesian block. The third (force) atom is still hidden inside the coarse
configuration kernel, yet `K_abc` depends on it. A gauge phase has no access
to that missing index.

### 2.6 Formulation that can recover atom-resolved precision

There are two sound routes:

1. **Centered Tensor3 vertex (exact and already validated).** Estimate FC3
   on the coarse ensemble, apply `Tensor3.Center(Far=3)` with the complete
   triplet-perimeter rule, Fourier-evaluate the centered tensor at every
   fine pair, and use the same vertex blocks on the `R -> two-phonon` map and
   their conjugate adjoints on the reverse map. This is the existing
   deterministic `d3_mode="tensor"` formulation. It gives 37.65 cm⁻¹ for
   2³→4³, compared with 34.79 cm⁻¹ for the native TDSCHA Lanczos calculation
   on the independently converged 4³ SSCHA reference. The former 4³→8³ and
   “full 8³” values are withdrawn because their 8³ reference was not
   independently converged.
2. **Atom-triplet-resolved stochastic kernel.** Split each of the three D3
   force channels by its pinned force atom, expose all three atom indices,
   and apply `K_abc` (including tied images) before summing the channels. The
   reverse channel must use the exact adjoint kernel. Permutation symmetry
   must be enforced across the three pinned channel families and the ASR
   imposed as a hard kernel constraint. This is no longer the unmodified
   coarse-cost trilinear fold; it requires slot-resolved kernel calls. The
   existing atom-pinned window path is a practical version; its former
   37.7 cm⁻¹ value was measured on the interpolated-auxiliary control.
   Complete-graph tuple centering is the route to matching Tensor3 image
   assignment exactly.

---

## 3. Class design

New module `Modules/QSpaceTrilinear.py`, class
`QSpaceTrilinearLanczos(QSpaceLanczos)`. Reused module-level helpers from
`QSpaceInterpolation`: `generate_fine_mesh`, `build_q_index_lookup`,
`_mesh_key`, `interpolate_dyn_fine`.

### 3.1 Attributes

```python
class QSpaceTrilinearLanczos(QL.QSpaceLanczos):
    def __init__(self, ensemble, fine_mesh, use_asr_dyn=True,
                 fine_w_threshold=None, atomic_phase=False,
                 lo_to_split=None, **kwargs):
        super().__init__(ensemble, lo_to_split=lo_to_split, **kwargs)
        # ---- keep coarse copies (kernel side) ----
        self.coarse_mesh   = np.array(self.dyn.GetSupercell())
        self.cq_points     = self.q_points.copy()     # coarse q list
        self.cn_q          = self.n_q
        self.cw_q, self.cpols_q = self.w_q.copy(), self.pols_q.copy()
        self.cvalid_modes  = self.valid_modes_q.copy()
        # X_q, Y_q stay coarse — they are only ever used by the kernel.

        # ---- swap the "public" arrays to the fine mesh ----
        fq, fidx = QI.generate_fine_mesh(self.uci_structure, fine_mesh)
        assert np.all(np.array(fine_mesh) % self.coarse_mesh == 0)
        w_f, pols_f = QI.interpolate_dyn_fine(self.dyn, fq, use_asr=use_asr_dyn)
        self._pin_commensurate(fq, w_f, pols_f)   # §3.2
        self.q_points, self.n_q, self.w_q, self.pols_q = fq, len(fq), w_f, pols_f
        self.fine_mesh, self._fine_idx = np.asarray(fine_mesh, int), fidx
        self._q_lookup = QI.build_q_index_lookup(fq, self.uci_structure, fine_mesh)
        self._build_valid_modes_fine(fine_w_threshold)
        self._build_corner_cache()                # §3.3

        # Hermitian-symmetric mesh factors (audit item 8)
        ratio = self.cn_q / float(self.n_q)
        self.qspace_scale3, self.qspace_scale4 = np.sqrt(ratio), ratio
```

`n_bands`, masses, ensemble Bloch data are untouched (coarse). All parent
methods that only touch `q_points / w_q / pols_q / valid_modes_q /
unique_pairs` (harmonic L, χ±, Wigner conversions, mask, Lanczos loop,
perturbation setup) now transparently operate on the fine mesh.

### 3.2 Fine dynamical matrix (FT-sign care)

`interpolate_dyn_fine` already: centers with `Tensor2`, applies ASR, calls
`t2.Interpolate(-q)` (**the sign fix** — regression-tested to reproduce
`dyn.dynmats` at commensurate q), enforces the TRI gauge
`e(−q) = conj(e(q))`. One addition, `_pin_commensurate`: at fine points that
coincide with coarse ones, **copy `w_q`/`pols_q` from the parent's
`DiagonalizeSupercell` output** rather than re-diagonalizing, so the mode
basis at Q (R-sector, `R1` passed to the kernel) is *identical* on both
sides — degenerate-subspace gauge differences would otherwise silently
misalign R1/f_pert.

Effective charges (REVISED 2026-07-23, see benchmark.md section 19): the
nonanalytic Gamma term is still never applied, but "warn + neglect" was a
misdescription of what actually happens. When `dyn.effective_charges` is set,
`ForceTensor` subtracts the Ewald dipole-dipole term before centering/ASR and
re-adds it at every interpolated q. That cycle is exactly the identity on the
coarse mesh (so no commensurate test sees it) and replaces the plain Fourier
continuation off-grid. Correct for a polar potential, wrong for a short-range
MLIP, where it produced 12 spurious imaginary modes on the CsSnI3 8^3
off-grid shell. Now controlled by the opt-in, interpolation-scoped
`ignore_effective_charges` flag on `QSpaceTrilinearLanczos` /
`interpolate_dyn_fine`; default False (unchanged behaviour), and the caller's
`dyn` is never mutated so IR response still sees Z*.

### 3.3 Trilinear geometry (exact integer arithmetic)

```python
def _build_corner_cache(self):
    """corners[iq_fine] = [(coarse_flat_index, weight), ...] (≤ 8, w>0)."""
    Nc, Nf = self.coarse_mesh, self.fine_mesh
    self._corners = []
    for n in self._fine_idx:                      # q′_frac = n / Nf
        jt = [divmod(Nc[d] * n[d], Nf[d]) for d in range(3)]   # j_d, r_d
        entries = []
        for eps in itertools.product((0, 1), repeat=3):
            w = 1.0
            for d in range(3):
                t = jt[d][1] / Nf[d]
                w *= t if eps[d] else (1.0 - t)
            if w == 0.0:
                continue                          # exact-node / face cases
            key = tuple((jt[d][0] + eps[d]) % Nc[d] for d in range(3))
            entries.append((self._coarse_lookup[key], w))
        self._corners.append(entries)
```

`divmod` keeps `j_d`, `t_d` exact (`t_d = r_d/N_f_d`), so commensurate fine
points collapse to a single corner with weight 1 — the identity limit is
exact by construction.

### 3.4 Fold and unfold (all Cartesian)

Basis transforms (`E = pols`, unitary; derived from `x = uᵀ·conj(E)`):
mode→Cartesian `M_cart = E₁ @ M_mode @ E₂.T`; Cartesian→mode
`M_mode = E₁.conj().T @ M_cart @ E₂.conj()`. The *same* pair of transforms
applies to both the α kernel (contracted with conjugated fields) and the
d2v dyadic outputs (unconjugated components) — verified analytically.

```python
def _fold_alpha_to_coarse(self, alpha1_fine):
    nb = self.n_bands
    A = np.zeros((self.cn_q, nb, nb), dtype=np.complex128)   # A[k] pairs with Q-k
    for p, (iq1, iq2) in enumerate(self.unique_pairs):        # fine unique pairs
        blocks = [(iq1, iq2, alpha1_fine[p])]
        if iq1 != iq2:
            blocks.append((iq2, iq1, alpha1_fine[p].T))       # reverse orientation
        for iqa, iqb, blk in blocks:                          # full fine BZ
            E1, E2 = self.pols_q[:, :, iqa], self.pols_q[:, :, iqb]
            blk_cart = E1 @ blk @ E2.T
            for ik, w in self._corners[iqa]:
                A[ik] += w * blk_cart
    # -> coarse mode basis, ordered as coarse unique pairs (kernel input)
    out = []
    fy = self._get_fy_coarse()
    for ik1, ik2 in self.c_unique_pairs:
        E1, E2 = self.cpols_q[:, :, ik1], self.cpols_q[:, :, ik2]
        A_mode = E1.conj().T @ A[ik1] @ E2.conj()
        out.append(fy[:, ik1][:, None] * A_mode * fy[:, ik2][None, :])
    return out          # A[Q-k] == A[k].T exact — asserted in tests

def _interp_d2v_to_fine(self, d2v_coarse):
    nb = self.n_bands
    D = np.zeros((self.cn_q, nb, nb), dtype=np.complex128)
    for p, (ik1, ik2) in enumerate(self.c_unique_pairs):      # -> Cartesian, both orientations
        E1, E2 = self.cpols_q[:, :, ik1], self.cpols_q[:, :, ik2]
        D[ik1] = E1 @ d2v_coarse[p] @ E2.T
        if ik1 != ik2:
            D[ik2] = D[ik1].T
    fine = []
    for iq1, iq2 in self.unique_pairs:                        # adjoint of the fold
        B = sum(w * D[ik] for ik, w in self._corners[iq1])
        E1, E2 = self.pols_q[:, :, iq1], self.pols_q[:, :, iq2]
        fine.append(E1.conj().T @ B @ E2.conj())
    return fine
```

This snippet shows the production coarse-Upsilon path. With
`atomic_phase=True`, the phase transport of §2.5 is inserted on both fold
and unfold; it is never a one-sided post-processing step.

(Loops shown for clarity; production code batches them as einsums over a
precomputed sparse fold matrix `W[iq_fine, ik_coarse]`.)

### 3.5 The L application (override)

```python
def apply_anharmonic_FT(self, transpose=False, **kwargs):
    if self.ignore_v3 and self.ignore_v4:
        return np.zeros(self.get_psi_size(), dtype=np.complex128)
    R1 = self.get_R1_q()                              # mode basis at Q (pinned == coarse)
    if self.ignore_v3:
        R1 = np.zeros_like(R1)
    alpha1_fine  = self._get_alpha1_bare()                 # fine χ±, no Upsilon yet
    alpha_coarse = self._fold_alpha_to_coarse(alpha1_fine)
    f_pert, d2v_coarse = self._call_julia_qspace_coarse(            # §3.6
        R1, self._flatten_blocks(alpha_coarse))
    d2v_fine = self._interp_d2v_to_fine(d2v_coarse)
    # assemble exactly like the parent: χ± factors at the fine pairs
    final_psi = np.zeros(self.get_psi_size(), dtype=np.complex128)
    final_psi[:self.n_bands] = f_pert
    chi_m, chi_p = self.get_chi_minus_q(), self.get_chi_plus_q()    # parent, fine tables
    for p, _ in enumerate(self.unique_pairs):
        self.set_block_in_psi(p, np.sqrt(-0.5*chi_m[p]) * d2v_fine[p], 'a', final_psi)
        self.set_block_in_psi(p, -np.sqrt(0.5*chi_p[p]) * d2v_fine[p], 'b', final_psi)
    return final_psi
```

### 3.6 Kernel call and symmetries (coarse side)

`_call_julia_qspace_coarse` is the parent's `_call_julia_qspace` with the
coarse attributes: coarse `X_q/Y_q/w_q/valid_modes`, coarse `iq_pert`
(lookup of Q in `cq_points`), coarse `q_pair_map`/`unique_pairs`, and
`scale3/scale4` (already threaded through to Julia). MPI/`GoParallel`
plumbing is inherited verbatim. `prepare_symmetrization` is overridden to
build the sparse symmetry matrices with the **coarse** `q_points/pols_q`
(the kernel rotates coarse fields; the fold geometry is fixed — same status
as the parent's fixed `alpha1` under symmetrized ensemble averaging).
`build_q_pair_map(iq_pert)` uses O(N_f) integer arithmetic (as in the
existing interp class), asserts Q is coarse-commensurate, and additionally
builds the coarse pair map for the kernel.

### 3.7 What is reused untouched

`run_FT`, `mask_dot_wigner`, `apply_L1_FT`, `get_chi_minus_q/plus_q`,
`_pack/_unpack`, `set_block_in_psi`,
`prepare_mode_q` / `prepare_perturbation_q` / `prepare_ir` / `prepare_raman`,
the whole Julia kernel `get_perturb_averages_qspace` (with its existing
`scale3/scale4` arguments; `prefiltered=False`).

---

## 4. Numerical care points

- **FT signs** (user warning): three distinct conventions coexist — ensemble
  `e^{−2πiq·R}/√N_q` (matches paper), CC dynmats `e^{−2πiq·(R_i−R_j)}/N_q`,
  Tensor2.Interpolate (opposite sign ⇒ call with `−q`). Each is pinned by an
  identity test at commensurate points (§6).
- **Acoustic / small ω on the fine mesh**: exact translations masked at fine
  Γ (parent logic); near-Γ acoustic modes are *physical* decay channels but
  have `1/ω`-divergent χ⁺ and Bose factors — expose `fine_w_threshold`
  (default: mask only `|ω| < CC.Phonons.__EPSILON_W__`, plus
  `ignore_small_w` behavior inherited).
- **LO-TO**: nonanalytic term not included in the interpolated dyn (warn),
  as in the existing module — phase 1 limitation.
- **Measured algorithmic control**: trilinear interpolation in q ≡ Bartlett
  (triangular) real-space centering of the correlation functions. On SnTe it
  under-captures D3: 46.11 cm⁻¹ instead of the converged-4³ native Lanczos
  value 34.79 cm⁻¹. Fourier Upsilon makes
  the under-capture worse (49.88–51.25 cm⁻¹); the full-Bloch atomic gauge is
  nearly neutral (45.98 cm⁻¹ with coarse Upsilon). The missing ingredient is
  atom-resolved FC3 centering/range, not harmonic/Upsilon interpolation or a
  Fourier phase convention. The apparent earlier 8×8×8 source-mesh trends
  are withdrawn: the old 4³ auxiliary matrix was interpolated from 2³, and
  the 8³ curve used a 2³ auxiliary/FC3 calculation on an 8³ integration
  grid.

---

## 5. Environment

Everything runs in the `sscha` micromamba environment. Install this checkout
editable before running the tests or material benchmarks:

```bash
micromamba run -n sscha pip install --no-build-isolation -e .   # meson-python dev install
micromamba run -n sscha python -c "import tdscha"                # smoke
micromamba run -n sscha pytest tests/test_trilinear -x           # new tests
micromamba run -n sscha python report/interpolation/scripts/bench_snte_trilinear8.py
micromamba run -n sscha python report/interpolation/scripts/bench_snte_oracle8.py
micromamba run -n sscha python report/interpolation/scripts/plot_snte_trilinear8_comparison.py
```

---

## 6. Tests (`tests/test_trilinear/`)

1. **Identity** — `fine_mesh == coarse mesh`: `L·ψ` equals the parent
   `QSpaceLanczos` to ~1e-12 for random ψ (fold collapses to identity), and
   full-spectrum Lanczos coefficients match.
2. **Dyn convention** — interpolated `D(k)` at every coarse k equals
   `dyn.dynmats[ik]`; pinned `ω/e` identical to `DiagonalizeSupercell`.
3. **Fold transpose symmetry** — `A(Q−k) = A(k)ᵀ` (machine precision), for
   TRI and non-TRI Q.
4. **Hermiticity** — `⟨φ, Lψ⟩ = conj(⟨ψ, Lφ⟩)` under `mask_dot_wigner`,
   random complex φ, ψ, several Q including non-TRI (guards the audit-item-5
   conjugation class of bugs).
5. **Effective-Φ³ permutation symmetry** — probe L with unit vectors to
   extract the interpolated 3-leg vertex; assert leg-exchange invariance
   (the user's explicit requirement, mirroring the Julia code's channels).
6. **Scale-factor necessity** — removing `scale3/scale4` breaks the
   `N_f→N_c` consistency of the self-energy on the toy chain (analogue of the
   existing test in `tests/test_interpolation`).
7. **Physics, toy chain** — coarse 3 → fine 6 vs a direct fine-supercell
   ensemble (reuse the existing harness); D3-only, then D3+D4 (scaled
   together, per the benchmark-design rule).
8. **Physics, real material** — SnTe (or Fm-3m clean cubic) 2×2×2 → 4×4×4:
   renormalization vs direct fine reference and vs the windowed scheme.
9. **Atomic gauge** — the phase-transported fold retains
   `A(Q-k)=A(k)^T`, and the complete D3+D4 L satisfies the masked
   Hermiticity identity at non-TRI Q.

---

## 7. Milestones

- **M1**: module skeleton, fine mesh + pinned dyn interpolation, corner
  cache, ψ layout on fine pairs, harmonic-only run; tests 1–2.
- **M2**: fold/unfold + anharmonic override + coarse kernel call; tests 3–6.
- **M3**: toy-chain physics validation (test 7).
- **M4**: real-material benchmark, comparison vs `QSpaceLanczosInterp`,
  short report section in `report/interpolation`.
- **M5**: Fourier-Upsilon and atom-position phase hypotheses benchmarked on
  SnTe. Complete: neither improves the result; Fourier-Upsilon paths removed
  and coarse Upsilon retained. The atomic phase remains only as a diagnostic.
- **M6**: matched 8×8×8 source-mesh convergence audit. **Reopened.** The
  former 4³ source and 8³ reference were not independently converged SSCHA
  calculations, so their numerical comparison is withdrawn. The new 4³
  reference is converged; recompute the 4³→8³ curves and native 8³ reference
  after an independently converged 8³ auxiliary matrix becomes available.
- **M7 candidate**: replace the universal corner kernel by the complete-graph
  atom-triplet kernel of §2.6, or reuse the already validated centered
  Tensor3 vertex. This requires exposing the force-leg atom and cannot use
  the unmodified coarse fold.

---

## 8. Resolved design decisions

1. **Pair-sector storage basis.** Keep the ψ two-phonon blocks
   in the *per-fine-pair mode basis* (Wigner a′/b′, exactly as the parent),
   because (a) the harmonic propagator and χ± are diagonal there, (b) **no
   quantity is ever interpolated across q in the polarization basis** — all
   corner mixing happens after exact, per-q unitary rotation to Cartesian,
   which is what the paper's "interpolate in Cartesian" requirement is
   about, and (c) it maximizes reuse of validated parent code. A literal
   Cartesian ψ storage is possible but adds per-pair basis conversions to
   *every* sector (harmonic included) with no change in the computed
   operator.
2. **Naming**: `Modules/QSpaceTrilinear.py`, class `QSpaceTrilinearLanczos`.
3. **Overleaf**: equation issues remain recorded in the audit above.
4. **LO-TO**: neglected in the interpolated dyn for phase 1; all benchmark
   comparisons strip it consistently.

---

## 9. Phase-winding topology of the SnTe vertex (why no corner scheme can work)

Probe: `report/interpolation/scripts/snte_phase_topology.py` (deterministic,
vertex level, no ensemble); data `report/interpolation/data/
snte_phase_topology.{json,npz}`; figures `figs/snte_phase_winding.pdf`,
`figs/snte_winding_scheme.pdf`; report Sec. "Why the phase of Phi3 rotates
in q space" (`sec:topology`).

Findings (Gamma-TO-projected vertex `V(q) = e_TO . Phi3(Gamma; q, -q)`):

1. The vertex is entirely on the Te–Sn bond: mixed Te–Sn atom blocks have
   norm ~0.28, Te–Te and Sn–Sn blocks ~1e-4 (x~1000 smaller).
2. The dominant Te–Sn element is a pure two-image interference:
   `V(q) = c (e^{2 pi i q.R1} - e^{2 pi i q.R2})`, c = 0.2208,
   R1 = (0,-2,1), R2 = (1,-1,0), residual harmonics < 1e-3 of c
   (3D 8^3 FFT; corrects the earlier two-path guess (0,0,-1)).
   Midpoint x_c = (R1+R2)/2 = (1/2,-3/2,1/2) = tau_Te - tau_Sn exactly:
   half-integer, on the Te–Sn bond. Phase = e^{2 pi i q.x_c}: rotates by exactly pi per
   BZ crossing (linear to 6e-7 deg); |V| is a smooth sine arch.
3. All 8 coarse (2^3) samples are real (Im/Re < 1e-12; TRI points), sign
   pattern (0,-,+,0,+,0,0,-); half of them ~0. Trilinear with real convex
   weights therefore produces an *identically real* fine vertex
   (max Im 8e-17 vs true 0.44): chord instead of arc. At (1/4,0,0) it keeps
   1/sqrt(2)=0.706 of the amplitude; at the cube centre (1/4,1/4,1/4) the
   corner sum cancels: ||V|| 0.442 -> 0.0007 (factor 1.6e-3) while the true
   norm is 71% of max there. This is the measured 29–35% |V|^2 survival.
4. No singularity in the true vertex: only parity-protected zeros at Gamma
   (odd Phi3 x three odd TO modes). No interior phase vortices on a 41x41
   scan of the (q1,q2,0) plane. The nodal surfaces live only in the real
   interpolated representation.
5. Quantized invariant: inversion + Phi3-oddness force the lattice
   harmonics to be antisymmetric about x_c with 2 x_c integer:
   `c_R = -c_{2 x_c - R}`. The class nu = 2 x_c mod 2 is Z2 per direction;
   SnTe: nu = (1,1,1) (nontrivial, Zak phase pi, SSH mid-bond class).
   Consequences (all verified): forced Sum_R c_R = 0 (zero at Gamma), real
   alternating-sign TRI samples, and protection: the rotation cannot be
   recovered from the (all-real) samples themselves; it must be supplied
   externally. A branch-consistent per-atom Bloch phase supplies it
   exactly (see section 10 below — this corrects the earlier blanket
   no-gauge claim); amplitudes remain unrecoverable by any phase, and the
   full cure is still the atom-resolved centering / centered Tensor3.

Practical rule: a coupling channel interpolates accurately with corner
weights iff its class is nu = 0 (site-centred). Detector: track arg V(q)
along one BZ crossing per axis (odd multiple of pi = nontrivial), or
compare vertex signs between the q=0 and q=1/2 TRI planes.

---

## 10. Per-atom phase gauge: correction of §9's no-gauge claim (2026-07-18)

User challenge: can a per-atom phase cure the failure? Answer: it cures
the PHASES exactly (for SnTe), it cannot cure the AMPLITUDES, and the Γ
D3-only benchmark only sees amplitudes. Two errors were found:

1. The "singular values of P are gauge-invariant" argument (report,
   earlier §) is invalid — it constrains norms, not alignment. A
   q-dependent gauge changes which function is interpolated.
2. `atomic_phase=True` had a branch bug: corner phases evaluated at
   centered-BZ-wrapped representatives flip sign for half-integer τ
   differences. FIXED in `Modules/QSpaceTrilinear.py`: `_corners` now
   stores `(ik, w, delta)` with `delta = corner − q_fine` unwrapped
   (|delta| ≤ 1/Nc) and `_atomic_delta_phase(delta)` replaces
   `_atomic_bloch_phase`. Consumers of the corner tuples updated
   (structural test, oracle scripts, bench_trilinear_phase). Class
   matches an independent gauged map to 0.0; 14/14 tests pass.

Measured (scripts `snte_phase_gauge_test.py`, `oracle_gauge_spectrum.py`):

- All nine Te–Sn channel centres x_c equal τ_Te−τ_Sn (raw stored coords)
  ⇒ the factorized atomic gauge IS the optimal per-channel phase.
  (Corrects §9 item 2: the centre is (1/2,−3/2,1/2), not (1/2,−1/2,−1/2);
  the earlier value came from an under-determined two-path FFT.)
- Vertex rel err² over 4³ fine pairs: plain 1.12 | buggy gauge 0.78 |
  corrected gauge 0.178 | best-of-125 integer branches 0.178 (m=0) |
  per-channel oracle strip 0.178. |V|² survival 0.375 → 0.422 only;
  cube-centre ‖V‖ 0.0007 → 0.156 (true 0.442).
- Spectra (tensor-D3 oracle, deterministic): Fourier 37.65 | plain
  trilinear 45.78 | phase-corrected trilinear 45.59 cm⁻¹. Only 0.19 of
  the 8.13 cm⁻¹ gap: the D3-only Γ response is |V|²χ — phase-blind —
  which also explains the old −0.13 cm⁻¹ stochastic result.

Bottom line: keep `atomic_phase=True` (now phase-exact; relevant for
D3×D4 cross terms / off-diagonal / non-Γ observables); the benchmark
error is amplitude loss from samples pinned at envelope nodes, curable
only by supplying the real-space image assignment (centering) or a
denser coarse mesh (L≥3 resolves the Nyquist sine).

---

## 11. Fourth order (D4) in the atom-centred Fourier scheme (2026-07-21)

**Status: COMPLETE.** Vertex map verified exact (machine zero); 10
operator-level tests green; spectral benchmark + figure done; report
section `sec:d4-atom-fourier` written and compiling.

Question: does the atom-centred Fourier continuation of Sec. 10 /
`sec:atom-fourier` (the `atom_fourier=True` kernel `P_ab(q,k)`, i.e. the
phase factor moved into the Fourier transform) also interpolate the
FOURTH-order vertex?

### 11.1 Why it factorizes (and why this is not the real-space D4 problem)

The D4 term maps the incoming two-phonon block at the fine pair
`(q', Q-q')` onto the outgoing block at `(q, Q-q)`.  Because the estimator
contracts the incoming block with CONJUGATED Bloch fields, the four vertex
legs carry

    leg 1 = a at +q     leg 2 = b at -q
    leg 3 = c at -q'    leg 4 = d at +q'

Pinning leg 1 to the home cell, translation invariance gives

    V(q,q')[a,b,c,d] = sum_{R2,R3,R4} Phi4(a0,bR2,cR3,dR4)
                       exp{2 pi i [ q.R2 + q'.(R3-R4) ]}.

The two momenta are INDEPENDENT and each couples to exactly one pair
offset: the q-harmonic is `R2` (minimal image near `d_ab = tau_a-tau_b`)
and the q'-harmonic is `R3-R4` (minimal image near `-d_cd`).  These are
precisely the harmonics continued by `P_ab(q,k)` on the unfold and
`conj(P_cd(q',k'))` on the fold.  **No new geometry is needed: the correct
four-leg continuation is the tensor product of the existing pairwise
kernel with itself.**

This is the decisive structural difference from the real-space D4
centering attempts of `sec:d4center` (`d4_center="leg"`, wrong sign;
`"reference"`, 50% capture).  There, minimal-image assignment is a genuine
four-body geometry problem that does not factorize into pairs.  In the
atom-centred q-space gauge the two independent momenta *do* factorize, so
the four-leg problem reduces exactly to two independent pair problems.

Consequence: `atom_fourier=True` already routes D4 through the correct map
in the existing fold/unfold architecture -- the D4 channel needs no new
code path.  What was missing was verification, tests, and a benchmark.

### 11.2 Deterministic vertex oracle (noise-free)

Script `report/interpolation/scripts/d4_atom_fourier_oracle.py`; data
`report/interpolation/data/d4_atom_fourier_oracle.json`.  The quartic
vertex of the anharmonic diatomic chain is known in closed form
(`6*g4*(delta_i-delta_j)^{otimes 4}` per bond per Cartesian component), so
the interpolation map is tested with zero stochastic noise: sample the
exact vertex at the coarse pairs, apply the production kernel, compare
against the exact fine vertex over ALL fine (q,q') pairs.

Nearest-neighbour quartic (range fits every coarse minimum-image window):

| coarse -> fine | atom_fourier rel err^2 | trilinear rel err^2 (off-grid) |
|---|---:|---:|
| 3 -> 9 | 1.4e-31 | 5.55e-2 (6.25e-2) |
| 2 -> 4 | 8.7e-33 | 1.75e-1 (2.33e-1) |
| 2 -> 6 | 3.1e-32 | 1.81e-1 (2.03e-1) |
| 4 -> 8 | 4.3e-32 | 1.91e-2 (2.55e-2) |

The atom-centred Fourier continuation reconstructs the exact four-leg
vertex to MACHINE ZERO, including at L=2 where every off-grid pair is a
Nyquist-tie case.  Ordinary trilinear loses 2-18%.  (That the L=2 result is
exact also independently confirms the sign/branch convention derived in
11.1: a single wrong sign destroys the cancellation.)

### 11.3 The honest limit: range, not order

Adding a second-neighbour quartic bond A(n)-A(n+2) (harmonics at |R|=2):

| coarse -> fine | atom_fourier | trilinear |
|---|---:|---:|
| 2 -> 4 (aliased: |R|=2 > L/2) | **2.50e-1** | 3.59e-1 |
| 4 -> 8 (resolved) | 4.5e-32 | 1.34e-2 |
| 5 -> 10 (resolved) | 2.2e-31 | 3.05e-2 |

The scheme is exact iff the quartic range fits the coarse minimum-image
window, and fails otherwise -- that failure is genuine aliasing of the
coarse samples, not a defect of the map: no q-space continuation can
recover a harmonic the coarse mesh does not resolve.  This is the same
Nc-convergence statement as for D3, one order up.

Note also that at L=4 the |R|=2 harmonic of the A-A bond is an EXACT
Nyquist tie (d_AA = 0), split 50/50 by `_nearest_alias_images`, and the
result is still exact: for a coupling symmetric under the leg exchange
that maps R -> -R, the tie split is not a compromise but the exact answer.

### 11.4 Done (2026-07-21)

- **operator-level tests** (`tests/test_trilinear/test_d4_atom_fourier.py`,
  10 tests, all green): exact vertex reconstruction; range-limit is
  aliasing not a defect; kernel mirror identity `P_ab(q,k)=P_ba(-q,-k)`;
  the three leg exchanges of Phi4; ASR preserved on all four legs (with
  non-cardinal + atom-pinned control maps that break it, proving teeth);
  isolated-D4 block Hermiticity at TRI/non-TRI Q; unfold mirror
  `D(Q-q)=D(q)^T`; commensurate identity to the parent (D4 non-vacuous).
- **spectral benchmark + figure**: `bench_d4_atom_fourier.py`,
  `make_d4_atom_fourier_fig.py` (`figs/d4_atom_fourier.pdf`),
  `d4_seed_noise_control.py` (noise floor).  atom_fourier captures the D4
  renormalization to within the 3-seed noise floor at every probe;
  trilinear under-captures and flips sign on the weak line.  Full table in
  `benchmark.md` section 12.
- **report section** `sec:d4-atom-fourier` (with `fig:d4atomfourier` and
  `eq:d4-two-momenta`), placed at the end of the D4 section; the report
  compiles (42 pages).

Subtlety found and documented: `ignore_v3=True` does NOT isolate a
Hermitian D4 operator -- it zeroes R1 (the D3 two-phonon OUTPUT channel)
but leaves the D3 input channel alpha -> f_pert active.  This asymmetry
(~2.9e-2 on the chain) is pre-existing behaviour of the parent
QSpaceLanczos, not of the interpolation.  D4 is isolated for the
Hermiticity test by projecting the R sector out of input and output
instead.

---

## 12. AUDIT + FIX: Nyquist ties were metric-blind (2026-07-21, user-requested)

**Status: FIXED and validated.** User asked to check whether Nyquist ties
(incl. >2-image 3D ties, e.g. bcc central atom to 8 replicas) are correctly
implemented.

**Problem.** `_nearest_alias_images` + the old `_build_atom_fourier_kernel`
resolved the minimal image and its ties SEPARABLY PER CARTESIAN AXIS in
FRACTIONAL coords (metric-free), then took a product over axes (matching
the report's product form eq:atom-fourier-kernel). Exact ONLY for
orthorhombic coarse lattices. On non-orthogonal cells the true min image
minimizes |(R-d).A| (metric A A^T couples axes), and its ties live on
non-axis-aligned WS faces. Audit `report/interpolation/scripts/
audit_nyquist_ties.py` (data `data/audit_nyquist_ties.json`): cubic exact
(incl. 8-fold WS-corner tie -> 1/8); fcc/bcc/hex kernels mis-assign
(genuine 4- and 6-fold ties, the 6-fold at weight 1/6 impossible from
products of halves; also MANUFACTURES spurious ties where metric is
unique).

**IMPORTANT NUANCE (user pushed back: "SnTe works well, why big
mismatch?").** The 23-36% kernel-NORM mismatch is NOT the observable error.
A physical coupling only excites the aliasing classes where it has coarse
weight; a short-ranged coupling excites only classes with a UNIQUE shortest
image, where separable==metric. Verified: the real SnTe Gamma-TO 2-image
coupling (R1=(0,-2,1),R2=(1,-1,0), each unique shortest image at 3.28A)
reconstructs 2^3->4^3 to machine zero (2e-32) with BOTH kernels. So the
2.9e-5 SnTe result is CORRECT, not fortuitous. The bug bites only couplings
pinned to a mis-assigned degenerate/spurious-tie class: minimal demo
(primitive bcc, d=(1/2,1/2,0), class (0,0,1)) separable rel err^2=0.50 vs
metric 1.1e-32. The user's canonical bcc central-atom d=(1/2,1/2,1/2) is
actually NOT a failing case (unique/separable images).

**Fix.** `tie_metric=True` (default). New static
`_metric_alias_images(d,Nc,metric)` = true 3D min-image set per class w/
equal tie split; `_build_atom_fourier_kernel_metric` builds the
non-separable kernel. Legacy separable kept as `tie_metric=False`
(orthorhombic fast path / control). Preserved: commensurate identity,
mirror identity P_ab(q,k)=P_ba(-q,-k) [img(κ;-d)=-img(-κ;d)], Hermiticity,
ASR.

**Tests.** `tests/test_trilinear/test_nyquist_ties.py` (8) +
`_toy_crystal3d.py` (minimal non-orthogonal 3D crystal): metric==brute
force incl. 4-fold tie; cubic==separable; metric fixes the bcc
reconstruction separable gets 50% wrong; commensurate + mirror identity +
operator Hermiticity on the non-orthogonal cell. Full trilinear suite
34 passed (was 26). Chain D3/D4 + SnTe results unchanged.

---

## 13. Performance optimization + CsSnI3 production stress test (2026-07-22)

**User-requested production case:** CsSnI3 P4/mbm Raman, native 4x4x4 ->
interpolated 8x8x8 (D3+D4, atom_fourier), T=500 K. Data under
`.../Perovskites/CsSnI3/SCHA/P4_mbm/4x4x4/T_500`. 10-atom cell (n_bands=30),
supercell 640 atoms, 12288-config ensemble (subsampled), tetragonal
(diagonal) cell. Q=Gamma is coarse => interpolation refines the internal
two-phonon loop 4^3->8^3 at 4^3 sampling cost. Scripts
`report/interpolation/scripts/cssni3_raman_interp.py` +
`make_cssni3_raman_fig.py`.

**Bottleneck analysis (profiled, 8^3):**
- Per L-application: Julia coarse kernel ~1.75s DOMINANT (= native 4x4x4
  cost, scales N_conf x N_sym; interpolation does NOT inflate it -- coarse
  kernel, fine resolution). fold+unfold ~0.3s after optimization.
- Construction ~45s one-time: CC `ForceTensor.Interpolate` (harmonic dyn ->
  512 fine q, 28s) + SetupFromPhonons/Tensor (~25s). Amortized.

**Optimizations (correctness-preserving, 34/34 tests green):**
1. atom_fourier kernel build: separated the phase
   exp(2pi i(q-x).R) = exp(2pi i q.R) conj(exp(2pi i x.R)) => per-pair
   kernel is ONE matmul over images R instead of a 3.3M-iteration Python
   loop (512x64x100 for 8^3 CsSnI3). ~10s -> ms.
2. fold/unfold: replaced Python iq x ik loops with np.repeat by a single
   einsum on the atom-block kernel ('qkab,qaibj->kaibj' fold,
   'pkab,kaibj->paibj' unfold). Eliminated 197k np.repeat/application;
   fold 0.22->0.09s, unfold 0.07->0.02s.

Residual one-time cost is CC's per-q harmonic interpolation; a batched
multi-q DFT of the centered FC2 would cut it but is amortized (future).

Key conceptual result: the interpolation's per-step overhead over native
4x4x4 is ~0.27s/step (fold+unfold); the expensive stochastic kernel is
reused at coarse cost. A native 8^3 would need an independent 8^3 ensemble
AND ~8x more Julia pairs per step. So the class delivers 8^3 resolution at
~4^3 cost -- the intended speedup.

---

## 14. RETRACTION: CsSnI3 spectral results were a Lanczos-instability artifact (2026-07-22)

User challenged the "massive" 4^3->8^3 peak shift; it was a BUG. **Section
13's CsSnI3 physical numbers (90->125/155) are RETRACTED.**

Cause: the NATIVE QSpaceLanczos (not the interpolation) has exponentially
growing tridiagonal coefficients for this soft perovskite -- b*c ~4x/step ->
1e41 by step 80 (a_n stay physical). Terminator meaningless; line shape
unstable to step count (peak flips 11 <-> 90). Correct spectrum (user's
saved June production, bounded |b|~1e-6) is SOFT-MODE DOMINATED ~29 cm-1.

Not interpolation's fault: reproduced in bare QSpaceLanczos, full 12288-cfg
ensemble, standard recipe; toy chain BOUNDED (ratio 0.99) under same code =>
system-specific/regression. Interpolation still validated: identity 3.2e-15
on this ensemble; **incommensurate vertex ratio off/comm = 1.00 (NO
anharmonicity loss at incommensurate points -- refutes that hypothesis)**;
chain/SnTe machine-exact; 34 tests green.

Two real script bugs also fixed: missing ens.init() after ens.split()
(stale forces_qspace, ~100x error); build ensemble around dyn_gen +
update_weights(final) per load_distributed_tdscha. Necessary but NOT
sufficient (native growth persists).

STANDS: +6% wall-time performance result; fold/unfold + kernel-build
optimizations; save_abc replot workflow. Diagnostics + correct reference
archived in Dropbox qspace_interpolation_8x8x8/. A valid CsSnI3 convergence
study needs the native coefficient-growth fixed first.

---

## 15. CsSnI3 CORRECTED (2026-07-22) — supersedes sections 13-14

User's push was right. Section 14's "fatal native growth" was wrong. The bug
was ensemble init in the driver: (1) ens.init() after ens.split(); (2)
Ensemble(dyn_gen)+update_weights(final). After the fix the spectrum is
STEP-STABLE (unpol0 4^3 = 14.0 cm-1, full = 33.1 cm-1, identical at
40/60/80/100 steps) despite b*c ~1e24 -> the growth is BENIGN (continued
fraction insensitive). Corrected rerun: native 4^3 and interp 8^3 both peak
33.1 cm-1 (soft-mode dominated), AGREE to L1=0.0012 -- physically correct
(one-phonon soft mode insensitive to internal mesh; interpolation faithful).
+5% wall time. Figures regenerated (cssni3_raman_interp.pdf,
cssni3_raman_analysis.pdf). Report sec:cssni3 rewritten as corrected result
(no longer a retraction). Scripts fixed + save_abc replot workflow.
