# Root-cause investigation: q-space anharmonicity lost at non-TRI q — full findings

**Date:** 2026-06-11.
**Test case throughout:** real 3×3×3 gold ensemble (`Examples/ensemble_gold`),
most anharmonic phonon iq=15 (star iq=13–18), band 1 (SSCHA 67.56 cm⁻¹),
D3-only (`ignore_v4=True`) for the probes unless stated. All runs via
`micromamba run -n sscha`, `JULIA_NUM_THREADS=1`. Errors always judged on the
**anharmonic renormalisation** (result − SSCHA), never on absolute frequencies.

---

## TL;DR

1. **The q-space bug is found and FIXED**: `tdscha_qspace.jl` used a
   sesquilinear (Hermitian) conjugation convention inside a bilinear
   (`q1 + q2 = q_pert`) block layout. The mis-conjugated estimator carries net
   momentum `2·q1`, so its ensemble average vanishes at every non-TRI q —
   the anharmonicity was reduced to stochastic noise. After the fix the
   q-space static Hessian reproduces the independent
   `ens.get_free_energy_hessian` reference to **0–3 % on the shift**
   (previously ~97 % of the shift was lost). All SnTe TRI regression tests
   still pass.
2. **A second, separate bug was uncovered in the REAL-SPACE code**: the
   `DynamicalLanczos` symmetrization applies block-truncated mode-space
   symmetry matrices that are **not orthogonal** on this ensemble (lossy
   projections), because the SSCHA dyn does **not commute with 972 of the
   1296 spglib symmetries** of the ideal supercell structure. The real-space
   Lanczos consequently *underestimates* the anharmonic renormalisation at
   non-TRI stars: its GF shift is 1.15 cm⁻¹ where the q-space Lanczos gives
   10.2 cm⁻¹ and the trusted static Hessian gives 10.92 cm⁻¹. I.e. after the
   q-space fix, **the real-space Lanczos is the outlier**, not q-space.

---

## 1. The q-space bug (FIXED): wrong conjugation convention in the Julia kernel

The q-space machinery pairs momenta as `q1 + q2 = q_pert`
(`QSpaceLanczos.build_q_pair_map`). That pairing implies the **bilinear**
(transpose-type) block convention: kernels must be contracted with
`conj(x(q))` where the real-space code uses plain `x` (since `x(−q) = conj(x(q))`
for the real ensemble fields), and dyadic outputs must use unconjugated band
components. `Modules/tdscha_qspace.jl` instead contracted sesquilinearly.

Momentum counting of the D3 estimator shows the consequence: the
accumulated average `⟨conj(W)·r1(q1)⊗conj(r2(q2))⟩` carries net momentum
`2·q1`, which vanishes identically by translational invariance **unless q1 is
TRI**. On all-TRI grids (2×2×2 SnTe) the conjugations are no-ops (Fourier
components real in the real-pol gauge) — everything agreed, which is why the
bug was never seen there. On non-TRI grids the D3/D4 averages reduce to noise
→ "anharmonicity washed out with more q points".

**Fix applied** (`Modules/tdscha_qspace.jl`, fused kernel + 3 standalone
variants; `Modules/QSpaceLanczos.py`):

- `weight_R  = Σ f_Y·conj(x_pert)·R1`   (was `f_Y·x_pert·conj(R1)`)
- `weight_Rf = Σ R1·conj(y_pert)`       (was `conj(R1)·y_pert`)
- `buffer_u[q1] += α1·conj(x_q2)`; reverse pair uses the **transpose** of α1
  with `conj(x_q1)` (was Hermitian conjugate, plain `x`)
- `local_w = conj(x_q1)·α1·conj(x_q2)`; off-diagonal pairs add `2·local_w`
  (was `local_w + conj(local_w)`)
- `buf_f_weight = Σ buffer_u·f_ψ·conj(y)` (was `conj(buffer_u)·f_ψ·y`)
- d2v outputs `r1(q1)⊗r2(q2)` etc. — **no conjugation** of the second factor
- `QSpaceLanczos._unpack_upper_triangle`: diagonal-pair `(q,q)` blocks are
  complex **symmetric** in this convention, not Hermitian → transpose fill.

### Verification chain (each step numerical, on the gold ensemble)

- **Per-config formulas, no symmetries:** three-way comparison of the D3
  a′-blocks of one anharmonic L application on the pure +q Bloch drive —
  numpy brute force ↔ fixed Julia kernel: ~1e-16; ↔ real-space reference:
  ~8e-8 uniformly (Bose-constant rounding only). Exact.
- **Methodological unlock:** the real-space code *can* be driven at pure +q —
  feed it `Re(e_+q)` and `Im(e_+q)` (both real vectors) and combine by
  linearity (`L v₊ = L v_re + i L v_im`). This removed the long-standing
  blocker for block-by-block comparisons.
- **X_q convention pinned:**
  `X_q[iq,i,ν] = (1/√N_q) Σ_a e^{−2πi q·R_a} ũ(a)·conj(pol_q[:,ν,iq])`,
  verified against a direct numpy FT (residual 1e-15 after the Å→Bohr factor
  1.8897).
- **q-space symmetry rep D(q′←q) validated:** the empirical linear map between
  independently FT-ed rotated/unrotated ensembles equals the constructed D to
  1e-11.
- **Symmetrized average validated against ground truth:** explicitly rotating
  the 50 configs in cartesian supercell space for each of the 48 PG ops,
  FT-ing directly, and averaging the validated formula (zero code symmetry
  machinery) reproduces the fixed q-space-with-symmetries blocks to **1e-11 on
  every pair**. The fixed q-space pipeline is exact end-to-end.

### End-to-end confirmation against the independent static reference

`tests/test_qspace/test_gold_nontri.py` now **passes in full**, including the
gold non-TRI Hessian vs `ens.get_free_energy_hessian(include_v4=True)`:

```
iq=1  band0: sscha=49.21 ref=43.72(d-5.49)  qspace=43.78(d-5.42)  shift rel-err=1%
iq=1  band1: sscha=67.31 ref=56.50(d-10.81) qspace=56.62(d-10.69) shift rel-err=1%
iq=1  band2: sscha=68.36 ref=58.94(d-9.42)  qspace=58.70(d-9.66)  shift rel-err=3%
iq=13 band0: sscha=49.53 ref=44.05(d-5.47)  qspace=44.03(d-5.50)  shift rel-err=0%
iq=13 band1: sscha=67.56 ref=56.64(d-10.92) qspace=56.64(d-10.92) shift rel-err=0%
iq=13 band2: sscha=67.95 ref=58.49(d-9.46)  qspace=58.60(d-9.35)  shift rel-err=1%
```

**TRI regressions:** `test_qspace_lanczos.py` + `test_qspace_hessian.py`
(SnTe 2×2×2: GF vs real space, reorthogonalization, Γ Hessian, symmetry,
mode-symmetry, checkpoint) — **7/7 pass**. SnTe Γ-optical control and
force-Parseval also pass.

---

## 2. The real-space bug (NEW, separate, NOT yet fixed): lossy symmetrization in `DynamicalLanczos`

After the q-space fix, the with-symmetry basis-independent moments still
disagreed (`m2`: q-space 2.4619e-13 vs real-space 1.5489e-13; anharmonic part
9.14× ≈ 3.02²). The ground-truth experiment (explicit cartesian rotations,
above) settled the direction: **q-space matches the truth; the real-space
symmetrized average does not.**

Diagnosis chain:

1. The real-space mode-space symmetry blocks (`l.symmetries`, from
   `cellconstructor.symmetries.GetSymmetriesOnModesDeg`) are **faithful** to
   their definition: stored = `e_degᵀ P e_deg` for all 1296 ops (1.5e-15),
   with `P` the cartesian permutation+rotation rep.
2. But the true blocks themselves are **non-orthogonal** for most ops: e.g. on
   the 6-mode star block at 67.561 cm⁻¹, `‖SᵀS − I‖ = 2.0` (rank-deficient —
   the op scatters most of the subspace *out* of the block).
3. Root cause: the supercell SSCHA dyn does **not commute** with 972 of the
   1296 spglib operations of the ideal structure:
   `‖P D Pᵀ − D‖/‖D‖ = 2.4e-2` for the broken ops (machine-zero for the 324
   good ones = 12 PG ops × 27 translations). spglib sees the ideal FCC
   positions (48 PG ops), but the dyn itself only respects a 12-op subgroup.
   The frequency degeneracies of the star are still exact, so the
   degenerate-block detection groups 6 modes — but those 6-dim spaces are
   **not invariant** under the broken ops.
4. Consequence: applying the block-restricted rep for a broken op is a
   **norm-losing projection** (not a rotation). Averaging over 1296 ops then
   systematically *suppresses* the anharmonic signal of non-TRI stars.
   This is the real-space Lanczos bias.

Effect at the observable level (gold, most anharmonic phonon, GF
renormalisation shift, `1/Re G(0)` with terminator, 60 steps):

| | shift (cm⁻¹) |
|---|---|
| trusted static Hessian (`get_free_energy_hessian`) | −10.92 |
| **q-space Lanczos (fixed)** | **−10.2** (consistent: static vs dynamic + terminator) |
| real-space Lanczos | −1.15 (~90 % of the renormalisation lost) |

`tests/test_qspace/test_gold_lanczos_gf.py` therefore now FAILS — but it fails
because its *reference* (the real-space Lanczos) is the broken side. The test
should be reworked to compare the q-space GF against the Hessian shift (or the
real-vs-q comparison marked xfail citing the real-space symmetrization bug).

**Open questions on the real-space side (not yet investigated):**
- Why does the dyn only respect 12 of 48 PG ops — was `dyn_gen_pop1_`
  generated/symmetrized with a smaller group (symprec? site symmetry?), or is
  this expected for this ensemble? If the dyn *should* have the full 48, the
  ensemble data needs re-symmetrization, and the real-space bias would shrink.
- Whether the same lossy-projection bias affects published real-space
  supercell TDSCHA results on systems whose dyn breaks part of the ideal
  structural symmetry.
- Possible code-level guards: filter the op list to those commuting with the
  dyn (cheap check `‖PDPᵀ−D‖`), or refuse block-truncation when the block rep
  is not orthogonal.

Note this also re-frames the historical picture: the *user-visible* symptom
("anharmonicity washed out with more q points") was produced by the q-space
conjugation bug, but the real-space numbers that served as reference at
non-TRI q were themselves biased low by the projection problem. The static
Hessian (independent implementation with its own tensor symmetrization) is
the only fully trusted reference and now agrees with the fixed q-space code.

---

## 3. Files changed

- `Modules/tdscha_qspace.jl` — conjugation-convention fix in
  `get_perturb_averages_qspace_fused` and the three standalone variants;
  docstrings/comments updated with the bilinear-convention rationale.
- `Modules/QSpaceLanczos.py` — `_unpack_upper_triangle`: symmetric (not
  Hermitian) fill of diagonal-pair blocks.
- Reinstalled into the `sscha` micromamba env (by the user).
- No changes to `DynamicalLanczos.py` / cellconstructor (real-space bug left
  unfixed, documented here and in `bug_report_qspace.md`).

## 4. Test status

| test | status |
|---|---|
| `test_gold_nontri.py::test_gold_force_parseval` | pass |
| `test_gold_nontri.py::test_snte_hessian_gamma_tri_control` | pass |
| `test_gold_nontri.py::test_gold_hessian_nontri` | **pass (was xfail — the q-space fix)** |
| `test_qspace_lanczos.py` (2 tests, SnTe TRI) | pass |
| `test_qspace_hessian.py` (5 tests, SnTe TRI) | pass |
| `test_gold_lanczos_gf.py::test_gold_lanczos_gf_most_anharmonic` | **FAILS — reference is the real-space Lanczos, which is the biased side (section 2); rework needed** |

## 5. Probe scripts (throwaway diagnostics, `/tmp`, can be deleted)

`probe_fix.py` (a/b coefficients), `probe_m2.py` / `probe_plusq.py`
(convention-free moments; +q drive of the real-space code via linearity),
`probe_brute.py` / `probe_brute2.py` (three-way no-symmetry block comparison),
`probe_sym.py` / `probe_conv.py` (q-space rep + FT-convention validation;
NB: a first apparent rep failure was a probe artefact — stale cached
`forces_qspace` in a deep-copied ensemble), `probe_symavg.py` (with-symmetry
three-way), `probe_truth.py` (ground-truth explicit-rotation group average),
`probe_realsym.py` / `probe_realsym2.py` (real-space block orthogonality;
stored-vs-true), `probe_commute.py` (dyn–symmetry commutation check).

## 6. Recommended next steps

1. Rework `test_gold_lanczos_gf.py`: assert the q-space GF shift against the
   free-energy-Hessian shift (tolerance ~10–15 % for static-vs-dynamic +
   terminator); add an xfail real-vs-q comparison documenting the real-space
   symmetrization bug.
2. Decide the real-space fix: either re-symmetrize the dyn/ensemble, or make
   `DynamicalLanczos.prepare_symmetrization` drop non-commuting ops (or use
   non-truncated reps).
3. Update `bug_report_qspace.md` to the final story (q-space conjugation bug
   fixed; real-space projection bias discovered).
