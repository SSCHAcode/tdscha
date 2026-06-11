# Bug: q-space loses the anharmonic renormalisation at non-TRI q-points

**Status:** CONFIRMED on real data.
**Date:** 2026-06-11 (revised)
**Components:** `Modules/QSpaceHessian.py`, `Modules/QSpaceLanczos.py`,
`Modules/tdscha_qspace.jl` — the q-space anharmonic machinery for a perturbation
/ response at a **non-time-reversal-invariant external momentum** `q != -q`.
**Reference (trusted):** `Modules/DynamicalLanczos.py`,
`sscha.Ensemble.get_free_energy_hessian`.

## Summary

When the external momentum of the anharmonic response is **TRI** (`q == -q`, i.e.
Gamma, or a half-BZ point such as X), the q-space code reproduces the real-space
anharmonic renormalisation exactly. When the external momentum is **non-TRI**
(`q != -q`), the q-space code recovers almost none of the anharmonic
renormalisation — the result stays close to the bare SSCHA phonon. This is the
reported "anharmonicity washed out when more q points are present".

The effect is invisible on the 2×2×2 SnTe data because every q-point of a
power-of-two cubic grid is TRI.

## Decisive evidence (static free energy Hessian, real 3×3×3 gold ensemble)

Data: `Examples/ensemble_gold` (50-config converged SSCHA ensemble, 1 atom/cell,
27 q-points, 26 non-TRI; only Gamma is TRI). The static Hessian is used because
it has **no Lanczos-convergence and no `+q` vs real-mode mapping ambiguity**:
both codes compute the same physical d²F/dR².

```
reference = ens.get_free_energy_hessian(include_v4=True)
q-space   = QSpaceHessian.compute_hessian_at_q(iq)
```

Renormalised frequencies (cm⁻¹), error judged on the renormalisation
(Hessian − SSCHA), **not** the absolute frequency:

| q-point            | band | SSCHA | reference (shift) | q-space (shift) | shift captured |
|--------------------|------|-------|-------------------|-----------------|----------------|
| iq=1  (non-TRI)    | 0    | 49.21 | 43.72 (−5.49)     | 49.06 (−0.15)   | **~3%**        |
| iq=1  (non-TRI)    | 1    | 67.31 | 56.50 (−10.81)    | 66.98 (−0.33)   | ~3%            |
| iq=13 (non-TRI)    | 1    | 67.56 | 56.64 (−10.92)    | 67.23 (−0.33)   | ~3%            |

So at non-TRI q the q-space Hessian misses ~97% of the anharmonic shift.

**Note on gold Gamma (not a valid control).** Gold has one atom per cell, so the
only modes at Gamma are the three acoustic modes, which are zero by construction
(and masked). The q-space and reference Hessians therefore "agree" at gold Gamma
trivially (0 = 0); that comparison is meaningless and is NOT evidence that the
TRI case works.

**The meaningful TRI controls** (real, renormalising optical modes at TRI
external momenta) come from the 2×2×2 SnTe data, where the q-space and real-space
results agree even for strongly renormalised modes:
- `test_qspace/test_qspace_hessian.py::test_qspace_hessian_gamma` — the SnTe
  Gamma-*optical* Hessian eigenvalues (which do renormalise) match the reference
  to <5%. Passes.
- SnTe Lanczos, most anharmonic mode (iq=4, a ±23 cm⁻¹ renormalisation, TRI):
  q-space and real-space renormalised frequencies coincide to **0.0%** on the
  shift; and the Gamma-optical / X-point `a/b/c` match to ~1e-7.

So the contrast is: TRI external q with genuine renormalisation → exact
agreement; non-TRI external q → ~97% of the renormalisation lost.

The same trend appears in the Lanczos Green function on gold (renormalised
frequency from `1/Re G(0)`), where the anharmonic shift is 24–52% too small in
q-space — but the Hessian comparison above is the clean, unambiguous
demonstration.

## Why only non-TRI external q (and why it was missed)

- Gamma (`q_ext = 0`, e.g. SnTe Gamma optical): every internal pair `(q1, q2)`
  with `q1 + q2 = 0` has `q2 = -q1`, so by time-reversal `w(q1) = w(q2)`.
  ✔ works (validated on SnTe, which has optical modes; on monatomic gold Gamma is
  only the zero acoustic modes and carries no information).
- A TRI external q (e.g. SnTe X-point): `q_ext == -q_ext`; the perturbed
  coefficients still match real-space to ~1e-7. ✔ works.
- A non-TRI external q (`q_ext != -q_ext`): the internal pairs generally have
  `w(q1) != w(q2)`, **and** the perturbation/response momentum is not its own
  inverse. ✘ the anharmonic renormalisation collapses.

The existing q-space tests only cover Gamma / TRI points:
`tests/test_qspace/test_qspace_hessian.py::test_qspace_hessian_gamma` and the
Gamma/X perturbations in `test_qspace_lanczos.py`. They pass, which is consistent
with the bug being confined to non-TRI external momenta — and is exactly why it
was missed.

## Where to look (root cause — localised, single line not yet pinned)

The error is already present in the **Lanczos** (`QSpaceLanczos`), and the static
Hessian merely inherits and amplifies it — both route the anharmonic average
through `QSpaceLanczos._call_julia_qspace`. Findings:

- It is in the **D3 (third-order) contribution**. The convention-free scalar
  `b_0^2 = ||L_anh v0||^2_mask` for a pure-R perturbation `v0` (which exercises
  only the D3 `d2v`-from-R path) is already wrong on the most anharmonic gold
  mode: `q / real = 0.44` (q-space ~2.3x too small). For gold (FCC, point group
  with inversion) `+q` and `-q` are symmetry-equivalent, so with consistent
  symmetrisation this ratio must be 1 — the 0.44 is a genuine discrepancy, not
  the `+q` vs real-mode mapping.
- Disproved hypotheses: (a) the momentum pairing sign in
  `build_q_pair_map` — the code's partner `q2 = q_pert - q1` and the alternative
  `q_pert + q1` give comparable non-zero D3 three-point correlations, so the
  pairing is not the bug; (b) the off-diagonal-block d2v assembly formula in
  `tdscha_qspace.jl` and the chi± conversion in `apply_anharmonic_FT` match the
  real-space `tdscha_core.jl` term-by-term on inspection.
- So the discrepancy is in the **D3 `d2v` value the q-space produces for a
  non-TRI external momentum** (`QSpaceLanczos.apply_anharmonic_FT` →
  `_call_julia_qspace` → `tdscha_qspace.jl::get_perturb_averages_qspace_fused`),
  but the precise erroneous term has not yet been isolated, because the
  d2v cannot be compared block-by-block across the two codes: the real-space
  code only accepts a *real* supercell perturbation (`+q` and `-q` mixed), so it
  cannot be driven with the pure `+q` Bloch mode the q-space uses.

**Reliable next step to pin the exact line:** instrument
`get_perturb_averages_qspace_fused` (and/or the Python `apply_anharmonic_FT`) to
dump the D3 `d2v` blocks for a non-TRI `iq_pert`, and compare against an
independent direct evaluation of the perturbed `<d2V/dQ dQ>` average built from
the q-space ensemble data (X_q, Y_q) — not from the optimised weight formula —
so the formula itself is checked rather than its re-implementation.

## Tests added

`tests/test_qspace/test_gold_nontri.py`:
- `test_gold_force_parseval` — the gold dyn is well-behaved (X and Y projections
  consistent). **passes**
- `test_gold_hessian_gamma_tri` — Gamma renormalisation matches exactly.
  **passes**
- `test_gold_hessian_nontri` — non-TRI renormalisation must match to 10%.
  **xfail (strict)**: currently the q-space shift is ~97% too small; a correct
  fix turns this into XPASS.

## Method notes

- Judge anharmonic agreement on the *renormalisation* (Hessian/Lanczos minus
  SSCHA), not the absolute frequency: a 0.1 cm⁻¹ absolute difference can be most
  of the physical effect.
- Compare with `use_symmetries=True`; q-space enforces translations via the
  Fourier/primitive-cell construction, the real-space side as explicit space-group
  operations (`n_syms = n_point_group × n_cells`).
- A synthetic dyn built with `GetDynQFromFCSupercell` is NOT a safe test bed here:
  it fails a force-Parseval check (`Σ|Y_q|² != Σ|Y_real|²`) even on TRI grids,
  which contaminates anharmonic comparisons. Use the real gold ensemble.
- Set `JULIA_NUM_THREADS=1` (threaded real-space D4 Julia routine can segfault
  under the PyCall GC).
