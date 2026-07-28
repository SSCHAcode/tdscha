# CsSnI3 q-space Lanczos investigation

Status (2026-07-22): root cause identified. The reported exponential growth
is real, but it is not an interpolation error and the earlier
`ensemble.init()` explanation was wrong.

## Findings

- **SnTe 4x4x4 validates the interpolation.** CellConstructor `Spectral`
  gives a 37.654 cm-1 peak. Atom-Fourier gives 38.044 cm-1 stochastically and
  37.719 cm-1 with the deterministic tensor (differences 0.390 and 0.065
  cm-1). The normalized line-shape L1 distances are 0.269 and 0.228, so this
  is strong peak agreement, not a pointwise-identical curve.
- The good CsSnI3 reference used neither `ensemble.init()` nor Lanczos
  reorthogonalization. Tests below use the same choices.
- An unrelated commit (`7be88036`) accidentally changed the q-space
  `run_FT` default from `reorthogonalize=False` to `True`. The historical
  default is restored.
- A one-step trace found the immediate numerical failure: evaluating
  `L_q * mask_dot` changed `L_q` itself because the result reused its NumPy
  storage. Thus `p_L = copy(L_q)` stopped matching `L_q`, and already the
  first Hermitian recurrence had `b != c`. Masked products now use an
  explicitly allocated `out=` buffer.
- The same bad coefficients occur on local main, `origin/main`, serial and
  MPI. Therefore this is not a bilinear/interpolation branch regression.

## Historical cause

- The correct run is recorded in `T_500/output_qslanc.log` on 2026-06-14.
  It used the final `fast_julia_startup` code and produced bounded
  coefficients for all 100 steps.
- tdscha PR #28 (`fast_julia_startup`) and python-sscha PR #419 changed Julia
  startup, but their final branch commits and merge commits contain no later
  q-space/ensemble numerical change. They did not introduce this regression.
- The environment used NumPy 2.4.6 for the June run. On 2026-07-06,
  `pip install deepmd-kit` installed `mendeleev==0.9.0` (`numpy<2`) and built
  NumPy 1.26.4 for Python 3.14. This is when the behavior changed.
- Re-running the one-step alias test with the cached NumPy 2.4.6 gives no
  shared storage, no mutation, and `b=c=1.1360633e-6` on 16 configurations.
  NumPy 1.26.4 shares the product storage with `L_q` and mutates it.

Thus the working-to-broken transition was an environment dependency downgrade,
not a tdscha commit. Commit `7be88036` did accidentally change the default
from no reorthogonalization to reorthogonalization, but that predates the June
reference and is a separate mistake.

## Quick test

Run 3-6 steps on 16 or 512 CsSnI3 configurations, with no init and no
reorthogonalization:

```bash
micromamba run -n sscha env \
  CSSNI3_PROBE_NCONF=16 CSSNI3_PROBE_STEPS=3 \
  CSSNI3_PROBE_REORTH=0 CSSNI3_PROBE_REFRESH=0 python \
  report/interpolation/scripts/probe_cssni3_lanczos_reorth.py
```

Before the fix, 16 configurations give `b0=2.18e-6`, `c0=1.12e-6` and the
sequence grows. After the fix, `b=c` to 5e-20. On all 12288 configurations
(8 MPI ranks), steps 3-6 are bounded at 0.66-1.33e-6 rather than growing
exponentially.

A full 100-step, 12288-configuration NumPy 2.4.6 rerun completed and wrote
`report/interpolation/data/cssni3_native4_full100_numpy246.json`. The first
20 coefficients reproduce the June reference within `1.1e-18`; both runs
have `max(abs(b))=2.9146e-6`, and the rerun keeps `b=c` within `3.7e-17`.
Their normalized isotropic Raman curves have L1 distance 0.077 at a stringent
2 cm-1 smearing (Fig. `cssni3_native4_rerun.pdf`). Thus the old bounded result
is reproducible. The claimed fatal interpolation failure is not supported:
SnTe agrees with the oracle, and the CsSnI3 growth came from the native
runtime/metric path.

The clean two-rank, full-ensemble CsSnI3 `4^3 -> 8^3` diagnostic completed
under NumPy 2.4.6 and is numerically consistent, but it is **not a valid
physical interpolation result**. The interpolated auxiliary matrix contains
12 imaginary modes at 10 off-grid q points (worst signed frequency
`-12.40 cm-1`). `allow_unstable=True` excluded every two-phonon channel
containing those modes; it did not stabilize them. Applying the real-space
ASR creates 8 of the 12 instabilities, while 4 remain without ASR. The
apparent 2 cm-1 agreement (`L1=0.033`) therefore compares the reference with
an incomplete Hilbert space. Low-smearing peaks are additionally unconverged
with 100 Lanczos steps. Future production runs now fail on imaginary fine
modes by default; masking is diagnostic only.

## RESOLVED 2026-07-23: the 12 imaginary modes were the inherited effective charges

The "12 imaginary modes at 10 off-grid q points" above are an interpolation
artifact, not a property of the CsSnI3 auxiliary dynamical matrix. Full
analysis in `benchmark.md` section 19 and report sec:cssni3-longrange.

`ForceTensor.Tensor2.SetupFromPhonons` subtracts the Ewald dipole-dipole term
when `dyn.effective_charges` is set, centers/ASR-projects only the remainder,
and `Interpolate` re-adds it. `lo_to_splitting=False` does NOT disable this.
The cycle is exactly the identity at commensurate q, hence invisible to every
regression test, and acts only off-grid. CsSnI3's forces come from a
short-range ML force field with no electrostatics, so the stored Z*/eps are
inherited DFT metadata about the material, not the potential; subtracting a
tail that is not in the data leaves a long-ranged remainder that centering
truncates and the ASR then mangles.

With `ignore_effective_charges=True`:
- imaginary modes on the full 8^3 mesh: **0** (min signed freq +7.8e-7 cm-1,
  the Gamma acoustic zero), with AND without ASR;
- the centered FCs already satisfy the ASR to 2.1e-10 -> the coarse dyn was
  well converged all along, and the "1.98% ASR FC change" was measuring the
  truncation error of a dipole tail that does not exist;
- max commensurate frequency residual 1.8e-6 cm-1, versus **19.5 cm-1** for
  the Z*+ASR path, which therefore does not even reproduce its own input
  (hidden in production by `reuse_commensurate=True`).

So the two hypotheses left open above ("soft branch under-resolved" /
"genuine instability between sampled q points") are both excluded, and the
`allow_unstable` masking is no longer needed for this system. Audit script
`report/interpolation/scripts/cssni3_longrange_audit.py`; corrected dispersion
`cssni3_dispersion_longrange.py`. Regression tests
`tests/test_interpolation/test_ignore_effective_charges.py`.

## CORRECTED production run 2026-07-23

The full 12288-config 8^3 Lanczos reran with `ignore_effective_charges=True`
and `allow_unstable=False` (0 imaginary modes confirmed on the production
path). Corrected low-energy comparison: L1(4 vs 8) = 0.028/0.078/0.160/0.249
at 2.0/1.0/0.5/0.25 cm-1, and **W8/W4 = 1.005-1.009 (weight conserved)** vs
0.984-0.9999 for the masked run (which lost weight by dropping channels). Fig
`figs/cssni3_full_lowenergy_noEC.pdf`, report fig:cssni3-lowenergy-noEC.

Separately uncovered: the env's NumPy 1.26.4 silently corrupts the q-space
Lanczos and `b == c` does NOT catch it (b-c is exactly 0 while b0 is 100x too
large). First rerun was invalid because of this; fixed by
`QSpaceLanczos.check_numpy_version()` (raises under NumPy < 2 from run_FT) plus
a PYTHONPATH pin to the cached 2.4.6. See numpy1_python314_qspace_issue.md.
