# Q-space Lanczos is corrupted by NumPy 1.26.4 built on Python 3.14

## Summary

`QSpaceLanczos.run_FT` can silently corrupt its Krylov vectors when tdscha is
run with NumPy 1.26.4 built from source on Python 3.14. A nominally
out-of-place masked product can reuse the input storage and mutate the vector:

```python
before = L_q.copy()
weighted = L_q * mask_dot

np.shares_memory(L_q, weighted)       # True in the failing environment
np.max(np.abs(L_q - before))          # 5.8e-8, not zero
```

The first Hermitian Lanczos step then has `b != c`, and subsequent
coefficients grow exponentially. The resulting spectrum is wrong without an
exception or warning.

This should be fixed independently on `main`, even though a defensive
workaround currently exists on the interpolation development branch.

## Affected environment

The failure has been reproduced with:

- Python 3.14.5
- NumPy 1.26.4, built locally from the PyPI source distribution
- tdscha q-space Lanczos through JuliaCall

It has **not** been reproduced with the same code, input, Python interpreter,
and Julia backend after placing NumPy 2.4.6 first on `PYTHONPATH`.

The evidence does not establish that every NumPy version below 2 is affected.
NumPy 1.26 predates Python 3.14, so the immediately actionable compatibility
bug is that tdscha currently permits this unsupported combination:

```toml
requires-python = ">=3.8"
dependencies = ["numpy", ...]
```

The build requirement is similarly only `numpy>=1.20.0`.

## Impact

The problem was found in a full CsSnI3 Raman calculation, but it is not
specific to interpolation. It occurs in the native `4x4x4`
`QSpaceLanczos` metric path before any `8x8x8` interpolation is used.

For a 16-configuration diagnostic, the old product gives
`b0=2.18e-6`, `c0=1.12e-6`. With an explicitly allocated product buffer,
`b=c` within `5e-20`. The corrupted recurrence eventually reaches enormous
coefficients, whereas the correct 100-step production sequence stays around
`1e-6`.

## Reproducer and invariant

The quickest end-to-end regression test is only three Lanczos steps on a
small q-space test ensemble, with neither ensemble reinitialization nor
reorthogonalization:

```python
lanczos = QSpaceLanczos(ensemble, use_wigner=True, lo_to_split=None)
lanczos.init()
lanczos.prepare_raman(unpolarized=0)
lanczos.run_FT(3, verbose=False, reorthogonalize=False)

np.testing.assert_allclose(lanczos.b_coeffs, lanczos.c_coeffs,
                           rtol=1e-11, atol=1e-18)
assert np.all(np.isfinite(lanczos.a_coeffs))
assert np.all(np.isfinite(lanczos.b_coeffs))
```

The test should also exercise the exact masked metric operation and assert
that its right-hand operand is unchanged. The existing small data under
`tests/test_qspace` should make this test fast and portable; the private
CsSnI3 data are not required in CI.

For the original CsSnI3 data, the diagnostic command used was:

```bash
CSSNI3_PROBE_NCONF=16 CSSNI3_PROBE_STEPS=3 \
CSSNI3_PROBE_REORTH=0 CSSNI3_PROBE_REFRESH=0 \
python report/interpolation/scripts/probe_cssni3_lanczos_reorth.py
```

## Root cause and history

The working June calculation and the failing July calculation initially
appeared to use the same tdscha code. A repository and environment audit
confirmed that they did:

1. The correct full-ensemble run is dated 2026-06-14. It used the tip of
   [`fast_julia_startup`](https://github.com/SSCHAcode/tdscha/pull/28), no
   `ensemble.init()`, and no Lanczos reorthogonalization.
2. The pull request only changed lazy Julia startup. Its final commit and
   merge commit contain no q-space numerical change. The matching
   [python-sscha PR](https://github.com/SSCHAcode/python-sscha/pull/419) also
   contains no change that explains the recurrence.
3. On 2026-07-06, installing `deepmd-kit` pulled `mendeleev==0.9.0`, whose
   `numpy<2` constraint downgraded the environment from NumPy 2.4.6 to a
   source build of NumPy 1.26.4. This is the working-to-failing transition.
4. Repeating the alias probe under cached NumPy 2.4.6 gives no shared storage,
   no input mutation, and `b=c=1.1360633e-6`.

An unrelated tdscha commit
[`7be88036`](https://github.com/SSCHAcode/tdscha/commit/7be880361967c7a9682b4f04a1bf852f4de78b74)
changed the `run_FT` default from `reorthogonalize=False` to `True`. That is a
separate regression: it predates the correct June calculation and does not
cause this NumPy-dependent vector mutation.

## Proposed fix

I suggest addressing both the immediate unsafe operation and the unsupported
dependency resolution:

1. Keep masked metric products explicitly out-of-place:

   ```python
   weighted_right = np.empty_like(right)
   np.multiply(right, mask_dot, out=weighted_right)
   value = np.vdot(left, weighted_right)
   ```

2. Add Python-version-aware NumPy lower bounds so that pip cannot resolve
   Python 3.14 to NumPy 1.26. The exact marker should follow the oldest NumPy
   release officially supporting each Python version.
3. Add a CI job for the newest supported Python/NumPy pair and the three-step
   Hermiticity test above.
4. Restore and test the historical `reorthogonalize=False` default separately.

A runtime error for a known unsupported Python/NumPy pair would also be safer
than allowing a calculation to complete with plausible but incorrect data.

## Validation

With NumPy 2.4.6 and the historical settings, a new eight-rank run using the
complete 12,288-configuration CsSnI3 ensemble completed all 100 native
`4x4x4` steps:

- the first 20 coefficients agree with the June reference within `1.1e-18`;
- both runs have `max(abs(b)) = 2.9146089e-6`;
- the rerun has `max(abs(b-c)) = 3.72e-17`;
- the normalized isotropic Raman curves have `L1 = 0.077` even at a stringent
  `2 cm-1` smearing and visually overlay.

This confirms that the old bounded result is reproducible and that the
exponential sequence was an environment-dependent numerical corruption, not
a physical instability or an interpolation effect.

## Update 2026-07-23: `b == c` does NOT detect the corruption

The out-of-place `metric_dot` workaround is **not sufficient**, and the
Hermiticity invariant proposed above is **not a valid acceptance test**.

Running the identical code and the identical full 12,288-configuration
CsSnI3 ensemble, native `4x4x4`, no interpolation, changing only NumPy:

| quantity | NumPy 2.4.6 | NumPy 1.26.4 |
| --- | --- | --- |
| `a[0]` | `-3.4223e-08` | `-3.4223e-08` (identical) |
| `a[1]` | `-1.0827e-06` | `-3.03e-06` |
| `b[0]` | `1.1270e-07` | **`4.526e-04`** |
| `b[1]` | `1.2340e-06` | `1.16e-05` |
| `max(abs(b-c))` | `1.7e-21` | **`0.0`** |

The NumPy 2.4.6 column reproduces `cssni3_native4_full100_numpy246.json` to
four digits. The NumPy 1.26.4 column is wrong, yet `b - c` is **exactly
zero** — cleaner than the correct run. The residual aliasing therefore
corrupts `b` and `c` symmetrically, so it lives in the operator application
or the Julia bridge, not in the metric products that `metric_dot` fixed.

A cheap invariant that *does* catch it: the Lanczos coefficients are bounded
by the spectral range of `L`, whose eigenvalues are two-phonon `omega^2`.
For CsSnI3 the phonon band tops out near `220 cm-1`, so no coefficient can
exceed `(440 cm-1)^2 = 4e-6 Ry^2`. The observed `b[0] = 4.5e-4` is two orders
of magnitude above that ceiling and is unphysical on inspection, whatever
`b - c` says.

Practical consequence: an `8x8x8` interpolated production run completed on
2026-07-23 under NumPy 1.26.4 with finite, `b == c`, 100-step coefficients
that were nonetheless invalid. It is archived under the
`INVALID_numpy1_` prefix in `report/interpolation/data/`. Production
submission now prepends the cached NumPy 2.4.6 to `PYTHONPATH`, forwards it
with `mpirun -x PYTHONPATH`, and `cssni3_raman_interp.check_numpy()` aborts
if the major version is below 2.

## Acceptance criteria

- Python package metadata prevents unsupported Python/NumPy combinations.
- The metric calculation cannot mutate either operand.
- A short q-space Lanczos test produces finite coefficients bounded by the
  two-phonon spectral range. **`b == c` alone is insufficient — the failing
  environment satisfies it exactly.**
- The default value of `reorthogonalize` is covered by a regression test.
