# Fast Julia loading for tdscha (and python-sscha)

## 1. Problem

`import tdscha` (and `import tdscha.QSpaceLanczos`) takes minutes on machines where
the PyJulia setup is not perfectly matched to the running Python interpreter.

Measured on this machine (micromamba env `sscha`, Python 3.13, Julia 1.12 via juliaup),
with `python -X importtime -c "import tdscha"`:

| Step                                        | Time      |
|---------------------------------------------|-----------|
| `sscha.Ensemble` (imported by tdscha)       | **194 s** |
| `tdscha.DynamicalLanczos` (self)            | 12.8 s    |
| `julia.Main` (Julia runtime boot)           | 2.3 s     |
| everything else (numpy, scipy, CC, ...)     | ~6 s      |
| **Total**                                   | **215 s** |

This cost is paid on *every* Python launch, even if the Julia mode is never used.

## 2. Root causes

### (a) Eager Julia initialization at import time

`Modules/DynamicalLanczos.py` and `Modules/QSpaceLanczos.py` boot the Julia runtime
and `include()` the `.jl` sources at *module import*, inside `try/except` blocks.
(`DynamicalLanczos.py` even contains **three duplicated copies** of the same block,
lines 38–116, plus three duplicated definitions of `is_julia_enabled()`.)
`sscha/Ensemble.py` in python-sscha does exactly the same for `fourier_gradient.jl`,
and tdscha imports `sscha.Ensemble` at the top level — so tdscha pays the sscha cost too.

### (b) PyJulia + PyCall: the libpython coupling

PyJulia talks to Julia through PyCall.jl. PyCall.jl is *compiled against one specific
libpython*. On this machine PyCall was built for `~/anaconda3/.../libpython3.11.so`,
while the interpreter running tdscha is `micromamba/envs/sscha/.../libpython3.13.so`.
The mismatch makes the fast path (`import julia.Main`) fail, and every importer falls
into the legacy workaround:

```python
jl = Julia(compiled_modules=False)
```

With `compiled_modules=False`, Julia is started with `--compiled-modules=no` and
**recompiles PyCall and all its dependencies from scratch on every Python launch**.
That is the 194 s. This is the same class of problem that the `python-jl` launcher
works around (statically linked Python), and it is intrinsic to the PyJulia/PyCall
architecture: any user whose Python is not the exact one PyCall was built for hits it.

### (c) No caching even on the happy path

Even when PyJulia works, `julia.Main.include("tdscha_core.jl")` re-parses and
re-lowers the source at import in every session, for users who may never call
the Julia mode.

## 3. Requirements for the fix

1. `import tdscha` and `import tdscha.QSpaceLanczos` must be fast (~1 s, no Julia boot).
2. Portable: `pip install tdscha` on a fresh computer must give the same result —
   no manual PyCall rebuilding, no `python-jl`, no conda-specific hacks.
3. The Julia runtime must still be available (it is the fastest computation mode,
   and `QSpaceLanczos` requires it).
4. tdscha and python-sscha must share a **single** Julia runtime per process
   (QSpace workflows use sscha's `fourier_gradient.jl` functions and tdscha's
   `tdscha_core.jl`/`tdscha_qspace.jl` functions in the same run).
5. Backward compatible with existing user scripts (`Lanczos`, `QSpaceLanczos`,
   `MODE_FAST_JULIA`, `is_julia_enabled()` keep working) and with MPI runs.

## 4. Design

Two orthogonal changes, applied to both tdscha and python-sscha (main-branch files only):

### 4.1 Replace PyJulia/PyCall with JuliaCall (PythonCall.jl)

[JuliaCall](https://juliapy.github.io/PythonCall.jl/) (`pip install juliacall`) is the
modern replacement for PyJulia:

* **No libpython coupling.** PythonCall.jl does not compile against a specific
  libpython, so there is no `compiled_modules=False` fallback and no `python-jl`.
  It works with any Python (conda, micromamba, system, statically linked).
* **Self-bootstrapping.** Through `juliapkg`, JuliaCall finds an existing Julia
  (e.g. juliaup) or **downloads and installs Julia automatically** on first use.
  A `juliapkg.json` file shipped inside the `tdscha` package declares the required
  Julia version; nothing else is needed on a fresh computer.
* **Native precompilation works.** PythonCall.jl is precompiled once per machine
  into Julia's package cache (pkgimages). The one-time setup (~3 min, automatic)
  is paid at first use, never again — unlike PyJulia's broken-path recompile
  on *every* launch.

Validated on this machine (Julia 1.12, juliacall 0.9.35):

| Step                                   | Time              |
|----------------------------------------|-------------------|
| one-time setup (first ever boot)       | 189 s (once per machine, automatic) |
| warm boot `from juliacall import Main` | 9.3 s             |
| `include("tdscha_core.jl")`            | 2.1 s             |
| `include("tdscha_qspace.jl")`          | 1.0 s             |

### 4.2 Lazy initialization through a single bridge module

All Julia access goes through a new module, `tdscha/JuliaExt.py` (mirrored as
`sscha/JuliaExt.py` in python-sscha). Nothing Julia-related happens at import time.

```
tdscha/JuliaExt.py
------------------
available()        -> bool   # cheap: importlib.util.find_spec, no Julia boot
get_main()         -> proxy  # boots Julia lazily on FIRST call, includes the
                             # package .jl files exactly once, returns a Main proxy
JuliaError                   # informative ImportError subclass with install hints
```

* `is_julia_enabled()` keeps existing semantics ("can I select `MODE_FAST_JULIA`?")
  but now answers from `available()` without booting Julia. The boot happens at the
  first real use (`Lanczos.prepare_symmetries`, `QSpaceLanczos` calls, ...).
* Thread-safe and idempotent (`threading.Lock`, cached state).
* If initialization fails, the error is cached and re-raised with a clear message
  (`pip install juliacall`), instead of today's silent `except: pass`.

#### Backend selection (interop with legacy installs)

One Julia runtime per process is mandatory (two libjulia initializations crash).
The bridge picks the backend in this order:

1. `SSCHA_JULIA_BACKEND` env var (`juliacall` | `pyjulia` | `none`) if set.
2. If PyJulia is **already initialized** in this process (`"julia.Main"` in
   `sys.modules` — e.g. an old python-sscha booted it first), reuse `julia.Main`.
3. Otherwise prefer `juliacall`, falling back to PyJulia if juliacall is absent.

Since both python-sscha and tdscha use the same rules, both packages converge on the
same runtime, and all functions live in the same `Main` namespace exactly as today.

#### Argument/return conversion (juliacall backend)

PyJulia copies numpy arrays into Julia `Array`s; juliacall instead passes a no-copy
`PyArray` wrapper, which **does not dispatch** on the strictly-typed signatures used
by `tdscha_core.jl` (e.g. `X::Matrix{T}`, `n_degeneracies::Vector{Int32}`). The
proxy returned by `get_main()` restores PyJulia call semantics (all validated):

| Python value                      | Conversion                                          |
|-----------------------------------|-----------------------------------------------------|
| `np.ndarray` (any strides/order)  | `juliacall.convert(Main.Array, x)` → `Array{T,N}`, same shape |
| `list`/`tuple` of same-dtype 1-D arrays | `juliacall.convert(Vector{Vector{T}}, x)` (needed by `init_sparse_symmetries_qspace`) |
| scalars (incl. numpy scalars)     | juliacall default (Float64/Int64/Bool/...)          |
| keyword arguments                 | converted with the same rules (python-sscha's `_wrapper_julia_*` helpers forward `**kwargs`) |
| returned Julia array              | `np.asarray(wrapper)` (buffer-protocol view, keeps owner alive) |
| returned Julia tuple              | converted element-wise (juliacall yields Python tuples) |

`proxy.eval(code)` maps to `Main.seval(code)` (juliacall) or `Main.eval(code)`
(PyJulia), and `proxy.include(path)` likewise — so the dynamically-defined helpers
in `QSpaceLanczos` keep working on both backends.

#### Threads and signals

**Implementation finding:** the kernels in `tdscha_core.jl` use
`Threads.@threads` loops that write into shared buffers allocated *outside*
the loop (`d2v_dR2`, `r1_aux`, `forces`, ...). They are **not thread-safe**:
they only ever produced correct results because PyJulia defaulted to a single
Julia thread. A first version of the bridge defaulted to `threads=auto` and
`tests/test_julia/test_julia.py` immediately caught the data race (wrong
continued-fraction value). The bridge therefore reproduces the PyJulia
defaults exactly. Before the first `import juliacall` it sets (only if unset,
so users stay in control):

* `PYTHON_JULIACALL_THREADS` = `$JULIA_NUM_THREADS` if defined, else `1`.
* `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` only when more than one thread is
  requested (required for Julia multithreading from Python, per PythonCall docs).

Note that `JULIA_NUM_THREADS > 1` produced racy results with PyJulia too:
this is a pre-existing bug of the kernels, see §7.

MPI: unchanged. Each rank lazily boots its own runtime when (and only when) it
calls a Julia function, exactly as each rank booted PyJulia before.

### 4.3 Packaging

* `Modules/juliapkg.json` (installed as `tdscha/juliapkg.json`): declares the
  required Julia version (`"julia": "^1.10"`). `SparseArrays`, `LinearAlgebra` and
  `InteractiveUtils` are Julia stdlibs — the old `Pkg.add(...)` fallback blocks are
  deleted.
* `pyproject.toml`: new optional extra `tdscha[julia]` → `juliacall`. (The runtime
  works without it; `MODE_FAST_SERIAL`/`MPI` need no Julia.)
* `meson.build`: install `Modules/JuliaExt.py` and `Modules/juliapkg.json`.
* python-sscha (companion patch, main-branch files only): same bridge as
  `sscha/JuliaExt.py`. `Ensemble.py` has ~28 `julia.Main.*` call sites; instead
  of touching each, the eager block is replaced by a lazy stand-in that keeps
  the existing syntax working unchanged:

  ```python
  class _LazyJuliaModule(object):
      @property
      def Main(self):
          return JuliaExt.get_main()   # boots Julia on first access

  julia = _LazyJuliaModule()
  __JULIA_EXT__ = JuliaExt.available()   # deprecated alias, no boot
  ```

  Because `__JULIA_EXT__` keeps its meaning (availability), `SchaMinimizer.py`
  — which reads `Ensemble.__JULIA_EXT__` to set `use_julia` — needs **no
  change at all**. The same applies to `Ensemble.fourier_gradient`, whose
  default is now "a backend is installed" rather than "the runtime booted":
  the boot happens inside `ensemble.init()` at the first Fourier-transform
  call.

## 5. Resulting behavior matrix

| Scenario                                            | Import time | First Julia use |
|-----------------------------------------------------|-------------|-----------------|
| Fresh machine, `pip install tdscha[julia]`          | ~6 s        | one-time ~3 min auto-setup, then ~12 s |
| Same machine, later sessions                        | ~6 s        | ~12 s (boot+include), then native speed |
| Julia never used (`MODE_FAST_SERIAL`/`MPI`)         | ~6 s        | — (never boots)  |
| Legacy: old sscha already booted PyJulia            | as before   | reuses `julia.Main`, no double runtime |
| No julia/juliacall installed                        | ~6 s        | clear `ImportError` with install hint |

### Measured after implementation (this machine)

| Quantity                              | Before     | After      |
|---------------------------------------|------------|------------|
| `import tdscha`                       | **215 s**  | **6.4 s**  |
| `import tdscha.QSpaceLanczos` (after `import tdscha`) | included above | < 1 ms |
| `is_julia_enabled()`                  | True       | True (no boot) |

The residual ~6 s is numpy/scipy/matplotlib/ase/cellconstructor/mpi4py import
time, unrelated to Julia. juliacall keeps its Julia project in
`<env>/julia_env/` (resolved automatically from the shipped `juliapkg.json`).

`tests/test_julia/test_julia.py` and `tests/test_julia/test_julia_wigner.py`
pass with the juliacall backend (numerical agreement with the reference
continued-fraction values).

## 6. Files changed

tdscha (this repo):
* `Modules/JuliaExt.py` — new bridge (lazy init, backend selection, conversions).
* `Modules/DynamicalLanczos.py` — delete the 3 duplicated eager blocks and duplicated
  `is_julia_enabled()`; route the 2 Julia call sites through the bridge.
* `Modules/QSpaceLanczos.py` — delete eager block; route the ~7 call sites through
  the bridge; error messages no longer mention `python-jl`.
* `Modules/juliapkg.json` — new.
* `meson.build`, `pyproject.toml` — packaging.

python-sscha (companion patch, branch `fast_load_julia`, main-branch files only):
* `Modules/JuliaExt.py` — new (same bridge, includes `fourier_gradient.jl`).
* `Modules/Ensemble.py` — eager block replaced by the `_LazyJuliaModule`
  stand-in (§4.2); all `julia.Main.*` call sites untouched.
* `Modules/juliapkg.json` — new.
* `meson.build`, `pyproject.toml` — packaging (`python-sscha[julia]` extra).
* `Modules/SchaMinimizer.py` — **unchanged** (the `__JULIA_EXT__` alias keeps
  its semantics).

## 7. Risks and future work

* **Pre-existing thread-safety bug in the Julia kernels** (found during this work):
  the `Threads.@threads` loops in `tdscha_core.jl` (e.g.
  `get_d2v_dR2_from_Y_pert_sym_fast`) accumulate into buffers shared across
  threads, so running with `JULIA_NUM_THREADS > 1` silently produces wrong
  numbers — with PyJulia as well as with juliacall. The bridge defaults to one
  thread, which is safe. Future fix: per-thread accumulators (or
  `OhMyThreads.jl`/chunked reduction), after which the default can become
  `auto` and the multithreaded Julia mode actually delivers its speedup.
* **First-call JIT latency** (~seconds per Julia function on first call in a session)
  is unchanged. Future work: move `tdscha_core.jl`/`tdscha_qspace.jl` into a proper
  Julia package with `PrecompileTools` workloads so pkgimages cache the compiled
  methods across sessions, cutting warm start to ~2–3 s.
* **PyJulia coexistence**: if a *new* tdscha boots juliacall and an *old* sscha later
  tries to boot PyJulia in the same process, the two runtimes conflict. Mitigated by
  releasing the python-sscha companion patch together with tdscha and by backend
  rule 2 (reuse PyJulia if it is already up).
* Numerical equivalence is enforced by the existing test suites
  (`tests/test_julia`, `tests/test_lanczos_fast`), which run the Julia mode.
  These tests caught the threading regression during development (§4.2),
  confirming they exercise the bridge end-to-end.

## 8. Full test-suite results and analysis of the failing tests

Full run after the migration, with both patched packages installed
(`OMP_NUM_THREADS=1 pytest -m "not release"`, 2026-06-12):

```
10 failed, 118 passed, 25 skipped in 753.80s (0:12:33)
```

All 118 previously passing tests still pass, including every Julia-mode test
(`tests/test_julia/`, `tests/test_lanczos_fast/`, most of `tests/test_qspace/`).
The 10 failures are all confined to `tests/test_qspace/` and **none of them is
caused by this migration**. They split into two groups.

### 8.1 Missing reference data (3 failures, environmental)

| Test | Error |
|------|-------|
| `test_gold_lanczos_gf.py::test_gold_lanczos_gf_most_anharmonic` | `ValueError: Error, file .../Examples/ensemble_gold/dyn_gen_pop1_1 does not exist.` |
| `test_gold_nontri.py::test_gold_force_parseval` | same missing file |
| `test_gold_nontri.py::test_gold_hessian_nontri` | same missing file |

All three abort in `CC.Phonons.Phonons(...)` while loading
`Examples/ensemble_gold/dyn_gen_pop1_*`: the gold ensemble directory is not
present in this working copy (`Examples/ensemble_gold/` is not on disk). The
tests never reach any Julia code, so they cannot be affected by the bridge.
They would fail identically on the unpatched branch.

### 8.2 Pre-existing q-space kernel bugs (7 failures, under active debugging)

| Test | Symptom |
|------|---------|
| `test_qspace_anharmonic_invariants.py::TestFpertReality::test_with_off_diagonal` | `f_pert has Im=1.35e+00 with off-diagonal pairs` (expected `< 1e-12`) |
| `test_qspace_anharmonic_invariants.py::TestFpertReality::test_off_diagonal_d4_only` | same invariant violated |
| `test_qspace_anharmonic_invariants.py::TestDiagonalD2vHermitian::test_mixed_pairs` | hermiticity of diagonal d2v blocks violated |
| `test_qspace_anharmonic_invariants.py::TestDiagonalD2vHermitian::test_d4_only_mixed` | same |
| `test_qspace_anharmonic_invariants.py::TestFlagGating::test_R1_zero_gives_d4_only_off_diagonal` | D3/D4 flag gating inconsistent for off-diagonal pairs |
| `test_qspace_hessian_1d.py::test_compare_real_vs_qspace_hessian` | real-space vs q-space Hessian mismatch |

These are physics-invariant checks on `get_perturb_averages_qspace`
(`tdscha_qspace.jl`) with synthetic inputs: with purely real inputs the
perturbation average `f_pert` must be real and the diagonal `d2v` blocks
Hermitian, and they are not whenever **off-diagonal (q1 ≠ q2) pairs** are
involved.

Evidence that these failures pre-date the migration and are independent of it:

1. **Backend cross-check (decisive).** The failures were reproduced with the
   legacy backend by forcing `SSCHA_JULIA_BACKEND=pyjulia`:
   `TestFpertReality::test_with_off_diagonal` fails with the **bit-identical**
   value `Im = 1.3502495550836942` under both pyjulia and juliacall, and
   `test_qspace_hessian_1d.py` fails identically under pyjulia as well. If the bridge's
   argument/return conversion were corrupting data, the two backends — which
   use completely different conversion machinery (PyCall copy vs
   `juliacall.convert` + `np.asarray`) — would not agree to the last bit.
2. **The bug is already documented on this branch.** `bug_hunting.md`
   (in-flight debugging notes in the repo root) describes exactly this defect:
   *"Suspect #3: `f_pert` D3 Contribution from Off-Diagonal Pairs"* — the D3
   accumulation `f_pert += w1 * y_pert; f_pert += w2 * f_Y[:, iq_pert] * x_pert`
   picks up a spurious imaginary part through the off-diagonal pair pathway
   (and a second pathway via Suspect #1 contaminates the Hessian results,
   which is consistent with the `test_qspace_hessian_1d` failure).
3. **The failing test files are themselves part of that debugging effort**:
   `tests/test_qspace/test_qspace_anharmonic_invariants.py` and the gold
   regression tests are untracked, in-flight files written to pin the bug
   down; they are not part of any previously green CI state.

Conclusion: the bit-identical cross-backend agreement turns these failures
into additional evidence that the bridge is numerically faithful. The actual
kernel bug in the off-diagonal pair handling of `tdscha_qspace.jl` is tracked
separately in `bug_hunting.md` and must be fixed independently of this work.

One incidental change was made to
`tests/test_qspace/test_qspace_anharmonic_invariants.py` (untracked file): its
module header used the old eager-PyJulia boilerplate, which inside a single
pytest process would have tried to start a second Julia runtime next to
juliacall (and on this machine costs ~100 s per boot through the
`compiled_modules=False` fallback). It now obtains the runtime from
`tdscha.JuliaExt` like the production code; the tests it contains fail before
and after this change for the reasons above.
