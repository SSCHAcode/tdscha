# Atom-Fourier interpolation polish plan

This checklist is the source of truth for preparing the pull-request branch.
It must be updated whenever a task starts or finishes.

## Scope

- Keep a single interpolation strategy: `atom_fourier`.
- Support both third- and fourth-order tensors (`d3` and `d4`).
- Share mesh, Fourier, validation, and reconstruction helpers wherever
  the tensor orders have the same mathematical operation.
- Remove experimental APIs, implementations, tests, and documentation from the
  pull-request branch. They remain recoverable from
  `backup/qspace-interpolation-attempts` at commit `0ceecd59`.
- Do not add report outputs, datasets, PDFs, caches, or generated test files.

## TODO

- [x] Preserve the pre-polish source and tests in a snapshot commit.
- [x] Create `backup/qspace-interpolation-attempts` at the snapshot.
- [x] Create and switch to the `atom-fourier-interpolation` branch.
- [x] Inventory interpolation implementations, public entry points, call sites,
      and tests.
- [x] Define the smallest user-facing API for `atom_fourier` d3/d4
      interpolation, including validation and error messages.
- [x] Refactor common mesh, centered-image, Fourier-transform,
      and reconstruction logic into order-independent helpers.
- [x] Retain only the `atom_fourier` implementation and remove obsolete
      interpolation modes, aliases, branches, and dead helpers.
- [x] Update callers to use the polished API without experimental switches.
- [x] Replace experiment-oriented tests with focused d3/d4 correctness tests.
- [x] Cover edge cases: identity mesh, anisotropic meshes, odd/even meshes,
      Nyquist ties, complex phases, tensor permutation symmetry, invalid shapes,
      and non-commensurate meshes.
- [x] Add concise user-facing API documentation and examples.
- [x] Run targeted interpolation tests and relevant Q-space regressions.
- [x] Review the complete diff for dead code, generated files, accidental report
      changes, naming consistency, and minimality.
- [x] Record final verification commands and results below.

## Verification record

- `pytest -q tests/test_atom_fourier tests/test_interpolation/test_mesh_and_dyn.py
  tests/test_interpolation/test_ignore_effective_charges.py`: 36 passed.
- The same focused run plus restart and distributed-loader tests: 41 passed.
- `pytest -q tests/test_qspace`: 50 passed, 7 skipped.
- `meson setup --reconfigure build && meson compile -C build`: passed.
- `python scripts/validate_docs.py docs/usage.md
  docs/api/qspace_atom_fourier.md`: passed.
- `python -m py_compile` on the interpolation modules: passed.
- `git diff --check`: passed.

## Inventory and API decision

- Remove the windowed, stochastic-centering, tensor, factor-kernel,
  piecewise-trilinear, atomic-phase, and separable-Nyquist implementations.
- Keep `generate_fine_mesh`, mesh-index lookup, and harmonic dynamical-matrix
  Fourier interpolation as shared infrastructure.
- Expose one implementation as
  `tdscha.QSpaceAtomFourier.QSpaceAtomFourierLanczos`; atom-Fourier and the
  true cell-metric image assignment are unconditional, so users do not need
  strategy flags.
- Expose distributed construction as
  `load_distributed_atom_fourier_tdscha`.
- Keep d3/d4 on the same fold/coarse-kernel/adjoint-unfold path, with the
  existing order-specific mesh normalization factors.
