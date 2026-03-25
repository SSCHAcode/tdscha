# Release Notes for TDSCHA v1.6.1

**Release Date:** March 25, 2026

This release introduces major performance optimizations and new computational capabilities, most notably the **q-space implementations** that dramatically accelerate calculations for periodic systems.

---

## Major New Features

### 1. QSpaceLanczos: Q-Space Spectral Calculations

A new module implementing the Lanczos algorithm in the Bloch (q-space) basis, exploiting momentum conservation for massive performance gains.

**Key Features:**
- **Momentum Conservation:** The 2-phonon sector is block-diagonal with pairs satisfying **q₁ + q₂ = q_pert**, reducing the ψ-vector size by ~N_cell compared to real-space
- **Hermitian Lanczos:** Uses complex arithmetic with Hermitian inner products for proper convergence
- **Julia Backend:** High-performance implementation via `Modules/tdscha_qspace.jl`
- **Bloch-Transformed Ensemble:** Direct q-space ensemble averaging without real-space supercell overhead

**Performance Impact:**
- Memory reduction: O(N_modes²) → O(n_bands²) for the 2-phonon sector
- Speedup scales as ~N_cell for spectral calculations
- Essential for large supercells where real-space Lanczos becomes prohibitive

**Usage:**
```python
import tdscha.QSpaceLanczos as QL

# Initialize q-space Lanczos solver
qlanc = QL.QSpaceLanczos(ensemble)
qlanc.init(use_symmetries=True)

# Compute spectral function at specific q-point
qlanc.prepare_q(0)  # Gamma point
qlanc.run_FT(n_steps=1000)
green = qlanc.get_green_function_continued_fraction(energies)
```

**Files Added:**
- `Modules/QSpaceLanczos.py` (1,232 lines)
- `Modules/tdscha_qspace.jl` (529 lines)
- `tests/test_qspace/test_qspace_lanczos.py` (273 lines)

---

### 2. QSpaceHessian: Fast Free Energy Hessian

The **fastest and most memory-efficient** method in the SSCHA ecosystem for computing the full anharmonic free energy Hessian, including fourth-order contributions.

**Key Features:**
- **Block-Diagonal Structure:** Translational symmetry makes the Liouvillian block-diagonal in q-space
- **Iterative Solver:** GMRES with harmonic preconditioner; BiCGSTAB fallback for difficult cases
- **Symmetry Exploitation:** Point-group symmetries reduce computation to the irreducible q-set
- **Hermiticity Enforcement:** Explicit symmetrization ensures real eigenvalues
- **Anharmonic Control:** `ignore_v3` and `ignore_v4` flags to exclude cubic/quartic terms

**Performance Impact:**

| Metric | Direct (`get_free_energy_hessian`) | **QSpaceHessian** |
|--------|-----------------------------------|-------------------|
| Memory | O(N_modes⁴) | O(n_bands⁴) |
| Time | O(N_modes⁶) | O(n_bands³ × N_q,irr) |
| Speedup (4×4×4 cell) | 1× | **~65,000×** |

For a 4×4×4 supercell (N_cell = 64, N_q,irr ≈ 4), calculations that would require **terabytes of RAM** are now feasible on a laptop.

**Usage:**
```python
import tdscha.QSpaceHessian as QH

# Compute full Hessian at all q-points
hess = QH.QSpaceHessian(ensemble)
hess.init()
hessian_dyn = hess.compute_full_hessian(tol=1e-6, max_iters=500)

# Or compute at single q-point
H_gamma = hess.compute_hessian_at_q(iq=0)
```

**Advanced Features:**
- **Dense Fallback:** For problematic q-points near phase transitions (`dense_fallback=True`)
- **Acoustic Mode Handling:** Proper null-space projection for non-self-conjugate q-points
- **Mode Degeneracy Exploitation:** Schur's lemma reduces GMRES solves for degenerate modes

**Files Added:**
- `Modules/QSpaceHessian.py` (701 lines)
- `tests/test_qspace/test_qspace_hessian.py` (170 lines)
- `docs/qspace-hessian.md` (comprehensive documentation)

---

### 3. Gamma-Only Optimization: 8× Speedup for Γ-Point Calculations

Major performance enhancement for Γ-point (q = 0) spectral and response calculations.

**Optimizations:**
- **Reduced Python-Julia Bridge Overhead:** Minimized data marshalling for gamma-only mode
- **O(n²) Permutations:** Replaced O(n³) matrix multiplications with direct index permutations in `apply_L_translations`
- **Symmetry-Aware:** Exploits point-group symmetries for additional speedup

**Impact:**
- 8× speedup for gamma-point IR/Raman response calculations
- Particularly effective for large supercells where the Python-Julia boundary crossing was a bottleneck
- Maintains full accuracy (verified against non-optimized implementation)

**Usage:**
The optimization is automatic when:
1. The perturbation is at the Γ-point (`q_pert = 0`)
2. `use_symmetries=True` is set
3. The Julia backend is active (`MODE_FAST_JULIA`)

**Files Modified:**
- `Modules/DynamicalLanczos.py` (gamma-only detection and optimization)
- `Modules/tdscha_core.jl` (translational projection optimization)

---

## Additional Improvements

### Documentation Overhaul
- **New Documentation Site:** Full mkdocs-based documentation at [sscha.eu/tdscha](http://sscha.eu/tdscha/)
- **API Reference:** Auto-generated from docstrings
- **Theory Guide:** Mathematical background on q-space methods
- **Examples:** SnTe spectral function at X point, IR response calculations

### Command-Line Interface
New executable scripts for common workflows:
- `tdscha-convergence-analysis` — Analyze Lanczos convergence
- `tdscha-plot-data` — Plot spectral functions
- `tdscha-output2abc` — Convert output formats
- `tdscha-hessian-convergence` — Monitor Hessian convergence

### Testing & Validation
- **Q-Space Correctness Tests:** Verify q-space vs. real-space Green functions match within 2.4×10⁻⁷ relative error
- **Wigner Representation Tests:** Compare Wigner vs. non-Wigner spectral functions
- **MPI Parallel Tests:** CI-validated MPI correctness
- **IR Perturbation Tests:** Modulus consistency between real-space and q-space Lanczos

### Bug Fixes
- **Q-Point Mapping:** Fixed errors for q-points where `-q` does not map to itself
- **Symmetry Construction:** Corrected point-group symmetry handling in q-space (Cartesian rotations, atom permutations)
- **Acoustic Modes:** Proper masking of acoustic modes (ω < 10⁻⁶) in ensemble averages
- **Version Synchronization:** Aligned version numbers across pyproject.toml, meson.build, and git tags

---

## Breaking Changes

None. This release is fully backward-compatible with v1.5.

---

## Migration Guide

### For Existing Users

No code changes required. Existing scripts will continue to work unchanged.

To benefit from new features:

1. **For faster Hessian calculations:** Replace `ensemble.get_free_energy_hessian()` with `QSpaceHessian`
2. **For large-supercell spectra:** Use `QSpaceLanczos` instead of `Lanczos` when memory is constrained
3. **For Γ-point response:** Ensure `use_symmetries=True` to activate 8× speedup automatically

### Dependencies

No new required dependencies. Optional dependencies:
- `spglib` — For symmetry exploitation (recommended)
- `julia` — Required for q-space modules and gamma-only optimization
- `mpi4py` — For MPI parallelization

---

## Contributors

- Lorenzo Monacelli (main development)
- Claude Opus 4.6 (q-space implementation assistance)

---

## References

1. R. Bianco, I. Errea, L. Paulatto, M. Calandra, and F. Mauri, *Phys. Rev. B* **96**, 014111 (2017) — Free energy Hessian theory
2. L. Monacelli, *Phys. Rev. B* **112**, 014109 (2025) — Static susceptibility formulation
3. L. Monacelli and F. Mauri, 2021 — Q-space Liouvillian block-diagonality

---

## Full Changelog

See the [git log](https://github.com/SSCHAcode/tdscha/compare/v1.5...v1.6.1) for all 43 commits since v1.5.

**Key Commits:**
- `47ab893c` — Gamma-only implementation with tests
- `31263868` — 8× speedup via reduced Python-Julia overhead
- `1a65d466` — QSpaceLanczos module with Bloch momentum conservation
- `b180a88f` — QSpaceHessian for free energy Hessian in q-space
- `7cd005ef` — Mode degeneracy exploitation (Schur's lemma)
- `a7ddc302` — Hermitian symmetry exploitation in q-pair blocks
- `ed693e2c` — Raman perturbation methods for QSpaceLanczos
