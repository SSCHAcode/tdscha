"""Slow SnTe 2x2x2 validation of symmetric-power factor centering."""
import os

import numpy as np
import pytest

import cellconstructor as CC
import cellconstructor.Phonons
import sscha.Ensemble

import tdscha.QSpaceFactorKernel as FK

try:
    import tdscha.QSpaceInterpolation as QI
    import tdscha.QSpaceLanczos as QL
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "test_julia", "data")
SC = (2, 2, 2)


@pytest.fixture(scope="module")
def snte_dyn():
    return CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), 3)


@pytest.fixture(scope="module")
def snte_fits(snte_dyn):
    structure = snte_dyn.structure
    return {
        order: FK.get_factor_fit(structure, SC, order, far=1,
                                 cost_power=1, max_rank=12)
        for order in (3, 4)
    }


@pytest.mark.slow
def test_snte_rank12_kernel_fit(snte_dyn, snte_fits):
    """The real SnTe geometry must retain exact constraints and improve
    substantially on the plain complete-graph assignment."""
    expected = {
        3: (0.488, 0.748, 1019),
        4: (0.630, 0.544, 15827),
    }
    for order, fit in snte_fits.items():
        rel, coverage, ntarget = expected[order]
        assert len(fit.windows) <= 12
        assert fit.diagnostics["max_class_dev"] < 1e-10
        assert fit.diagnostics["n_target_tuples"] == ntarget
        assert fit.diagnostics["rel_residual"] == pytest.approx(rel,
                                                                  abs=0.03)
        assert fit.diagnostics["coverage"] == pytest.approx(coverage,
                                                              abs=0.03)
        assert fit.diagnostics["coverage"] > \
            fit.diagnostics["coverage_plain"] + 0.35
        for win in fit.windows:
            assert win.check_class_sums(SC,
                                        nat=snte_dyn.structure.N_atoms) \
                < 1e-10


@pytest.mark.slow
def test_snte_sampled_constrained_fit(snte_dyn):
    """Exercise the production target stream and sparse exact factors on
    the real SnTe geometry without enumerating its rank-four target."""
    fits = {
        order: FK.get_factor_fit(
            snte_dyn.structure, SC, order, far=1, cost_power=1,
            max_rank=12, fit_mode="constrained", target_mode="sample",
            sample_classes=256, validation_classes=128,
            max_exact_classes=0, seed=8100 + order)
        for order in (3, 4)
    }
    for fit in fits.values():
        diagnostics = fit.diagnostics
        assert diagnostics["target_mode"] == "sample"
        assert diagnostics["constraint_residual"] < 1e-10
        assert diagnostics["train_stream"]["classes_visited"] == 256
        assert diagnostics["validation_stream"]["classes_visited"] == 128
        assert len(fit.windows) <= 12
        assert fit.plain_coeff == pytest.approx(1.0, abs=1e-12)
        for window in fit.windows:
            assert window.class_sum == 0.0
            assert window.check_class_sums(SC,
                                           nat=snte_dyn.structure.N_atoms) \
                < 1e-12


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_Q, reason="Julia ext not available")
def test_snte_2x2x2_to_4x4x4_factor_smoke(snte_dyn, snte_fits):
    """Run the fitted D3/D4 kernel on a genuine off-grid SnTe mode."""
    ensemble = sscha.Ensemble.Ensemble(snte_dyn, 250)
    ensemble.load_bin(DATA_DIR, 1)
    lanc = QI.QSpaceLanczosInterp(
        ensemble, fine_mesh=(4, 4, 4), window_design="factor",
        factor_fit_mode="individual", factor_rank=12, factor_far=1,
        factor_cost=1)
    lanc.init(use_symmetries=True)
    iq = int(np.where((lanc._fine_idx == [1, 0, 0]).all(axis=1))[0][0])
    lanc.prepare_mode_q(iq, 5)
    lanc.run_FT(3, verbose=False)

    a = np.asarray(lanc.a_coeffs)
    b = np.asarray(lanc.b_coeffs)
    c = np.asarray(lanc.c_coeffs)
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(b))
    assert np.all(np.abs(np.imag(a)) < 1e-11)
    n = min(len(b), len(c))
    assert n > 0
    assert np.max(np.abs(b[:n] - c[:n]) /
                  np.maximum(np.abs(b[:n]), 1e-14)) < 1e-7


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_Q, reason="Julia ext not available")
def test_snte_constrained_stream_runtime_smoke(snte_dyn):
    ensemble = sscha.Ensemble.Ensemble(snte_dyn, 250)
    ensemble.load_bin(DATA_DIR, 1)
    lanc = QI.QSpaceLanczosInterp(
        ensemble, fine_mesh=(4, 4, 4), window_design="factor",
        factor_fit_mode="constrained", factor_target_mode="sample",
        factor_sample_classes=256, factor_validation_classes=128,
        factor_max_exact_classes=0, factor_seed=8100,
        factor_rank=12, factor_far=1)
    lanc.init(use_symmetries=True)
    iq = int(np.where((lanc._fine_idx == [1, 0, 0]).all(axis=1))[0][0])
    lanc.prepare_mode_q(iq, 5)
    lanc.run_FT(2, verbose=False)
    b = np.asarray(lanc.b_coeffs)
    c = np.asarray(lanc.c_coeffs)
    assert np.all(np.isfinite(lanc.a_coeffs))
    assert np.all(np.isfinite(b))
    n = min(len(b), len(c))
    assert np.max(np.abs(b[:n] - c[:n]) /
                  np.maximum(np.abs(b[:n]), 1e-14)) < 1e-7
