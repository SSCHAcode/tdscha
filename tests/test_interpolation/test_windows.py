"""
Tests for the designed multitaper windows ("stochastic centering") and the
batched slot-resolved kernel (plan sections 5.2-5.5, milestones M3/M4).
"""
import os, sys
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest

import _toy_chain as TC

try:
    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceInterpolation as QI
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

pytestmark = pytest.mark.skipif(not _HAS_Q, reason="QSpaceLanczos/Julia not available")


# ----------------------------------------------------------------------
# Window design (pure numpy, no Julia)
# ----------------------------------------------------------------------
def test_plain_kernel_partition_of_unity():
    """The tent kernel of the plain window must satisfy the partition of
    unity exactly (commensurate limit) for every L."""
    for L in (2, 3, 4, 5, 6):
        S = QI._kernel_sym(np.ones(L), np.ones(L), np.ones(L))
        assert np.max(np.abs(QI._partition_residual(S, L))) < 1e-12, L


@pytest.mark.parametrize("L,K", [(3, 3), (4, 2)])
def test_designed_kernel_is_minimal_image(L, K):
    """The fitted window passes must reproduce the minimal-image target
    kernel (numerically exactly for these sizes) with partition of unity."""
    passes = QI.design_windows_1d(L, K=K)
    target = QI.minimal_image_target_1d(L)
    Ktot = sum(QI._kernel_sym(*p) for p in passes)
    assert np.sqrt(np.mean((Ktot - target) ** 2)) / L < 1e-6
    assert np.max(np.abs(QI._partition_residual(Ktot, L))) / L < 1e-6


def test_L2_design_is_plain():
    """For L=2 every nonzero difference is a Wigner-Seitz tie: the plain
    window IS the minimal-image kernel and the design must return it."""
    passes = QI.design_windows_1d(2)
    assert len(passes) == 1
    for wv in passes[0]:
        assert np.allclose(wv, 1.0)


# ----------------------------------------------------------------------
# Kernel equivalences (Julia)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_system():
    dyn = TC.build_dyn(3)
    ens = TC.make_ensemble(dyn, 300.0, 150, seed=9, g3=0.2, g4=0.3)
    return dyn, ens


def test_batched_equals_scalar_kernel(small_system):
    """The BLAS-3 batched slot kernel must reproduce the scalar one to
    machine precision on random inputs (windowed fields, both flags on)."""
    import tdscha.JuliaExt as JuliaExt
    dyn, ens = small_system
    li = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, 6),
                                window_design="minimal_image")
    li.init(use_symmetries=True)
    li.prepare_mode_q(1, 4)   # non-TRI interpolated q

    jl = JuliaExt.get_main()
    rs = np.random.RandomState(3)
    nb = li.n_bands
    R1 = rs.randn(nb) + 1j * rs.randn(nb)
    n_al = len(li.unique_pairs) * nb ** 2
    alpha1 = rs.randn(n_al) + 1j * rs.randn(n_al)

    up = np.array(li.unique_pairs, dtype=np.int32) + 1
    vm = np.array(li.valid_modes_q, dtype=np.bool_)
    iz, iw, iv = li._window_passes[0]
    fz, fw, fv = (li._field_sets[i] for i in (iz, iw, iv))
    args = (fz[0], fz[1], fw[0], fw[1], fv[0], fv[1],
            li.w_q, li.rho, R1, alpha1, float(li.T), True, True,
            int(li.iq_pert) + 1, up, 1, int(li.n_syms_qspace * li.N), vm,
            0.7, 0.5, True)
    r_scalar = np.array(jl.get_perturb_averages_qspace_slots(*args, False))
    r_batch = np.array(jl.get_perturb_averages_qspace_slots(*args, True))
    scale = np.max(np.abs(r_scalar))
    assert np.max(np.abs(r_scalar - r_batch)) < 1e-12 * scale


def test_windowed_trivial_mesh_equals_plain(small_system):
    """On a coarse mesh where every window design is trivial (L <= 2 per
    dimension), the minimal_image path must equal the plain path exactly."""
    dyn2 = TC.build_dyn(2)
    ens2 = TC.make_ensemble(dyn2, 300.0, 150, seed=4, g3=0.2, g4=0.3)
    la = QI.QSpaceLanczosInterp(ens2, fine_mesh=(1, 1, 2), window_design="plain")
    lb = QI.QSpaceLanczosInterp(ens2, fine_mesh=(1, 1, 2),
                                window_design="minimal_image")
    for lanc in (la, lb):
        lanc.init(use_symmetries=True)
        lanc.prepare_mode_q(0, 5)
        lanc.run_FT(5, verbose=False)
    a1, a2 = np.array(la.a_coeffs), np.array(lb.a_coeffs)
    assert np.max(np.abs(a1 - a2) / np.abs(a1)) < 1e-12


def test_origin_shift_is_unbiased(small_system):
    """Origin averaging is a pure variance reduction: with the FULL set of
    origins along the chain, the estimator remains a valid minimal-image
    estimator (finite-N values differ, expectation identical). Sanity: run
    both and require agreement at the statistical level, plus Hermiticity."""
    dyn, ens = small_system
    r = {}
    for orig in (1, 3):
        lm = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, 6),
                                    window_design="minimal_image",
                                    window_origins=orig)
        lm.init(use_symmetries=True)
        lm.prepare_mode_q(1, 4)
        lm.run_FT(10, verbose=False)
        b = np.array(lm.b_coeffs)
        c = np.array(lm.c_coeffs)
        n = min(len(b), len(c))
        assert np.max(np.abs(b[:n] - c[:n]) / np.abs(b[:n])) < 1e-8
        r[orig] = TC.lanczos_effective_freq(lm) - lm.w_q[4, 1]
    # statistically compatible (they share the same data): loose bound
    assert abs(r[1] - r[3]) < 0.5 * max(abs(r[1]), abs(r[3]))


# ----------------------------------------------------------------------
# Physics: windows must beat the plain interpolation
# ----------------------------------------------------------------------
def test_minimal_image_beats_plain():
    """Coarse (1,1,3) -> fine (1,1,6) vs a direct (1,1,6) ensemble, ALL
    modes: the designed windows must reduce the aggregate renormalization
    error well below the plain-window one.

    Reference values at these exact settings (N=4000, 35 steps):
      noise floor (two direct seeds)  ~ 0.103
      plain window                    ~ 0.299
      minimal_image, 3 origins        ~ 0.122
    The full mode set is required: with a small probe subset the comparison
    is statistically unstable (both estimators sit at the noise floor and
    can swap order). Runtime ~6 min; this is the decisive physics test of
    the stochastic-centering machinery.
    """
    T, N, G3, N_STEPS = 300.0, 4000, 0.1, 35
    dync, dynf = TC.build_dyn(3), TC.build_dyn(6)
    ensf = TC.make_ensemble(dynf, T, N, seed=101, g3=G3)
    ensc = TC.make_ensemble(dync, T, N, seed=303, g3=G3)

    ld = QL.QSpaceLanczos(ensf, lo_to_split=None)
    lp = QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, 6), window_design="plain")
    lm = QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, 6),
                                window_design="minimal_image",
                                window_origins=3)
    for l in (ld, lp, lm):
        l.init(use_symmetries=True)

    def renorms(lanc):
        out = {}
        for iq in range(lanc.n_q):
            for band in range(lanc.n_bands):
                if not lanc.valid_modes_q[band, iq]:
                    continue
                lanc.prepare_mode_q(iq, band)
                lanc.run_FT(N_STEPS, verbose=False)
                out[(tuple(np.round(lanc.q_points[iq], 6)), band)] = \
                    TC.lanczos_effective_freq(lanc) - lanc.w_q[band, iq]
        return out

    def agg(r1, r2):
        ks = sorted(set(r1) & set(r2))
        return (sum(abs(r1[k] - r2[k]) for k in ks)
                / sum(abs(r2[k]) for k in ks))

    r_dir = renorms(ld)
    err_p = agg(renorms(lp), r_dir)
    err_m = agg(renorms(lm), r_dir)

    assert err_m < 0.7 * err_p, \
        "windows did not clearly improve: plain {:.3f} vs windows {:.3f}".format(
            err_p, err_m)
    assert err_m < 0.25, \
        "windowed interpolation error too large: {:.3f}".format(err_m)
