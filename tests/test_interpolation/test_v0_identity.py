"""
V0 back-compatibility: QSpaceLanczosInterp with fine_mesh == coarse mesh must
reproduce QSpaceLanczos exactly (same physics, machine precision on the
gauge-invariant Lanczos coefficients), both with and without the field
pre-filter. Also checks the field Parseval identity (raw mode) and that the
vertex rescaling factors are exactly 1 on-grid.
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

T = 300.0
N_CONF = 300
N_STEPS = 6
L = 3


@pytest.fixture(scope="module")
def system():
    dyn = TC.build_dyn(L)
    ens = TC.make_ensemble(dyn, T, N_CONF, seed=7, g3=0.2, g4=0.3)
    return dyn, ens


def _run(lanc, iq, band):
    lanc.init(use_symmetries=True)
    lanc.prepare_mode_q(iq, band)
    lanc.run_FT(N_STEPS, verbose=False)
    return np.array(lanc.a_coeffs), np.array(lanc.b_coeffs)


@pytest.mark.parametrize("prefilter", [False, True])
def test_v0_coefficients_identical(system, prefilter):
    dyn, ens = system
    lc = QL.QSpaceLanczos(ens, lo_to_split=None)
    li = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, L), prefilter=prefilter)

    assert li.qspace_scale3 == 1.0 and li.qspace_scale4 == 1.0

    # Gamma optical mode and a non-Gamma point, full anharmonicity
    for iq_c, band in [(0, 5), (1, 2)]:
        iq_f = li.find_fine_q(lc.q_points[iq_c])
        a_c, b_c = _run(lc, iq_c, band)
        a_i, b_i = _run(li, iq_f, band)
        n = min(len(a_c), len(a_i))
        assert n > 2
        assert np.max(np.abs(a_c[:n] - a_i[:n]) / np.abs(a_c[:n])) < 1e-10, \
            "prefilter={} iq={} band={}".format(prefilter, iq_c, band)
        m = min(len(b_c), len(b_i))
        assert np.max(np.abs(b_c[:m] - b_i[:m]) / np.abs(b_c[:m])) < 1e-10


def test_v0_field_parseval(system):
    """Raw-mode fields at commensurate q must carry identical spectral
    weight as the standard coarse construction (per q-point)."""
    dyn, ens = system
    lc = QL.QSpaceLanczos(ens, lo_to_split=None)
    li = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, L), prefilter=False)

    for iq_c in range(lc.n_q):
        iq_f = li.find_fine_q(lc.q_points[iq_c])
        v_c = lc.valid_modes_q[:, iq_c]
        v_f = li.valid_modes_q[:, iq_f]
        assert np.sum(v_c) == np.sum(v_f)
        for A_c, A_f in [(lc.X_q, li.X_q), (lc.Y_q, li.Y_q)]:
            s_c = np.sum(np.abs(A_c[iq_c][:, v_c]) ** 2)
            s_f = np.sum(np.abs(A_f[iq_f][:, v_f]) ** 2)
            assert abs(s_f / s_c - 1.0) < 1e-8


def test_hermiticity_off_grid(system):
    """On an incommensurate fine mesh the Lanczos must stay Hermitian:
    real a coefficients and |b - c| at machine level."""
    dyn, ens = system
    li = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, 2 * L))
    li.init(use_symmetries=True)
    # a genuinely interpolated (non-coarse, non-TRI) q-point: n_z = 1 -> 1/6
    iq = int(np.where((li._fine_idx == [0, 0, 1]).all(axis=1))[0][0])
    li.prepare_mode_q(iq, 4)
    li.run_FT(N_STEPS, verbose=False)
    a = np.array(li.a_coeffs)
    b = np.array(li.b_coeffs)
    c = np.array(li.c_coeffs)
    assert np.all(np.abs(np.imag(a)) < 1e-12)
    n = min(len(b), len(c))
    assert np.max(np.abs(b[:n] - c[:n]) / np.abs(b[:n])) < 1e-8
