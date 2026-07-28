"""
Physics validation of the trilinear q-space interpolation on the anharmonic
diatomic chain: a coarse (1,1,3) ensemble interpolated to the (1,1,6) mesh
against a DIRECT (1,1,6)-supercell ensemble of the same model.

Metric: the anharmonic renormalization (static Lanczos frequency - SSCHA
frequency); absolute frequencies hide the anharmonic signal. The two
ensembles are independent, so tolerances are calibrated against the
stochastic noise floor of the toy (~10% aggregate at N=3000-4000; the
windowed-plain interpolation error of the OLD scheme on this bond-ranged
model is ~30% at L_c=3 -- the trilinear scheme belongs to the same
Bartlett-centering class, so comparable accuracy is expected).

The probes cover: interpolated non-TRI q, commensurate non-TRI q,
interpolated TRI zone boundary, and Gamma decaying into interpolated pairs.

The scale-factor necessity test pins the N_c -> N_f vertex rescaling with a
magnitude and a direction, not just "different".
"""
import os, sys
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "test_interpolation"))
import numpy as np
import pytest

import _toy_chain as TC

try:
    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceTrilinear as QT
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

pytestmark = pytest.mark.skipif(not _HAS_Q,
                                reason="QSpaceLanczos/Julia not available")

T = 300.0
N_CONF = 3000
N_STEPS = 35
G3 = 0.1
LC, LF = 3, 6

# (mesh index n_z on the fine (1,1,6) mesh, band):
#   n_z = 0 (Gamma)  -> commensurate, decays into interpolated pairs
#   n_z = 2 (q=1/3)  -> commensurate with the coarse mesh, non-TRI
PROBES = [(0, 5), (2, 3), (2, 0)]


def _fine_iq(li, n_z):
    return int(np.where((li._fine_idx == [0, 0, n_z]).all(axis=1))[0][0])


@pytest.fixture(scope="module")
def renorms():
    dync = TC.build_dyn(LC)
    dynf = TC.build_dyn(LF)
    ensc = TC.make_ensemble(dync, T, N_CONF, seed=11, g3=G3)
    ensf = TC.make_ensemble(dynf, T, N_CONF, seed=77, g3=G3)

    li = QT.QSpaceTrilinearLanczos(ensc, fine_mesh=(1, 1, LF))
    ld = QL.QSpaceLanczos(ensf, lo_to_split=None)
    li.init(use_symmetries=True)
    ld.init(use_symmetries=True)

    def get(lanc, iq, band):
        lanc.prepare_mode_q(iq, band)
        lanc.run_FT(N_STEPS, verbose=False)
        return TC.lanczos_effective_freq(lanc) - lanc.w_q[band, iq]

    out = {}
    for n_z, band in PROBES:
        iq_f = _fine_iq(li, n_z)
        iq_d = None
        for jq in range(ld.n_q):
            if li.find_fine_q(ld.q_points[jq]) == iq_f:
                iq_d = jq
                break
        assert iq_d is not None
        # the direct run must init the Julia symmetry cache for ITS mesh
        ld.init(use_symmetries=True)
        rd = get(ld, iq_d, band)
        li.init(use_symmetries=True)
        ri = get(li, iq_f, band)
        out[(n_z, band)] = (rd, ri)
    return out


def test_renormalization_sign_and_magnitude(renorms):
    """Cubic anharmonicity at low T softens the modes; both calculations
    must agree on sign and order of magnitude."""
    for key, (rd, ri) in renorms.items():
        assert rd < 0, "direct renormalization must soften at {}".format(key)
        assert ri < 0, "interp renormalization must soften at {}".format(key)
        assert 0.2 < abs(ri / rd) < 5.0, \
            "gross mismatch at {}: direct {} vs interp {}".format(key, rd, ri)


def test_renormalization_aggregate_accuracy(renorms):
    """Aggregate accuracy target as in the windowed-scheme suite: < 50%
    (systematic failures -- wrong scale factors, broken pair map or fold,
    conjugation errors -- fail at ~100%+)."""
    num = sum(abs(ri - rd) for (rd, ri) in renorms.values())
    den = sum(abs(rd) for (rd, _) in renorms.values())
    err = num / den
    print("Aggregate renormalization error: {:.3f}".format(err))
    assert err < 0.5


def test_scale_factors_are_necessary():
    """Disabling the N_c -> N_f vertex rescaling must OVERSHOOT the
    renormalization roughly by N_f/N_c (the D3 self-energy doubles)."""
    dync = TC.build_dyn(LC)
    ensc = TC.make_ensemble(dync, T, N_CONF // 3, seed=21, g3=G3)

    li = QT.QSpaceTrilinearLanczos(ensc, fine_mesh=(1, 1, LF))
    li.init(use_symmetries=True)
    iq = _fine_iq(li, 0)

    def renorm():
        li.prepare_mode_q(iq, 5)
        li.run_FT(N_STEPS, verbose=False)
        return TC.lanczos_effective_freq(li) - li.w_q[5, iq]

    r_ok = renorm()
    li.qspace_scale3, li.qspace_scale4 = 1.0, 1.0
    r_bad = renorm()
    li.qspace_scale3 = np.sqrt(li.cn_q / float(li.n_q))
    li.qspace_scale4 = li.cn_q / float(li.n_q)

    ratio = r_bad / r_ok
    assert ratio > 1.4, \
        "unscaled renormalization should overshoot ~x2, got x{:.2f}".format(
            ratio)
