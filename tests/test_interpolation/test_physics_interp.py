"""
Physics validation of the q-mesh interpolation on the anharmonic diatomic
chain (see _toy_chain.py): a coarse (1,1,3) ensemble interpolated to the
(1,1,6) mesh is compared against a DIRECT (1,1,6)-supercell ensemble of the
same model.

Design of the tests
-------------------
- The comparison metric is the anharmonic renormalization
  (static Lanczos frequency - SSCHA frequency), the established way to judge
  anharmonic agreement (absolute frequencies hide the anharmonic signal).
- The two ensembles are statistically independent, so the assertions are
  calibrated against the measured stochastic noise floor of the toy
  (~10% aggregate at N=4000; the plain-window interpolation error on this
  bond-ranged model is ~30% at L_c=3). The tolerances catch every
  systematic failure mode observed during development:
    * missing/wrong vertex rescaling  -> factor ~N_f/N_c = 2 error,
    * missing field pre-filter        -> factor ~0.5 deficit,
    * broken q-pair map or TRI gauge  -> O(1) garbage at non-TRI points.
- test_scale_factors_are_necessary deliberately disables the N_c -> N_f
  vertex rescaling and asserts the renormalization OVERSHOOTS by ~N_f/N_c:
  this pins the scaling with a sign and a magnitude, not just "different".

Runtime: ~2-4 minutes (Julia kernel, N=3000 configs, 12 Lanczos runs).
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
N_CONF = 3000
N_STEPS = 35
G3 = 0.1
LC, LF = 3, 6

# Representative (mesh index n_z, band) probes on the fine (1,1,6) mesh:
#   n_z = 1 (q=1/6)  -> interpolated, NON-TRI q-point (q != -q)
#   n_z = 2 (q=1/3)  -> commensurate with the coarse mesh, non-TRI
#   n_z = 3 (q=1/2)  -> interpolated zone boundary (TRI)
#   n_z = 0 (Gamma)  -> commensurate, decay into interpolated pairs
PROBES = [(0, 5), (1, 0), (1, 4), (2, 3), (3, 2), (3, 5)]


@pytest.fixture(scope="module")
def renorms():
    dync = TC.build_dyn(LC)
    dynf = TC.build_dyn(LF)
    ensc = TC.make_ensemble(dync, T, N_CONF, seed=11, g3=G3)
    ensf = TC.make_ensemble(dynf, T, N_CONF, seed=77, g3=G3)

    li = QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, LF))
    ld = QL.QSpaceLanczos(ensf, lo_to_split=None)
    li.init(use_symmetries=True)
    ld.init(use_symmetries=True)

    def get(lanc, iq, band):
        lanc.prepare_mode_q(iq, band)
        lanc.run_FT(N_STEPS, verbose=False)
        return TC.lanczos_effective_freq(lanc) - lanc.w_q[band, iq]

    out = {}
    for n_z, band in PROBES:
        iq_f = int(np.where((li._fine_idx == [0, 0, n_z]).all(axis=1))[0][0])
        # locate the same q on the direct object through the interp lookup
        # (both live on the same fine mesh)
        iq_d = None
        for jq in range(ld.n_q):
            if li.find_fine_q(ld.q_points[jq]) == iq_f:
                iq_d = jq
                break
        assert iq_d is not None
        out[(n_z, band)] = (get(ld, iq_d, band), get(li, iq_f, band), li, ld)
    return out


def test_renormalization_sign_and_magnitude(renorms):
    """Physical sanity: cubic anharmonicity at low T softens the modes;
    both calculations must agree on sign and order of magnitude."""
    for key, (rd, ri, li, ld) in renorms.items():
        assert rd < 0, "direct renormalization must be a softening {}".format(key)
        assert ri < 0, "interp renormalization must be a softening {}".format(key)
        # order of magnitude (cm^-1 scale of this model: 3 - 40 cm^-1)
        assert 0.2 < abs(ri / rd) < 5.0, \
            "gross mismatch at {}: direct {} vs interp {}".format(key, rd, ri)


def test_renormalization_aggregate_accuracy(renorms):
    """Aggregate accuracy: with the pre-filtered fields the plain-window
    interpolation reproduces the direct fine-supercell renormalization
    within 50% aggregate (measured: ~30% interpolation + ~10% noise at
    this size; a missing rescaling or filter fails at 100%+)."""
    num = sum(abs(ri - rd) for (rd, ri, _, _) in renorms.values())
    den = sum(abs(rd) for (rd, _, _, _) in renorms.values())
    err = num / den
    assert err < 0.5, "aggregate interpolation error too large: {:.3f}".format(err)


def test_scale_factors_are_necessary(renorms):
    """Disable the N_c -> N_f vertex rescaling: the renormalization must
    overshoot by roughly N_f/N_c = 2 (the D3 bubble doubles when the pair
    sum doubles without diluting the vertices). This pins both the presence
    AND the magnitude of the rescaling."""
    # reuse the interp object from the fixture
    (rd, ri, li, ld) = renorms[(0, 5)]

    s3, s4 = li.qspace_scale3, li.qspace_scale4
    try:
        li.qspace_scale3 = 1.0
        li.qspace_scale4 = 1.0
        iq_f = int(np.where((li._fine_idx == [0, 0, 0]).all(axis=1))[0][0])
        li.prepare_mode_q(iq_f, 5)
        li.run_FT(N_STEPS, verbose=False)
        r_noscale = TC.lanczos_effective_freq(li) - li.w_q[5, iq_f]
    finally:
        li.qspace_scale3 = s3
        li.qspace_scale4 = s4

    ratio = r_noscale / ri
    assert 1.5 < ratio < 3.0, \
        ("without vertex rescaling the renormalization should be ~2x "
         "(N_f/N_c): got {:.3f} (scaled {:.4g}, unscaled {:.4g} Ry)"
         .format(ratio, ri, r_noscale))


def test_two_phonon_sector_is_fine(renorms):
    """Structural: the interpolated Lanczos must have pair blocks at the
    interpolated q-points (phonon decay into interpolated pairs)."""
    (rd, ri, li, ld) = renorms[(0, 5)]
    li.build_q_pair_map(0)  # Gamma perturbation
    partner_nz = sorted(li._fine_idx[iq2][2] for (_, iq2) in li.unique_pairs)
    # pairs (q, -q): partners must include the interpolated n_z = 5 (= -1/6)
    assert len(li.unique_pairs) == LF // 2 + 1
    assert 5 in partner_nz or 1 in partner_nz, \
        "no interpolated decay channels at Gamma"
