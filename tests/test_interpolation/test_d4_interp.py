"""
Fourth-order (D4) interpolation validation for the q-mesh interpolated
Lanczos (see QSpaceInterpolation.py and _toy_chain.py).

What is (and is NOT) interpolated for D4
----------------------------------------
The D4 four-field averages ARE interpolated off-grid (the Bloch fields are
evaluated at the fine-mesh q, exactly as for D3) and rescaled by the vertex
factor scale4 = N_c/N_f.  They are, by design, NOT run through the designed
multitaper / ASR window passes: the D4 term always uses a single plain
full-period window pass (QSpaceInterpolation._call_julia_qspace, "D4 terms:
single plain-window pass"; report section on the windowed passes).  Two of
the tests below lock exactly that contract.

Why D4 cannot be tested with a purely-quartic model
---------------------------------------------------
The static one-phonon self-energy of a bare quartic vertex is the tadpole
(1/2) sum_k d4 chi_k, a constant shift of omega^2 that SSCHA already resums
into the auxiliary frequencies w_q.  What survives in the TDSCHA response is
the coupling of D4 to the two-phonon sector, which is only populated by D3.
Consequently a model with g3 = 0 gives EXACTLY zero anharmonic
renormalization (test_pure_quartic_renorm_is_negligible), and D4 must be
exercised through a MIXED (cubic + quartic) model, isolating its effect by
differencing ignore_v4.

Test strategy
-------------
* Renormalization level (gauge invariant, physical): a mixed-anharmonicity
  coarse (1,1,3) ensemble interpolated to (1,1,6) must reproduce a DIRECT
  (1,1,6) ensemble of the same model, INCLUDING the D4 contribution; and
  dropping D4 from the interpolation must spoil that agreement badly (proves
  the tolerance actually bites on the fourth order).
* Operator level (interp-vs-interp, identical basis -> no gauge ambiguity):
  the Frobenius norm of the D4 block of the anharmonic operator is a
  gauge-invariant scalar.  It must scale exactly linearly in scale4, and be
  bit-identical across window designs (plain / minimal_image / asr).

Runtime: the renormalization tests run the Julia kernel over N=3000 configs
(a few minutes); the operator tests are cheap (single kernel calls).
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

pytestmark = pytest.mark.skipif(not _HAS_Q,
                                reason="QSpaceLanczos/Julia not available")

T = 300.0
LC, LF = 3, 6
G3, G4 = 0.1, 1.0           # mixed anharmonicity: D3 softens, D4 hardens


def _renorm(lanc, iq, band, n_steps):
    lanc.prepare_mode_q(iq, band)
    lanc.run_FT(n_steps, verbose=False)
    return TC.lanczos_effective_freq(lanc) - lanc.w_q[band, iq]


# =====================================================================
#  Renormalization-level tests (mixed model, interp vs direct)
# =====================================================================
N_CONF = 3000
N_STEPS = 30
# probes on the fine (1,1,6) mesh: interpolated non-TRI (n_z=1), commensurate
# non-TRI (n_z=2), interpolated zone boundary (n_z=3), Gamma decay (n_z=0)
PROBES = [(0, 5), (1, 4), (2, 3), (3, 5)]


@pytest.fixture(scope="module")
def d4_renorms():
    """Full and D4-off renormalizations, interpolated and direct, for the
    mixed model.  Returns per-probe (d_full, d_no4, i_full, i_no4)."""
    dync = TC.build_dyn(LC)
    dynf = TC.build_dyn(LF)
    ensc = TC.make_ensemble(dync, T, N_CONF, seed=11, g3=G3, g4=G4)
    ensf = TC.make_ensemble(dynf, T, N_CONF, seed=77, g3=G3, g4=G4)

    li = QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, LF))
    ld = QL.QSpaceLanczos(ensf, lo_to_split=None)
    li.init(use_symmetries=True)
    ld.init(use_symmetries=True)

    out = {}
    for n_z, band in PROBES:
        iq_f = int(np.where((li._fine_idx == [0, 0, n_z]).all(axis=1))[0][0])
        iq_d = None
        for jq in range(ld.n_q):
            if li.find_fine_q(ld.q_points[jq]) == iq_f:
                iq_d = jq
                break
        assert iq_d is not None

        ld.ignore_v4 = False
        d_full = _renorm(ld, iq_d, band, N_STEPS)
        ld.ignore_v4 = True
        d_no4 = _renorm(ld, iq_d, band, N_STEPS)
        ld.ignore_v4 = False

        li.ignore_v4 = False
        i_full = _renorm(li, iq_f, band, N_STEPS)
        li.ignore_v4 = True
        i_no4 = _renorm(li, iq_f, band, N_STEPS)
        li.ignore_v4 = False

        out[(n_z, band)] = (d_full, d_no4, i_full, i_no4)
    return out


def test_d4_channel_is_active_and_hardens(d4_renorms):
    """Sanity that the test actually exercises the fourth order: the D4
    contribution (full minus D4-off) must be a sizeable, POSITIVE shift
    (the quartic stiffening opposes the cubic softening) in BOTH the direct
    and the interpolated calculation."""
    for key, (d_full, d_no4, i_full, i_no4) in d4_renorms.items():
        d_D4 = d_full - d_no4
        i_D4 = i_full - i_no4
        # the pure-cubic renormalization is a softening (< 0); D4 pushes back
        assert d_no4 < 0, "cubic part must soften at {}".format(key)
        assert d_D4 > 0.15 * abs(d_no4), \
            "direct D4 contribution too small at {}: {:.3g} vs cubic {:.3g}" \
            .format(key, d_D4, d_no4)
        assert i_D4 > 0.0, "interp D4 contribution wrong sign at {}".format(key)


def test_d4_renormalization_interpolates(d4_renorms):
    """The interpolated FULL renormalization (D3 + D4) reproduces the direct
    one within the noise+plain-window floor, AND dropping D4 from the
    interpolation spoils the agreement by a large factor -- so the tolerance
    genuinely constrains the fourth order rather than passing on D3 alone."""
    den = sum(abs(d_full) for (d_full, _, _, _) in d4_renorms.values())
    err_full = sum(abs(i_full - d_full)
                   for (d_full, _, i_full, _) in d4_renorms.values()) / den
    err_nod4 = sum(abs(i_no4 - d_full)
                   for (d_full, _, _, i_no4) in d4_renorms.values()) / den

    assert err_full < 0.5, \
        "interp+D4 does not reproduce direct: err {:.3f}".format(err_full)
    # a missing/broken D4 leaves the interpolation at ~1.3 aggregate error
    assert err_nod4 > 2.0 * err_full and err_nod4 > 0.9, \
        ("the D4 term barely matters here (err_full={:.3f}, err_nod4={:.3f}): "
         "the test would not detect a broken fourth order"
         .format(err_full, err_nod4))


# =====================================================================
#  Operator-level tests (interp vs interp: identical basis, no gauge issue)
# =====================================================================
def _d2v_d4_norm(li, iq, band, seed=3, scale4_mult=1.0):
    """Frobenius norm of the D4 block of the anharmonic operator for a fixed,
    deterministic two-phonon input state.  Gauge-invariant scalar.

    R1 is set to zero so only the alpha1 -> d2v (D4) path contributes; the
    two-phonon (pair-block) part of psi is filled deterministically so alpha1
    -- hence the D4 output -- is non-trivial."""
    li.prepare_mode_q(iq, band)
    nb = li.n_bands
    rng = np.random.RandomState(seed)
    vec = rng.randn(li.psi.shape[0] - nb) + 1j * rng.randn(li.psi.shape[0] - nb)
    li.psi[:nb] = 0.0
    li.psi[nb:] = vec

    saved = li.qspace_scale4
    li.qspace_scale4 = saved * scale4_mult
    try:
        R1 = np.zeros(nb, dtype=np.complex128)
        alpha1 = li._flatten_blocks(li.get_alpha1_beta1_wigner_q(get_alpha=True))
        _, d2v_blocks = li._call_julia_qspace(R1, alpha1)
    finally:
        li.qspace_scale4 = saved
    return np.sqrt(sum(np.sum(np.abs(b) ** 2) for b in d2v_blocks))


@pytest.fixture(scope="module")
def op_ensemble():
    return TC.make_ensemble(TC.build_dyn(LC), T, 800, seed=11, g3=G3, g4=G4)


def test_d4_operator_scales_linearly_with_scale4(op_ensemble):
    """scale4 = N_c/N_f multiplies the D4 average linearly: doubling it must
    exactly double the D4 operator norm (pins the vertex rescaling, and that
    it is applied to the fourth order specifically)."""
    li = QI.QSpaceLanczosInterp(op_ensemble, fine_mesh=(1, 1, LF))
    li.init(use_symmetries=True)
    iq = int(np.where((li._fine_idx == [0, 0, 1]).all(axis=1))[0][0])

    base = _d2v_d4_norm(li, iq, 4, scale4_mult=1.0)
    doubled = _d2v_d4_norm(li, iq, 4, scale4_mult=2.0)
    assert base > 1e-20, "D4 operator is trivially zero -- test is vacuous"
    assert abs(doubled / base - 2.0) < 1e-9, \
        "D4 does not scale linearly with scale4: ratio {:.6f}".format(
            doubled / base)


def test_d4_ignores_window_design(op_ensemble):
    """The fourth order is NOT windowed: the D4 operator output must be
    bit-identical whether the D3 estimator uses the plain, minimal-image, or
    ASR window design (D4 always runs on the plain full-period pass)."""
    iq_nz = 1
    norms = {}
    for design in ["plain", "minimal_image", "asr"]:
        li = QI.QSpaceLanczosInterp(op_ensemble, fine_mesh=(1, 1, LF),
                                    window_design=design)
        li.init(use_symmetries=True)
        iq = int(np.where((li._fine_idx == [0, 0, iq_nz]).all(axis=1))[0][0])
        norms[design] = _d2v_d4_norm(li, iq, 4)

    base = norms["plain"]
    assert base > 1e-20
    for design, val in norms.items():
        assert abs(val - base) <= 1e-12 * base, \
            ("D4 operator depends on window_design={} (rel diff {:.2e}): the "
             "fourth order must stay on the plain pass"
             .format(design, abs(val - base) / base))


# =====================================================================
#  Degeneracy guard: a purely quartic model must NOT renormalize
# =====================================================================
def test_pure_quartic_renorm_is_negligible():
    """A g3=0, g4!=0 model must give ~zero anharmonic renormalization (the D4
    tadpole is resummed into the SSCHA frequencies; only the D3-driven
    two-phonon coupling activates D4).  Compared against the same-size purely
    cubic model, which does renormalize.  Guards against a spurious D4
    one-phonon tadpole leaking into the operator."""
    N, ns = 2000, 20
    dynf = TC.build_dyn(LF)
    ens_q = TC.make_ensemble(dynf, T, N, seed=77, g3=0.0, g4=1.0)
    ens_c = TC.make_ensemble(dynf, T, N, seed=77, g3=0.1, g4=0.0)

    lq = QL.QSpaceLanczos(ens_q, lo_to_split=None); lq.init(use_symmetries=True)
    lc = QL.QSpaceLanczos(ens_c, lo_to_split=None); lc.init(use_symmetries=True)

    r_quartic = _renorm(lq, 0, 5, ns)
    r_cubic = _renorm(lc, 0, 5, ns)

    # toggling D4 in the purely quartic model must also do nothing
    lq.ignore_v4 = True
    r_quartic_no4 = _renorm(lq, 0, 5, ns)

    assert abs(r_cubic) > 1e-5, "cubic reference did not renormalize"
    assert abs(r_quartic) < 0.05 * abs(r_cubic), \
        "purely quartic model renormalized ({:.3g}); spurious D4 tadpole?" \
        .format(r_quartic)
    assert abs(r_quartic - r_quartic_no4) < 0.05 * abs(r_cubic), \
        "D4 toggle changed the purely quartic renormalization"
