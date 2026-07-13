"""
Fourth-order (D4) atomic centering: channel split and commensurate identity.

The pin-one-leg D4 centering (d4_center="leg") splits the four-phonon
estimator by the position of its force leg (external w/v, internal w/v),
exactly as the D3 estimator is split into its three force channels.  Two
invariants are locked here:

* Channel split: on identical (plain) fields, the sum of the W-force gated
  call (channels ew+iw) and the V-force gated call (ev+iv) must equal the
  all-channel D4 call bit-for-bit.  This protects the 4-way permutation
  split of the estimator against regressions in the Julia kernels.

* Commensurate identity: interpolating a coarse ensemble onto ITS OWN mesh,
  the centered D4 operator must reproduce the plain-pass D4 operator to
  machine precision.  Each centered term is linear in the delta-windowed
  force field and the atomic windows collapse to plain at commensurate q,
  so the pinned-atom/origin sum equals the plain estimator configuration
  by configuration.  Checked for both d4_center="leg" and the legacy
  d4_center="reference".
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
LC = 3
G3, G4 = 0.1, 1.0


@pytest.fixture(scope="module")
def ens():
    return TC.make_ensemble(TC.build_dyn(LC), T, 400, seed=5, g3=G3, g4=G4)


def _prep(li, iq_nz=1, band=4, seed=3):
    """Deterministic two-phonon state; returns (R1, alpha1_flat)."""
    iq = int(np.where((li._fine_idx == [0, 0, iq_nz]).all(axis=1))[0][0])
    li.prepare_mode_q(iq, band)
    nb = li.n_bands
    rng = np.random.RandomState(seed)
    vec = rng.randn(li.psi.shape[0] - nb) + 1j * rng.randn(li.psi.shape[0] - nb)
    li.psi[:nb] = 0.0
    li.psi[nb:] = vec
    R1 = np.zeros(nb, dtype=np.complex128)
    alpha1 = li._flatten_blocks(li.get_alpha1_beta1_wigner_q(get_alpha=True))
    if li.qspace_prefiltered:
        alpha1 = li._fold_alpha1(alpha1)
    return R1, alpha1


def test_d4_channel_split_equals_full(ens):
    """(ew+iw) + (ev+iv) gated D4 calls == the all-channel D4 call on
    identical plain fields."""
    li = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, 2 * LC))
    li.init(use_symmetries=True)
    R1, alpha1 = _prep(li)
    plain = (li.X_q, li.Y_q)

    full = li._call_slots(plain, plain, plain, R1, alpha1, False, True)
    part_w = li._call_slots(plain, plain, plain, R1, alpha1, False, True,
                            d4_channels=(True, False, True, False))
    part_v = li._call_slots(plain, plain, plain, R1, alpha1, False, True,
                            d4_channels=(False, True, False, True))
    split = part_w + part_v

    ref = np.max(np.abs(full))
    assert ref > 1e-20, "D4 output is trivially zero -- test is vacuous"
    assert np.max(np.abs(split - full)) < 1e-12 * ref, \
        "D4 force-channel split does not sum to the full D4 call"


@pytest.mark.parametrize("mode", ["leg", "reference"])
def test_d4_center_commensurate_identity(ens, mode):
    """Centered D4 == plain D4 when interpolating onto the coarse mesh."""
    ops = {}
    for d4c in (False, mode):
        li = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, LC),
                                    window_design="atomic_delta",
                                    window_far=2, d4_center=d4c)
        li.init(use_symmetries=True)
        R1, alpha1 = _prep(li)
        ops[d4c] = np.asarray(li._call_julia_qspace(R1, alpha1)[1])

    ref = max(np.max(np.abs(b)) for b in ops[False])
    assert ref > 1e-20
    err = max(np.max(np.abs(a - b))
              for a, b in zip(ops[mode], ops[False]))
    assert err < 1e-10 * ref, \
        ("d4_center='%s' breaks the commensurate identity: rel err %.2e"
         % (mode, err / ref))
