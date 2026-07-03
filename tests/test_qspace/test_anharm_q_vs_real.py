"""
Compare the ANHARMONIC part of the Lanczos propagator L between the real-space
DynamicalLanczos and the q-space QSpaceLanczos.

The Lanczos coefficients a/b/c are scalar products <v|L^k|v> and are therefore
*basis independent*: when the perturbation is the same physical phonon the two
representations must produce identical coefficients (up to ensemble noise).

Contents
--------
  test_coeffs_tri
      All-TRI 2x2x2 supercell (every q satisfies q == -q) with the real SnTe-like
      ensemble.  The codes AGREE for D3, D4 and full anharmonic, at Gamma and at
      an X point.  This is a valid regression test and it passes.

  test_synthetic_dyn_force_parseval_healthcheck
      Historically an xfail "negative control" blaming the hand-built dyn for
      inconsistent force projections (Sum|Y_q|^2 / Sum|Y_real|^2 ~ 0.04).  The
      real cause was a STALENESS TRAP in the ensemble: ens.generate()
      precomputes u_disps_qspace and zeroes forces_qspace; assigning
      ens.forces afterwards does not refresh them, and QSpaceLanczos skips
      ens.init() when u_disps_qspace is already present -- so the q-space code
      silently used zero forces.  _build_nontri.make_ensemble now calls
      ens.init() after assigning the forces and both projections are exact;
      the test now passes as a positive regression guard.
"""
from __future__ import print_function
import os
import sys
os.environ.setdefault("JULIA_NUM_THREADS", "1")
import numpy as np
import pytest

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Methods
import cellconstructor.Units

import sscha, sscha.Ensemble
import tdscha.DynamicalLanczos as DL

try:
    import tdscha.QSpaceLanczos as QL
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

import _build_nontri as B

pytestmark = pytest.mark.skipif(not _HAS_Q, reason="QSpaceLanczos/Julia not available")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _match_mode(dyn, iq, band):
    """Supercell-mode index whose frequency matches q-mode (iq, band)."""
    ws_sc, pols_sc, w_q, _ = dyn.DiagonalizeSupercell(return_qmodes=True)
    ss = dyn.structure.generate_supercell(dyn.GetSupercell())
    trans = CC.Methods.get_translations(pols_sc, ss.get_masses_array())
    good = ws_sc[~trans]
    return int(np.where(np.abs(good - w_q[band, iq]) < 1e-7)[0][0])


def _run_real(ens, mode_index, n_steps, iv3, iv4):
    lanc = DL.Lanczos(ens, lo_to_split=None)
    lanc.ignore_harmonic = False
    lanc.ignore_v3, lanc.ignore_v4 = iv3, iv4
    lanc.use_wigner = True
    lanc.mode = DL.MODE_FAST_JULIA
    lanc.init(use_symmetries=True)
    lanc.prepare_mode(mode_index)
    lanc.run_FT(n_steps, run_simm=True, verbose=False)
    return np.array(lanc.a_coeffs), np.array(lanc.b_coeffs)


def _run_q(ens, iq, band, n_steps, iv3, iv4):
    q = QL.QSpaceLanczos(ens, lo_to_split=None)
    q.ignore_harmonic = False
    q.ignore_v3, q.ignore_v4 = iv3, iv4
    q.init(use_symmetries=True)
    q.prepare_mode_q(iq, band)
    q.run_FT(n_steps, verbose=False, reorthogonalize=True)
    return np.array(q.a_coeffs), np.array(q.b_coeffs)


def _relerr(x, y):
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    s = np.maximum(np.abs(x), np.abs(y))
    s[s == 0] = 1.0
    return np.max(np.abs(x - y) / s)


def _compare_coeffs(ens, dyn, iq, band, n_steps=5):
    mode_index = _match_mode(dyn, iq, band)
    worst = 0.0
    for iv3, iv4 in [(False, True), (True, False), (False, False)]:
        ar, br = _run_real(ens, mode_index, n_steps, iv3, iv4)
        aq, bq = _run_q(ens, iq, band, n_steps, iv3, iv4)
        worst = max(worst, _relerr(ar, aq))
        if len(br) and len(bq):
            worst = max(worst, _relerr(br, bq))
    return worst


def _force_parseval_ratio(ens):
    """Sum|Y_q|^2 / Sum|Y_real|^2 over valid modes -- must be 1 for consistency."""
    lanc = DL.Lanczos(ens, lo_to_split=None)
    lanc.ignore_v3, lanc.ignore_v4, lanc.use_wigner = False, True, True
    lanc.mode = DL.MODE_FAST_JULIA
    lanc.init(use_symmetries=True)
    q = QL.QSpaceLanczos(ens, lo_to_split=None)
    q.ignore_v3, q.ignore_v4 = False, True
    q.init(use_symmetries=True)
    Yr = np.sum(np.abs(lanc.Y) ** 2)
    Xr = np.sum(np.abs(lanc.X) ** 2)
    Yq = Xq = 0.0
    for iq in range(q.n_q):
        v = q.valid_modes_q[:, iq]
        Xq += np.sum(np.abs(q.X_q[iq][:, v]) ** 2)
        Yq += np.sum(np.abs(q.Y_q[iq][:, v]) ** 2)
    return Xq / Xr, Yq / Yr


# --------------------------------------------------------------------------
# 1. all-TRI 2x2x2, real ensemble -- the codes agree (valid regression test)
# --------------------------------------------------------------------------
def test_coeffs_tri():
    T = 250.0
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), 3)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)

    # displacement AND force projections must be consistent on real data
    xr, yr = _force_parseval_ratio(ens)
    assert abs(xr - 1.0) < 1e-6, "X Parseval broken on real ensemble: {}".format(xr)
    assert abs(yr - 1.0) < 1e-6, "Y Parseval broken on real ensemble: {}".format(yr)

    # Gamma optical (band 5) and X-point (iq=5, band 2)
    worst_gamma = _compare_coeffs(ens, dyn, 0, 5)
    worst_x = _compare_coeffs(ens, dyn, 5, 2)
    assert worst_gamma < 1e-4, "TRI Gamma coeffs disagree: {:.2e}".format(worst_gamma)
    assert worst_x < 1e-4, "TRI X-point coeffs disagree: {:.2e}".format(worst_x)


# --------------------------------------------------------------------------
# 2. hand-built dyn health check (RESOLVED).
#    The historical "force-Parseval artifact" was NOT a property of the
#    hand-built dyn: ens.generate() precomputes u_disps_qspace and leaves
#    forces_qspace = 0, and assigning ens.forces afterwards does not refresh
#    them (QSpaceLanczos skips ens.init() when u_disps_qspace is present).
#    make_ensemble now calls ens.init() after assigning the forces, and both
#    projections are consistent. This test guards against a regression of
#    that staleness trap.
# --------------------------------------------------------------------------
def test_synthetic_dyn_force_parseval_healthcheck():
    dyn = B.build_dyn()
    ens = B.make_ensemble(dyn, 300.0, 3000, seed=1)
    xr, yr = _force_parseval_ratio(ens)
    assert abs(xr - 1.0) < 1e-6, "X Parseval: {:.6f}".format(xr)
    assert abs(yr - 1.0) < 1e-6, "Y Parseval: {:.6f} (stale forces_qspace?)".format(yr)


if __name__ == "__main__":
    T = 250.0
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), 3)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)
    print("TRI X/Y Parseval:", _force_parseval_ratio(ens))
    print("TRI worst coeff rel-err (Gamma):", _compare_coeffs(ens, dyn, 0, 5))
