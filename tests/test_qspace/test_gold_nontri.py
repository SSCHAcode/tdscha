"""
Real NON-TRI benchmark: q-space vs real-space anharmonic renormalisation on a
3x3x3 gold supercell (Examples/ensemble_gold, 50-config converged SSCHA
ensemble, 1 atom/cell, 27 q-points of which 26 are NOT time-reversal invariant;
only Gamma is TRI).

This is the regime the all-TRI 2x2x2 SnTe data never exercises, and it exposes a
real bug: the q-space code reproduces the anharmonic renormalisation exactly when
the external momentum is TRI (SnTe Gamma-optical / X point) but loses almost all
of it at non-TRI q-points.

NB: gold is monatomic, so its Gamma has only the three (zero) acoustic modes and
is NOT a usable control -- the TRI control here is SnTe Gamma-optical, whose modes
genuinely renormalise.

The decisive probe is the *static free energy Hessian*, which has no Lanczos
convergence or +q/real-mode mapping ambiguity:

    reference   = ens.get_free_energy_hessian()           (real-space, trusted)
    q-space     = QSpaceHessian.compute_hessian_at_q(iq)

The error must be judged on the *renormalisation* (Hessian freq - SSCHA freq),
not the absolute frequency.  Example (gold iq=13, band 1): SSCHA 67.56 ->
reference 56.64 (shift -10.92) but q-space 67.23 (shift -0.33): only ~3% of the
anharmonic effect is captured.
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
    import tdscha.QSpaceHessian as QH
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

pytestmark = pytest.mark.skipif(not _HAS_Q, reason="QSpaceLanczos/Julia not available")

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'Examples', 'ensemble_gold')
SNTE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'test_julia', 'data')
NQIRR = 6
T = 300.0
CM = CC.Units.RY_TO_CM

# NON-TRI q-points of gold (incl. the most anharmonic one, iq=13).
# NB: gold has 1 atom/cell, so its Gamma carries only the (zero) acoustic modes
# and is NOT a usable control.  The TRI control is SnTe Gamma-optical, which has
# real modes that renormalise -- see test_snte_hessian_gamma_tri_control.
NONTRI_Q = [1, 13]


def _load():
    dyn = CC.Phonons.Phonons(os.path.join(GOLD, "dyn_gen_pop1_"), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(GOLD, 1)
    return dyn, ens


def _freqs(ev):
    return np.sign(ev) * np.sqrt(np.abs(ev)) * CM


def _hessian_freqs_per_q(ens, qlist):
    """Return {iq: (ref_freqs, qspace_freqs)} for the free energy Hessian."""
    ref = ens.get_free_energy_hessian(include_v4=True, use_symmetries=True)
    qh = QH.QSpaceHessian(ens, verbose=False, lo_to_split=None)
    qh.init(use_symmetries=True)
    out = {}
    for iq in qlist:
        wref, _ = ref.DyagDinQ(iq)
        ev_ref = np.sort(np.sign(wref) * wref ** 2)
        Hq = qh.compute_hessian_at_q(iq, tol=1e-8, max_iters=1500)
        pol = qh.pols_q[:, :, iq]
        Phi = pol @ Hq @ np.conj(pol).T
        ev_q = np.sort(np.real(np.linalg.eigvalsh(Phi)))
        out[iq] = (_freqs(ev_ref), _freqs(ev_q))
    return out


def test_gold_force_parseval():
    """Sanity: on this REAL dyn the X and Y projections are consistent."""
    dyn, ens = _load()
    lanc = DL.Lanczos(ens, lo_to_split=None)
    lanc.ignore_v3, lanc.ignore_v4, lanc.use_wigner = False, True, True
    lanc.mode = DL.MODE_FAST_JULIA
    lanc.init(use_symmetries=True)
    q = QL.QSpaceLanczos(ens, lo_to_split=None)
    q.ignore_v3, q.ignore_v4 = False, True
    q.init(use_symmetries=True)
    Xr = np.sum(np.abs(lanc.X) ** 2)
    Yr = np.sum(np.abs(lanc.Y) ** 2)
    Xq = Yq = 0.0
    for iq in range(q.n_q):
        v = q.valid_modes_q[:, iq]
        Xq += np.sum(np.abs(q.X_q[iq][:, v]) ** 2)
        Yq += np.sum(np.abs(q.Y_q[iq][:, v]) ** 2)
    assert abs(Xq / Xr - 1.0) < 1e-4
    assert abs(Yq / Yr - 1.0) < 1e-4


def test_snte_hessian_gamma_tri_control():
    """TRI control with REAL renormalising modes: SnTe Gamma-optical Hessian.

    Gold's Gamma is only the (zero) acoustic modes, so it is no control.  SnTe
    (2x2x2, all-TRI) has Gamma optical modes that genuinely renormalise (here to
    an unstable, imaginary mode); the q-space and reference Hessians must agree.
    Compares the Hessian eigenvalues (w^2 with sign), as in
    test_qspace_hessian.py::test_qspace_hessian_gamma.
    """
    dyn = CC.Phonons.Phonons(os.path.join(SNTE, "dyn_gen_pop1_"), 3)
    ens = sscha.Ensemble.Ensemble(dyn, 250.0)
    ens.load_bin(SNTE, 1)

    ref = ens.get_free_energy_hessian(include_v4=True, use_symmetries=True)
    w, _ = ref.DyagDinQ(0)
    ev_ref = np.sort(w ** 2 * np.sign(w))
    ev_ref = ev_ref[np.abs(ev_ref) > 1e-8]      # drop acoustic

    qh = QH.QSpaceHessian(ens, verbose=False, lo_to_split=None)
    qh.init(use_symmetries=True)
    Hq = qh.compute_hessian_at_q(0, tol=1e-8, max_iters=1000)
    pol = qh.pols_q[:, :, 0]
    Phi = pol @ Hq @ np.conj(pol).T
    ev_q = np.sort(np.real(np.linalg.eigvalsh(Phi)))
    ev_q = ev_q[np.abs(ev_q) > 1e-8]

    n = min(len(ev_ref), len(ev_q))
    rel = np.max(np.abs(ev_ref[:n] - ev_q[:n]) / np.maximum(np.abs(ev_ref[:n]), 1e-15))
    f_ref = _freqs(ev_ref[:n])  # signed cm-1 (negative = unstable)
    f_q = _freqs(ev_q[:n])
    print("\nSnTe Gamma optical (signed cm-1): ref={}  qspace={}  rel-err={:.2%}".format(
        np.round(f_ref, 2), np.round(f_q, 2), rel))
    assert rel < 0.05, "SnTe Gamma-optical Hessian mismatch: {:.2%}".format(rel)


def test_gold_hessian_nontri():
    """At non-TRI q the q-space Hessian must reproduce the anharmonic shift.

    The anharmonic renormalisation (SSCHA -> full Hessian) must agree between
    real-space and q-space to within 3%.  This checks the bilinear (complex
    symmetric) conjugation convention is consistent: at non-TRI q, a
    sesquilinear (Hermitian) convention suppresses the ensemble average by
    momentum mismatch, while the correct bilinear convention conserves it.
    """
    dyn, ens = _load()
    w_sscha = dyn.DiagonalizeSupercell(return_qmodes=True)[2] * CM  # (band, q)
    res = _hessian_freqs_per_q(ens, NONTRI_Q)
    for iq in NONTRI_Q:
        ref, qsp = res[iq]
        for b in range(len(ref)):
            ws = w_sscha[b, iq]
            if ws < 1.0:
                continue
            ref_shift = ws - ref[b]
            q_shift = ws - qsp[b]
            rel = abs(ref_shift - q_shift) / max(abs(ref_shift), 1e-12)
            print("iq={} band{}: sscha={:.2f} ref={:.2f}(d{:+.2f}) "
                  "qspace={:.2f}(d{:+.2f})  shift rel-err={:.0%}".format(
                      iq, b, ws, ref[b], -ref_shift, qsp[b], -q_shift, rel))
            # the anharmonic renormalisation must agree to within 3%
            assert rel < 0.03, (
                "iq={} band{}: anharmonic shift real={:.3f} q={:.3f} cm-1 "
                "(rel-err {:.0%})".format(iq, b, ref_shift, q_shift, rel))


if __name__ == "__main__":
    dyn, ens = _load()
    res = _hessian_freqs_per_q(ens, NONTRI_Q)
    w_sscha = dyn.DiagonalizeSupercell(return_qmodes=True)[2] * CM
    for iq in NONTRI_Q:
        ref, qsp = res[iq]
        print("\ngold iq={} (non-TRI):".format(iq))
        for b in range(len(ref)):
            ws = w_sscha[b, iq]
            print("  band{}: sscha={:7.2f}  ref={:7.2f}  qspace={:7.2f}".format(
                b, ws, ref[b], qsp[b]))
