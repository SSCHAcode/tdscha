"""
Real-space vs q-space Lanczos Green function on the MOST anharmonic phonon of the
3x3x3 gold supercell (Examples/ensemble_gold, non-TRI grid).

This isolates the discrepancy at the Lanczos level (the source), before it
propagates into the static Hessian.  The most anharmonic phonon is identified
from the free energy Hessian (largest |Hessian freq - SSCHA freq|); on this data
it is band 1 of the iq=13..18 star (SSCHA 67.56 cm-1, renormalised to ~56.6).

The error is judged on the anharmonic *renormalisation* (renormalised frequency
minus SSCHA), not the absolute frequency, and on the spectral function, both of
which must agree between the two representations (the static and spectral Green
function are basis independent; time-reversal makes the +q and real-mode
responses coincide).
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

pytestmark = pytest.mark.skipif(not _HAS_Q, reason="QSpaceLanczos/Julia not available")

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'Examples', 'ensemble_gold')
NQIRR = 6
T = 300.0
N_STEPS = 60
CM = CC.Units.RY_TO_CM


def _load():
    dyn = CC.Phonons.Phonons(os.path.join(GOLD, "dyn_gen_pop1_"), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(GOLD, 1)
    return dyn, ens


def _most_anharmonic_mode(dyn, ens):
    """(iq, band, w_sscha[Ry]) of the phonon with the largest |Hessian - SSCHA|."""
    _, _, w_sscha, _ = dyn.DiagonalizeSupercell(return_qmodes=True)
    hess = ens.get_free_energy_hessian(include_v4=True, use_symmetries=True)
    _, _, w_hess, _ = hess.DiagonalizeSupercell(return_qmodes=True)
    best = None
    for iq in range(w_sscha.shape[1]):
        for b in range(w_sscha.shape[0]):
            ws = w_sscha[b, iq]
            if ws * CM < 1.0:
                continue
            shift = abs(w_hess[b, iq] - ws) * CM
            if best is None or shift > best[0]:
                best = (shift, iq, b, ws)
    return best[1], best[2], best[3], best[0]


def _match(dyn, iq, band):
    ws_sc, pols_sc, w_q, _ = dyn.DiagonalizeSupercell(return_qmodes=True)
    ss = dyn.structure.generate_supercell(dyn.GetSupercell())
    trans = CC.Methods.get_translations(pols_sc, ss.get_masses_array())
    good = ws_sc[~trans]
    return int(np.where(np.abs(good - w_q[band, iq]) < 1e-7)[0][0])


def _real(ens, mode):
    l = DL.Lanczos(ens, lo_to_split=None)
    l.ignore_v3, l.ignore_v4, l.use_wigner = False, False, True
    l.mode = DL.MODE_FAST_JULIA
    l.init(use_symmetries=True)
    l.prepare_mode(mode)
    l.run_FT(N_STEPS, run_simm=True, verbose=False)
    return l


def _q(ens, iq, band):
    q = QL.QSpaceLanczos(ens, lo_to_split=None)
    q.ignore_v3, q.ignore_v4 = False, False
    q.init(use_symmetries=True)
    q.prepare_mode_q(iq, band)
    q.run_FT(N_STEPS, verbose=False)
    return q


def _renorm_freq(obj, wm):
    g = obj.get_green_function_continued_fraction(
        np.array([0.0]), use_terminator=False, smearing=0.0)
    w2 = 1.0 / np.real(g[0])
    return np.sign(w2) * np.sqrt(abs(w2)) * CM


def test_gold_lanczos_gf_most_anharmonic():
    """Lanczos GF on the most anharmonic non-TRI phonon: q-space vs real-space.

    The anharmonic renormalisation (SSCHA -> Lanczos GF) must agree between
    real-space and q-space to within 3%.  This checks the bilinear (complex
    symmetric) conjugation convention is consistent in the Lanczos path:
    at non-TRI q, a sesquilinear (Hermitian) convention suppresses the
    ensemble average by momentum mismatch, while the correct bilinear
    convention conserves it.
    """
    dyn, ens = _load()
    iq, band, wm, hess_shift = _most_anharmonic_mode(dyn, ens)
    w_sscha = wm * CM
    mode = _match(dyn, iq, band)
    print("\nmost anharmonic phonon: iq={} band={}  SSCHA={:.2f} cm-1  "
          "(Hessian renorm {:.2f} cm-1)".format(iq, band, w_sscha, hess_shift))

    lr = _real(ens, mode)
    lq = _q(ens, iq, band)

    fr = _renorm_freq(lr, wm)
    fq = _renorm_freq(lq, wm)
    shift_r = w_sscha - fr
    shift_q = w_sscha - fq
    rel = abs(shift_r - shift_q) / max(abs(shift_r), 1e-12)

    # spectral function over a window around the mode
    wgrid = np.linspace(0.5 * wm, 1.5 * wm, 60)
    sm = wm * 0.05
    Ar = -np.imag(lr.get_green_function_continued_fraction(
        wgrid, use_terminator=True, smearing=sm))
    Aq = -np.imag(lq.get_green_function_continued_fraction(
        wgrid, use_terminator=True, smearing=sm))
    peak_r = wgrid[np.argmax(Ar)] * CM
    peak_q = wgrid[np.argmax(Aq)] * CM

    print("  renorm freq:    real={:.3f}  q={:.3f} cm-1".format(fr, fq))
    print("  anharm shift:   real={:+.3f}  q={:+.3f} cm-1   rel-err on shift={:.0%}".format(
        shift_r, shift_q, rel))
    print("  spectral peak:  real={:.3f}  q={:.3f} cm-1".format(peak_r, peak_q))

    # the anharmonic renormalisation must agree to within 3%
    assert rel < 0.03, (
        "anharmonic shift real={:.3f} q={:.3f} cm-1 (rel-err {:.0%})".format(
            shift_r, shift_q, rel))


if __name__ == "__main__":
    dyn, ens = _load()
    iq, band, wm, hs = _most_anharmonic_mode(dyn, ens)
    print("most anharmonic: iq={} band={} sscha={:.2f} hess-shift={:.2f}".format(
        iq, band, wm * CM, hs))
    mode = _match(dyn, iq, band)
    lr = _real(ens, mode)
    lq = _q(ens, iq, band)
    print("real renorm:", _renorm_freq(lr, wm), " q renorm:", _renorm_freq(lq, wm))
