"""Stochastic interpolated TD-SCHA against the analytic anharmonic chain.

End-to-end through the public API: CellConstructor dynamical matrix,
sscha.Ensemble.generate (T = 0, even/odd sampling) with forces from the
polynomial potential, mode exclusion of the five zero-frequency bands
(ensemble.ignore_small_w), QSpaceAtomFourierLanczos interpolation 9 -> 36,
Lanczos and continued fraction. The reference is the exact bubble and RPA
Green function of polynomial_reference.py. Monte Carlo tolerances: the seed
is fixed; the bound (15%) is about twice the observed statistical error and
half the distance between bubble and RPA, so a wrong vertex fails the test.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('JULIA_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / 'benchmarks' / 'interpolation'))

try:
    import tdscha.QSpaceLanczos as QL
    _AVAILABLE = QL.__JULIA_EXT__
except Exception:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason='Julia backend not installed')

import polynomial_reference as reference
import stochastic_chain as chain

COARSE, FINE, ETA = 9, 36, .015
FREQUENCIES = np.linspace(.6, 1.65, 211)


@pytest.fixture(scope='module')
def ensemble():
    return chain.make_ensemble(chain.build_dyn(COARSE), 8000, seed=1)


def test_forces_match_reference_potential():
    reference.CUBIC, reference.QUARTIC = chain.STRONG['cubic'], chain.STRONG['quartic']
    x = np.random.default_rng(0).normal(size=(4, COARSE))
    sigma2 = reference.counterterms(master_size=COARSE)
    expected = np.array([reference.energy_force(row, sigma2)[1] for row in x])
    assert np.allclose(chain.chain_forces(x, sigma2), expected, atol=1e-13)


def test_mode_exclusion_keeps_only_the_chain_band(ensemble):
    import tdscha.QSpaceAtomFourier as AF
    lanczos = AF.QSpaceAtomFourierLanczos(ensemble, fine_mesh=(1, 1, FINE))
    assert np.all(lanczos.valid_modes_q.sum(axis=0) == 1)
    q = lanczos._fine_idx[:, 2] / FINE
    w = lanczos.w_q[lanczos.valid_modes_q] / chain.OMEGA0
    assert np.allclose(w, reference.frequency(q), atol=1e-8)


@pytest.mark.parametrize('quartic', [False, True])
def test_stochastic_spectrum_matches_analytic(ensemble, quartic):
    stochastic = chain.spectral_function(ensemble, FINE, FREQUENCIES, ETA, quartic)
    exact = chain.analytic_spectral_function(FINE, FREQUENCIES, ETA, quartic)
    error = np.linalg.norm(stochastic - exact) / np.linalg.norm(exact)
    assert error < .15, error
    step = FREQUENCIES[1] - FREQUENCIES[0]
    assert abs(FREQUENCIES[stochastic.argmax()] - FREQUENCIES[exact.argmax()]) <= 2 * step
