"""Static limits of the independent analytic benchmark, without Julia."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                     / 'benchmarks' / 'interpolation'))
import polynomial_reference as ref


@pytest.mark.parametrize('thermal', [0., .3])
@pytest.mark.parametrize('quartic', [False, True])
def test_static_degenerate_pair_limit(thermal, quartic):
    # Q=0: all pair frequencies coincide. Independently evaluate the
    # thermodynamic static pair susceptibility, including the Bose derivative.
    n = 18
    w = ref.frequency(np.arange(n)/n)
    occupation = np.zeros(n) if thermal == 0 else 1/np.expm1(w/thermal)
    derivative = np.zeros(n) if thermal == 0 else -occupation*(1+occupation)/thermal
    lam = -((1+2*occupation)/(2*w)-derivative)/(4*w*w)
    v, W = ref.vertices(n, 0.)
    pair = np.linalg.solve(np.eye(n)-lam[:, None]*W, lam*v) if quartic else lam*v
    expected = np.vdot(v, pair)
    sigma, response = ref.response(n, 0., 0j, quartic=quartic, thermal=thermal)
    assert np.isfinite(response)
    np.testing.assert_allclose(sigma, expected, rtol=1e-13)
