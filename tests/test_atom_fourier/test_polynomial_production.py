"""Analytic finite-range Phi4 versus the actual Julia interpolation operator.

No Monte Carlo tolerances: degree-seven Gaussian cubature exactly integrates
the polynomial force estimator. Missing Julia is a skip, a failed runtime is
an error. See benchmarks/interpolation/README.md for scope and conventions.
"""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault('JULIA_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / 'benchmarks' / 'interpolation'))
import polynomial_reference as reference
import production_validation as production

pytestmark = pytest.mark.skipif(not production.JE.available(),
                                reason='Julia backend not installed')


@pytest.mark.parametrize('nc,nf,Q,thermal', [
    (9, 9, 0., 0.), (9, 18, 0., 0.), (9, 36, 1/3, 0.),
    (9, 72, 1/3, .3), (8, 24, .25, .3),
])
def test_polynomial_production_operator_and_response(nc, nf, Q, thermal):
    production.probe(nc, nf, Q, embedded=True, thermal=thermal)


@pytest.mark.parametrize('parent', [False, True])
@pytest.mark.parametrize('ignore_v3,ignore_v4', [(True, False), (False, True), (True, True)])
def test_independent_vertex_switches(parent, ignore_v3, ignore_v4):
    production.probe(9, 9 if parent else 18, 1/3, embedded=True,
                     ignore_v3=ignore_v3, ignore_v4=ignore_v4, thermal=.3,
                     parent=parent)


def test_dense_complex_mode_blocks():
    production.rotated_action()


@pytest.mark.parametrize('mutation', ['missing_scale4', 'conjugate_kernel'])
def test_polynomial_detects_broken_interpolation(mutation):
    _, _, errors = production.probe(9, 18, 1/3, embedded=True, check=False,
                                    mutation=mutation)
    assert errors['D4'] > .01


def test_cutoff_must_be_resolved():
    _, _, errors = production.probe(3, 18, 0., check=False)
    assert errors['D4'] > .01


def test_polynomial_force_and_response_first_control():
    report = reference.benchmark()
    assert report['response_first_external_Q_error'] > .01
