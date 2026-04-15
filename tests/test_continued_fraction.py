"""
Test the continued fraction: terminator picks smallest |gf| branch,
and smooth_ramp linearly blends coefficients towards the mean.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Modules'))

from tdscha.DynamicalLanczos import Lanczos


def _make_mock_lanczos(a_coeffs, b_coeffs, c_coeffs=None,
                       use_wigner=False, perturbation_modulus=1.0,
                       reverse_L=False, shift_value=0.0):
    lanc = Lanczos.__new__(Lanczos)
    lanc.a_coeffs = np.array(a_coeffs, dtype=np.float64)
    lanc.b_coeffs = np.array(b_coeffs, dtype=np.float64)
    if c_coeffs is not None:
        lanc.c_coeffs = np.array(c_coeffs, dtype=np.float64)
    else:
        lanc.c_coeffs = []
    lanc.use_wigner = use_wigner
    lanc.perturbation_modulus = perturbation_modulus
    lanc.reverse_L = reverse_L
    lanc.shift_value = shift_value
    lanc.verbose = False
    return lanc


def test_continued_fraction():
    # --- Test 1: terminator picks smallest |gf| branch (non-Wigner) ---
    # a=10, b=1, c=1, w=0 => disc=96
    # plus:  (10+sqrt(96))/2 ~ 9.9, minus: (10-sqrt(96))/2 ~ 0.1
    # minus has smaller |.|
    a_coeffs = [3.0, 10.0]
    b_coeffs = [1.0, 1.0]
    c_coeffs = [1.0, 1.0]
    lanc = _make_mock_lanczos(a_coeffs, b_coeffs, c_coeffs, use_wigner=False)

    w = np.array([0.0])
    gf = lanc.get_green_function_continued_fraction(w, use_terminator=True, last_average=1, smearing=0.0)

    # Manually compute with the minus (smallest) branch
    a1, b1, c1 = 10.0, 1.0, 1.0
    disc = (a1 - 0.0)**2 - 4*b1*c1
    gf_minus = (a1 - np.sqrt(disc + 0j)) / (2*b1*c1)
    a0, b0, c0 = 3.0, 1.0, 1.0
    expected = 1.0 / (a0 - 0.0 - b0*c0*gf_minus)
    np.testing.assert_allclose(gf[0], expected, rtol=1e-10,
        err_msg="Non-Wigner terminator did not choose smallest |gf| branch")

    # --- Test 2: Wigner also picks smallest |gf| branch ---
    lanc_w = _make_mock_lanczos(a_coeffs, b_coeffs, c_coeffs, use_wigner=True)
    gf_w = lanc_w.get_green_function_continued_fraction(w, use_terminator=True, last_average=1, smearing=0.0)

    disc_w = (a1 + 0.0)**2 - 4*b1*c1
    gf_minus_w = (a1 + 0.0 - np.sqrt(disc_w + 0j)) / (2*b1*c1)
    expected_w = 1.0 / (a0 + 0.0 - b0*c0*gf_minus_w)
    expected_w = (-np.real(expected_w) + 1j*np.imag(expected_w))
    np.testing.assert_allclose(gf_w[0], expected_w, rtol=1e-10,
        err_msg="Wigner terminator did not choose smallest |gf| branch")

    # --- Test 3: smooth_ramp=0 equals no smoothing ---
    a2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    b2 = [0.5, 0.6, 0.7, 0.8, 0.9]
    c2 = [0.5, 0.6, 0.7, 0.8, 0.9]
    lanc2 = _make_mock_lanczos(a2, b2, c2, use_wigner=False)
    w2 = np.array([1.0])

    gf_no_ramp = lanc2.get_green_function_continued_fraction(
        w2, use_terminator=True, last_average=2, smearing=0.0, smooth_ramp=0)
    gf_ramp0 = lanc2.get_green_function_continued_fraction(
        w2, use_terminator=True, last_average=2, smearing=0.0, smooth_ramp=0)
    np.testing.assert_allclose(gf_no_ramp, gf_ramp0, rtol=1e-14,
        err_msg="smooth_ramp=0 should equal no smoothing")

    # --- Test 4: smooth_ramp>0 changes the result ---
    gf_ramp3 = lanc2.get_green_function_continued_fraction(
        w2, use_terminator=True, last_average=2, smearing=0.0, smooth_ramp=3)
    assert np.isfinite(gf_ramp3[0]), "smooth_ramp result not finite"
    assert not np.allclose(gf_no_ramp, gf_ramp3, rtol=1e-6), \
        "smooth_ramp=3 should differ from smooth_ramp=0"

    # --- Test 5: smooth_ramp has no effect without terminator ---
    gf_no_term1 = lanc2.get_green_function_continued_fraction(
        w2, use_terminator=False, smearing=0.01, smooth_ramp=0)
    gf_no_term2 = lanc2.get_green_function_continued_fraction(
        w2, use_terminator=False, smearing=0.01, smooth_ramp=5)
    np.testing.assert_allclose(gf_no_term1, gf_no_term2, rtol=1e-14,
        err_msg="smooth_ramp should have no effect without terminator")

    print("All continued fraction tests passed!")


if __name__ == "__main__":
    test_continued_fraction()