"""
Test that compares the spectral function computed via Q-space Lanczos
and real-space Lanczos (both using Wigner mode) on the same SnTe-like system.

The benchmark is on the Green function (real and imaginary parts)
evaluated at omega=0 and at the frequency of the perturbed mode.

This test perturbs a Gamma-point optical mode. Because the optical modes
at Gamma are triply degenerate, we must carefully project the supercell
eigenvector onto q-space bands to get the correct mapping.
"""
from __future__ import print_function

import numpy as np
import os
import pytest

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Methods
import cellconstructor.symmetries

import sscha, sscha.Ensemble
import tdscha.DynamicalLanczos as DL

from tdscha.Parallel import pprint as print

# Test parameters
N_STEPS = 10
T = 250
NQIRR = 3

# Tolerance on the Green function comparison (relative)
GF_RTOL = 0.01  # 1% relative tolerance

# Data directory (reuse test_julia data)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')


def _find_gamma_mode_mapping():
    """Find the correct mapping between a supercell mode and q-space band at Gamma.

    Because the Gamma optical modes are degenerate, we cannot simply match by
    frequency — the eigenvector gauge within the degenerate subspace may differ.
    Instead, we project the supercell eigenvector onto the q-space band basis
    to find the correct band_index.

    Returns
    -------
    mode_index : int
        Supercell mode index (after translation removal) for the selected mode.
    band_index : int
        Matching band index at Gamma (0-based).
    w_q : ndarray
        Frequencies at each q-point, shape (n_bands, n_q).
    pols_q : ndarray
        Polarization vectors at each q-point, shape (3*nat_uc, n_bands, n_q).
    """
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    ws_sc, pols_sc, w_q, pols_q = dyn.DiagonalizeSupercell(return_qmodes=True)

    super_structure = dyn.structure.generate_supercell(dyn.GetSupercell())
    m = super_structure.get_masses_array()  # (nat_sc,)
    trans_mask = CC.Methods.get_translations(pols_sc, m)
    good_ws = ws_sc[~trans_mask]
    orig_indices = np.where(~trans_mask)[0]

    n_bands = 3 * dyn.structure.N_atoms
    n_cell = np.prod(dyn.GetSupercell())
    nat_uc = dyn.structure.N_atoms
    nat_sc = super_structure.N_atoms
    itau = super_structure.get_itau(dyn.structure) - 1  # 0-indexed

    # Find first non-acoustic Gamma band
    for band in range(n_bands):
        if w_q[band, 0] > 1e-6:
            target_freq = w_q[band, 0]
            break
    else:
        raise ValueError("No non-acoustic Gamma mode found")

    # Find the supercell mode matching this frequency
    mode_index = np.argmin(np.abs(good_ws - target_freq))
    freq_diff = abs(good_ws[mode_index] - target_freq)
    assert freq_diff < 1e-8, (
        "Mode frequency mismatch: {:.10e} vs {:.10e}".format(
            good_ws[mode_index], target_freq))

    # Project the supercell eigenvector onto Gamma q-space bands
    orig_mode = orig_indices[mode_index]
    pol_sc_mode = pols_sc[:, orig_mode]  # (3*N_sc,)

    # Sum over supercell images to get Gamma component
    pol_gamma = np.zeros(3 * nat_uc)
    for i_sc in range(nat_sc):
        i_uc = itau[i_sc]
        pol_gamma[3*i_uc:3*i_uc+3] += pol_sc_mode[3*i_sc:3*i_sc+3]
    pol_gamma /= np.sqrt(n_cell)

    # Project onto q-space bands: R1[nu] = conj(pol_q[:, nu, 0]).T @ pol_gamma
    R1 = np.conj(pols_q[:, :, 0]).T @ pol_gamma
    band_index = np.argmax(np.abs(R1))

    assert np.abs(R1[band_index]) > 0.99, (
        "Projection onto q-space bands is not clean: |R1| = {}, R1 = {}".format(
            np.abs(R1), R1))

    return mode_index, band_index, w_q, pols_q


def _setup_realspace_lanczos(mode_index):
    """Set up and run a real-space Lanczos calculation (Wigner, serial C)."""
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)

    lanc = DL.Lanczos(ens, lo_to_split=None)
    lanc.ignore_harmonic = False
    lanc.ignore_v3 = False
    lanc.ignore_v4 = False
    lanc.use_wigner = True
    lanc.mode = DL.MODE_FAST_SERIAL
    lanc.init(use_symmetries=True)
    lanc.prepare_mode(mode_index)
    lanc.run_FT(N_STEPS, run_simm=True, verbose=False)
    return lanc


def _setup_qspace_lanczos(iq, band_index):
    """Set up and run a Q-space Lanczos calculation."""
    try:
        import tdscha.QSpaceLanczos as QL
    except ImportError:
        pytest.skip("Julia not available for QSpaceLanczos")

    if not QL.__JULIA_EXT__:
        pytest.skip("Julia extension not loaded")

    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)

    qlanc = QL.QSpaceLanczos(ens, lo_to_split=None)
    qlanc.ignore_harmonic = False
    qlanc.ignore_v3 = False
    qlanc.ignore_v4 = False
    qlanc.init(use_symmetries=True)
    qlanc.prepare_mode_q(iq, band_index)
    qlanc.run_FT(N_STEPS, verbose=False)
    return qlanc


def test_qspace_vs_realspace_green_function(verbose=False):
    """
    Compare Green functions from Q-space and real-space Lanczos
    at omega=0 and at the perturbed mode frequency.

    The renormalized frequency from 1/Re[G(0)] must agree within tolerance.
    """
    try:
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia not available for QSpaceLanczos")

    mode_index, band_index, w_q, pols_q = _find_gamma_mode_mapping()

    if verbose:
        print()
        print("=== Q-space vs real-space Lanczos comparison ===")
        print("Testing Gamma band {} (freq {:.2f} cm-1)".format(
            band_index, w_q[band_index, 0] * CC.Units.RY_TO_CM))
        print("  -> supercell mode index {}".format(mode_index))
        print()

    # Run both calculations
    lanc_real = _setup_realspace_lanczos(mode_index)
    lanc_qspace = _setup_qspace_lanczos(0, band_index)

    # Get the harmonic frequency of the perturbed mode (in Ry)
    w_mode = lanc_real.w[mode_index]

    if verbose:
        print("Lanczos coefficients comparison:")
        print("  a_0 real-space: {:.10e}".format(lanc_real.a_coeffs[0]))
        print("  a_0 q-space:    {:.10e}".format(lanc_qspace.a_coeffs[0]))
        print("  b_0 real-space: {:.10e}".format(lanc_real.b_coeffs[0]))
        print("  b_0 q-space:    {:.10e}".format(lanc_qspace.b_coeffs[0]))
        print()

    # Small imaginary broadening
    smearing = w_mode * 0.1

    # Evaluate the Green function at omega=0 and at the mode frequency
    w_points = np.array([0.0, w_mode])

    gf_real = lanc_real.get_green_function_continued_fraction(
        w_points, use_terminator=False, smearing=smearing
    )
    gf_qspace = lanc_qspace.get_green_function_continued_fraction(
        w_points, use_terminator=False, smearing=smearing
    )

    real_real = np.real(gf_real)
    real_qspace = np.real(gf_qspace)
    spectral_real = -np.imag(gf_real)
    spectral_qspace = -np.imag(gf_qspace)

    if verbose:
        print("At omega = 0:")
        print("  Re[G] real-space:  {:.10e}".format(real_real[0]))
        print("  Re[G] q-space:     {:.10e}".format(real_qspace[0]))
        print("  Im[G] real-space:  {:.10e}".format(spectral_real[0]))
        print("  Im[G] q-space:     {:.10e}".format(spectral_qspace[0]))

        print()
        print("At omega = w_mode ({:.2f} cm-1):".format(
            w_mode * CC.Units.RY_TO_CM))
        print("  Re[G] real-space:  {:.10e}".format(real_real[1]))
        print("  Re[G] q-space:     {:.10e}".format(real_qspace[1]))
        print("  Im[G] real-space:  {:.10e}".format(spectral_real[1]))
        print("  Im[G] q-space:     {:.10e}".format(spectral_qspace[1]))

    # --- Assertions ---
    # Renormalized frequency from Re[G(0)]: w^2 = 1/Re[G(0)]
    w2_real = 1.0 / real_real[0]
    w2_qspace = 1.0 / real_qspace[0]
    freq_real = np.sign(w2_real) * np.sqrt(np.abs(w2_real)) * CC.Units.RY_TO_CM
    freq_qspace = np.sign(w2_qspace) * np.sqrt(np.abs(w2_qspace)) * CC.Units.RY_TO_CM

    if verbose:
        print()
        print("Renormalized frequency (real-space):  {:.6f} cm-1".format(freq_real))
        print("Renormalized frequency (Q-space):     {:.6f} cm-1".format(freq_qspace))
        print("Difference:                           {:.6f} cm-1".format(
            abs(freq_real - freq_qspace)))

    assert abs(freq_real - freq_qspace) < 0.1, (
        "Renormalized frequencies differ too much: "
        "real-space={:.4f}, q-space={:.4f} cm-1".format(freq_real, freq_qspace)
    )

    # Compare spectral function at omega = w_mode
    ref_spectral = max(abs(spectral_real[1]), abs(spectral_qspace[1]))
    if ref_spectral > 1e-15:
        rel_diff_spectral = abs(spectral_real[1] - spectral_qspace[1]) / ref_spectral
        print("Relative diff in spectral function at w_mode: {:.6e}".format(
            rel_diff_spectral))
        assert rel_diff_spectral < GF_RTOL, (
            "Spectral functions at w_mode differ by {:.4e} "
            "(tolerance: {})".format(rel_diff_spectral, GF_RTOL)
        )

    # Compare real part of GF at omega = w_mode
    ref_real_wm = max(abs(real_real[1]), abs(real_qspace[1]))
    if ref_real_wm > 1e-15:
        rel_diff_real = abs(real_real[1] - real_qspace[1]) / ref_real_wm
        print("Relative diff in Re[G] at w_mode:              {:.6e}".format(
            rel_diff_real))
        assert rel_diff_real < GF_RTOL, (
            "Real parts of GF at w_mode differ by {:.4e} "
            "(tolerance: {})".format(rel_diff_real, GF_RTOL)
        )

    if verbose:
        print()
        print("=== All benchmarks passed ===")


if __name__ == "__main__":
    test_qspace_vs_realspace_green_function(verbose=True)
