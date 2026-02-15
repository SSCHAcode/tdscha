"""
Test that compares the spectral function computed via Wigner mode
and normal (non-Wigner) mode on the same SnTe-like system.

The benchmark is on the Green function (real and imaginary parts)
evaluated at omega=0 and at the frequency of the perturbed mode.
"""
from __future__ import print_function

import numpy as np
import os

import cellconstructor as CC
import cellconstructor.Phonons

import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL

from tdscha.Parallel import pprint as print


MODE_INDEX = 10
N_STEPS = 50
T = 250
NQIRR = 3

# Tolerance on the Green function comparison (relative)
GF_RTOL = 0.001  # 0.1% relative tolerance on spectral function values


def _setup_lanczos(use_wigner):
    """Set up and run a Lanczos calculation with the given Wigner flag."""
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    dyn = CC.Phonons.Phonons("data/dyn_gen_pop1_", NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin("data", 1)

    lanc = DL.Lanczos(ens)
    lanc.ignore_harmonic = False
    lanc.ignore_v3 = False
    lanc.ignore_v4 = False
    lanc.use_wigner = use_wigner
    lanc.mode = DL.MODE_FAST_SERIAL
    lanc.init(use_symmetries=True)
    lanc.prepare_mode(MODE_INDEX)
    lanc.run_FT(N_STEPS, run_simm=use_wigner, verbose=False)
    return lanc


def test_wigner_vs_nowigner_green_function(verbose=False):
    """
    Compare Green functions from Wigner and non-Wigner Lanczos
    at omega=0 and at the perturbed mode frequency.

    The spectral function (imaginary part of the Green function)
    must agree between the two approaches within tolerance.
    """
    lanc_wigner = _setup_lanczos(use_wigner=True)
    lanc_nowigner = _setup_lanczos(use_wigner=False)

    # Get the harmonic frequency of the perturbed mode (in Ry)
    w_mode = lanc_wigner.w[MODE_INDEX]

    # Small imaginary broadening to regularize the Green function
    smearing = w_mode * 0.1

    # Evaluate the Green function at omega=0 and at the mode frequency
    w_points = np.array([0.0, w_mode])

    gf_wigner = lanc_wigner.get_green_function_continued_fraction(
        w_points, use_terminator=False, smearing=smearing
    )
    gf_nowigner = lanc_nowigner.get_green_function_continued_fraction(
        w_points, use_terminator=False, smearing=smearing
    )

    spectral_wigner = -np.imag(gf_wigner)
    spectral_nowigner = -np.imag(gf_nowigner)

    real_wigner = np.real(gf_wigner)
    real_nowigner = np.real(gf_nowigner)

    # --- Benchmark at omega = 0 ---
    if verbose:
        print()
        print("=== Green function comparison: Wigner vs non-Wigner ===")
        print()
        print(f"At omega = 0:")
        print(f"  Origina w {lanc_wigner.w[MODE_INDEX] * CC.Units.RY_TO_CM:.2f} cm-1")
        print(f"  Re[G] Wigner:    {real_wigner[0]:.10e}")
        print(f"  Re[G] noWigner:  {real_nowigner[0]:.10e}")
        print(f"  Im[G] Wigner:    {spectral_wigner[0]:.10e}")
        print(f"  Im[G] noWigner:  {spectral_nowigner[0]:.10e}")

        # --- Benchmark at omega = w_mode ---
        print()
        print(f"At omega = w_mode ({w_mode * CC.Units.RY_TO_CM:.2f} cm-1):")
        print(f"  Re[G] Wigner:    {real_wigner[1]:.10e}")
        print(f"  Re[G] noWigner:  {real_nowigner[1]:.10e}")
        print(f"  Im[G] Wigner:    {spectral_wigner[1]:.10e}")
        print(f"  Im[G] noWigner:  {spectral_nowigner[1]:.10e}")

    # --- Assertions ---
    # At omega=0, the real part of the Green function gives the
    # renormalized frequency: w^2 = 1/Re[G(0)].
    # Compare the extracted frequencies.
    w2_wigner = 1.0 / real_wigner[0]
    w2_nowigner = 1.0 / real_nowigner[0]
    freq_wigner = np.sign(w2_wigner) * np.sqrt(np.abs(w2_wigner)) * CC.Units.RY_TO_CM
    freq_nowigner = np.sign(w2_nowigner) * np.sqrt(np.abs(w2_nowigner)) * CC.Units.RY_TO_CM

    if verbose:
        print()
        print(f"Renormalized frequency (Wigner):    {freq_wigner:.6f} cm-1")
        print(f"Renormalized frequency (noWigner):  {freq_nowigner:.6f} cm-1")
        print(f"Difference:                         {abs(freq_wigner - freq_nowigner):.6f} cm-1")

    # Frequency difference should be small (< 0.01 cm-1)
    assert abs(freq_wigner - freq_nowigner) < 0.01, (
        f"Renormalized frequencies differ too much: "
        f"Wigner={freq_wigner:.4f}, noWigner={freq_nowigner:.4f} cm-1"
    )

    # Compare spectral function at omega = w_mode
    # Use the larger value as reference for relative comparison
    ref_spectral = max(abs(spectral_wigner[1]), abs(spectral_nowigner[1]))
    if ref_spectral > 1e-15:
        rel_diff_spectral = abs(spectral_wigner[1] - spectral_nowigner[1]) / ref_spectral
        print(f"Relative diff in spectral function at w_mode: {rel_diff_spectral:.6e}")
        assert rel_diff_spectral < GF_RTOL, (
            f"Spectral functions at w_mode differ by {rel_diff_spectral:.4e} "
            f"(tolerance: {GF_RTOL})"
        )

    # Compare real part of GF at omega = w_mode
    ref_real = max(abs(real_wigner[1]), abs(real_nowigner[1]))
    if ref_real > 1e-15:
        rel_diff_real = abs(real_wigner[1] - real_nowigner[1]) / ref_real
        print(f"Relative diff in Re[G] at w_mode:              {rel_diff_real:.6e}")
        assert rel_diff_real < GF_RTOL, (
            f"Real parts of GF at w_mode differ by {rel_diff_real:.4e} "
            f"(tolerance: {GF_RTOL})"
        )

    if verbose:
        print()
        print("=== All benchmarks passed ===")


if __name__ == "__main__":
    test_wigner_vs_nowigner_green_function(verbose=True)
