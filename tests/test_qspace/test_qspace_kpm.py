"""
Regression test for Q-space KPM spectral function.

Compares the KPM spectral function against the continued-fraction result
for a Gamma-point optical perturbation on the standard q-space benchmark.
"""
from __future__ import print_function

import numpy as np

import cellconstructor as CC
import cellconstructor.Methods
import cellconstructor.Phonons

import sscha, sscha.Ensemble
import tdscha.QSpaceKPM as QK

from tdscha.Parallel import pprint as print

from test_qspace_lanczos import DATA_DIR, NQIRR, _setup_qspace_lanczos


N_MOMENTS = 256
N_MOMENTS_HARMONIC = 256
PEAK_RTOL = 0.08


def _find_high_gamma_mode_mapping():
    dyn = CC.Phonons.Phonons("{}/dyn_gen_pop1_".format(DATA_DIR), NQIRR)
    ws_sc, pols_sc, w_q, pols_q = dyn.DiagonalizeSupercell(return_qmodes=True)

    super_structure = dyn.structure.generate_supercell(dyn.GetSupercell())
    m = super_structure.get_masses_array()
    trans_mask = CC.Methods.get_translations(pols_sc, m)
    good_ws = ws_sc[~trans_mask]
    orig_indices = np.where(~trans_mask)[0]

    n_cell = np.prod(dyn.GetSupercell())
    nat_uc = dyn.structure.N_atoms
    nat_sc = super_structure.N_atoms
    itau = super_structure.get_itau(dyn.structure) - 1

    band_index = np.where(w_q[:, 0] > 1e-6)[0][-1]
    target_freq = w_q[band_index, 0]
    mode_index = np.argmin(np.abs(good_ws - target_freq))
    orig_mode = orig_indices[mode_index]
    pol_sc_mode = pols_sc[:, orig_mode]

    pol_gamma = np.zeros(3 * nat_uc)
    for i_sc in range(nat_sc):
        i_uc = itau[i_sc]
        pol_gamma[3 * i_uc:3 * i_uc + 3] += pol_sc_mode[3 * i_sc:3 * i_sc + 3]
    pol_gamma /= np.sqrt(n_cell)

    R1 = np.conj(pols_q[:, :, 0]).T @ pol_gamma
    band_index = np.argmax(np.abs(R1))
    return mode_index, band_index, w_q, pols_q


def _setup_qspace_kpm(iq, band_index, ignore_v3=False, ignore_v4=False):
    from test_qspace_lanczos import DATA_DIR, T, NQIRR

    dyn = CC.Phonons.Phonons("{}/dyn_gen_pop1_".format(DATA_DIR), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)

    kpm = QK.QSpaceKPM(ens, lo_to_split=None)
    kpm.ignore_harmonic = False
    kpm.ignore_v3 = ignore_v3
    kpm.ignore_v4 = ignore_v4
    kpm.init(use_symmetries=True)
    kpm.prepare_mode_q(iq, band_index)
    return kpm


def test_qspace_kpm_physics_regression(verbose=False):
    mode_index, band_index, w_q, pols_q = _find_high_gamma_mode_mapping()

    if verbose:
        print()
        print("=== Q-space KPM physics regression ===")
        print("Testing Gamma band {} (freq {:.2f} cm-1)".format(
            band_index, w_q[band_index, 0] * CC.Units.RY_TO_CM))

    # Setup and run the Lanczos file
    lanc_cf = _setup_qspace_lanczos(0, band_index)
    w_mode = w_q[band_index, 0]

    kpm = _setup_qspace_kpm(0, band_index)
    kpm.run_KPM(N_MOMENTS, verbose=False)

    w_min = max(0.0, 0.3 * w_mode)
    w_max = 1.4 * w_mode
    w_array = np.linspace(w_min, w_max, 81)

    cf_smearing = 0.10 * w_mode
    spectral_cf = -np.imag(
        lanc_cf.get_green_function_continued_fraction(
            w_array, use_terminator=False, smearing=cf_smearing))
    spectral_kpm = kpm.get_spectral_function_KPM(w_array, regularization="jackson")

    peak_cf = w_array[np.argmax(spectral_cf)]
    peak_kpm = w_array[np.argmax(spectral_kpm)]
    rel_peak = abs(peak_kpm - peak_cf) / peak_cf

    if verbose:
        print("Peak CF:  {:.6f} cm-1".format(peak_cf * CC.Units.RY_TO_CM))
        print("Peak KPM: {:.6f} cm-1".format(peak_kpm * CC.Units.RY_TO_CM))
        print("Relative peak diff: {:.6e}".format(rel_peak))
        import matplotlib.pyplot as plt
        plt.plot(w_array * CC.Units.RY_TO_CM, spectral_cf, label="Continued fraction")
        plt.plot(w_array * CC.Units.RY_TO_CM, spectral_kpm, label="KPM")
        plt.axvline(w_mode * CC.Units.RY_TO_CM, color="C0", linestyle="--", label="CF peak")
        plt.legend()
        plt.show()

    assert rel_peak < PEAK_RTOL, (
        "KPM peak differs too much from continued fraction: {:.4e}".format(rel_peak))


def test_qspace_kpm_harmonic_peak_exact(verbose=False):
    mode_index, band_index, w_q, pols_q = _find_high_gamma_mode_mapping()
    w_mode = w_q[band_index, 0]

    if verbose:
        print()
        print("=== Q-space KPM exact harmonic peak ===")
        print("Testing Gamma band {} (harmonic freq {:.2f} cm-1)".format(
            band_index, w_mode * CC.Units.RY_TO_CM))

    kpm = _setup_qspace_kpm(0, band_index, ignore_v3=True, ignore_v4=True)
    kpm.run_KPM(N_MOMENTS_HARMONIC, bound_factor=1.0, verbose=False)

    dw = 0.002 * w_mode
    w_array = w_mode + dw * np.arange(-100, 101)
    spectral_kpm = kpm.get_spectral_function_KPM(w_array, regularization="jackson")
    i_peak = np.argmax(spectral_kpm)

    if verbose:
        print("Harmonic peak: {:.10e} Ry".format(w_mode))
        print("KPM peak:      {:.10e} Ry".format(w_array[i_peak]))

    assert i_peak == len(w_array) // 2, "KPM harmonic peak is not exactly at the harmonic frequency"


def test_qspace_kpm_save_restore_continuation(tmp_path):
    """Test that save_status/load_status enables exact continuation of KPM."""
    import os
    mode_index, band_index, w_q, pols_q = _find_high_gamma_mode_mapping()

    # Reference: run 10 moments in one shot
    kpm_ref = _setup_qspace_kpm(0, band_index)
    kpm_ref.run_KPM(10, verbose=False)
    moments_ref = kpm_ref.kpm_moments.copy()

    # Split run: 5 moments, save, restore on fresh object, continue to 10
    kpm_a = _setup_qspace_kpm(0, band_index)
    kpm_a.run_KPM(5, verbose=False)
    save_file = os.path.join(str(tmp_path), "kpm_checkpoint")
    kpm_a.save_status(save_file)

    kpm_b = _setup_qspace_kpm(0, band_index)
    kpm_b.load_status(save_file)
    kpm_b.run_KPM(10, verbose=False)

    # Moments must match exactly (deterministic, no floating-point reordering)
    np.testing.assert_array_equal(
        kpm_b.kpm_moments, moments_ref,
        err_msg="Continued KPM moments differ from one-shot reference")

    # Spectral functions must also match
    w_mode = w_q[band_index, 0]
    w_array = np.linspace(0.3 * w_mode, 1.4 * w_mode, 81)
    spec_ref = kpm_ref.get_spectral_function_KPM(w_array)
    spec_cont = kpm_b.get_spectral_function_KPM(w_array)
    np.testing.assert_array_equal(
        spec_cont, spec_ref,
        err_msg="Continued KPM spectral function differs from one-shot reference")


if __name__ == "__main__":
    test_qspace_kpm_physics_regression(verbose=True)
    test_qspace_kpm_harmonic_peak_exact(verbose=True)
