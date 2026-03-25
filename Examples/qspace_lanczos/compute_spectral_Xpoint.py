"""
Q-Space Lanczos: phonon spectral function at the X point of SnTe
================================================================

This example computes the anharmonic phonon spectral function for SnTe
at the X point (zone boundary) using the Q-space Lanczos algorithm.

The Q-space Lanczos exploits Bloch momentum conservation to drastically
reduce the size of the two-phonon sector, giving a speedup proportional
to the number of unit cells in the supercell (N_cell = 8 for this 2x2x2
supercell).

Requirements:
    - Julia with SparseArrays package
    - spglib
    - The SnTe ensemble from tests/test_julia/data/

Usage:
    python compute_spectral_Xpoint.py

    # Or with MPI parallelism:
    mpirun -np 4 python compute_spectral_Xpoint.py
"""
from __future__ import print_function

import numpy as np
import os, sys

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Units

import sscha, sscha.Ensemble
import tdscha.QSpaceLanczos as QL

from tdscha.Parallel import pprint as print

# ========================
# Parameters
# ========================
# Path to the SnTe ensemble data (2x2x2 supercell, 2 atoms/unit cell)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'tests', 'test_julia', 'data')
NQIRR = 3           # Number of irreducible q-points in the dynamical matrix
TEMPERATURE = 250    # Temperature in Kelvin
N_STEPS = 50         # Number of Lanczos steps (increase for production)
SAVE_DIR = "output"  # Directory for checkpoints and results


def main():
    # ========================
    # 1. Load ensemble
    # ========================
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, TEMPERATURE)
    ens.load_bin(DATA_DIR, 1)

    # ========================
    # 2. Create Q-space Lanczos
    # ========================
    qlanc = QL.QSpaceLanczos(ens)
    qlanc.ignore_v3 = False   # Include 3-phonon interactions
    qlanc.ignore_v4 = False   # Include 4-phonon interactions
    qlanc.init(use_symmetries=True)

    # ========================
    # 3. Inspect q-points and pick the X point
    # ========================
    print("Available q-points:")
    for iq, q in enumerate(qlanc.q_points):
        freqs = qlanc.w_q[:, iq] * CC.Units.RY_TO_CM
        print("  iq={}: q = ({:8.5f}, {:8.5f}, {:8.5f})  "
              "freqs = {} cm-1".format(iq, q[0], q[1], q[2],
              np.array2string(freqs, precision=1, separator=', ')))

    # The zone-boundary X point in a 2x2x2 FCC supercell is at iq=5,6,7
    # (equivalent by cubic symmetry). We pick iq=5.
    iq_pert = 5
    print()
    print("Selected q-point: iq={}".format(iq_pert))
    print("  q = {}".format(qlanc.q_points[iq_pert]))
    print()

    # ========================
    # 4. Run Lanczos for each band at the X point
    # ========================
    n_bands = qlanc.n_bands  # 6 bands for SnTe (3 * 2 atoms)

    for band in range(n_bands):
        freq = qlanc.w_q[band, iq_pert] * CC.Units.RY_TO_CM

        # Skip acoustic modes (zero frequency at Gamma only; at X all modes
        # have finite frequency, but we check anyway for safety)
        if qlanc.w_q[band, iq_pert] < 1e-6:
            print("Skipping acoustic band {} (freq = {:.2f} cm-1)".format(
                band, freq))
            continue

        print("=" * 50)
        print("Band {}: {:.2f} cm-1".format(band, freq))
        print("=" * 50)

        qlanc.prepare_mode_q(iq_pert, band)
        qlanc.run_FT(N_STEPS, save_dir=SAVE_DIR,
                      prefix="Xpoint_band{}".format(band), verbose=True)
        qlanc.save_status(os.path.join(SAVE_DIR,
                          "Xpoint_band{}_final.npz".format(band)))

    # ========================
    # 5. Plot the total spectral function at X
    # ========================
    print()
    print("=" * 50)
    print("Computing total spectral function at X point")
    print("=" * 50)

    # Frequency grid (cm-1)
    w_cm = np.linspace(0, 200, 500)
    w_ry = w_cm / CC.Units.RY_TO_CM
    smearing = 3.0 / CC.Units.RY_TO_CM  # 3 cm-1 broadening

    total_spectral = np.zeros_like(w_cm)

    for band in range(n_bands):
        if qlanc.w_q[band, iq_pert] < 1e-6:
            continue

        result_file = os.path.join(SAVE_DIR,
                                   "Xpoint_band{}_final.npz".format(band))
        if not os.path.exists(result_file):
            print("  Band {} result not found, skipping".format(band))
            continue

        # Load and compute Green function
        from tdscha.DynamicalLanczos import Lanczos
        lanc_tmp = Lanczos()
        lanc_tmp.load_status(result_file)

        gf = lanc_tmp.get_green_function_continued_fraction(
            w_ry, smearing=smearing, use_terminator=True)
        spectral = -np.imag(gf)
        total_spectral += spectral

        # Print the renormalized frequency from G(0)
        gf0 = lanc_tmp.get_green_function_continued_fraction(
            np.array([0.0]), smearing=smearing, use_terminator=True)
        w2 = 1.0 / np.real(gf0[0])
        w_ren = np.sign(w2) * np.sqrt(np.abs(w2)) * CC.Units.RY_TO_CM
        print("  Band {}: harmonic = {:.2f} cm-1, "
              "renormalized = {:.2f} cm-1".format(
                  band,
                  qlanc.w_q[band, iq_pert] * CC.Units.RY_TO_CM,
                  w_ren))

    # Save the spectrum to a text file
    output_file = os.path.join(SAVE_DIR, "spectral_Xpoint.dat")
    np.savetxt(output_file,
               np.column_stack([w_cm, total_spectral]),
               header="omega(cm-1)  spectral_function(arb.units)",
               fmt="%.6f")
    print()
    print("Spectral function saved to {}".format(output_file))

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(w_cm, total_spectral, 'b-', linewidth=1.5)

        # Mark harmonic frequencies
        for band in range(n_bands):
            freq = qlanc.w_q[band, iq_pert] * CC.Units.RY_TO_CM
            if freq > 1e-3:
                ax.axvline(freq, color='r', linestyle='--', alpha=0.5,
                           linewidth=0.8)

        ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.set_ylabel("Spectral function (arb. units)")
        ax.set_title("SnTe phonon spectral function at X point (T = {} K)".format(
            TEMPERATURE))
        ax.set_xlim(0, 200)
        ax.legend(["Anharmonic (TD-SCHA)", "Harmonic frequencies"],
                  loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, "spectral_Xpoint.pdf"))
        print("Plot saved to {}/spectral_Xpoint.pdf".format(SAVE_DIR))
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
