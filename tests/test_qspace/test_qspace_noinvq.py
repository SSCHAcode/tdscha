"""
Test that compares the static limit of green functions at a
q point different from Gamma on the SnTe-like system.

This compares the new q-space implementation, the old real-space implementation of Lanczos
and the get_free_energy_hessian one.
"""

import numpy as np
import os, pytest

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Units

import sscha, sscha.Ensemble
import tdscha.DynamicalLanczos as DL
import tdscha.QSpaceLanczos as QL


N_STEPS = 30
T = 250
NQIRR = 3

IQ = 1
NMODE = 0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')


def test_qspace_noinvq(test_v3 = True, test_v4=True):
    # Load the original dynamical matrix
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)

    # Load the ensemble
    ensemble = sscha.Ensemble.Ensemble(dyn, T)
    ensemble.load_bin(DATA_DIR, 1)

    # Compute the free energy hessian
    hessian = dyn.Copy()
    if test_v3:
        hessian = ensemble.get_free_energy_hessian(include_v4 = test_v4)

    # Initialize the q-space and real-space Lanczos solvers
    q_lanczos = QL.QSpaceLanczos(ensemble)
    q_lanczos.ignore_v3 = not test_v3
    if test_v3:
        q_lanczos.ignore_v4 = not test_v4
    else:
        q_lanczos.ignore_v4 = True
    q_lanczos.init()

    r_lanczos = DL.Lanczos(ensemble)
    r_lanczos.ignore_v3 = not test_v3
    if test_v3:
        r_lanczos.ignore_v4 = not test_v4
    else:
        r_lanczos.ignore_v4 = True
    r_lanczos.init()

    # Run the q-space Lanczos solver
    q_lanczos.prepare_mode_q(IQ, NMODE)

    # Get the equivalence with r_lanczos mode
    mode_id = np.argmin(np.abs(r_lanczos.w - q_lanczos.w_q[NMODE, IQ]))
    r_lanczos.prepare_mode(mode_id)

    # Run the two lanczos
    q_lanczos.run_FT(N_STEPS)
    r_lanczos.run_FT(N_STEPS)

    # Get the static frequencies
    gf_q = q_lanczos.get_green_function_continued_fraction(np.array([0.0]), smearing=0.0, use_terminator=False)
    gf_r = r_lanczos.get_green_function_continued_fraction(np.array([0.0]), smearing=0.0, use_terminator=False)

    w_qlanczos = np.sqrt(np.abs(np.real(1.0/gf_q[0]))) * np.sign(np.real(gf_q[0]))
    w_rlanczos = np.sqrt(np.abs(np.real(1.0/gf_r[0]))) * np.sign(np.real(gf_r[0]))

    if verbose:
        print()
        print(" ---------- TEST SINGLE Q RESULTS ---------- ")
        print(" v3 = ", test_v3, " v4 = ", test_v4)
        print(f"Q-space Lanczos frequency: {w_qlanczos*CC.Units.RY_TO_CM:.6f} cm-1")
        print(f"Real-space Lanczos frequency: {w_rlanczos*CC.Units.RY_TO_CM:.6f} cm-1")

    # Compute the free energy hessian frequency
    Dmat = hessian.dynmats[IQ]
    m_sqrt = np.sqrt(q_lanczos.m)
    Dmat /= np.outer(m_sqrt, m_sqrt)
    w2_hessian = q_lanczos.pols_q[:, NMODE, IQ] @ Dmat @ q_lanczos.pols_q[:, NMODE, IQ]
    w_hessian = np.real(np.sqrt(np.abs(w2_hessian)) * np.sign(w2_hessian))

    if verbose:
        print(f"Free energy hessian frequency: {w_hessian*CC.Units.RY_TO_CM:.6f} cm-1")

    # Compare the results
    assert np.isclose(w_qlanczos, w_rlanczos, atol=1e-5), f"Q-space and real-space Lanczos frequencies differ: {w_qlanczos*CC.Units.RY_TO_CM:.6f} cm-1 vs {w_rlanczos*CC.Units.RY_TO_CM:.6f} cm-1"
    assert np.isclose(w_qlanczos, w_hessian, atol=1e-5), f"Q-space Lanczos and free energy hessian frequencies differ: {w_qlanczos*CC.Units.RY_TO_CM:.6f} cm-1 vs {w_hessian*CC.Units.RY_TO_CM:.6f} cm-1"
    

    
    
if __name__ == "__main__":
    verbose = True
    test_qspace_noinvq(test_v3=True, test_v4=False)
    test_qspace_noinvq(test_v3=True, test_v4=True)
