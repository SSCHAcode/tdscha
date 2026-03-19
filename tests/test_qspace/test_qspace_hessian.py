"""
Test Q-Space Hessian against reference ens.get_free_energy_hessian().

Uses the SnTe 2×2×2 test data from tests/test_julia/data/ (T=250K, NQIRR=3).
"""
from __future__ import print_function

import numpy as np
import os
import pytest

import cellconstructor as CC
import cellconstructor.Phonons

import sscha, sscha.Ensemble

# Test parameters
T = 250
NQIRR = 3

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')


def _load_ensemble():
    """Load the SnTe test ensemble."""
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)
    return ens


def test_qspace_hessian_gamma():
    """Compare Gamma-point Hessian eigenvalues: Q-space vs reference."""
    try:
        import tdscha.QSpaceHessian as QH
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia or QSpaceHessian not available")

    ens = _load_ensemble()

    # -- Reference: ens.get_free_energy_hessian --
    ref_hessian = ens.get_free_energy_hessian(include_v4=True,
                                               use_symmetries=True)
    ref_dynmat_gamma = ref_hessian.dynmats[0]
    w, pols = ref_hessian.DyagDinQ(0)
    w = w**2 * np.sign(w)  # Convert to eigenvalues of the Hessian
    # Filter out acoustic (near-zero) eigenvalues
    ref_non_acoustic = w[np.abs(w) > 1e-8]

    # -- Q-space Hessian at Gamma --
    qh = QH.QSpaceHessian(ens, verbose=True, lo_to_split=None)
    qh.init(use_symmetries=True)
    H_q_gamma = qh.compute_hessian_at_q(0, tol=1e-8, max_iters=1000)

    # Transform to Cartesian
    pol = qh.pols_q[:, :, 0]
    Phi_q = pol @ H_q_gamma @ np.conj(pol).T
    qspace_evals = np.sort(np.real(np.linalg.eigvalsh(Phi_q)))
    qspace_non_acoustic = qspace_evals[np.abs(qspace_evals) > 1e-8]

    print()
    print("=== Gamma-point Hessian eigenvalue comparison ===")
    print("Reference (ens.get_free_energy_hessian):")
    for i, ev in enumerate(ref_non_acoustic):
        print("  {:2d}: {:.8e}".format(i, ev))
    print("Q-space Hessian:")
    for i, ev in enumerate(qspace_non_acoustic):
        print("  {:2d}: {:.8e}".format(i, ev))

    # Compare eigenvalues
    n_compare = min(len(ref_non_acoustic), len(qspace_non_acoustic))
    assert n_compare > 0, "No non-acoustic eigenvalues found"

    # Use relative tolerance on eigenvalues
    for i in range(n_compare):
        ref_val = ref_non_acoustic[i]
        qsp_val = qspace_non_acoustic[i]
        rel_diff = abs(ref_val - qsp_val) / max(abs(ref_val), 1e-15)
        print("  eigenvalue {}: ref={:.8e}, qsp={:.8e}, rel_diff={:.2e}".format(
            i, ref_val, qsp_val, rel_diff))
        assert rel_diff < 0.05, (
            "Eigenvalue {} differs too much: ref={:.6e}, qsp={:.6e}, "
            "rel_diff={:.4e}".format(i, ref_val, qsp_val, rel_diff))

    print("=== Gamma-point Hessian test PASSED ===")


def test_qspace_hessian_symmetry():
    """Verify eigenvalues at symmetry-equivalent q-points match."""
    try:
        import tdscha.QSpaceHessian as QH
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia or QSpaceHessian not available")

    ens = _load_ensemble()

    qh = QH.QSpaceHessian(ens, verbose=True, lo_to_split=None)
    qh.init(use_symmetries=True)

    # Compute full hessian (all q-points)
    hessian = qh.compute_full_hessian(tol=1e-8, max_iters=1000)

    # For each star, check that eigenvalues are equivalent
    print()
    print("=== Symmetry equivalence test ===")
    for iq_irr in qh.irr_qpoints:
        star = qh.q_star_map[iq_irr]
        if len(star) <= 1:
            continue

        ref_H = qh.H_q_dict[iq_irr]
        ref_evals = np.sort(np.real(np.linalg.eigvalsh(ref_H)))

        for iq_rot, R_cart, t_cart in star:
            if iq_rot == iq_irr:
                continue
            rot_H = qh.H_q_dict[iq_rot]
            rot_evals = np.sort(np.real(np.linalg.eigvalsh(rot_H)))

            max_diff = np.max(np.abs(ref_evals - rot_evals))
            print("  q_irr={}, q_rot={}: max eigenvalue diff = {:.2e}".format(
                iq_irr, iq_rot, max_diff))
            assert max_diff < 1e-6, (
                "Eigenvalues at q={} and q={} differ by {:.2e}".format(
                    iq_irr, iq_rot, max_diff))

    print("=== Symmetry equivalence test PASSED ===")


    hessian_other = ens.get_free_energy_hessian(include_v4=True, use_symmetries=True)
    print()
    print("=== Test the complete Hessian matrix ===")
    badvalues = []
    for iq, q in enumerate(hessian.q_tot):
        w_good, _ = hessian_other.DyagDinQ(iq)
        w_test, _ = hessian.DyagDinQ(iq)

        print("  q = {}".format(q))
        for k in range(len(w_good)):
            w_g = w_good[k]
            w_t = w_test[k]
            rel_diff = abs(w_g - w_t) / max(abs(w_g), 1e-15)

            if iq == 0 and abs(w_g) < 1e-9:
                continue

            print("    mode {}: ref={:.8e}, test={:.8e}, rel_diff={:.2e}".format(
                k, w_g, w_t, rel_diff))
            if rel_diff > 0.05:
                badvalues.append((k, q, w_g, w_t, rel_diff))
            
    print("=== Summary of modes with large discrepancies (>5% relative difference) ===")
    for klist in badvalues:
        k, q, w_g, w_t, rel_diff = klist
        assert rel_diff < 0.05, (
                "Mode {} at q={} differs too much: ref={:.6e}, test={:.6e}, "
                "rel_diff={:.4e}".format(k, q, w_g, w_t, rel_diff))

    print("=== Test the complete Hessian matrix  PASSED ===")

if __name__ == "__main__":
    test_qspace_hessian_gamma()
    test_qspace_hessian_symmetry()
