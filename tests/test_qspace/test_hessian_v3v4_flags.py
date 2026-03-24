"""
Test Q-Space Hessian v3/v4 exclusion flags.

This test verifies that:
1. Excluding only v4 gives same result as ens.get_free_energy_hessian(include_v4=False)
2. Excluding both v3 and v4 gives the harmonic result
3. Changing flags after initialization properly propagates to QSpaceLanczos

Uses the SnTe 2×2×2 test data from tests/test_julia/data/ (T=250K, NQIRR=3).
"""
from __future__ import print_function

import numpy as np
import os
import sys
import pytest

import cellconstructor as CC
import cellconstructor.Phonons
from cellconstructor.Settings import ParallelPrint as print

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


def test_hessian_exclude_v4_only():
    """Compare Q-space Hessian with ignore_v4=True vs reference with include_v4=False.
    
    This test verifies that setting ignore_v4=True in QSpaceHessian gives the same
    result as ens.get_free_energy_hessian(include_v4=False).
    """
    try:
        import tdscha.QSpaceHessian as QH
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia or QSpaceHessian not available")

    ens = _load_ensemble()

    print()
    print("=" * 60)
    print("TEST: Exclude v4 only (compare with include_v4=False)")
    print("=" * 60)

    # -- Reference: ens.get_free_energy_hessian with include_v4=False --
    print("\n[1] Computing reference Hessian (include_v4=False)...")
    ref_hessian = ens.get_free_energy_hessian(include_v4=False,
                                               use_symmetries=True)

    # -- Q-space Hessian with ignore_v4=True --
    print("\n[2] Computing Q-space Hessian with ignore_v4=True...")
    qh = QH.QSpaceHessian(ens, verbose=True, ignore_v3=False, ignore_v4=True, 
                          lo_to_split=None)
    qh.init(use_symmetries=True)
    qspace_hessian = qh.compute_full_hessian(tol=1e-8, max_iters=1000)

    # -- Compare eigenvalues at all q-points --
    print("\n[3] Comparing eigenvalues at all q-points...")
    nq = len(ref_hessian.q_tot)
    max_rel_diff_all = 0.0
    
    for iq in range(nq):
        w_ref, _ = ref_hessian.DyagDinQ(iq)
        w_test, _ = qspace_hessian.DyagDinQ(iq)
        
        # Filter out acoustic modes at Gamma
        if iq == 0:
            mask = np.abs(w_ref) > 1e-8
            w_ref = w_ref[mask]
            w_test = w_test[mask]
        
        # Compare eigenvalues
        for i in range(len(w_ref)):
            ref_val = w_ref[i]
            test_val = w_test[i]
            rel_diff = abs(ref_val - test_val) / max(abs(ref_val), 1e-15)
            max_rel_diff_all = max(max_rel_diff_all, rel_diff)
            
            if rel_diff > 0.01:  # Print details for significant differences
                print("  q={}, mode {}: ref={:.8e}, test={:.8e}, rel_diff={:.2e}".format(
                    iq, i, ref_val, test_val, rel_diff))
    
    print("\n[4] Results:")
    print("  Maximum relative difference across all modes: {:.2e}".format(max_rel_diff_all))
    
    # Assert that results match within tolerance
    assert max_rel_diff_all < 0.05, (
        "Q-space Hessian with ignore_v4=True differs from reference "
        "(include_v4=False): max_rel_diff={:.4e}".format(max_rel_diff_all))
    
    print("\n" + "=" * 60)
    print("TEST PASSED: Exclude v4 only")
    print("=" * 60)


def test_hessian_exclude_v3_and_v4():
    """Compare Q-space Hessian with ignore_v3=True, ignore_v4=True vs harmonic.
    
    This test verifies that setting both ignore_v3=True and ignore_v4=True gives
    the harmonic (SSCHA) result, matching ens.get_free_energy_hessian with both
    include_v3=False and include_v4=False.
    """
    try:
        import tdscha.QSpaceHessian as QH
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia or QSpaceHessian not available")

    ens = _load_ensemble()

    print()
    print("=" * 60)
    print("TEST: Exclude both v3 and v4 (harmonic result)")
    print("=" * 60)

    # -- Reference: ens.get_free_energy_hessian with both v3 and v4 excluded --
    print("\n[1] Computing reference Hessian (harmonic only)...")
    # Note: get_free_energy_hessian doesn't have explicit include_v3 flag,
    # but we can compare against the SSCHA dynamical matrix (harmonic part)
    ref_dyn = ens.current_dyn.Copy()
    
    # Get harmonic frequencies from SSCHA dyn
    w_ref_list = []
    for iq in range(len(ref_dyn.q_tot)):
        w, _ = ref_dyn.DyagDinQ(iq)
        if iq == 0:
            # Filter acoustic modes
            w = w[np.abs(w) > 1e-8]
        w_ref_list.append(w)

    # -- Q-space Hessian with both ignore_v3=True and ignore_v4=True --
    print("\n[2] Computing Q-space Hessian with ignore_v3=True, ignore_v4=True...")
    qh = QH.QSpaceHessian(ens, verbose=True, ignore_v3=True, ignore_v4=True,
                          lo_to_split=None)
    qh.init(use_symmetries=True)
    qspace_hessian = qh.compute_full_hessian(tol=1e-8, max_iters=1000)

    # -- Compare eigenvalues at all q-points --
    print("\n[3] Comparing eigenvalues at all q-points...")
    max_rel_diff_all = 0.0
    
    for iq in range(len(ref_dyn.q_tot)):
        w_ref = w_ref_list[iq]
        w_test, _ = qspace_hessian.DyagDinQ(iq)
        
        # Filter acoustic modes at Gamma
        if iq == 0:
            mask = np.abs(w_test) > 1e-8
            w_test = w_test[mask]
        
        # Compare eigenvalues
        for i in range(min(len(w_ref), len(w_test))):
            ref_val = w_ref[i]
            test_val = w_test[i]
            rel_diff = abs(ref_val - test_val) / max(abs(ref_val), 1e-15)
            max_rel_diff_all = max(max_rel_diff_all, rel_diff)
            
            if rel_diff > 0.01:  # Print details for significant differences
                print("  q={}, mode {}: ref={:.8e}, test={:.8e}, rel_diff={:.2e}".format(
                    iq, i, ref_val, test_val, rel_diff))
    
    print("\n[4] Results:")
    print("  Maximum relative difference across all modes: {:.2e}".format(max_rel_diff_all))
    
    # Assert that results match within tolerance
    assert max_rel_diff_all < 0.05, (
        "Q-space Hessian with ignore_v3=True, ignore_v4=True differs from "
        "harmonic reference: max_rel_diff={:.4e}".format(max_rel_diff_all))
    
    print("\n" + "=" * 60)
    print("TEST PASSED: Exclude both v3 and v4 (harmonic)")
    print("=" * 60)


def test_hessian_flags_consistency():
    """Test that flags are properly passed to QSpaceLanczos.
    
    Verifies that the ignore_v3 and ignore_v4 flags set on QSpaceHessian
    are properly propagated to the underlying QSpaceLanczos instance.
    """
    try:
        import tdscha.QSpaceHessian as QH
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia or QSpaceHessian not available")

    ens = _load_ensemble()

    print()
    print("=" * 60)
    print("TEST: Flags consistency check")
    print("=" * 60)

    # Test various flag combinations at initialization
    flag_combinations = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]

    for ignore_v3, ignore_v4 in flag_combinations:
        print("\n  Testing initialization with ignore_v3={}, ignore_v4={}...".format(
            ignore_v3, ignore_v4))
        
        qh = QH.QSpaceHessian(ens, verbose=False, 
                              ignore_v3=ignore_v3, ignore_v4=ignore_v4,
                              lo_to_split=None)
        
        # Check that flags are set on QSpaceHessian
        assert qh.ignore_v3 == ignore_v3, (
            "qh.ignore_v3 ({}) != expected ({})".format(qh.ignore_v3, ignore_v3))
        assert qh.ignore_v4 == ignore_v4, (
            "qh.ignore_v4 ({}) != expected ({})".format(qh.ignore_v4, ignore_v4))
        
        # Check that flags are propagated to QSpaceLanczos
        assert qh.qlanc.ignore_v3 == ignore_v3, (
            "qh.qlanc.ignore_v3 ({}) != expected ({})".format(
                qh.qlanc.ignore_v3, ignore_v3))
        assert qh.qlanc.ignore_v4 == ignore_v4, (
            "qh.qlanc.ignore_v4 ({}) != expected ({})".format(
                qh.qlanc.ignore_v4, ignore_v4))
        
        print("    PASSED: Flags correctly set and propagated at init")

    print("\n" + "=" * 60)
    print("TEST PASSED: Flags consistency at initialization")
    print("=" * 60)


def test_hessian_flags_setattr_propagation():
    """Test that changing flags after initialization propagates to QSpaceLanczos.
    
    Verifies that the __setattr__ override correctly propagates flag changes
    to the underlying QSpaceLanczos instance.
    """
    try:
        import tdscha.QSpaceHessian as QH
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia or QSpaceHessian not available")

    ens = _load_ensemble()

    print()
    print("=" * 60)
    print("TEST: Flags __setattr__ propagation")
    print("=" * 60)

    # Initialize with both flags False
    print("\n[1] Initializing with ignore_v3=False, ignore_v4=False...")
    qh = QH.QSpaceHessian(ens, verbose=False, ignore_v3=False, ignore_v4=False,
                          lo_to_split=None)
    
    assert qh.ignore_v3 == False and qh.qlanc.ignore_v3 == False
    assert qh.ignore_v4 == False and qh.qlanc.ignore_v4 == False
    print("    Initial state confirmed")
    
    # Change ignore_v3
    print("\n[2] Changing ignore_v3 to True...")
    qh.ignore_v3 = True
    assert qh.ignore_v3 == True, "qh.ignore_v3 not updated"
    assert qh.qlanc.ignore_v3 == True, "qh.qlanc.ignore_v3 not propagated"
    print("    ignore_v3 change propagated correctly")
    
    # Change ignore_v4
    print("\n[3] Changing ignore_v4 to True...")
    qh.ignore_v4 = True
    assert qh.ignore_v4 == True, "qh.ignore_v4 not updated"
    assert qh.qlanc.ignore_v4 == True, "qh.qlanc.ignore_v4 not propagated"
    print("    ignore_v4 change propagated correctly")
    
    # Change both back to False
    print("\n[4] Changing both flags back to False...")
    qh.ignore_v3 = False
    qh.ignore_v4 = False
    assert qh.ignore_v3 == False and qh.qlanc.ignore_v3 == False
    assert qh.ignore_v4 == False and qh.qlanc.ignore_v4 == False
    print("    Both flags reset correctly")
    
    # Test setting to same value (should not cause issues)
    print("\n[5] Setting flags to same values (no-op)...")
    qh.ignore_v3 = False  # Already False
    qh.ignore_v4 = False  # Already False
    assert qh.ignore_v3 == False and qh.qlanc.ignore_v3 == False
    assert qh.ignore_v4 == False and qh.qlanc.ignore_v4 == False
    print("    No-op assignment handled correctly")

    print("\n" + "=" * 60)
    print("TEST PASSED: Flags __setattr__ propagation")
    print("=" * 60)


if __name__ == "__main__":
    # Run all tests
    test_hessian_flags_consistency()
    test_hessian_flags_setattr_propagation()
    test_hessian_exclude_v4_only()
    test_hessian_exclude_v3_and_v4()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
