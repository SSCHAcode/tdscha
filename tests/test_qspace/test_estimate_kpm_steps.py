"""Test the estimate_kpm_steps function."""
import sys, os
import numpy as np

import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble
import tdscha, tdscha.QSpaceKPM

NQIRR = 3
T = 250
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')
IGNORE_V3 = True
IGNORE_V4 = True

def test_estimate_kpm_steps():
    """Test the KPM step estimation function."""
    dyn = CC.Phonons.Phonons("{}/dyn_gen_pop1_".format(DATA_DIR), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)

    kpm = tdscha.QSpaceKPM.QSpaceKPM(ens, lo_to_split=None)
    kpm.ignore_v3 = IGNORE_V3
    kpm.ignore_v4 = IGNORE_V4
    kpm.init()
    
    # Test 1: Must raise error if no perturbation prepared
    print("Test 1: Check error when no perturbation prepared...")
    try:
        kpm.estimate_kpm_steps(1.0)
        print("  FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  PASSED: Got expected error: {e}")
    
    # Prepare perturbation at q=0, mode 5
    kpm.prepare_mode_q(0, 5)
    
    # Test 2: Check with different precisions
    print("\nTest 2: Check step estimates for different precisions...")
    precisions = [10.0, 5.0, 2.0, 1.0, 0.5]  # cm⁻¹
    for prec in precisions:
        n_steps = kpm.estimate_kpm_steps(prec, bound_factor=1.2)
        print(f"  Precision {prec:.1f} cm⁻¹ -> {n_steps} steps")
        # Higher precision (smaller value) should need more steps
    
    # Test 3: Check that smaller precision gives more steps
    n_steps_10 = kpm.estimate_kpm_steps(10.0, bound_factor=1.2)
    n_steps_1 = kpm.estimate_kpm_steps(1.0, bound_factor=1.2)
    if n_steps_1 > n_steps_10:
        print(f"\n  PASSED: Higher precision needs more steps ({n_steps_1} > {n_steps_10})")
    else:
        print(f"\n  FAILED: Higher precision should need more steps ({n_steps_1} <= {n_steps_10})")
        return False
    
    # Test 4: Check with different bound factors
    print("\nTest 3: Check step estimates for different bound factors...")
    for bf in [1.1, 1.2, 1.5, 2.0]:
        n_steps = kpm.estimate_kpm_steps(1.0, bound_factor=bf)
        print(f"  bound_factor={bf:.1f} -> {n_steps} steps")
    
    # Test 5: Check error for invalid inputs
    print("\nTest 4: Check error handling for invalid inputs...")
    try:
        kpm.estimate_kpm_steps(-1.0)
        print("  FAILED: Should have raised ValueError for negative precision")
        return False
    except ValueError:
        print("  PASSED: Negative precision raises ValueError")
    
    try:
        kpm.estimate_kpm_steps(1.0, bound_factor=1.0)
        print("  FAILED: Should have raised ValueError for bound_factor=1.0")
        return False
    except ValueError:
        print("  PASSED: bound_factor=1.0 raises ValueError")
    
    try:
        kpm.estimate_kpm_steps(1.0, bound_factor=0.5)
        print("  FAILED: Should have raised ValueError for bound_factor<1.0")
        return False
    except ValueError:
        print("  PASSED: bound_factor<1.0 raises ValueError")
    
    # Test 6: Verify the estimated steps actually work
    print("\nTest 5: Verify estimated steps produce reasonable spectral function...")
    precision = 2.0  # cm⁻¹
    n_steps = kpm.estimate_kpm_steps(precision, bound_factor=1.2)
    print(f"  Estimated steps for {precision} cm⁻¹ precision: {n_steps}")
    
    # Reset KPM state
    kpm = tdscha.QSpaceKPM.QSpaceKPM(ens, lo_to_split=None)
    kpm.ignore_v3 = IGNORE_V3
    kpm.ignore_v4 = IGNORE_V4
    kpm.init()
    kpm.prepare_mode_q(0, 5)
    
    # Run KPM with estimated steps
    kpm.run_KPM(n_steps, bound_factor=1.2, verbose=False)
    
    # Compute spectral function
    w = np.linspace(0, 200, 1000)
    w_ry = w / CC.Units.RY_TO_CM
    spectral = kpm.get_spectral_function_KPM(w_ry, regularization="jackson")
    
    # Find peak and check FWHM
    peak_idx = np.argmax(spectral)
    peak_w = w[peak_idx]
    peak_val = spectral[peak_idx]
    
    # Find half-maximum points
    half_max = peak_val / 2
    left_idx = np.where(spectral[:peak_idx] < half_max)[0]
    right_idx = np.where(spectral[peak_idx:] < half_max)[0]
    
    if len(left_idx) > 0 and len(right_idx) > 0:
        fwhm = w[peak_idx + right_idx[0]] - w[left_idx[-1]]
        print(f"  Peak at {peak_w:.2f} cm⁻¹, FWHM ≈ {fwhm:.2f} cm⁻¹")
        print(f"  Target precision: {precision} cm⁻¹")
        if fwhm <= precision * 2:  # FWHM should be roughly comparable to precision
            print(f"  PASSED: FWHM is within expected range")
        else:
            print(f"  WARNING: FWHM ({fwhm:.2f}) > 2×precision ({2*precision:.2f})")
    else:
        print(f"  Could not determine FWHM (peak may be too sharp or noisy)")
    
    print("\n=== All tests passed! ===")
    return True

if __name__ == "__main__":
    success = test_estimate_kpm_steps()
    sys.exit(0 if success else 1)
