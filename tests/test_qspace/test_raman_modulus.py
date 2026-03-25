"""
Test that compares the perturbation modulus for Raman between Q-space Lanczos
and real-space Lanczos (both using Wigner mode) on the SnTe-like system.

The perturbation modulus should match exactly (within numerical tolerance).
"""
from __future__ import print_function

import numpy as np
import os
import sys
import pytest

import cellconstructor.Phonons as CC
import cellconstructor.Methods
import cellconstructor.symmetries

import sscha.Ensemble
import tdscha.DynamicalLanczos as DL

from tdscha.Parallel import pprint as print

# Data directory (reuse test_julia data)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')

# Temperature
T = 250
NQIRR = 3


def _create_dummy_raman_tensor(dyn):
    """Create a dummy Raman tensor for testing."""
    nat = dyn.structure.N_atoms
    # Create a simple Raman tensor: R_{αβ,i} = δ_{αβ} * sign(i) where i is atom index
    # For 2 atoms: atom 0 gets +I, atom 1 gets -I (satisfies translation invariance)
    raman_tensor = np.zeros((3, 3, 3 * nat))
    for i in range(nat):
        sign = 1.0 if i % 2 == 0 else -1.0
        for alpha in range(3):
            raman_tensor[alpha, alpha, 3*i + alpha] = sign
    return raman_tensor


def _setup_realspace_lanczos_raman(pol_vec_in, pol_vec_out, unpolarized=None):
    """Set up a real-space Lanczos calculation with Raman perturbation."""
    dyn = CC.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)
    
    # Add dummy Raman tensor if missing (AFTER loading binary data)
    # Lanczos uses ens.current_dyn, not ens.dyn_0
    if ens.current_dyn.raman_tensor is None:
        ens.current_dyn.raman_tensor = _create_dummy_raman_tensor(ens.current_dyn)

    lanc = DL.Lanczos(ens, lo_to_split=None)
    lanc.ignore_harmonic = False
    lanc.ignore_v3 = False
    lanc.ignore_v4 = False
    lanc.use_wigner = True
    lanc.mode = DL.MODE_FAST_SERIAL
    lanc.init(use_symmetries=True)
    
    if unpolarized is None:
        lanc.prepare_raman(pol_vec_in=pol_vec_in, pol_vec_out=pol_vec_out)
    else:
        lanc.prepare_raman(unpolarized=unpolarized)
    
    return lanc


def _setup_qspace_lanczos_raman(pol_vec_in, pol_vec_out, unpolarized=None):
    """Set up a Q-space Lanczos calculation with Raman perturbation."""
    try:
        import tdscha.QSpaceLanczos as QL
    except ImportError:
        pytest.skip("Julia not available for QSpaceLanczos")

    if not QL.__JULIA_EXT__:
        pytest.skip("Julia extension not loaded")

    dyn = CC.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)
    
    # Add dummy Raman tensor if missing (AFTER loading binary data)
    # Lanczos uses ens.current_dyn, not ens.dyn_0
    if ens.current_dyn.raman_tensor is None:
        ens.current_dyn.raman_tensor = _create_dummy_raman_tensor(ens.current_dyn)

    qlanc = QL.QSpaceLanczos(ens, lo_to_split=None)
    qlanc.ignore_harmonic = False
    qlanc.ignore_v3 = False
    qlanc.ignore_v4 = False
    qlanc.init(use_symmetries=True)
    
    if unpolarized is None:
        qlanc.prepare_raman(pol_vec_in=pol_vec_in, pol_vec_out=pol_vec_out)
    else:
        qlanc.prepare_raman(unpolarized=unpolarized)
    
    return qlanc


def test_raman_perturbation_modulus(verbose=True):
    """
    Compare perturbation_modulus after prepare_raman between real-space and q-space.
    """
    try:
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia not available for QSpaceLanczos")

    # Test simple polarized Raman
    test_cases = [
        # (pol_in, pol_out, description)
        (np.array([1., 0., 0.]), np.array([1., 0., 0.]), "xx"),
        (np.array([1., 0., 0.]), np.array([0., 1., 0.]), "xy"),
        (np.array([0., 1., 0.]), np.array([0., 1., 0.]), "yy"),
        (np.array([0., 0., 1.]), np.array([0., 0., 1.]), "zz"),
    ]
    
    for pol_in, pol_out, desc in test_cases:
        if verbose:
            print(f"\n=== Testing Raman perturbation modulus {desc} ===")
        
        lanc_real = _setup_realspace_lanczos_raman(pol_in, pol_out)
        lanc_qspace = _setup_qspace_lanczos_raman(pol_in, pol_out)
        
        mod_real = lanc_real.perturbation_modulus
        mod_qspace = lanc_qspace.perturbation_modulus
        
        if verbose:
            print(f"Real-space perturbation_modulus = {mod_real:.12e}")
            print(f"Q-space perturbation_modulus    = {mod_qspace:.12e}")
            print(f"Relative difference = {abs(mod_real - mod_qspace) / max(mod_real, mod_qspace):.2e}")
        
        # The modulus should agree within 1e-10 relative tolerance
        rtol = 1e-10
        atol = 1e-14
        np.testing.assert_allclose(mod_qspace, mod_real, rtol=rtol, atol=atol,
                                   err_msg=f"Raman perturbation modulus mismatch for {desc}")
    
    # Test unpolarized components (with prefactors)
    for unpolarized in range(7):
        if verbose:
            print(f"\n=== Testing unpolarized Raman component {unpolarized} ===")
        
        lanc_real = _setup_realspace_lanczos_raman(None, None, unpolarized=unpolarized)
        lanc_qspace = _setup_qspace_lanczos_raman(None, None, unpolarized=unpolarized)
        
        mod_real = lanc_real.perturbation_modulus
        mod_qspace = lanc_qspace.perturbation_modulus
        
        if verbose:
            print(f"Real-space perturbation_modulus = {mod_real:.12e}")
            print(f"Q-space perturbation_modulus    = {mod_qspace:.12e}")
            print(f"Relative difference = {abs(mod_real - mod_qspace) / max(mod_real, mod_qspace):.2e}")
        
        # The modulus should agree within 1e-10 relative tolerance
        rtol = 1e-10
        atol = 1e-14
        np.testing.assert_allclose(mod_qspace, mod_real, rtol=rtol, atol=atol,
                                   err_msg=f"Unpolarized Raman component {unpolarized} modulus mismatch")
    
    if verbose:
        print("\n=== All Raman perturbation modulus tests passed ===")


def test_unpolarized_raman_modulus(verbose=True):
    """
    Compare perturbation_modulus after prepare_unpolarized_raman between real-space and q-space.
    """
    try:
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia not available for QSpaceLanczos")

    for index in range(7):
        if verbose:
            print(f"\n=== Testing prepare_unpolarized_raman index {index} ===")
        
        dyn = CC.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
        
        ens = sscha.Ensemble.Ensemble(dyn, T)
        ens.load_bin(DATA_DIR, 1)
        
        # Add dummy Raman tensor if missing (AFTER loading binary data)
        # Lanczos uses ens.current_dyn, not ens.dyn_0
        if ens.current_dyn.raman_tensor is None:
            ens.current_dyn.raman_tensor = _create_dummy_raman_tensor(ens.current_dyn)

        # Real-space
        lanc_real = DL.Lanczos(ens, lo_to_split=None)
        lanc_real.ignore_harmonic = False
        lanc_real.ignore_v3 = False
        lanc_real.ignore_v4 = False
        lanc_real.use_wigner = True
        lanc_real.mode = DL.MODE_FAST_SERIAL
        lanc_real.init(use_symmetries=True)
        lanc_real.prepare_unpolarized_raman(index=index)
        
        # Q-space
        qlanc = QL.QSpaceLanczos(ens, lo_to_split=None)
        qlanc.ignore_harmonic = False
        qlanc.ignore_v3 = False
        qlanc.ignore_v4 = False
        qlanc.init(use_symmetries=True)
        qlanc.prepare_unpolarized_raman(index=index)
        
        mod_real = lanc_real.perturbation_modulus
        mod_qspace = qlanc.perturbation_modulus
        
        if verbose:
            print(f"Real-space perturbation_modulus = {mod_real:.12e}")
            print(f"Q-space perturbation_modulus    = {mod_qspace:.12e}")
            print(f"Relative difference = {abs(mod_real - mod_qspace) / max(mod_real, mod_qspace):.2e}")
        
        # The modulus should agree within 1e-10 relative tolerance
        rtol = 1e-10
        atol = 1e-14
        np.testing.assert_allclose(mod_qspace, mod_real, rtol=rtol, atol=atol,
                                   err_msg=f"prepare_unpolarized_raman index {index} modulus mismatch")
    
    if verbose:
        print("\n=== All prepare_unpolarized_raman tests passed ===")


if __name__ == "__main__":
    # Run the tests with verbose output
    test_raman_perturbation_modulus(verbose=True)
    test_unpolarized_raman_modulus(verbose=True)
    print("\nAll tests passed.")