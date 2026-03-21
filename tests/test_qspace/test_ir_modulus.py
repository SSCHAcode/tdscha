"""
Test that compares the perturbation modulus for IR between Q-space Lanczos
and real-space Lanczos (both using Wigner mode) on the SnTe-like system.

The perturbation modulus should match exactly (within numerical tolerance).
"""
from __future__ import print_function

import numpy as np
import os
import sys
import pytest

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Methods
import cellconstructor.symmetries

import sscha, sscha.Ensemble
import tdscha.DynamicalLanczos as DL

from tdscha.Parallel import pprint as print

# Data directory (reuse test_julia data)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')

# Temperature
T = 250
NQIRR = 3


def _setup_realspace_lanczos_ir(pol_vec):
    """Set up a real-space Lanczos calculation with IR perturbation."""
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    # Create ASR-satisfying effective charges: +I for first atom, -I for second, etc.
    nat = dyn.structure.N_atoms
    effective_charges = np.zeros((nat, 3, 3))
    for i in range(nat):
        # Alternate signs to satisfy ASR
        sign = 1.0 if i % 2 == 0 else -1.0
        effective_charges[i] = sign * np.eye(3)
    # Ensure ASR is exactly satisfied
    total_charge = np.sum(effective_charges, axis=0)
    if np.max(np.abs(total_charge)) > 1e-12:
        # Adjust to exactly satisfy ASR
        effective_charges -= total_charge / nat
    
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)

    lanc = DL.Lanczos(ens, lo_to_split=None)
    lanc.ignore_harmonic = False
    lanc.ignore_v3 = False
    lanc.ignore_v4 = False
    lanc.use_wigner = True
    lanc.mode = DL.MODE_FAST_SERIAL
    lanc.init(use_symmetries=True)
    lanc.prepare_ir(effective_charges=effective_charges, pol_vec=pol_vec)
    return lanc


def _setup_qspace_lanczos_ir(pol_vec):
    """Set up a Q-space Lanczos calculation with IR perturbation."""
    try:
        import tdscha.QSpaceLanczos as QL
    except ImportError:
        pytest.skip("Julia not available for QSpaceLanczos")

    if not QL.__JULIA_EXT__:
        pytest.skip("Julia extension not loaded")

    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    nat = dyn.structure.N_atoms
    # Create ASR-satisfying effective charges: +I for first atom, -I for second, etc.
    effective_charges = np.zeros((nat, 3, 3))
    for i in range(nat):
        # Alternate signs to satisfy ASR
        sign = 1.0 if i % 2 == 0 else -1.0
        effective_charges[i] = sign * np.eye(3)
    # Ensure ASR is exactly satisfied
    total_charge = np.sum(effective_charges, axis=0)
    if np.max(np.abs(total_charge)) > 1e-12:
        # Adjust to exactly satisfy ASR
        effective_charges -= total_charge / nat
    
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)

    qlanc = QL.QSpaceLanczos(ens, lo_to_split=None)
    qlanc.ignore_harmonic = False
    qlanc.ignore_v3 = False
    qlanc.ignore_v4 = False
    qlanc.init(use_symmetries=True)
    qlanc.prepare_ir(effective_charges=effective_charges, pol_vec=pol_vec)
    return qlanc


def test_ir_perturbation_modulus(verbose=True):
    """
    Compare perturbation_modulus after prepare_ir between real-space and q-space.
    """
    try:
        import tdscha.QSpaceLanczos as QL
        if not QL.__JULIA_EXT__:
            pytest.skip("Julia extension not loaded")
    except ImportError:
        pytest.skip("Julia not available for QSpaceLanczos")

    # Test two polarization vectors
    for pol_vec in (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])):
        if verbose:
            print(f"\n=== Testing IR perturbation modulus with pol_vec = {pol_vec} ===")
        
        lanc_real = _setup_realspace_lanczos_ir(pol_vec)
        lanc_qspace = _setup_qspace_lanczos_ir(pol_vec)
        
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
                                   err_msg=f"IR perturbation modulus mismatch for pol_vec {pol_vec}")
        
        # Also compare the R sector of psi (first n_modes/n_bands entries)
        # Real-space psi is real, q-space psi is complex (but should be real at Gamma)
        # Note: shapes are different - q-space has n_bands (unit cell modes),
        # real-space has n_modes (supercell modes minus translations)
        # We can't compare directly, but we can check that the modulus matches
        if verbose:
            n_modes_real = lanc_real.n_modes
            n_bands_qspace = lanc_qspace.n_bands
            print(f"Real-space n_modes = {n_modes_real}, Q-space n_bands = {n_bands_qspace}")
            
            # Extract R sector
            psi_r_real = lanc_real.psi[:n_modes_real]
            psi_r_qspace = lanc_qspace.psi[:n_bands_qspace]
            print(f"Real-space R sector norm = {np.linalg.norm(psi_r_real):.12e}")
            print(f"Q-space R sector norm = {np.linalg.norm(psi_r_qspace):.12e}")
    
    if verbose:
        print("\n=== All IR perturbation modulus tests passed ===")


if __name__ == "__main__":
    # Run the test with verbose output
    test_ir_perturbation_modulus(verbose=True)
    print("\nAll tests passed.")