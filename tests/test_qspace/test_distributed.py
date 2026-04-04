"""
Tests for MPI-distributed configuration loading in QSpaceLanczos.

Run with MPI (compares GoParallel vs distributed approach):
    micromamba run -n sscha mpirun -np 2 python tests/test_qspace/test_distributed.py

The tests compare results from:
1. Regular QSpaceLanczos (uses GoParallel internally)
2. load_distributed_tdscha (new distributed approach)

Note: These tests require at least 2 MPI processes and will be skipped if run
with only 1 process (e.g., via pytest without mpirun).
"""
from __future__ import print_function

import numpy as np
import os
import sys

import pytest

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Settings

import sscha, sscha.Ensemble

import tdscha.QSpaceLanczos as QL
import tdscha.QSpaceKPM as QK
import tdscha.QSpaceHessian as QH

from tdscha.QSpaceLanczos import load_distributed_tdscha


# Test parameters
N_STEPS = 5
T = 250
NQIRR = 3

# Data directory (reuse test_julia data)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')


def pprint(*args):
    """Print only on rank 0."""
    if cellconstructor.Settings.am_i_the_master():
        print(*args)


def _find_gamma_mode(dyn):
    """Find a non-acoustic Gamma mode for testing."""
    ws_sc, pols_sc, w_q, pols_q = dyn.DiagonalizeSupercell(return_qmodes=True)

    super_structure = dyn.structure.generate_supercell(dyn.GetSupercell())
    m = super_structure.get_masses_array()
    trans_mask = CC.Methods.get_translations(pols_sc, m)
    good_ws = ws_sc[~trans_mask]
    orig_indices = np.where(~trans_mask)[0]

    n_bands = 3 * dyn.structure.N_atoms

    # Find first non-acoustic Gamma band
    for band in range(n_bands):
        if w_q[band, 0] > 1e-6:
            target_freq = w_q[band, 0]
            break
    else:
        raise ValueError("No non-acoustic Gamma mode found")

    # Find the supercell mode matching this frequency
    mode_index = np.argmin(np.abs(good_ws - target_freq))
    orig_mode = orig_indices[mode_index]

    # Project onto q-space bands
    nat_uc = dyn.structure.N_atoms
    nat_sc = super_structure.N_atoms
    n_cell = np.prod(dyn.GetSupercell())
    itau = super_structure.get_itau(dyn.structure) - 1

    pol_sc_mode = pols_sc[:, orig_mode]
    pol_gamma = np.zeros(3 * nat_uc)
    for i_sc in range(nat_sc):
        i_uc = itau[i_sc]
        pol_gamma[3*i_uc:3*i_uc+3] += pol_sc_mode[3*i_sc:3*i_sc+3]
    pol_gamma /= np.sqrt(n_cell)

    R1 = np.conj(pols_q[:, :, 0]).T @ pol_gamma
    band_index = np.argmax(np.abs(R1))

    return band_index


def _get_n_procs():
    """Get number of MPI processes."""
    return cellconstructor.Settings.GetNProc()


def _create_dyn():
    """Create a test dynamical matrix."""
    return CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)


def test_distributed_attributes_set():
    """Test that distributed attributes are correctly set after load_distributed_tdscha."""
    n_procs = _get_n_procs()
    if n_procs < 2:
        pytest.skip("This test requires mpirun -np 2")

    pprint("=" * 60)
    pprint("TEST: Distributed attributes are correctly set")
    pprint("=" * 60)

    # Create dynamical matrix
    dyn = _create_dyn()

    # Use load_distributed_tdscha with data_dir and dyn
    qlanc = load_distributed_tdscha(DATA_DIR, 1, dyn, T, lo_to_split=None, use_symmetries=True)
    
    pprint(f"Rank {n_procs - 1} has N_local = {qlanc._N_local}")
    pprint(f"Global N_global = {qlanc._N_global}")
    pprint(f"_distributed = {qlanc._distributed}")
    
    # Check distributed attributes
    assert hasattr(qlanc, '_distributed'), "Missing _distributed attribute"
    assert qlanc._distributed == True, f"_distributed should be True, got {qlanc._distributed}"
    assert hasattr(qlanc, '_N_global'), "Missing _N_global attribute"
    assert hasattr(qlanc, '_N_local'), "Missing _N_local attribute"
    
    # N_local should be less than or equal to ceil(N_global / n_procs)
    max_expected = int(np.ceil(qlanc._N_global / n_procs))
    assert qlanc._N_local <= max_expected, \
        f"_N_local ({qlanc._N_local}) > ceil(_N_global/n_procs) ({max_expected})"


def test_distributed_memory_footprint():
    """Test that distributed mode uses less memory by storing only N_local configs."""
    n_procs = _get_n_procs()
    if n_procs < 2:
        pytest.skip("This test requires mpirun -np 2")

    pprint("=" * 60)
    pprint("TEST: Distributed memory footprint")
    pprint("=" * 60)

    # Create dynamical matrix
    dyn = _create_dyn()

    # Use load_distributed_tdscha (distributes the 1 config)
    qlanc = load_distributed_tdscha(DATA_DIR, 1, dyn, T, lo_to_split=None, use_symmetries=True)
    
    pprint(f"X_q shape: {qlanc.X_q.shape}")
    pprint(f"N_local: {qlanc._N_local}")
    pprint(f"N_global: {qlanc._N_global}")
    
    # After distribution, X_q should have only N_local configs
    # (not N_global, because configs are distributed)
    max_expected = int(np.ceil(qlanc._N_global / n_procs))
    
    assert qlanc.X_q.shape[1] <= max_expected, \
        f"X_q has {qlanc.X_q.shape[1]} configs (expected <= {max_expected})"


def test_distributed_lanczos_run():
    """Test that Lanczos runs correctly with distributed configurations."""
    n_procs = _get_n_procs()
    if n_procs < 2:
        pytest.skip("This test requires mpirun -np 2")

    pprint("=" * 60)
    pprint("TEST: Distributed Lanczos run")
    pprint("=" * 60)

    # Create dynamical matrix
    dyn = _create_dyn()

    # Use load_distributed_tdscha
    qlanc = load_distributed_tdscha(DATA_DIR, 1, dyn, T, lo_to_split=None, use_symmetries=True)
    qlanc.ignore_harmonic = False
    qlanc.ignore_v3 = False
    qlanc.ignore_v4 = False
    
    # Prepare perturbation
    iq = 0
    band = _find_gamma_mode(dyn)
    qlanc.prepare_mode_q(iq, band)
    
    pprint("Running Lanczos FT...")
    qlanc.run_FT(N_STEPS, verbose=False)
    
    # Check that coefficients are finite
    assert all(np.isfinite(qlanc.a_coeffs)), "Lanczos a_coeffs contain NaN/Inf"
    assert all(np.isfinite(qlanc.b_coeffs)), "Lanczos b_coeffs contain NaN/Inf"


def test_distributed_hessian():
    """Test that Hessian computation works with distributed configurations."""
    n_procs = _get_n_procs()
    if n_procs < 2:
        pytest.skip("This test requires mpirun -np 2")

    pprint("=" * 60)
    pprint("TEST: Distributed Hessian computation")
    pprint("=" * 60)

    # Create dynamical matrix
    dyn = _create_dyn()

    # Use load_distributed_tdscha
    qlanc = load_distributed_tdscha(DATA_DIR, 1, dyn, T, lo_to_split=None, use_symmetries=False)
    qlanc.ignore_harmonic = False
    qlanc.ignore_v3 = False
    qlanc.ignore_v4 = False
    
    # Create Hessian from distributed Lanczos
    hess = QH.QSpaceHessian.from_qspace_lanczos(qlanc, verbose=False, use_symmetries=False)
    
    pprint("Computing Hessian...")
    hess.compute_full_hessian()
    
    # Check that we have results for Gamma
    assert 0 in hess.H_q_dict, "Missing Gamma point in Hessian results"
    
    # Check that eigenvalues are reasonable
    H_gamma = hess.H_q_dict[0]
    evals = np.linalg.eigvalsh(H_gamma)
    
    pprint(f"Hessian eigenvalues at Gamma: {evals}")
    
    # All eigenvalues should be non-negative (for stable system)
    assert np.all(evals >= -1e-10), "Negative eigenvalues in Hessian"


def test_distributed_kpm():
    """Test that KPM works with distributed configurations."""
    n_procs = _get_n_procs()
    if n_procs < 2:
        pytest.skip("This test requires mpirun -np 2")

    pprint("=" * 60)
    pprint("TEST: Distributed KPM")
    pprint("=" * 60)

    # Create dynamical matrix
    dyn = _create_dyn()

    # Use load_distributed_tdscha
    qlanc = load_distributed_tdscha(DATA_DIR, 1, dyn, T, lo_to_split=None, use_symmetries=True)
    
    # Prepare perturbation
    iq = 0
    band = _find_gamma_mode(dyn)
    qlanc.prepare_mode_q(iq, band)
    
    # Create KPM from distributed Lanczos
    pprint("Creating KPM from distributed Lanczos...")
    kpm = QK.QSpaceKPM.from_qspace_lanczos(qlanc)
    kpm.prepare_mode_q(iq, band)
    
    # Estimate and run KPM
    pprint("Running KPM...")
    n_moments = kpm.estimate_kpm_steps(precision_cm=50)
    n_moments = min(n_moments, 16)  # Cap for test speed
    kpm.run_KPM(n_moments, verbose=False)
    
    # Check that moments are finite
    assert all(np.isfinite(kpm.kpm_moments)), "KPM moments contain NaN/Inf"


def test_goparallel_vs_distributed():
    """Compare results from GoParallel (regular) vs load_distributed_tdscha.
    
    This test must be run with mpirun -np 2 to test the distributed approach.
    Both approaches should give identical results.
    """
    n_procs = _get_n_procs()
    if n_procs < 2:
        pytest.skip("This test requires mpirun -np 2")

    if not QL.__JULIA_EXT__:
        pytest.skip("Julia extension not loaded")

    pprint("=" * 60)
    pprint("TEST: GoParallel vs Distributed approach")
    pprint("=" * 60)
    
    # Create dynamical matrix
    dyn = _create_dyn()

    # Test 1: Regular QSpaceLanczos with GoParallel
    # Each rank has full data but Julia work is distributed via GoParallel
    pprint()
    pprint("--- Test 1: GoParallel (regular) ---")
    
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)
    
    qlanc_goparallel = QL.QSpaceLanczos(ens, lo_to_split=None)
    qlanc_goparallel.ignore_harmonic = False
    qlanc_goparallel.ignore_v3 = False
    qlanc_goparallel.ignore_v4 = False
    qlanc_goparallel.init(use_symmetries=True)
    
    # Prepare perturbation
    iq = 0
    band = _find_gamma_mode(dyn)
    qlanc_goparallel.prepare_mode_q(iq, band)
    
    # Note: _distributed is False by default
    qlanc_goparallel._distributed = False
    
    pprint(f"_distributed = {qlanc_goparallel._distributed}")
    pprint(f"N = {qlanc_goparallel.N}")
    
    pprint("Running FT with GoParallel...")
    qlanc_goparallel.run_FT(N_STEPS, verbose=False)
    
    a_coeffs_goparallel = qlanc_goparallel.a_coeffs.copy()
    b_coeffs_goparallel = qlanc_goparallel.b_coeffs.copy()
    
    # Test 2: load_distributed_tdscha
    pprint()
    pprint("--- Test 2: Distributed approach ---")
    
    # Use the new distributed function with data_dir
    qlanc_dist = load_distributed_tdscha(DATA_DIR, 1, dyn, T, lo_to_split=None, use_symmetries=True)
    qlanc_dist.ignore_harmonic = False
    qlanc_dist.ignore_v3 = False
    qlanc_dist.ignore_v4 = False
    
    qlanc_dist.prepare_mode_q(iq, band)
    
    pprint(f"_distributed = {qlanc_dist._distributed}")
    pprint(f"N = {qlanc_dist.N}")
    pprint(f"N_local = {qlanc_dist._N_local}")
    
    pprint("Running FT with distributed approach...")
    qlanc_dist.run_FT(N_STEPS, verbose=False)
    
    a_coeffs_dist = qlanc_dist.a_coeffs.copy()
    b_coeffs_dist = qlanc_dist.b_coeffs.copy()
    
    # Print comparison
    pprint()
    pprint("RESULTS COMPARISON:")
    pprint()
    pprint(f"GoParallel a_coeffs: {a_coeffs_goparallel}")
    pprint(f"Distributed a_coeffs: {a_coeffs_dist}")
    pprint()
    pprint(f"GoParallel b_coeffs: {b_coeffs_goparallel}")
    pprint(f"Distributed b_coeffs: {b_coeffs_dist}")
    
    # Compare results
    np.testing.assert_allclose(a_coeffs_goparallel, a_coeffs_dist, rtol=1e-8,
                               err_msg="a_coeffs mismatch between GoParallel and distributed")
    np.testing.assert_allclose(b_coeffs_goparallel, b_coeffs_dist, rtol=1e-8,
                               err_msg="b_coeffs mismatch between GoParallel and distributed")
