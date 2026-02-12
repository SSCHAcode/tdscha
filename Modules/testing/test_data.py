#!/usr/bin/env python3
"""
Test data utilities for TD-SCHA doctests.

This module provides lightweight test data from the test_julia test suite
for use in documentation examples and doctests.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Try to import TD-SCHA dependencies
try:
    import cellconstructor as CC
    import cellconstructor.Phonons
    import sscha.Ensemble
    import tdscha.DynamicalLanczos as DL
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    CC = None
    DL = None
    sscha = None

# Define mode constants
if HAS_DEPENDENCIES:
    MODE_FAST_SERIAL = DL.MODE_FAST_SERIAL
    MODE_FAST_MPI = DL.MODE_FAST_MPI
    MODE_FAST_JULIA = DL.MODE_FAST_JULIA
else:
    MODE_FAST_SERIAL = 1
    MODE_FAST_MPI = 2
    MODE_FAST_JULIA = 3


def get_test_data_path():
    """
    Get the path to the test_julia test data.
    
    Returns
    -------
    Path
        Path object pointing to tests/test_julia/data directory
    
    Raises
    ------
    FileNotFoundError
        If data directory cannot be found
    """
    # Try multiple possible locations
    possible_paths = [
        # Development environment (from repo root, relative to this file)
        Path(__file__).parent.parent.parent / "tests" / "test_julia" / "data",
        # Installed package (unlikely to have test data)
        Path(__file__).parent.parent / "tests" / "test_julia" / "data",
        # Relative to current working directory (CI runs from repo root)
        Path.cwd() / "tests" / "test_julia" / "data",
        # Relative to environment variable TD_SCHA_SOURCE_DIR
        Path(os.environ.get("TD_SCHA_SOURCE_DIR", "")) / "tests" / "test_julia" / "data",
        # Look for the data in the source directory (if we can find it)
        Path(sys.prefix) / ".." / "tests" / "test_julia" / "data",  # virtualenv
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Last resort: check if we're in a test environment and data might be elsewhere
    # Look for any parent directory containing 'tests/test_julia/data'
    current = Path(__file__).parent
    for _ in range(5):  # Go up at most 5 levels
        test_path = current / "tests" / "test_julia" / "data"
        if test_path.exists():
            return test_path
        current = current.parent
    
    raise FileNotFoundError(
        "Could not find test_julia data directory. "
        "Tried: " + ", ".join(str(p) for p in possible_paths)
    )


def load_test_ensemble(temperature=250, n_configs=None):
    """
    Load the test ensemble from test_julia data.
    
    Parameters
    ----------
    temperature : float
        Temperature in Kelvin (default: 250)
    n_configs : int or None
        Number of configurations to load (default: all)
    
    Returns
    -------
    sscha.Ensemble.Ensemble
        Loaded ensemble object
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Required dependencies not available: cellconstructor, sscha")
    
    data_path = get_test_data_path()
    
    # Load dynamical matrix (3 irreducible q-points as in test_julia)
    dyn = CC.Phonons.Phonons(str(data_path / "dyn_gen_pop1_"), 3)
    
    # Create ensemble
    ens = sscha.Ensemble.Ensemble(dyn, temperature)
    
    # Load binary ensemble data
    ens.load_bin(str(data_path), 1)
    
    # Limit configurations if requested
    if n_configs is not None and n_configs < ens.N:
        # Select first n_configs
        mask = np.zeros(ens.N, dtype=bool)
        mask[:n_configs] = True
        ens = ens.split(mask)
    
    return ens


def create_test_lanczos(ensemble, mode=MODE_FAST_SERIAL, verbose=False, **kwargs):
    """
    Create a Lanczos object for the given test ensemble.
    
    Note: This function does NOT initialize the Lanczos object.
    The user must call lanczos.init() explicitly in documentation examples.
    
    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble
        Ensemble to use (must be provided)
    mode : int
        Lanczos computation mode (default: MODE_FAST_SERIAL)
    verbose : bool
        If False, suppress verbose output (default: False)
    **kwargs : dict
        Additional arguments passed to Lanczos constructor
    
    Returns
    -------
    DL.Lanczos
        Uninitialized Lanczos object (call .init() to initialize)
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Required dependencies not available: cellconstructor, sscha, tdscha")
    
    if ensemble is None:
        raise ValueError("ensemble must be provided")
    
    # Create Lanczos object (not initialized)
    lanczos = DL.Lanczos(ensemble, **kwargs)
    
    # Set computation mode (doesn't require initialization)
    lanczos.mode = mode
    lanczos.verbose = verbose
    
    return lanczos


def get_test_mode_frequencies(ensemble=None):
    """
    Get mode frequencies from test ensemble.
    
    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble or None
        Ensemble to use (default: load test ensemble)
    
    Returns
    -------
    tuple
        (frequencies_cm, mode_indices) where frequencies_cm is array
        of frequencies in cm⁻¹ and mode_indices is array of mode indices
        Note: Returns (None, None) if dependencies not available
    """
    if not HAS_DEPENDENCIES:
        return None, None
    
    if ensemble is None:
        ensemble = load_test_ensemble()
    
    # Create and initialize Lanczos to get frequencies
    lanczos = DL.Lanczos(ensemble)
    lanczos.init(use_symmetries=True)
    
    # Convert from Ry to cm⁻¹ (assuming CC.Units is available)
    try:
        from cellconstructor import Units
        ry_to_cm = Units.RY_TO_CM
    except ImportError:
        # Fallback conversion factor
        ry_to_cm = 109737.315685
    
    frequencies_cm = lanczos.w * ry_to_cm
    
    # Sort by frequency
    sorted_indices = np.argsort(frequencies_cm)
    
    return frequencies_cm, sorted_indices


# Example usage and test
if __name__ == "__main__":
    print("Testing test_data module...")
    
    try:
        # Test 1: Load ensemble
        print("1. Loading test ensemble...")
        ens = load_test_ensemble(n_configs=10)
        print(f"   Success: {ens.N} configurations loaded")
        
        # Test 2: Create Lanczos (not initialized)
        print("2. Creating Lanczos object (not initialized)...")
        lanczos = create_test_lanczos(ens)
        print(f"   Success: Lanczos object created (not initialized)")
        print(f"   Note: User must call lanczos.init() to initialize")
        
        # Test 3: Get frequencies (requires initialization)
        print("3. Getting mode frequencies...")
        freqs, indices = get_test_mode_frequencies(ens)
        if freqs is not None:
            print(f"   Lowest frequency: {freqs[indices[0]]:.1f} cm⁻¹")
            print(f"   Highest frequency: {freqs[indices[-1]]:.1f} cm⁻¹")
        else:
            print("   Dependencies not available, skipping frequency test")
        
        # Test 4: Demonstrate workflow
        print("4. Demonstrating complete workflow...")
        if HAS_DEPENDENCIES:
            # Create and initialize Lanczos
            lanczos = DL.Lanczos(ens)
            lanczos.init(use_symmetries=True)
            print(f"   Initialized Lanczos with {len(lanczos.w)} modes")
            
            # Prepare perturbation
            lanczos.prepare_mode(10)
            print(f"   Prepared perturbation for mode 10")
            
            # Run a few steps
            lanczos.run_FT(2, debug=False)
            print(f"   Ran 2 Lanczos steps")
            print(f"   Coefficients: a={lanczos.a_coeffs[:2]}, b={lanczos.b_coeffs[:2]}")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)