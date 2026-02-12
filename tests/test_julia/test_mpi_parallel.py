"""
Test that MPI parallelization produces identical results to serial execution.

The pytest function runs a serial Lanczos (np=1), then spawns mpirun -np 2
as a subprocess running the helper script _mpi_worker.py, and compares
the Lanczos coefficients and spectral function.
"""

from __future__ import print_function

import numpy as np
import subprocess
import sys
import os
import tempfile
import shutil

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Units

import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL

from tdscha.Parallel import pprint as print


def _run_serial(data_dir, n_steps=5):
    """Run Lanczos in serial and return (a_coeffs, b_coeffs, spectrum)."""
    cwd = os.getcwd()
    os.chdir(data_dir)
    try:
        T = 250
        NQIRR = 3

        dyn = CC.Phonons.Phonons("data/dyn_gen_pop1_", NQIRR)
        ens = sscha.Ensemble.Ensemble(dyn, T)
        ens.load_bin("data", 1)

        nat_uc = dyn.structure.N_atoms
        ec = np.zeros((nat_uc, 3, 3))
        ec[0] = np.eye(3)
        ec[1] = -np.eye(3)

        lanc = DL.Lanczos(ens)
        lanc.gamma_only = True
        lanc.mode = DL.MODE_FAST_JULIA
        lanc.init(use_symmetries=True)
        lanc.prepare_ir(effective_charges=ec, pol_vec=np.array([1., 0., 0.]))
        lanc.run_FT(n_steps)

        w_cm = np.linspace(0, 400, 1000)
        w_ry = w_cm / CC.Units.RY_TO_CM
        smearing_ry = 5.0 / CC.Units.RY_TO_CM
        gf = lanc.get_green_function_continued_fraction(
            w_ry, smearing=smearing_ry, use_terminator=False)
        spectrum = -np.imag(gf)

        return (np.array(lanc.a_coeffs),
                np.array(lanc.b_coeffs),
                spectrum)
    finally:
        os.chdir(cwd)


def test_mpi_parallel():
    """Verify that mpirun -np 2 gives the same results as serial."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(test_dir, "_mpi_worker.py")
    n_steps = 5

    # Check that mpirun is available
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        import pytest
        pytest.skip("mpirun/mpiexec not found")

    # 1. Serial run
    print("Running serial (np=1)...")
    a_serial, b_serial, spec_serial = _run_serial(test_dir, n_steps)

    # 2. MPI run (np=2) via subprocess
    print("Running MPI (np=2)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "mpi_result.npz")
        env = os.environ.copy()
        env["JULIA_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"

        cmd = [mpirun, "-np", "2", sys.executable, worker_script,
               "--n-steps", str(n_steps), "--output", outfile]

        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=300, env=env)
        if result.returncode != 0:
            print("MPI stdout:", result.stdout[-2000:] if result.stdout else "")
            print("MPI stderr:", result.stderr[-2000:] if result.stderr else "")
            raise RuntimeError("mpirun failed with exit code {}".format(
                result.returncode))

        # Load MPI results
        data = np.load(outfile)
        a_mpi = data["a_coeffs"]
        b_mpi = data["b_coeffs"]
        spec_mpi = data["spectrum"]

    # 3. Compare
    print("Comparing results...")
    print("  a_coeffs serial: {}".format(a_serial))
    print("  a_coeffs mpi:    {}".format(a_mpi))

    atol = 1e-10
    rtol = 1e-8

    assert np.allclose(a_serial, a_mpi, atol=atol, rtol=rtol), \
        "a_coeffs mismatch: max_diff={:.2e}".format(
            np.max(np.abs(a_serial - a_mpi)))

    assert np.allclose(b_serial, b_mpi, atol=atol, rtol=rtol), \
        "b_coeffs mismatch: max_diff={:.2e}".format(
            np.max(np.abs(b_serial - b_mpi)))

    assert np.allclose(spec_serial, spec_mpi, atol=atol, rtol=rtol), \
        "spectrum mismatch: max_diff={:.2e}".format(
            np.max(np.abs(spec_serial - spec_mpi)))

    print("PASSED: MPI (np=2) results match serial.")


if __name__ == "__main__":
    test_mpi_parallel()
