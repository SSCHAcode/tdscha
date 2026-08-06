"""Shared fixtures for the tdscha test suite."""
from __future__ import print_function

import shutil
import subprocess
import sys

import pytest


def _probe_mpirun():
    """Return (launcher, n_ranks_seen) for a two-rank ``mpirun`` job.

    ``n_ranks_seen`` is what the *child* processes report as the size of
    ``MPI.COMM_WORLD``.  It is 2 on a working installation.  It is 1 when the
    mpi4py in use was built against a different MPI than the launcher on
    PATH -- typically a pip wheel bundling its own runtime next to a
    system-packaged mpirun.  Each process then initialises its own singleton
    communicator and ``mpirun -np 2`` silently becomes two unrelated
    serial jobs.
    """
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None:
        return None, 0
    probe = ("from mpi4py import MPI; "
             "print('COMM_WORLD_SIZE', MPI.COMM_WORLD.Get_size())")
    try:
        proc = subprocess.run(
            [launcher, "-np", "2", sys.executable, "-c", probe],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    except (subprocess.TimeoutExpired, OSError):
        return launcher, 0
    sizes = [int(line.split()[1])
             for line in proc.stdout.decode().splitlines()
             if line.startswith("COMM_WORLD_SIZE")]
    if proc.returncode != 0 or not sizes:
        return launcher, 0
    return launcher, min(sizes)


@pytest.fixture(scope="session")
def multi_rank_mpirun():
    """An ``mpirun`` that really runs several ranks in one MPI job.

    Skips when there is no MPI at all, and fails -- loudly, with the
    diagnosis -- when there is an mpirun that does not actually produce a
    multi-rank job.  That second case is the dangerous one: the distributed
    tests would still run, every rank would believe it owned the whole
    ensemble, and a test suite that never exercised MPI would report
    success.
    """
    pytest.importorskip("mpi4py")
    launcher, size = _probe_mpirun()
    if launcher is None:
        pytest.skip("no mpirun/mpiexec found")
    if size == 2:
        return launcher
    if size == 1:
        pytest.fail(
            "'{} -np 2' starts two INDEPENDENT single-rank jobs: each child "
            "reports MPI.COMM_WORLD size 1.\n\n"
            "mpi4py was built against a different MPI implementation than "
            "this launcher -- usually a pip wheel that bundles its own "
            "runtime installed next to a system mpirun. Every distributed "
            "test would then pass while testing nothing, so this is a hard "
            "failure rather than a skip.\n\n"
            "Rebuild mpi4py against the MPI that owns this launcher:\n"
            "    pip install --no-binary=mpi4py --force-reinstall mpi4py"
            .format(launcher))
    pytest.fail(
        "'{} -np 2' did not run: the MPI installation is broken.".format(
            launcher))
