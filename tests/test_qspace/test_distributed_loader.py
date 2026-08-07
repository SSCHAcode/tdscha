"""Distributing the ensemble must not change the Lanczos coefficients.

``load_distributed_tdscha`` scatters the configurations across MPI ranks so no
rank holds a full replica.  This checks the property that makes it usable: a
distributed run reproduces the replicated one, for the plain q-space Lanczos
*and* for the atom-Fourier interpolated one.

The interpolated case is the delicate one.  Building
``QSpaceAtomFourierLanczos`` interpolates the dynamical matrix, and that goes
through CellConstructor's ``ForceTensor`` (``Center`` and ``Apply_ASR``), each
of which ends in an unconditional ``Settings.broadcast``.  Left inside the
master-only branch it deadlocks -- master in the ASR broadcast, workers in the
metadata broadcast.  ``prepare_distributed_construction`` moves exactly that
step in front of the master/worker split so every rank runs it together, and
the master then reads the configurations alone.  A regression here shows up as
a hang, so these tests carry a timeout and treat expiry as failure.

Each case runs in its own ``mpirun`` because every matrix-vector product is a
collective: a rank that built a second, replicated object and stepped it alone
would hang the others.
"""
from __future__ import print_function

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE = os.path.join(HERE, "_distributed_probe.py")
DATA = os.path.join(REPO, "tests", "test_julia", "data")
TIMEOUT = 900

pytest.importorskip("mpi4py")
if not os.path.isdir(DATA):
    pytest.skip("q-space test ensemble not available", allow_module_level=True)


def _run(launcher, mode, out, n_ranks):
    cmd = [launcher, "-np", str(n_ranks), sys.executable, PROBE, mode, DATA,
           out]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, timeout=TIMEOUT,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        pytest.fail("%s did not finish in %ds -- most likely a deadlock "
                    "between mismatched MPI collectives." % (mode, TIMEOUT))
    assert proc.returncode == 0, \
        "%s failed (rc=%d):\n%s" % (mode, proc.returncode,
                                    proc.stdout.decode()[-3000:])
    assert os.path.exists(out), "%s produced no output" % mode
    return np.load(out, allow_pickle=True)


def _assert_same_coeffs(serial, dist, msg):
    for name in "abc":
        s, d = serial[name], dist[name]
        assert s.shape == d.shape, \
            "%s: %s shape %s != %s" % (msg, name, s.shape, d.shape)
        diff = float(np.max(np.abs(s - d)))
        scale = max(float(np.max(np.abs(s))), 1e-30)
        # Not bit-exact by construction: the distributed reduction sums the
        # per-rank partials in a different order than the replicated loop.
        assert diff / scale < 1e-10, \
            "%s: %s differs by %.3e (rel %.3e)" % (msg, name, diff,
                                                   diff / scale)


@pytest.mark.parametrize("kind", ["plain", "tri"])
def test_distributed_matches_replicated(kind, tmp_path, multi_rank_mpirun):
    serial = _run(multi_rank_mpirun, "serial-%s" % kind,
                  str(tmp_path / "serial.npz"), 1)
    dist = _run(multi_rank_mpirun, "dist-%s" % kind,
                str(tmp_path / "dist.npz"), 2)

    # The distributed object really did split the configurations ...
    assert bool(dist["distributed"]) is True
    assert int(dist["n_global"]) == int(serial["n_global"])
    assert int(dist["n_local"]) < int(serial["n_local"]), \
        "each rank should hold a strict subset of the configurations"
    # ... and it is still the class we asked for.
    assert str(dist["cls"]) == str(serial["cls"])
    # The ensemble Bloch fields are indexed by the COARSE q count in both
    # cases: for the interpolated class that is not n_q, which is why the
    # loader ships the true leading dimension instead of assuming one.
    assert int(dist["xq_nq"]) == int(serial["xq_nq"])

    _assert_same_coeffs(serial, dist, "distributed %s" % kind)


def test_interpolated_master_only_matches_build_everywhere(
        tmp_path, multi_rank_mpirun):
    """The production loader must reproduce the replicating oracle exactly.

    ``build_on_all_ranks=True`` rebuilds the whole object identically on every
    rank, so it cannot disagree with itself about the interpolated mode basis.
    The production path instead interpolates on all ranks and then takes the
    master's copy of that basis; if those two ever produced different
    polarization vectors, the ensemble Bloch fields would be projected in one
    gauge and contracted in another, and the coefficients would move.
    """
    oracle = _run(multi_rank_mpirun, "oracle-tri",
                  str(tmp_path / "oracle.npz"), 2)
    dist = _run(multi_rank_mpirun, "dist-tri",
                str(tmp_path / "dist.npz"), 2)

    assert bool(dist["distributed"]) is True
    assert int(dist["n_global"]) == int(oracle["n_global"])
    assert int(dist["n_local"]) == int(oracle["n_local"])
    _assert_same_coeffs(oracle, dist, "master-only vs build-everywhere")


def test_collective_left_in_the_constructor_fails_loudly(
        tmp_path, multi_rank_mpirun):
    """The historical defect must not be able to come back silently.

    A collective the master runs alone does not raise anywhere: MPI matches
    it against whatever the workers happen to be waiting in and hands them
    the wrong payload.  Before the sentinel in the metadata, that produced a
    hang -- and, when it did not hang, a plausible but wrong spectrum.  The
    loader must now diagnose it and stop the job.
    """
    cmd = [multi_rank_mpirun, "-np", "2", sys.executable, PROBE,
           "guard-tri", DATA, str(tmp_path / "unused.npz")]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, timeout=TIMEOUT,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        pytest.fail("a mismatched collective must abort, not hang")
    output = proc.stdout.decode()
    assert proc.returncode != 0, \
        "a mismatched collective must not be reported as success:\n%s" % (
            output[-3000:])
    assert "distributed loader received something other than its own" \
        in output, output[-3000:]
    assert not os.path.exists(str(tmp_path / "unused.npz"))
