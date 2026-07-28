"""Distributing the ensemble must not change the Lanczos coefficients.

``load_distributed_tdscha`` scatters the configurations across MPI ranks so no
rank holds a full replica.  This checks the property that makes it usable: a
distributed run reproduces the replicated one, for the plain q-space Lanczos
*and* for the atom-Fourier interpolated one.

The interpolated case needs its own strategy. ``QSpaceAtomFourierLanczos``
performs MPI collectives inside its constructor
(``interpolate_dyn_fine`` -> ``ForceTensor.Apply_ASR`` -> ``broadcast``), so the
master-builds-then-scatters path deadlocks: the master blocks in the ASR
broadcast while the workers block in the metadata broadcast. The interpolated
loader therefore builds on every rank and slices afterwards
(``build_on_all_ranks=True``).  A regression here shows up as a hang, so these
tests carry a timeout and treat expiry as failure.

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


def _run(mode, out, n_ranks):
    cmd = ["mpirun", "-np", str(n_ranks), sys.executable, PROBE, mode, DATA, out]
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
def test_distributed_matches_replicated(kind, tmp_path):
    serial = _run("serial-%s" % kind, str(tmp_path / "serial.npz"), 1)
    dist = _run("dist-%s" % kind, str(tmp_path / "dist.npz"), 2)

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
