"""``Spectroscopy`` must distribute the ensemble, and get the same answer.

The driver takes an ``EnsembleSource`` -- where the ensemble lives on disk --
and routes the q-space backends through the distributed loaders, so the
configurations are read once by the master and scattered.  Two properties have
to hold together, and neither is worth much alone:

* no rank holds a replica (that is the point of the change), and
* the spectrum is the one a single, undistributed process computes.

The interpolated backend additionally has to survive the collective inside its
own construction (CellConstructor's ``ForceTensor`` broadcasts while imposing
the ASR).  That failure mode is a hang, so the cases carry a timeout and treat
expiry as a failure.
"""
from __future__ import print_function

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE = os.path.join(HERE, "_distributed_spectroscopy_probe.py")
DATA = os.path.join(REPO, "tests", "test_julia", "data")
REQUESTS = ("axis_0", "axis_1", "axis_2", "mixed")
TIMEOUT = 900

pytest.importorskip("mpi4py")
if not os.path.isdir(DATA):
    pytest.skip("q-space test ensemble not available", allow_module_level=True)


def _run(backend, workdir, n_ranks):
    os.makedirs(workdir, exist_ok=True)
    cmd = ["mpirun", "-np", str(n_ranks), sys.executable, PROBE,
           backend, DATA, workdir]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, timeout=TIMEOUT,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        pytest.fail(
            "%s on %d ranks did not finish in %ds -- most likely a deadlock "
            "between mismatched MPI collectives."
            % (backend, n_ranks, TIMEOUT))
    assert proc.returncode == 0, \
        "%s on %d ranks failed (rc=%d):\n%s" % (
            backend, n_ranks, proc.returncode, proc.stdout.decode()[-4000:])
    return [np.load(os.path.join(workdir, "rank_%d.npz" % rank))
            for rank in range(n_ranks)]


@pytest.mark.parametrize("backend", ["qspace", "atom_fourier"])
def test_spectroscopy_distributes_and_reproduces_the_spectrum(
        backend, tmp_path):
    single = _run(backend, str(tmp_path / "single"), 1)[0]
    ranks = _run(backend, str(tmp_path / "parallel"), 2)

    # The symmetry planner is not affected by the distribution: the three
    # Cartesian directions are one orbit, the mixed vector another.
    assert int(single["n_independent_runs"]) == 2
    # One engine for both runs, and none at all once everything is restored.
    assert int(single["n_engines"]) == 1
    assert int(single["n_engines_total"]) == 1

    for rank, data in enumerate(ranks):
        assert int(data["n_independent_runs"]) == 2
        assert int(data["n_engines"]) == 1
        assert int(data["n_engines_total"]) == 1
        assert bool(data["distributed"]) is True, \
            "rank %d did not take the distributed path" % rank
        assert int(data["n_global"]) == int(single["n_global"])
        assert int(data["n_local"]) < int(data["n_global"]), \
            "rank %d holds a full replica of the configurations" % rank
        assert str(data["cls"]) == str(single["cls"])

    assert (sum(int(data["n_local"]) for data in ranks)
            == int(single["n_global"])), \
        "the configurations must be partitioned, not shared or dropped"

    for name in REQUESTS:
        reference = single[name]
        scale = max(float(np.max(np.abs(reference))), 1e-30)
        for rank, data in enumerate(ranks):
            # Not bit-exact: the distributed reduction sums the per-rank
            # partials in a different order than the replicated loop.
            difference = float(np.max(np.abs(reference - data[name])))
            assert difference / scale < 1e-10, \
                "%s on rank %d differs by %.3e (rel %.3e)" % (
                    name, rank, difference, difference / scale)

    # The three symmetry-equivalent directions must assemble to one spectrum.
    for name in ("axis_1", "axis_2"):
        np.testing.assert_allclose(single[name], single["axis_0"],
                                   rtol=1e-10, atol=0)


@pytest.mark.parametrize("backend", ["qspace", "atom_fourier"])
def test_restarted_analysis_matches_the_run_that_produced_it(
        backend, tmp_path):
    workdir = str(tmp_path / "restart")
    _run(backend, workdir, 2)
    for rank in range(2):
        produced = np.load(os.path.join(workdir, "rank_%d.npz" % rank))
        restored = np.load(os.path.join(workdir, "restored_%d.npz" % rank))
        for name in REQUESTS:
            np.testing.assert_allclose(restored[name], produced[name],
                                       rtol=1e-12, atol=0)
