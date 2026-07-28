"""Restart correctness for the q-space Lanczos.

A long production run (e.g. the CsSnI3 8^3 convergence study at 400+ Lanczos
steps) must be resumable: if 400 steps turn out to be too few, the run has to
continue from a checkpoint rather than start over.  Restart is implemented by
``DynamicalLanczos.save_status`` / ``load_status`` (full Krylov state) plus the
``i_step = len(self.a_coeffs)`` continuation branch of ``run_FT``.

These tests pin that behaviour to be **bit-exact**: a chain of shorter runs
with a save/load in between must reproduce, coefficient for coefficient, the
single-shot run of the same total length.  They cover the exact production
configuration (``reorthogonalize=False``, the default) for both the plain
``QSpaceLanczos`` and the interpolating ``QSpaceTrilinearLanczos``.
"""
from __future__ import print_function

import os
import sys

import numpy as np
import pytest

# _toy_chain lives with the interpolation tests.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "test_interpolation"))

import cellconstructor as CC
import cellconstructor.Phonons
import sscha
import sscha.Ensemble

try:
    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceTrilinear as QT
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

pytestmark = pytest.mark.skipif(not _HAS_Q,
                                reason="QSpaceLanczos/Julia not available")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "test_julia", "data")
NQIRR = 3
T = 250


def _assert_bit_exact(ref, other, msg):
    for name in ("a_coeffs", "b_coeffs", "c_coeffs"):
        r = np.asarray(getattr(ref, name), dtype=float)
        o = np.asarray(getattr(other, name), dtype=float)
        assert r.shape == o.shape, \
            "%s: %s length %s != %s" % (msg, name, r.shape, o.shape)
        if r.size:
            d = np.max(np.abs(r - o))
            scale = max(np.max(np.abs(r)), 1e-30)
            assert d <= 1e-10 * scale, \
                "%s: %s differs by %.3e (rel %.3e)" % (msg, name, d, d / scale)


def _make_plain():
    dyn = CC.Phonons.Phonons(os.path.join(DATA_DIR, "dyn_gen_pop1_"), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)
    q = QL.QSpaceLanczos(ens, lo_to_split=None)
    q.ignore_harmonic = False
    q.ignore_v3 = False
    q.ignore_v4 = False
    q.init(use_symmetries=True)
    q.prepare_mode_q(0, 3)
    return q


def _make_tri():
    import _toy_chain as TC
    dync = TC.build_dyn(3)
    ensc = TC.make_ensemble(dync, 300.0, 3000, seed=11, g3=0.1)
    li = QT.QSpaceTrilinearLanczos(ensc, fine_mesh=(1, 1, 6), atom_fourier=True)
    li.init(use_symmetries=True)
    li.prepare_mode_q(0, 5)
    return li


@pytest.mark.parametrize("maker,total", [(_make_plain, 10), (_make_tri, 12)])
def test_in_memory_restart_is_bit_exact(maker, total):
    """Calling run_FT twice on the same object continues the recursion exactly."""
    ref = maker()
    ref.run_FT(total, verbose=False)

    split = maker()
    split.run_FT(total // 2, verbose=False)
    split.run_FT(total - total // 2, verbose=False)

    _assert_bit_exact(ref, split, "in-memory restart")


@pytest.mark.parametrize("maker,total", [(_make_plain, 10), (_make_tri, 12)])
def test_disk_restart_is_bit_exact(maker, total, tmp_path):
    """save_status -> fresh object + init + load_status -> continue == single shot.

    This is the production resume path: the object is rebuilt from scratch
    (kernel included, for the trilinear class) and only the Krylov state is
    reloaded from disk.
    """
    ref = maker()
    ref.run_FT(total, verbose=False)

    ckpt = str(tmp_path / "ckpt")
    first = maker()
    first.run_FT(total // 2, verbose=False)
    first.save_status(ckpt)
    del first

    resumed = maker()          # identical construction + init + perturbation
    resumed.load_status(ckpt)
    resumed.run_FT(total - total // 2, verbose=False)

    _assert_bit_exact(ref, resumed, "disk restart")


def test_multi_chunk_disk_restart(tmp_path):
    """Three chunks with a save/load at every boundary still match one shot.

    Guards against any state that survives one restart but drifts across
    repeated ones ("more than 400 steps needed" may chain several restarts).
    """
    total = 12
    ref = _make_plain()
    ref.run_FT(total, verbose=False)

    ckpt = str(tmp_path / "chain")
    lanc = _make_plain()
    done = 0
    for chunk in (4, 4, 4):
        lanc.run_FT(chunk, verbose=False)
        lanc.save_status(ckpt)
        done += chunk
        reloaded = _make_plain()
        reloaded.load_status(ckpt)
        lanc = reloaded
    assert done == total
    _assert_bit_exact(ref, lanc, "multi-chunk disk restart")
