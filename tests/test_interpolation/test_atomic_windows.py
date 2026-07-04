"""Pure geometry tests for atom-resolved pinned windows."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _toy_chain as TC

try:
    import tdscha.QSpaceInterpolation as QI
    _OK = True
except Exception:
    _OK = False

pytestmark = pytest.mark.skipif(not _OK, reason="tdscha.QSpaceInterpolation not importable")


def test_atomic_leg_images_break_l2_tie_by_basis_offset():
    """At L=2 the cell-index kernel cannot choose between +/-1 images, but
    the atomic-basis distance can."""
    s = TC.build_unit_structure()
    imgs = QI._atomic_leg_images(
        s, np.array([1, 1, 2]), far=2, a=0, b=1,
        d=np.array([0, 0, 1]))
    assert len(imgs) == 1
    assert imgs[0][0] == (0, 0, -1)
    assert imgs[0][1] == pytest.approx(1.0)


def test_atomic_window_passes_are_partitioned_by_class():
    """Every atom/class leg window must assign unit total weight among its
    geometry-selected images."""
    s = TC.build_unit_structure()
    L = np.array([1, 1, 2])
    passes = QI.get_atomic_window_passes(s, L, far=2)
    assert len(passes) == s.N_atoms

    for a, (z, w, v) in enumerate(passes):
        assert z == {(a, (0, 0, 0)): 1.0}
        assert w == v
        for b in range(s.N_atoms):
            for dz in range(L[2]):
                total = 0.0
                for (bb, ext), val in w.items():
                    if bb == b and np.array_equal(np.asarray(ext) % L,
                                                  [0, 0, dz]):
                        total += val
                assert total == pytest.approx(1.0)


def test_extended_window_to_qweights_keeps_supercell_phase():
    """An image outside the fundamental supercell maps to the periodic atom
    plus the correct Bloch phase across the supercell."""
    s = TC.build_unit_structure()
    L = np.array([1, 1, 2])
    sc = s.generate_supercell(L)
    itau = sc.get_itau(s) - 1
    r_lat = sc.coords - s.coords[itau]
    frac = np.linalg.solve(s.unit_cell.T, r_lat.T).T
    cell_idx = np.round(frac).astype(int) % L

    bg = s.get_reciprocal_vectors() / (2 * np.pi)
    q_points = np.array([np.array([0.0, 0.0, 0.25]) @ bg])
    W = QI._window_map_to_qweights(
        {(1, (0, 0, -1)): 1.0}, q_points, s, itau, cell_idx, L)

    k = np.where((itau == 1) & (cell_idx[:, 2] == 1))[0][0]
    assert W.shape == (1, sc.N_atoms)
    assert W[0, k] == pytest.approx(-1.0 + 0.0j)
    assert np.count_nonzero(np.abs(W) > 1e-12) == 1
