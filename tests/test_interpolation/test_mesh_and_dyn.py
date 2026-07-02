"""
Unit tests for the fine-mesh utilities and the interpolated dynamical matrix
(no Lanczos runs; no Julia needed except through module imports).

Covers:
- fine mesh generation (Gamma first, closure under q -> -q and pair map)
- O(1) index lookup consistency with the O(n^2) distance search
- interpolated dyn: exact at commensurate points, EXACT everywhere for the
  nearest-neighbor spring chain (its force constants are strictly range-1,
  so centered Fourier interpolation has zero error -- a sharp test),
- time-reversal gauge e(-q) = conj(e(q)),
- acoustic sum rule of the interpolated dyn (w_acoustic -> 0 smoothly).
"""
import os, sys
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest

import cellconstructor as CC
import cellconstructor.Methods

import _toy_chain as TC

try:
    import tdscha.QSpaceInterpolation as QI
    _OK = True
except Exception:
    _OK = False

pytestmark = pytest.mark.skipif(not _OK, reason="tdscha.QSpaceInterpolation not importable")


def test_mesh_generation_and_lookup():
    unit = TC.build_unit_structure()
    mesh = (2, 3, 4)
    q_points, idx = QI.generate_fine_mesh(unit, mesh)

    assert len(q_points) == 24
    assert np.allclose(q_points[0], 0.0), "Gamma must be first"

    lookup = QI.build_q_index_lookup(q_points, unit, mesh)
    assert len(lookup) == 24, "lookup keys must be unique"

    bg = unit.get_reciprocal_vectors() / (2 * np.pi)

    # every q must find itself through the hash
    for iq, q in enumerate(q_points):
        assert lookup[QI._mesh_key(q, unit, mesh)] == iq

    # closure under q -> -q, verified against the O(n^2) distance search
    for iq, q in enumerate(q_points):
        jq = lookup[QI._mesh_key(-q, unit, mesh)]
        d = CC.Methods.get_min_dist_into_cell(bg, -q, q_points[jq])
        assert d < 1e-8


def test_pair_map_matches_parent_search():
    """The O(N_f) mesh-index pair map must agree with the parent's
    O(n^2) minimum-distance search on the same q list."""
    unit = TC.build_unit_structure()
    mesh = (1, 2, 3)
    q_points, idx = QI.generate_fine_mesh(unit, mesh)
    lookup = QI.build_q_index_lookup(q_points, unit, mesh)
    bg = unit.get_reciprocal_vectors() / (2 * np.pi)

    for iq_pert in range(len(q_points)):
        n_pert = idx[iq_pert]
        for iq1 in range(len(q_points)):
            # index arithmetic
            key = tuple((n_pert - idx[iq1]) % np.asarray(mesh))
            iq2_fast = lookup[key]
            # reference: q2 = q_pert - q1 modulo G
            q_target = q_points[iq_pert] - q_points[iq1]
            d = CC.Methods.get_min_dist_into_cell(bg, q_target, q_points[iq2_fast])
            assert d < 1e-8, (iq_pert, iq1)


def test_dyn_interpolation_exact_for_range1_model():
    """The spring chain has strictly nearest-cell force constants: centered
    Fourier interpolation from ANY supercell L >= 2 must reproduce the
    L' = 2L dispersion exactly (not just approximately)."""
    dyn2 = TC.build_dyn(3)
    dyn4 = TC.build_dyn(6)

    q_fine, _ = QI.generate_fine_mesh(dyn2.structure, (1, 1, 6))
    w_int, pols_int = QI.interpolate_dyn_fine(dyn2, q_fine, use_asr=True)

    # reference frequencies from the direct 1x1x6 dyn
    lookup = QI.build_q_index_lookup(q_fine, dyn2.structure, (1, 1, 6))
    for jq, q in enumerate(dyn4.q_tot):
        iq = lookup[QI._mesh_key(np.asarray(q), dyn2.structure, (1, 1, 6))]
        w_ref, _ = dyn4.DyagDinQ(jq)
        assert np.max(np.abs(np.sort(w_ref) - np.sort(w_int[:, iq]))) < 1e-9, \
            "interpolated dispersion wrong at q={}".format(q)


def test_tensor2_sign_convention():
    """Pin the Tensor2.Interpolate phase convention: Interpolate(-q) must
    reproduce dyn.dynmats[q] at commensurate NON-TRI q (a q <-> -q swap is
    invisible to frequencies and to the constructive TRI gauge, so it must
    be pinned at the dynamical-matrix level)."""
    import cellconstructor.ForceTensor
    dyn = TC.build_dyn(3)
    uc = dyn.structure
    sc = uc.generate_supercell(dyn.GetSupercell())
    t2 = CC.ForceTensor.Tensor2(uc, sc, dyn.GetSupercell())
    t2.SetupFromPhonons(dyn)
    t2.Center()
    for iq, q in enumerate(dyn.q_tot):
        if np.linalg.norm(q) < 1e-8:
            continue
        D_ref = dyn.dynmats[iq]
        D_minus = t2.Interpolate(-np.asarray(q), asr=False, lo_to_splitting=False)
        assert np.max(np.abs(D_minus - D_ref)) < 1e-12, \
            "Tensor2.Interpolate sign convention changed!"


def test_dyn_interpolation_tri_gauge_and_asr():
    dyn = TC.build_dyn(3)
    mesh = (1, 1, 6)
    q_fine, _ = QI.generate_fine_mesh(dyn.structure, mesh)
    w_int, pols_int = QI.interpolate_dyn_fine(dyn, q_fine, use_asr=True)
    lookup = QI.build_q_index_lookup(q_fine, dyn.structure, mesh)

    # TRI gauge: e(-q) = conj(e(q)), w(-q) = w(q)
    for iq, q in enumerate(q_fine):
        jq = lookup[QI._mesh_key(-q, dyn.structure, mesh)]
        assert np.allclose(w_int[:, iq], w_int[:, jq], atol=1e-12)
        assert np.allclose(pols_int[:, :, jq], np.conj(pols_int[:, :, iq]),
                           atol=1e-10)

    # ASR: at Gamma exactly three zero modes; smallest nonzero acoustic
    # frequency at the closest-to-Gamma fine point must be positive and small
    w_gamma = np.sort(np.abs(w_int[:, 0]))
    assert np.all(w_gamma[:3] < 1e-7), "Gamma translations must be at zero"
    assert np.all(w_gamma[3:] > 1e-4), "optical modes must be finite"

    # acoustic branch grows away from Gamma (stability)
    assert np.all(w_int[:, 1:] > -1e-8), "no imaginary frequencies off Gamma"


def test_asr_zero_mode_projection():
    """Rigid shifts of displacements and constant force offsets must be
    exactly projected out of the fine-mesh fields (plain window):
    the estimator fields are invariant under u -> u + const and
    f -> f + const/atom (acoustic sum rule at the field level)."""
    dyn = TC.build_dyn(3)
    ens = TC.make_ensemble(dyn, 250.0, 30, seed=5, g3=0.1)

    li_ref = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, 6), prefilter=False)
    X_ref, Y_ref = li_ref.X_q.copy(), li_ref.Y_q.copy()

    # rigid displacement shift + constant per-atom force offset
    ens.u_disps = ens.u_disps + np.tile([0.13, -0.07, 0.21],
                                        ens.u_disps.shape[1] // 3)[None, :]
    ens.forces = ens.forces + np.array([0.011, -0.023, 0.005])[None, None, :]

    li_shift = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, 6), prefilter=False)

    assert np.max(np.abs(li_shift.X_q - X_ref)) < 1e-10, \
        "rigid displacement leaked into the fields"
    assert np.max(np.abs(li_shift.Y_q - Y_ref)) < 1e-10, \
        "constant force offset leaked into the fields"
