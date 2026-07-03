"""Tests for the design-level ASR windows (plan section 5.7).

The "asr" window design uses doubled-support (2L) windows constrained to
exactly uniform class sums, fitted to the projection of the minimal-image
kernel onto the acoustic-sum-rule subspace. Centering, ASR, commensurate
exactness and w<->v symmetrization hold simultaneously in one constrained
fit (no iterative ASR/symmetrization alternation as in ForceTensor).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _toy_chain as TC

import tdscha.QSpaceInterpolation as QI


# ----------------------------------------------------------------------
# kernel-level checks (no ensembles, fast)
# ----------------------------------------------------------------------

def _class_sum_spread(w, L):
    cs = np.array([w[d] + w[d + L] for d in range(L)])
    return np.max(cs) - np.min(cs)


def _partition_dev(K, L, S):
    cls = QI._class_images_gen(L, S)
    flat = K.ravel()
    return max(abs(sum(flat[i] for i in cls[c]) - L) for c in cls)


def _asr_col_dev(K, L, S):
    n = 2 * S - 1
    dev = 0.0
    for i in range(n):
        sums = []
        for d2 in range(L):
            tot = sum(K[i, j] for j in
                      (d2 + b * L + S - 1 for b in range(-4, 5))
                      if 0 <= j < n)
            sums.append(tot)
        dev = max(dev, max(sums) - min(sums))
    return dev


def _asr_diag_dev(K, L, S):
    n = 2 * S - 1
    dev = 0.0
    for Delta in range(-(n - 1), n):
        sums = []
        for d in range(L):
            tot, hit = 0.0, False
            for b in range(-4, 5):
                i, j = d + b * L + S - 1, d + Delta + b * L + S - 1
                if 0 <= i < n and 0 <= j < n:
                    tot += K[i, j]
                    hit = True
            if hit:
                sums.append(tot)
        if len(sums) > 1:
            dev = max(dev, max(sums) - min(sums))
    return dev


@pytest.mark.parametrize("L", [3, 4])
def test_asr_design_constraints_exact(L):
    """Uniform class sums, partition of unity and all three kernel ASR
    conditions must hold at machine precision; the fit must realize the
    ASR-projected target closely."""
    S = 2 * L
    passes = QI.get_window_design_asr(L, K=3)
    for p in passes:
        for w in p:
            assert len(w) == S
            assert _class_sum_spread(w, L) < 1e-10

    Kt = sum(QI._kernel_sym(*p) for p in passes)
    assert _partition_dev(Kt, L, S) < 1e-10
    assert _asr_col_dev(Kt, L, S) < 1e-10
    assert _asr_col_dev(Kt.T, L, S) < 1e-10
    assert _asr_diag_dev(Kt, L, S) < 1e-10

    tgt = QI.asr_projected_target_1d(L, S)
    rms = np.sqrt(np.mean((Kt - tgt) ** 2))
    assert rms < {3: 0.05, 4: 0.12}[L]


def test_projected_target_is_asr_and_closer_than_plain():
    """The ASR projection of the minimal-image kernel satisfies the
    constraints exactly and is strictly closer to L*M than the tent."""
    L, S = 3, 6
    tgt_mi = QI.embed_minimal_image_target_1d(L, S)
    tgt = QI.asr_projected_target_1d(L, S)
    assert _partition_dev(tgt, L, S) < 1e-10
    assert _asr_col_dev(tgt, L, S) < 1e-10
    assert _asr_diag_dev(tgt, L, S) < 1e-10

    plain = np.zeros(S)
    plain[:L] = 1.0
    K_plain = QI._kernel_sym(plain, plain, plain)
    d_proj = np.sqrt(np.mean((tgt - tgt_mi) ** 2))
    d_plain = np.sqrt(np.mean((K_plain - tgt_mi) ** 2))
    assert d_proj < 0.6 * d_plain


def test_asr_design_l2_is_plain():
    """For L <= 2 the tent is the minimal-image kernel and plain windows
    already sit on the constraint manifold: nothing to design."""
    for L in (1, 2):
        passes = QI.get_window_design_asr(L, K=3)
        assert len(passes) == 1
        for w in passes[0]:
            assert np.allclose(w[:L], 1.0)
            assert np.allclose(w[L:], 0.0)


def test_decay_weighted_projection_short_range():
    """The decay-weighted projection concentrates fidelity at small
    spreads (where the physical Phi3 lives)."""
    L, S = 3, 6
    tgt_mi = QI.embed_minimal_image_target_1d(L, S)
    uni = QI.asr_projected_target_1d(L, S)
    dec = QI.asr_projected_target_1d(L, S, decay_weight=0.4)
    # error on the spread<=1 entries must shrink
    n = 2 * S - 1
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if QI._spread_ext(i - (S - 1), j - (S - 1)) <= 1:
                mask[i, j] = True
    e_uni = np.max(np.abs((uni - tgt_mi)[mask]))
    e_dec = np.max(np.abs((dec - tgt_mi)[mask]))
    assert e_dec < 0.5 * e_uni


# ----------------------------------------------------------------------
# estimator-level checks (small ensembles)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def commensurate_pair():
    """asr-design and plain interp objects on fine mesh == coarse mesh."""
    L = 3
    dyn = TC.build_dyn(L)
    T, N, seed, g3 = 250.0, 40, 42, 0.1

    def make(design, **kw):
        ens = TC.make_ensemble(TC.build_dyn(L), T, N, seed=seed, g3=g3)
        lanc = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, L),
                                      window_design=design, **kw)
        lanc.init(use_symmetries=True)
        return lanc

    return make("plain"), make("asr")


def test_asr_commensurate_identity(commensurate_pair):
    """On a fully commensurate fine mesh every second-period phase is 1,
    each pass collapses to (class-sum product) x plain, and the hard
    partition constraint sum_r c_z c_w c_v = 1 makes the pass sum
    reproduce the plain estimator exactly, configuration by
    configuration."""
    plain, asr = commensurate_pair
    band = int(np.where(plain.valid_modes_q[:, 0])[0][0])

    for lanc in (plain, asr):
        lanc.prepare_mode_q(0, band)
    R1 = plain.get_R1_q()
    nb = plain.n_bands
    alpha1 = plain._flatten_blocks(
        [np.zeros((nb, nb), dtype=np.complex128)
         for _ in plain.unique_pairs])

    fp_p, d2v_p = plain._call_julia_qspace(R1, alpha1.copy())
    fp_a, d2v_a = asr._call_julia_qspace(R1, alpha1.copy())

    scale = max(np.max(np.abs(fp_p)), 1e-30)
    assert np.max(np.abs(fp_a - fp_p)) < 1e-9 * scale
    for bp, ba in zip(d2v_p, d2v_a):
        bscale = max(np.max(np.abs(bp)), 1e-30)
        assert np.max(np.abs(ba - bp)) < 1e-9 * bscale


def test_asr_fields_flag_is_noop_for_asr_design(commensurate_pair):
    """No per-config field projection is applied under the asr design
    (the class-sum construction makes the q->0 acoustic contraction vanish
    automatically): toggling asr_fields must not change the fields."""
    _, asr = commensurate_pair
    ens = TC.make_ensemble(TC.build_dyn(3), 250.0, 40, seed=42, g3=0.1)
    lanc2 = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, 3),
                                   window_design="asr", asr_fields=False)
    lanc2.init(use_symmetries=True)
    for (Xa, Ya), (Xb, Yb) in zip(asr._field_sets, lanc2._field_sets):
        assert np.array_equal(Xa, Xb)
        assert np.array_equal(Ya, Yb)


@pytest.mark.slow
def test_acoustic_vertex_decay():
    """The interpolated D3 vertex on an acoustic leg q2 -> 0 must vanish.

    The leak channel is an acoustic leg at FIXED FINITE partner q1 (the
    q_pert = 0 pair line is leak-free by phase cancellation), and it only
    contracts tensor entries whose legs span three distinct cells, so the
    toy needs the three-body term. The minimal_image design must show the
    plateau (ASR leak); the asr design must track the plain (exact-ASR)
    decay."""
    T, N, g3, g3b, Lc, Lf = 300.0, 2000, 0.02, 0.4, 3, 24

    def make(design, **kw):
        ens = TC.make_ensemble(TC.build_dyn(Lc), T, N, seed=303, g3=g3,
                               g3b=g3b)
        lanc = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, Lf),
                                      window_design=design, **kw)
        lanc.init(use_symmetries=True)
        return lanc

    def acoustic_leak(lanc, n_probe=1):
        iq_pert = int(lanc._q_lookup[(0, 0, Lf // 2)])   # q_pert = 1/2
        band = int(np.argmax(np.where(lanc.valid_modes_q[:, iq_pert],
                                      lanc.w_q[:, iq_pert], -np.inf)))
        lanc.prepare_mode_q(iq_pert, band)
        R1 = lanc.get_R1_q()
        nb = lanc.n_bands
        alpha1 = lanc._flatten_blocks(
            [np.zeros((nb, nb), dtype=np.complex128)
             for _ in lanc.unique_pairs])
        _, d2v = lanc._call_julia_qspace(R1, alpha1)
        iq2 = int(lanc._q_lookup[(0, 0, n_probe)])       # acoustic leg
        iq1 = int(lanc.q_pair_map[iq2])
        lo, hi = min(iq1, iq2), max(iq1, iq2)
        blk = d2v[lanc.unique_pairs.index((lo, hi))]
        w2 = np.where(lanc.valid_modes_q[:, iq2],
                      np.abs(lanc.w_q[:, iq2]), np.inf)
        ac = int(np.argmin(w2))
        row = blk[ac, :] if lo == iq2 else blk[:, ac]
        return float(np.linalg.norm(row))

    leak_plain = acoustic_leak(make("plain"))
    leak_mimg = acoustic_leak(make("minimal_image"))
    leak_asr = acoustic_leak(make("asr"))

    # measured at N=4000, Lf=48 (q2=0.042): plain 2.0e-6, mimg 7.9e-6,
    # asr 3.6e-6; thresholds leave room for the smaller N here
    assert leak_mimg > 2.0 * leak_plain
    assert leak_asr < 0.65 * leak_mimg
