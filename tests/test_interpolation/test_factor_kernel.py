"""
Tests of the symmetric-power factor centering (new_plan.tex,
Modules/QSpaceFactorKernel.py + window_design="factor").

Validation ladder implemented here (new_plan.tex section "Validation and
known pitfalls"):
 1. geometry target: permutation covariance, tie splitting, partition
    normalization to N_c;
 2. factor windows: exactly uniform folded class sums (the manifold that
    makes every hard constraint structural);
 3. kernel level, materialized on a small chain: partition of unity,
    per-leg ASR image-sum constancy, permutation symmetry of the
    orbit-summed kernel;
 4. per-configuration commensurate identity of the runtime operator
    (Lanczos a/b coefficients equal the plain coarse Lanczos);
 5. Hermiticity |b - c| on a genuinely interpolated mesh.
"""
import os, sys
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import numpy as np
import pytest

import _toy_chain as TC
import tdscha.QSpaceFactorKernel as FK

try:
    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceInterpolation as QI
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

L = 3
SC = (1, 1, L)


@pytest.fixture(scope="module")
def chain():
    dyn = TC.build_dyn(L)
    return dyn.structure


# =====================================================================
# 1. geometry target
# =====================================================================

def test_target_partition_normalization(chain):
    """Every folded class must carry total weight N_c (the class sum of
    the plain tent kernel), exactly."""
    for order in (3, 4):
        atoms, cells, w = FK.build_target_tuples(chain, SC, order, far=2)
        n_c = int(np.prod(SC))
        sums = {}
        for i in range(len(w)):
            key = (tuple(atoms[i]),
                   tuple(tuple(c % np.array(SC)) for c in cells[i]))
            sums[key] = sums.get(key, 0.0) + w[i]
        vals = np.array(list(sums.values()))
        assert np.max(np.abs(vals - n_c)) < 1e-10
        nat = chain.N_atoms
        assert len(sums) == nat ** order * n_c ** (order - 1)


def test_target_permutation_covariance(chain):
    """The complete-graph cost is leg-symmetric: swapping two legs (and
    re-gauging to keep leg 0 at cell 0) maps target tuples to target
    tuples with the same weight."""
    atoms, cells, w = FK.build_target_tuples(chain, SC, 3, far=2)
    tset = {}
    for i in range(len(w)):
        key = (tuple(atoms[i]), tuple(tuple(c) for c in cells[i]))
        tset[key] = tset.get(key, 0.0) + w[i]
    for key, val in tset.items():
        (a1, a2, a3) = key[0]
        (c1, c2, c3) = [np.array(c) for c in key[1]]
        # swap legs 1 and 2 (both non-anchor: no re-gauge needed)
        k_sw = ((a1, a3, a2), (tuple(c1), tuple(c3), tuple(c2)))
        assert abs(tset.get(k_sw, 0.0) - val) < 1e-10
        # swap anchor and leg 1, re-gauge leg order: anchor at cell 0
        k_an = ((a2, a1, a3),
                ((0, 0, 0), tuple(c1 - c2), tuple(c3 - c2)))
        assert abs(tset.get(k_an, 0.0) - val) < 1e-10


def test_target_tie_split(chain):
    """Weights are N_c / n_ties: every weight divides N_c exactly and
    tuples of one class share equal weights."""
    atoms, cells, w = FK.build_target_tuples(chain, SC, 3, far=2)
    n_c = int(np.prod(SC))
    ratios = n_c / w
    assert np.max(np.abs(ratios - np.round(ratios))) < 1e-8


# =====================================================================
# 2. factor windows: the uniform-class-sum manifold
# =====================================================================

def test_dictionary_class_sums(chain):
    wins = FK.default_dictionary(chain, SC, far=3)
    assert len(wins) > 4
    for win in wins:
        assert win.class_sum in (0.0, 1.0)
        assert win.check_class_sums(SC) < 1e-10
        if win.class_sum == 1.0:
            # every folded class must be covered
            covered = set()
            Lv = np.array(SC)
            for (a, u) in win.entries:
                covered.add((a, tuple(np.array(u) % Lv)))
            assert len(covered) == chain.N_atoms * int(np.prod(SC))


def test_plain_window_is_type1(chain):
    a0w = FK.plain_window(chain.N_atoms, SC)
    assert a0w.check_class_sums(SC, nat=chain.N_atoms) < 1e-15


def test_missing_type1_class_is_rejected(chain):
    """The structural commensurate-identity proof needs every folded
    atom/cell class, not only the classes present in the sparse map."""
    bad = FK.FactorWindow({(0, (0, 0, 0)): 1.0}, 1.0,
                          label="missing-classes")
    assert bad.check_class_sums(SC, nat=chain.N_atoms) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="uniform class-sum"):
        FK.fit_symmetric_factors(chain, SC, 3, dictionary=[bad], max_rank=1)


# =====================================================================
# 3. kernel-level constraints (materialized, small chain)
# =====================================================================

def _corr_dict(win, nat, order, box_lo, box_hi):
    atoms, cells, vals = FK.dense_kernel(win, nat, SC, order,
                                         (box_lo, box_hi))
    out = {}
    for i in range(len(vals)):
        if abs(vals[i]) > 1e-15:
            key = (tuple(atoms[i]), tuple(tuple(c) for c in cells[i]))
            out[key] = vals[i]
    return atoms, cells, vals


def test_kernel_constraints_on_manifold(chain):
    """For a genuinely oscillating type-1 window: (a) folded partition
    sums equal N_c for every class; (b) per-leg ASR: the image sum at
    fixed other legs is independent of the summed (atom, cell); (c) the
    orbit-summed kernel is permutation symmetric."""
    nat = chain.N_atoms
    order = 3
    win = FK.gaussian_window(chain, SC, chain.coords[0], 3.0, far=2)
    at, cl, vals = win.arrays()
    lo = cl.min(axis=0).copy()
    hi = cl.max(axis=0).copy()
    box_lo = lo - hi
    box_hi = hi - lo
    atoms, cells, kv = FK.dense_kernel(win, nat, SC, order,
                                       (box_lo, box_hi))

    Lv = np.array(SC)
    n_c = int(np.prod(SC))

    # (a) partition: for the raw power of a type-1 window the class sum
    # is N_c for every folded class (matches K0 => correction sums to 0)
    sums = {}
    for i in range(len(kv)):
        key = (tuple(atoms[i]),
               tuple(tuple(c % Lv) for c in cells[i]))
        sums[key] = sums.get(key, 0.0) + kv[i]
    vals_arr = np.array(list(sums.values()))
    assert len(sums) == nat ** order * n_c ** (order - 1)
    assert np.max(np.abs(vals_arr - n_c)) < 1e-8

    # (b) ASR on leg 2: fix leg 1 (atom, extended cell) and the folded
    # class of leg 2; the image sum must be the same for every
    # (atom, folded cell) of leg 2
    kdict = {}
    for i in range(len(kv)):
        key = (tuple(atoms[i]), tuple(tuple(c) for c in cells[i]))
        kdict[key] = kdict.get(key, 0.0) + kv[i]
    # choose a fixed leg-1 state with support
    a1, a2 = 0, 1
    c2 = (0, 0, 1)
    ref = None
    for b3 in range(nat):
        for d3 in itertools.product(range(SC[0]), range(SC[1]),
                                    range(SC[2])):
            tot = 0.0
            image_ranges = []
            for axis in range(3):
                mlo = (box_lo[axis] - d3[axis]) // SC[axis] - 1
                mhi = (box_hi[axis] - d3[axis]) // SC[axis] + 2
                image_ranges.append(range(mlo, mhi))
            for image in itertools.product(*image_ranges):
                cell3 = tuple(d3[axis] + image[axis] * SC[axis]
                              for axis in range(3))
                tot += kdict.get(((a1, a2, b3),
                                  ((0, 0, 0), c2, cell3)), 0.0)
            if ref is None:
                ref = tot
            assert abs(tot - ref) < 1e-8, \
                "ASR image sum depends on the summed leg ({}, {})".format(
                    b3, d3)

    # (c) permutation symmetry (legs 1 <-> 2)
    for i in range(0, len(kv), 7):
        if abs(kv[i]) < 1e-12:
            continue
        key_sw = ((atoms[i, 0], atoms[i, 2], atoms[i, 1]),
                  (tuple(cells[i, 0]), tuple(cells[i, 2]),
                   tuple(cells[i, 1])))
        assert abs(kdict.get(key_sw, 0.0) - kv[i]) < 1e-8


def test_gram_identity_matches_dense(chain):
    """The matrix-free Gram <corr_n(A), corr_n(B)> = sum_v g_AB(v)^n must
    equal the dense inner product of the materialized kernels."""
    nat = chain.N_atoms
    win_a = FK.gaussian_window(chain, SC, chain.coords[0], 3.0, far=1)
    win_b = FK.minimal_image_window(chain, SC, 1, far=1)
    order = 3
    for win in (win_a, win_b):
        at, cl, _ = win.arrays()
    lo = np.minimum(win_a.arrays()[1].min(axis=0),
                    win_b.arrays()[1].min(axis=0))
    hi = np.maximum(win_a.arrays()[1].max(axis=0),
                    win_b.arrays()[1].max(axis=0))
    box = (lo - hi, hi - lo)
    atoms, cells, va = FK.dense_kernel(win_a, nat, SC, order, box)
    _, _, vb = FK.dense_kernel(win_b, nat, SC, order, box)
    dense = float(np.dot(va, vb))
    fast = FK.corr_inner_product(win_a, win_b, order)
    assert abs(dense - fast) < 1e-8 * max(1.0, abs(dense))


def test_streamed_target_overlaps_match_materialized(chain):
    """The exact target stream must reproduce dense target contractions
    while holding only a bounded tuple batch."""
    wins = [FK.minimal_image_window(chain, SC, 0, far=2),
            FK.gaussian_window(chain, SC, chain.coords[0], 3.0, far=2)]
    target = FK.build_target_tuples(chain, SC, 3, far=2)
    dense = np.array([
        np.dot(target[2], FK.eval_corr_at_tuples(
            win, target[0], target[1], chain.N_atoms)) for win in wins
    ])
    stream = FK.GeometryTargetStream(chain, SC, 3, far=2,
                                     batch_tuples=7)
    streamed, tt, info = FK.stream_target_overlaps(
        wins, stream, chain.N_atoms)
    assert np.allclose(streamed, dense, rtol=1e-12, atol=1e-12)
    assert tt == pytest.approx(float(np.dot(target[2], target[2])))
    assert info["classes_visited"] == stream.total_classes
    assert info["tuples_visited"] == len(target[2])


def test_implicit_plain_algebra_matches_materialized(chain):
    plain = FK.plain_window(chain.N_atoms, SC)
    local = FK.local_gaussian_window(chain, chain.coords[0], 1.5)
    target = FK.build_target_tuples(chain, SC, 3, far=1)
    explicit_values = FK.eval_corr_at_tuples(
        plain, target[0], target[1], chain.N_atoms)
    implicit_values = FK.plain_corr_at_tuples(target[1], SC)
    assert np.array_equal(explicit_values, implicit_values)
    assert FK.plain_power_norm(chain.N_atoms, SC, 3) == pytest.approx(
        FK.corr_inner_product(plain, plain, 3))
    cross, info = FK.sparse_plain_inner_product(
        local, SC, 3, exact_limit=1000000)
    assert not info["sampled"]
    assert cross == pytest.approx(FK.corr_inner_product(local, plain, 3),
                                  rel=1e-12, abs=1e-12)


def test_constraint_gram_recognizes_termwise_manifold(chain):
    """Uniform type-0/type-1 factors must lie in the analytic joint
    partition+ASR nullspace without constructing constraint rows."""
    dictionary = FK.default_dictionary(chain, SC, far=2)[:8]
    plain = FK.plain_window(chain.N_atoms, SC)
    refs = np.array([0.0 if win.class_sum == 0.0 else 1.0
                     for win in dictionary])
    gram, partition, asr = FK.constraint_gram(
        dictionary, plain, SC, chain.N_atoms, 3,
        reference_weights=refs)
    assert np.max(np.abs(gram)) < 1e-9
    assert np.max(np.abs(partition)) < 1e-7
    assert np.max(np.abs(asr)) < 1e-7


def test_constraint_gram_detects_general_local_factor(chain):
    local = FK.local_gaussian_window(chain, chain.coords[0], 1.5)
    plain = FK.plain_window(chain.N_atoms, SC)
    gram, partition, asr = FK.constraint_gram(
        [local], plain, SC, chain.N_atoms, 3)
    assert gram[0, 0] > 0
    assert partition[0, 0] > 0
    assert asr[0, 0] > 0


def _brute_asr_residual_inner(win_a, win_b, nat, supercell, order):
    """One-leg ASR residual inner product for order three, used only to
    validate the analytic correlation formula."""
    assert order == 3
    Lvec = np.asarray(supercell, dtype=int)
    cells_a = win_a.arrays()[1]
    cells_b = win_b.arrays()[1]
    lo = np.minimum(cells_a.min(axis=0), cells_b.min(axis=0))
    hi = np.maximum(cells_a.max(axis=0), cells_b.max(axis=0))
    deltas = list(itertools.product(
        range(lo[0] - hi[0], hi[0] - lo[0] + 1),
        range(lo[1] - hi[1], hi[1] - lo[1] + 1),
        range(lo[2] - hi[2], hi[2] - lo[2] + 1)))

    def values(window):
        folded = window.folded_sums(supercell, nat)
        atoms, cells, _ = window.arrays()
        cells = np.unique(cells, axis=0)
        dense, origin = FK._window_dense(window, nat)
        dims = np.asarray(dense.shape[1:])
        rows = []
        for a1 in range(nat):
            for a2 in range(nat):
                for delta in deltas:
                    h = np.zeros((nat,) + tuple(Lvec))
                    for u in cells:
                        i1 = u - origin
                        i2 = u + np.asarray(delta) - origin
                        if np.any(i2 < 0) or np.any(i2 >= dims):
                            continue
                        prefactor = (dense[(a1,) + tuple(i1)]
                                     * dense[(a2,) + tuple(i2)])
                        if prefactor == 0:
                            continue
                        for a3 in range(nat):
                            for d3 in itertools.product(
                                    range(Lvec[0]), range(Lvec[1]),
                                    range(Lvec[2])):
                                residue = tuple((u + d3) % Lvec)
                                h[(a3,) + d3] += \
                                    prefactor * folded[(a3,) + residue]
                    rows.append((h - np.mean(h)).ravel())
        return np.concatenate(rows)

    return float(order) * np.dot(values(win_a), values(win_b))


def test_asr_constraint_gram_matches_brute_force(chain):
    win_a = FK.local_gaussian_window(chain, chain.coords[0], 1.5)
    win_b = FK.local_gaussian_window(chain, chain.coords[1], 2.0)
    plain = FK.plain_window(chain.N_atoms, SC)
    _, _, asr = FK.constraint_gram(
        [win_a, win_b], plain, SC, chain.N_atoms, 3)
    brute = _brute_asr_residual_inner(
        win_a, win_b, chain.N_atoms, SC, 3)
    assert asr[0, 1] == pytest.approx(brute, rel=1e-10, abs=1e-10)


def test_constrained_stream_fit_matches_feasible_dictionary(chain):
    """The full C_n U nullspace path must retain exact feasibility and
    avoid materializing the geometry target."""
    dictionary = [win for win in FK.default_dictionary(chain, SC, far=2)
                  if win.class_sum == 1.0][:6]
    fit = FK.fit_constrained_factors(
        chain, SC, 3, far=2, dictionary=dictionary, max_rank=4,
        target_mode="exact", validation_classes=0)
    assert len(fit.windows) <= 4
    assert fit.diagnostics["constraint_residual"] < 1e-8
    assert fit.diagnostics["target_mode"] == "exact"
    assert fit.diagnostics["train_stream"]["classes_visited"] == \
        chain.N_atoms * (chain.N_atoms * np.prod(SC)) ** 2


def test_large_supercell_stream_and_sparse_dictionary_are_bounded(chain):
    """Production setup must depend on the sample and local support, not on
    the number of folded rank-n tuples."""
    huge = (1, 1, 100000)
    stream = FK.GeometryTargetStream(
        chain, huge, 4, far=1, sample_classes=5, seed=11,
        batch_tuples=2, max_combinations=20000)
    first = next(stream.iter_batches())
    assert stream.total_classes == \
        chain.N_atoms * (chain.N_atoms * np.prod(huge)) ** 3
    assert first[0].shape[1] == 4
    assert len(first[2]) < 100

    small = FK.sparse_feasible_dictionary(
        chain, (1, 1, 10), sigmas=[1.0], bond_centers=False,
        quotient_shifts=[(0, 0, 1)])
    large = FK.sparse_feasible_dictionary(
        chain, huge, sigmas=[1.0], bond_centers=False,
        quotient_shifts=[(0, 0, 1)])
    assert [len(win.entries) for win in small] == \
        [len(win.entries) for win in large]
    for win in large:
        assert win.check_class_sums(huge, nat=chain.N_atoms) < 1e-12
    fit = FK.fit_constrained_factors(
        chain, huge, 3, far=1, dictionary=large, max_rank=2,
        target_mode="sample", sample_classes=5, validation_classes=0,
        plain_shift_samples=32, plain_exact_limit=0,
        max_combinations=20000)
    assert len(fit.windows) <= 2
    assert fit.diagnostics["train_stream"]["total_classes"] == \
        chain.N_atoms * (chain.N_atoms * np.prod(huge)) ** 2
    assert all(item["sampled"]
               for item in fit.diagnostics["plain_overlap_sampling"])


def test_factor_fit_disk_roundtrip(chain, tmp_path):
    dictionary = FK.sparse_feasible_dictionary(
        chain, SC, sigmas=[1.0], bond_centers=False,
        quotient_shifts=[(0, 0, 1)])
    fit = FK.fit_constrained_factors(
        chain, SC, 3, far=1, dictionary=dictionary, max_rank=2,
        target_mode="sample", sample_classes=12, validation_classes=4)
    path = tmp_path / "fit.npz"
    fit.save(path)
    restored = FK.FactorFit.load(path)
    assert restored.order == fit.order
    assert restored.plain_coeff == pytest.approx(fit.plain_coeff)
    assert np.allclose(restored.coeffs, fit.coeffs)
    assert restored.diagnostics["fit_mode"] == "joint-constrained"
    assert [window.entries for window in restored.windows] == \
        [window.entries for window in fit.windows]


def test_outer_factor_optimization_rebuilds_fit(chain):
    fit = FK.optimize_constrained_factors(
        chain, SC, 3, sweeps=1, scale_trials=(0.8, 1.2),
        base_sigmas=[1.0], quotient_shifts=[(0, 0, 1)],
        max_rank=2, far=1, target_mode="sample", sample_classes=12,
        validation_classes=4)
    history = fit.diagnostics["factor_optimization"]["history"]
    assert len(history) == 3
    assert fit.diagnostics["factor_optimization"]["selected_scale"] in \
        (0.8, 1.0, 1.2)
    assert fit.diagnostics["constraint_residual"] < 1e-12


# =====================================================================
# 4-5. runtime: commensurate identity + Hermiticity
# =====================================================================

@pytest.fixture(scope="module")
def ensemble():
    dyn = TC.build_dyn(L)
    return TC.make_ensemble(dyn, 300.0, 200, seed=7, g3=0.2, g4=0.3)


def _run(lanc, iq, band, nstep=6):
    lanc.init(use_symmetries=True)
    lanc.prepare_mode_q(iq, band)
    lanc.run_FT(nstep, verbose=False)
    return np.array(lanc.a_coeffs), np.array(lanc.b_coeffs)


@pytest.mark.skipif(not _HAS_Q, reason="Julia ext not available")
def test_factor_commensurate_identity(ensemble):
    """fine mesh == coarse mesh: the factor-mode operator must equal the
    plain coarse Lanczos PER CONFIGURATION (the type-1 collapse cancels
    every correction pass exactly at commensurate q).  This is the
    runtime realization of the partition-of-unity gate."""
    lc = QL.QSpaceLanczos(ensemble, lo_to_split=None)
    li = QI.QSpaceLanczosInterp(ensemble, fine_mesh=(1, 1, L),
                                window_design="factor", factor_rank=6)
    assert len(li._factor_passes[3]) > 0, \
        "no correction retained: the identity test would be trivial"
    for iq_c, band in [(0, 5), (1, 2)]:
        iq_f = li.find_fine_q(lc.q_points[iq_c])
        a_c, b_c = _run(lc, iq_c, band)
        a_i, b_i = _run(li, iq_f, band)
        n = min(len(a_c), len(a_i))
        assert np.max(np.abs(a_c[:n] - a_i[:n]) / np.abs(a_c[:n])) < 1e-10
        m = min(len(b_c), len(b_i))
        assert np.max(np.abs(b_c[:m] - b_i[:m]) / np.abs(b_c[:m])) < 1e-10


@pytest.mark.skipif(not _HAS_Q, reason="Julia ext not available")
def test_constrained_stream_runtime_identity(ensemble):
    """The production streaming fitter must feed its sparse factors through
    the existing runtime while retaining exact coarse-grid identity."""
    lc = QL.QSpaceLanczos(ensemble, lo_to_split=None)
    li = QI.QSpaceLanczosInterp(
        ensemble, fine_mesh=(1, 1, L), window_design="factor",
        factor_fit_mode="constrained", factor_target_mode="sample",
        factor_sample_classes=30, factor_validation_classes=12,
        factor_rank=4)
    assert li._factor_passes[3]
    for _, field_index in li._factor_passes[3]:
        assert field_index >= 0
    iq_c, band = 1, 3
    iq_f = li.find_fine_q(lc.q_points[iq_c])
    a_c, b_c = _run(lc, iq_c, band, nstep=3)
    a_i, b_i = _run(li, iq_f, band, nstep=3)
    assert np.allclose(a_c, a_i, rtol=1e-10, atol=1e-12)
    assert np.allclose(b_c, b_i, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_Q, reason="Julia ext not available")
def test_factor_hermiticity_off_grid(ensemble):
    li = QI.QSpaceLanczosInterp(ensemble, fine_mesh=(1, 1, 2 * L),
                                window_design="factor", factor_rank=6)
    li.init(use_symmetries=True)
    iq = int(np.where((li._fine_idx == [0, 0, 1]).all(axis=1))[0][0])
    li.prepare_mode_q(iq, 4)
    li.run_FT(6, verbose=False)
    a = np.array(li.a_coeffs)
    b = np.array(li.b_coeffs)
    c = np.array(li.c_coeffs)
    assert np.all(np.abs(np.imag(a)) < 1e-12)
    n = min(len(b), len(c))
    assert np.max(np.abs(b[:n] - c[:n]) / np.abs(b[:n])) < 1e-8


@pytest.mark.skipif(not _HAS_Q, reason="Julia ext not available")
def test_factor_d4_mixed_mode_identity(ensemble):
    """window_design='atomic_delta' + d4_center='factor' on the
    commensurate mesh must reproduce the same atomic-D3 operator with a
    plain D4 pass. Comparing identical interpolation paths isolates D4
    from the atomic-delta field projector."""
    iq_c, band = 1, 3
    results = {}
    for d4_center in (False, "factor"):
        li = QI.QSpaceLanczosInterp(
            ensemble, fine_mesh=(1, 1, L),
            window_design="atomic_delta", d4_center=d4_center,
            factor_rank=6)
        iq_f = li.find_fine_q(li.dyn.q_tot[iq_c])
        results[d4_center] = _run(li, iq_f, band)
    for left, right in zip(results[False], results["factor"]):
        n = min(len(left), len(right))
        scale = np.maximum(np.abs(left[:n]), 1e-14)
        assert np.max(np.abs(left[:n] - right[:n]) / scale) < 1e-8
