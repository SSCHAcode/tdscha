"""Unit tests for the adaptive Schur fill of QSpaceHessian.

By Schur's lemma the cross block of G between two copies of the SAME
irrep is c*U with a unitary intertwiner U that is nontrivial whenever
eigh picks arbitrary bases in the two degenerate subspaces: the scalar
c*I shortcut is exact only for distinct irreps (zero coupling). These
tests exercise _adaptive_schur_fill directly with an exact solver, so
they need no Julia extension and no ensemble data.
"""
import numpy as np
import pytest

try:
    from tdscha.QSpaceHessian import _adaptive_schur_fill
except ModuleNotFoundError:
    # Only a genuinely missing dependency is a legitimate skip. A bare
    # ImportError would also swallow "cannot import name _adaptive_schur_fill",
    # i.e. exactly the regression these tests exist to catch: the suite would
    # then report green while never running a single assertion.
    pytest.skip("tdscha.QSpaceHessian not importable", allow_module_level=True)


def _exact_solver(G_true):
    """solve_column callable backed by L = G_true^-1 (exact columns)."""
    L = np.linalg.inv(G_true)

    def solve_column(i):
        e = np.zeros(G_true.shape[0], dtype=np.complex128)
        e[i] = 1.0
        return np.linalg.solve(L, e), 0, 0.0

    return solve_column


def _two_triplets(c12):
    """G_true with two 3-dim blocks (a*I, b*I) and cross c12*U, U random."""
    rng = np.random.default_rng(42)
    d, nb = 3, 6
    U = np.linalg.qr(rng.normal(size=(d, d))
                     + 1j * rng.normal(size=(d, d)))[0]
    G = np.zeros((nb, nb), dtype=np.complex128)
    G[:d, :d] = 2.3 * np.eye(d)
    G[d:, d:] = 4.1 * np.eye(d)
    G[:d, d:] = c12 * U
    G[d:, :d] = np.conj(c12) * U.conj().T
    return G, U


def test_repeated_irrep_intertwiner_exact():
    """Two copies of the same irrep with a nontrivial intertwiner: the
    coupling must be detected and the fill must reproduce G exactly
    (the old c*I shortcut was wrong by ~7% on the eigenvalues here)."""
    c12 = 0.7 * np.exp(0.6j)
    G_true, U = _two_triplets(c12)
    # Sanity: the intertwiner really is nontrivial, so c*I would be wrong
    assert np.max(np.abs(c12 * U - c12 * np.eye(3))) > 0.1

    solve = _exact_solver(G_true)
    schedule = [(0, [0, 1, 2]), (3, [3, 4, 5])]
    rep_x = {b: solve(b)[0] for b, _ in schedule}

    G = np.zeros_like(G_true)
    full = _adaptive_schur_fill(G, schedule, rep_x, solve, 6, 1e-6, True)
    assert full == {0, 3}, "repeated-irrep coupling not detected"

    G = (G + G.conj().T) / 2
    assert np.max(np.abs(G - G_true)) < 1e-12

    ev = np.linalg.eigvalsh(np.linalg.inv(G))
    ev_true = np.linalg.eigvalsh(np.linalg.inv(G_true))
    assert np.max(np.abs(ev - ev_true) / np.abs(ev_true)) < 1e-12


def test_distinct_irreps_keep_scalar_path():
    """Zero coupling (distinct irreps): no full solve, and the scalar
    Schur fill is exact."""
    G_true, _ = _two_triplets(0.0)
    solve = _exact_solver(G_true)
    schedule = [(0, [0, 1, 2]), (3, [3, 4, 5])]
    rep_x = {b: solve(b)[0] for b, _ in schedule}

    G = np.zeros_like(G_true)
    full = _adaptive_schur_fill(G, schedule, rep_x, solve, 6, 1e-6, True)
    assert full == set(), "spurious coupling detected on distinct irreps"

    G = (G + G.conj().T) / 2
    assert np.max(np.abs(G - G_true)) < 1e-12


def test_singletons_and_no_mode_symmetry():
    """All-singleton schedules (use_mode_symmetry False or no degeneracy)
    must reproduce the full columns verbatim."""
    rng = np.random.default_rng(7)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    G_true = A @ A.conj().T + 5 * np.eye(4)
    solve = _exact_solver(G_true)
    schedule = [(i, [i]) for i in range(4)]
    rep_x = {b: solve(b)[0] for b, _ in schedule}

    for ums in (True, False):
        G = np.zeros_like(G_true)
        full = _adaptive_schur_fill(G, schedule, rep_x, solve, 4, 1e-6, ums)
        assert full == set()
        assert np.max(np.abs(G - G_true)) < 1e-12


def _block_diag_irrep(consts, dim, mixer=None):
    """G with `len(consts)` copies of one dim-dimensional irrep.

    Each copy m contributes consts[m] * I_dim on its diagonal block; `mixer`,
    if given, is the (ncopies, ncopies) hermitian matrix of cross-block
    constants, so that the block (m, n) is mixer[m, n] * I_dim. This is the
    exact structure Schur's lemma allows in a symmetry-adapted basis.
    """
    n = len(consts) * dim
    G = np.zeros((n, n), dtype=np.complex128)
    for m, cm in enumerate(consts):
        G[m * dim:(m + 1) * dim, m * dim:(m + 1) * dim] = cm * np.eye(dim)
    if mixer is not None:
        for m in range(len(consts)):
            for n_ in range(len(consts)):
                if m != n_:
                    G[m * dim:(m + 1) * dim, n_ * dim:(n_ + 1) * dim] = \
                        mixer[m, n_] * np.eye(dim)
    return G


def _rotate(G, rng, blocks):
    """Rotate each degenerate block by a random unitary.

    This is what eigh does in practice: inside a degenerate subspace the basis
    is arbitrary, which is exactly why the cross block is c*U and not c*I.
    """
    n = G.shape[0]
    U = np.eye(n, dtype=np.complex128)
    for b in blocks:
        k = len(b)
        M = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
        Q, _ = np.linalg.qr(M)
        U[np.ix_(b, b)] = Q
    return U.conj().T @ G @ U


def test_single_reducible_block_is_detected():
    """A lone degenerate block can already be reducible.

    Two copies of the same irrep degenerate at the SAME frequency land in one
    block, so there is no partner block to reveal the coupling. The scalar
    shortcut is wrong for it, and the leakage of the representative column
    onto the rest of its own block is what exposes it.
    """
    rng = np.random.default_rng(11)
    block = list(range(4))
    G_true = _block_diag_irrep([2.0, 3.5], dim=2,
                               mixer=np.array([[0.0, 0.9], [0.9, 0.0]]))
    G_true = _rotate(G_true, rng, [block])
    solve = _exact_solver(G_true)
    schedule = [(0, block)]
    rep_x = {0: solve(0)[0]}

    G = np.zeros_like(G_true)
    full = _adaptive_schur_fill(G, schedule, rep_x, solve, 4, 1e-6, True)
    assert full == {0}, "reducible single block not detected"
    G = (G + G.conj().T) / 2
    assert np.max(np.abs(G - G_true)) < 1e-10


def test_soft_mode_spectator_does_not_raise_the_threshold():
    """A large-norm spectator block must not hide a later coupling.

    A block with a large column norm (a soft mode: the columns of G go as
    1/w^2) must not raise the detection threshold for the pairs examined
    after it. Carrying a running maximum of `scale` across the pair loop
    makes the threshold monotonically non-decreasing, so such a spectator
    sitting between two coupled blocks masks their coupling entirely. The
    spectator is therefore at index 1, with the coupled copies at 0 and 2,
    and its constant is large so that its column norm dominates.
    """
    rng = np.random.default_rng(3)
    n = 6
    G_true = np.zeros((n, n), dtype=np.complex128)
    # two copies of a 2-dim irrep, coupled, at indices 0-1 and 4-5
    G_true[0:2, 0:2] = 1.0 * np.eye(2)
    G_true[4:6, 4:6] = 1.2 * np.eye(2)
    G_true[0:2, 4:6] = 3e-4 * np.eye(2)
    G_true[4:6, 0:2] = 3e-4 * np.eye(2)
    # spectator in between, with a column norm ~1e3 times the coupled pair
    # and no coupling of its own
    G_true[2:4, 2:4] = 1e3 * np.eye(2)
    blocks = [[0, 1], [2, 3], [4, 5]]
    G_true = _rotate(G_true, rng, blocks)
    G_true = (G_true + G_true.conj().T) / 2
    solve = _exact_solver(G_true)
    schedule = [(b[0], b) for b in blocks]
    rep_x = {b[0]: solve(b[0])[0] for b in blocks}

    G = np.zeros_like(G_true)
    full = _adaptive_schur_fill(G, schedule, rep_x, solve, n, 1e-6, True)
    assert full == {0, 4}, "coupling masked by the soft-mode spectator"
    G = (G + G.conj().T) / 2
    assert np.max(np.abs(G - G_true)) < 1e-10


def test_detection_is_order_independent():
    """The result must not depend on the order the blocks are listed in.

    The dimension shortcut reads the set of self-reducible blocks, so that set
    has to be complete before any pair is examined; deciding both in one pass
    makes the outcome depend on the iteration order.
    """
    rng = np.random.default_rng(5)
    # a self-reducible 4-dim block coupled to a 2-dim block: the pair has
    # different dimensions, so it is only examined because one is reducible
    G_true = np.zeros((6, 6), dtype=np.complex128)
    G_true[:4, :4] = _block_diag_irrep([2.0, 2.6], dim=2,
                                       mixer=np.array([[0.0, 0.8],
                                                       [0.8, 0.0]]))
    G_true[4:, 4:] = 1.5 * np.eye(2)
    G_true[:2, 4:] = 0.4 * np.eye(2)
    G_true[4:, :2] = 0.4 * np.eye(2)
    blocks_a = [[0, 1, 2, 3], [4, 5]]
    G_true = _rotate(G_true, rng, blocks_a)
    G_true = (G_true + G_true.conj().T) / 2
    solve = _exact_solver(G_true)

    results = []
    for blocks in (blocks_a, list(reversed(blocks_a))):
        schedule = [(b[0], b) for b in blocks]
        rep_x = {b[0]: solve(b[0])[0] for b in blocks}
        G = np.zeros_like(G_true)
        full = _adaptive_schur_fill(G, schedule, rep_x, solve, 6, 1e-6, True)
        G = (G + G.conj().T) / 2
        results.append((full, np.max(np.abs(G - G_true))))

    assert results[0][0] == results[1][0], "detection depends on block order"
    assert results[0][0] == {0, 4}, "expected both coupled blocks to be solved"
    for _, err in results:
        assert err < 1e-10
