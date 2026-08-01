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
