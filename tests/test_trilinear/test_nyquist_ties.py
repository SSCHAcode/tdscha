"""Nyquist-tie handling of the atom-centred Fourier kernel.

The minimal image of an aliasing class and its ties must be resolved with
the TRUE 3D cell metric, not separably per Cartesian axis.  On an
orthorhombic cell the two rules agree; on a non-orthogonal cell the
separable rule can miss genuine >2-fold ties, invent spurious ties, or pick
the wrong image, and a coupling pinned to such a class is then mis-continued
off the coarse grid.

These tests cover:

1. ``_metric_alias_images`` vs a brute-force 3D minimum image, including a
   genuine four-fold tie a separable product cannot represent;
2. equivalence of the metric and separable kernels on a cubic cell;
3. on a non-orthogonal (bcc-primitive) cell: the metric kernel reconstructs
   a coupling pinned to a mis-assigned class that the separable kernel gets
   wrong by 50%;
4. the metric kernel keeps the commensurate identity and the mirror
   identity P_ab(q,k) = P_ba(-q,-k) that guarantee Hermiticity.
"""
import itertools
import os
import sys

os.environ.setdefault("JULIA_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import pytest

import _toy_crystal3d as TC3

try:
    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceTrilinear as QT
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

pytestmark = pytest.mark.skipif(not _HAS_Q,
                                reason="QSpaceLanczos/Julia not available")

NCART = 3


# =====================================================================
# 1. Geometry: metric minimum image vs brute force
# =====================================================================
def _brute_min_images(d, Nc, A, tol=1e-7):
    d = np.asarray(d, float)
    Nc = np.asarray(Nc, int)
    out = {}
    span = range(-3, 4)
    for k in itertools.product(*[range(int(n)) for n in Nc]):
        best, best_d2 = [], np.inf
        for m in itertools.product(span, span, span):
            R = np.array([k[i] + int(Nc[i]) * m[i] for i in range(3)], float)
            cart = (R - d) @ A
            d2 = float(cart @ cart)
            if d2 < best_d2 - tol:
                best_d2, best = d2, [tuple(int(x) for x in R)]
            elif abs(d2 - best_d2) < tol:
                best.append(tuple(int(x) for x in R))
        out[k] = {R: 1.0 / len(best) for R in best}
    return out


@pytest.mark.parametrize("d,expect_max_mult", [
    ((0.5, 0.5, 0.0), 4),      # genuine four-fold non-separable tie
    ((0.5, 0.5, 0.5), 1),      # user's bcc central-atom case: unique
    ((0.25, 0.1, 0.6), 1),
])
def test_metric_alias_images_matches_bruteforce(d, expect_max_mult):
    a = 3.0
    A = a * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
    metric = A @ A.T
    Nc = [2, 2, 2]
    got = QT.QSpaceTrilinearLanczos._metric_alias_images(
        np.array(d), Nc, metric)
    ref = _brute_min_images(d, Nc, A)
    max_mult = max(len(v) for v in got.values())
    assert max_mult == expect_max_mult
    for k in ref:
        g = {R: w for R, w in got[k]}
        assert set(g) == set(ref[k])
        for R in g:
            assert abs(g[R] - ref[k][R]) < 1e-9


def test_metric_alias_images_cubic_equals_separable():
    """On a cubic (box) cell the metric assignment reduces to the separable
    per-axis product, including the eight-fold WS-corner tie."""
    A = np.eye(3)
    metric = np.eye(3)
    Nc = [2, 2, 2]
    cls = QT.QSpaceTrilinearLanczos
    for d in [(0.5, 0.5, 0.5), (0.5, 0.0, 0.0), (0.25, 0.5, 0.75)]:
        met = cls._metric_alias_images(np.array(d), Nc, metric)
        # rebuild the separable assignment
        sep = {}
        per = []
        for ax in range(3):
            pa = {}
            for alias in range(Nc[ax]):
                pa[alias] = cls._nearest_alias_images(alias, d[ax], Nc[ax])
            per.append(pa)
        for k in itertools.product(*[range(n) for n in Nc]):
            entries = {}
            for rx, wx in per[0][k[0]]:
                for ry, wy in per[1][k[1]]:
                    for rz, wz in per[2][k[2]]:
                        entries[(int(rx), int(ry), int(rz))] = wx * wy * wz
            sep[k] = entries
        for k in met:
            g = {R: w for R, w in met[k]}
            assert set(g) == set(sep[k])
            for R in g:
                assert abs(g[R] - sep[k][R]) < 1e-9


# =====================================================================
# 2/3. Kernel reconstruction on a real (non-orthogonal) class instance
# =====================================================================
@pytest.fixture(scope="module")
def system3d():
    dyn = TC3.build_dyn((2, 2, 2))
    ens = TC3.make_ensemble(dyn, N=24, seed=3)
    return dyn, ens


def _pair_kernel(li, iq, ia, ib):
    """P_ab(q_fine=iq, k_coarse) as a length-cn_q complex vector."""
    return li._atom_fourier_kernel[iq, :, ia, ib]


def _reconstruct(li, coupling, ia, ib):
    """Reconstruct V(q) = sum amp e^{2 pi i q.R} from its coarse samples
    with the (ia,ib) pair kernel; return relative error^2 on the fine mesh.

    coupling: list of (R (3-int), amplitude)."""
    Nc = li.coarse_mesh
    LF = li.fine_mesh
    coarse_frac = np.asarray(li._coarse_idx, float) / Nc
    fine_frac = np.asarray(li._fine_idx, float) / LF

    def Vtrue(q):
        return sum(amp * np.exp(2j * np.pi * np.dot(q, R))
                   for R, amp in coupling)

    Vc = np.array([Vtrue(x) for x in coarse_frac])
    num = den = 0.0
    for iq, q in enumerate(fine_frac):
        vt = Vtrue(q)
        vi = np.dot(_pair_kernel(li, iq, ia, ib), Vc)
        num += abs(vi - vt) ** 2
        den += abs(vt) ** 2
    return num / den


def test_metric_kernel_fixes_nonorthogonal_reconstruction(system3d):
    """On the bcc-primitive cell, a coupling pinned to the class the
    separable rule mis-assigns is reconstructed exactly by the metric
    kernel and wrongly (about 50%) by the separable one."""
    _, ens = system3d
    li_m = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(4, 4, 4),
                                     atom_fourier=True, tie_metric=True,
                                     allow_unstable=True)
    li_s = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(4, 4, 4),
                                     atom_fourier=True, tie_metric=False,
                                     allow_unstable=True)

    # Pair (0,1): d = tau0 - tau1 = -(0.5,0.5,0.0).  Class (0,0,1) is the
    # metric/separable disagreement; the physically shortest image there is
    # unique for the metric but a spurious (0,0,+-1) tie for separable.
    A = np.asarray(li_m.uci_structure.unit_cell, float)
    metric = A @ A.T
    tau = np.linalg.solve(A.T, li_m.uci_structure.coords.T).T
    d = tau[0] - tau[1]
    imgs = QT.QSpaceTrilinearLanczos._metric_alias_images(
        d, li_m.coarse_mesh, metric)
    class_key = (0, 0, 1)
    minset = imgs[class_key]
    assert len(minset) == 1, "expected a unique metric image for this class"
    coupling = [(minset[0][0], 1.0)]

    err_m = _reconstruct(li_m, coupling, 0, 1)
    err_s = _reconstruct(li_s, coupling, 0, 1)
    assert err_m < 1e-20, "metric kernel not exact: {}".format(err_m)
    assert err_s > 0.1, "separable kernel unexpectedly fine: {}".format(err_s)


def test_metric_kernel_commensurate_identity(system3d):
    """The metric kernel is exactly cardinal on the coarse mesh."""
    _, ens = system3d
    li = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(4, 4, 4),
                                   atom_fourier=True, allow_unstable=True)
    nat = len(li.uci_structure.coords)
    for jq in range(li.cn_q):
        iq = li._fine_of_coarse[jq]
        for ik in range(li.cn_q):
            expected = 1.0 if ik == jq else 0.0
            for ia in range(nat):
                for ib in range(nat):
                    assert abs(li._atom_fourier_kernel[iq, ik, ia, ib]
                               - expected) < 1e-12


def test_metric_kernel_mirror_identity(system3d):
    """P_ab(q,k) = P_ba(-q,-k): the leg-exchange symmetry that keeps the
    folded operator Hermitian, preserved by the metric ties."""
    _, ens = system3d
    li = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(4, 4, 4),
                                   atom_fourier=True, allow_unstable=True)
    K = li._atom_fourier_kernel

    def neg_fine(iq):
        n = (-li._fine_idx[iq]) % li.fine_mesh
        return li._q_lookup[tuple(n)]

    def neg_coarse(ik):
        n = (-np.array(li._coarse_idx[ik])) % li.coarse_mesh
        return li._coarse_lookup[tuple(n)]

    for iq in range(li.n_q):
        jq = neg_fine(iq)
        for ik in range(li.cn_q):
            jk = neg_coarse(ik)
            assert np.allclose(K[iq, ik], K[jq, jk].T, atol=1e-12)


def test_metric_kernel_hermiticity_operator(system3d):
    """The full atom-Fourier L with metric ties is Hermitian under the
    masked inner product on the non-orthogonal cell."""
    _, ens = system3d
    li = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(4, 4, 4),
                                   atom_fourier=True, allow_unstable=True)
    li.init(use_symmetries=True)
    # Q = Gamma (coarse); build the pair map and probe Hermiticity.
    iq0 = int(np.where((li._fine_idx == [0, 0, 0]).all(axis=1))[0][0])
    li.build_q_pair_map(iq0)
    li.reset_q()

    rng = np.random.default_rng(7)
    n = li.get_psi_size()
    mask = li.mask_dot_wigner()
    phi = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    d1 = np.conj(phi).dot(li.apply_full_L(psi.copy()) * mask)
    d2 = np.conj(psi).dot(li.apply_full_L(phi.copy()) * mask)
    assert abs(d1 - np.conj(d2)) / max(abs(d1), abs(d2), 1e-300) < 1e-9
