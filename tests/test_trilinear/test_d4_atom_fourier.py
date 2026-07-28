"""Fourth-order (D4) interpolation in the atom-centred Fourier scheme.

The D4 term maps the incoming two-phonon block at the fine pair
``(q', Q-q')`` onto the outgoing block at ``(q, Q-q)``.  Because the
estimator contracts the incoming block with CONJUGATED Bloch fields, the
four vertex legs carry the momenta

    leg 1 = a at +q      leg 2 = b at -q
    leg 3 = c at -q'     leg 4 = d at +q'

so the two momenta are independent and each couples to exactly ONE pair
offset.  The correct four-leg continuation is therefore the tensor product
of the pairwise atom-centred kernel with itself -- ``P_ab(q,k)`` on the
unfold and ``conj(P_cd(q',k'))`` on the fold, which is what the production
fold/unfold already applies.  These tests verify that claim:

1. exact reconstruction of the closed-form quartic vertex of the chain
   (machine zero for atom_fourier, finite error for plain trilinear);
2. leg-exchange permutation symmetry, via the kernel mirror identity
   P_ab(q,k) = P_ba(-q,-k), and at the operator level;
3. acoustic sum rule: preserved exactly on all four legs, because the
   kernel is cardinal at Gamma AND independent of the atoms of the
   non-interpolated legs.  Deliberately broken control maps show the ASR
   tests have teeth;
4. the four-leg range limit: aliased quartic harmonics are NOT recovered.
"""
import os
import sys

os.environ.setdefault("JULIA_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "test_interpolation"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "report",
                                "interpolation", "scripts"))

import itertools

import numpy as np
import pytest

import _toy_chain as TC

try:
    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceTrilinear as QT
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

pytestmark = pytest.mark.skipif(not _HAS_Q,
                                reason="QSpaceLanczos/Julia not available")

T = 300.0
N_CONF = 60
L = 3
LF = 9
G3 = 0.2
G4 = 0.3

NCART = 3
NAT = 2
NDIM = NAT * NCART


# =====================================================================
# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def system():
    dyn = TC.build_dyn(L)
    ens = TC.make_ensemble(dyn, T, N_CONF, seed=17, g3=G3, g4=G4)
    return dyn, ens


@pytest.fixture(scope="module")
def li_af(system):
    _, ens = system
    return QT.QSpaceTrilinearLanczos(ens, fine_mesh=(1, 1, LF),
                                     atom_fourier=True)


@pytest.fixture(scope="module")
def li_tri(system):
    _, ens = system
    return QT.QSpaceTrilinearLanczos(ens, fine_mesh=(1, 1, LF))


def _fine_index(li, nz):
    return int(np.where((li._fine_idx == [0, 0, nz]).all(axis=1))[0][0])


# =====================================================================
# 1. Exact reconstruction of the closed-form quartic vertex
# =====================================================================
def _dense_trilinear(li):
    W = np.zeros((li.n_q, li.cn_q), dtype=np.float64)
    for iq, entries in enumerate(li._corners):
        for ik, w, _delta in entries:
            W[iq, ik] += w
    return W


def _interp_vertex(V_coarse, Pq, Pqp, cn_q):
    """V_int(q,q') = sum_{k,k'} P_ab(q,k) V(k,k') conj(P_cd(q',k'))."""
    out = np.zeros((NDIM,) * 4, dtype=np.complex128)
    for k in range(cn_q):
        for kp in range(cn_q):
            out += (Pq[k][:, :, None, None]
                    * np.conj(Pqp[kp])[None, None, :, :]
                    * V_coarse[k, kp])
    return out


def test_d4_vertex_reconstruction(li_af, li_tri):
    """The atom-centred Fourier kernel reconstructs the exact four-leg
    quartic vertex of the chain; plain trilinear does not."""
    import d4_atom_fourier_oracle as ORC

    coarse_frac = np.array([c[2] / float(L) for c in li_af._coarse_idx])
    fine_frac = np.array([n[2] / float(LF) for n in li_af._fine_idx])
    cn = li_af.cn_q

    V_coarse = np.empty((cn, cn) + (NDIM,) * 4, dtype=np.complex128)
    for k in range(cn):
        for kp in range(cn):
            V_coarse[k, kp] = ORC.exact_vertex(coarse_frac[k],
                                               coarse_frac[kp], G4)

    W = _dense_trilinear(li_tri)

    def pairs_af(iq):
        return [np.repeat(np.repeat(li_af._atom_fourier_kernel[iq, k],
                                    NCART, axis=0), NCART, axis=1)
                for k in range(cn)]

    def pairs_tri(iq):
        return [np.full((NDIM, NDIM), W[iq, k], dtype=np.complex128)
                for k in range(cn)]

    err = {"af": [0.0, 0.0], "tri": [0.0, 0.0]}
    for iq in range(li_af.n_q):
        for iqp in range(li_af.n_q):
            exact = ORC.exact_vertex(fine_frac[iq], fine_frac[iqp], G4)
            den = np.sum(np.abs(exact) ** 2)
            for tag, builder in [("af", pairs_af), ("tri", pairs_tri)]:
                got = _interp_vertex(V_coarse, builder(iq), builder(iqp), cn)
                err[tag][0] += np.sum(np.abs(got - exact) ** 2)
                err[tag][1] += den

    rel_af = err["af"][0] / err["af"][1]
    rel_tri = err["tri"][0] / err["tri"][1]
    assert rel_af < 1e-24, "atom-Fourier D4 vertex not exact: {}".format(rel_af)
    # The test must be able to discriminate: plain trilinear is far off.
    assert rel_tri > 1e-3, "control (trilinear) unexpectedly exact"


def test_d4_range_limit_is_aliasing_not_a_defect(system):
    """A quartic harmonic longer than the coarse minimum-image window is
    genuinely aliased: the atom-centred continuation cannot recover it."""
    import d4_atom_fourier_oracle as ORC

    _, ens = system
    # L = 2 cannot represent the second-neighbour |R| = 2 quartic bond.
    dyn2 = TC.build_dyn(2)
    ens2 = TC.make_ensemble(dyn2, T, 20, seed=5, g3=G3, g4=G4)
    li2 = QT.QSpaceTrilinearLanczos(ens2, fine_mesh=(1, 1, 4),
                                    atom_fourier=True)
    cn = li2.cn_q
    coarse_frac = np.array([c[2] / 2.0 for c in li2._coarse_idx])
    fine_frac = np.array([n[2] / 4.0 for n in li2._fine_idx])

    V_coarse = np.empty((cn, cn) + (NDIM,) * 4, dtype=np.complex128)
    for k in range(cn):
        for kp in range(cn):
            V_coarse[k, kp] = ORC.exact_vertex(coarse_frac[k],
                                               coarse_frac[kp], G4,
                                               g4_long=0.15)

    def pairs(iq):
        return [np.repeat(np.repeat(li2._atom_fourier_kernel[iq, k],
                                    NCART, axis=0), NCART, axis=1)
                for k in range(cn)]

    num = den = 0.0
    for iq in range(li2.n_q):
        for iqp in range(li2.n_q):
            exact = ORC.exact_vertex(fine_frac[iq], fine_frac[iqp], G4,
                                     g4_long=0.15)
            got = _interp_vertex(V_coarse, pairs(iq), pairs(iqp), cn)
            num += np.sum(np.abs(got - exact) ** 2)
            den += np.sum(np.abs(exact) ** 2)
    assert num / den > 1e-2, "aliased range unexpectedly reconstructed"

    # ... while the same coupling on a mesh that resolves it is exact.
    dyn4 = TC.build_dyn(4)
    ens4 = TC.make_ensemble(dyn4, T, 20, seed=5, g3=G3, g4=G4)
    li4 = QT.QSpaceTrilinearLanczos(ens4, fine_mesh=(1, 1, 8),
                                    atom_fourier=True)
    cn4 = li4.cn_q
    cf = np.array([c[2] / 4.0 for c in li4._coarse_idx])
    ff = np.array([n[2] / 8.0 for n in li4._fine_idx])
    Vc = np.empty((cn4, cn4) + (NDIM,) * 4, dtype=np.complex128)
    for k in range(cn4):
        for kp in range(cn4):
            Vc[k, kp] = ORC.exact_vertex(cf[k], cf[kp], G4, g4_long=0.15)

    def pairs4(iq):
        return [np.repeat(np.repeat(li4._atom_fourier_kernel[iq, k],
                                    NCART, axis=0), NCART, axis=1)
                for k in range(cn4)]

    num = den = 0.0
    for iq in range(li4.n_q):
        for iqp in range(li4.n_q):
            exact = ORC.exact_vertex(ff[iq], ff[iqp], G4, g4_long=0.15)
            got = _interp_vertex(Vc, pairs4(iq), pairs4(iqp), cn4)
            num += np.sum(np.abs(got - exact) ** 2)
            den += np.sum(np.abs(exact) ** 2)
    assert num / den < 1e-24, "resolved range not exact: {}".format(num / den)


# =====================================================================
# 2. Leg-exchange permutation symmetry
# =====================================================================
def test_kernel_mirror_identity(li_af):
    """P_ab(q,k) = P_ba(-q,-k).

    This identity is exactly what makes the interpolated vertex invariant
    under exchanging the two legs of a pair (q1 <-> q2 = Q - q1), for D3
    and for both D4 pairs.  It is the atom-resolved generalization of the
    corner-mirror property of the trilinear weights.
    """
    li = li_af
    K = li._atom_fourier_kernel
    nq, cn = li.n_q, li.cn_q

    def neg_fine(iq):
        n = (-li._fine_idx[iq]) % li.fine_mesh
        return li._q_lookup[tuple(n)]

    def neg_coarse(ik):
        n = (-np.array(li._coarse_idx[ik])) % li.coarse_mesh
        return li._coarse_lookup[tuple(n)]

    for iq in range(nq):
        jq = neg_fine(iq)
        for ik in range(cn):
            jk = neg_coarse(ik)
            assert np.allclose(K[iq, ik], K[jq, jk].T, atol=1e-13), \
                "mirror identity broken at (iq={}, ik={})".format(iq, ik)


def test_d4_vertex_leg_permutations(li_af):
    """The interpolated four-leg vertex obeys the three independent leg
    exchanges of Phi4:  1<->2 (with q -> -q), 3<->4 (with q' -> -q'), and
    (1,2) <-> (3,4) (with q <-> q')."""
    import d4_atom_fourier_oracle as ORC

    li = li_af
    cn = li.cn_q
    coarse_frac = np.array([c[2] / float(L) for c in li._coarse_idx])
    fine_frac = np.array([n[2] / float(LF) for n in li._fine_idx])

    V_coarse = np.empty((cn, cn) + (NDIM,) * 4, dtype=np.complex128)
    for k in range(cn):
        for kp in range(cn):
            V_coarse[k, kp] = ORC.exact_vertex(coarse_frac[k],
                                               coarse_frac[kp], G4)

    def pairs(iq):
        return [np.repeat(np.repeat(li._atom_fourier_kernel[iq, k],
                                    NCART, axis=0), NCART, axis=1)
                for k in range(cn)]

    def neg_fine(iq):
        n = (-li._fine_idx[iq]) % li.fine_mesh
        return li._q_lookup[tuple(n)]

    def interp(iq, iqp):
        return _interp_vertex(V_coarse, pairs(iq), pairs(iqp), cn)

    scale = np.max(np.abs(interp(1, 2)))
    for iq, iqp in [(1, 2), (4, 7), (0, 3), (5, 5)]:
        V = interp(iq, iqp)
        # legs 1 <-> 2, q -> -q
        V12 = interp(neg_fine(iq), iqp)
        assert np.allclose(V, V12.transpose(1, 0, 2, 3), atol=1e-10 * scale)
        # legs 3 <-> 4, q' -> -q'
        V34 = interp(iq, neg_fine(iqp))
        assert np.allclose(V, V34.transpose(0, 1, 3, 2), atol=1e-10 * scale)
        # pair exchange (1,2) <-> (3,4).  Legs 3,4 carry (-q', +q') while
        # legs 1,2 carry (+q, -q), so the exchanged configuration is the
        # vertex at (-q', -q), not at (q', q).
        Vpp = interp(neg_fine(iqp), neg_fine(iq))
        assert np.allclose(V, Vpp.transpose(2, 3, 0, 1), atol=1e-10 * scale)


# =====================================================================
# 3. Acoustic sum rule
# =====================================================================
def _random_asr_clusters(seed, n_clusters=5, n_sites=3, max_cell=1):
    """Random quartic 'generalized bond' couplings.

    Each cluster is a set of sites ``(atom, cart, cell)`` with weights
    ``sigma_i`` summing to ZERO, contributing

        Phi4 = g * ( sum_i sigma_i delta_{s_i} )^{otimes 4}

    to the real-space quartic tensor (this is exactly the structure of the
    chain's bond term, ``6 g4 (delta_i - delta_j)^{otimes 4}``).  Such a
    coupling is by construction

    * fully symmetric under the four leg exchanges (a 4th tensor power),
    * translation covariant (it is defined on real-space sites),
    * exactly ASR-satisfying on every leg, because summing a leg over its
      site index gives ``sum_i sigma_i = 0``.

    Cells are kept inside the coarse minimum-image window so that the
    coupling is representable on the coarse mesh.
    """
    rng = np.random.default_rng(seed)
    clusters = []
    for _ in range(n_clusters):
        sigma = rng.normal(size=n_sites)
        sigma -= sigma.mean()                       # sum_i sigma_i = 0
        # All sites of a cluster share ONE Cartesian component, exactly as
        # the chain's bonds do.  The ASR is a sum over ATOMS at fixed
        # Cartesian component, so it would NOT hold for a cluster whose
        # weights were spread over several components.
        cart = int(rng.integers(NCART))
        sites = []
        for i in range(n_sites):
            sites.append((int(rng.integers(NAT)), cart,
                          int(rng.integers(-max_cell, max_cell + 1))))
        clusters.append((float(rng.normal()), sigma, sites))
    return clusters


def _cluster_bloch(cluster, k):
    """U(k) over the (atom, cart) index:  U = sum_i sigma_i e^{2 pi i k n_i}
    restricted to site i's (atom, cart) component."""
    _g, sigma, sites = cluster
    U = np.zeros(NDIM, dtype=np.complex128)
    for s, (atom, cart, cell) in zip(sigma, sites):
        U[NCART * atom + cart] += s * np.exp(2j * np.pi * k * cell)
    return U


def _asr_vertex(clusters, q, qp):
    """Exact Bloch vertex of the cluster couplings, oracle convention.

    Carrying out the real-space sums with leg 1 pinned to the home cell
    gives the closed form

        V(q,q')[a,b,c,d] = sum_clusters g * U_a(-q) U_b(q) U_c(q') U_d(-q')

    with legs at momenta (+q, -q, -q', +q') as in the module docstring.
    """
    V = np.zeros((NDIM,) * 4, dtype=np.complex128)
    for cluster in clusters:
        g = cluster[0]
        V += g * np.einsum("a,b,c,d->abcd",
                           _cluster_bloch(cluster, -q),
                           _cluster_bloch(cluster, q),
                           _cluster_bloch(cluster, qp),
                           _cluster_bloch(cluster, -qp))
    return V


def test_d4_asr_preserved(li_af, li_tri):
    """The interpolation preserves the acoustic sum rule on all four legs.

    Two independent mechanisms are required, and both are checked:

    * legs 3 and 4 carry momentum +-q', which the kernel of legs 1,2 does
      not see: ``P_ab`` must be independent of the atoms of the other
      pair.  (An atom-pinned window map, as in the windowed scheme, breaks
      exactly this -- which is why that scheme needed an explicit ASR
      repair.)
    * legs 1 and 2 reach the ASR only at q = Gamma, where the kernel must
      be exactly cardinal.
    """
    li = li_af
    clusters = _random_asr_clusters(seed=4)

    coarse_frac = np.array([c[2] / float(L) for c in li._coarse_idx])
    cn = li.cn_q

    V_coarse = np.empty((cn, cn) + (NDIM,) * 4, dtype=np.complex128)
    for k in range(cn):
        for kp in range(cn):
            V_coarse[k, kp] = _asr_vertex(clusters, coarse_frac[k],
                                          coarse_frac[kp])

    # Sanity: the constructed coarse vertex really satisfies the ASR.
    def asr_residuals(V):
        """Max |sum over the atom index of each leg| (Cartesian blocks)."""
        res = []
        for axis in range(4):
            s = V.reshape((NAT, NCART) * 4)
            s = s.sum(axis=2 * axis)
            res.append(np.max(np.abs(s)))
        return res

    igamma_c = li._coarse_lookup[(0, 0, 0)]
    scale = np.max(np.abs(V_coarse))
    # legs 1,2 ASR holds only at q = Gamma; legs 3,4 only at q' = Gamma.
    for kp in range(cn):
        r = asr_residuals(V_coarse[igamma_c, kp])
        assert max(r[0], r[1]) < 1e-10 * scale
    for k in range(cn):
        r = asr_residuals(V_coarse[k, igamma_c])
        assert max(r[2], r[3]) < 1e-10 * scale

    def kernels(lanc, iq, use_atoms):
        if use_atoms:
            return [np.repeat(np.repeat(lanc._atom_fourier_kernel[iq, k],
                                        NCART, axis=0), NCART, axis=1)
                    for k in range(cn)]
        W = _dense_trilinear(lanc)
        return [np.full((NDIM, NDIM), W[iq, k], dtype=np.complex128)
                for k in range(cn)]

    igamma_f = _fine_index(li, 0)
    for lanc, use_atoms, tag in [(li, True, "atom_fourier"),
                                 (li_tri, False, "trilinear")]:
        for iqp in range(lanc.n_q):
            V = _interp_vertex(V_coarse, kernels(lanc, igamma_f, use_atoms),
                               kernels(lanc, iqp, use_atoms), cn)
            r = asr_residuals(V)
            assert max(r[0], r[1]) < 1e-10 * scale, \
                "{}: ASR broken on legs 1,2 at fine q'={}".format(tag, iqp)
        for iq in range(lanc.n_q):
            V = _interp_vertex(V_coarse, kernels(lanc, iq, use_atoms),
                               kernels(lanc, igamma_f, use_atoms), cn)
            r = asr_residuals(V)
            assert max(r[2], r[3]) < 1e-10 * scale, \
                "{}: ASR broken on legs 3,4 at fine q={}".format(tag, iq)


def test_d4_asr_control_maps_fail(li_af):
    """The ASR tests have teeth: a non-cardinal map and an atom-pinned map
    both break the sum rule that the production kernel preserves."""
    li = li_af
    clusters = _random_asr_clusters(seed=4)
    coarse_frac = np.array([c[2] / float(L) for c in li._coarse_idx])
    cn = li.cn_q

    V_coarse = np.empty((cn, cn) + (NDIM,) * 4, dtype=np.complex128)
    for k in range(cn):
        for kp in range(cn):
            V_coarse[k, kp] = _asr_vertex(clusters, coarse_frac[k],
                                          coarse_frac[kp])
    scale = np.max(np.abs(V_coarse))

    def asr_leg(V, axis):
        s = V.reshape((NAT, NCART) * 4).sum(axis=2 * axis)
        return np.max(np.abs(s))

    igamma_f = _fine_index(li, 0)
    rng = np.random.default_rng(0)

    # (a) NON-CARDINAL map: smears Gamma over the other coarse points, so
    #     the legs 1,2 ASR (which only holds at Gamma) is destroyed.
    noncardinal = []
    for k in range(cn):
        M = np.full((NDIM, NDIM), 1.0 / cn, dtype=np.complex128)
        noncardinal.append(M)
    ident = [np.full((NDIM, NDIM), 1.0 if k == li._coarse_lookup[(0, 0, 0)]
                     else 0.0, dtype=np.complex128) for k in range(cn)]
    V = _interp_vertex(V_coarse, noncardinal, ident, cn)
    assert max(asr_leg(V, 0), asr_leg(V, 1)) > 1e-6 * scale, \
        "non-cardinal control did not break the legs 1,2 ASR"

    # (b) ATOM-PINNED map: weights that depend on the atoms of the OTHER
    #     pair cannot be written as P_ab, and break the legs 3,4 ASR.
    pinned = []
    for k in range(cn):
        M = np.asarray(rng.normal(size=(NDIM, NDIM)), dtype=np.complex128)
        pinned.append(M)
    V = _interp_vertex(V_coarse, ident, pinned, cn)
    assert max(asr_leg(V, 2), asr_leg(V, 3)) > 1e-6 * scale, \
        "atom-pinned control did not break the legs 3,4 ASR"


# =====================================================================
# 4. Operator level: the shipped code path
# =====================================================================
@pytest.mark.parametrize("nz_pert", [0, 3])
def test_d4_operator_block_hermiticity(system, nz_pert):
    """The pure-D4 two-phonon block of L is Hermitian, at Gamma and at a
    non-TRI coarse Q.

    The D4 term is the only anharmonic channel that maps the two-phonon
    sector onto itself, so it is isolated by projecting the R sector out of
    both the input and the output.  (Note that setting ``ignore_v3=True``
    does NOT isolate a Hermitian D4 operator: it zeroes R1, killing the D3
    two-phonon OUTPUT channel while leaving the D3 input channel
    alpha -> f_pert active.  That asymmetry is pre-existing behaviour of
    the parent QSpaceLanczos, measured at 2.9e-2 on this chain.)
    """
    _, ens = system
    li = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(1, 1, LF),
                                   atom_fourier=True)
    li.init(use_symmetries=True)
    li.build_q_pair_map(_fine_index(li, nz_pert))
    li.reset_q()

    nb = li.n_bands
    rng = np.random.default_rng(77 + nz_pert)
    n = li.get_psi_size()
    mask = li.mask_dot_wigner()

    def two_phonon(vec):
        out = vec.copy()
        out[:nb] = 0.0
        return out

    def apply_d4(vec):
        return two_phonon(li.apply_full_L(two_phonon(vec)))

    phi = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    d1 = np.conj(phi).dot(apply_d4(psi) * mask)
    d2 = np.conj(psi).dot(apply_d4(phi) * mask)
    assert abs(d1 - np.conj(d2)) / max(abs(d1), abs(d2), 1e-300) < 1e-9


def test_d4_operator_unfold_mirror(system):
    """The D4 output path (the unfold) reproduces the reversed pair
    orientation as the exact transpose -- the operator-level statement of
    the q1 <-> q2 leg exchange."""
    _, ens = system
    li = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(1, 1, LF),
                                   atom_fourier=True)
    li.init(use_symmetries=True)
    li.build_q_pair_map(_fine_index(li, 3))

    rng = np.random.default_rng(123)
    nb = li.n_bands
    # Coarse d2v blocks with the kernel's own pair symmetry.
    d2v = []
    for ik1, ik2 in li.c_unique_pairs:
        b = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        if ik1 == ik2:
            b = 0.5 * (b + b.T)
        d2v.append(b)

    fine = li._interp_d2v_to_fine(d2v)
    # Rebuild the full fine Cartesian field and check D(Q-q) = D(q)^T.
    D = {}
    for p, (iq1, iq2) in enumerate(li.unique_pairs):
        E1 = li.pols_q[:, :, iq1]
        E2 = li.pols_q[:, :, iq2]
        D[(iq1, iq2)] = E1 @ fine[p] @ E2.T

    # Independently interpolate the mirrored orientation and compare.
    Dc = np.zeros((li.cn_q, nb, nb), dtype=np.complex128)
    for p, (ik1, ik2) in enumerate(li.c_unique_pairs):
        E1 = li.cpols_q[:, :, ik1]
        E2 = li.cpols_q[:, :, ik2]
        Dc[ik1] = E1 @ d2v[p] @ E2.T
        if ik1 != ik2:
            Dc[ik2] = Dc[ik1].T

    for (iq1, iq2), block in D.items():
        mirrored = np.zeros((nb, nb), dtype=np.complex128)
        for ik in range(li.cn_q):
            P = li._atom_fourier_cart_kernel(iq2, ik)
            mirrored += P * Dc[ik]
        assert np.allclose(block, mirrored.T, atol=1e-10), \
            "unfold mirror broken at fine pair ({}, {})".format(iq1, iq2)


def _chain_coeffs(lanc, iq, band, steps=5, ignore_v4=False):
    lanc.ignore_v4 = ignore_v4
    lanc.init(use_symmetries=True)
    lanc.prepare_mode_q(iq, band)
    lanc.run_FT(steps, verbose=False)
    return np.array(lanc.a_coeffs)


def test_d4_commensurate_identity(system):
    """fine mesh == coarse mesh: the atom-Fourier class with D4 active must
    reproduce the parent QSpaceLanczos exactly.

    A pure-D4 run cannot be used for this: with D3 = 0 the one-phonon start
    never couples to the two-phonon sector and the Lanczos chain terminates
    after one step.  D4 is therefore probed inside the D3-opened continuum,
    and the test also asserts that D4 really does change the chain (so that
    the identity is not passing vacuously).
    """
    _, ens = system
    iq_c, band = 1, 2

    lc = QL.QSpaceLanczos(ens, lo_to_split=None)
    a_parent = _chain_coeffs(lc, iq_c, band)

    li = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(1, 1, L),
                                   atom_fourier=True)
    iq_f = li.find_fine_q(lc.q_points[iq_c])
    a_interp = _chain_coeffs(li, iq_f, band)

    n = min(len(a_parent), len(a_interp))
    assert n > 2
    assert np.max(np.abs(a_parent[:n] - a_interp[:n])
                  / np.maximum(np.abs(a_parent[:n]), 1e-30)) < 1e-10

    # D4 must actually contribute, otherwise the identity is vacuous.
    li_no4 = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(1, 1, L),
                                       atom_fourier=True)
    a_no4 = _chain_coeffs(li_no4, iq_f, band, ignore_v4=True)
    m = min(n, len(a_no4))
    rel = np.max(np.abs(a_interp[:m] - a_no4[:m])
                 / np.maximum(np.abs(a_interp[:m]), 1e-30))
    assert rel > 1e-6, "D4 does not affect the chain: test is vacuous"
