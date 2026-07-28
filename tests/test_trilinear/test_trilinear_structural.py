"""
Structural tests of the trilinear q-space interpolation
(QSpaceTrilinearLanczos), on the P1 anharmonic diatomic chain:

1. identity: fine_mesh == coarse mesh reproduces QSpaceLanczos exactly
   (gauge-invariant Lanczos coefficients, machine precision);
2. geometry: corner weights sum to one, commensurate points collapse to a
   single corner, pinned commensurate dyn equals the coarse one;
3. fold transpose symmetry: A(Q - k) = A(k)^T exactly (the permutation
   symmetry of the interpolated vertex), at TRI and non-TRI Q;
4. Hermiticity of the full L under the masked inner product, at TRI and
   non-TRI Q (guards the conjugation-convention class of bugs);
5. the Hermitian-symmetric mesh factors sqrt(N_c/N_f), N_c/N_f.
"""
import os, sys
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "test_interpolation"))
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
N_CONF = 300
N_STEPS = 6
L = 3
LF = 6


@pytest.fixture(scope="module")
def system():
    dyn = TC.build_dyn(L)
    ens = TC.make_ensemble(dyn, T, N_CONF, seed=7, g3=0.2, g4=0.3)
    return dyn, ens


@pytest.fixture(scope="module")
def li_fine(system):
    dyn, ens = system
    return QT.QSpaceTrilinearLanczos(ens, fine_mesh=(1, 1, LF))


def _run(lanc, iq, band):
    lanc.init(use_symmetries=True)
    lanc.prepare_mode_q(iq, band)
    lanc.run_FT(N_STEPS, verbose=False)
    return np.array(lanc.a_coeffs), np.array(lanc.b_coeffs)


# =====================================================================
def test_identity_coefficients(system):
    """fine mesh == coarse mesh: the fold is the identity and the Lanczos
    coefficients must match QSpaceLanczos to machine precision."""
    dyn, ens = system
    lc = QL.QSpaceLanczos(ens, lo_to_split=None)
    li = QT.QSpaceTrilinearLanczos(ens, fine_mesh=(1, 1, L))

    assert li.qspace_scale3 == 1.0 and li.qspace_scale4 == 1.0

    for iq_c, band in [(0, 5), (1, 2)]:
        iq_f = li.find_fine_q(lc.q_points[iq_c])
        a_c, b_c = _run(lc, iq_c, band)
        a_i, b_i = _run(li, iq_f, band)
        n = min(len(a_c), len(a_i))
        assert n > 2
        assert np.max(np.abs(a_c[:n] - a_i[:n]) / np.abs(a_c[:n])) < 1e-10
        m = min(len(b_c), len(b_i))
        assert np.max(np.abs(b_c[:m] - b_i[:m]) / np.abs(b_c[:m])) < 1e-10


# =====================================================================
def test_corner_geometry(li_fine):
    li = li_fine
    for iq, entries in enumerate(li._corners):
        w_sum = sum(e[1] for e in entries)
        assert abs(w_sum - 1.0) < 1e-14, "weights at fine iq={}".format(iq)
    # commensurate fine points: exactly one corner, weight 1, the point itself
    for jq in range(li.cn_q):
        iq = li._fine_of_coarse[jq]
        entries = li._corners[iq]
        assert len(entries) == 1
        assert entries[0][0] == jq and abs(entries[0][1] - 1.0) < 1e-15
    # non-commensurate points: more than one corner
    n_multi = sum(1 for e in li._corners if len(e) > 1)
    assert n_multi == li.n_q - li.cn_q


def test_pinned_commensurate_dyn(system, li_fine):
    dyn, ens = system
    lc = QL.QSpaceLanczos(ens, lo_to_split=None)
    li = li_fine
    for jq in range(lc.n_q):
        iq = li.find_fine_q(lc.q_points[jq])
        assert np.allclose(li.w_q[:, iq], lc.w_q[:, jq], atol=1e-14)
        assert np.allclose(li.pols_q[:, :, iq], lc.pols_q[:, :, jq],
                           atol=1e-14)


def test_scale_factors(li_fine):
    li = li_fine
    ratio = li.cn_q / float(li.n_q)
    assert abs(li.qspace_scale3 - np.sqrt(ratio)) < 1e-15
    assert abs(li.qspace_scale4 - ratio) < 1e-15
    assert ratio == pytest.approx(L / float(LF))


# =====================================================================
def _random_alpha_fine(li, seed):
    """Random fine alpha blocks with the storage symmetry: diagonal pairs
    (q' = Q - q') have complex-symmetric blocks."""
    rng = np.random.default_rng(seed)
    nb = li.n_bands
    blocks = []
    for iq1, iq2 in li.unique_pairs:
        b = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        if iq1 == iq2:
            b = 0.5 * (b + b.T)
        blocks.append(b)
    return blocks


@pytest.mark.parametrize("nz_pert", [0, 2, 5])   # Gamma (TRI), and non-TRI Q
def test_fold_transpose_symmetry(li_fine, nz_pert):
    """A(Q - k) = A(k)^T exactly: the q1 <-> q2 permutation symmetry of the
    folded (interpolated) vertex kernel."""
    li = li_fine
    iq_pert = int(np.where((li._fine_idx == [0, 0, nz_pert]).all(axis=1))[0][0])
    # Q must be coarse-commensurate: nz on the fine (1,1,6) mesh maps to the
    # coarse (1,1,3) mesh only for even nz
    if nz_pert % (LF // L) != 0:
        with pytest.raises(ValueError):
            li.build_q_pair_map(iq_pert)
        return
    li.build_q_pair_map(iq_pert)
    A = li._fold_alpha_cart(_random_alpha_fine(li, seed=3 + nz_pert))
    for ik in range(li.cn_q):
        ik2 = int(li.c_q_pair_map[ik])
        assert np.allclose(A[ik2], A[ik].T, atol=1e-12), \
            "fold transpose broken at coarse ik={} (Q index {})".format(
                ik, nz_pert)


def test_pair_map_rejects_incommensurate_q(li_fine):
    """Perturbations off the coarse mesh must be rejected."""
    li = li_fine
    iq = int(np.where((li._fine_idx == [0, 0, 1]).all(axis=1))[0][0])
    with pytest.raises(ValueError):
        li.build_q_pair_map(iq)


# =====================================================================
@pytest.mark.parametrize("nz_pert", [0, 2])      # Gamma and non-TRI coarse Q
def test_hermiticity(li_fine, nz_pert):
    """<phi, L psi> = conj(<psi, L phi>) under the masked inner product."""
    li = li_fine
    li.init(use_symmetries=True)
    iq_pert = int(np.where((li._fine_idx == [0, 0, nz_pert]).all(axis=1))[0][0])
    li.build_q_pair_map(iq_pert)
    li.reset_q()

    rng = np.random.default_rng(42 + nz_pert)
    n = li.get_psi_size()
    mask = li.mask_dot_wigner()

    def rand_psi():
        return (rng.normal(size=n) + 1j * rng.normal(size=n))

    phi, psi = rand_psi(), rand_psi()
    L_psi = li.apply_full_L(psi.copy())
    L_phi = li.apply_full_L(phi.copy())

    d1 = np.conj(phi).dot(L_psi * mask)
    d2 = np.conj(psi).dot(L_phi * mask)
    scale = max(abs(d1), abs(d2), 1e-300)
    assert abs(d1 - np.conj(d2)) / scale < 1e-9, \
        "L not Hermitian at Q index {}: {} vs conj({})".format(
            nz_pert, d1, d2)


def test_atomic_gauge_hermiticity(system):
    """The full-Bloch atomic phase must be applied as a gauge pair on fold
    and unfold; its inverse-adjoint return path keeps the full L Hermitian."""
    _, ens = system
    li = QT.QSpaceTrilinearLanczos(
        ens, fine_mesh=(1, 1, LF), atomic_phase=True)
    li.init(use_symmetries=True)
    iq = int(np.where((li._fine_idx == [0, 0, 2]).all(axis=1))[0][0])
    li.build_q_pair_map(iq)
    li.reset_q()

    A = li._fold_alpha_cart(_random_alpha_fine(li, seed=90))
    for ik in range(li.cn_q):
        ik2 = int(li.c_q_pair_map[ik])
        assert np.allclose(A[ik2], A[ik].T, atol=1e-12)

    rng = np.random.default_rng(91)
    n = li.get_psi_size()
    phi = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    mask = li.mask_dot_wigner()
    d1 = np.conj(phi).dot(li.apply_full_L(psi.copy()) * mask)
    d2 = np.conj(psi).dot(li.apply_full_L(phi.copy()) * mask)
    assert abs(d1 - np.conj(d2)) / max(abs(d1), abs(d2), 1e-300) < 1e-9


def test_atom_fourier_cardinal_and_harmonic(system):
    """The atom-centred kernel is cardinal on the coarse mesh and exactly
    reproduces each Fourier image selected by the intracell separation."""
    _, ens = system
    li = QT.QSpaceTrilinearLanczos(
        ens, fine_mesh=(1, 1, LF), atom_fourier=True)

    # Every atom-pair kernel is the identity at commensurate points.
    for jq in range(li.cn_q):
        iq = li._fine_of_coarse[jq]
        for ik in range(li.cn_q):
            expected = 1.0 if ik == jq else 0.0
            assert np.allclose(li._atom_fourier_kernel[iq, ik], expected,
                               atol=1e-12)

    tau = np.linalg.solve(li.uci_structure.unit_cell.T,
                          li.uci_structure.coords.T).T
    ia, ib = 0, 1
    d = tau[ia, 2] - tau[ib, 2]
    images = li._nearest_alias_images(1, d, L)
    q_coarse = np.arange(L, dtype=float) / L
    samples = np.array([
        sum(w * np.exp(2j * np.pi * q * R) for R, w in images)
        for q in q_coarse])
    for iq, n in enumerate(li._fine_idx):
        q = n[2] / float(LF)
        expected = sum(w * np.exp(2j * np.pi * q * R)
                       for R, w in images)
        got = sum(li._atom_fourier_kernel[iq, ik, ia, ib]
                  * samples[li._coarse_idx[ik][2]]
                  for ik in range(li.cn_q))
        assert np.allclose(got, expected, atol=1e-12)


def test_atom_fourier_hermiticity(system):
    """The cardinal map and its adjoint preserve pair symmetry and L."""
    _, ens = system
    li = QT.QSpaceTrilinearLanczos(
        ens, fine_mesh=(1, 1, LF), atom_fourier=True)
    li.init(use_symmetries=True)
    iq = int(np.where((li._fine_idx == [0, 0, 2]).all(axis=1))[0][0])
    li.build_q_pair_map(iq)
    li.reset_q()

    A = li._fold_alpha_cart(_random_alpha_fine(li, seed=190))
    for ik in range(li.cn_q):
        ik2 = int(li.c_q_pair_map[ik])
        assert np.allclose(A[ik2], A[ik].T, atol=1e-12)

    rng = np.random.default_rng(191)
    n = li.get_psi_size()
    phi = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    mask = li.mask_dot_wigner()
    d1 = np.conj(phi).dot(li.apply_full_L(psi.copy()) * mask)
    d2 = np.conj(psi).dot(li.apply_full_L(phi.copy()) * mask)
    assert abs(d1 - np.conj(d2)) / max(abs(d1), abs(d2), 1e-300) < 1e-9
