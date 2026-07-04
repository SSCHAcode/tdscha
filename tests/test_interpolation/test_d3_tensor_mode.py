"""
Hybrid tensor-D3 mode (d3_mode="tensor", Interpolation_plan.md section 5.8):
the D3 vertex is interpolated deterministically from a centered
cellconstructor Tensor3 (the same object the Spectral d3 bubble consumes),
while D4 stays on the stochastic plain pass. Validation on the toy chain
with the EXACT third-order tensor of the bond model:

1. operator-level convention test: on the coarse-commensurate (identity)
   mesh the deterministic D3 action (d2v blocks + f_pert through
   apply_anharmonic_FT) must agree with the stochastic estimator within
   the stochastic error, at a NON-TRI q_pert (catches any sign /
   conjugation / transpose / normalization mistake as an O(1) error);
2. Hermiticity on a genuinely interpolated fine mesh (|b - c| at machine
   level: the two deterministic maps must be exact mutual adjoints);
3. physics: interp (1,1,3) -> (1,1,6) static renormalizations vs a direct
   fine-supercell ensemble. The exact-tensor centering must beat the
   plain-window interpolation error (~30% aggregate on this bond-ranged
   model) and sit near the stochastic noise floor (~10%).

Runtime: ~2-3 minutes (dominated by the physics fixture).
"""
import os, sys
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest

import cellconstructor as CC
import cellconstructor.ForceTensor

import _toy_chain as TC

try:
    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceInterpolation as QI
    _HAS_Q = QL.__JULIA_EXT__
except Exception:
    _HAS_Q = False

pytestmark = pytest.mark.skipif(not _HAS_Q, reason="QSpaceLanczos/Julia not available")

T = 300.0
G3 = 0.1
LC, LF = 3, 6


def exact_phi3_tensor(dyn, g3):
    """Exact third-order tensor of the bond model on the dyn supercell,
    centered + ASR (Ry/Bohr^3, the convention the Spectral bubble uses)."""
    unit = dyn.structure
    L = dyn.GetSupercell()[2]
    sc_struct = unit.generate_supercell(dyn.GetSupercell())
    nat_sc = sc_struct.N_atoms
    bonds = TC.get_bonds(sc_struct, unit, L)
    phi3 = np.zeros((3 * nat_sc, 3 * nat_sc, 3 * nat_sc))
    for (i, j, _k) in bonds:
        for alpha in range(3):
            for (a, sa) in ((i, 1.0), (j, -1.0)):
                for (b, sb) in ((i, 1.0), (j, -1.0)):
                    for (c, sc_) in ((i, 1.0), (j, -1.0)):
                        phi3[3 * a + alpha, 3 * b + alpha, 3 * c + alpha] += \
                            2.0 * g3 * sa * sb * sc_
    t3 = CC.ForceTensor.Tensor3(unit, sc_struct, dyn.GetSupercell())
    t3.SetupFromTensor(phi3)
    t3.Center(Far=3)
    t3.Apply_ASR()
    return t3


@pytest.fixture(scope="module")
def coarse_system():
    dyn = TC.build_dyn(LC)
    ens = TC.make_ensemble(dyn, T, 4000, seed=3, g3=G3)
    t3 = exact_phi3_tensor(dyn, G3)
    return dyn, ens, t3


def test_requires_tensor(coarse_system):
    dyn, ens, t3 = coarse_system
    with pytest.raises(ValueError):
        QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, LF), d3_mode="tensor")


def test_forces_plain_windows(coarse_system):
    dyn, ens, t3 = coarse_system
    with pytest.warns(UserWarning):
        li = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, LF),
                                    d3_mode="tensor", d3_tensor=t3,
                                    window_design="minimal_image")
    assert li.window_design == "plain"


def test_operator_convention_on_grid(coarse_system):
    """Deterministic D3 action == stochastic estimator on the identity mesh
    (where the plain estimator is exact in expectation), non-TRI q_pert."""
    dyn, ens, t3 = coarse_system

    ls = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, LC))
    lt = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, LC),
                                d3_mode="tensor", d3_tensor=t3)
    outs = []
    for lanc in (ls, lt):
        lanc.init(use_symmetries=True)
        lanc.ignore_v4 = True          # isolate the D3 channel (g4 = 0)
        # non-TRI q_pert: n_z = 1 -> q = 1/3
        iq = int(np.where((lanc._fine_idx == [0, 0, 1]).all(axis=1))[0][0])
        lanc.prepare_mode_q(iq, 2)
        rs = np.random.RandomState(42)
        psi = (rs.randn(lanc.get_psi_size())
               + 1j * rs.randn(lanc.get_psi_size()))
        lanc.psi = psi
        outs.append(lanc.apply_anharmonic_FT())

    o_s, o_t = outs
    scale = np.linalg.norm(o_s)
    assert scale > 0
    # The residual is pure sampling noise of the stochastic side: measured
    # 1/sqrt(N) convergence rel = 0.134 / 0.101 / 0.047 at N = 2k/8k/32k
    # with the least-squares scalar fit -> 0.999. Any convention error
    # (sign, conjugation, transpose, off-diagonal factor 2, the 1/2 of
    # f_pert, N_f normalization) is O(30%)-O(1) and N-independent.
    rel = np.linalg.norm(o_s - o_t) / scale
    cos = np.abs(np.vdot(o_s, o_t)) / (np.linalg.norm(o_s) * np.linalg.norm(o_t))
    assert rel < 0.18, "stochastic vs tensor D3 mismatch: rel={:.3f}".format(rel)
    assert cos > 0.99, "direction mismatch: cos={:.5f}".format(cos)
    nb = ls.n_bands
    for sl, name in [(slice(0, nb), "f_pert"), (slice(nb, None), "d2v")]:
        cfit = np.vdot(o_s[sl], o_t[sl]) / np.vdot(o_s[sl], o_s[sl])
        assert 0.85 < abs(cfit) < 1.15, \
            "{} scalar fit off: |c|={:.3f}".format(name, abs(cfit))


def test_hermiticity_interpolated(coarse_system):
    """The two deterministic D3 maps are exact mutual adjoints: |b - c|
    must stay at machine level on a genuinely interpolated mesh."""
    dyn, ens, t3 = coarse_system
    li = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, LF),
                                d3_mode="tensor", d3_tensor=t3)
    li.init(use_symmetries=True)
    # genuinely interpolated, non-TRI q: n_z = 1 -> q = 1/6
    iq = int(np.where((li._fine_idx == [0, 0, 1]).all(axis=1))[0][0])
    li.prepare_mode_q(iq, 4)
    li.run_FT(6, verbose=False)
    a = np.array(li.a_coeffs)
    b = np.array(li.b_coeffs)
    c = np.array(li.c_coeffs)
    assert np.all(np.abs(np.imag(a)) < 1e-12)
    n = min(len(b), len(c))
    assert np.max(np.abs(b[:n] - c[:n]) / np.abs(b[:n])) < 1e-8


@pytest.fixture(scope="module")
def physics_renorms(coarse_system):
    dyn, ens, t3 = coarse_system
    dynf = TC.build_dyn(LF)
    ensf = TC.make_ensemble(dynf, T, 4000, seed=77, g3=G3)

    lt = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, LF),
                                d3_mode="tensor", d3_tensor=t3)
    ld = QL.QSpaceLanczos(ensf, lo_to_split=None)
    lt.init(use_symmetries=True)
    ld.init(use_symmetries=True)

    def get(lanc, iq, band):
        lanc.prepare_mode_q(iq, band)
        lanc.run_FT(35, verbose=False)
        return TC.lanczos_effective_freq(lanc) - lanc.w_q[band, iq]

    # interpolated non-TRI (1/6), commensurate (1/3), interpolated ZB (1/2)
    probes = [(1, 0), (1, 4), (2, 3), (3, 2)]
    out = {}
    bg = dyn.structure.get_reciprocal_vectors() / (2 * np.pi)
    for n_z, band in probes:
        iq_f = int(np.where((lt._fine_idx == [0, 0, n_z]).all(axis=1))[0][0])
        iq_d = None
        for jq in range(ld.n_q):
            if CC.Methods.get_min_dist_into_cell(
                    bg, np.asarray(ld.q_points[jq]),
                    np.asarray(lt.q_points[iq_f])) < 1e-6:
                iq_d = jq
                break
        assert iq_d is not None
        out[(n_z, band)] = (get(lt, iq_f, band), get(ld, iq_d, band))
    return out


def test_physics_tensor_mode(physics_renorms):
    """Aggregate renormalization error vs the direct fine ensemble must be
    near the stochastic noise floor (~10%), far below the plain-window
    error (~30%) measured on this model."""
    num = 0.0
    den = 0.0
    for (interp, direct) in physics_renorms.values():
        assert np.sign(interp) == np.sign(direct)
        num += abs(interp - direct)
        den += abs(direct)
    agg = num / den
    assert agg < 0.20, "aggregate renorm error {:.3f}".format(agg)
