"""d3/d4 operator tests for the production atom-Fourier path."""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JULIA_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "test_interpolation"))

import _toy_chain as chain

try:
    import tdscha.QSpaceAtomFourier as AF
    import tdscha.QSpaceLanczos as QL
    _AVAILABLE = QL.__JULIA_EXT__
except Exception:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE, reason="QSpaceLanczos/Julia not available")

TEMPERATURE = 300.0
COARSE = 3
FINE = 6


@pytest.fixture(scope="module")
def system():
    dyn = chain.build_dyn(COARSE)
    ensemble = chain.make_ensemble(
        dyn, TEMPERATURE, 120, seed=7, g3=0.2, g4=0.3)
    return dyn, ensemble


@pytest.fixture(scope="module")
def interpolated(system):
    _, ensemble = system
    return AF.QSpaceAtomFourierLanczos(
        ensemble, fine_mesh=(1, 1, FINE))


def _fine_index(lanczos, z_index):
    matches = np.where(
        (lanczos._fine_idx == [0, 0, z_index]).all(axis=1))[0]
    return int(matches[0])


def _random_vector(rng, size):
    return rng.normal(size=size) + 1j * rng.normal(size=size)


def _run(lanczos, iq, band, steps=5):
    lanczos.init(use_symmetries=True)
    lanczos.prepare_mode_q(iq, band)
    lanczos.run_FT(steps, verbose=False)
    return np.asarray(lanczos.a_coeffs), np.asarray(lanczos.b_coeffs)


def test_identity_mesh_reproduces_parent_d3_and_d4(system):
    """At equal meshes the full d3+d4 operator equals QSpaceLanczos."""
    _, ensemble = system
    parent = QL.QSpaceLanczos(ensemble, lo_to_split=None)
    atom_fourier = AF.QSpaceAtomFourierLanczos(
        ensemble, fine_mesh=(1, 1, COARSE))

    parent_a, parent_b = _run(parent, iq=1, band=2)
    fine_iq = atom_fourier.find_fine_q(parent.q_points[1])
    interp_a, interp_b = _run(atom_fourier, iq=fine_iq, band=2)

    assert len(parent_a) > 2
    assert np.allclose(interp_a, parent_a, rtol=1e-10, atol=1e-14)
    assert np.allclose(interp_b, parent_b, rtol=1e-10, atol=1e-14)

    no_d4 = AF.QSpaceAtomFourierLanczos(
        ensemble, fine_mesh=(1, 1, COARSE))
    no_d4.ignore_v4 = True
    no_d4_a, _ = _run(no_d4, iq=fine_iq, band=2)
    assert not np.allclose(no_d4_a, interp_a, rtol=1e-7, atol=1e-14)


def test_atom_fourier_accepts_directional_lo_to_and_pins_gamma_basis():
    dyn = chain.build_dyn(COARSE)
    charges = np.zeros((dyn.structure.N_atoms, 3, 3))
    charges[0] = 1.5 * np.eye(3)
    charges[1] = -1.5 * np.eye(3)
    dyn.effective_charges = charges
    dyn.dielectric_tensor = np.diag([2.0, 3.0, 5.0])
    ensemble = chain.make_ensemble(
        dyn, TEMPERATURE, 12, seed=17, g3=0.2, g4=0.3)
    direction = np.array([1.0, 2.0, 3.0])

    parent = QL.QSpaceLanczos(ensemble, lo_to_split=direction)
    interpolated = AF.QSpaceAtomFourierLanczos(
        ensemble, fine_mesh=(1, 1, 2 * COARSE),
        lo_to_split=direction, allow_unstable=True)
    gamma_fine = interpolated._fine_of_coarse[0]
    np.testing.assert_allclose(
        interpolated.w_q[:, gamma_fine], parent.w_q[:, 0], atol=1e-14)
    np.testing.assert_allclose(
        interpolated.pols_q[:, :, gamma_fine], parent.pols_q[:, :, 0],
        atol=1e-14)

    short_range_parent = QL.QSpaceLanczos(ensemble, lo_to_split=None)
    short_range = AF.QSpaceAtomFourierLanczos(
        ensemble, fine_mesh=(1, 1, 2 * COARSE),
        lo_to_split=direction, ignore_effective_charges=True,
        allow_unstable=True)
    gamma_short = short_range._fine_of_coarse[0]
    np.testing.assert_allclose(
        short_range.w_q[:, gamma_short], short_range_parent.w_q[:, 0],
        atol=1e-14)
    np.testing.assert_allclose(
        short_range.pols_q[:, :, gamma_short],
        short_range_parent.pols_q[:, :, 0], atol=1e-14)

    # Suppression is interpolation-local: the original Z* remains available
    # and can still prepare an IR perturbation.
    np.testing.assert_allclose(short_range.dyn.effective_charges, charges)
    short_range.init(use_symmetries=False)
    short_range.prepare_ir(pol_vec=[1, 0, 0])
    assert short_range.perturbation_modulus > 0


def test_commensurate_frequencies_and_normalization(
        system, interpolated):
    _, ensemble = system
    parent = QL.QSpaceLanczos(ensemble, lo_to_split=None)

    for coarse_iq, fine_iq in enumerate(interpolated._fine_of_coarse):
        assert np.allclose(
            interpolated.w_q[:, fine_iq], parent.w_q[:, coarse_iq],
            atol=1e-14)
        assert np.allclose(
            interpolated.pols_q[:, :, fine_iq],
            parent.pols_q[:, :, coarse_iq], atol=1e-14)

    ratio = interpolated.cn_q / float(interpolated.n_q)
    assert interpolated.qspace_scale3 == pytest.approx(np.sqrt(ratio))
    assert interpolated.qspace_scale4 == pytest.approx(ratio)


def test_off_coarse_perturbation_is_rejected(interpolated):
    with pytest.raises(ValueError, match="coarse mesh"):
        interpolated.build_q_pair_map(_fine_index(interpolated, 1))
    with pytest.raises(ValueError, match="iq_pert"):
        interpolated.build_q_pair_map(interpolated.n_q)


@pytest.mark.parametrize("z_index", [0, 2])
def test_d3_operator_is_hermitian(interpolated, z_index):
    """With d4 disabled, d3 couples the one/two-phonon sectors adjointly."""
    interpolated.ignore_v4 = True
    interpolated.init(use_symmetries=True)
    interpolated.build_q_pair_map(_fine_index(interpolated, z_index))
    interpolated.reset_q()

    rng = np.random.default_rng(30 + z_index)
    size = interpolated.get_psi_size()
    mask = interpolated.mask_dot_wigner()
    left = _random_vector(rng, size)
    right = _random_vector(rng, size)
    lhs = np.vdot(
        left, interpolated.apply_full_L(right.copy()) * mask)
    rhs = np.vdot(
        right, interpolated.apply_full_L(left.copy()) * mask)
    assert lhs == pytest.approx(np.conj(rhs), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("z_index", [0, 2])
def test_d4_two_phonon_operator_is_hermitian(interpolated, z_index):
    """Projecting out the one-phonon sector isolates the d4 block."""
    interpolated.ignore_v4 = False
    interpolated.init(use_symmetries=True)
    interpolated.build_q_pair_map(_fine_index(interpolated, z_index))
    interpolated.reset_q()

    rng = np.random.default_rng(40 + z_index)
    size = interpolated.get_psi_size()
    n_bands = interpolated.n_bands
    mask = interpolated.mask_dot_wigner()

    def two_phonon(vector):
        vector = vector.copy()
        vector[:n_bands] = 0.0
        return vector

    def apply_d4(vector):
        return two_phonon(
            interpolated.apply_full_L(two_phonon(vector)))

    left = _random_vector(rng, size)
    right = _random_vector(rng, size)
    lhs = np.vdot(left, apply_d4(right) * mask)
    rhs = np.vdot(right, apply_d4(left) * mask)
    assert lhs == pytest.approx(np.conj(rhs), rel=1e-9, abs=1e-12)
