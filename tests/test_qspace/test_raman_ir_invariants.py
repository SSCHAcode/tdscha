"""Fast regression tests for Raman invariants and IR backend scaling."""

import importlib

import numpy as np
import pytest

import tdscha.DynamicalLanczos as DL
import tdscha.QSpaceLanczos as QL


N_CELL = 4
N_ATOMS = 2
N_CART = 3 * N_ATOMS
MASSES = np.array([1.0, 4.0])


class _FakeDyn:
    def __init__(self):
        self.raman_tensor = np.zeros((3, 3, N_CART))
        components = {
            (0, 0): [1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
            (1, 1): [-3.0, 5.0, 2.0, -1.0, 7.0, 4.0],
            (2, 2): [8.0, 1.0, -4.0, 3.0, 2.0, -6.0],
            (0, 1): [2.0, 3.0, -1.0, 5.0, -2.0, 7.0],
            (0, 2): [-5.0, 4.0, 6.0, 2.0, 1.0, -3.0],
            (1, 2): [7.0, -6.0, 5.0, -4.0, 3.0, -2.0],
        }
        for (i, j), values in components.items():
            self.raman_tensor[i, j] = values
            self.raman_tensor[j, i] = values

        self.effective_charges = np.array([
            [[1.0, 2.0, -1.0], [3.0, -2.0, 4.0], [5.0, 1.0, 2.0]],
            [[-2.0, 1.0, 3.0], [4.0, 2.0, -3.0], [1.0, -5.0, 6.0]],
        ])

    def GetRamanVector(self, pol_in, pol_out):
        return np.einsum(
            "i,j,ijk->k", pol_in, pol_out, self.raman_tensor)

    @staticmethod
    def GetSupercell():
        return np.array([2, 1, 2])


class _FakeStructure:
    @staticmethod
    def get_masses_array():
        return MASSES.copy()


class _RealHarness(DL.Lanczos):
    def __init__(self, dyn):
        self.dyn = dyn
        masses_uc = np.repeat(MASSES, 3)
        self.m = np.tile(masses_uc, N_CELL)
        self.n_modes = N_CART
        gamma = np.ones(N_CELL) / np.sqrt(N_CELL)
        self.pols = np.kron(gamma[:, None], np.eye(N_CART))
        self.psi = np.zeros(self.n_modes)
        self.symmetrize = False
        self.ignore_small_w = True

    def reset(self):
        pass


class _QSpaceHarness(QL.QSpaceLanczos):
    def __init__(self, dyn):
        self.dyn = dyn
        self.uci_structure = _FakeStructure()
        self.n_bands = N_CART
        self.pols_q = np.eye(N_CART, dtype=np.complex128)[:, :, None]
        self.psi = np.zeros(self.n_bands, dtype=np.complex128)

    def build_q_pair_map(self, iq):
        assert iq == 0

    def reset_q(self):
        self.psi = np.zeros(self.n_bands, dtype=np.complex128)


@pytest.fixture
def dyn():
    return _FakeDyn()


def _raw_invariants(tensor):
    return [
        tensor[0, 0] + tensor[1, 1] + tensor[2, 2],
        tensor[0, 0] - tensor[1, 1],
        tensor[0, 0] - tensor[2, 2],
        tensor[1, 1] - tensor[2, 2],
        tensor[0, 1],
        tensor[0, 2],
        tensor[1, 2],
    ]


def _projected_unit_cell(vector):
    masses = np.repeat(MASSES, 3)
    return np.sqrt(N_CELL) * vector / np.sqrt(masses)


def test_shared_builder_returns_all_seven_invariants(dyn):
    lanc = _RealHarness(dyn)
    raw = _raw_invariants(dyn.raman_tensor)
    scales = [1 / 3] + [1 / np.sqrt(2)] * 3 + [np.sqrt(3)] * 3

    for index, expected_raw in enumerate(raw):
        np.testing.assert_allclose(
            lanc._build_raman_vector(
                unpolarized=index, normalized=False),
            expected_raw)
        np.testing.assert_allclose(
            lanc._build_raman_vector(
                unpolarized=index, normalized=True),
            expected_raw * scales[index])


@pytest.mark.parametrize("backend", [_RealHarness, _QSpaceHarness])
def test_polarized_and_coherently_mixed_raman(backend, dyn):
    lanc = backend(dyn)
    pol_in = np.array([0.5, -1.0, 2.0])
    pol_out = np.array([1.5, 0.25, -0.75])
    pol_in_2 = np.array([-0.5, 2.0, 1.0])
    pol_out_2 = np.array([1.0, -1.5, 0.5])

    direct = dyn.GetRamanVector(pol_in, pol_out)
    lanc.prepare_raman(pol_vec_in=pol_in, pol_vec_out=pol_out)
    np.testing.assert_allclose(lanc.psi[:N_CART], _projected_unit_cell(direct))
    np.testing.assert_allclose(
        lanc.perturbation_modulus,
        np.vdot(_projected_unit_cell(direct),
                _projected_unit_cell(direct)).real)

    direct_mixed = direct + dyn.GetRamanVector(pol_in_2, pol_out_2)
    lanc.prepare_raman(
        pol_vec_in=pol_in, pol_vec_out=pol_out, mixed=True,
        pol_in_2=pol_in_2, pol_out_2=pol_out_2)
    np.testing.assert_allclose(
        lanc.psi[:N_CART], _projected_unit_cell(direct_mixed))
    np.testing.assert_allclose(
        lanc.perturbation_modulus,
        np.vdot(_projected_unit_cell(direct_mixed),
                _projected_unit_cell(direct_mixed)).real)


@pytest.mark.parametrize("backend", [_RealHarness, _QSpaceHarness])
def test_both_unpolarized_apis_have_equivalent_weighted_intensities(
        backend, dyn):
    normalized_weights = [45] + [7] * 6
    lanc = backend(dyn)

    for index, normalized_weight in enumerate(normalized_weights):
        lanc.prepare_raman(unpolarized=index)
        normalized_intensity = (
            normalized_weight * lanc.perturbation_modulus)

        lanc.prepare_unpolarized_raman(index=index)
        raw_intensity = (
            lanc.get_prefactors_unpolarized_raman(index)
            * lanc.perturbation_modulus)

        np.testing.assert_allclose(normalized_intensity, raw_intensity)


def test_real_and_qspace_raman_vectors_and_moduli_match(dyn):
    real = _RealHarness(dyn)
    qspace = _QSpaceHarness(dyn)
    cases = [
        {"pol_vec_in": np.array([0.5, -1.0, 2.0]),
         "pol_vec_out": np.array([1.5, 0.25, -0.75])},
        {"pol_vec_in": np.array([1.0, 0.0, 0.0]),
         "pol_vec_out": np.array([0.0, 1.0, 0.0]),
         "mixed": True,
         "pol_in_2": np.array([0.0, 0.0, 1.0]),
         "pol_out_2": np.array([0.0, 1.0, 0.0])},
    ]
    cases.extend({"unpolarized": index} for index in range(7))

    for kwargs in cases:
        real.prepare_raman(**kwargs)
        qspace.prepare_raman(**kwargs)
        np.testing.assert_allclose(
            qspace.psi[:N_CART], real.psi[:N_CART])
        np.testing.assert_allclose(
            qspace.perturbation_modulus, real.perturbation_modulus)


def test_generic_add_uses_complete_accumulated_perturbation(dyn):
    vector_1 = np.arange(1, N_CART + 1, dtype=float)
    vector_2 = np.array([-2.0, 1.0, 4.0, -3.0, 5.0, 2.0])

    real = _RealHarness(dyn)
    vector_1_sc = np.tile(vector_1, N_CELL)
    vector_2_sc = np.tile(vector_2, N_CELL)
    real.prepare_perturbation(vector_1_sc, masses_exp=-1)
    real.prepare_perturbation(vector_2_sc, masses_exp=-1, add=True)
    expected = _projected_unit_cell(vector_1 + vector_2)
    np.testing.assert_allclose(real.psi[:N_CART], expected)
    np.testing.assert_allclose(real.perturbation_modulus, expected @ expected)

    qspace = _QSpaceHarness(dyn)
    qspace.prepare_perturbation_q(0, vector_1 * np.sqrt(N_CELL))
    qspace.prepare_perturbation_q(
        0, vector_2 * np.sqrt(N_CELL), add=True)
    np.testing.assert_allclose(qspace.psi[:N_CART], expected)
    np.testing.assert_allclose(
        qspace.perturbation_modulus, np.vdot(expected, expected).real)


def test_ir_real_qspace_parity_and_powder_average(dyn):
    real = _RealHarness(dyn)
    qspace = _QSpaceHarness(dyn)
    moduli = []

    for pol_vec in np.eye(3):
        direct = np.einsum(
            "abc,b->ac", dyn.effective_charges, pol_vec).ravel()
        expected = _projected_unit_cell(direct)

        real.prepare_ir(pol_vec=pol_vec)
        qspace.prepare_ir(pol_vec=pol_vec)
        np.testing.assert_allclose(real.psi[:N_CART], expected)
        np.testing.assert_allclose(qspace.psi[:N_CART], expected)
        np.testing.assert_allclose(
            qspace.perturbation_modulus, real.perturbation_modulus)
        moduli.append(real.perturbation_modulus)

    powder_average = sum(moduli) / 3
    direct_powder_average = sum(
        np.vdot(
            _projected_unit_cell(
                np.einsum(
                    "abc,b->ac", dyn.effective_charges, pol_vec).ravel()),
            _projected_unit_cell(
                np.einsum(
                    "abc,b->ac", dyn.effective_charges, pol_vec).ravel())
        ).real
        for pol_vec in np.eye(3)
    ) / 3
    np.testing.assert_allclose(powder_average, direct_powder_average)


def test_atom_fourier_lanczos_inherits_qspace_raman_implementation():
    try:
        module = importlib.import_module("tdscha.QSpaceAtomFourier")
    except ImportError:
        pytest.skip("QSpaceAtomFourier is only present on the 1.8 branch")

    cls = module.QSpaceAtomFourierLanczos
    assert issubclass(cls, QL.QSpaceLanczos)
    assert "prepare_raman" not in cls.__dict__
    assert "prepare_unpolarized_raman" not in cls.__dict__
    assert "prepare_perturbation_q" not in cls.__dict__


def test_qspace_uses_only_backend_hook_for_optical_perturbations():
    assert "prepare_ir" not in QL.QSpaceLanczos.__dict__
    assert "prepare_raman" not in QL.QSpaceLanczos.__dict__
    assert "prepare_unpolarized_raman" not in QL.QSpaceLanczos.__dict__
    assert "_prepare_gamma_cartesian_perturbation" in (
        QL.QSpaceLanczos.__dict__)


@pytest.mark.parametrize("backend", [_RealHarness, _QSpaceHarness])
@pytest.mark.parametrize("method_name", [
    "prepare_unpolarized_raman_FT",
    "prepare_anharmonic_raman_FT",
    "prepare_anharmonic_raman_FT_2ph",
])
def test_unvalidated_two_phonon_raman_is_disabled(backend, method_name, dyn):
    lanczos = backend(dyn)
    with pytest.raises(NotImplementedError, match="Two-phonon Raman"):
        getattr(lanczos, method_name)()


@pytest.mark.parametrize("backend", [_RealHarness, _QSpaceHarness])
@pytest.mark.parametrize("method_name", [
    "prepare_anharmonic_ir_FT",
    "prepare_anharmonic_ir",
])
def test_unvalidated_configuration_dependent_ir_is_disabled(
        backend, method_name, dyn):
    lanczos = backend(dyn)
    with pytest.raises(NotImplementedError, match="Configuration-dependent"):
        getattr(lanczos, method_name)()
