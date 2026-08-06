"""Tests for the backend-neutral spectroscopy API and symmetry algebra."""

import itertools

import numpy as np
import pytest

import tdscha.Spectroscopy as SP


def _cubic_rotations():
    """Return the 24 proper signed-permutation rotations of a cube."""
    rotations = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[list(permutation)]
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = np.diag(signs) @ permutation_matrix
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    return rotations


def _random_symmetric_raman(seed=1234, n_coordinates=11):
    generator = np.random.default_rng(seed)
    tensor = generator.normal(size=(3, 3, n_coordinates))
    return (tensor + tensor.swapaxes(0, 1)) / 2


def test_normalized_and_raw_raman_definitions_are_equivalent():
    tensor = _random_symmetric_raman()
    normalized_total = 0.0
    raw_total = 0.0

    for component in SP.RAMAN_COMPONENTS:
        normalized = SP.build_raman_vector(
            tensor, component.coefficients("normalized"))
        raw = SP.build_raman_vector(
            tensor, component.coefficients("raw"))
        normalized_total += component.weight("normalized") * normalized**2
        raw_total += component.weight("raw") * raw**2

    np.testing.assert_allclose(normalized_total, raw_total)
    np.testing.assert_allclose(
        SP.get_unpolarized_raman_weights("normalized"),
        [45, 7, 7, 7, 7, 7, 7])
    np.testing.assert_allclose(
        SP.get_unpolarized_raman_weights("raw"),
        [5, 3.5, 3.5, 3.5, 21, 21, 21])


def test_component_vectors_match_the_placzek_formula():
    tensor = _random_symmetric_raman(n_coordinates=7)
    responses = []
    for component in SP.RAMAN_COMPONENTS:
        vector = SP.build_raman_vector(
            tensor, component.coefficients("normalized"))
        responses.append(vector**2)

    assembled = 45 * responses[0] + 7 * sum(responses[1:])
    alpha = (tensor[0, 0] + tensor[1, 1] + tensor[2, 2]) / 3
    gamma_squared = (
        ((tensor[0, 0] - tensor[1, 1])**2
         + (tensor[0, 0] - tensor[2, 2])**2
         + (tensor[1, 1] - tensor[2, 2])**2) / 2
        + 3 * (tensor[0, 1]**2 + tensor[0, 2]**2
               + tensor[1, 2]**2))
    np.testing.assert_allclose(assembled, 45 * alpha**2 + 7 * gamma_squared)


def test_polarized_coefficients_match_direct_symmetric_contraction():
    tensor = _random_symmetric_raman()
    incoming = np.array([0.5, -1.0, 2.0])
    outgoing = np.array([1.5, 0.25, -0.75])
    coefficients = SP.raman_coefficients_from_polarizations(
        incoming, outgoing)

    actual = SP.build_raman_vector(tensor, coefficients)
    expected = np.einsum("a,b,abk->k", incoming, outgoing, tensor)
    np.testing.assert_allclose(actual, expected)


def test_ir_builder_matches_existing_axis_convention():
    effective_charges = np.arange(18, dtype=float).reshape(2, 3, 3)
    direction = np.array([0.25, -0.5, 1.5])
    np.testing.assert_allclose(
        SP.build_ir_vector(effective_charges, direction),
        np.einsum("abc,b->ac", effective_charges, direction).ravel())


def test_specs_are_immutable_and_validate_physical_inputs():
    ir = SP.IRPolarizationPerturbation([2, 0, 0])
    np.testing.assert_allclose(ir.as_array(), [1, 0, 0])
    returned = ir.as_array()
    returned[0] = 10
    np.testing.assert_allclose(ir.as_array(), [1, 0, 0])

    with pytest.raises(ValueError, match="must not be zero"):
        SP.IRPolarizationPerturbation([0, 0, 0])
    with pytest.raises(ValueError, match="must be symmetric"):
        SP.RamanTensorPerturbation([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="0 to 6"):
        SP.get_raman_component(7)


def test_cubic_ir_orbit_reduces_three_axes_to_one_run():
    group = SP.SymmetryGroup.from_matrices(_cubic_rotations())
    vectors = np.eye(3)
    orbits = SP.find_perturbation_orbits(
        vectors, SP.vector_representations_for_ir(group), group)

    assert len(orbits) == 1
    orbit = orbits[0]
    assert orbit.members == (0, 1, 2)
    assert len(orbit.stabilizer) == 8
    assert len(orbit.left_cosets) == 3
    assert len(orbit.right_cosets) == 3
    assert set(orbit.characters) == {-1.0, 1.0}


def test_left_and_right_cosets_are_complete_partitions():
    group = SP.SymmetryGroup.from_matrices(_cubic_rotations())
    orbit = SP.find_perturbation_orbits(
        np.eye(3), SP.vector_representations_for_ir(group), group)[0]
    for cosets in (orbit.left_cosets, orbit.right_cosets):
        flattened = [member for coset in cosets for member in coset]
        assert sorted(flattened) == list(range(len(group)))
        assert all(len(coset) == len(orbit.stabilizer)
                   for coset in cosets)


def test_cubic_raman_components_reduce_to_three_orbits():
    group = SP.SymmetryGroup.from_matrices(_cubic_rotations())
    representations = SP.vector_representations_for_symmetric_raman(group)
    vectors = [SP.symmetric_raman_vector(
        component.coefficients("normalized"))
        for component in SP.RAMAN_COMPONENTS]
    orbits = SP.find_perturbation_orbits(vectors, representations, group)

    assert [orbit.members for orbit in orbits] == [
        (0,), (1, 2, 3), (4, 5, 6)]
    assert [len(orbit.left_cosets) for orbit in orbits] == [1, 3, 3]


def test_identity_group_never_overreduces():
    group = SP.SymmetryGroup.from_matrices([np.eye(3)])
    vectors = np.eye(3)
    orbits = SP.find_perturbation_orbits(
        vectors, SP.vector_representations_for_ir(group), group)
    assert [orbit.members for orbit in orbits] == [(0,), (1,), (2,)]


def test_orbit_analysis_rejects_a_mismatched_representation():
    group = SP.SymmetryGroup.from_matrices(_cubic_rotations())
    representations = list(SP.vector_representations_for_ir(group))
    representations[1] = np.eye(3)
    with pytest.raises(ValueError, match="multiplication table"):
        SP.find_perturbation_orbits(np.eye(3), representations, group)


def test_symmetry_group_rejects_non_groups():
    with pytest.raises(ValueError, match="closed"):
        SP.SymmetryGroup.from_matrices([
            np.eye(3),
            np.diag([-1, 1, 1]),
            np.diag([1, -1, 1]),
        ])


def test_request_registry_has_one_canonical_manifest_representation():
    job = SP.Spectroscopy(None, backend="qspace", workdir="spectroscopy")
    job.add_raman_polarized([1, 0, 0], [0, 1, 0], name="raman_xy")
    job.add_raman_unpolarized(name="raman_powder")
    job.add_ir_polarized([2, 0, 0], name="ir_x")
    job.add_ir_unpolarized(name="ir_powder")

    manifest = job.manifest()
    assert manifest["schema_version"] == 3
    assert manifest["backend"] == "qspace"
    assert [request["name"] for request in manifest["requests"]] == [
        "raman_xy", "raman_powder", "ir_x", "ir_powder"]
    assert len(manifest["requests"][1]["perturbations"]) == 7
    assert manifest["requests"][1]["weights"] == [45, 7, 7, 7, 7, 7, 7]
    assert len(manifest["requests"][3]["perturbations"]) == 3

    with pytest.raises(ValueError, match="already exists"):
        job.add_ir_polarized([1, 0, 0], name="ir_x")
    with pytest.raises(ValueError, match="letters"):
        job.add_ir_unpolarized(name="bad/name")
    assert job.requests["raman_powder"].observable == "raman_unpolarized"


def test_per_request_effective_charges_are_immutable():
    charges = np.arange(18, dtype=float).reshape(2, 3, 3)
    job = SP.Spectroscopy(None)
    job.add_ir_unpolarized("ir", effective_charges=charges)
    charges[:] = 0
    stored = np.asarray(job.requests["ir"].source)
    assert np.any(stored != 0)
