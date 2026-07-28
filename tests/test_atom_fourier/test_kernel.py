"""Geometry and exact-reconstruction tests for atom-Fourier interpolation."""

import itertools
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JULIA_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "test_interpolation"))

import _toy_chain as chain
import _toy_crystal3d as crystal3d

try:
    import tdscha.QSpaceAtomFourier as AF
    import tdscha.QSpaceInterpolation as interpolation
    import tdscha.QSpaceLanczos as QL
    _AVAILABLE = QL.__JULIA_EXT__
except Exception:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE, reason="QSpaceLanczos/Julia not available")


def _fractional_points(indices, mesh):
    return np.asarray(indices, dtype=float) / np.asarray(mesh, dtype=float)


def _negated_index(index, indices, mesh, lookup):
    key = tuple((-np.asarray(indices[index])) % np.asarray(mesh))
    return lookup[key]


def _pair_images(lanczos, atom_a, atom_b):
    cell = np.asarray(lanczos.uci_structure.unit_cell, dtype=float)
    tau = np.linalg.solve(cell.T, lanczos.uci_structure.coords.T).T
    return lanczos._metric_alias_images(
        tau[atom_a] - tau[atom_b],
        lanczos.coarse_mesh, cell @ cell.T)


def _pair_function(points, images, coefficients):
    values = np.zeros(len(points), dtype=np.complex128)
    for coefficient, entries in zip(coefficients, images.values()):
        for lattice_vector, weight in entries:
            values += coefficient * weight * np.exp(
                2j * np.pi * (
                    points @ np.asarray(lattice_vector, dtype=float)))
    return values


@pytest.fixture(scope="module")
def nonorthogonal_system():
    dyn = crystal3d.build_dyn((2, 2, 2))
    ensemble = crystal3d.make_ensemble(dyn, N=24, seed=3)
    lanczos = AF.QSpaceAtomFourierLanczos(
        ensemble, fine_mesh=(4, 4, 4), allow_unstable=True)
    return dyn, ensemble, lanczos


@pytest.mark.parametrize(
    "mesh", [(1, 1, 1), (1, 3, 6), np.array([2, 4, 3])])
def test_validate_mesh(mesh):
    assert np.array_equal(
        interpolation.validate_mesh(mesh), np.asarray(mesh, dtype=int))


@pytest.mark.parametrize(
    "mesh", [None, (), (1, 2), (1, 2, 3, 4), (0, 2, 2),
             (-1, 2, 2), (1.5, 2, 2), (True, 2, 2)])
def test_validate_mesh_rejects_invalid_values(mesh):
    with pytest.raises(ValueError, match="positive integer|three entries"):
        interpolation.validate_mesh(mesh)


def test_generate_anisotropic_odd_even_mesh():
    dyn = chain.build_dyn(3)
    q_points, indices = interpolation.generate_fine_mesh(
        dyn.structure, (1, 3, 4))
    assert q_points.shape == (12, 3)
    assert indices.shape == (12, 3)
    assert tuple(indices[0]) == (0, 0, 0)
    lookup = interpolation.build_q_index_lookup(
        q_points, dyn.structure, (1, 3, 4))
    assert len(lookup) == 12
    for iq, q in enumerate(q_points):
        assert lookup[
            interpolation.mesh_key(
                q, dyn.structure, (1, 3, 4))] == iq


def test_mesh_helpers_reject_invalid_q_arrays():
    dyn = chain.build_dyn(3)
    with pytest.raises(ValueError, match="shape"):
        interpolation.mesh_key(
            np.zeros(2), dyn.structure, (1, 1, 3))
    with pytest.raises(ValueError, match="shape"):
        interpolation.build_q_index_lookup(
            np.zeros(3), dyn.structure, (1, 1, 3))
    with pytest.raises(ValueError, match="at least one"):
        interpolation.interpolate_dyn_fine(
            dyn, np.empty((0, 3)))


def test_constructor_rejects_noncommensurate_fine_mesh():
    dyn = chain.build_dyn(3)
    ensemble = chain.make_ensemble(
        dyn, 300.0, 12, seed=2, g3=0.2, g4=0.3)
    with pytest.raises(ValueError, match="integer multiple"):
        AF.QSpaceAtomFourierLanczos(
            ensemble, fine_mesh=(1, 1, 4))

    with pytest.raises(ValueError, match="w_min_guard"):
        AF.QSpaceAtomFourierLanczos(
            ensemble, fine_mesh=(1, 1, 6), w_min_guard=0)


def test_metric_images_match_brute_force():
    cell = 3.0 * np.array(
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
    metric = cell @ cell.T
    mesh = np.array([2, 2, 2])
    displacement = np.array([0.5, 0.5, 0.0])
    got = AF.QSpaceAtomFourierLanczos._metric_alias_images(
        displacement, mesh, metric)

    for key in itertools.product(*(range(n) for n in mesh)):
        candidates = []
        for shift in itertools.product(range(-3, 4), repeat=3):
            vector = np.array(key) + mesh * np.array(shift)
            delta = vector - displacement
            candidates.append(
                (float(delta @ metric @ delta), tuple(vector)))
        minimum = min(item[0] for item in candidates)
        expected = {
            vector for distance, vector in candidates
            if abs(distance - minimum) < 1e-7}
        assert {vector for vector, _ in got[key]} == expected
        assert sum(weight for _, weight in got[key]) == pytest.approx(1.0)

    assert max(len(images) for images in got.values()) == 4


def test_metric_search_expands_for_highly_skewed_cell():
    """Minimum images outside the old fixed ±2 search are still found."""
    cell = np.array(
        [[1.0, 0.0, 0.0], [10.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    got = AF.QSpaceAtomFourierLanczos._metric_alias_images(
        np.array([0.0, 0.5, 0.0]), (1, 1, 1), cell @ cell.T)
    images = {vector for vector, _ in got[(0, 0, 0)]}
    assert images == {(5, 0, 0), (-5, 1, 0)}
    assert sum(weight for _, weight in got[(0, 0, 0)]) == pytest.approx(1.0)


def test_kernel_is_cardinal_and_mirror_symmetric(nonorthogonal_system):
    _, _, lanczos = nonorthogonal_system
    kernel = lanczos._atom_fourier_kernel
    n_atoms = lanczos.uci_structure.N_atoms

    for coarse_iq, fine_iq in enumerate(lanczos._fine_of_coarse):
        for coarse_jq in range(lanczos.cn_q):
            expected = 1.0 if coarse_iq == coarse_jq else 0.0
            assert np.allclose(
                kernel[fine_iq, coarse_jq],
                np.full((n_atoms, n_atoms), expected), atol=1e-12)

    for fine_iq in range(lanczos.n_q):
        minus_fine = _negated_index(
            fine_iq, lanczos._fine_idx, lanczos.fine_mesh,
            lanczos._q_lookup)
        for coarse_iq in range(lanczos.cn_q):
            minus_coarse = _negated_index(
                coarse_iq, lanczos._coarse_idx, lanczos.coarse_mesh,
                lanczos._coarse_lookup)
            assert np.allclose(
                kernel[fine_iq, coarse_iq],
                kernel[minus_fine, minus_coarse].T, atol=1e-12)


def test_d3_pair_interpolation_is_exact(nonorthogonal_system):
    """Every representable third-order pair harmonic is reconstructed."""
    _, _, lanczos = nonorthogonal_system
    coarse_q = _fractional_points(
        lanczos._coarse_idx, lanczos.coarse_mesh)
    fine_q = _fractional_points(
        lanczos._fine_idx, lanczos.fine_mesh)
    rng = np.random.default_rng(8)

    for atom_a in range(lanczos.uci_structure.N_atoms):
        for atom_b in range(lanczos.uci_structure.N_atoms):
            images = _pair_images(lanczos, atom_a, atom_b)
            coefficients = (
                rng.normal(size=len(images))
                + 1j * rng.normal(size=len(images)))
            coarse = _pair_function(coarse_q, images, coefficients)
            exact = _pair_function(fine_q, images, coefficients)
            reconstructed = (
                lanczos._atom_fourier_kernel[:, :, atom_a, atom_b]
                @ coarse)
            assert np.allclose(reconstructed, exact, atol=2e-12)


def test_d4_pair_product_interpolation_is_exact(nonorthogonal_system):
    """The d4 continuation is the tensor product of two exact pair maps."""
    _, _, lanczos = nonorthogonal_system
    coarse_q = _fractional_points(
        lanczos._coarse_idx, lanczos.coarse_mesh)
    fine_q = _fractional_points(
        lanczos._fine_idx, lanczos.fine_mesh)
    rng = np.random.default_rng(9)
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for pair_index, (atom_a, atom_b) in enumerate(pairs):
        images_ab = _pair_images(lanczos, atom_a, atom_b)
        coeff_ab = (
            rng.normal(size=len(images_ab))
            + 1j * rng.normal(size=len(images_ab)))
        coarse_ab = _pair_function(coarse_q, images_ab, coeff_ab)
        fine_ab = _pair_function(fine_q, images_ab, coeff_ab)
        kernel_ab = lanczos._atom_fourier_kernel[
            :, :, atom_a, atom_b]

        atom_c, atom_d = pairs[(pair_index + 1) % len(pairs)]
        images_cd = _pair_images(lanczos, atom_c, atom_d)
        coeff_cd = (
            rng.normal(size=len(images_cd))
            + 1j * rng.normal(size=len(images_cd)))
        coarse_cd = _pair_function(coarse_q, images_cd, coeff_cd)
        fine_cd = _pair_function(fine_q, images_cd, coeff_cd)
        kernel_cd = lanczos._atom_fourier_kernel[
            :, :, atom_c, atom_d]

        coarse_d4 = np.outer(coarse_ab, coarse_cd)
        reconstructed = kernel_ab @ coarse_d4 @ kernel_cd.T
        assert np.allclose(
            reconstructed, np.outer(fine_ab, fine_cd), atol=5e-12)
