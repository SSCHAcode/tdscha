"""High-level, backend-neutral definitions for optical spectroscopy.

This module is intentionally independent from the Lanczos implementations.
It contains the physical definitions and finite-group algebra that are shared
by real-space, q-space, and interpolation backends.  Execution and
checkpointing are delegated to one private workflow layer without duplicating
the Lanczos recursion or continued-fraction analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Tuple

import numpy as np


_DEFAULT_TOLERANCE = 1e-10
_RAMAN_SCHEMA_VERSION = 1
_SPECTROSCOPY_SCHEMA_VERSION = 3


def _finite_array(value, shape, name):
    """Return a copied float array after common validation."""
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(
            "{} must have shape {}, got {}".format(name, shape, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return np.array(array, copy=True)


def _validate_lo_to_split(value):
    """Return a JSON-stable LO--TO specification shared by all backends."""
    if value is None:
        return None
    if isinstance(value, str):
        if value != "random":
            raise ValueError("lo_to_split must be None, 'random', or a vector")
        return value
    direction = _finite_array(value, (3,), "lo_to_split")
    if np.linalg.norm(direction) <= _DEFAULT_TOLERANCE:
        raise ValueError("lo_to_split direction must not be zero")
    return [float(component) for component in direction]


def _same_lo_to_split(left, right):
    try:
        return _validate_lo_to_split(left) == _validate_lo_to_split(right)
    except ValueError:
        return False


def _tuple_vector(value, name="vector", normalize=False):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("{} must be a non-empty one-dimensional array".format(
            name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    norm = np.linalg.norm(array)
    if norm <= _DEFAULT_TOLERANCE:
        raise ValueError("{} must not be zero".format(name))
    if normalize:
        array = array / norm
    return tuple(float(item) for item in array)


def _tuple_matrix3(value, name, symmetric=False):
    array = _finite_array(value, (3, 3), name)
    if symmetric and not np.allclose(
            array, array.T, atol=_DEFAULT_TOLERANCE, rtol=0):
        raise ValueError("{} must be symmetric".format(name))
    return tuple(tuple(float(item) for item in row) for row in array)


class PerturbationKind(str, Enum):
    """Kinds of stable equilibrium optical perturbations."""

    RAMAN = "raman"
    IR = "ir"
    CARTESIAN = "cartesian"


@dataclass(frozen=True)
class RamanComponent:
    """One canonical component of the Placzek unpolarized invariant."""

    index: int
    label: str
    raw_coefficients: Tuple[Tuple[float, ...], ...]
    normalized_scale: float
    normalized_weight: float
    raw_weight: float

    def coefficients(self, convention="normalized"):
        """Return a fresh 3x3 coefficient tensor for the requested convention."""
        coefficients = np.asarray(self.raw_coefficients, dtype=float)
        if convention == "raw":
            return np.array(coefficients, copy=True)
        if convention == "normalized":
            return coefficients * self.normalized_scale
        raise ValueError(
            "Raman convention must be 'normalized' or 'raw', got {!r}".format(
                convention))

    def weight(self, convention="normalized"):
        """Return the response weight matching ``coefficients(convention)``."""
        if convention == "normalized":
            return self.normalized_weight
        if convention == "raw":
            return self.raw_weight
        raise ValueError(
            "Raman convention must be 'normalized' or 'raw', got {!r}".format(
                convention))


def _symmetric_component(i, j, value=1.0):
    """Coefficient tensor whose contraction selects one symmetric component."""
    result = np.zeros((3, 3), dtype=float)
    if i == j:
        result[i, j] = value
    else:
        result[i, j] = value / 2
        result[j, i] = value / 2
    return _tuple_matrix3(result, "Raman component", symmetric=True)


def _diagonal_component(values):
    return _tuple_matrix3(np.diag(values), "Raman component", symmetric=True)


RAMAN_COMPONENTS = (
    RamanComponent(0, "trace", _diagonal_component((1, 1, 1)),
                   1 / 3, 45, 5),
    RamanComponent(1, "xx_minus_yy", _diagonal_component((1, -1, 0)),
                   1 / np.sqrt(2), 7, 7 / 2),
    RamanComponent(2, "xx_minus_zz", _diagonal_component((1, 0, -1)),
                   1 / np.sqrt(2), 7, 7 / 2),
    RamanComponent(3, "yy_minus_zz", _diagonal_component((0, 1, -1)),
                   1 / np.sqrt(2), 7, 7 / 2),
    RamanComponent(4, "xy", _symmetric_component(0, 1),
                   np.sqrt(3), 7, 21),
    RamanComponent(5, "xz", _symmetric_component(0, 2),
                   np.sqrt(3), 7, 21),
    RamanComponent(6, "yz", _symmetric_component(1, 2),
                   np.sqrt(3), 7, 21),
)


def get_raman_component(index):
    """Return one canonical Raman component by stable integer index."""
    if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < 7:
        raise ValueError("Raman component index must be an integer from 0 to 6")
    return RAMAN_COMPONENTS[int(index)]


def get_unpolarized_raman_weights(convention="normalized"):
    """Return all seven unpolarized Raman response weights."""
    return np.array(
        [component.weight(convention) for component in RAMAN_COMPONENTS],
        dtype=float)


def raman_coefficients_from_polarizations(
        incoming, outgoing, symmetric=True):
    """Return the tensor selecting a polarized Raman contraction.

    The stable non-resonant API uses ``symmetric=True``.  The optional raw
    outer product exists only so legacy Lanczos calls retain their exact
    incoming/outgoing convention for a non-symmetric input tensor.
    """
    incoming = _finite_array(incoming, (3,), "incoming polarization")
    outgoing = _finite_array(outgoing, (3,), "outgoing polarization")
    if np.linalg.norm(incoming) <= _DEFAULT_TOLERANCE:
        raise ValueError("incoming polarization must not be zero")
    if np.linalg.norm(outgoing) <= _DEFAULT_TOLERANCE:
        raise ValueError("outgoing polarization must not be zero")
    coefficients = np.outer(incoming, outgoing)
    if symmetric:
        coefficients = (coefficients + coefficients.T) / 2
    return coefficients


def build_raman_vector(raman_tensor, coefficients):
    """Contract a Raman derivative with one symmetric optical tensor.

    ``raman_tensor`` must start with the two optical Cartesian axes and may
    have any remaining atomic-coordinate shape.  The returned array retains
    those remaining axes.
    """
    raman_tensor = np.asarray(raman_tensor)
    if raman_tensor.ndim < 3 or raman_tensor.shape[:2] != (3, 3):
        raise ValueError(
            "raman_tensor must have shape (3, 3, ...), got {}".format(
                raman_tensor.shape))
    if not np.all(np.isfinite(raman_tensor)):
        raise ValueError("raman_tensor must contain only finite values")
    coefficients = _finite_array(coefficients, (3, 3),
                                 "Raman coefficient tensor")
    return np.einsum("ab,ab...->...", coefficients, raman_tensor)


def build_ir_vector(effective_charges, direction):
    """Contract equilibrium effective charges with an electric-field direction."""
    effective_charges = np.asarray(effective_charges)
    if effective_charges.ndim != 3 or effective_charges.shape[1:] != (3, 3):
        raise ValueError(
            "effective_charges must have shape (n_atoms, 3, 3), got {}".format(
                effective_charges.shape))
    if not np.all(np.isfinite(effective_charges)):
        raise ValueError("effective_charges must contain only finite values")
    direction = _finite_array(direction, (3,), "IR direction")
    return np.einsum("abc,b->ac", effective_charges, direction).ravel()


@dataclass(frozen=True)
class RamanTensorPerturbation:
    """A stable one-phonon Raman perturbation specification."""

    coefficients: Tuple[Tuple[float, ...], ...]

    def __init__(self, coefficients):
        object.__setattr__(
            self, "coefficients",
            _tuple_matrix3(coefficients, "Raman coefficient tensor",
                           symmetric=True))
        if np.linalg.norm(self.as_array()) <= _DEFAULT_TOLERANCE:
            raise ValueError("Raman coefficient tensor must not be zero")

    @property
    def kind(self):
        return PerturbationKind.RAMAN

    def as_array(self):
        return np.array(self.coefficients, dtype=float)

    def transformed(self, rotation):
        rotation = _finite_array(rotation, (3, 3), "rotation")
        coefficients = self.as_array()
        return RamanTensorPerturbation(
            rotation @ coefficients @ rotation.T)


@dataclass(frozen=True)
class IRPolarizationPerturbation:
    """A stable one-phonon IR perturbation with a unit direction."""

    direction: Tuple[float, ...]

    def __init__(self, direction):
        direction = _finite_array(direction, (3,), "IR direction")
        object.__setattr__(
            self, "direction", _tuple_vector(
                direction, name="IR direction", normalize=True))

    @property
    def kind(self):
        return PerturbationKind.IR

    def as_array(self):
        return np.array(self.direction, dtype=float)

    def transformed(self, rotation):
        rotation = _finite_array(rotation, (3, 3), "rotation")
        return IRPolarizationPerturbation(rotation @ self.as_array())


@dataclass(frozen=True)
class CartesianPerturbation:
    """An explicit nonzero unit-cell Cartesian perturbation vector."""

    vector: Tuple[float, ...]

    def __init__(self, vector):
        object.__setattr__(self, "vector", _tuple_vector(vector))

    @property
    def kind(self):
        return PerturbationKind.CARTESIAN

    def as_array(self):
        return np.array(self.vector, dtype=float)


@dataclass(frozen=True)
class SymmetryGroup:
    """A validated finite group and its multiplication table."""

    matrices: Tuple[Tuple[Tuple[float, ...], ...], ...]
    multiplication_table: Tuple[Tuple[int, ...], ...]
    identity_index: int

    @classmethod
    def from_matrices(cls, matrices, tolerance=_DEFAULT_TOLERANCE):
        arrays = tuple(
            _finite_array(matrix, (3, 3), "symmetry matrix")
            for matrix in matrices)
        if not arrays:
            raise ValueError("at least one symmetry matrix is required")
        for matrix in arrays:
            if not np.allclose(matrix.T @ matrix, np.eye(3),
                               atol=tolerance, rtol=0):
                raise ValueError("symmetry matrices must be orthogonal")

        def find_index(target):
            matches = [
                index for index, candidate in enumerate(arrays)
                if np.allclose(target, candidate, atol=tolerance, rtol=0)]
            if len(matches) != 1:
                raise ValueError(
                    "symmetry matrices must be unique and closed under "
                    "multiplication")
            return matches[0]

        identity_index = find_index(np.eye(3))
        multiplication = tuple(tuple(
            find_index(left @ right) for right in arrays) for left in arrays)
        immutable = tuple(_tuple_matrix3(matrix, "symmetry matrix")
                          for matrix in arrays)
        return cls(immutable, multiplication, identity_index)

    def __len__(self):
        return len(self.matrices)

    def matrix(self, index):
        return np.array(self.matrices[index], dtype=float)

    def left_cosets(self, subgroup):
        """Partition the group into left cosets of a validated subgroup."""
        subgroup = tuple(sorted(set(int(index) for index in subgroup)))
        if not subgroup or self.identity_index not in subgroup:
            raise ValueError("subgroup must contain the group identity")
        if any(index < 0 or index >= len(self) for index in subgroup):
            raise ValueError("subgroup contains an invalid group index")
        subgroup_set = set(subgroup)
        for left in subgroup:
            for right in subgroup:
                if self.multiplication_table[left][right] not in subgroup_set:
                    raise ValueError("indices do not form a subgroup")

        unseen = set(range(len(self)))
        cosets = []
        while unseen:
            representative = min(unseen)
            coset = tuple(sorted(
                self.multiplication_table[representative][member]
                for member in subgroup))
            cosets.append(coset)
            unseen.difference_update(coset)
        return tuple(cosets)

    def right_cosets(self, subgroup):
        """Partition the group into right cosets ``H g``.

        These are the cosets used by the spectroscopy ensemble reduction:
        the stabilizer projector accounts for ``H`` while one transformed
        ensemble representative is evaluated for each ``H g``.
        """
        subgroup = tuple(sorted(set(int(index) for index in subgroup)))
        # Reuse the subgroup validation in left_cosets.
        self.left_cosets(subgroup)
        unseen = set(range(len(self)))
        cosets = []
        while unseen:
            representative = min(unseen)
            coset = tuple(sorted(
                self.multiplication_table[member][representative]
                for member in subgroup))
            cosets.append(coset)
            unseen.difference_update(coset)
        return tuple(cosets)


def find_atom_permutation(structure, rotation, translation, tolerance=1e-5):
    """Return the atom permutation for one Cartesian space-group operation."""
    rotation = _finite_array(rotation, (3, 3), "rotation")
    translation = _finite_array(translation, (3,), "translation")
    lattice = np.asarray(structure.unit_cell, dtype=float).T
    inverse_lattice = np.linalg.inv(lattice)
    atom_types = np.asarray(structure.get_atomic_types())
    permutation = np.full(structure.N_atoms, -1, dtype=int)
    for atom in range(structure.N_atoms):
        mapped = rotation @ structure.coords[atom] + translation
        for candidate in range(structure.N_atoms):
            if atom_types[candidate] != atom_types[atom]:
                continue
            difference = mapped - structure.coords[candidate]
            fractional = inverse_lattice @ difference
            fractional -= np.round(fractional)
            if np.linalg.norm(lattice @ fractional) <= tolerance:
                permutation[atom] = candidate
                break
        if permutation[atom] < 0:
            raise ValueError(
                "Could not map atom {} under the supplied symmetry".format(
                    atom))
    if len(set(permutation.tolist())) != structure.N_atoms:
        raise ValueError("Symmetry atom mapping is not a permutation")
    return permutation


def get_gamma_symmetry_representation(structure, tolerance=1e-8,
                                      symprec=1e-5, supercell=None):
    """Build mesh-compatible Gamma point-group representations.

    When ``supercell`` is supplied, operations which do not map that
    supercell translation lattice onto itself are excluded.  This prevents
    an anisotropic finite sampling mesh from spuriously identifying optical
    perturbations that are equivalent only in the infinite crystal.
    """
    try:
        import spglib
    except ImportError as error:
        raise ImportError(
            "spglib is required for spectroscopy symmetry reduction") from error

    symmetry = spglib.get_symmetry(
        structure.get_spglib_cell(), symprec=symprec)
    if symmetry is None:
        raise ValueError("spglib could not determine structure symmetries")
    rotations_fractional = symmetry["rotations"]
    translations_fractional = symmetry["translations"]
    lattice = np.asarray(structure.unit_cell, dtype=float).T
    inverse_lattice = np.linalg.inv(lattice)

    if supercell is None:
        supercell_matrix = np.eye(3, dtype=float)
    else:
        supercell = np.asarray(supercell)
        if supercell.shape == (3,):
            supercell_matrix = np.diag(supercell.astype(float))
        elif supercell.shape == (3, 3):
            supercell_matrix = supercell.astype(float)
        else:
            raise ValueError("supercell must have shape (3,) or (3, 3)")
        if abs(np.linalg.det(supercell_matrix)) <= tolerance:
            raise ValueError("supercell matrix must be invertible")
    inverse_supercell = np.linalg.inv(supercell_matrix)

    unique = {}
    for index, rotation in enumerate(rotations_fractional):
        mesh_rotation = (
            inverse_supercell @ rotation.astype(float) @ supercell_matrix)
        if not np.allclose(mesh_rotation, np.rint(mesh_rotation),
                           atol=max(tolerance, 1e-8), rtol=0):
            continue
        unique.setdefault(rotation.tobytes(), index)
    indices = tuple(unique.values())
    rotations = tuple(
        lattice @ rotations_fractional[index].astype(float) @ inverse_lattice
        for index in indices)
    translations = tuple(
        lattice @ translations_fractional[index] for index in indices)
    group = SymmetryGroup.from_matrices(rotations, tolerance=tolerance)

    dimension = 3 * structure.N_atoms
    representations = []
    atom_tolerance = max(symprec * 10, tolerance * 10)
    for rotation, translation in zip(rotations, translations):
        permutation = find_atom_permutation(
            structure, rotation, translation, tolerance=atom_tolerance)
        representation = np.zeros((dimension, dimension), dtype=float)
        for atom, mapped_atom in enumerate(permutation):
            representation[3 * mapped_atom:3 * mapped_atom + 3,
                           3 * atom:3 * atom + 3] = rotation
        representations.append(representation)

    # Validate the atom-space matrices against the same multiplication table.
    for left in range(len(group)):
        for right in range(len(group)):
            product = group.multiplication_table[left][right]
            if not np.allclose(
                    representations[left] @ representations[right],
                    representations[product], atol=max(tolerance, 1e-8),
                    rtol=0):
                raise ValueError(
                    "Gamma atom-space symmetry matrices do not form the "
                    "same representation as the point group")
    return group, tuple(representations)


def _real_phase(reference, candidate, tolerance):
    """Return +1/-1 when two nonzero real vectors differ only by that phase."""
    reference = np.asarray(reference, dtype=float).ravel()
    candidate = np.asarray(candidate, dtype=float).ravel()
    if reference.shape != candidate.shape:
        return None
    scale = max(np.linalg.norm(reference), np.linalg.norm(candidate))
    if scale <= np.finfo(float).tiny:
        return None
    for phase in (1.0, -1.0):
        if np.linalg.norm(candidate - phase * reference) <= tolerance * scale:
            return phase
    return None


@dataclass(frozen=True)
class PerturbationOrbit:
    """Symmetry reconstruction data for one representative perturbation."""

    representative: int
    members: Tuple[int, ...]
    operations: Tuple[int, ...]
    phases: Tuple[float, ...]
    stabilizer: Tuple[int, ...]
    characters: Tuple[float, ...]
    left_cosets: Tuple[Tuple[int, ...], ...]
    right_cosets: Tuple[Tuple[int, ...], ...]


def find_perturbation_orbits(vectors, representations, group,
                             tolerance=_DEFAULT_TOLERANCE):
    """Partition real perturbation vectors into sign-aware symmetry orbits.

    Parameters
    ----------
    vectors : sequence of one-dimensional arrays
        Requested perturbations in a common vector representation.
    representations : sequence of square arrays
        Representation matrix corresponding to each element of ``group``.
    group : SymmetryGroup
        Multiplication information for the same ordered group elements.

    Notes
    -----
    The sign is retained in the reconstruction metadata.  This is sufficient
    for diagonal response functions; future cross-response assembly must use
    the stored sign rather than discarding it.
    """
    vectors = tuple(np.asarray(vector, dtype=float).ravel()
                    for vector in vectors)
    if not vectors:
        return ()
    dimension = vectors[0].size
    if dimension == 0:
        raise ValueError("perturbation vectors must not be empty")
    for vector in vectors:
        if vector.size != dimension or not np.all(np.isfinite(vector)):
            raise ValueError(
                "all perturbation vectors must be finite and have one size")
        if np.linalg.norm(vector) <= np.finfo(float).tiny:
            raise ValueError("perturbation vectors must not be zero")

    representations = tuple(np.asarray(item, dtype=float)
                            for item in representations)
    if len(representations) != len(group):
        raise ValueError(
            "one representation matrix is required per group element")
    for representation in representations:
        if representation.shape != (dimension, dimension):
            raise ValueError(
                "representation matrices must have shape ({0}, {0})".format(
                    dimension))
    for left in range(len(group)):
        for right in range(len(group)):
            product = group.multiplication_table[left][right]
            if not np.allclose(
                    representations[left] @ representations[right],
                    representations[product], atol=tolerance, rtol=0):
                raise ValueError(
                    "representation matrices do not follow the supplied "
                    "group multiplication table")

    remaining = set(range(len(vectors)))
    orbits = []
    while remaining:
        representative = min(remaining)
        reference = vectors[representative]
        members = []
        operations = []
        phases = []
        for member in sorted(remaining):
            for operation, representation in enumerate(representations):
                phase = _real_phase(
                    vectors[member], representation @ reference, tolerance)
                if phase is not None:
                    members.append(member)
                    operations.append(operation)
                    phases.append(phase)
                    break

        stabilizer = []
        characters = []
        for operation, representation in enumerate(representations):
            character = _real_phase(
                reference, representation @ reference, tolerance)
            if character is not None:
                stabilizer.append(operation)
                characters.append(character)

        left_cosets = group.left_cosets(stabilizer)
        right_cosets = group.right_cosets(stabilizer)
        orbit = PerturbationOrbit(
            representative=representative,
            members=tuple(members),
            operations=tuple(operations),
            phases=tuple(phases),
            stabilizer=tuple(stabilizer),
            characters=tuple(characters),
            left_cosets=left_cosets,
            right_cosets=right_cosets)
        orbits.append(orbit)
        remaining.difference_update(members)
    return tuple(orbits)


def vector_representations_for_ir(group):
    """Return the ordinary Cartesian-vector representation of a group."""
    return tuple(group.matrix(index) for index in range(len(group)))


def vector_representations_for_symmetric_raman(group):
    """Return 6x6 representations acting on symmetric Raman tensors."""
    basis = []
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        tensor = np.zeros((3, 3), dtype=float)
        tensor[i, j] = 1
        tensor[j, i] = 1
        basis.append(tensor)
    representations = []
    for operation in range(len(group)):
        rotation = group.matrix(operation)
        columns = []
        for tensor in basis:
            transformed = rotation @ tensor @ rotation.T
            columns.append(np.array([
                transformed[0, 0], transformed[1, 1], transformed[2, 2],
                transformed[0, 1], transformed[0, 2],
                transformed[1, 2]], dtype=float))
        representations.append(np.column_stack(columns))
    return tuple(representations)


def symmetric_raman_vector(coefficients):
    """Encode a symmetric 3x3 coefficient tensor in the shared 6-vector basis."""
    coefficients = _finite_array(
        coefficients, (3, 3), "Raman coefficient tensor")
    if not np.allclose(coefficients, coefficients.T,
                       atol=_DEFAULT_TOLERANCE, rtol=0):
        raise ValueError("Raman coefficient tensor must be symmetric")
    return np.array([
        coefficients[0, 0], coefficients[1, 1], coefficients[2, 2],
        coefficients[0, 1], coefficients[0, 2], coefficients[1, 2]],
        dtype=float)


def _load_phonons(source, nqirr, name):
    """Return a CellConstructor dynamical matrix from a path or an object."""
    if source is None:
        return None
    if isinstance(source, (str, os.PathLike)):
        if nqirr is None:
            raise ValueError(
                "{} was given as a file prefix, so the number of irreducible "
                "q-points must be given too".format(name))
        if (not isinstance(nqirr, (int, np.integer))
                or isinstance(nqirr, bool) or int(nqirr) < 1):
            raise ValueError(
                "the number of irreducible q-points for {} must be a "
                "positive integer".format(name))
        import cellconstructor.Phonons

        return cellconstructor.Phonons.Phonons(
            os.fspath(source), int(nqirr))
    if nqirr is not None:
        raise ValueError(
            "{} is already a dynamical matrix object, so its number of "
            "irreducible q-points must not be given".format(name))
    for attribute in ("structure", "dynmats", "GetSupercell"):
        if not hasattr(source, attribute):
            raise TypeError(
                "{} must be a file prefix or a CellConstructor Phonons "
                "object, got {}".format(name, type(source).__name__))
    return source


@dataclass(frozen=True)
class EnsembleSource:
    """Where a stochastic ensemble lives on disk, and how to reweight it.

    :class:`Spectroscopy` takes one of these instead of a loaded ensemble so
    that the configurations are read **once, by the MPI master**, and
    scattered: every rank keeps only its ``N / n_procs`` slice of the
    Bloch-transformed displacements and forces.  Handing a fully loaded
    ``sscha.Ensemble.Ensemble`` to every rank instead replicates the
    configurations, which is what makes a large ensemble impossible to run.

    Nothing here is heavy.  The dynamical matrices are small and are held on
    every rank -- the run plan, the symmetry analysis, the fingerprint, and
    the spectral analysis all need them.  Only the configurations are
    distributed.

    Parameters
    ----------
    data_dir : str or os.PathLike
        Directory holding the binary ensemble files.
    population : int
        Population identifier of the ensemble inside ``data_dir``.
    dyn : str, os.PathLike, or CC.Phonons.Phonons
        The dynamical matrix the ensemble was *generated* with, either as a
        CellConstructor file prefix or as an already loaded object.
    T : float
        Temperature in Kelvin at which the ensemble was generated.
    nqirr : int, optional
        Number of irreducible q-points; required when ``dyn`` is a prefix.
    n_configs : int, optional
        Read only the first ``n_configs`` configurations.  ``None`` reads
        every configuration in the population.
    final_dyn : str, os.PathLike, or CC.Phonons.Phonons, optional
        The converged solution.  When given, the ensemble is reweighted onto
        it and it becomes the reference dynamical matrix of the calculation:
        the Raman tensor, the effective charges, and the mode basis all come
        from it.  Production runs should always set it.
    final_nqirr : int, optional
        Number of irreducible q-points; required when ``final_dyn`` is a
        prefix.
    final_T : float, optional
        Temperature of the reweighted ensemble.  Defaults to ``T``.
    """

    data_dir: str
    population: int
    dyn: object
    T: float
    nqirr: Optional[int] = None
    n_configs: Optional[int] = None
    final_dyn: object = None
    final_nqirr: Optional[int] = None
    final_T: Optional[float] = None

    def __post_init__(self):
        data_dir = os.fspath(self.data_dir)
        if not data_dir:
            raise ValueError("data_dir must be a non-empty path")
        if not os.path.isdir(data_dir):
            raise ValueError(
                "the ensemble directory {!r} does not exist".format(data_dir))
        object.__setattr__(self, "data_dir", data_dir)

        if (not isinstance(self.population, (int, np.integer))
                or isinstance(self.population, bool)):
            raise ValueError("population must be an integer")
        object.__setattr__(self, "population", int(self.population))

        temperature = float(self.T)
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("T must be a finite non-negative temperature")
        object.__setattr__(self, "T", temperature)

        if self.n_configs is not None:
            if (not isinstance(self.n_configs, (int, np.integer))
                    or isinstance(self.n_configs, bool)
                    or int(self.n_configs) < 1):
                raise ValueError("n_configs must be a positive integer")
            object.__setattr__(self, "n_configs", int(self.n_configs))

        if self.final_dyn is None and self.final_nqirr is not None:
            raise ValueError(
                "final_nqirr was given without a final_dyn")
        if self.final_dyn is None and self.final_T is not None:
            raise ValueError(
                "final_T was given without a final_dyn: the ensemble is not "
                "reweighted, so its temperature is T")

        object.__setattr__(self, "_generating_dyn", _load_phonons(
            self.dyn, self.nqirr, "dyn"))
        object.__setattr__(self, "_converged_dyn", _load_phonons(
            self.final_dyn, self.final_nqirr, "final_dyn"))

        if self.final_T is not None:
            final_temperature = float(self.final_T)
            if not np.isfinite(final_temperature) or final_temperature < 0:
                raise ValueError(
                    "final_T must be a finite non-negative temperature")
            object.__setattr__(self, "final_T", final_temperature)

    @property
    def generating_dyn(self):
        """The dynamical matrix the configurations were sampled from."""
        return self._generating_dyn

    @property
    def converged_dyn(self):
        """The reweighting target, or ``None`` when there is none."""
        return self._converged_dyn

    @property
    def reference_dyn(self):
        """The ensemble's ``current_dyn``: the converged one when reweighting."""
        if self._converged_dyn is not None:
            return self._converged_dyn
        return self._generating_dyn

    @property
    def reference_temperature(self):
        """The ensemble's ``current_T`` after any reweighting."""
        if self._converged_dyn is not None and self.final_T is not None:
            return float(self.final_T)
        return float(self.T)

    def load_ensemble(self):
        """Read the whole ensemble into memory on the calling process.

        This is the replicated path.  It is what ``backend="real"`` needs --
        the real-space Lanczos parallelizes over a replicated ensemble -- and
        it is the wrong thing to call from a q-space driver, which must go
        through the distributed loaders instead.
        """
        import sscha.Ensemble

        ensemble = sscha.Ensemble.Ensemble(self.generating_dyn, self.T)
        if self.n_configs is None:
            ensemble.load_bin(self.data_dir, self.population)
        else:
            ensemble.load_bin(self.data_dir, self.population,
                              n_configs=self.n_configs)
        if self.converged_dyn is not None:
            ensemble.update_weights(self.converged_dyn,
                                    self.reference_temperature)
        return ensemble

    def describe(self):
        """Return the JSON-stable identity checked when a run is resumed.

        Deliberately *not* the absolute path.  What makes two ensembles
        different is the population, how many of its configurations are read,
        and how they are reweighted -- none of which the fingerprint of the
        dynamical matrix can see, since the same matrix generates every
        population.  The directory enters only through its name, so that
        moving a finished calculation to another machine does not invalidate
        its checkpoint while ``pop1/`` and ``pop2/`` still do not collide.
        The full path is recorded separately, for provenance.
        """
        return {
            "directory_name": os.path.basename(
                os.path.normpath(os.path.abspath(self.data_dir))),
            "population": self.population,
            "n_configs": self.n_configs,
            "temperature": float(self.T),
            "reweighted": self.converged_dyn is not None,
            "reference_temperature": self.reference_temperature,
        }

    def provenance(self):
        """Return where this ensemble was read from, for the record only."""
        def path_of(value):
            if isinstance(value, (str, os.PathLike)):
                return os.fspath(value)
            return None

        return {
            "data_dir": os.path.abspath(self.data_dir),
            "dyn": path_of(self.dyn),
            "final_dyn": path_of(self.final_dyn),
        }


@dataclass(frozen=True)
class SpectroscopyRequest:
    """One named user request, possibly containing several perturbations."""

    name: str
    observable: str
    perturbations: Tuple[object, ...]
    weights: Tuple[float, ...]
    convention: Optional[str] = None
    source: Optional[Tuple] = None


class Spectroscopy:
    """Restartable polarized and unpolarized Raman/IR calculation driver.

    The ensemble is normally given as an :class:`EnsembleSource` -- where it
    lives on disk -- rather than as a loaded ``sscha.Ensemble.Ensemble``.
    Under ``mpirun`` the q-space backends then read the configurations on
    the master alone and scatter them, so each rank holds ``N / n_procs`` of
    them and a large ensemble does not have to fit in memory ``n_procs``
    times.  :meth:`from_ensemble_path` is the shorthand for the common case.

    A loaded ensemble is still accepted and still works; it is the
    replicated path, appropriate for small systems and for
    ``backend="real"``, whose real-space Lanczos parallelizes over a
    replicated ensemble by design.

    ``ignore_v3`` and ``ignore_v4`` control the anharmonic vertices in every
    backend.  ``lo_to_split`` uses one common convention: ``None`` disables
    the nonanalytic Gamma correction, ``"random"`` delegates the direction
    to CellConstructor, and a finite nonzero three-vector selects it.  The
    deprecated placement of these values inside ``backend_options`` remains
    accepted so existing scripts continue to run.
    """

    SUPPORTED_BACKENDS = ("real", "qspace", "atom_fourier")

    @classmethod
    def from_ensemble_path(cls, data_dir, population, dyn, T, nqirr=None,
                           n_configs=None, final_dyn=None, final_nqirr=None,
                           final_T=None, **options):
        """Build a driver that reads its ensemble from ``data_dir``.

        Shorthand for ``Spectroscopy(EnsembleSource(...), **options)``; see
        :class:`EnsembleSource` for the meaning of the ensemble arguments and
        :meth:`__init__` for the rest.
        """
        return cls(EnsembleSource(
            data_dir=data_dir, population=population, dyn=dyn, T=T,
            nqirr=nqirr, n_configs=n_configs, final_dyn=final_dyn,
            final_nqirr=final_nqirr, final_T=final_T), **options)

    def __init__(self, ensemble, backend="qspace", workdir="spectroscopy",
                 use_symmetries=True, symmetry_tolerance=1e-8,
                 ignore_v3=None, ignore_v4=None, lo_to_split=None,
                 backend_options=None):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                "backend must be one of {}, got {!r}".format(
                    self.SUPPORTED_BACKENDS, backend))
        try:
            workdir = os.fspath(workdir)
        except TypeError as error:
            raise ValueError("workdir must be a non-empty path") from error
        if not workdir:
            raise ValueError("workdir must be a non-empty path")
        if (ensemble is not None
                and not isinstance(ensemble, EnsembleSource)
                and not (hasattr(ensemble, "current_dyn")
                         and hasattr(ensemble, "current_T"))):
            raise TypeError(
                "ensemble must be an EnsembleSource, an "
                "sscha.Ensemble.Ensemble, or None for load-only analysis; "
                "got {}".format(type(ensemble).__name__))
        self.ensemble = ensemble
        self.backend = backend
        self.workdir = workdir
        self.use_symmetries = bool(use_symmetries)
        self.symmetry_tolerance = float(symmetry_tolerance)
        if self.symmetry_tolerance <= 0:
            raise ValueError("symmetry_tolerance must be positive")
        self.backend_options = dict(backend_options or {})

        # These physics switches used to be hidden in backend_options.  Keep
        # accepting that spelling so existing scripts remain restartable, but
        # expose one backend-independent public API from now on.
        def resolve_legacy_option(name, explicit, default):
            legacy = self.backend_options.pop(name, None)
            if explicit is not None and legacy is not None:
                if name == "lo_to_split":
                    same = _same_lo_to_split(explicit, legacy)
                else:
                    same = bool(explicit) == bool(legacy)
                if not same:
                    raise ValueError(
                        "Conflicting {!r} values were supplied explicitly "
                        "and through backend_options".format(name))
            value = explicit if explicit is not None else legacy
            return default if value is None else value

        self.ignore_v3 = bool(resolve_legacy_option(
            "ignore_v3", ignore_v3, False))
        self.ignore_v4 = bool(resolve_legacy_option(
            "ignore_v4", ignore_v4, False))
        self.lo_to_split = _validate_lo_to_split(resolve_legacy_option(
            "lo_to_split", lo_to_split, None))
        self._requests = {}
        self._run_specs = {}
        self._request_maps = {}
        self._results = {}
        self._manifest_data = None

    @property
    def requests(self):
        return MappingProxyType(self._requests.copy())

    @property
    def ensemble_source(self):
        """The :class:`EnsembleSource`, or ``None`` for a loaded ensemble."""
        return self.ensemble if isinstance(self.ensemble, EnsembleSource) \
            else None

    @property
    def reference_dyn(self):
        """The dynamical matrix every observable and symmetry is defined on.

        This is the ensemble's ``current_dyn``: the converged solution when
        the ensemble is reweighted onto one.  It carries the Raman tensor,
        the Born effective charges, and the electronic dielectric tensor.
        It is small and is held on every MPI rank, unlike the configurations.
        """
        if self.ensemble is None:
            return None
        if isinstance(self.ensemble, EnsembleSource):
            return self.ensemble.reference_dyn
        return self.ensemble.current_dyn

    @property
    def reference_temperature(self):
        """The ensemble's ``current_T`` after any reweighting."""
        if self.ensemble is None:
            return None
        if isinstance(self.ensemble, EnsembleSource):
            return self.ensemble.reference_temperature
        return float(self.ensemble.current_T)

    def _require_reference_dyn(self, what):
        dyn = self.reference_dyn
        if dyn is None:
            raise ValueError(
                "An ensemble source or a loaded ensemble is required to "
                "{}".format(what))
        return dyn

    def _add_request(self, request):
        if not isinstance(request.name, str) or not request.name:
            raise ValueError("request name must be a non-empty string")
        if any(character not in
               "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
               for character in request.name):
            raise ValueError(
                "request name may contain only letters, numbers, '_' and '-'")
        if request.name in self._requests:
            raise ValueError(
                "a spectroscopy request named {!r} already exists".format(
                    request.name))
        self._requests[request.name] = request
        return request.name

    def add_raman_polarized(self, incoming, outgoing, name):
        incoming = _finite_array(incoming, (3,), "incoming polarization")
        outgoing = _finite_array(outgoing, (3,), "outgoing polarization")
        if np.linalg.norm(incoming) <= _DEFAULT_TOLERANCE:
            raise ValueError("incoming polarization must not be zero")
        if np.linalg.norm(outgoing) <= _DEFAULT_TOLERANCE:
            raise ValueError("outgoing polarization must not be zero")
        incoming = incoming / np.linalg.norm(incoming)
        outgoing = outgoing / np.linalg.norm(outgoing)
        coefficients = raman_coefficients_from_polarizations(
            incoming, outgoing)
        return self.add_raman_tensor(coefficients, name=name)

    def add_raman_tensor(self, tensor, name):
        perturbation = RamanTensorPerturbation(tensor)
        return self._add_request(SpectroscopyRequest(
            name=name, observable="raman_polarized",
            perturbations=(perturbation,), weights=(1.0,)))

    def add_raman_unpolarized(self, name, convention="normalized"):
        perturbations = tuple(RamanTensorPerturbation(
            component.coefficients(convention))
            for component in RAMAN_COMPONENTS)
        weights = tuple(float(component.weight(convention))
                        for component in RAMAN_COMPONENTS)
        return self._add_request(SpectroscopyRequest(
            name=name, observable="raman_unpolarized",
            perturbations=perturbations, weights=weights,
            convention=convention))

    def add_raman_vector(self, vector, name):
        """Add an explicitly prepared unit-cell Raman Cartesian vector."""
        perturbation = CartesianPerturbation(vector)
        return self._add_request(SpectroscopyRequest(
            name=name, observable="raman_polarized",
            perturbations=(perturbation,), weights=(1.0,)))

    def add_ir_polarized(self, direction, name, effective_charges=None):
        source = None
        if effective_charges is not None:
            effective_charges = np.asarray(effective_charges, dtype=float)
            if (effective_charges.ndim != 3 or
                    effective_charges.shape[1:] != (3, 3)):
                raise ValueError(
                    "effective_charges must have shape (n_atoms, 3, 3)")
            if not np.all(np.isfinite(effective_charges)):
                raise ValueError("effective_charges must be finite")
            source = tuple(tuple(tuple(float(value) for value in row)
                                 for row in atom)
                           for atom in effective_charges)
        perturbation = IRPolarizationPerturbation(direction)
        return self._add_request(SpectroscopyRequest(
            name=name, observable="ir_polarized",
            perturbations=(perturbation,), weights=(1.0,), source=source))

    def add_ir_unpolarized(self, name, effective_charges=None):
        source = None
        if effective_charges is not None:
            effective_charges = np.asarray(effective_charges, dtype=float)
            if (effective_charges.ndim != 3 or
                    effective_charges.shape[1:] != (3, 3)):
                raise ValueError(
                    "effective_charges must have shape (n_atoms, 3, 3)")
            source = tuple(tuple(tuple(float(value) for value in row)
                                 for row in atom)
                           for atom in effective_charges)
        perturbations = tuple(
            IRPolarizationPerturbation(direction) for direction in np.eye(3))
        return self._add_request(SpectroscopyRequest(
            name=name, observable="ir_unpolarized",
            perturbations=perturbations, weights=(1 / 3,) * 3,
            source=source))

    def add_ir_vector(self, vector, name):
        """Add an explicitly prepared unit-cell IR Cartesian vector."""
        perturbation = CartesianPerturbation(vector)
        return self._add_request(SpectroscopyRequest(
            name=name, observable="ir_polarized",
            perturbations=(perturbation,), weights=(1.0,)))

    def add_cartesian_perturbation(self, vector, observable, name):
        if not isinstance(observable, str) or not observable:
            raise ValueError("observable must be a non-empty string")
        perturbation = CartesianPerturbation(vector)
        return self._add_request(SpectroscopyRequest(
            name=name, observable=observable,
            perturbations=(perturbation,), weights=(1.0,)))

    def manifest(self):
        """Return the JSON-compatible, execution-independent request manifest."""
        requests = []
        for request in self._requests.values():
            perturbations = []
            for perturbation in request.perturbations:
                if isinstance(perturbation, RamanTensorPerturbation):
                    value = [list(row) for row in perturbation.coefficients]
                elif isinstance(perturbation, IRPolarizationPerturbation):
                    value = list(perturbation.direction)
                else:
                    value = list(perturbation.vector)
                perturbations.append({
                    "kind": perturbation.kind.value,
                    "value": value,
                })
            requests.append({
                "name": request.name,
                "observable": request.observable,
                "convention": request.convention,
                "weights": list(request.weights),
                "source": request.source,
                "perturbations": perturbations,
            })
        source = self.ensemble_source
        return {
            "schema_version": _SPECTROSCOPY_SCHEMA_VERSION,
            "raman_schema_version": _RAMAN_SCHEMA_VERSION,
            "backend": self.backend,
            "use_symmetries": self.use_symmetries,
            "symmetry_tolerance": self.symmetry_tolerance,
            "ignore_v3": self.ignore_v3,
            "ignore_v4": self.ignore_v4,
            "lo_to_split": self.lo_to_split,
            "backend_options": self.backend_options,
            "ensemble_source": source.describe() if source is not None
            else None,
            "ensemble_provenance": source.provenance() if source is not None
            else None,
            "requests": requests,
        }

    def plan_calculations(self):
        """Return the symmetry-reduced run plan without executing Lanczos."""
        dyn = self._require_reference_dyn("build a run plan")
        if not self._requests:
            raise ValueError("Add at least one Raman or IR request first")
        from tdscha import _SpectroscopyWorkflow as workflow

        self._run_specs, self._request_maps, group_order = (
            workflow.build_execution_plan(
                self._requests, dyn,
                use_symmetries=self.use_symmetries,
                tolerance=self.symmetry_tolerance))
        return {
            "group_order": group_order,
            "n_requested_components": sum(
                len(request.perturbations)
                for request in self._requests.values()),
            "n_independent_runs": len(self._run_specs),
            "runs": {
                run_id: {
                    "kind": spec.kind,
                    "vector": list(spec.vector),
                    "stabilizer": list(spec.stabilizer),
                    "characters": list(spec.characters),
                    "cosets": [list(coset) for coset in spec.cosets],
                }
                for run_id, spec in self._run_specs.items()
            },
            "request_components": {
                name: [
                    {
                        "run_id": component.run_id,
                        "weight": component.weight,
                        "phase": component.phase,
                        "component_index": component.component_index,
                    }
                    for component in components
                ]
                for name, components in self._request_maps.items()
            },
        }

    def _execution_manifest(self, n_steps, run_options):
        from tdscha import _SpectroscopyWorkflow as workflow

        plan = self.plan_calculations()
        manifest = self.manifest()
        dyn = self._require_reference_dyn("write an execution manifest")
        temperature = float(self.reference_temperature)
        structure = dyn.structure
        supercell = np.asarray(dyn.GetSupercell())
        unit_cell_volume = float(abs(np.linalg.det(structure.unit_cell)))
        manifest.update({
            "ensemble_fingerprint": workflow.reference_fingerprint(
                dyn, temperature),
            "temperature": temperature,
            "unit_cell_volume_angstrom3": unit_cell_volume,
            "supercell_volume_angstrom3": unit_cell_volume * float(
                np.prod(supercell)),
            "dielectric_tensor": workflow.json_compatible(
                getattr(dyn, "dielectric_tensor", None)),
            "target_steps": int(n_steps),
            "run_options": workflow.json_compatible(run_options),
            "group_order": plan["group_order"],
            "request_components": plan["request_components"],
            "runs": {},
        })
        for run_id, run in plan["runs"].items():
            manifest["runs"][run_id] = dict(
                run, state="pending", completed_steps=0, converged=False,
                error=None)
        return workflow.json_compatible(manifest)

    @staticmethod
    def _validate_restart_manifest(existing, current):
        # ``ensemble_provenance`` is intentionally absent: it records the
        # absolute paths the ensemble was read from, which must not stop a
        # calculation from resuming after it has been moved.
        fields = (
            "schema_version", "raman_schema_version", "backend",
            "use_symmetries", "symmetry_tolerance", "backend_options",
            "ignore_v3", "ignore_v4", "lo_to_split",
            "requests", "ensemble_fingerprint", "ensemble_source",
            "run_options", "request_components")
        mismatches = [field for field in fields
                      if existing.get(field) != current.get(field)]
        if mismatches:
            raise ValueError(
                "Spectroscopy checkpoint is incompatible in: {}".format(
                    ", ".join(mismatches)))

    def run(self, n_steps, save_each=10, resume=True, verbose=True,
            run_options=None):
        """Run every symmetry-inequivalent perturbation to ``n_steps``.

        ``n_steps`` is the total requested number of Lanczos coefficients,
        including work restored from checkpoints.

        The backend engine is built once and reused for every independent
        perturbation: preparing a perturbation resets the whole Lanczos
        state, and reading a production ensemble is minutes of I/O that must
        not be repeated per run.  It is built lazily, so a fully restored
        calculation reloads nothing.
        """
        from tdscha import _SpectroscopyWorkflow as workflow

        if not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
            raise ValueError("n_steps must be a positive integer")
        if not isinstance(save_each, (int, np.integer)) or int(save_each) < 1:
            raise ValueError("save_each must be a positive integer")
        n_steps = int(n_steps)
        save_each = int(save_each)
        run_options = dict(run_options or {})
        current = self._execution_manifest(n_steps, run_options)
        workdir = Path(self.workdir)
        manifest_path = workdir / "manifest.json"
        workflow.ensure_directory(workdir / "runs")

        if manifest_path.exists() and resume:
            with open(manifest_path, "r", encoding="utf-8") as stream:
                existing = json.load(stream)
            self._validate_restart_manifest(existing, current)
            for run_id, run in current["runs"].items():
                if run_id in existing.get("runs", {}):
                    old = existing["runs"][run_id]
                    run["state"] = old.get("state", "pending")
                    run["completed_steps"] = old.get("completed_steps", 0)
                    run["converged"] = old.get("converged", False)
                    run["error"] = old.get("error")
                    run["analysis"] = old.get("analysis")
        self._manifest_data = current
        workflow.atomic_write_json(manifest_path, current)

        engine_options = dict(self.backend_options)
        engine_options.update(
            ignore_v3=self.ignore_v3,
            ignore_v4=self.ignore_v4,
            lo_to_split=(
                np.asarray(self.lo_to_split, dtype=float)
                if isinstance(self.lo_to_split, list)
                else self.lo_to_split))
        # Built on first use.  Every rank walks the same run list and makes
        # the same skip decisions, so the construction -- which is collective
        # for the distributed backends -- stays matched across ranks.
        engine = None

        for run_id, spec in self._run_specs.items():
            run_dir = workdir / "runs" / run_id
            status_path = run_dir / "status.npz"
            result_path = run_dir / "result.npz"
            metadata_path = run_dir / "metadata.json"
            entry = current["runs"][run_id]
            completed = int(entry.get("completed_steps", 0))
            portable_path = run_dir / "lanczos.abc"
            if (resume and entry.get("state") == "complete" and
                    (completed >= n_steps or entry.get("converged", False)) and
                    (result_path.exists() or portable_path.exists())):
                if result_path.exists():
                    self._results[run_id] = workflow.load_result(result_path)
                else:
                    analysis = entry.get("analysis") or {}
                    self._results[run_id] = workflow.load_abc_result(
                        portable_path, run_id, current["temperature"],
                        use_wigner=analysis.get("use_wigner", True),
                        reverse=analysis.get("reverse", False),
                        shift=analysis.get("shift", 0.0))
                continue

            if engine is None:
                engine = workflow.create_backend(
                    self.ensemble, self.backend, engine_options,
                    use_symmetries=self.use_symmetries)
            workflow.prepare_engine(
                engine, spec.as_array(), self.use_symmetries, spec,
                self.symmetry_tolerance)
            entry["analysis"] = workflow.analysis_metadata(
                engine, self.backend)
            if resume and status_path.exists():
                engine.load_status(str(status_path))
                completed = workflow.completed_steps(engine, self.backend)
            else:
                completed = 0

            entry.update(state="running", completed_steps=completed, error=None)
            workflow.atomic_write_json(metadata_path, entry)
            workflow.atomic_write_json(manifest_path, current)
            try:
                while completed < n_steps:
                    chunk = min(save_each, n_steps - completed)
                    workflow.run_engine_chunk(
                        engine, self.backend, chunk, verbose, run_options)
                    completed = workflow.completed_steps(engine, self.backend)
                    workflow.atomic_save_status(engine, status_path)
                    entry["completed_steps"] = completed
                    if workflow.engine_converged(engine, self.backend):
                        entry["converged"] = True
                    workflow.atomic_write_json(metadata_path, entry)
                    workflow.atomic_write_json(manifest_path, current)
                    if entry["converged"]:
                        break

                workflow.save_result(
                    engine, self.backend, run_id, result_path)
                workflow.atomic_save_abc(engine, portable_path)
                self._results[run_id] = workflow.load_result(result_path)
                entry.update(state="complete", completed_steps=completed,
                             error=None)
            except Exception as error:
                entry.update(state="failed", completed_steps=completed,
                             error="{}: {}".format(
                                 type(error).__name__, error))
                workflow.atomic_write_json(metadata_path, entry)
                workflow.atomic_write_json(manifest_path, current)
                raise
            workflow.atomic_write_json(metadata_path, entry)
            workflow.atomic_write_json(manifest_path, current)
        return self

    @classmethod
    def load(cls, workdir):
        """Load a completed or partially completed calculation for analysis."""
        from tdscha import _SpectroscopyWorkflow as workflow

        workdir = Path(workdir)
        with open(workdir / "manifest.json", "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        instance = cls(
            None, backend=manifest["backend"], workdir=workdir,
            use_symmetries=manifest.get("use_symmetries", True),
            symmetry_tolerance=manifest.get("symmetry_tolerance", 1e-8),
            ignore_v3=manifest.get("ignore_v3"),
            ignore_v4=manifest.get("ignore_v4"),
            lo_to_split=manifest.get("lo_to_split"),
            backend_options=manifest.get("backend_options", {}))
        instance._manifest_data = manifest
        from tdscha import _SpectroscopyWorkflow as workflow_module
        instance._request_maps = {
            name: tuple(workflow_module.RequestComponent(**component)
                        for component in components)
            for name, components in manifest["request_components"].items()
        }
        for run_id, entry in manifest["runs"].items():
            result_path = workdir / "runs" / run_id / "result.npz"
            abc_path = workdir / "runs" / run_id / "lanczos.abc"
            if entry.get("state") == "complete" and result_path.exists():
                instance._results[run_id] = workflow.load_result(result_path)
            elif entry.get("state") == "complete" and abc_path.exists():
                analysis = entry.get("analysis") or {}
                instance._results[run_id] = workflow.load_abc_result(
                    abc_path, run_id, manifest["temperature"],
                    use_wigner=analysis.get("use_wigner", True),
                    reverse=analysis.get("reverse", False),
                    shift=analysis.get("shift", 0.0))
        return instance

    def _request_manifest(self, name):
        if self._manifest_data is None:
            requests = self.manifest()["requests"]
        else:
            requests = self._manifest_data["requests"]
        for request in requests:
            if request["name"] == name:
                return request
        raise KeyError("Unknown spectroscopy request {!r}".format(name))

    def _evaluate_request(self, name, frequencies, quantity, options):
        from tdscha import _SpectroscopyWorkflow as workflow

        if name not in self._request_maps:
            raise KeyError("Unknown or unplanned spectroscopy request {!r}".format(
                name))
        cache = {}
        total = np.zeros_like(
            np.asarray(frequencies, dtype=float),
            dtype=np.complex128 if quantity == "green" else float)
        for component in self._request_maps[name]:
            if component.run_id is None:
                continue
            if component.run_id not in self._results:
                raise RuntimeError(
                    "Result {} required by {!r} is incomplete".format(
                        component.run_id, name))
            if component.run_id not in cache:
                result = self._results[component.run_id]
                if quantity == "green":
                    cache[component.run_id] = workflow.evaluate_green_function(
                        result, frequencies, **options)
                else:
                    cache[component.run_id] = workflow.evaluate_response(
                        result, frequencies, **options)
            total += component.weight * cache[component.run_id]
        return total

    def green_function(self, name, frequencies, **options):
        """Return the weighted complex response for one named request."""
        return self._evaluate_request(
            name, frequencies, "green", dict(options))

    def response(self, name, frequencies, **options):
        """Return the Bose-free weighted spectral response ``-Im G``."""
        return self._evaluate_request(
            name, frequencies, "response", dict(options))

    def raman_spectrum(self, name, frequencies, kind="stokes",
                       temperature=None, laser_frequency=None, **options):
        """Return Raman response, Stokes, or anti-Stokes intensity."""
        request = self._request_manifest(name)
        if not request["observable"].startswith("raman"):
            raise ValueError("Request {!r} is not Raman".format(name))
        frequencies = np.asarray(frequencies, dtype=float)
        spectrum = self.response(name, frequencies, **options)
        if kind == "response":
            return spectrum
        if kind not in ("stokes", "anti_stokes"):
            raise ValueError(
                "kind must be 'response', 'stokes', or 'anti_stokes'")
        if np.any(frequencies <= 0):
            raise ValueError("Thermal Raman frequencies must be positive")
        if temperature is None:
            if self._manifest_data is not None:
                temperature = self._manifest_data["temperature"]
            else:
                temperature = self.reference_temperature
            if temperature is None:
                raise ValueError(
                    "No temperature is available: pass temperature=, or run "
                    "or load the calculation first")
        import tdscha.DynamicalLanczos as DL
        occupation = DL.bose_occupation(frequencies, float(temperature))
        spectrum = spectrum * (
            occupation + 1 if kind == "stokes" else occupation)
        if laser_frequency is not None:
            laser_frequency = float(laser_frequency)
            scattered = (laser_frequency - frequencies
                         if kind == "stokes"
                         else laser_frequency + frequencies)
            if np.any(scattered <= 0):
                raise ValueError(
                    "laser_frequency must exceed every Stokes shift")
            spectrum = spectrum * scattered**4
        return spectrum

    def ir_susceptibility(self, name, frequencies, **options):
        """Return the projected ionic susceptibility in Hartree atomic units.

        The Lanczos Green function uses CellConstructor's Rydberg frequency
        and mass convention.  Converting its displacement response to the
        conventional Hartree atomic units contributes the factor two below.

        The Green function is computed from the gamma perturbation
        ``Z* . direction * sqrt(n_cell)`` (``prepare_ir`` scales the unit-cell
        charge vector by ``sqrt(n_cell)``), so it already carries the
        ``n_cell`` factor.  ``dielectric_function`` therefore divides by the
        **supercell** volume ``V = n_cell * V_unit_cell``, exactly matching the
        CellConstructor non-analytic LO-TO term ``8*pi/V`` (the 8 is the
        Rydberg ``e^2 = 2``).
        """
        request = self._request_manifest(name)
        if not request["observable"].startswith("ir"):
            raise ValueError("Request {!r} is not IR".format(name))
        return 2 * self.green_function(name, frequencies, **options)

    def dielectric_function(self, name, frequencies,
                            epsilon_infinity=None, ionic_prefactor=None,
                            electronic_projection=None, **options):
        """Return projected ``epsilon_inf + (4 pi / Omega) chi_ionic``.

        The default volume is the **supercell** volume converted from the
        CellConstructor Angstrom convention to Bohr cubed.  This is required
        because the Lanczos perturbation carries ``sqrt(n_cell)``
        (``prepare_ir``), so ``chi_ionic`` already includes the ``n_cell``
        factor and the volume must cancel it.  Together with the factor of two
        in ``ir_susceptibility`` (Rydberg ``e^2 = 2``), the total prefactor is
        ``8*pi / V_supercell``, matching the CellConstructor non-analytic LO-TO
        term.  ``ionic_prefactor`` can override the full prefactor when a
        different electromagnetic/unit convention is needed.
        """
        request = self._request_manifest(name)
        if not request["observable"].startswith("ir"):
            raise ValueError("Request {!r} is not IR".format(name))
        if self._manifest_data is None:
            raise RuntimeError("Run or load the calculation before analysis")
        if epsilon_infinity is None and self.reference_dyn is not None:
            epsilon_infinity = getattr(
                self.reference_dyn, "dielectric_tensor", None)
        if epsilon_infinity is None:
            epsilon_infinity = self._manifest_data.get("dielectric_tensor")
        if epsilon_infinity is None:
            raise ValueError("No electronic dielectric tensor is available")
        epsilon_infinity = _finite_array(
            epsilon_infinity, (3, 3), "epsilon_infinity")
        if electronic_projection is not None:
            electronic = float(electronic_projection)
        elif request["observable"] == "ir_unpolarized":
            electronic = np.trace(epsilon_infinity) / 3
        elif request["perturbations"][0]["kind"] == "ir":
            direction = np.asarray(
                request["perturbations"][0]["value"], dtype=float)
            direction = direction / np.linalg.norm(direction)
            electronic = direction @ epsilon_infinity @ direction
        else:
            raise ValueError(
                "An explicit IR vector requires electronic_projection")
        if ionic_prefactor is None:
            from cellconstructor.Units import A_TO_BOHR
            volume_angstrom3 = self._manifest_data.get(
                "supercell_volume_angstrom3")
            if volume_angstrom3 is None:
                # Backward compatibility with manifests written before the
                # supercell volume was stored.
                dyn = self._require_reference_dyn(
                    "recover the supercell volume of an old manifest")
                n_cell = float(np.prod(np.asarray(dyn.GetSupercell())))
                volume_angstrom3 = (
                    self._manifest_data["unit_cell_volume_angstrom3"]
                    * n_cell)
            volume_bohr3 = volume_angstrom3 * float(A_TO_BOHR)**3
            ionic_prefactor = 4 * np.pi / volume_bohr3
        return (electronic + float(ionic_prefactor)
                * self.ir_susceptibility(name, frequencies, **options))


__all__ = [
    "CartesianPerturbation", "EnsembleSource", "IRPolarizationPerturbation",
    "PerturbationKind", "PerturbationOrbit", "RAMAN_COMPONENTS",
    "RamanComponent", "RamanTensorPerturbation", "Spectroscopy",
    "SpectroscopyRequest", "SymmetryGroup", "build_ir_vector",
    "build_raman_vector", "find_perturbation_orbits",
    "find_atom_permutation", "get_gamma_symmetry_representation",
    "get_raman_component", "get_unpolarized_raman_weights",
    "raman_coefficients_from_polarizations", "symmetric_raman_vector",
    "vector_representations_for_ir",
    "vector_representations_for_symmetric_raman",
]
