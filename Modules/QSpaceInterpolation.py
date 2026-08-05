"""Shared q-mesh and harmonic interpolation utilities.

The anharmonic interpolation lives in :mod:`tdscha.QSpaceAtomFourier`.
This module contains only the order-independent mesh operations and the
second-order force-constant interpolation needed to build its fine harmonic
basis.
"""

import itertools

import numpy as np

import cellconstructor as CC
import cellconstructor.ForceTensor
import cellconstructor.Methods


def validate_mesh(mesh, name="mesh"):
    """Return a three-component positive integer mesh.

    Parameters
    ----------
    mesh : array-like
        Three positive integer-valued entries.
    name : str
        Name used in validation errors.
    """
    values = np.asarray(mesh, dtype=object)
    if values.shape != (3,):
        raise ValueError(
            "{} must contain exactly three entries, got shape {}".format(
                name, values.shape))
    if any(isinstance(value, (bool, np.bool_)) for value in values.flat):
        raise ValueError("{} entries must be positive integers".format(name))
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "{} entries must be positive integers".format(name)) from error
    if (not np.all(np.isfinite(numeric))
            or not np.all(numeric == np.rint(numeric))
            or np.any(numeric <= 0)):
        raise ValueError(
            "{} entries must be positive integers, got {}".format(
                name, tuple(values)))
    return numeric.astype(int)


def generate_fine_mesh(structure, mesh):
    """Generate a Gamma-centred uniform q mesh with Gamma first.

    Returns
    -------
    q_points : ndarray, shape (prod(mesh), 3)
        Cartesian q vectors in the convention used by ``Phonons.q_tot``.
    indices : ndarray, shape (prod(mesh), 3)
        Integer mesh indices. Fractional q coordinates are ``indices / mesh``
        modulo reciprocal lattice vectors.
    """
    mesh = validate_mesh(mesh)
    reciprocal = structure.get_reciprocal_vectors() / (2.0 * np.pi)
    indices = np.array(
        list(itertools.product(*(range(int(n)) for n in mesh))),
        dtype=int)
    fractional = indices / mesh[None, :]
    fractional -= np.floor(fractional + 0.5)
    return fractional @ reciprocal, indices


def mesh_key(q, structure, mesh, tol=1e-6):
    """Return the integer index of a mesh q point modulo the reciprocal cell.

    Raises ``ValueError`` when ``q`` is not commensurate with ``mesh``.
    """
    mesh = validate_mesh(mesh)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number")
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite Cartesian vector of shape (3,)")
    fractional = np.asarray(structure.unit_cell) @ q
    scaled = fractional * mesh
    nearest = np.rint(scaled)
    if np.max(np.abs(scaled - nearest)) > tol * np.max(mesh):
        raise ValueError(
            "q-point {} is not on mesh {} (fractional q * mesh = {})".format(
                q, tuple(mesh), scaled))
    return tuple(nearest.astype(int) % mesh)


def build_q_index_lookup(q_points, structure, mesh, tol=1e-6):
    """Map integer mesh indices to positions in a q-point array."""
    q_points = np.asarray(q_points, dtype=np.float64)
    if q_points.ndim != 2 or q_points.shape[1] != 3:
        raise ValueError(
            "q_points must have shape (n_q, 3), got {}".format(
                q_points.shape))
    lookup = {}
    for iq, q in enumerate(q_points):
        key = mesh_key(q, structure, mesh, tol)
        if key in lookup:
            raise ValueError(
                "q_points contains duplicate mesh point {} at indices {} "
                "and {}".format(key, lookup[key], iq))
        lookup[key] = iq
    return lookup


def _matching_q(q, candidates, reciprocal, tol=1e-6):
    """Index of ``q`` in ``candidates`` modulo a reciprocal vector, or -1."""
    for index, candidate in enumerate(candidates):
        distance = CC.Methods.get_min_dist_into_cell(
            reciprocal, np.asarray(q), np.asarray(candidate))
        if distance < tol:
            return index
    return -1


def interpolate_dyn_fine(
        dyn, q_points, use_asr=True, reuse_commensurate=True,
        ignore_effective_charges=False, lo_to_split=None, verbose=False):
    """Fourier-interpolate a dynamical matrix at arbitrary q points.

    The real-space second-order force constants are centred before
    interpolation and optionally projected onto the acoustic sum rule.
    Commensurate input matrices are reused exactly, and time-reversed pairs
    share conjugate eigenvectors.

    ``ignore_effective_charges=True`` removes Born charges and the dielectric
    tensor on a private copy used only for this interpolation. This is useful
    when the ensemble forces came from a strictly short-range potential but
    ``dyn`` contains long-range metadata inherited from another calculation.
    The caller's object is never modified.

    ``lo_to_split`` controls the nonanalytic Gamma limit: ``None`` disables
    it, ``"random"`` lets CellConstructor choose a direction, and a finite
    nonzero three-vector selects an explicit propagation direction.  At
    nonzero q the usual tensorial dipole--dipole interpolation is retained.

    Returns
    -------
    frequencies : ndarray, shape (3 * n_atoms, n_q)
        Signed phonon frequencies in Ry.
    polarizations : ndarray, shape (3 * n_atoms, 3 * n_atoms, n_q)
        Complex polarization vectors.
    """
    q_points = np.asarray(q_points, dtype=np.float64)
    if q_points.ndim != 2 or q_points.shape[1] != 3:
        raise ValueError(
            "q_points must have shape (n_q, 3), got {}".format(
                q_points.shape))
    if len(q_points) == 0:
        raise ValueError("q_points must contain at least one q-point")
    if not np.all(np.isfinite(q_points)):
        raise ValueError("q_points must contain only finite values")

    if isinstance(lo_to_split, str):
        if lo_to_split != "random":
            raise ValueError(
                "lo_to_split must be None, 'random', or a three-vector")
        q_direct = None
        use_gamma_nonanalytic = True
    elif lo_to_split is None:
        q_direct = None
        use_gamma_nonanalytic = False
    else:
        q_direct = np.asarray(lo_to_split, dtype=float)
        if (q_direct.shape != (3,) or not np.all(np.isfinite(q_direct)) or
                np.linalg.norm(q_direct) <= 1e-14):
            raise ValueError("lo_to_split must be a finite nonzero three-vector")
        use_gamma_nonanalytic = True

    if ignore_effective_charges:
        # The flag is local to harmonic interpolation.  It suppresses the
        # complete dipolar correction, including the directional Gamma
        # limit, without modifying dyn.effective_charges.  Those charges can
        # therefore still define an IR perturbation downstream.
        q_direct = None
        use_gamma_nonanalytic = False

    work_dyn = dyn
    if ignore_effective_charges and dyn.effective_charges is not None:
        work_dyn = dyn.Copy()
        work_dyn.effective_charges = None
        work_dyn.dielectric_tensor = None

    structure = work_dyn.structure
    supercell = validate_mesh(work_dyn.GetSupercell(), "coarse mesh")
    super_structure = structure.generate_supercell(supercell)
    tensor2 = CC.ForceTensor.Tensor2(
        structure, super_structure, supercell)
    tensor2.SetupFromPhonons(work_dyn)
    tensor2.Center()
    if use_asr:
        tensor2.Apply_ASR()

    n_q = len(q_points)
    n_bands = 3 * structure.N_atoms
    masses = np.repeat(structure.get_masses_array(), 3)
    mass_factor = 1.0 / np.sqrt(np.outer(masses, masses))
    reciprocal = structure.get_reciprocal_vectors() / (2.0 * np.pi)

    commensurate = np.full(n_q, -1, dtype=int)
    if reuse_commensurate:
        for iq, q in enumerate(q_points):
            commensurate[iq] = _matching_q(
                q, work_dyn.q_tot, reciprocal)

    negative = np.full(n_q, -1, dtype=int)
    for iq, q in enumerate(q_points):
        negative[iq] = _matching_q(-q, q_points, reciprocal)

    frequencies = np.zeros((n_bands, n_q), dtype=np.float64)
    polarizations = np.zeros(
        (n_bands, n_bands, n_q), dtype=np.complex128)
    done = np.zeros(n_q, dtype=bool)

    for iq, q in enumerate(q_points):
        if done[iq]:
            continue
        if commensurate[iq] >= 0:
            force_constants = np.array(
                work_dyn.dynmats[commensurate[iq]], dtype=np.complex128)
        else:
            # Tensor2 and Phonons use opposite Fourier phase conventions.
            at_gamma = np.linalg.norm(
                CC.Methods.get_min_dist_into_cell(
                    reciprocal, np.asarray(q), np.zeros(3))) < 1e-8
            force_constants = tensor2.Interpolate(
                -q, asr=False,
                lo_to_splitting=(use_gamma_nonanalytic and at_gamma),
                q_direct=(q_direct if at_gamma else None))

        dynamical = force_constants * mass_factor
        dynamical = 0.5 * (dynamical + dynamical.conj().T)
        if negative[iq] == iq:
            dynamical = dynamical.real

        eigenvalues, eigenvectors = np.linalg.eigh(dynamical)
        frequencies[:, iq] = (
            np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)))
        polarizations[:, :, iq] = eigenvectors
        done[iq] = True

        jq = negative[iq]
        if jq >= 0 and jq != iq and not done[jq]:
            frequencies[:, jq] = frequencies[:, iq]
            polarizations[:, :, jq] = eigenvectors.conj()
            done[jq] = True

    if verbose:
        print(
            "Interpolated the dynamical matrix at {} q-points "
            "({} commensurate points reused).".format(
                n_q, int(np.sum(commensurate >= 0))))

    return frequencies, polarizations
