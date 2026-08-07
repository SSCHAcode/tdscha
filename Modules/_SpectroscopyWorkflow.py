"""Private execution, persistence, and symmetry helpers for Spectroscopy.

The public physical definitions live in :mod:`tdscha.Spectroscopy`.  This
module keeps filesystem and backend orchestration out of that definition
layer and never reimplements Lanczos recursions or continued fractions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np

import cellconstructor.Settings as Parallel


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    kind: str
    vector: Tuple[float, ...]
    stabilizer: Tuple[int, ...]
    characters: Tuple[float, ...]
    cosets: Tuple[Tuple[int, ...], ...]
    group_rotations: Tuple[Tuple[Tuple[float, ...], ...], ...]

    def as_array(self):
        return np.asarray(self.vector, dtype=float)


@dataclass(frozen=True)
class RequestComponent:
    # ``None`` denotes a symmetry-forbidden (identically zero) optical
    # component.  It participates in the observable definition, but does not
    # require a Lanczos calculation.
    run_id: Optional[str]
    weight: float
    phase: float
    component_index: int


@dataclass(frozen=True)
class SpectroscopyResult:
    run_id: str
    method: str
    temperature: float
    perturbation_modulus: float
    data: dict


def json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_compatible(item)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("Value {!r} is not JSON serializable".format(value))


def array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        if array is None:
            digest.update(b"<none>")
            continue
        array = np.asarray(array)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        if array.dtype.kind in "OUS":
            digest.update(json.dumps(array.tolist(), sort_keys=True).encode())
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def reference_fingerprint(dyn, temperature):
    """Fingerprint the reference dynamical matrix and temperature.

    This is what identifies a calculation for restart purposes.  It
    deliberately does not touch the configurations: they live only on the
    master once the ensemble is loaded distributed, so any fingerprint over
    them would either be unavailable or force a collective.  The
    configurations are identified separately, by the manifest's
    ``ensemble_source`` entry.
    """
    structure = dyn.structure
    dynmats = getattr(dyn, "dynmats", ())
    arrays = [
        structure.unit_cell, structure.coords,
        np.asarray(structure.get_atomic_types()),
        np.asarray(dyn.GetSupercell()),
    ]
    arrays.extend(dynmats)
    arrays.extend([
        getattr(dyn, "raman_tensor", None),
        getattr(dyn, "effective_charges", None),
        getattr(dyn, "dielectric_tensor", None),
        np.asarray([float(temperature)], dtype=float),
    ])
    return array_fingerprint(*arrays)


def canonical_vector(vector, tolerance):
    vector = np.asarray(vector, dtype=float).ravel()
    norm = np.linalg.norm(vector)
    significant = np.flatnonzero(np.abs(vector) > tolerance * norm)
    if significant.size and vector[significant[0]] < 0:
        vector = -vector
    return vector


def vector_run_id(vector, tolerance):
    vector = canonical_vector(vector, tolerance)
    # Optical vertices have physical units and no universal absolute scale.
    # Hash the deterministic canonical bytes; tolerance is used only to pick
    # the sign, never to quantize a small physical perturbation to zero.
    digest = hashlib.sha256(np.ascontiguousarray(vector).tobytes())
    return "p_" + digest.hexdigest()[:20]


def _request_vector(request, perturbation, dyn):
    import tdscha.Spectroscopy as SP

    if isinstance(perturbation, SP.RamanTensorPerturbation):
        if dyn.raman_tensor is None:
            raise ValueError(
                "Raman request {!r} requires a Raman tensor".format(
                    request.name))
        return SP.build_raman_vector(
            dyn.raman_tensor, perturbation.as_array()).ravel()
    if isinstance(perturbation, SP.IRPolarizationPerturbation):
        effective_charges = request.source
        if effective_charges is None:
            effective_charges = dyn.effective_charges
        if effective_charges is None:
            raise ValueError(
                "IR request {!r} requires effective charges".format(
                    request.name))
        return SP.build_ir_vector(
            np.asarray(effective_charges), perturbation.as_array())
    return perturbation.as_array().ravel()


def build_execution_plan(requests, dyn, use_symmetries=True, tolerance=1e-8):
    """Build globally deduplicated runs and per-request reconstruction maps."""
    import tdscha.Spectroscopy as SP

    if use_symmetries:
        group, representations = SP.get_gamma_symmetry_representation(
            dyn.structure, tolerance=tolerance,
            supercell=dyn.GetSupercell())
    else:
        group = SP.SymmetryGroup.from_matrices([np.eye(3)])
        dimension = 3 * dyn.structure.N_atoms
        representations = (np.eye(dimension),)

    run_specs = {}
    request_maps = {}
    flattened = []
    for request in requests.values():
        vectors = tuple(_request_vector(request, perturbation, dyn)
                        for perturbation in request.perturbations)
        if any(vector.size != representations[0].shape[0]
               for vector in vectors):
            raise ValueError(
                "Request {!r} has a Cartesian vector incompatible with the "
                "unit-cell structure".format(request.name))
        components = [None] * len(vectors)
        norms = np.asarray([np.linalg.norm(vector) for vector in vectors])
        reference_norm = float(np.max(norms, initial=0.0))
        zero_threshold = (
            100 * np.finfo(float).eps * reference_norm
            if reference_norm > 0 else 0.0)
        for index, (perturbation, vector, norm) in enumerate(zip(
                request.perturbations, vectors, norms)):
            if norm <= zero_threshold:
                components[index] = RequestComponent(
                    run_id=None,
                    weight=float(request.weights[index]),
                    phase=1.0,
                    component_index=index)
            else:
                flattened.append(
                    (request, index, perturbation, vector))
        request_maps[request.name] = components

    vectors = tuple(item[3] for item in flattened)
    orbits = SP.find_perturbation_orbits(
        vectors, representations, group, tolerance=tolerance)
    for orbit in orbits:
        _, _, perturbation, representative_vector = (
            flattened[orbit.representative])
        run_id = vector_run_id(representative_vector, tolerance)
        canonical = canonical_vector(representative_vector, tolerance)
        if run_id in run_specs:
            existing = run_specs[run_id].as_array()
            scale = max(np.linalg.norm(existing),
                        np.linalg.norm(canonical), np.finfo(float).tiny)
            if np.linalg.norm(existing - canonical) > tolerance * scale:
                raise RuntimeError("Perturbation hash collision")
        else:
            run_specs[run_id] = RunSpec(
                run_id=run_id,
                kind=perturbation.kind.value,
                vector=tuple(float(item) for item in canonical),
                stabilizer=orbit.stabilizer,
                characters=orbit.characters,
                cosets=orbit.right_cosets,
                group_rotations=group.matrices)

        for member, phase in zip(orbit.members, orbit.phases):
            member_request, member_index, _, _ = flattened[member]
            request_maps[member_request.name][member_index] = RequestComponent(
                run_id=run_id,
                weight=float(member_request.weights[member_index]),
                phase=float(phase),
                component_index=member_index)

    request_maps = {
        name: tuple(components) for name, components in request_maps.items()
    }
    return run_specs, request_maps, len(group)


def _create_distributed_backend(source, backend, options, use_symmetries):
    """Build a q-space engine whose configurations live on one rank each.

    The master reads the ensemble and scatters the Bloch-transformed
    configurations; no rank ever holds a replica.  Anything in the
    construction that performs an MPI collective is run by every rank first,
    through ``prepare_distributed_construction`` -- see
    ``QSpaceLanczos.load_distributed_tdscha``.
    """
    loader_options = dict(
        use_symmetries=use_symmetries,
        n_configs=source.n_configs,
        final_dyn=source.converged_dyn,
        final_T=(source.reference_temperature
                 if source.converged_dyn is not None else None),
        lo_to_split=options.pop("lo_to_split", None))

    if backend == "qspace":
        import tdscha.QSpaceLanczos as QL
        return QL.load_distributed_tdscha(
            source.data_dir, source.population, source.generating_dyn,
            source.T, **loader_options, **options)
    if backend == "atom_fourier":
        import tdscha.QSpaceAtomFourier as QAF
        fine_mesh = options.pop("fine_mesh", None)
        if fine_mesh is None:
            raise ValueError(
                "backend='atom_fourier' needs backend_options={'fine_mesh': "
                "(m1, m2, m3), ...}")
        return QAF.load_distributed_atom_fourier_tdscha(
            source.data_dir, source.population, source.generating_dyn,
            source.T, fine_mesh, **loader_options, **options)
    raise AssertionError("unreachable backend {!r}".format(backend))


def _create_replicated_backend(ensemble, backend, options):
    """Build an engine from an ensemble already in this process's memory."""
    if backend == "real":
        import tdscha.DynamicalLanczos as DL
        return DL.Lanczos(ensemble, **options)
    if backend == "qspace":
        import tdscha.QSpaceLanczos as QL
        return QL.QSpaceLanczos(ensemble, **options)
    if backend == "atom_fourier":
        import tdscha.QSpaceAtomFourier as QAF
        return QAF.QSpaceAtomFourierLanczos(ensemble, **options)
    raise AssertionError("unreachable backend {!r}".format(backend))


def create_backend(ensemble, backend, options, use_symmetries=True):
    """Build the Lanczos engine for one spectroscopy calculation.

    ``ensemble`` is either a :class:`~tdscha.Spectroscopy.EnsembleSource` --
    the production path, where the configurations are read once by the MPI
    master and scattered -- or a loaded ``sscha.Ensemble.Ensemble``, which
    is replicated on every rank.

    ``backend="real"`` has no distributed loader: the real-space Lanczos
    parallelizes by splitting a *replicated* ensemble across ranks, so from
    a source it loads the ensemble on every rank.  That is correct for the
    small systems this backend is for, and is why the q-space backends exist
    for the large ones.
    """
    import tdscha.Spectroscopy as SP

    if backend not in ("real", "qspace", "atom_fourier"):
        raise ValueError("Unsupported spectroscopy backend {!r}".format(
            backend))

    options = dict(options)
    runtime_flags = {}
    for name in ("ignore_v3", "ignore_v4", "ignore_harmonic",
                 "ignore_small_w"):
        if name in options:
            runtime_flags[name] = options.pop(name)

    if isinstance(ensemble, SP.EnsembleSource) and backend != "real":
        engine = _create_distributed_backend(
            ensemble, backend, options, use_symmetries)
    else:
        if isinstance(ensemble, SP.EnsembleSource):
            ensemble = ensemble.load_ensemble()
        elif backend != "real" and Parallel.GetNProc() > 1:
            warnings.warn(
                "Spectroscopy was given an already loaded ensemble, so every "
                "one of the {} ranks holds a full copy of the "
                "configurations. Pass an EnsembleSource (or use "
                "Spectroscopy.from_ensemble_path) to have the master read "
                "them once and scatter them instead.".format(
                    Parallel.GetNProc()))
        engine = _create_replicated_backend(ensemble, backend, options)

    for name, value in runtime_flags.items():
        setattr(engine, name, value)
    return engine


def prepare_engine(engine, vector, use_symmetries, run_spec=None,
                   symmetry_tolerance=1e-8):
    # Gamma optical perturbations can separate the expensive point-group
    # average from the cheap translation projector in the real-space engine.
    if hasattr(engine, "gamma_only"):
        engine.gamma_only = True
    engine.init(use_symmetries=use_symmetries)
    engine._prepare_gamma_cartesian_perturbation(np.asarray(vector, dtype=float))
    if (use_symmetries and run_spec is not None and
            hasattr(engine, "configure_spectroscopy_symmetry")):
        engine.configure_spectroscopy_symmetry(
            run_spec.group_rotations, run_spec.stabilizer,
            run_spec.characters, run_spec.cosets,
            tolerance=max(float(symmetry_tolerance), 1e-7))


def analysis_metadata(engine, backend):
    """Return the small set of conventions needed to read portable results."""
    metadata = {"use_wigner": bool(engine.use_wigner)}
    metadata.update(
        reverse=bool(engine.reverse_L),
        shift=float(engine.shift_value),
    )
    full_order = int(getattr(engine, "n_syms", 1))
    active = getattr(engine, "_spectroscopy_coset_indices", None)
    metadata["symmetry_reduction"] = {
        "enabled": active is not None,
        "full_group_order": full_order,
        "ensemble_representatives": (
            len(active) if active is not None else full_order),
        "stabilizer_order": (
            len(engine._spectroscopy_stabilizer_indices)
            if active is not None else 1),
    }
    return metadata


def completed_steps(engine, backend):
    return len(engine.a_coeffs)


def engine_converged(engine, backend):
    """Detect a terminated three-term recursion with no next Krylov vector."""
    return len(engine.a_coeffs) > len(engine.b_coeffs)


def run_engine_chunk(engine, backend, target_or_count, verbose, run_options):
    options = dict(run_options)
    options.pop("verbose", None)
    engine.run_FT(target_or_count, verbose=verbose, **options)


def _barrier():
    Parallel.barrier()


def _master():
    return Parallel.am_i_the_master()


def ensure_directory(path):
    if _master():
        Path(path).mkdir(parents=True, exist_ok=True)
    _barrier()


def atomic_write_json(path, data):
    path = Path(path)
    ensure_directory(path.parent)
    temporary = path.with_name("." + path.name + ".tmp")
    if _master():
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(json_compatible(data), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    _barrier()


def atomic_save_status(engine, path):
    path = Path(path)
    ensure_directory(path.parent)
    temporary = path.with_name("." + path.stem + ".tmp.npz")
    engine.save_status(str(temporary))
    _barrier()
    if _master():
        os.replace(temporary, path)
    _barrier()


def atomic_save_abc(engine, path):
    """Save portable Lanczos coefficients without duplicating their format."""
    path = Path(path)
    ensure_directory(path.parent)
    temporary = path.with_name("." + path.name + ".tmp")
    if _master():
        engine.save_abc(str(temporary))
        os.replace(temporary, path)
    _barrier()


def save_result(engine, backend, run_id, path):
    path = Path(path)
    ensure_directory(path.parent)
    temporary = path.with_name("." + path.stem + ".tmp.npz")
    if _master():
        common = dict(
            run_id=np.asarray(run_id),
            method=np.asarray("lanczos"),
            temperature=np.float64(engine.T),
            perturbation_modulus=np.float64(engine.perturbation_modulus),
        )
        common.update(
            a_coeffs=np.asarray(engine.a_coeffs),
            b_coeffs=np.asarray(engine.b_coeffs),
            c_coeffs=np.asarray(engine.c_coeffs),
            use_wigner=np.bool_(engine.use_wigner),
            reverse=np.bool_(engine.reverse_L),
            shift=np.float64(engine.shift_value),
        )
        np.savez_compressed(temporary, **common)
        os.replace(temporary, path)
    _barrier()


def load_result(path):
    with np.load(path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    run_id = str(raw.pop("run_id").item())
    method = str(raw.pop("method").item())
    temperature = float(raw.pop("temperature"))
    modulus = float(raw.pop("perturbation_modulus"))
    return SpectroscopyResult(
        run_id=run_id, method=method, temperature=temperature,
        perturbation_modulus=modulus, data=raw)


def load_abc_result(path, run_id, temperature, use_wigner=True,
                    reverse=False, shift=0.0):
    """Load a portable ``.abc`` calculation into the common result model."""
    import tdscha.DynamicalLanczos as DL

    engine = DL.Lanczos(None)
    engine.load_abc(str(path))
    return SpectroscopyResult(
        run_id=run_id, method="lanczos", temperature=float(temperature),
        perturbation_modulus=float(engine.perturbation_modulus),
        data={
            "a_coeffs": np.asarray(engine.a_coeffs),
            "b_coeffs": np.asarray(engine.b_coeffs),
            "c_coeffs": np.asarray(engine.c_coeffs),
            "use_wigner": np.asarray(bool(use_wigner)),
            "reverse": np.asarray(bool(reverse)),
            "shift": np.asarray(float(shift)),
        })


def evaluate_green_function(result, frequencies, **options):
    import tdscha.DynamicalLanczos as DL

    engine = DL.Lanczos(None)
    engine.a_coeffs = list(result.data["a_coeffs"])
    engine.b_coeffs = list(result.data["b_coeffs"])
    engine.c_coeffs = list(result.data["c_coeffs"])
    engine.T = result.temperature
    engine.perturbation_modulus = result.perturbation_modulus
    engine.use_wigner = bool(result.data["use_wigner"])
    engine.reverse_L = bool(result.data["reverse"])
    engine.shift_value = float(result.data["shift"])
    engine.verbose = False
    return engine.get_green_function_continued_fraction(
        np.asarray(frequencies, dtype=float), **options)


def evaluate_response(result, frequencies, **options):
    return -np.imag(evaluate_green_function(result, frequencies, **options))


__all__ = [
    "RequestComponent", "RunSpec", "SpectroscopyResult", "analysis_metadata",
    "atomic_save_abc",
    "atomic_save_status", "atomic_write_json", "build_execution_plan",
    "completed_steps", "create_backend", "engine_converged",
    "ensure_directory", "evaluate_green_function", "evaluate_response",
    "json_compatible", "load_abc_result", "load_result", "prepare_engine",
    "reference_fingerprint", "run_engine_chunk", "save_result",
]
