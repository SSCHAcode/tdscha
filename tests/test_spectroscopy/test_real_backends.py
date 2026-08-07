"""Small real-engine integration tests for the Spectroscopy driver."""

from pathlib import Path

import numpy as np
import pytest

import cellconstructor as CC
import sscha.Ensemble

import tdscha.Spectroscopy as SP
from tdscha import _SpectroscopyWorkflow as workflow


DATA = Path(__file__).parents[1] / "test_julia" / "data"


def _ensemble():
    dyn = CC.Phonons.Phonons(str(DATA / "dyn_gen_pop1_"), 3)
    ensemble = sscha.Ensemble.Ensemble(dyn, 250)
    ensemble.load_bin(str(DATA), 1)
    return ensemble


@pytest.mark.parametrize("backend", ["real", "qspace"])
def test_driver_executes_real_engines_and_loads_results(backend, tmp_path):
    ensemble = _ensemble()
    charges = np.zeros((ensemble.current_dyn.structure.N_atoms, 3, 3))
    charges[0] = np.eye(3)
    charges[1] = -np.eye(3)
    job = SP.Spectroscopy(
        ensemble, backend=backend,
        workdir=tmp_path / backend,
        use_symmetries=False,
        ignore_v3=True, ignore_v4=True)
    job.add_ir_polarized(
        [1, 0, 0], "ir_x", effective_charges=charges)
    job.run(2, save_each=1, verbose=False)

    loaded = SP.Spectroscopy.load(tmp_path / backend)
    frequencies = np.linspace(0.001, 0.003, 5)
    response = loaded.response(
        "ir_x", frequencies, use_terminator=False, smearing=1e-4)
    assert response.shape == frequencies.shape
    assert np.all(np.isfinite(response))
    assert np.all(response >= 0)


def test_driver_executes_atom_fourier_backend(tmp_path):
    ensemble = _ensemble()
    charges = np.zeros((ensemble.current_dyn.structure.N_atoms, 3, 3))
    charges[0] = np.eye(3)
    charges[1] = -np.eye(3)
    mesh = tuple(int(value) for value in
                 ensemble.current_dyn.GetSupercell())
    job = SP.Spectroscopy(
        ensemble, backend="atom_fourier",
        workdir=tmp_path / "atom_fourier", use_symmetries=False,
        ignore_v3=True, ignore_v4=True,
        backend_options={"fine_mesh": mesh})
    job.add_ir_polarized([1, 0, 0], "ir_x", effective_charges=charges)
    job.run(1, verbose=False)
    assert np.all(np.isfinite(job.response(
        "ir_x", np.linspace(0.001, 0.003, 3),
        use_terminator=False, smearing=1e-4)))


@pytest.mark.parametrize("backend", ["real", "qspace"])
def test_stabilizer_coset_kernel_matches_full_group_average(backend, tmp_path):
    ensemble = _ensemble()
    charges = np.zeros((ensemble.current_dyn.structure.N_atoms, 3, 3))
    charges[0] = np.eye(3)
    charges[1] = -np.eye(3)
    job = SP.Spectroscopy(
        ensemble, backend=backend, workdir=tmp_path / backend,
        use_symmetries=True)
    job.add_ir_polarized(
        [1, 0, 0], "ir_x", effective_charges=charges)
    job.plan_calculations()
    spec = next(iter(job._run_specs.values()))

    full = workflow.create_backend(ensemble, backend, {})
    workflow.prepare_engine(full, spec.as_array(), True)
    reduced = workflow.create_backend(ensemble, backend, {})
    workflow.prepare_engine(
        reduced, spec.as_array(), True, spec, job.symmetry_tolerance)

    assert reduced._spectroscopy_symmetry_count(
        reduced.n_syms) < reduced.n_syms
    np.testing.assert_allclose(
        reduced.apply_anharmonic_FT(), full.apply_anharmonic_FT(),
        rtol=2e-10, atol=2e-12)
