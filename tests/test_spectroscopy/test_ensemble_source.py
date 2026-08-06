"""``EnsembleSource`` validation and backend routing.

These are the cheap, serial guarantees around the distributed loading.  That
the loading itself is correct is checked under ``mpirun`` by
``test_distributed_spectroscopy.py``.
"""

import os

import numpy as np
import pytest

import cellconstructor as CC

import tdscha.Spectroscopy as SP
from tdscha import _SpectroscopyWorkflow as workflow


class _Dyn:
    def __init__(self):
        structure = CC.Structure.Structure(1)
        structure.unit_cell = np.eye(3) * 5.0
        structure.coords[0] = np.zeros(3)
        structure.atoms = ["X"]
        structure.masses = {"X": 1.0}
        structure.has_unit_cell = True
        self.structure = structure
        self.dynmats = [np.eye(3)]
        self.effective_charges = np.eye(3)[None, :, :]
        self.dielectric_tensor = np.eye(3) * 2.5

    @staticmethod
    def GetSupercell():
        return np.ones(3, dtype=int)


def _source(tmp_path, **overrides):
    arguments = dict(data_dir=str(tmp_path), population=1, dyn=_Dyn(),
                     T=250.0)
    arguments.update(overrides)
    return SP.EnsembleSource(**arguments)


def test_missing_directory_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        _source(tmp_path / "absent")


def test_a_path_needs_the_number_of_irreducible_q_points(tmp_path):
    with pytest.raises(ValueError, match="irreducible q-points"):
        _source(tmp_path, dyn=str(tmp_path / "dyn_"))
    with pytest.raises(ValueError, match="must not be given"):
        _source(tmp_path, nqirr=3)


def test_the_reference_is_the_converged_matrix_when_reweighting(tmp_path):
    generating, converged = _Dyn(), _Dyn()
    plain = _source(tmp_path, dyn=generating)
    assert plain.reference_dyn is generating
    assert plain.reference_temperature == 250.0
    assert plain.converged_dyn is None

    reweighted = _source(tmp_path, dyn=generating, final_dyn=converged,
                         final_T=100.0)
    assert reweighted.reference_dyn is converged
    assert reweighted.reference_temperature == 100.0

    # Without an explicit final temperature the ensemble keeps T.
    assert _source(tmp_path, dyn=generating,
                   final_dyn=converged).reference_temperature == 250.0


def test_reweighting_arguments_require_a_target(tmp_path):
    with pytest.raises(ValueError, match="final_T was given"):
        _source(tmp_path, final_T=10.0)
    with pytest.raises(ValueError, match="final_nqirr was given"):
        _source(tmp_path, final_nqirr=2)


def test_identity_separates_populations_but_survives_a_move(tmp_path):
    first = (tmp_path / "ensemble").resolve()
    first.mkdir()
    moved = (tmp_path / "elsewhere" / "ensemble").resolve()
    moved.mkdir(parents=True)

    dyn = _Dyn()
    here = SP.EnsembleSource(str(first), 1, dyn, 250.0)
    there = SP.EnsembleSource(str(moved), 1, dyn, 250.0)
    other_population = SP.EnsembleSource(str(first), 2, dyn, 250.0)
    fewer = SP.EnsembleSource(str(first), 1, dyn, 250.0, n_configs=10)

    assert here.describe() == there.describe()
    assert here.describe() != other_population.describe()
    assert here.describe() != fewer.describe()
    # The paths are still recorded, they just do not gate a restart.
    assert here.provenance()["data_dir"] != there.provenance()["data_dir"]


def test_the_driver_rejects_something_that_is_neither(tmp_path):
    with pytest.raises(TypeError, match="EnsembleSource"):
        SP.Spectroscopy(object(), workdir=tmp_path / "work")


def test_the_manifest_carries_the_ensemble_identity(tmp_path):
    job = SP.Spectroscopy(_source(tmp_path), backend="qspace",
                          workdir=tmp_path / "work")
    job.add_ir_polarized([1, 0, 0], "ir")
    manifest = job.manifest()
    assert manifest["ensemble_source"]["population"] == 1
    assert manifest["ensemble_provenance"]["data_dir"] == os.path.abspath(
        str(tmp_path))

    analysis_only = SP.Spectroscopy(None, workdir=tmp_path / "work")
    assert analysis_only.manifest()["ensemble_source"] is None


@pytest.mark.parametrize("backend", ["qspace", "atom_fourier"])
def test_a_source_routes_the_q_space_backends_through_the_loader(
        backend, tmp_path, monkeypatch):
    """The whole point: a source must never be loaded rank-locally here."""
    source = _source(tmp_path, final_dyn=_Dyn(), final_T=100.0,
                     n_configs=64)
    monkeypatch.setattr(
        SP.EnsembleSource, "load_ensemble",
        lambda self: pytest.fail(
            "a q-space backend must not replicate the ensemble"))

    calls = {}

    def record(name):
        def loader(*args, **kwargs):
            calls["name"] = name
            calls["args"] = args
            calls["kwargs"] = kwargs
            return "engine"
        return loader

    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceAtomFourier as QAF
    monkeypatch.setattr(QL, "load_distributed_tdscha", record("qspace"))
    monkeypatch.setattr(QAF, "load_distributed_atom_fourier_tdscha",
                        record("atom_fourier"))

    options = {"lo_to_split": None}
    if backend == "atom_fourier":
        options["fine_mesh"] = (2, 2, 2)
    engine = workflow.create_backend(source, backend, options,
                                     use_symmetries=False)

    assert engine == "engine"
    assert calls["name"] == backend
    assert calls["args"][:2] == (source.data_dir, source.population)
    assert calls["args"][2] is source.generating_dyn
    assert calls["args"][3] == 250.0
    assert calls["kwargs"]["final_dyn"] is source.converged_dyn
    assert calls["kwargs"]["final_T"] == 100.0
    assert calls["kwargs"]["n_configs"] == 64
    assert calls["kwargs"]["use_symmetries"] is False
    if backend == "atom_fourier":
        assert calls["args"][4] == (2, 2, 2)


def test_the_interpolated_backend_asks_for_its_mesh(tmp_path):
    with pytest.raises(ValueError, match="fine_mesh"):
        workflow.create_backend(_source(tmp_path), "atom_fourier", {})


def test_an_unknown_backend_is_refused_before_anything_is_loaded(tmp_path):
    with pytest.raises(ValueError, match="Unsupported spectroscopy backend"):
        workflow.create_backend(_source(tmp_path), "nonsense", {})
