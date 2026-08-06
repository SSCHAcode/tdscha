"""End-to-end tests for spectroscopy planning, restart, and analysis."""

import json
from pathlib import Path

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
        tensor = np.arange(27, dtype=float).reshape(3, 3, 3) / 10
        self.raman_tensor = (tensor + tensor.swapaxes(0, 1)) / 2

    @staticmethod
    def GetSupercell():
        return np.ones(3, dtype=int)


class _Ensemble:
    def __init__(self):
        self.current_dyn = _Dyn()
        self.current_T = 300.0


class _FakeEngine:
    created = 0

    def __init__(self, temperature):
        type(self).created += 1
        self.T = temperature
        self.a_coeffs = []
        self.b_coeffs = []
        self.c_coeffs = []
        self.perturbation_modulus = 1.0
        self.use_wigner = True
        self.reverse_L = False
        self.shift_value = 0.0

    def init(self, use_symmetries=True):
        self.use_symmetries = use_symmetries

    def _prepare_gamma_cartesian_perturbation(self, vector):
        self.vector = np.asarray(vector)
        self.perturbation_modulus = float(self.vector @ self.vector)

    def run_FT(self, count, verbose=False, **kwargs):
        for _ in range(count):
            self.a_coeffs.append(-0.4)
            self.b_coeffs.append(0.02)
            self.c_coeffs.append(0.02)

    def save_status(self, path):
        np.savez_compressed(
            path, a=self.a_coeffs, b=self.b_coeffs, c=self.c_coeffs)

    def load_status(self, path):
        with np.load(path) as archive:
            self.a_coeffs = list(archive["a"])
            self.b_coeffs = list(archive["b"])
            self.c_coeffs = list(archive["c"])

    def save_abc(self, path):
        abc = np.column_stack(
            (self.a_coeffs, self.b_coeffs, self.c_coeffs))
        np.savetxt(
            path, abc,
            header="perturbation_modulus = {}\na; b; c".format(
                self.perturbation_modulus))


class _InterruptingEngine(_FakeEngine):
    def run_FT(self, count, verbose=False, **kwargs):
        if self.a_coeffs:
            raise RuntimeError("simulated interruption")
        super().run_FT(count, verbose=verbose, **kwargs)


def test_structure_symmetry_reduces_cubic_ir_to_one_run():
    ensemble = _Ensemble()
    job = SP.Spectroscopy(ensemble, backend="real")
    job.add_ir_unpolarized("powder")
    plan = job.plan_calculations()

    assert plan["group_order"] == 48
    assert plan["n_requested_components"] == 3
    assert plan["n_independent_runs"] == 1


def test_separately_named_symmetry_equivalent_requests_share_one_run():
    ensemble = _Ensemble()
    job = SP.Spectroscopy(ensemble, backend="real")
    job.add_ir_polarized([1, 0, 0], "ir_x")
    job.add_ir_polarized([0, 1, 0], "ir_y")
    plan = job.plan_calculations()

    assert plan["n_requested_components"] == 2
    assert plan["n_independent_runs"] == 1
    assert (plan["request_components"]["ir_x"][0]["run_id"] ==
            plan["request_components"]["ir_y"][0]["run_id"])


def test_anisotropic_mesh_excludes_incompatible_cubic_rotations():
    ensemble = _Ensemble()
    ensemble.current_dyn.GetSupercell = lambda: np.array([1, 2, 3])
    group, _ = SP.get_gamma_symmetry_representation(
        ensemble.current_dyn.structure,
        supercell=ensemble.current_dyn.GetSupercell())
    assert len(group) == 8


def test_run_resume_load_and_analyze(monkeypatch, tmp_path):
    ensemble = _Ensemble()
    _FakeEngine.created = 0

    def make_engine(_ensemble, _backend, _options, **_):
        return _FakeEngine(_ensemble.current_T)

    monkeypatch.setattr(workflow, "create_backend", make_engine)
    job = SP.Spectroscopy(
        ensemble, backend="real", workdir=tmp_path / "spectroscopy")
    job.add_ir_unpolarized("powder")
    job.run(5, save_each=2, verbose=False)

    assert _FakeEngine.created == 1
    manifest_path = tmp_path / "spectroscopy" / "manifest.json"
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert len(manifest["runs"]) == 1
    assert next(iter(manifest["runs"].values()))["state"] == "complete"
    assert next(iter(manifest["runs"].values()))["completed_steps"] == 5
    analysis = next(iter(manifest["runs"].values()))["analysis"]
    assert "symmetry_reduction" in analysis
    run_id = next(iter(manifest["runs"]))
    assert (tmp_path / "spectroscopy" / "runs" / run_id /
            "status.npz").exists()

    # A completed resume does not construct or rerun the backend.
    job.run(5, save_each=2, verbose=False)
    assert _FakeEngine.created == 1

    loaded = SP.Spectroscopy.load(tmp_path / "spectroscopy")
    frequencies = np.linspace(0.1, 0.2, 5)
    response = loaded.response(
        "powder", frequencies, use_terminator=False, smearing=0.01)
    assert response.shape == frequencies.shape
    assert np.all(response >= 0)
    np.testing.assert_allclose(
        loaded.ir_susceptibility(
            "powder", frequencies, use_terminator=False, smearing=0.01),
        2 * loaded.green_function(
            "powder", frequencies, use_terminator=False, smearing=0.01))

    epsilon = loaded.dielectric_function(
        "powder", frequencies, epsilon_infinity=np.eye(3) * 2.5,
        ionic_prefactor=0.0, use_terminator=False, smearing=0.01)
    np.testing.assert_allclose(epsilon, 2.5)

    # The full tensor is persisted in the manifest, so load-only analysis
    # needs no duplicate epsilon_infinity argument.
    inferred = loaded.dielectric_function(
        "powder", frequencies, ionic_prefactor=0.0,
        use_terminator=False, smearing=0.01)
    np.testing.assert_allclose(inferred, 2.5)

    # The portable abc files are sufficient for automatic load-only analysis.
    result_path = (tmp_path / "spectroscopy" / "runs" / run_id /
                   "result.npz")
    result_path.unlink()
    abc_loaded = SP.Spectroscopy.load(tmp_path / "spectroscopy")
    np.testing.assert_allclose(
        abc_loaded.response(
            "powder", frequencies, use_terminator=False, smearing=0.01),
        response)


def test_dielectric_tensor_is_projected_not_scalarized(monkeypatch, tmp_path):
    ensemble = _Ensemble()
    ensemble.current_dyn.dielectric_tensor = np.array([
        [2.0, 0.4, 0.0], [0.4, 4.0, 0.0], [0.0, 0.0, 8.0]])
    monkeypatch.setattr(
        workflow, "create_backend",
        lambda ens, backend, options, **_: _FakeEngine(ens.current_T))
    job = SP.Spectroscopy(
        ensemble, backend="real", workdir=tmp_path / "tensor")
    direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    job.add_ir_polarized(direction, "polarized")
    job.add_ir_unpolarized("powder")
    job.run(2, verbose=False)
    loaded = SP.Spectroscopy.load(tmp_path / "tensor")
    frequencies = np.linspace(0.1, 0.2, 3)

    polarized = loaded.dielectric_function(
        "polarized", frequencies, ionic_prefactor=0.0,
        use_terminator=False, smearing=0.01)
    powder = loaded.dielectric_function(
        "powder", frequencies, ionic_prefactor=0.0,
        use_terminator=False, smearing=0.01)
    np.testing.assert_allclose(polarized, 3.4)
    np.testing.assert_allclose(powder, 14.0 / 3.0)

    override = np.diag([10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        loaded.dielectric_function(
            "polarized", frequencies, epsilon_infinity=override,
            ionic_prefactor=0.0, use_terminator=False, smearing=0.01),
        15.0)


def test_dielectric_function_uses_supercell_volume(monkeypatch, tmp_path):
    """The default IR prefactor must be 4*pi/V_supercell, not 4*pi/V_unit.

    The Lanczos perturbation carries sqrt(n_cell) (prepare_ir), so the Green
    function already includes the n_cell factor and the volume in
    epsilon_inf + (4*pi/Omega)*chi_ionic must be the supercell volume.
    """
    ensemble = _Ensemble()
    ensemble.current_dyn.GetSupercell = lambda: np.array([2, 1, 2])
    monkeypatch.setattr(
        workflow, "create_backend",
        lambda ens, backend, options, **_: _FakeEngine(ens.current_T))
    job = SP.Spectroscopy(
        ensemble, backend="real", workdir=tmp_path / "supercell_ir")
    job.add_ir_unpolarized("powder")
    job.run(2, verbose=False)
    loaded = SP.Spectroscopy.load(tmp_path / "supercell_ir")

    from cellconstructor.Units import A_TO_BOHR
    manifest = loaded._manifest_data
    assert manifest["supercell_volume_angstrom3"] == pytest.approx(
        manifest["unit_cell_volume_angstrom3"] * 4)
    assert manifest["unit_cell_volume_angstrom3"] == pytest.approx(125.0)

    frequencies = np.linspace(0.1, 0.2, 3)
    options = dict(use_terminator=False, smearing=0.01)
    default = loaded.dielectric_function("powder", frequencies, **options)

    supercell_bohr3 = (
        manifest["supercell_volume_angstrom3"] * float(A_TO_BOHR)**3)
    explicit = loaded.dielectric_function(
        "powder", frequencies,
        ionic_prefactor=4 * np.pi / supercell_bohr3, **options)
    np.testing.assert_allclose(default, explicit)

    unit_bohr3 = (
        manifest["unit_cell_volume_angstrom3"] * float(A_TO_BOHR)**3)
    wrong = loaded.dielectric_function(
        "powder", frequencies,
        ionic_prefactor=4 * np.pi / unit_bohr3, **options)
    assert not np.allclose(default, wrong)


def test_backend_physics_flags_are_public_and_legacy_compatible():
    ensemble = _Ensemble()
    explicit = SP.Spectroscopy(
        ensemble, ignore_v3=True, ignore_v4=False,
        lo_to_split=[1, 2, 3])
    manifest = explicit.manifest()
    assert manifest["ignore_v3"] is True
    assert manifest["ignore_v4"] is False
    assert manifest["lo_to_split"] == [1.0, 2.0, 3.0]
    assert "ignore_v3" not in manifest["backend_options"]

    legacy = SP.Spectroscopy(
        ensemble,
        backend_options={"ignore_v3": True, "ignore_v4": True,
                         "lo_to_split": [0, 0, 1]})
    assert legacy.ignore_v3 and legacy.ignore_v4
    assert legacy.lo_to_split == [0.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="Conflicting 'ignore_v3'"):
        SP.Spectroscopy(
            ensemble, ignore_v3=False,
            backend_options={"ignore_v3": True})


def test_restart_manifest_rejects_changed_requests(monkeypatch, tmp_path):
    ensemble = _Ensemble()
    monkeypatch.setattr(
        workflow, "create_backend",
        lambda ens, backend, options, **_: _FakeEngine(ens.current_T))
    workdir = tmp_path / "spectroscopy"
    first = SP.Spectroscopy(ensemble, backend="real", workdir=workdir)
    first.add_ir_polarized([1, 0, 0], "ir")
    first.run(2, verbose=False)

    changed = SP.Spectroscopy(ensemble, backend="real", workdir=workdir)
    changed.add_ir_polarized([0, 1, 0], "ir")
    try:
        changed.run(2, verbose=False)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("Changed requests must invalidate a restart")


def test_interrupted_run_resumes_from_native_status(monkeypatch, tmp_path):
    ensemble = _Ensemble()
    workdir = tmp_path / "interrupted"
    monkeypatch.setattr(
        workflow, "create_backend",
        lambda ens, backend, options, **_: _InterruptingEngine(ens.current_T))
    interrupted = SP.Spectroscopy(
        ensemble, backend="real", workdir=workdir,
        use_symmetries=False)
    interrupted.add_ir_polarized([1, 0, 0], "ir")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.run(5, save_each=2, verbose=False)

    monkeypatch.setattr(
        workflow, "create_backend",
        lambda ens, backend, options, **_: _FakeEngine(ens.current_T))
    resumed = SP.Spectroscopy(
        ensemble, backend="real", workdir=workdir,
        use_symmetries=False)
    resumed.add_ir_polarized([1, 0, 0], "ir")
    resumed.run(5, save_each=2, verbose=False)

    with open(workdir / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    entry = next(iter(manifest["runs"].values()))
    assert entry["state"] == "complete"
    assert entry["completed_steps"] == 5


def test_unpolarized_and_polarized_raman_assembly(monkeypatch, tmp_path):
    ensemble = _Ensemble()
    monkeypatch.setattr(
        workflow, "create_backend",
        lambda ens, backend, options, **_: _FakeEngine(ens.current_T))
    job = SP.Spectroscopy(
        ensemble, backend="real", workdir=tmp_path / "raman",
        use_symmetries=False)
    job.add_raman_unpolarized("powder")
    job.add_raman_polarized(
        [1, 1, 0], [0, 1, 1], name="polarized")
    plan = job.plan_calculations()
    assert plan["n_requested_components"] == 8
    job.run(2, verbose=False)

    frequencies = np.linspace(0.1, 0.2, 4)
    response = job.raman_spectrum(
        "powder", frequencies, kind="response",
        use_terminator=False, smearing=0.01)
    stokes = job.raman_spectrum(
        "powder", frequencies, kind="stokes",
        use_terminator=False, smearing=0.01)
    anti = job.raman_spectrum(
        "powder", frequencies, kind="anti_stokes",
        use_terminator=False, smearing=0.01)
    assert np.all(response >= 0)
    assert np.all(stokes > anti)
    np.testing.assert_allclose(
        stokes - anti, response, rtol=1e-12, atol=1e-12)

    polarized = job.raman_spectrum(
        "polarized", frequencies, kind="response",
        use_terminator=False, smearing=0.01)
    assert np.all(polarized >= 0)


@pytest.mark.parametrize("scale", [1.0, 1.0e-14])
def test_symmetry_forbidden_raman_components_need_no_run(scale):
    ensemble = _Ensemble()
    ensemble.current_dyn.raman_tensor[:] = 0.0
    # A cubic T2-like Raman tensor: only the normalized xy component is
    # active.  The other six powder-average components are exactly zero and
    # must contribute zero without creating invalid Lanczos perturbations.
    ensemble.current_dyn.raman_tensor[0, 1, 0] = scale
    ensemble.current_dyn.raman_tensor[1, 0, 0] = scale

    job = SP.Spectroscopy(
        ensemble, backend="real", use_symmetries=False)
    job.add_raman_unpolarized("powder")
    plan = job.plan_calculations()

    assert plan["n_requested_components"] == 7
    assert plan["n_independent_runs"] == 1
    components = plan["request_components"]["powder"]
    assert sum(component["run_id"] is None for component in components) == 6
