"""Direct full-group/coset parity on a cubic 3x3x3 q mesh.

This test intentionally stops at ``apply_anharmonic_FT``.  Its second
application feeds the first anharmonic image back into the operator, so the
input has a nonzero two-phonon sector and exercises the stabilizer projection
of both ``f`` and ``d2v_dr2``.  Every q point is covered; on an odd mesh only
Gamma is time-reversal invariant.
"""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JULIA_NUM_THREADS", "1")

import cellconstructor.Phonons as Phonons
import sscha.Ensemble

try:
    import tdscha.QSpaceLanczos as QL
    import tdscha.QSpaceAtomFourier as QAF
    _AVAILABLE = QL.__JULIA_EXT__
except Exception:
    _AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _AVAILABLE, reason="QSpaceLanczos/Julia not available")

DATA = Path(__file__).parents[1] / "test_julia" / "data"
MESH = (3, 3, 3)


@pytest.fixture(scope="module")
def cubic_odd_mesh_ensemble():
    coarse = Phonons.Phonons(str(DATA / "dyn_gen_pop1_"), 3)
    dyn = coarse.Interpolate(
        coarse.GetSupercell(), MESH, symmetrize=True)
    # The algebraic parity does not assume a model for Y: Julia explicitly
    # replicates and rotates each X/Y configuration.  A deterministic random
    # force field is the sharpest small-N probe because it avoids accidental
    # zeros in D3 and D4 while only four configurations keep this test cheap.
    np.random.seed(731)
    ensemble = sscha.Ensemble.Ensemble(dyn, 250.0)
    ensemble.generate(4)
    rng = np.random.default_rng(991)
    ensemble.forces = rng.normal(scale=2e-3, size=ensemble.forces.shape)
    n_configurations = len(ensemble.structures)
    ensemble.energies = np.zeros(n_configurations)
    ensemble.force_computed = np.ones(n_configurations, dtype=bool)
    ensemble.init()
    return ensemble


def _new_engine(ensemble):
    engine = QL.QSpaceLanczos(ensemble, lo_to_split=None)
    engine.init(use_symmetries=True)
    return engine


def test_all_q_full_replica_matches_coset_and_inner_group(
        cubic_odd_mesh_ensemble):
    full = _new_engine(cubic_odd_mesh_ensemble)
    reduced = _new_engine(cubic_odd_mesh_ensemble)

    assert tuple(full.dyn.GetSupercell()) == MESH
    assert full.n_q == 27
    assert full.n_syms_qspace == 48

    reciprocal = full.uci_structure.get_reciprocal_vectors() / (2 * np.pi)
    tri = []
    reductions = []
    # Opposite sublattice displacement: optical at Gamma and nonzero at each
    # finite q.  A Cartesian vector is preferable to a band index because it
    # is insensitive to arbitrary gauges within degenerate eigenspaces.
    perturbation = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])

    for iq, qpoint in enumerate(full.q_points):
        # Explicitly determine TRI status modulo a reciprocal lattice vector.
        import cellconstructor.Methods as Methods
        tri.append(Methods.get_min_dist_into_cell(
            reciprocal, qpoint, -qpoint) < 1e-7)

        full.prepare_perturbation_q(iq, perturbation)
        reduced.prepare_perturbation_q(iq, perturbation)
        metadata = reduced.configure_qspace_perturbation_symmetry()
        reductions.append(metadata)

        first_full = full.apply_anharmonic_FT()
        first_reduced = reduced.apply_anharmonic_FT()
        np.testing.assert_allclose(
            first_reduced, first_full, rtol=3e-9, atol=3e-11,
            err_msg="first anharmonic application differs at iq={}".format(iq))

        # The first image contains d2v_dr2 in both Wigner two-phonon sectors.
        assert np.linalg.norm(first_full[full.n_bands:]) > 1e-13
        full.psi = first_full.copy()
        reduced.psi = first_reduced.copy()
        second_full = full.apply_anharmonic_FT()
        second_reduced = reduced.apply_anharmonic_FT()
        np.testing.assert_allclose(
            second_reduced, second_full, rtol=3e-9, atol=3e-11,
            err_msg="two-phonon input differs at iq={}".format(iq))

    assert tri == [True] + [False] * 26
    assert reductions[0]["coset_representatives"] < 48
    assert any(item["coset_representatives"] < 48
               for item in reductions[1:])


@pytest.mark.parametrize("coarse_iq", [0, 1])
def test_atom_fourier_uses_coarse_stabilizer_for_gamma_and_nontri(
        cubic_odd_mesh_ensemble, coarse_iq):
    full = QAF.QSpaceAtomFourierLanczos(
        cubic_odd_mesh_ensemble, fine_mesh=MESH, lo_to_split=None)
    reduced = QAF.QSpaceAtomFourierLanczos(
        cubic_odd_mesh_ensemble, fine_mesh=MESH, lo_to_split=None)
    full.init(use_symmetries=True)
    reduced.init(use_symmetries=True)
    fine_iq = int(full._fine_of_coarse[coarse_iq])
    perturbation = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    full.prepare_perturbation_q(fine_iq, perturbation)
    reduced.prepare_perturbation_q(fine_iq, perturbation)
    metadata = reduced.configure_qspace_perturbation_symmetry()

    assert metadata["full_group_order"] == 48
    assert metadata["coset_representatives"] < 48
    first_full = full.apply_anharmonic_FT()
    first_reduced = reduced.apply_anharmonic_FT()
    np.testing.assert_allclose(
        first_reduced, first_full, rtol=3e-9, atol=3e-11)
    full.psi = first_full.copy()
    reduced.psi = first_reduced.copy()
    np.testing.assert_allclose(
        reduced.apply_anharmonic_FT(), full.apply_anharmonic_FT(),
        rtol=3e-9, atol=3e-11)
