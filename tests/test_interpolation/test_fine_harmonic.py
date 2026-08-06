"""``FineHarmonicInterpolation``: the value the distributed loaders inject.

The interpolation is computed by every MPI rank and then handed to a
constructor that would otherwise recompute it.  Two things must hold: it must
be the interpolation that constructor would have produced, and it must be
recognisably *not* it when the inputs differ.  Nothing else in the pipeline
would notice a mismatch -- the ensemble would simply be contracted in a mode
basis that does not belong to it.
"""

import os

import numpy as np
import pytest

import cellconstructor as CC
import cellconstructor.Phonons

import tdscha.QSpaceInterpolation as QI

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "test_julia", "data"))
MESH = (2, 2, 4)

if not os.path.isdir(DATA):
    pytest.skip("q-space test dynamical matrix not available",
                allow_module_level=True)


@pytest.fixture(scope="module")
def dyn():
    return CC.Phonons.Phonons(os.path.join(DATA, "dyn_gen_pop1_"), 3)


@pytest.fixture(scope="module")
def harmonic(dyn):
    return QI.build_fine_harmonic(dyn, MESH)


def test_it_reproduces_the_direct_interpolation(dyn, harmonic):
    q_points, indices = QI.generate_fine_mesh(dyn.structure, MESH)
    frequencies, polarizations = QI.interpolate_dyn_fine(
        dyn, q_points, use_asr=True)

    np.testing.assert_array_equal(harmonic.q_points, q_points)
    np.testing.assert_array_equal(harmonic.indices, indices)
    np.testing.assert_array_equal(harmonic.frequencies, frequencies)
    np.testing.assert_array_equal(harmonic.polarizations, polarizations)
    assert harmonic.n_q == int(np.prod(MESH))
    assert harmonic.n_bands == 3 * dyn.structure.N_atoms


def test_it_accepts_the_matrix_it_came_from(dyn, harmonic):
    harmonic.validate_for(dyn, MESH)
    harmonic.validate_for(dyn.Copy(), MESH)


def test_a_copy_through_an_ensemble_is_still_the_same_matrix(dyn, harmonic):
    """The loader interpolates ``final_dyn``; the constructor sees a copy.

    ``Ensemble`` hands back a ``current_dyn`` whose Gamma block has been
    demoted from complex to real without any value changing.  If the
    identity check noticed that, every distributed interpolated run would be
    refused; if it noticed nothing at all, it would be worthless.
    """
    ensemble_module = pytest.importorskip("sscha.Ensemble")
    ensemble = ensemble_module.Ensemble(dyn, 250.0)
    ensemble.load_bin(DATA, 1)
    harmonic.validate_for(ensemble.current_dyn, MESH)

    ensemble.update_weights(dyn, 250.0)
    harmonic.validate_for(ensemble.current_dyn.Copy(), MESH)


def test_it_refuses_a_different_mesh(dyn, harmonic):
    with pytest.raises(ValueError, match="built on mesh"):
        harmonic.validate_for(dyn, (2, 2, 2))


def test_it_refuses_different_interpolation_settings(dyn, harmonic):
    with pytest.raises(ValueError, match="not built from"):
        harmonic.validate_for(dyn, MESH, use_asr=False)
    with pytest.raises(ValueError, match="not built from"):
        harmonic.validate_for(dyn, MESH, lo_to_split=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not built from"):
        harmonic.validate_for(dyn, MESH, ignore_effective_charges=True)


def test_it_refuses_a_different_dynamical_matrix(dyn, harmonic):
    other = dyn.Copy()
    other.dynmats[0] = np.asarray(other.dynmats[0]) * 1.01
    with pytest.raises(ValueError, match="not built from"):
        harmonic.validate_for(other, MESH)


def test_the_constructor_refuses_a_foreign_interpolation(dyn, harmonic):
    """The injection point itself must reject it, not just the value type."""
    pytest.importorskip("sscha.Ensemble")
    import tdscha.QSpaceAtomFourier as AF

    with pytest.raises(TypeError, match="FineHarmonicInterpolation"):
        AF.QSpaceAtomFourierLanczos(
            ensemble=object(), fine_mesh=MESH,
            harmonic_interpolation="not an interpolation")
