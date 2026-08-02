"""Regression test for the N_eff integer-truncation bug in
``tdscha.QSpaceLanczos.load_distributed_tdscha``.

The bug
-------
``load_distributed_tdscha`` used to set, on every rank::

    qlanc.N_eff = int(np.sum(qlanc.rho))

The Julia kernel ``get_perturb_averages_qspace`` returns a result already
divided by ``n_syms * sum(rho_local)`` -- a *float*, see ``tdscha_qspace.jl``
(``N_eff = sum(rho)``, ``norm_factor = n_syms * N_eff``).
``_call_julia_qspace_distributed`` then multiplies by ``self.N_eff`` to undo
exactly that division, MPI-Allreduces and divides by ``N_eff_global``.  The
cancellation is exact only if ``self.N_eff == float(sum(rho_local))``.
Truncating to ``int`` breaks it as soon as ``rho`` is non-integer, i.e. after
``ensemble.update_weights`` -- the normal production situation.  The result is
a *silently* wrong (mis-weighted) anharmonic average, hence a wrong Hessian /
Lanczos spectrum.

The test
--------
Run under ``mpirun -np 2``.  Load the in-repo test ensemble
(``tests/test_julia/data``, 10 configs), call ``update_weights`` with a
slightly different dynamical matrix so that ``rho != 1``, then compute the
TDSCHA Lanczos coefficients twice:

  * path A -- ordinary ``QSpaceLanczos`` (full ensemble on every rank, Julia
    work split by ``GoParallel``); ``N_eff`` comes from the base class as
    ``np.sum(rho)`` and is therefore correct;
  * path B -- ``load_distributed_tdscha`` (config slices per rank), which is
    the code path containing the bug.

The two must agree.  On the unfixed code the local weight sums are truncated
(here 4.9461 -> 4 and 4.0589 -> 4), which mis-weights each rank's contribution
by up to ~20%.

Why the Lanczos and not the free-energy Hessian: the GOLD **size 2** ensemble
is purely harmonic (its forces are exactly odd in u, ``f(u)+f(-u) = 2.2e-15``),
so its anharmonic operator contributes nothing and a Hessian-level
serial-vs-distributed comparison on it is vacuous -- it agrees bitwise even on
the buggy code.  GOLD size >= 3 *is* genuinely anharmonic and is the minimum
case for correctness validation.  The ``run_FT`` Lanczos on the small in-repo
ensemble populates the two-phonon sector, is far cheaper than a size-3
Hessian, and its averages go through exactly the same ``N_eff``
normalisation.

Usage
-----
    mpirun -np 2 python test_neff_cast.py     # the actual test
    python test_neff_cast.py                  # re-execs itself under mpirun -np 2

It is also collected by pytest as ``test_neff_not_truncated``.
"""

import os
import subprocess
import sys

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Settings as Parallel
import sscha.Ensemble
import tdscha.QSpaceLanczos as QL
from tdscha.QSpaceLanczos import load_distributed_tdscha

# In-repo test ensemble (the one used by tests/test_qspace/test_distributed.py)
ENS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "test_julia", "data")
POPULATION = 1
NQIRR = 3
T = 250.0

N_STEPS = 6
IQ = 0
SCALE = 1.05     # perturbation of the dynamical matrix used by update_weights


def _load_dyns():
    dyn0 = CC.Phonons.Phonons(os.path.join(ENS_DIR, "dyn_gen_pop1_"), NQIRR)
    # A slightly different dyn, so that update_weights yields non-integer rho.
    dyn_f = dyn0.Copy()
    for i in range(len(dyn_f.dynmats)):
        dyn_f.dynmats[i] = dyn_f.dynmats[i] * SCALE
    return dyn0, dyn_f


def run_mpi():
    """Body of the test; must be executed under mpirun with >= 2 ranks."""
    n_procs = Parallel.GetNProc()
    assert n_procs >= 2, "this test must be run with mpirun -np 2 (got %d)" % n_procs

    dyn0, dyn_f = _load_dyns()

    # ---------------- path A: reference (GoParallel, full ensemble) --------
    ens = sscha.Ensemble.Ensemble(dyn0, T)
    ens.load_bin(ENS_DIR, POPULATION)
    ens.update_weights(dyn_f, T)

    rho = np.asarray(ens.rho, dtype=np.float64)
    assert np.abs(rho - np.round(rho)).max() > 1e-6, \
        "rho is (near) integer: the test would not exercise the bug"

    ref = QL.QSpaceLanczos(ens, lo_to_split=None)
    ref.ignore_v3 = False
    ref.ignore_v4 = False
    ref.init(use_symmetries=True)
    assert not ref._distributed
    band = int(np.argmax(ref.w_q[:, IQ]))
    ref.prepare_mode_q(IQ, band)
    ref.run_FT(N_STEPS, verbose=False)
    a_ref = np.array(ref.a_coeffs, dtype=np.float64)
    b_ref = np.array(ref.b_coeffs, dtype=np.float64)

    # ---------------- path B: distributed (the buggy code path) ------------
    dist = load_distributed_tdscha(ENS_DIR, POPULATION, dyn0, T, lo_to_split=None,
                                   use_symmetries=True,
                                   final_dyn=dyn_f, final_T=T)
    dist.ignore_v3 = False
    dist.ignore_v4 = False
    assert dist._distributed
    dist.prepare_mode_q(IQ, band)
    dist.run_FT(N_STEPS, verbose=False)
    a_dis = np.array(dist.a_coeffs, dtype=np.float64)
    b_dis = np.array(dist.b_coeffs, dtype=np.float64)

    ok = True
    if Parallel.am_i_the_master():
        print("N_eff local  = %r  (sum rho_local = %.10f)"
              % (dist.N_eff, float(np.sum(dist.rho))))
        print("a_ref = %s" % np.array2string(a_ref, precision=10))
        print("a_dis = %s" % np.array2string(a_dis, precision=10))
        for name, x, y in (("a_coeffs", a_ref, a_dis), ("b_coeffs", b_ref, b_dis)):
            scale = max(np.abs(x).max(), 1e-30)
            rel = np.abs(x - y).max() / scale
            print("max |ref - dist| / max|ref|  [%s] = %.6e" % (name, rel))
            if not np.allclose(x, y, rtol=1e-8, atol=1e-12 * scale):
                ok = False
        if not ok:
            print("FAIL: distributed Lanczos differs from the reference. "
                  "N_eff must be float(sum(rho)), not int(sum(rho)).")
        else:
            print("OK: distributed Lanczos matches the GoParallel reference")

    return ok


def _run_under_mpirun():
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.call(["mpirun", "-np", "2", sys.executable,
                            os.path.abspath(__file__), "--inner"], env=env)


def test_neff_not_truncated():
    assert _run_under_mpirun() == 0


if __name__ == "__main__":
    if "--inner" in sys.argv:
        sys.exit(0 if run_mpi() else 1)
    sys.exit(_run_under_mpirun())
