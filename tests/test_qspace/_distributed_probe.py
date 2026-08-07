"""Worker for test_distributed_loader.py -- run under mpirun, dump a/b/c.

Usage:  python _distributed_probe.py <mode> <data_dir> <out.npz>
        mode = serial-plain | dist-plain | serial-tri | dist-tri
               | oracle-tri | guard-tri

``oracle-tri`` is the build-on-every-rank reference: it replicates the
ensemble during construction, which is exactly what the production loader
avoids, so it exists only to pin the master-only path against it.

``guard-tri`` reintroduces the historical defect on purpose and must fail
loudly rather than hang or return a corrupted object.

Separate processes (rather than one job that builds both) because every
matrix-vector product goes through a collective reduction: a rank that built a
second, non-distributed object and stepped it alone would hang the others.
"""
from __future__ import print_function

import os
import sys

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import sscha
import sscha.Ensemble

import tdscha.QSpaceLanczos as QL
import tdscha.QSpaceAtomFourier as AF
import cellconstructor.Settings as Parallel

T = 250.0
NQIRR = 3
FINE = (2, 2, 4)          # multiple of the 2x2x2 coarse supercell
NSTEPS = 6
BAND = 3
POP = 1


def build(mode, data_dir):
    dyn = CC.Phonons.Phonons(os.path.join(data_dir, "dyn_gen_pop%d_" % POP),
                             NQIRR)
    if mode == "serial-plain":
        ens = sscha.Ensemble.Ensemble(dyn, T)
        ens.load_bin(data_dir, POP)
        lanc = QL.QSpaceLanczos(ens, lo_to_split=None)
        lanc.init(use_symmetries=True)
    elif mode == "dist-plain":
        lanc = QL.load_distributed_tdscha(data_dir, POP, dyn, T,
                                          use_symmetries=True)
    elif mode == "serial-tri":
        ens = sscha.Ensemble.Ensemble(dyn, T)
        ens.load_bin(data_dir, POP)
        lanc = AF.QSpaceAtomFourierLanczos(ens, fine_mesh=FINE)
        lanc.init(use_symmetries=True)
    elif mode == "dist-tri":
        lanc = AF.load_distributed_atom_fourier_tdscha(
            data_dir, POP, dyn, T, fine_mesh=FINE,
            use_symmetries=True)
    elif mode == "guard-tri":
        # Deliberately leaves the collective interpolation inside the
        # constructor, i.e. reintroduces the defect the loader guards
        # against.  The workers must notice that the master's ForceTensor
        # broadcast, not the loader's metadata, reached them.
        class Unprepared(AF.QSpaceAtomFourierLanczos):
            @classmethod
            def prepare_distributed_construction(cls, dyn, **kwargs):
                return {}

        lanc = QL.load_distributed_tdscha(
            data_dir, POP, dyn, T, use_symmetries=True,
            lanczos_class=Unprepared, fine_mesh=FINE)
    elif mode == "oracle-tri":
        lanc = QL.load_distributed_tdscha(
            data_dir, POP, dyn, T, use_symmetries=True,
            lanczos_class=AF.QSpaceAtomFourierLanczos,
            build_on_all_ranks=True, fine_mesh=FINE)
    else:
        raise ValueError("unknown mode %s" % mode)
    return lanc


def main():
    mode, data_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    lanc = build(mode, data_dir)

    # The distributed object must still describe the *global* ensemble.
    n_global = getattr(lanc, "_N_global", lanc.N)

    lanc.prepare_mode_q(0, BAND)
    lanc.run_FT(NSTEPS, verbose=False, reorthogonalize=False, optimized=True)

    if Parallel.am_i_the_master():
        np.savez(out,
                 a=np.asarray(lanc.a_coeffs, dtype=float),
                 b=np.asarray(lanc.b_coeffs, dtype=float),
                 c=np.asarray(lanc.c_coeffs, dtype=float),
                 n_global=n_global,
                 cls=type(lanc).__name__,
                 distributed=bool(getattr(lanc, "_distributed", False)),
                 xq_nq=lanc.X_q.shape[0],
                 n_local=lanc.X_q.shape[1])
        print("%s: wrote %s" % (mode, out))


if __name__ == "__main__":
    main()
