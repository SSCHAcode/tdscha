"""
MPI worker script for test_mpi_parallel.py.

Run via: mpirun -np N python _mpi_worker.py --n-steps 5 --output result.npz

Runs a Lanczos computation under MPI and saves the results (rank 0 only).
"""

import numpy as np
import argparse
import os
import sys

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Units

import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL
from tdscha.Parallel import am_i_the_master


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)

    T = 250
    NQIRR = 3

    dyn = CC.Phonons.Phonons("data/dyn_gen_pop1_", NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin("data", 1)

    nat_uc = dyn.structure.N_atoms
    ec = np.zeros((nat_uc, 3, 3))
    ec[0] = np.eye(3)
    ec[1] = -np.eye(3)

    lanc = DL.Lanczos(ens)
    lanc.gamma_only = True
    lanc.mode = DL.MODE_FAST_JULIA
    lanc.init(use_symmetries=True)
    lanc.prepare_ir(effective_charges=ec, pol_vec=np.array([1., 0., 0.]))
    lanc.run_FT(args.n_steps)

    w_cm = np.linspace(0, 400, 1000)
    w_ry = w_cm / CC.Units.RY_TO_CM
    smearing_ry = 5.0 / CC.Units.RY_TO_CM
    gf = lanc.get_green_function_continued_fraction(
        w_ry, smearing=smearing_ry, use_terminator=False)
    spectrum = -np.imag(gf)

    if am_i_the_master():
        np.savez(args.output,
                 a_coeffs=np.array(lanc.a_coeffs),
                 b_coeffs=np.array(lanc.b_coeffs),
                 spectrum=spectrum)


if __name__ == "__main__":
    main()
