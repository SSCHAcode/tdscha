from __future__ import print_function

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons

import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL

import scipy, scipy.sparse

from tdscha.Parallel import pprint as print

import sys, os
import mpi4py



def test_lanczos():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    T = 250
    NQIRR = 3

    dyn = CC.Phonons.Phonons("data/dyn_gen_pop1_", NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin("data", 1)
    
    save_dir = 'lanc_wigner'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    lanc = DL.Lanczos(ens)
    N_STEPS = 5
    lanc.ignore_harmonic = False
    lanc.ignore_v3 = False
    lanc.ignore_v4 = False
    lanc.use_wigner = True
    lanc.init(use_symmetries = True)
    lanc.prepare_mode(10)

    lanc.run_FT(2 * N_STEPS, save_dir = 'wigner_nowigner_julia', debug = False, run_simm = lanc.use_wigner, prefix = 'lanczos_wigner')


if __name__ == "__main__":
    test_lanczos()
