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
    

    lanc = DL.Lanczos(ens)
    N_STEPS = 10
    lanc.ignore_harmonic = False
    lanc.ignore_v3 = False
    lanc.ignore_v4 = False
    lanc.mode = DL.MODE_FAST_JULIA
    lanc.init(use_symmetries = True)
    lanc.prepare_mode(10)

    lanc.run_FT(N_STEPS, save_dir = 'julia', debug = False)

    # HARDCODE the last c value
    gf = lanc.get_green_function_continued_fraction(np.array([0]), smearing = 0)
    w = np.sign(np.real(gf)) * np.sqrt(np.abs(1./np.real(gf))) * CC.Units.RY_TO_CM
    w_reference = -36.93203026
    assert np.isclose(w, w_reference, rtol = 1e-3), "TEST FAILED: LAST C VALUE CALCULATED: {} | EXPECTED C VALUE: {}".format(w, w_reference)


if __name__ == "__main__":
    test_lanczos()
