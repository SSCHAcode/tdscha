from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

import sscha
import tdscha, tdscha.DynamicalLanczos as DL
import tdscha.Perturbations as perturbations
import sscha.Ensemble

import time

import scipy, scipy.sparse

import cellconstructor.Settings as Parallel
from tdscha.Parallel import pprint as print
from tdscha.Parallel import *

import cellconstructor.Units as units

import sys, os


def prepare_raman_unpol(d3 = False, d4 = False,
                        steps = 4, N_polarizations = 7, debug = False):
    """
    PREPARE THE RAMAN CALCULATIONS
    
    Parameters:
    -----------
        -d3: bool, if true we include d3
        -d4: bool, if true we include d4
        -steps: int the Lanczos steps to do
        -cluster: bool
        -N_polarization: the number of polarization to get the unpolarized signal
    """
    if N_polarizations > 7:
        raise ValueError('The number of Raman polarization larger than 7')
        
    # The SCHA dynamical matrix
    dyn = CC.Phonons.Phonons('../scha_ensemble/dyn_gen_pop1_', 1)

    # Get the SCHA ensemble
    ens = sscha.Ensemble.Ensemble(dyn, 0.)
    # Load the ensemble
    ens.load_bin('../scha_ensemble')
    # Here you can eventually add the reweighting changing the ens.current_dyn
    # Assign the raman tns to the current dyn (3, 3, 3 * N_at_uc)
    ens.current_dyn.raman_tensor =  np.load('eq_ramantns.npy')
    
    # Get the Raman tensor ensemble
    # The Raman tensors computed on the enemble generated shape is (ens.N, 3, 3, 3 * N_at_sc)
    raman_tns_ens = np.load('raman_tns_ens.npy')
    
    for i in range(N_polarizations):
        print('\nUNPOLARIZED RAMAN component {}'.format(i))
        # Prepare the Lanczos
        lanczos = DL.Lanczos(ens, use_wigner = True)
        # Chose the anharmonic vertex
        lanczos.ignore_v3 = not d3
        lanczos.ignore_v4 = not d4
        # Set up the mode of execution
        lanczos.mode = DL.MODE_FAST_JULIA
        
        # Save the Lanczos status in a custom directory
        dir_status = 'tdscha_raman_'
        if d3:
            dir_status += 'D3_'
        if d4:
            dir_status += 'D4_'
        dir_status += '{}'.format(i)

        os.mkdir(dir_status)
        os.chdir(dir_status)
        
        # Prepare the unpolarized raman
        lanczos.prepare_unpolarized_raman_FT(index = i, debug = False,
                                             eq_raman_tns = ens.current_dyn.raman_tensor,
                                             use_symm = True,
                                             ens_av_raman = ens, raman_tns_ens = raman_tns_ens,
                                             add_2ph = True)
    
        #Save the status as we run TDSCHA with another script in this directory
        lanczos.save_status('initial')
        lanczos.dyn.save_qe('lanczos_dyn')
            
        os.chdir('../')
    
    
    
    
if __name__ == "__main__":
    """
    PREPARE THE LANCZOS CALCULATIONS
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    prepare_raman_unpol()