from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
# from mpi4py import MPI

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


def prepare_ir_unpol(d3 = False, d4 = False, N_polarizations = 3):
    """
    PREPARE THE IR CALCULATIONS
    
    Parameters:
    -----------
        -d3: bool, if true we include d3
        -d4: bool, if true we include d4
        -N_polarization: the number of polarization to get the unpolarized IR signal
    """
    if N_polarizations > 3:
        raise ValueError('The number of IR polarization larger than 3')
        
    # The SCHA dynamical matrix
    dyn = CC.Phonons.Phonons('../scha_ensemble/dyn_gen_pop1_', 1)

    # Get the SCHA ensemble
    ens = sscha.Ensemble.Ensemble(dyn, 0.)
    # Load the SCHA ensemble
    ens.load_bin('../scha_ensemble')
    # Assign the effective charges to the current dyn (N_at_uc, E_comp, cart_comp)
    ens.current_dyn.effective_charges =  np.load('eq_effchgs.npy')
    
    # Get the eff charge ensemble (ens.N, N_at_sc, E_comp, cart_comp)
    effchgs_ens = np.load('eff_chgs_ens.npy')
    
    for i in range(N_polarizations):
        print('\nUNPOLARIZED IR component {}'.format(i))
        # Prepare the Lanczos
        lanczos = DL.Lanczos(ens, use_wigner = True)
        # Chose the anharmonic vertex
        lanczos.ignore_v3 = not d3
        lanczos.ignore_v4 = not d4
        # Set up the mode of execution
        lanczos.mode = DL.MODE_FAST_JULIA
        
        # Save the Lanczos status in a custom directory
        dir_status = 'tdscha_ir_'
        if d3:
            dir_status += 'D3_'
        if d4:
            dir_status += 'D4_'
        dir_status += '{}'.format(i)

        os.mkdir(dir_status)
        os.chdir(dir_status)
        
        # Prepare the unpolarized IR
        lanczos.prepare_anharmonic_ir_FT(ec = effchgs_ens, ec_eq = ens.current_dyn.effective_charges,
                                         pol_vec_light = np.eye(3)[i,:], add_two_ph = True, symmetrize = True, ensemble = ens)
        #Save the status
        lanczos.save_status('initial')
        lanczos.dyn.save_qe('lanczos_dyn')
            
        os.chdir('../')
    
    
if __name__ == "__main__":
    """
    PREPARE THE LANCZOS CALCULATIONS
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    prepare_ir_unpol()