from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import cellconstructor as CC
import cellconstructor.Phonons

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


if __name__ == '__main__':
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # COMMAND TO RUN
    # nohup mpirun --use-hwthread-cpus -np 16 python3 -u ./run_anharmonic_lanczos.py > output_y_1.log &

    # Select the directory
    dir_to_run = sys.argv[1]
    
    steps = 5
    save  = 1
    step_in = 0
    optimize = True
        
    dyn = CC.Phonons.Phonons(os.path.join(dir_to_run, 'lanczos_dyn'), 1)
    ens = sscha.Ensemble.Ensemble(dyn, 0.)
    ens.generate(2)
    
    # Prepare the Lanczos with the dummy ensemble
    lanczos = DL.Lanczos(ens) 
    
    print('Load LANCZO from', os.path.join(dir_to_run,'initial'))
    # Load the status of the Lanczos
    lanczos.load_status(os.path.join(dir_to_run,'initial'))
    
    lanczos.mode = DL.MODE_FAST_JULIA
    lanczos.use_wigner = True
    
    Parallel.barrier()
    if Parallel.am_i_the_master():
        print()
        print('++++++++++++++++++++++++++++++++++')
        print('RUNNING THE LANCZOS WITH D3 AND D4')
        print('The calculation is saved in {}'.format(dir_to_run))
        print('Steps     {}'.format(steps))
        print('Save each {}'.format(save))
        print('D3 = {} D4 = {}'.format(not(lanczos.ignore_v3), not(lanczos.ignore_v4)))
        print('++++++++++++++++++++++++++++++++++')
        print()
        print()

    
    # # Prepare input file for the cluster
    # lanczos.prepare_input_files(root_name = os.path.join(lanczos_dir,"tdscha"), n_steps = steps,\
    #                             start_from_scratch = True, directory = lanczos_dir)
    
    #lanczos.initialized= True
    
    lanczos.run_FT(steps, verbose = True, debug = False,\
                   save_dir = dir_to_run, prefix = 'lanczos',\
                   save_each = save, run_simm = lanczos.use_wigner , optimized = optimize)
    
    Parallel.barrier()
    if Parallel.am_i_the_master():
        np.save(os.path.join(dir_to_run,'steps'), len(lanczos.a_coeffs))
