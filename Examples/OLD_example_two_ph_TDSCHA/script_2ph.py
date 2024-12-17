from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

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

def identify_polarization():
    """
    IDENTIFY THE POLARIZATION VECTORS
    """
            
    in_pol = np.zeros(3)
    out_pol = np.zeros(3)
    in_pol2 = None
    out_pol2 = None
    
    if which_pol in all_pol[:3]: # xx yy zz
        if which_pol == all_pol[0]:
            INDEX = 0
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = in_pol
            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))
        elif which_pol == all_pol[1]:
            INDEX = 1
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = in_pol
            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))
        else:
            INDEX = 2
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = in_pol
            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))

    elif which_pol in all_pol[3:-3]: #xy xz yz
        if which_pol == all_pol[3]:
            INDEX = 3
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = pol_dic[all_pol[INDEX][1]]
            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))
        elif which_pol == all_pol[4]:
            INDEX = 4
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = pol_dic[all_pol[INDEX][1]]
            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))
        else:
            INDEX = 5
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = pol_dic[all_pol[INDEX][1]]
            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))
    else:
        if which_pol == all_pol[6]:
            INDEX = 6
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = in_pol

            in_pol2  = pol_dic[all_pol[INDEX][1]]
            out_pol2 = in_pol2

            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))
            print('in pol2 \n{} \n out pol2 \n{}'.format(in_pol2,out_pol2))
        elif which_pol == all_pol[7]:
            INDEX = 7
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = in_pol

            in_pol2  = pol_dic[all_pol[INDEX][1]]
            out_pol2 = in_pol2

            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))
            print('in pol2 \n{} \n out pol2 \n{}'.format(in_pol2,out_pol2))
        else:
            INDEX = 8
            print('Polarizattion {}'.format(all_pol[INDEX]))
            in_pol  = pol_dic[all_pol[INDEX][0]]
            out_pol = in_pol

            in_pol2  = pol_dic[all_pol[INDEX][1]]
            out_pol2 = in_pol2

            print('in pol \n{} \n out pol \n{}'.format(in_pol,out_pol))
            print('in pol2 \n{} \n out pol2 \n{}'.format(in_pol2,out_pol2))
            
    print()

    return in_pol, out_pol, in_pol2, out_pol2

if __name__ == "__main__":
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    len_input = len(sys.argv[1:])
    
    
    # Temperature
    T = 400.
    print('The temperature (K) is {}'.format(T))
    print()
    
    all_pol = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xy2', 'xz2', 'yz2']
    pol_dic = {'x': np.array([1,0,0]), 'y': np.array([0,1,0]), 'z' : np.array([0,0,1])}
    # Select the polarization
    which_pol = sys.argv[1] 
    if not(which_pol in all_pol):
        raise ValueError('Try with {}'.format(all_pol))
    # Get the polarization vectors
    in_pol, out_pol, in_pol2, out_pol2 = identify_polarization()
            
    # Use of two phonon
    which_ph = sys.argv[2]
    if which_ph == '1':
        print('One ph eff was selected')
        use_two_ph = False
    elif which_ph == '2':
        print('Two ph eff was selected')
        use_two_ph = True
    else:
        raise ValueError('Try with 1 or 2')
    print()
        
    # Use of symmetries in the Raman vertex
    which_symm = sys.argv[3]
    if which_symm == 'symm':
        print('Symmetrize the Raman vertex')
        symmetrize_raman = True
    elif which_symm == 'nosymm':
        print('NOT symmetrize the Raman vertex')
        symmetrize_raman = False
    else:
        raise ValueError('Try with symm or nosymm')
    print()
    
    ####### ENSEMBLE PREPARATION #######
    # Get the dynmats on 3x3x3
    dyn = CC.Phonons.Phonons('final_dyn_T400_',4)
    # This ensemble is used to average the Raman tensors
    ens = sscha.Ensemble.Ensemble(dyn, T0 = T)
    
    # Get the supercell structures on which we compute the Raman tensor
    dir_structures = 'structures/'
    # Get the number of supercell structure on 3x3x3
    N_structures = 0
    for fname in os.listdir(dir_structures):
        if fname.endswith('.scf'):
            head_file = fname[:-10]
            N_structures += 1
    print('Reading the {} structures on which we compute the Raman tensors'.format(N_structures))
    print()
    all_structures = []
    for i in range(N_structures):
        my_struct = CC.Structure.Structure()
        my_struct.read_scf(os.path.join(dir_structures,'{}_{:05}.scf'.format(head_file,i)))
        all_structures.append(my_struct)
    
    # Init from the structures
    print('Init the ensemble form the structures')
    print()
    ens.init_from_structures(all_structures)
 
    # Load the Raman tensors. the shape must be N_configs,
    raman_tensors = np.load('raman_tensors.npy')
    if raman_tensors.shape != (ens.N, 3, 3, 3 * dyn.structure.N_atoms * np.prod(dyn.GetSupercell())):
        raise ValueError('The shape must be N_configs, 3, 3, 3 * N_at_sc')

    
    
    ####### LANCZOS PREPARATION ########
    lanczos = DL.Lanczos(ens)
    print('Now the Lanzos is initialized with the same ensemble that we use to average the Raman tensor')
    print()
    # IF you want to perfrom an anharmonic calculations initialize the lanczos with the ensemble containing the energies/forces configurations
    # Then use another ensemble to average the Raman tensors in prepare_anharmonic_FT
    
    # Perfrom an harmonic calcularos
    lanczos.ignore_v3 = True
    lanczos.ignore_v4 = True
    lanczos.use_wigner = True
    # Reset and initialize
    lanczos.reset()
    lanczos.init() # use_symmetries = True depractaed
    
    # I pass to prepare_anharmonic_Raman_FT the ensemble ens which is the one on which we perform the averages of the Raman tensors
    if (out_pol2 is None) and (in_pol2 is None):
        lanczos.prepare_anharmonic_raman_FT(raman = raman_tensors, raman_eq = None,\
                                            pol_in = in_pol, pol_out = out_pol,\
                                            add_two_ph = use_two_ph, symmetrize = symmetrize_raman, ensemble = ens)
    else:
        lanczos.prepare_anharmonic_raman_FT(raman = raman_tensors, raman_eq = None,\
                                            pol_in = in_pol, pol_out = out_pol, mixed = True, pol_in_2 = in_pol2, pol_out_2 = out_pol2,\
                                            add_two_ph = use_two_ph, symmetrize = symmetrize_raman, ensemble = ens)
        
    # Number of Lanczos steps
    steps = 100
    # Save each 100 steps (save only the last step)
    save = 100
    dir_data_lanc = 'data/raman_{}_pol_{}_Nraman_{}_symm_{}_steps_{}'.format(which_ph, which_pol, int(ens.N), symmetrize_raman, steps)
    
    if os.path.isdir(dir_data_lanc):
        raise ValueError('There is already a calculation with this name')
    optimize = True
    lanczos.run_FT(steps, verbose = True, debug = False, save_dir = dir_data_lanc,\
                   prefix = 'lanczos', save_each = save, run_simm = lanczos.use_wigner, optimized = optimize)
    
    if len(os.listdir(dir_data_lanc)) == 0:
        lanczos.save_status("%s/%s_STEP%d" % (dir_data_lanc, 'lanczos', len(lanczos.a_coeffs) - 1))
        
    