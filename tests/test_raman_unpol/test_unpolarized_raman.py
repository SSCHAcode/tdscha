import cellconstructor as CC
import cellconstructor.Phonons as phonons
import sscha
import sscha.Ensemble as ensemble
import os
import numpy as np
import sys, subprocess
import ase, ase.io
import matplotlib.pyplot as plt
import scipy

import tdscha, tdscha.DynamicalLanczos as DL
import tdscha.Perturbations as perturbations
  
def custom_unpol(raman_tns, i):
    """
    TEST THE GENERATION OF THE UNPOLARIZED RAMAN PERTURBATIONS
    """
    if i == 0:
        raman_v  = np.einsum('abc, a, b -> c', raman_tns, np.array([1,0,0]), np.array([1,0,0]))
        raman_v += np.einsum('abc, a, b -> c', raman_tns, np.array([0,1,0]), np.array([0,1,0]))
        raman_v += np.einsum('abc, a, b -> c', raman_tns, np.array([0,0,1]), np.array([0,0,1]))
        
    elif i == 1:
        raman_v  = np.einsum('abc, a, b -> c', raman_tns, np.array([1,0,0]), np.array([1,0,0]))
        raman_v -= np.einsum('abc, a, b -> c', raman_tns, np.array([0,1,0]), np.array([0,1,0]))
        
    elif i == 2:
        raman_v  = np.einsum('abc, a, b -> c', raman_tns, np.array([1,0,0]), np.array([1,0,0]))
        raman_v -= np.einsum('abc, a, b -> c', raman_tns, np.array([0,0,1]), np.array([0,0,1]))
        
    elif i == 3:
        raman_v  = np.einsum('abc, a, b -> c', raman_tns, np.array([0,1,0]), np.array([0,1,0]))
        raman_v -= np.einsum('abc, a, b -> c', raman_tns, np.array([0,0,1]), np.array([0,0,1]))
        
    elif i == 4:
        raman_v  = np.einsum('abc, a, b -> c', raman_tns, np.array([1,0,0]), np.array([0,1,0]))
        
    elif i == 5:
        raman_v  = np.einsum('abc, a, b -> c', raman_tns, np.array([1,0,0]), np.array([0,0,1]))
        
    elif i == 6:
        raman_v  = np.einsum('abc, a, b -> c', raman_tns, np.array([0,1,0]), np.array([0,0,1]))
        
    return raman_v

if __name__ == "__main__":
    """
    TEST OF RAMAN UNPOLARIZED COMPONENTS
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    
    dir_dyn = './dyn_pop1_'
    dyn = CC.Phonons.Phonons(dir_dyn, 3)
    
    raman_tns = "./raman_tensor_eq.npy"
    
    dyn.raman_tensor = np.load(raman_tns)
    
    # print(dyn.raman_tensor.shape)
    ens = ensemble.Ensemble(dyn, 0.)
    ens.generate(2)
    
    
    
    for i in range(7):
        # Prepare the Lanczos
        lanczos = DL.Lanczos(ens, use_wigner = True)
        lanczos.prepare_unpolarized_raman(index = i, debug = True)
        raman_v = np.load('raman_v_{}.npy'.format(i))
        perc_diff = np.abs((raman_v - custom_unpol(dyn.raman_tensor, i)) * 100/raman_v)
        # print(i, perc_diff.max())
        # print(i, perc_diff.min())
        # print(np.abs(raman_v - custom_unpol(dyn.raman_tensor, i)))
        # print()
        
        if perc_diff.max() > 1e-10:
            raise ValueError('The unpolarized Raman component {} is not implemented correctly'.format(i))
            
        #print(lanczos.get_prefactors_unpolarized_raman(i))
        