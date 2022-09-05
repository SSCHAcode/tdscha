#!python
from __future__ import print_function
from __future__ import division

import sys, os

import numpy as np

import sscha,tdscha, tdscha.DynamicalLanczos
import sscha.Ensemble


smearing = float(sys.argv[1])


#Define each polarization term and its relative multilplication factor to get the unpolarized Raman response
name_list=["xxxx","yyyy","zzzz","xxyy","xxzz","yyzz","xyxy","xzxz","yzyz"]
factor = np.array([8,8,8,2,2,2,14,14,14],dtype=float)

NQIRR = 1

dyn = CC.Phonons.Phonons("dyn",NQIRR)

T = 300
W_START = 0
W_END = 5000 / CC.Units.RY_TO_CM 
NW = 5000
SMEARING = smearing / CC.Units.RY_TO_CM


final_gf = np.zeros(NW, dtype=np.complex128)



for j,jq in enumerate(name_list):

    
    name_root = "tdscha_lanczos_"+str(jq)
    name_dir = "cluster_"+str(jq)
    
    data = tdscha.DynamicalLanczos.Lanczos()
    data.load_from_input_files(name_root,directory=name_dir)
    
    nat = data.nat
    N_iters = len(data.a_coeffs) - 1

    
    dynamical_noterm = np.zeros((N_iters, NW),dtype=np.complex128) 
    w_array = np.linspace(W_START, W_END, NW)


    a_coeffs = np.copy(data.a_coeffs)
    b_coeffs = np.copy(data.b_coeffs)
    c_coeffs = np.copy(data.c_coeffs)

    for i in range(N_iters):
        data.a_coeffs = a_coeffs[:i+1]
        data.b_coeffs = b_coeffs[:i]
        data.c_coeffs = c_coeffs[:i]

        gf = data.get_green_function_continued_fraction(w_array, use_terminator = False, smearing = SMEARING)
        dynamical_noterm[i, :] +=  gf * factor[j]

        if i % 10 == 0:
            sys.stderr.write("\rProgress %% %d" % (i * 100 / N_iters))
            sys.stderr.flush()
            sys.stderr.write("\n")
    if data.perturbation_modulus == 1:
        raise ValueError("Error perturbation modulus is not loaded correctly")
    else:
        final_gf[:] += dynamical_noterm[-1,:] 

print ("Saving the results...")

savename = "Green_function_sm"+str(int(smearing))
np.savetxt(savename,final_gf)
