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


if __name__ == '__main__':
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    lanc = DL.Lanczos()
    lanc_julia = DL.Lanczos()
    
    save_dir = 'lanc_wigner'
    lanc.load_status(os.path.join(save_dir,'lanczos_wigner_STEP10'))
    lanc_julia.load_status(os.path.join(save_dir,'lanczos_wigner_julia_STEP10'))
    
    gf = lanc.get_green_function_continued_fraction(np.array([0]))
    w2 = 1 / np.real(gf)
    w = np.sign(w2[0]) * np.sqrt(np.abs(w2[0]))
    w *= CC.Units.RY_TO_CM
    
    gf_julia = lanc.get_green_function_continued_fraction(np.array([0]))
    w2_julia = 1 / np.real(gf_julia)
    w_julia = np.sign(w2_julia[0]) * np.sqrt(np.abs(w2_julia[0]))
    w_julia *= CC.Units.RY_TO_CM
    
    print()
    print('Compare the frequencies')
    print('JULIA freq in cm-1', w_julia)
    print('C++   freq in cm-1', w)
    print('diff in cm-1', w_julia - w)
    assert np.abs(w_julia - w) < 1e-10 
    print()
    print('a coeff diff')
    delta_a = np.abs(lanc.a_coeffs - lanc_julia.a_coeffs)
    print('MIN', delta_a.min())
    print('MAX', delta_a.max())
    print('b coeff diff')
    delta_b = np.abs(lanc.b_coeffs - lanc_julia.b_coeffs)
    print('MIN', delta_b.min())
    print('MAX', delta_b.max())
    print('c coeff diff')
    delta_c = np.abs(lanc.c_coeffs - lanc_julia.c_coeffs)
    print('MIN', delta_c.min())
    print('MAX', delta_c.max())
    print('basis P')
    delta_P = np.abs(lanc.basis_P - lanc_julia.basis_P)
    # print(delta_P)
    print('MIN', delta_P.min())
    print('MAX', delta_P.max())
    print('basis Q')
    delta_Q = np.abs(lanc.basis_Q - lanc_julia.basis_Q)
    # print(delta_P)
    print('MIN', delta_Q.min())
    print('MAX', delta_Q.max())
    
    # print((delta_P*100/np.abs(lanc.basis_P)).max())
    