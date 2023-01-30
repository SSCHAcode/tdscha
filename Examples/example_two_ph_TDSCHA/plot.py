import numpy as np
import matplotlib.pyplot as plt

import cellconstructor as CC
import cellconstructor.Phonons

import sscha
import tdscha, tdscha.DynamicalLanczos as DL
import sscha.Ensemble

import time

import scipy, scipy.sparse

from sscha.Parallel import pprint as print

import cellconstructor.Units as units

import sys, os

import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble
import json
import tdscha, tdscha.DynamicalLanczos as DL
import time, numpy as np

import matplotlib.pyplot as plt

def get_green_function_cluster(directory, use_wigner = False, classical = False, classical_factor = 1, delta = 10, my_steps = None):
    """
    Computes the dynamical green function WITHOUT TERMINATOR from the cluster
    
    Pramaneters
    -----------
        -direcotry: directory where lanczos_STEPxxx.npz are saved
        -use_wigner: bool, if True the Wigner representaiton is used
        -classical: bool, if True we read the results from a classical calculation
        -classical_factor: float, the factor used to multiply the masses to get a classical calculation
        -delta: float, the smearing in cm-1
        -my_steps: int the number of steps to read
    """
    # Load the lanczos
    lanczos = DL.Lanczos()
    lanczos.a_coeffs = []
    lanczos.b_coeffs = []
    lanczos.c_coeffs = []
    lanczos.perturbation_modulus = 0.

    with open('{}/tdscha.json'.format(directory), 'r') as f:
        data = json.load(f)
    print(data['data']['perturbation_modulus'])
    abc_coeff = np.loadtxt('{}/tdscha.abc'.format(directory))


    lanczos.perturbation_modulus = data['data']['perturbation_modulus']
    if my_steps is None:
        lanczos.a_coeffs = list(abc_coeff[:,0])
        lanczos.b_coeffs = list(abc_coeff[:,1])
        lanczos.c_coeffs = list(abc_coeff[:,2])
    else:
        print('MY STEPS', my_steps)
        lanczos.a_coeffs = list(abc_coeff[:my_steps,0])
        lanczos.b_coeffs = list(abc_coeff[:my_steps,1])
        lanczos.c_coeffs = list(abc_coeff[:my_steps,2])
        
    lanczos.use_wigner = True

    # Get the dynamical green function
    if not classical:
        green_function = lanczos.get_green_function_continued_fraction(w / CC.Units.RY_TO_CM, use_terminator = False, smearing = delta / CC.Units.RY_TO_CM)
    else:
        print('This is a Classic calculation')
        green_function = lanczos.get_green_function_continued_fraction(w /classical_factor /CC.Units.RY_TO_CM, use_terminator = False,\
                                                                       smearing = delta /classical_factor / CC.Units.RY_TO_CM)

    return -np.imag(green_function)


def get_green_function(directory, use_wigner = False, classical = False, classical_factor = 1, delta = 10, my_steps = 100):
    """
    Computes the dynamical green function WITHOUT TERMINATOR
    
    Pramaneters
    -----------
        -direcotry: directory where lanczos_STEPxxx.npz are saved
        -use_wigner: bool, if True the Wigner representaiton is used
        -classical: bool, if True we read the results from a classical calculation
        -classical_factor: float, the factor used to multiply the masses to get a classical calculation
        -delta: float, the smearing in cm-1
        -my_steps: int the number of steps to read
    """
    # Lanczos status file
    steps = my_steps
    DATA_FILE = "{}/lanczos_STEP{}.npz".format(directory, steps)
    print()
    print('DIRECTORY = {}'.format(directory))
    print('DATA FILE = {}'.format(DATA_FILE))
    print('STEPS DONE = {}'.format(steps))
    
    # Load the lanczos
    lanczos = DL.Lanczos()
    lanczos.load_status(DATA_FILE)
    lanczos.use_wigner = True

    # Get the dynamical green function
    if not classical:
        green_function = lanczos.get_green_function_continued_fraction(w / CC.Units.RY_TO_CM,\
                                                                       use_terminator = False, smearing = delta / CC.Units.RY_TO_CM)
    else:
        print('This is a Classic calculation')
        green_function = lanczos.get_green_function_continued_fraction(w /classical_factor /CC.Units.RY_TO_CM, use_terminator = False,\
                                                                       smearing = delta /classical_factor / CC.Units.RY_TO_CM)
    
    return -np.imag(green_function)

def plot_spectral(imag_chi = None, imag_chi_extra = None, labels = None, extra_labels = None):
    """
    Plot the -Im(\chi)
    
    Parameters:
    -----------
        -imag_chi: the signal to plot
        -imag_chi_extra: a list of signal to plot
        -lable_1: the label for imag_chi
        -extra_labels: a list of labels for imag_chi_extra
    """
    # Plot 
    plt.figure(dpi = 500)
    
    plt.title('Unpolarized spectra BaTiO$_3$ $3x3x3$ cubic T=400 K $\\delta$={} cm-1'.format(delta_in))
    plt.xlabel("$\\omega$ [cm-1]")
    
    if labels == '1':
        labels = '$\\left\\langle \\Xi \\right\\rangle$'
        
    if labels == '2':
        labels = '$\\left\\langle \\Xi \\right\\rangle$ + $\\left\\langle \\frac{\\partial \\Xi}{\partial R} \\right\\rangle$'
        
    if labels == 'eq' :
        labels = '$\\Xi(\\mathcal{R})$'

    if imag_chi is not None:
        plt.plot(w, imag_chi, linestyle = 'solid', label = '{}'.format(labels))
        
    if imag_chi_extra is not None:
        if len(imag_chi_extra) != len(extra_labels):
            raise ValueError('The data to plot must match the labels for the legend')
        
        for i, _gf_ in enumerate(imag_chi_extra):
            plt.plot(w, _gf_ , linewidth=0.8, label = '{}'.format(extra_labels[i]))
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if name_fig is not None:
        print('Saving fig')
        os.chdir(total_path)
        plt.savefig('{}.eps'.format(name_fig))
        
    return
 
def get_unpolarized_signal(GF_in):
    """
    GET THE UNPOLARIZED SIGNAL
    ==========================
    
    The unpolarized signal is defined as
    \chi(\omega)_{\text{unpolarized}} =  
    +8 \left(\chi(\omega)_{\alpha_{xx}\alpha_{xx}}  +  \chi(\omega)_{\alpha_{yy}\alpha_{yy}}  +  \chi(\omega)_{\alpha_{zz}\alpha_{zz}}\right)
    +14\left(\chi(\omega)_{\alpha_{xy}\alpha_{xy}}  +  \chi(\omega)_{\alpha_{xz}\alpha_{xz}}  +  \chi(\omega)_{\alpha_{zy}\alpha_{zy}}\right)
    +2 \left(\chi(\omega)_{\alpha_{xx}\alpha_{yy}}  +  \chi(\omega)_{\alpha_{xx}\alpha_{zz}}  +  \chi(\omega)_{\alpha_{yy}\alpha_{zz}}\right)
    
    Parameters
    ----------
        -GF_in: np.array of shape (9,N_FREQS)
    """
    GF = np.copy(GF_in)
    
    GF[6,:] -= (GF[0,:] + GF[1,:])
    GF[7,:] -= (GF[0,:] + GF[2,:])
    GF[8,:] -= (GF[1,:] + GF[2,:])
    GF[6:,:] /= 2
    
    GF_UNPOL = np.zeros(len(GF[0,:]))
    GF_UNPOL = 8 * (GF[0,:] + GF[1,:] + GF[2,:]) + 14 * (GF[3,:] + GF[4,:] + GF[5,:]) + 2 * (GF[6,:] + GF[7,:] + GF[8,:])
    
    return GF_UNPOL
    
if __name__ == "__main__":
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # Directory eith the Lanczos calculations
    dir_lanczos_scha = []
    pols = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xy2', 'xz2', 'yz2']
    for p in pols:
        dir_lanczos_scha.append('data/raman_2_pol_{}_Nraman_50_symm_True_steps_100'.format(p))
        
        
    # Directory eith the Lanczos calculations
    dir_lanczos_1_scha = []
    pols = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xy2', 'xz2', 'yz2']
    for p in pols:
        dir_lanczos_1_scha.append('data/raman_1_pol_{}_Nraman_50_symm_True_steps_100'.format(p))

    try:
        name_fig = sys.argv[1]
    except:
        name_fig = None
             
    # The smearing in cm-1
    delta_in = 30
    
    FREQ_START = 0
    FREQ_END = 1000
    
    # Range definition in cm-1
    N_FREQS = 10000
    
    # Get the frequency array for plotting in cm-1
    w = np.linspace(FREQ_START, FREQ_END, N_FREQS)
    
    # Prepare the Green fucntion
    gf_scha = np.zeros((len(pols),N_FREQS))
    # Prepare the Green fucntion
    gf_scha_1 = np.zeros((len(pols),N_FREQS))
    
    # Get all the polarizations
    for i in range(len(pols)):
        gf_scha[i,:]   = get_green_function(dir_lanczos_scha[i],   delta = delta_in)
        try:
            gf_scha_1[i,:] = get_green_function(dir_lanczos_1_scha[i], delta = delta_in)
        except:
            gf_scha_1[i,:] = get_green_function(dir_lanczos_1_scha[i], delta = delta_in, my_steps = 91)
        
        

    def bose_pref(Temp = 400, calc = 'stokes'):
        """
        ADD THE BOSE FACTOR IF WE WANT TO COMPARE WITH EXPERIMENTS
        ==========================================================
        
        The unpolarized Raman signal should be multiplied by
            1 + n(omega) for the STOKES (neg freq)
            n(omega)     for the ANTI-STOKES (pos freq)
        if you want to compare TDSCHA with experiments
        
        Paramenters:
        ------------
            -calc: string, stokes or antistokes
        
        """
        if Temp > 0.5:
            T_ry = (Temp/CC.Units.RY_TO_KELVIN)
            bose_pref =  1/(np.exp(w/(T_ry * CC.Units.RY_TO_CM)) - 1)
            if calc == 'stokes':
                print('Stokes')
                bose_pref += 1
            else:
                print('Anti-Stokes')
        else:
            bose_pref = 0.
            
        return bose_pref
    
    # This is the unpolarized Raman signal
    gf_unpol_scha    = get_unpolarized_signal(gf_scha[:,:]) # * bose_pref(Temp=400)
    
    # This is the unpolarized Raman signal
    gf_unpol_scha_1  = get_unpolarized_signal(gf_scha_1[:,:]) # * bose_pref(Temp=400)
    
    # Plot the unpolarized signal
    plot_spectral(gf_unpol_scha, [gf_unpol_scha_1],\
                  labels = '$\\left\\langle \\Xi \\right\\rangle + \\left\\langle \\frac{\\partial \\Xi}{\partial R} \\right\\rangle$',\
                  extra_labels = ['$\\left\\langle \\Xi \\right\\rangle$'])
    
    # By plotting the two phonon dos you can check if the raman vertex modulates the signal
    
    # dyn = CC.Phonons.Phonons('final_dyn_T400_', 4)
#     two_ph_dos = dyn.get_two_phonon_dos(w/CC.Units.RY_TO_CM, delta_in/CC.Units.RY_TO_CM, 400.)
    
#     # Plot the two phonon dos
#     plot_spectral(two_ph_dos, labels = '2-phonon dos')
    
