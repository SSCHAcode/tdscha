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
import tdscha, tdscha.DynamicalLanczos as DL
import time, numpy as np
import matplotlib.pyplot as plt
import json
import brokenaxes
from brokenaxes import brokenaxes


def get_green_function(directory, omega, delta, use_wigner = False, index = 0):
    """
    Computes the dynamical green function obtained from a Lanczos calculation
    """
    # Lanczos status file
    steps     = np.load("{}/steps.npy".format(directory))
    DATA_FILE = "{}/lanczos_STEP{}.npz".format(directory, steps)
    print()
    print('DIRECTORY = {}'.format(directory))
    print('DATA FILE = {}'.format(DATA_FILE))
    print('STEPS DONE = {}'.format(steps))
    
    # Load the lanczos
    lanczos = DL.Lanczos()
    # Load the satus
    lanczos.load_status(DATA_FILE)
    # Add the prefactor for the unpolarized Raman
    pref = lanczos.get_prefactors_unpolarized_raman(index)

    # Convert into RY before getting the Green Function
    green_function = lanczos.get_green_function_continued_fraction(omega / CC.Units.RY_TO_CM,
                                                                   use_terminator = False,
                                                                   smearing = delta / CC.Units.RY_TO_CM)

    return -np.imag(pref * green_function)


if __name__ == "__main__":
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # The freq range in cm-1
    OMEGA = np.linspace(0, 4300, 1000)
    # The smearing function
    SMEARING = 30
    # The nuber of polarization
    Npol = 7
    
    # TDCHA DIR 
    my_dir = './tdscha_raman_POL'
    
    # Set to zero the green function
    GF = np.zeros(len(OMEGA))
    for pol in range(Npol):
        print('\n->POLARIZATION {}'.format(pol))
        dir_data = my_dir.replace('POL', '{}'.format(pol))
        # Get the Raman signal (or Green Function)
        GF += get_green_function(dir_data, OMEGA, SMEARING, use_wigner = True, index = pol)

    plt.plot(OMEGA, GF)
    plt.savefig('Raman_1ph_2ph_tdscha.png', dpi = 500)
    plt.show()