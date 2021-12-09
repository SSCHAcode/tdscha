from __future__ import print_function

import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble

import tdscha, tdscha.DynamicalLanczos as DL
import time, numpy as np
import matplotlib.pyplot as plt
import nonlinear_sscha.Conversion as conv

# The dynamical matrix used to generate the ensemble
ORIGINAL_DYN = "H2_data/dyn_0"
NQIRR = 1  # Number of irreducible q points of the dynamical matrix

# The final dynamical matrix (after the SSCHA minimization)
FINAL_DYN = "H2_data/final_dyn"

# Temperature (in K) used to generate the ensemble
TEMPERATURE = 300
# Temperature for the Lanczos calculation (If None the same as TEMPERATURE)
FINAL_TEMPERATURE = None

# The ensemble data
ENSEMBLE_DIR = "ensemble_H2" # Directory of the ensemble
N_CONFIGS = 1000 # Number of configurations
POPULATION_ID = 1 # Population of the ensemble
LOAD_BIN = True # If true, load a binary ensemble


# Here the input of the TDSCHA calculation
LANCZOS_STEPS   = 100 # Number of Lanczos step
USE_THIRD_ORDER = True  # Use the third order in the calculation
USE_FOURTH_ORDER = True # Use the fourth order (2x computational cost)
USE_WIGNER = False
SAVE_EACH = 1 # Save the result each tot steps (for restart)

SAVE_PREFIX = "tdscha_lanczos" # The name of this calculation
    
MODE_PERTURBATION_ID = 0

if FINAL_TEMPERATURE is None:
    FINAL_TEMPERATURE = TEMPERATURE
    
    
# Load the dynamical matrix
dyn       = CC.Phonons.Phonons(ORIGINAL_DYN, NQIRR)
final_dyn = CC.Phonons.Phonons(FINAL_DYN, NQIRR)

# Load the ensemble
ens = sscha.Ensemble.Ensemble(dyn, TEMPERATURE)
if LOAD_BIN:
    ens.load_bin(ENSEMBLE_DIR, POPULATION_ID)
else:
    ens.load(ENSEMBLE_DIR, POPULATION_ID, N_CONFIGS)

ens.update_weights(final_dyn, FINAL_TEMPERATURE)

static_freqs = []
RE_green = []
IM_green = []

for USE_WIGNER in [False, True]:      
    for MODE_PERTURBATION_ID in range(3):
        print("\nPrepare the Lanczos...\n")
        print("Should I use Wigner? {}\n".format(USE_WIGNER))
        print("Wich mode are we looking at? {}\n".format(MODE_PERTURBATION_ID))
        
        # Save Folder
        if not USE_WIGNER:
            # The folder in which the data are saved
            SAVE_FOLDER = "Lanczos_H2_{}".format(MODE_PERTURBATION_ID) 
        else:
            # The folder in which the data are saved
            SAVE_FOLDER = "Lanczos_H2_wigner_{}".format(MODE_PERTURBATION_ID) 

        lanczos = DL.Lanczos(ens)
        lanczos.ignore_v3 = not USE_THIRD_ORDER
        lanczos.ignore_v4 = not USE_FOURTH_ORDER
        lanczos.use_wigner = USE_WIGNER
        lanczos.init()

        # Here the single mode code
        lanczos.prepare_mode(MODE_PERTURBATION_ID)
        
        # Run the calculation
        print("Running...")
        lanczos.run_FT(LANCZOS_STEPS, save_dir = SAVE_FOLDER,
                       prefix    = SAVE_PREFIX,
                       save_each = SAVE_EACH)

        # Lanczos status file
        steps = len(lanczos.a_coeffs) - 1
        if not USE_WIGNER:
            DATA_FILE = "Lanczos_H2_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, steps)
        else:
            DATA_FILE = "Lanczos_H2_wigner_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, steps)
        
        # If you submitted with the tdscha-lanczos.x
        # Copy all the files inside the directory
        LOAD_FROM_EXECUTABLE = False
        DIRECTORY = "submit_on_cluster"
        PREFIX = "tdscha_lanczos"

        FREQ_START = 0 # cm-1
        FREQ_END = 6000 # cm-1
        N_FREQS = 10000

        # Load the lanczos
        lanczos_2 = DL.Lanczos()

        if not LOAD_FROM_EXECUTABLE:
            lanczos_2.load_status(DATA_FILE)
        else:
            lanczos_2.load_from_input_files(PREFIX, DIRECTORY)

        w_static  = np.array([0])
        gf_static = lanczos_2.get_green_function_continued_fraction(w_static)
        hessian = 1 / np.real(gf_static[0])
        static_frequency = np.sign(hessian) * np.sqrt(np.abs(hessian))
        static_frequency *= CC.Units.RY_TO_CM
        print("The static frequency of the simualated mode is: {} cm-1".format(static_frequency))
        static_freqs.append(static_frequency)

        # Get the frequency array for plotting
        w = np.linspace(FREQ_START, FREQ_END, N_FREQS)

        # Convert in Ry
        w_ry = w / CC.Units.RY_TO_CM

        # Get the dynamical green function
        green_function = lanczos_2.get_green_function_continued_fraction(w_ry)

        # Plot the imaginary part
        plt.figure(dpi = 150)
        plt.xlabel("Frequency [cm-1]")
        plt.ylabel("- Im(G)/Re(G)")
        plt.plot(w, -np.imag(green_function), label = '-Im(G)')
        plt.plot(w, np.real(green_function), label = 'Re(G)')
        plt.legend(loc = 'best')
        plt.tight_layout()
        
        RE_green.append(+np.real(green_function))
        IM_green.append(-np.imag(green_function))    
        
        
# Save the static frequencies       
np.savetxt('freq_stat.txt', np.asarray(static_freqs).reshape((2,3)))
np.save('real_G', np.asarray(RE_green).reshape(2,3,N_FREQS))
np.save('imag_G', np.asarray(IM_green).reshape(2,3,N_FREQS))