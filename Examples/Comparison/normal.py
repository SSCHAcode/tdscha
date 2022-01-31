from __future__ import print_function

import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble

import tdscha, tdscha.DynamicalLanczos as DL
import time, numpy as np


# INPUT VARIABLES -------------------

# The dynamical matrix used to generate the ensemble
ORIGINAL_DYN = "../IceEnsembleTest/dyn"
NQIRR = 1  # Number of irreducible q points of the dynamical matrix

# The final dynamical matrix (after the SSCHA minimization)
FINAL_DYN = "../IceEnsembleTest/dyn_population2_"

# Temperature (in K) used to generate the ensemble
TEMPERATURE = 0. #100
# Temperature for the Lanczos calculation (If None the same as TEMPERATURE)
FINAL_TEMPERATURE = None

MODE_PERTURBATION_ID = 2
# 0 => The lowest energy mode of the final sscha matrix (excluding translations)

USE_WIGNER = False

# The ensemble data
ENSEMBLE_DIR = "../IceEnsembleTest" # Directory of the ensemble
N_CONFIGS = 10000 # Number of configurations
POPULATION_ID = 2 # Population of the ensemble
LOAD_BIN = False # If true, load a binary ensemble

# Here the input of the TDSCHA calculation
LANCZOS_STEPS = 500 # Number of Lanczos step
USE_THIRD_ORDER = True  # Use the third order in the calculation
USE_FOURTH_ORDER = True # Use the fourth order (2x computational cost)
SAVE_EACH = 1 # Save the result each tot steps (for restart)

if USE_FOURTH_ORDER and USE_FOURTH_ORDER:
    SAVE_FOLDER = "data_md_{}".format(MODE_PERTURBATION_ID) # The folder in which the data are saved
elif not USE_FOURTH_ORDER:
    SAVE_FOLDER = "data_no_D4_md_{}".format(MODE_PERTURBATION_ID) # The folder in which the data are saved
else:
    SAVE_FOLDER = "data_harm_md_{}".format(MODE_PERTURBATION_ID) # The folder in which the data are saved
    
SAVE_PREFIX = "tdscha_lanczos" # The name of this calculation

# You need to choose the response to which perturbation to compute.
# By default, we compute only the perturabtion along one mode.
# Mode ids are ordered as the respective frequencies in the supercell
# (excluding translations)


# NOTE: If you want to compute IR or Raman,
#       go to the code below and uncomment the respective region.


if FINAL_TEMPERATURE is None:
    FINAL_TEMPERATURE = TEMPERATURE

INFO = """

This simple example executes a TDSCHA perturbation therory calculation.

This script is a template.
You can easily edit the variables defining at the beginning of the script.

This scripts runs the Lanczos locally. You can use multiprocessors with:

mpirun -np NPROC python run_local.py > output

Substituting NPROC with the number of processors.

NOTE. This scripts by default simulates a single mode specified by input.
If you want Raman or IR, uncomment the relative part of the code below.


-------------------------------------------
Input data: Check if it is what you expect

The dynamical matrix that generated the ensemble is = {}
The number of irreducible q points are = {}
Temperature of the ensemble generated = {}
The dynamical matrix at the end of the SSCHA simulation is = {}

The ensemble location is {}
The number of configurations are {}
The population id of the ensemble is {}

The number of steps in the Lanczos simulation is {}
Does the calculation include third order? {}
Does the calculation include fourth order? {}
The status is saved each {} steps
The directory to save the status is {}
The prefix for the status name is {}

""".format(ORIGINAL_DYN, NQIRR, TEMPERATURE, FINAL_DYN,
           ENSEMBLE_DIR, N_CONFIGS, POPULATION_ID,
           LANCZOS_STEPS, USE_THIRD_ORDER, USE_FOURTH_ORDER,
           SAVE_EACH, SAVE_FOLDER, SAVE_PREFIX)
           
print(INFO)

# ============== HERE THE SCRIPT ====================


# Load the dynamical matrix
dyn = CC.Phonons.Phonons(ORIGINAL_DYN, NQIRR)
final_dyn = CC.Phonons.Phonons(FINAL_DYN, NQIRR)

# Load the ensemble
print("Loading the ensemble...")
t1 = time.time()
ens = sscha.Ensemble.Ensemble(dyn, TEMPERATURE)
if LOAD_BIN:
    ens.load_bin(ENSEMBLE_DIR, POPULATION_ID)
else:
    ens.load(ENSEMBLE_DIR, POPULATION_ID, N_CONFIGS)
t2 = time.time()
print("Time to load the ensemble: {} s".format(t2-t1))


print("Updating the ensemble to the final dynamical matrix...")
ens.update_weights(final_dyn, FINAL_TEMPERATURE)

print("Prepare the Lanczos...")
lanczos = DL.Lanczos(ens)
lanczos.ignore_v3 = not USE_THIRD_ORDER
lanczos.ignore_v4 = not USE_FOURTH_ORDER
lanczos.use_wigner = USE_WIGNER
lanczos.init()

# # Here we prepare the perturbation.
# # We have 3 possible kinds of perturbations:
# #   1) A single mode (default)
# #   2) Raman
# #   3) IR
# #
# # If you want Raman or IR, uncomment the relative sections
# # and comment the one of the single mode.
# # NOTE: Only one perturbation at time is calculated.
# print("Preparing the perturbation...")
# print("Selected mode ID: {} | w = {} cm-1. Check if this is what you expect.".format(MODE_PERTURBATION_ID, lanczos.w[MODE_PERTURBATION_ID] * CC.Units.RY_TO_CM))

# # Here the single mode code (comment if you want IR or RAMAN)
lanczos.prepare_mode(MODE_PERTURBATION_ID)

# # Here the IR perturbation (uncomment)
# # NOTE: you need the effective charges inside the final dynamical matrix.
# # # Select the polarization of the light (cartesian coordinates)
# # light_polarization = np.array([1,0,0])
# # lanczos.prepare_ir(light_polarization)

# # Here the Raman perturbation (uncomment)
# # NOTE: you need the Raman Tensor inside the final dynamical matrix.
# ## Select the incoming and outcoming light polarization
# # light_polarization_in = np.array([1,0,0])
# # light_polarization_out = np.array([1,0,0])
# # lanczos.prepare_raman(light_polarization_in, light_polarization_out)

t1 = time.time()

# Run the calculation
print("Running...")
lanczos.run_FT(LANCZOS_STEPS, save_dir = SAVE_FOLDER,
               prefix = SAVE_PREFIX,
               save_each = SAVE_EACH)

t2 = time.time()
print('Total time required in sec {}'.format(t2 - t1))




