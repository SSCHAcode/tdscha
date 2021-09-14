from __future__ import print_function

import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble

import tdscha, tdscha.DynamicalLanczos as DL
import time, numpy as np


# INPUT VARIABLES -------------------


# The dynamical matrix used to generate the ensemble
ORIGINAL_DYN = "../IceEnsembleTest/dyn"
NQIRR = 1  # Number of irreducible q points of the dynamical matrix


# Here the input of the TDSCHA calculation
RESTART_FROM_FILE = "data/tdscha_lanczos_STEP5.npz" # The name of the Lanczos file to restart.
LANCZOS_STEPS = 100 # Number of new Lanczos step to do
SAVE_EACH = 5 # Save the result each tot steps (for restart)
SAVE_FOLDER = "data" # The folder in which the data are saved
SAVE_PREFIX = "tdscha_lanczos_restarted" # The name of this calculation


INFO = """

This simple example executes a TDSCHA perturbation therory calculation.

This script is a template.
You can easily edit the variables defining at the beginning of the script.

It restarts from a previous stopped calculation.
You can restart from the files saved automatically during the local execution.

This scripts runs the Lanczos locally. You can use multiprocessors with:

mpirun -np NPROC python run_local.py > output

Substituting NPROC with the number of processors.

NOTE. This scripts by default simulates a single mode specified by input.
If you want Raman or IR, uncomment the relative part of the code below.


-------------------------------------------
Input data: Check if it is what you expect

Restarting from {}
The number of steps in the Lanczos simulation is {}
The status is saved each {} steps
The directory to save the status is {}
The prefix for the status name is {}

""".format(RESTART_FROM_FILE,
           LANCZOS_STEPS, 
           SAVE_EACH, SAVE_FOLDER, SAVE_PREFIX)
           
print(INFO)

# ============== HERE THE SCRIPT ====================

dyn = CC.Phonons.Phonons(ORIGINAL_DYN, NQIRR)
# Create a fake ensemble, we need it only to initialize the symmetries
ens = sscha.Ensemble.Ensemble(dyn, 0) 
ens.generate(2)


lanczos = DL.Lanczos(ens)
lanczos.init() # Initialize the symmetries
lanczos.load_status(RESTART_FROM_FILE)

# Run the calculation
print("Running...")
lanczos.run_FT(LANCZOS_STEPS, save_dir = SAVE_FOLDER,
               prefix = SAVE_PREFIX,
               save_each = SAVE_EACH)
