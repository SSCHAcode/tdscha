import sys, os
import numpy as np

import cellconstructor as CC, cellconstructor.Phonons
from cellconstructor.Settings import ParallelPrint as print

import sscha, sscha.Ensemble, sscha.SchaMinimizer
import tdscha
import tdscha.QSpaceLanczos as QSL

# ========== INPUT PARAMETERS ==========

# Dynamical matrix to generate the ensemble
original_dyn = "../free_energy_hessian/data_cs_sn_i3/dyn_gen_pop10_"
# Number of irreducible q points
nqirr = 4
# Directory of the ensemble (saved in binary)
ensemble_dir = "../free_energy_hessian/data_cs_sn_i3"
# Population ID of the ensemble
population_id = 10
# Temperature of the ensemble (in K)
temperature = 450.0

# Account for fourth-order scattering?
include_v4 = True

# Number of lanczos step (the higher, the more smooth the IR spectrum, but also the more expensive)
n_lanczos_steps = 100

# IR specific parameters
# Effective charges (espresso ph.x output with epsil = .true.)
effective_charges_file = "../free_energy_hessian/data_cs_sn_i3/effective_charges.pho"
# Electric field polarization
polarization_direction = np.array([1, 0, 0])  # Example: polarization along x-axis
# Electric field momentum versor (used for LO-TO splitting)
q_vector_direction = np.array([0, 0, 1])  # Example: q vector along z-axis

# where to save the results
ir_results_file = "ir_polx_qz.npz"

# ========== END OF INPUT PARAMETERS ==========

def run_hessian_calculation():
    """
    Run the calculation of the free energy Hessian.
    """
    print("Loading the original dynamical matrix and the ensemble...")
    dyn = CC.Phonons.Phonons(original_dyn, nqirr) 
    ensemble = sscha.Ensemble.Ensemble(dyn, temperature)
    ensemble.load_bin(ensemble_dir, population_id)

    # If no final dyn, run the SSCHA minimization first
    if final_dyn is None:
        print("No final dynamical matrix provided. Running SSCHA minimization first...")
        minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
        minim.run()
        minim.dyn.save_qe("final_auxiliary_dyn")
        ensemble = minim.ensemble
    else:
        print("Final dynamical matrix provided. Loading it and updating the ensemble weights...")
        # Load the final dynamical matrix and update the ensemble weights
        final_dynmat = CC.Phonons.Phonons(final_dyn, nqirr)
        ensemble.update_weights(final_dynmat, temperature)

    # Load the effective charges
    ensemble.current_dyn.ReadInfoFromESPRESSO(effective_charges_file)

    # Run the Free energy Hessian calculation
    print("Running the IR calculation...")
    qspace_lanczos = QSL.QSpaceLanczos(ensemble, use_wigner=True, lo_to_split=q_vector_direction)
    qspace_lanczos.ignore_v4 = not include_v4
    qspace_lanczos.init()
    qspace_lanczos.prepare_ir(pol_vec = polarization_direction)

    # Run
    qspace_lanczos.run_FT(n_lanczos_steps)
    qspace_lanczos.save_status(ir_results_file)


if __name__ == "__main__":
    run_hessian_calculation()

