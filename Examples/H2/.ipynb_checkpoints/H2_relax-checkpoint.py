import numpy as np
import matplotlib.pyplot as plt
import os

import time
import cellconstructor as CC
import cellconstructor.Methods as methods
import cellconstructor.Units as units

import sscha, sscha.Ensemble, sscha.SchaMinimizer
import sscha.Relax, sscha.Utilities

import nonlinear_sscha
import nonlinear_sscha.NonLinearStructure as NLS
import nonlinear_sscha.NonLinearEnsemble as NLE
import nonlinear_sscha.Conversion as conv

import ase
from ase.visualize import view

import H2model 
import H2model.Calculator

# The force field
ff_calculator = H2model.Calculator.ToyModelCalculator()
ff_calculator.E = 0.001

Nconfigs = 5000

k_harm =  (2. * ff_calculator.H2_a**2 * ff_calculator.H2_D) * conv.HA_TO_RY
re     = ff_calculator.H2_re * conv.AU_TO_ANGSTROM

struct = CC.Structure.Structure(2)
struct.masses = {"H" : 918.58996499058958}
struct.coords = np.array([[-re/2., 0., 0.],
                          [+re/2., 0., 0.]])
struct.unit_cell = 5. * np.eye(3)
struct.has_unit_cell = True

Cart_dyn = CC.Phonons.Phonons(struct)

diagonal = 0.1 * k_harm * np.eye(3)

Cart_dyn.dynmats[0][0:3,0:3] = np.copy(diagonal)
Cart_dyn.dynmats[0][3:6,3:6] = np.copy(diagonal)
CC.symmetries.CustomASR(Cart_dyn.dynmats[0])

# Define the ensemble
ensemble = sscha.Ensemble.Ensemble(Cart_dyn, T0 = 300., supercell = Cart_dyn.GetSupercell())

# Setup the minimizer
minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
minim.root_representation = "normal" 
minim.precond_dyn    = True
minim.min_step_dyn   = 0.1
minim.kong_liu_ratio = 0.5

# Setup the relaxer
relax = sscha.Relax.SSCHA(minim, ase_calculator = ff_calculator, N_configs = Nconfigs, max_pop = 10)

# Run the minimization
relax.relax()

# Create the directory
path = 'H2_data'
os.mkdir(path)

# Lets plot the Free energy, gradient and the Kong-Liu effective sample size
relax.minim.plot_results(save_filename = '{}/LinearSCHA_minimization_E={:.0e}'.format(path, ff_calculator.E))
# Save the final dyn
relax.minim.dyn.save_qe("{}/final_dyn".format(path))
# Save the initial dyn
relax.minim.ensemble.dyn_0.save_qe("{}/dyn_0".format(path))

# Save the ensemble in binary
relax.minim.ensemble.save_bin('ensemble_H2')