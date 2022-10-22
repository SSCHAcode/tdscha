from __future__ import print_function
from __future__ import division
from tdscha.Parallel import pprint as print
import tdscha 
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries
import sscha, sscha.Ensemble as Ensemble
from time import time 
import spglib
import numpy as np

##################
# SYMMETRIZATION #
##################

def symmetrize_d1alpha_dR(d1alpha_dR_in, dyn, verbose = False):
    """
    SYMMETRIZE THE EFFECTIVE CHARGES IN SUPERCELL
    =============================================

    This method symmetrize the Raman tensors
    
    Parameters:
    -----------
        -d1alpha_dR_in: average of the polarizability moment first derivative, np.array with shape = (E_field, E_field, 3 * N_at_sc)
        -dyn: the current Phonons object of the ensemble object

    Returnes:
    ---------
        -d1alpha_dR_out: symmetrized average of the polarizability first derivative, np.array with shape = (E_field, E_field, 3 * N_at_sc)
    """
    # Initialize the symmetries in the supercell
    super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    symmetries = cellconstructor.symmetries.QE_Symmetry(super_dyn.structure)
    symmetries.SetupFromSPGLIB()
    
    # Prepare the result
    d1alpha_dR_out = d1alpha_dR_in.copy()

    if verbose:
        print('Symmetrizing the Raman tensor in the supercell!')
        print('[spglib] identified {} symmetries'.format(symmetries.QE_nsym))
        print('[spglib] identified {} translations'.format(symmetries.QE_translation_nr))
        print()

    # The result has shape = (E_field, E_field, 3 * N_at_sc)
    d1alpha_dR_out = d1alpha_dR_in.copy()
    symmetries.ApplySymmetryToRamanTensor(d1alpha_dR_out)

    return d1alpha_dR_out





def symmetrize_d1M_dR(d1M_dR_in, dyn, verbose = False):
    """
    SYMMETRIZE THE EFFECTIVE CHARGES IN SUPERCELL
    =============================================

    This method symmetrize the effective charges generated in supercell needed for an IR spectra calculation
    considering postion modulation of effective charges (see prepare_anahrmonic_ir method)

    ..math:
        P_{\alpha} = \sum_{i=1}^{3N_at_sc} Z(alpha, i) u_i

        S_{\beta,\alpha} P_alpha = \sum_{ij=1}^{3N_at_sc} Z(alpha, j) S_sc_{j,i} u_i

        Z \longrightarrow S^\dagger Z S_sc
        
    The sum rule says:
    ..math:
        \sum_{k=1}^{N_at_sc} Z_{k,E,cart} = 0 for E = x,y,z and cart = x,y,z

    Parameters:
    -----------
        -d1M_dR_in: average of the dipole moment first derivative (i.e. the effective charges), np.array with shape = (3 * N_at_sc, E_comp)
        -dyn: the current Phonons object of the ensemble object

    Returnes:
    ---------
        -d1M_dR_out: symmetrized average of the dipole moment first derivative (i.e. the effective charges), np.array with shape = (3 * N_at_sc, E_comp)
    """
    # Initialize the symmetries in the supercell
    super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    symmetries = cellconstructor.symmetries.QE_Symmetry(super_dyn.structure)
    symmetries.SetupFromSPGLIB()
    
    # Prepare the result
    d1M_dR_out = d1M_dR_in.copy()

    if verbose:
        print('Symmetrizing the effective charges in the supercell!')
        print('[spglib] identified {} symmetries'.format(symmetries.QE_nsym))
        print('[spglib] identified {} translations'.format(symmetries.QE_translation_nr))
        print()

    # Reshape the result in np.array with shape = (N_at_sc, E_field, cart)
    temp = np.einsum('abc -> acb', d1M_dR_out.reshape((super_dyn.structure.N_atoms, 3, 3)))
    symmetries.ApplySymmetryToEffCharge(temp)
    
    # Reshape in np.array with shape = (3 * N_at_sc, E_field)
    d1M_dR_out = np.einsum('abc -> acb', temp).reshape((super_dyn.structure.N_atoms * 3, 3))

    return d1M_dR_out





def symmetrize_d2M_dR(d2M_dR_in, dyn, verbose = False):
    """
    SYMMETRIZE THE SECOND ORDER EFFECTIVE CHARGES
    =============================================

    Parameters:
    -----------
        -d2M_dR_in: average of the second derivative of the dipole moment (i.e. the second order dipole moment), np.array with shape (3 * N_at_sc, 3 * N_at_sc, 3)
             the last component is the E_field
        -dyn: the current Phonons object of the ensemble object

    Returns:
    --------
        -d2M_dR_out: symmetrized second order effective charges,  np.array with shape (3 * N_at_sc, 3 * N_at_sc, 3), the last component is the E_field
    """
    # Initialize the symmetries in the supercell
    super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    symmetries = cellconstructor.symmetries.QE_Symmetry(super_dyn.structure)
    symmetries.SetupFromSPGLIB()

    # Prepare the results
    d2M_dR_out = d2M_dR_in.copy()
            
    if verbose:
        print('Symmetrizing the second order EFF CHARGES in the supercell!')
        print('[spglib] identified {} symmetries'.format(symmetries.QE_nsym))
        print('[spglib] identified {} translations'.format(symmetries.QE_translation_nr))
        print()

    # ASR is imposed
    symmetries.ApplySymmetryToSecondOrderEffCharge(d2M_dR_out, apply_asr = True)

    # Apply permutation symmetry on the first two indices
    d2M_dR_out += np.einsum("abc -> bac", d2M_dR_out)
    d2M_dR_out /= 2

    return d2M_dR_out


def symmetrize_d2alpha_dR(d2alpha_dR_in, dyn, verbose = False):
    """
    SYMMETRIZE THE SECOND ORDER RAMAN TENSORS
    =========================================

    Parameters:
    -----------
        -d2alpha_dR_in: average of the second derivative of the polarizability, np.array with shape (3, 3, 3 * N_at_sc, 3 * N_at_sc)
             the last component is the E_field
        -dyn: the current Phonons object of the ensemble object

    Returns:
    --------
        -d2alpha_dR_out: symmetrized second order Raman tensor,  np.array with shape (3, 3, 3 * N_at_sc, 3 * N_at_sc)
    """
    # Initialize the symmetries in the supercell
    super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    symmetries = cellconstructor.symmetries.QE_Symmetry(super_dyn.structure)
    symmetries.SetupFromSPGLIB()

    # Prepare the results
    d2alpha_dR_out = d2alpha_dR_in.copy()
            
    if verbose:
        print('Symmetrizing the second order RAMAN tensor in the supercell!')
        print('[spglib] identified {} symmetries'.format(symmetries.QE_nsym))
        print('[spglib] identified {} translations'.format(symmetries.QE_translation_nr))
        print()

    # ASR is imposed
    symmetries.ApplySymmetryToSecondOrderRamanTensor(d2alpha_dR_out, apply_asr = True)

    return d2alpha_dR_out





    
#####################
# ONE PHONON VERTEX #
#####################

def get_d1M_dR_av(ensemble, effective_charges, symmetrize = False):
    """
    Get the average of the effective charges over the ensemble
    
    Parameters:
    -----------
        -ensemble: the scha ensemble object on which perorm the averages
        -effective_charges: the effective charges, np.array with shape (N_configs, N_at_sc, E_field, 3)
        
    Returns:
    --------
        -d1M_dR: the averages of effective charges, np.array with shape (3 * N_at_sc, E_filed)
    """
    assert len(effective_charges) == ensemble.N, """
Error, the number of effective charges ({})
       does not match with the ensemble size ({}).
""".format(len(effective_charges), ensemble.N)

    # Create the effective charge array
    dyn = ensemble.current_dyn
    # Get the number of configurations
    n_rand = ensemble.N
    # Nuber of atoms in the unit cell
    nat = dyn.structure.N_atoms
    # NUmber of atoms in the supercell
    nat_sc = nat * np.prod(dyn.GetSupercell())
    # Prepare the effective charges, np.array with shape =  (n_rand, 3 * N_at_sc, E_field)
    new_eff_charge = np.zeros((n_rand, 3 * nat_sc, 3), dtype = np.double, order = "F")

    for i in range(n_rand):
        new_eff_charge[i, :, :] = np.einsum("abc ->bac", effective_charges[i]).reshape((3, 3*nat_sc)).T
        
    # Effective sample size
    N_effective = np.sum(ensemble.rho)
    # Get the average of the effective charges, np.array with shape (3 * N_at_sc, E_field)
    d1M_dR = np.einsum("iab, i", new_eff_charge, ensemble.rho) / N_effective
    
    if symmetrize:
        print('Symmetrizing the one phonon IR in the supercell')
        d1M_dR = symmetrize_d1M_dR(d1M_dR, ensemble.current_dyn, verbose = True)

    return d1M_dR



def get_d1alpha_dR_av(ensemble, raman_tensors, symmetrize = False):
    """
    Get the average of the raman tensors over the ensemble
    
    Parameters:
    -----------
        -ensemble: the scha ensemble object on which perform the averages
        -raman_tensors: the raman_tensors, np.array with shape (N_configs, E_filed, E_field, 3 * N_at_sc)
        
    Returns:
    --------
        -d1alpha_dR: the averages of raman_tensor, np.array with shape (E_filed, E_field, 3 * N_at_sc)
    """
    # The size of the Raman tensor is controlled in the main

    # Effective sample size
    N_effective = np.sum(ensemble.rho)
    # Get the average of raman tensors, np.array with shape (E_field, E_field, 3 * N_atoms)
    d1alpha_dR = np.einsum("iabc, i", raman_tensors, ensemble.rho) / N_effective
    
    # print(symmetrize)
    if symmetrize:
        print('Symmetrizing the one phonon Raman in the supercell')
        d1alpha_dR = symmetrize_d1alpha_dR(d1alpha_dR, ensemble.current_dyn)

    return d1alpha_dR




#################
# TWO PH VERTEX #
#################


def get_d2M_dR_av(ensemble, effective_charges, w_pols = None, symmetrize = False):
    """
    COMPUTE THE SECOND DERIVATIVE OF THE DIPOLE MOMENT
    ==================================================

    Using the effective charges, we compute the derivative of the dipole moment
    
     Parameters:
    -----------
        -ensemble: the scha ensemble object on which perorm the averages
        -effective_charges: the effective charges, np.array with shape (N_configs, N_at_sc, E_field, cart_comp)
        
    Returns:
    --------
        -d2M_dR: the averages of the dipole moment second derivative, np.array with shape (3 * N_at_sc, 3 * N_at_sc, E_filed)
        
    """
    assert len(effective_charges) == ensemble.N, """
Error, the number of effective charges ({})
       does not match with the ensemble size ({}).
""".format(len(effective_charges), ensemble.N)

    T = ensemble.current_T
    dyn = ensemble.current_dyn

    nat = dyn.structure.N_atoms
    nat_sc = nat * np.prod(dyn.GetSupercell())
    n_rand = ensemble.N

    # Get the upsilon matrix, shape = (3 * N_at_sc, 3 * N_at_sc)
    ups_mat = dyn.GetUpsilonMatrix(T, w_pols = None)

    # Get the v, np.array with shape = (3 * N_at_sc)
    v_disp = np.einsum("ab, ib -> ia", ups_mat, ensemble.u_disps * CC.Units.A_TO_BOHR)

    # Create the effective charge array, np.array with shape (N_rand, 3 * N_at_sc, E_field)
    new_eff_charge = np.zeros((n_rand, 3 * nat_sc, 3), dtype = np.double, order = "F")

    for i in range(n_rand):
        new_eff_charge[i, :, :] = np.einsum("abc ->bac", effective_charges[i]).reshape((3, 3*nat_sc)).T
    
    N_effective = np.sum(ensemble.rho)

    # Get the average of the second derivative of the dipole moment, np.array with shape = (3 * N_at_sc, 3 * N_at_sc, 3)
    d2M_dR = np.einsum("i, ia, ibc -> abc", ensemble.rho, v_disp, new_eff_charge) / N_effective

    # Apply permutation symmetry before symmetrize
    d2M_dR += np.einsum("abc -> bac", d2M_dR)
    d2M_dR /= 2
    
    if symmetrize:
        d2M_dR = symmetrize_d2M_dR(d2M_dR, ensemble.current_dyn, verbose = True)

    return d2M_dR
    



def get_d2alpha_dR_av(ensemble, raman, w_pols = None, symmetrize = False):
    """
    COMPUTE THE SECOND DERIVATIVE OF THE DIPOLE MOMENT
    ==================================================

    Using the effective charges, we compute the derivative of the dipole moment
    
     Parameters:
    -----------
        -ensemble: the scha ensemble object on which perorm the averages
        -raman: raman tensors, np.array with shape (N_configs, E_field, E_filed, 3 * N_at_sc)
        
    Returns:
    --------
        -d2alpha_dR: the averages of the polarizability second derivative, np.array with shape (E_field, E_filed, 3 * N_at_sc)
        
    """
    assert len(raman) == ensemble.N, """
Error, the number of Raman tenors ({})
       does not match with the ensemble size ({}).
""".format(len(raman), ensemble.N)

    T = ensemble.current_T
    dyn = ensemble.current_dyn

    nat = dyn.structure.N_atoms
    nat_sc = nat * np.prod(dyn.GetSupercell())
    n_rand = ensemble.N

    # Get the upsilon matrix, shape = (3 * N_at_sc, 3 * N_at_sc) 1/BOHR^2
    ups_mat = dyn.GetUpsilonMatrix(T, w_pols = None)

    # Get the v in 1/BOHR, np.array with shape = (N_configs, 3 * N_at_sc)
    v_disp = np.einsum("ab, ib -> ia", ups_mat, ensemble.u_disps * CC.Units.A_TO_BOHR)
    
    # The effective sample size
    N_effective = np.sum(ensemble.rho)

    # Get the average of the second derivative of the polarizability, np.array with shape = (E_field, E_field, 3 * N_at_sc, 3 * N_at_sc)
    d2alpha_dR = np.einsum("i, ia, ibcd -> bcda", ensemble.rho, v_disp, raman) / N_effective

    # Apply permutation symmetry before symmetrize
    d2alpha_dR += np.einsum("abcd -> badc", d2alpha_dR)
    d2alpha_dR /= 2
    
    if symmetrize:
        d2alpha_dR = symmetrize_d2alpha_dR(d2alpha_dR, ensemble.current_dyn, verbose = True)
        

    return d2alpha_dR
        

