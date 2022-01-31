import tdscha 
import cellconstructor as CC
import sscha, sscha.Ensemble as Ensemble

import numpy as np


def get_ir_perturbation(light_in, light_out, ensemble, effective_charges, w_pols = None):
    """
    GET THE IR PERTURBATION
    =======================

    This method computes the correct IR perturbation, which includes the effect
    of non uniform effective charges.

    Parameters
    ----------
        light_in : ndarray(size = 3)
            The polarization versors of the incoming light
        light_out : ndarray(size = 3)
            The polarization versors of the outcoming light
        ensemble : sscha.Ensemble.Ensemble
            The ensemble on which you want to compute the polarizaiton
        effective_charges : list of ndarray (size = (nat, 3, 3))
            Each element of the list is a effective charge.
            They are the d^2 E / dR deps
            Where R is the atomic Cartesian coordinate and correspond to first and last index.
            eps is the electric field, and correspond to the medium index.
        w_pols : (w, pols)
            Optional. If given, avoid the diagonalization of the current dynamical matrix
    """
    raise NotImplementedError("Error, not yet implemented.")


def get_M_av(ensemble, effective_charges):
    """
    Get the average of the effective charges over the ensemble
    """

    assert len(effective_charges) == ensemble.N, """
Error, the number of effective charges ({})
       does not match with the ensemble size ({}).
""".format(len(effective_charges), ensemble.N)

    # Create the effective charge array
    dyn = ensemble.current_dyn

    n_rand = ensemble.N
    nat = dyn.structure.N_atoms
    nat_sc = nat * np.prod(dyn.GetSupercell())
    new_eff_charge = np.zeros((n_rand, 3 * nat_sc, 3), dtype = np.double, order = "F")

    for i in range(n_rand):
        new_eff_charge[i, :, :] = np.einsum("abc ->bac", effective_charges[i]).reshape((3, 3*nat_sc)).T
    
    N_effective = np.sum(ensemble.rho)
    av_eff = np.einsum("iab, i", new_eff_charge, ensemble.rho) / N_effective

    # TODO: Apply symmetries
    return av_eff


def get_dM_dR(ensemble, effective_charges, w_pols = None):
    """
    COMPUTE THE DERIVATIVE OF THE DIPOLE MOMENT
    ===========================================

    Using the effective charges, we compute the derivative of the dipole moment
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

    # Get the upsilon matrix
    ups_mat = dyn.GetUpsilonMatrix(T, w_pols = w_pols)

    # Get the v 
    v_disp = np.einsum("ab, ib -> ia", ups_mat, ensemble.u_disps * CC.Units.A_TO_BOHR)

    # Create the effective charge array
    new_eff_charge = np.zeros((n_rand, 3 * nat_sc, 3), dtype = np.double, order = "F")

    for i in range(n_rand):
        new_eff_charge[i, :, :] = np.einsum("abc ->bac", effective_charges[i]).reshape((3, 3*nat_sc)).T
    
    N_effective = np.sum(ensemble.rho)

    dM_dR = np.einsum("i, ia, ibc -> abc", ensemble.rho, v_disp, new_eff_charge) / N_effective

    # Apply permutation symmetry
    dM_dR += np.einsum("abc -> bac", dM_dR)
    dM_dR /= 2

    # TODO: Apply the symmetries

    return dM_dR
    




