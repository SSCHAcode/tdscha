import tdscha 
import cellconstructor as CC
import sscha, sscha.Ensemble as Ensemble

from tdscha.DynamicalLanczos import __RyToK__
import numpy as np

from tdscha.DynamicalLanczos import TYPE_DP


def get_ir_perturbation(light_in, ensemble, effective_charges, frequencies, pols):
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
        frequencies : ndarray(n_modes)
            The frequencies of the active frequencies. 
        pols : ndarray(3n_atoms, n_modes)
            The polarization vectors


    Results
    -------
        psi_right, psi_left : ndarray
            The two vectors of the response function.
            psi_right is the perturbation (p in the Monacelli Mauri PRB, 2021)
            psi_left is the response (q in the Monacelli Mauri PRB, 2021)
    """

    Z_av = get_Z_av(ensemble, effective_charges)
    dZ_dR_av = get_dZ_dR(ensemble, effective_charges)

    # Contract the effective charges with the two dypole moments
    Z_av_new = np.einsum("a, ab -> b", light_in, Z_av)
    dZ_dR_av_new = np.einsum("a, abc -> bc", light_in, dZ_dR_av)

    supercell_structure = ensemble.current_dyn.structure.generate_supercell(ensemble.current_dyn.GetSupercell())
    mass = np.tile(supercell_structure.get_masses_array(), (3, 1)).T.ravel()

    standard_response = np.einsum("a, ab -> b", Z_av_new / np.sqrt(mass), pols)

    dZ_over_M = dZ_dR_av_new / np.outer( np.sqrt(mass), np.sqrt(mass))
    dZ_munu = np.einsum("ab, ai, bj -> ij", dZ_over_M, pols, pols)


    # Prepare the symmetric representation of ReA and Y
    T = ensemble.current_T
    n_modes = len(frequencies)
    i_a = np.tile(np.arange(n_modes), (n_modes,1)).ravel()
    i_b = np.tile(np.arange(n_modes), (n_modes,1)).T.ravel()

    new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
    new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
    
    w_a = frequencies[new_i_a]
    w_b = frequencies[new_i_b]

    N_w2 = len(w_a)

    psi_right = np.zeros(n_modes + 2 * N_w2, dtype = TYPE_DP)
    psi_left = np.zeros(n_modes + 2 * N_w2, dtype = TYPE_DP)

    # TODO: CHECK THE SIGN
    psi_right[:n_modes] = -standard_response
    psi_left[:n_modes] = standard_response

    n_a = np.zeros(np.shape(w_a), dtype = TYPE_DP)
    n_b = np.zeros(np.shape(w_a), dtype = TYPE_DP)
    if T > 0:
        n_a = 1 / (np.exp( w_a / np.double(T / __RyToK__)) - 1)
        n_b = 1 / (np.exp( w_b / np.double(T / __RyToK__)) - 1)

    # Get the places where the perturbations start
    start_Y = n_modes
    start_A = n_modes + N_w2

    psi_right[start_Y : start_A] = 2 * w_a / (2*n_a + 1) + 2 * w_b / (2*n_b + 1)
    psi_right[start_A :] = 2 * (n_a + 1) * n_a * w_a  / (2 * n_a + 1)    
    psi_right[start_A :] += 2 * (n_b + 1) * n_b * w_b  / (2 * n_b + 1)    

    psi_left[start_Y : start_A] = -(2*n_a + 1)*(2*n_b + 1) / (8 * w_a * w_b)

    for i, xa in enumerate(new_i_a):
        xb = new_i_b[i]

        psi_right[start_Y + i] *= dZ_munu[xa, xb]
        psi_right[start_A + i] *= dZ_munu[xa, xb]
        psi_left[start_Y + i]  *= dZ_munu[xa, xb]


    return psi_right, psi_left

def get_Z_av(ensemble, effective_charges):
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


def get_dZ_dR(ensemble, effective_charges, mean_eff_charge, w_pols = None):
    """
    COMPUTE THE DERIVATIVE OF THE DIPOLE MOMENT
    ===========================================

    Using the effective charges, we compute the derivative of the dipole moment

    Parameters
    ----------
        ensemble : sscha.Ensemble.Ensemble
            The SSCHA ensemble
        effective_charges: list
            List of the effective charges, each one must be a ndarray
            of size  (nat, 3, 3). The middle index indicates the electric field.
        mean_eff_charge: ndarray(size = (nat, 3, 3))
            The average effective charge.
        w_pols : tuple
            frequencies and polarization vectors of the current dynamical matrix
            if given, avoids to dyagonalize the dynamical matrix once again.
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
        new_eff_charge[i, :, :] = np.einsum("abc ->bac", effective_charges[i] - mean_eff_charge).reshape((3, 3*nat_sc)).T
    
    N_effective = np.sum(ensemble.rho)

    dM_dR = np.einsum("i, ia, ibc -> abc", ensemble.rho, v_disp, new_eff_charge) / N_effective

    # Apply permutation symmetry
    dM_dR += np.einsum("abc -> bac", dM_dR)
    dM_dR /= 2

    # TODO: Apply the symmetries

    return dM_dR
    




