"""
    This file contains all the necessary items to perfom a SSCHA calculation on the 1D chain and to save and analyse the results
"""

import chain
import numpy as np
import matplotlib.pyplot as plt
import os

import cellconstructor as CC, cellconstructor.Structure, cellconstructor.Manipulate
import cellconstructor.Phonons
import cellconstructor.Units as units

import ase
from ase.units import Ry, Bohr
import ase.calculators.calculator as calc

import sscha, sscha.Ensemble, sscha.SchaMinimizer
import sscha.Relax
import sscha.Utilities

def CellConstructor(**kwargs):
    """
        Function to create the files with the dynamical matrices of the supercell of the chain in QE format.

        First, I use the class 'supercell' and its methods, taken from 'chain', to calculate the dynamical matrix
        for a supercell with N_cells.
        The second order approximation of the BO energy around the atomic positions is used.
        There are two contributions to the total energy: ionic and electronic.
        The first is harmonic in the distance between atoms and in particular is zero when the BLA is absent.
        The second is the total energy of the occupied electronic states of a tight-binding n.n. model.

        Then, I create a structure using the 'cellconstructor' methods, I pass the useful parameters
        and in particular the dynamical matrices.

        Parameters
        ----------
            N_cells : int
                Number of cells in the supercell
            atoms_per_cell : int
                Number of atoms per single cell
            cell_length : float
                The length of the unit cell (expected in Angstrom)
            onsite : float or array of floats
                Onsite energy of each atom in the supercell
                If float, an array is created with alternating +onsite -onsite
            hopping : float
                Hopping energy in the case of equidistant atoms
            N_k : int
                Number of k-points for the electronic Hamiltonian
            mass : float or array of floats
                Mass of each atom in the supercell
                If float, an array of ones*mass is created
            displacement : float or array of floats
                Displacement of each atom from the case of equidistant atoms
                It must be given in fractional units
                If float, an array with [0, displacement, 0, displacement, ...] is created
            el_ph : float
                Electron-phonon coupling parameter
            k_ela : float
                Elastic constant to account for ion-ion harmonic interaction
            default : bool
                If True, all displacements are set to 0
            file_name : string
                String with the name of the file where the dynamical matrices will be saved
    """

    # ==========================================================================
    # ----- create the supercell using the correspondent class in 'chain' ------
    # ==========================================================================

    N_cells = kwargs['N_cells']
    atoms_per_cell = kwargs['atoms_per_cell']
    cell_length = kwargs['cell_length'] # length of the unit cell (in A)
    displacement = kwargs['displacement'] # in fractional units (w.r.t. the unit cell length)
    default = kwargs['default'] # if True, all displacements will be set to zero
    onsite = kwargs['onsite'] # on-site energy on the atoms (in eV)
    t = kwargs['hopping'] # hopping energy (in eV)
    beta = kwargs['el_ph'] # el-ph coupling parameter (in eV/A)
    N_k = kwargs['N_k'] # number of k-points for the electronic calculation
    k_ela = kwargs['k_ela'] # elastic constant for the ionic contribution (in eV/A^2)
    mass = kwargs['mass'] # in Ry to be consistent with SSCHA code (carbon mass = 10947.)

    # create the supercell; if default=True, all displacements will be set to zero
    supercell = chain.supercell(N_cells = N_cells, atoms_per_cell = atoms_per_cell, cell_length = cell_length, displacement = displacement,
                                default = default, onsite = onsite, hopping = t, el_ph = beta, k_ela = k_ela, mass = mass)

    # compute electronic bands without plotting them
    eps_k, psi_k = supercell.compute_elec_bands(N_k=N_k, plot=False)

    # compute the force constants matrix
    force_constants = supercell.compute_force_constants(psi_k=psi_k, eps_k=eps_k)

    # compute the dynamical matrices for each q without plotting phonons dispersion
    dyn_matrix, omega_q, v_q = supercell.compute_dyn_matrix(force_constants, plot=False)
    from cellconstructor.Units import RY_TO_CM
    RyToTHz=3289.84377
    print("omega from chain (cm): \n", omega_q)
    print("omega from chain (THz): \n", omega_q*RyToTHz/RY_TO_CM)

    # ==========================================================================
    # --- create the structure using CellConstructor and passing parameters ----
    # ==========================================================================

    # create the structure with N_atoms in the unit cell (i.e. the atoms_per_cell of the supercell instance)
    N_atoms = atoms_per_cell
    structure = CC.Structure.Structure(N_atoms)

    # initialise the unit cell
    structure.has_unit_cell = True
    structure.unit_cell = 10*np.eye(3) # basis vectors of the unit cell (in A); ndarray of dimension (3,3)
    structure.unit_cell[0,0] = cell_length

    # atoms' coordinates (in Angstrom)
    structure.coords[:,:] = 0 # ndarray of dimension (N_atoms, 3)
    structure.coords[0,0] = supercell.atom[0].position*cell_length # in Angstrom
    structure.coords[1,0] = supercell.atom[1].position*cell_length # in Angstrom
    # name of each atom
    structure.atoms[0] = "C"
    structure.atoms[1] = "C"
    # dictionary with masses of the atoms associated with their name (mass in Ry)
    structure.set_masses( {"C": mass, "C": mass} ) # carbon mass: 10947 Ry

    # to generate the dynamical matrix at Gamma point (q=0)
    dyn = CC.Phonons.Phonons(structure) # takes the structure in input

    # to generate N_q evenly-spaced q-points in the range from q_min to q_max (q in unit of 2pi/A)
    N_q = N_cells+1
    q_min = 0.
    q_max = 1./cell_length
    q = np.linspace(q_min,q_max,N_q)

    # to interpolate the dynamical matrix on a N_qx1x1 mesh (starting from the 1x1x1)
    dyn_supercell = dyn.Interpolate((1,1,1), (N_cells, 1,1))

    for iq in range(len(dyn_supercell.dynmats[:])):
        # iq-th dynamical matrix, dimension (3N_atoms x 3N_atoms)
        dyn_supercell.dynmats[iq][:,:] = 0
        for alpha in range(N_atoms):
            for gamma in range(N_atoms):
                # I fill only the (alpha_x,gamma_x) term for each matrix
                dyn_supercell.dynmats[iq][alpha*3,gamma*3] = dyn_matrix[alpha,gamma,iq]*mass #*Bohr**2/Ry)
        dyn_supercell.q_tot[iq][:] = [q[iq],0,0] # vector with components of the iq-th q-point (units of 2pi/A)

    # adjust for equivalent q-points
    dyn_supercell.AdjustQStar()
    dyn_supercell.Symmetrize()

    # to avoid files redundancy, I remove the pre-existing files
    file_name = kwargs['file_name']
    for fname in os.listdir("."):
        if fname.startswith(file_name):
            os.remove(fname)
    # save the dynamical matrices in QE format
    dyn_supercell.save_qe(file_name)


class ToyModelCalculator(calc.Calculator):

    def __init__(self, *args, **kwargs):
        """
            This class defines the calculator for the 1d-chain toy model

            Parameters
            ----------
                N_cells : int
                    Number of cells in the supercell
                atoms_per_cell : int
                    Number of atoms per single cell
                cell_length : float
                    The length of the unit cell (expected in Angstrom)
                onsite : float or array of floats
                    Onsite energy of each atom in the supercell
                    If float, an array is created with alternating +onsite -onsite
                hopping : float
                    Hopping energy in the case of equidistant atoms
                N_k : int
                    Number of k-points for the electronic Hamiltonian
                mass : float or array of floats
                    Mass of each atom in the supercell
                    If float, an array of ones*mass is created
                displacement : float or array of floats
                    Displacement of each atom from the case of equidistant atoms
                    It must be given in fractional units
                    If float, an array with [0, displacement, 0, displacement, ...] is created
                el_ph : float
                    Electron-phonon coupling parameter
                k_ela : float
                    Elastic constant to account for ion-ion harmonic interaction
                default : bool
                    If True, all displacements are set to 0
        """

        calc.Calculator.__init__(self, *args, **kwargs)

        # Setup what properties the calculator can load
        self.implemented_properties = ["energy", "forces", "effective charges"]

        # # load the dynamical matrices
        # N_files = 0
        # for fname in os.listdir("."):
        #     if fname.startswith("start_dyn"):
        #         N_files += 1
        # start_dyn = CC.Phonons.Phonons("start_dyn", N_files)

        # Parameters of the 1D toy-model
        self.N_cells = kwargs['N_cells']
        self.atoms_per_cell = kwargs['atoms_per_cell']
        self.cell_length = kwargs['cell_length'] # length of the unit cell (in A)
        self.onsite = kwargs['onsite'] # on-site energy on the atoms (in eV)
        self.hopping = kwargs['hopping'] # hopping energy (in eV)
        self.el_ph = kwargs['el_ph'] # el-ph coupling parameter (in eV/A)
        self.N_k = kwargs['N_k'] # number of k-points for the electronic calculation
        self.k_ela = kwargs['k_ela'] # elastic constant for the ionic contribution (in eV/A^2)
        self.N_atoms = self.atoms_per_cell*self.N_cells

    def calculate(self, atoms=None, *args, **kwargs):
        """
        COMPUTES ENERGY AND FORCES IN eV and eV/ ANGSTROM
        =================================================

        Returns:
        -------
            self.results: a dict with energy and forces
        """

        self.N_atoms = self.N_cells*self.atoms_per_cell
        calc.Calculator.calculate(self, atoms, *args, **kwargs)

        N_atoms = self.N_atoms

        # Energy and forces in eV and eV/A
        energy = 0. # (in eV)
        forces = np.zeros((N_atoms, 3), dtype = np.double) # (eV/A) only the component along x can be non-zero

        # I take the positions of the displaced atoms from the SSCHA calculation
        coords = atoms.get_positions() # array (N_atoms, 3) with atoms' coordinates (in A)
        position = np.array([coords[i,0] for i in range(N_atoms)]) # I only need the components along x
        position /= self.cell_length # convert in units of cell_length (fractional units)
        # print("POSIZIONI DA SSCHA: ",position)

        # compute the equilibrium positions of each atom in the case of equidistant atoms (in fractional units)
        eq_position = np.array([[cell_label + atom_label/self.atoms_per_cell for atom_label in range(self.atoms_per_cell)] for cell_label in range(self.N_cells)]).ravel()
        # print("EQ POSITION: ",eq_position)

        # calculate the displacement of each atom (in fractional units)
        displacement = position - eq_position
        # print("DISPLACEMENT: ",displacement)

        # generate supercell with displaced atoms (mass is a dummy variable can be any float)
        supercell = chain.supercell(displacement = displacement, default = False, N_cells = self.N_cells, atoms_per_cell = self.atoms_per_cell, mass = 1.,
                                    cell_length = self.cell_length, onsite = self.onsite, hopping = self.hopping, el_ph = self.el_ph, k_ela = self.k_ela)
        # print("POSIZIONI DA CHAIN: ",[supercell.atom[i].position for i in range(N_atoms)])

        # compute eigen-energies and wave functions of the supercell with displaced atoms
        eps_k, psi_k = supercell.compute_elec_bands(N_k = self.N_k, plot = False)

        # calculate the total energy of the supercell with displaced atoms
        energy = supercell.E_tot(eps_k)

        # calculate the forces acting on each atom using perturbation theory
        forces[:,0] = supercell.compute_forces(psi_k)

        # Compute the effective charges
        zeff = np.zeros((N_atoms, 3, 3), dtype = np.double)
        for alpha in range(N_atoms):
            zeff[alpha, 0, 0] = supercell.Zeff_alpha(alpha = alpha,
                                                     eps_k = eps_k,
                                                     psi_k = psi_k)

        self.results = {"energy": energy, "forces": forces, "effective charges" : zeff}

        return self.results


def test_forces(*args, **kwargs):
    """
        This function test that the forces calculated with the calculator are the opposite of the derivatives of the total energies calculated with the calculator

            Parameters
            ----------
                N_cells : int
                    Number of cells in the supercell
                atoms_per_cell : int
                    Number of atoms per single cell
                cell_length : float
                    The length of the unit cell (expected in Angstrom)
                onsite : float or array of floats
                    Onsite energy of each atom in the supercell
                    If float, an array is created with alternating +onsite -onsite
                hopping : float
                    Hopping energy in the case of equidistant atoms
                N_k : int
                    Number of k-points for the electronic Hamiltonian
                el_ph : float
                    Electron-phonon coupling parameter
                k_ela : float
                    Elastic constant to account for ion-ion harmonic interaction
                plot : bool
                    If True, minus the numerical gradient of the energy vs the analytical gradient is plotted
                starting_file : string
                    String with the name of the file with the starting matrices
            Return
            ------
                check : bool
                    If True, minus the numerial gradient of the forces correspond to the forces computed with the calculator
    """

    # load the dynamical matrices
    start_file = kwargs['starting_file']
    N_files = 0
    for fname in os.listdir("."):
        if fname.startswith(start_file):
            N_files += 1
    start_dyn = CC.Phonons.Phonons(start_file, N_files)
    # to avoid immaginary frequencies
    start_dyn.ForcePositiveDefinite()
    start_dyn.Symmetrize()

    # setup the calculator's properties
    N_cells = kwargs['N_cells']
    atoms_per_cell = kwargs['atoms_per_cell']
    cell_length = kwargs['cell_length'] # length of the unit cell (in A)
    onsite = kwargs['onsite'] # on-site energy on the atoms (in eV)
    hopping = kwargs['hopping'] # hopping energy (in eV)
    N_k = kwargs['N_k'] # number of k-points for the electronic calculation
    el_ph = kwargs['el_ph'] # el-ph coupling parameter (in eV/A)
    k_ela = kwargs['k_ela'] # elastic constant for the ionic contribution (in eV/A^2)

    # setup the calculator
    calc = ToyModelCalculator(N_cells = N_cells, atoms_per_cell = atoms_per_cell, cell_length = cell_length,
                              onsite = onsite, hopping = hopping, N_k = N_k, el_ph = el_ph, k_ela = k_ela)

    # we compute the energies and the forces using the calculator on a random structure extracted with given dynamical matrices at a given temperature
    TEMPERATURE = 0
    struct = start_dyn.ExtractRandomStructures(T = TEMPERATURE, lock_low_w = True)[0]

    # we move the atom i of the extracted structure for different displacements in an interval and compute energies and forces for each
    atom_i = 0

    ddispl = 0.01
    N_displ = 50
    displ_max = 0.4
    displ = np.linspace(struct.coords[atom_i,0], struct.coords[atom_i, 0] + displ_max, N_displ)

    energy = np.zeros(N_displ)
    forces = np.zeros(N_displ)

    for i in range(N_displ):
        struct.coords[atom_i, 0] = displ[i]
        atoms = struct.get_ase_atoms()
        atoms.set_calculator(calc)
        en = atoms.get_total_energy()
        f = atoms.get_forces()

        energy[i] = en
        forces[i] = f[atom_i, 0]

    tol = 1e-4
    check = np.all(forces[1:-1] + np.gradient(energy, displ)[1:-1] < tol)

    if kwargs['plot']:
        plt.plot(displ, -np.gradient(energy, displ), label = "Numerical gradient")
        plt.plot(displ, forces, label = "Analytical gradient")
        plt.plot(displ, forces+np.gradient(energy, displ), label=  "Difference")
        plt.legend()
        plt.show()

    return check


def plot_BLA():
    """
        This function plots the behaviour of the BLA vs the minimisation steps
    """

    # I load all the structures of the minimisation from the file atom_pos.scf. There is one strucure for each step
    all_structs = CC.Manipulate.load_scf_trajectory("atom_pos.scf")

    def get_BLA(struct):
        return 2*np.abs(np.abs(struct.coords[1,0] - struct.coords[0,0]) - struct.unit_cell[0,0])

    all_BLA = [get_BLA(s) for s in all_structs]

    plt.plot(all_BLA, marker = ".")
    plt.ylabel("BLA [$\AA$]")
    plt.show()


def auto_relax_SSCHA(*args, **kwargs):
    """
        This function initialise and perform automatically the SSCHA minimisation

        Parameters for the minimisation
        ------------------------------
            TEMPERATURE : int
                Temperature at which the minimisation is performed
            N_CONFIG : int
                Number of configuration to extract at each recalculation
            MAX_ITERATIONS : int
                How many SSCHA recalculation to do automatically at most
            plot : bool
                If True, a plot of the results of the minimisation is shown
            minim_struct : bool
                If False, does not minimise the structure and calculate the properties of the input structure
            starting_file : string
                String with the name of the file with the starting matrices
            data_file_name : string
                String with the name of the file where the data of the minimisation will be stored
        Parameters to setup the calculator
        ----------------------------------
            N_cells : int
                Number of cells in the supercell
            atoms_per_cell : int
                Number of atoms per single cell
            cell_length : float
                The length of the unit cell (expected in Angstrom)
            onsite : float or array of floats
                Onsite energy of each atom in the supercell
                If float, an array is created with alternating +onsite -onsite
            hopping : float
                Hopping energy in the case of equidistant atoms
            N_k : int
                Number of k-points for the electronic Hamiltonian
            el_ph : float
                Electron-phonon coupling parameter
            k_ela : float
                Elastic constant to account for ion-ion harmonic interaction
    """

    # parameters for the minimisation
    TEMPERATURE = kwargs['TEMPERATURE']
    N_CONFIG = kwargs['N_CONFIG']
    MAX_ITERATIONS = kwargs['MAX_ITERATIONS']

    # load the dynamical matrices
    start_file = kwargs['starting_file']
    N_files = 0
    for fname in os.listdir("."):
        if fname.startswith(start_file):
            N_files += 1
    start_dyn = CC.Phonons.Phonons(start_file, N_files)
    # to avoid immaginary frequencies
    start_dyn.Symmetrize()
    start_dyn.ForcePositiveDefinite()
    # to obtain the classical limit
    # start_dyn.structure.masses["C"] = 1e8

    # setup the ensemble
    ensemble = sscha.Ensemble.Ensemble(start_dyn, TEMPERATURE)
    ensemble.ignore_small_w = True

    # setup the calculator's properties
    N_cells = kwargs['N_cells']
    atoms_per_cell = kwargs['atoms_per_cell']
    cell_length = kwargs['cell_length'] # length of the unit cell (in A)
    onsite = kwargs['onsite'] # on-site energy on the atoms (in eV)
    hopping = kwargs['hopping'] # hopping energy (in eV)
    N_k = kwargs['N_k'] # number of k-points for the electronic calculation
    el_ph = kwargs['el_ph'] # el-ph coupling parameter (in eV/A)
    k_ela = kwargs['k_ela'] # elastic constant for the ionic contribution (in eV/A^2)

    # setup the calculator
    calc = ToyModelCalculator(N_cells = N_cells, atoms_per_cell = atoms_per_cell, cell_length = cell_length,
                              onsite = onsite, hopping = hopping, N_k = N_k, el_ph = el_ph, k_ela = k_ela)

    # setup the minimizer
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
    minim.kong_liu_ratio = 0.5
    minim.meaningful_factor = 0.0001
    minim.minim_struct = kwargs['minim_struct'] # if False, does not minimize the structure
    minim.set_minimization_step(0.01)

    # utilities to save useful info
    IOutils = sscha.Utilities.IOInfo()
    IOutils.SetupAtomicPositions("atom_pos.scf") # save the atomic position at each step
    IOutils.SetupSaving("frequencies.dat") # save the frequencies

    # setup the automatic relaxation
    relax = sscha.Relax.SSCHA(minim, ase_calculator = calc, N_configs = N_CONFIG, max_pop = MAX_ITERATIONS)
    relax.setup_custom_functions(custom_function_post = IOutils.CFP_SaveAll)
    relax.relax(get_stress = False, start_pop = 1)

    # save the results
    relax.minim.finalize()
    relax.minim.dyn.save_qe("final_dyn")
    # displacement = 0.0 # fractional units
    # fname = "minim_data_%f.dat"%displacement
    fname = kwargs['data_file_name']
    plot = kwargs['plot']
    relax.minim.plot_results(save_filename = fname, plot = plot)


def get_hessian(*args, **kwargs):
    """
        This function initialise and perform automatically the SSCHA minimisation

        Parameters for the minimisation
        ------------------------------
            TEMPERATURE : int
                Temperature at which the minimisation is performed
            N_CONFIG : int
                Number of configuration to extract at each recalculation
            MAX_ITERATIONS : int
                How many SSCHA recalculation to do automatically at most
            plot : bool
                If True, a plot of the results of the minimisation is shown
            minim_struct : bool
                If False, does not minimise the structure and calculate the properties of the input structure

        Parameters to setup the calculator
        ----------------------------------
            N_cells : int
                Number of cells in the supercell
            atoms_per_cell : int
                Number of atoms per single cell
            cell_length : float
                The length of the unit cell (expected in Angstrom)
            onsite : float or array of floats
                Onsite energy of each atom in the supercell
                If float, an array is created with alternating +onsite -onsite
            hopping : float
                Hopping energy in the case of equidistant atoms
            N_k : int
                Number of k-points for the electronic Hamiltonian
            el_ph : float
                Electron-phonon coupling parameter
            k_ela : float
                Elastic constant to account for ion-ion harmonic interaction
    """
