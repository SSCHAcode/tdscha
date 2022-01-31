import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class atom:
    def __init__(self, *args, **kwargs):
        """
            This class represent a single atom
            All lengths are in fractional units

            Parameters
            ----------
                atoms_per_cell : int
                    Number of atoms per single cell
                cell_label : int
                    The label of the cell which contains the atom
                atom_label : int
                    The label of the atom inside the cell
                displacement : float
                    Displacement of the atom from the equilibrium position of equidistant atoms
                onsite_energy : float
                    Onsite energy of the atom (in eV)
                mass : float (in Ry)
                    Mass of the atom
        """

        self.atoms_per_cell = kwargs['atoms_per_cell'] # number of atoms in a single cell
        self.cell_label = kwargs['cell_label'] # label of the cell
        self.atom_label = kwargs['atom_label'] # label of the atom in the cell
        self.displacement = kwargs['displacement'] # displacement from equilibrium in fractional units
        self.onsite_energy = kwargs['onsite_en'] # on-site energy of the atom in tight-binding approximation, in eV
        self.mass = kwargs['mass'] # mass of the atom (in Ry if we want consistency with SSCHA)
        self.eq_position = self.cell_label + self.atom_label/self.atoms_per_cell # eq. position along the chain in the case equidistant atoms
        self.position = self.eq_position + self.displacement # position of the atom taking into account also the displacement

    def __str__(self):
        """
            Redefine what the default print() method does when an "atom" instance is passed
        """
        return "Atom %d in cell %d, eq position: %.4f (fractional units), displacement: %.4f (fractional units), on-site energy: %.2f eV, mass: %.2f Ry"\
                % (self.atom_label,self.cell_label,self.eq_position,self.displacement,self.onsite_energy,self.mass)


class supercell:
    def __init__(self, *args, **kwargs):
        """
            This class represent the supercell of the 1D-chain.

            Parameters
            ----------
                cell_length : float
                    The length of the unit cell (in Angstrom)
                N_cells : int
                    Number of cells in the supercell
                atoms_per_cell : int
                    Number of atoms per single cell
                onsite : float or array of floats
                    Onsite energy of each atom in the supercell (in eV)
                    If float, an array is created with alternating +onsite -onsite
                hopping : float
                    Hopping energy in the case of equidistant atoms (in eV)
                mass : float or array of floats
                    Mass of each atom in the supercell (in Ry)
                    If float, an array of ones*mass is created
                displacement : float or array of floats
                    Displacement of each atom from the case of equidistant atoms
                    It must be given in fractional units
                    If float, an array with [0, displacement, 0, displacement, ...] is created
                el_ph : float
                    Electron-phonon coupling parameter (in eV/Angstrom)
                k_ela : float
                    Elastic constant to account for ion-ion harmonic interaction (in eV/Angstrom^2)
                default : bool
                    If True, all displacements are set to 0
        """

        self.cell_length = kwargs['cell_length'] # length of the cell
        self.N_cells = kwargs['N_cells'] # number of cells in the supercell (better if even number)
        self.atoms_per_cell = kwargs['atoms_per_cell'] # number of atoms per cell
        self.N_atoms = self.N_cells*self.atoms_per_cell # total number of atoms in the supercell
        self.supercell_length = self.N_cells*self.cell_length # length of the supercell
        self.onsite = kwargs['onsite']
        self.hopping = kwargs['hopping']
        self.el_ph = kwargs['el_ph']
        self.k_ela = kwargs['k_ela']
        self.mass = kwargs['mass']
        self.displacement = kwargs['displacement'] # it must be given in fractional units
        self.default = kwargs['default']

        # we create the atoms in the supercell, given a number of cells and a number of atoms per cell
        N_atoms = self.N_atoms
        N_cells = self.N_cells
        atoms_per_cell = self.atoms_per_cell
        cell_length = self.cell_length
        beta = self.el_ph
        mass = self.mass
        onsite = self.onsite
        displacement = self.displacement

        if self.default:
            displacement = np.zeros(N_atoms)
            self.displacement = displacement
        elif isinstance(displacement, float):
            # displacement = np.array([[ displacement*(1-(-1)**(i+j*atoms_per_cell)) for i in range(atoms_per_cell)] for j in range(N_cells) ])
            displ = displacement
            displacement = np.array([(-1)**(i+1)*(displ*0.5) for i in range(N_atoms)])
            displacement += displ*0.5
            self.displacement = displacement
        if isinstance(onsite, float):
            Delta = onsite
            onsite = np.array([(-1)**i*Delta for i in range(N_atoms)]) #array with the on-site energy for each atom (in eV)
            self.onsite = onsite
        if isinstance(mass, float):
            mass = np.ones(N_atoms)*mass #array with the mass of each atom (in Ry)
            self.mass = mass

        # array where each atom in the supercell is stored with its properties using the 'atom' class
        # This array is the 'computer' equivalent of the physical array of atoms
        self.atom = np.array(   [  [    atom(atoms_per_cell=atoms_per_cell,atom_label=i,cell_label=j,displacement=displacement[i+j*atoms_per_cell],
                                            onsite_en=onsite[i+j*atoms_per_cell], mass=mass[i+j*atoms_per_cell])
                                    for i in range(atoms_per_cell)]
                                for j in range(N_cells)]
                            ).ravel()

        # array with the variation of hopping due to the displacement of each atom, to account for e-ph coupling
        self.dt = -beta*cell_length*np.array([(self.atom[(i+1)%N_atoms].displacement-self.atom[i].displacement) for i in range(N_atoms)])

    def print_info(self):
        """
            To print info about the atoms in the supercell using the __str__ method of the class "atom"
        """
        for i in range(self.N_atoms):
            print(self.atom[i])
        pass


    def draw(self):
        """
            To draw the supercell with all the atoms, highlighing their displacement from eq positions
        """

        N_atoms = self.N_atoms
        fig = plt.figure()
        fig.set_size_inches(11.5,3.)
        plt.yticks([])
        plt.xlabel('[$\AA$]',fontsize = 20)
        ax = plt.gca()
        atoms_pos = np.array([self.atom[i].position for i in range(N_atoms)])*self.cell_length # in Angstrom
        eq_pos = np.array([self.atom[i].eq_position for i in range(N_atoms)])*self.cell_length # in Angstrom
        y = np.zeros(N_atoms)
        ax.scatter(atoms_pos,y,s=150,c='r',marker='o')
        ax.scatter(eq_pos,y,s=20,c='k',marker='.')
        ax.tick_params(axis='x', labelsize=20)
        ax.xaxis.grid(True, which='minor')
        ax.legend(['atom displaced','equilibrium position if equidistant atoms'])
        for i in range(N_atoms):
            ax.plot([atoms_pos[i],eq_pos[i]],[0.,0.],color='k') # in Angstrom


    def compute_elec_bands(self, N_k, plot=False, **kwargs):
        """
            Compute the electronic eigen-energies and eigen-functions in k space

            Parameters
            ----------
                N_k : int
                    Number of k points to be used
                plot : bool
                    If true, a plot of the electronic bands is done using the plot_elec_bands() function
            Return
            ------
                eps_k : array of shape (N_atoms,N_k) containing floats
                    Array with the eigenenergy for each band and k-point (in eV)
                psi_k : array of shape (N_atoms,N_atoms,N_k) containing complex_
                    Array with the eigenvectors for each band and for each k-point
        """

        N_atoms = self.N_atoms
        supercell_length = self.supercell_length
        cell_length = self.cell_length
        t = self.hopping # hopping energy for equidistant atoms, in eV
        dt = self.dt # variation of hopping due to the displacement of the atoms

        # quasi-momentum k
        N_k = N_k
        k_min = -np.pi/supercell_length
        k_max = np.pi/supercell_length
        k = np.linspace(k_min,k_max,N_k)

        # we find eigenvalues eps(k) and eigenvectors psi(k) for each k
        eps_k = np.zeros((N_atoms,N_k))
        psi_k = np.zeros((N_atoms,N_atoms,N_k),dtype=np.complex_)

        for i_k in range(N_k):
            # we build the Hamiltonian matrix of the system for a given k
            H = np.zeros((N_atoms,N_atoms),dtype=np.complex_)
            if N_atoms!=2:
                for i in range(N_atoms):
                    # H[i,i] = self.atom[i].onsite_energy*0.5
                    # H[0,N_atoms-1] = -(t+dt[N_atoms-1])*np.exp(-1j*k[i_k]*cell_length*(self.atom[i].displacement - self.atom[N_atoms-1].displacement + 0.5))
                    # H[i,(i+1)%N_atoms] = -(t+dt[i])*np.exp(1j*k[i_k]*cell_length*(self.atom[(i+1)%N_atoms].displacement - self.atom[i].displacement + 0.5))
                    # H[N_atoms-1,0] = 0
                    for j in range(i,N_atoms):
                        if j==i:
                            H[i,j]=self.atom[i].onsite_energy*0.5
                        elif j==i+1:
                            H[i,j]=-(t+dt[i])*np.exp(1j*k[i_k]*cell_length*(self.atom[j].displacement - self.atom[i].displacement + 0.5))
                        elif i==0 and j==(N_atoms-1):
                            H[i,j]=-(t+dt[j])*np.exp(-1j*k[i_k]*cell_length*(self.atom[i].displacement - self.atom[j].displacement + 0.5))
            else:
                H[0,0] = self.atom[0].onsite_energy*0.5
                H[1,1] = self.atom[1].onsite_energy*0.5
                H[0,1] = -(t+dt[0])*np.exp(1j*k[i_k]*cell_length*(self.atom[1].displacement - self.atom[0].displacement + 0.5)) \
                         -(t+dt[1])*np.exp(-1j*k[i_k]*cell_length*(self.atom[0].displacement - self.atom[1].displacement + 0.5))
            # print(H)
            H += np.matrix.getH(H)
            eps_k[:,i_k], psi_k[:,:,i_k] = np.linalg.eigh(H)

        if plot:
            self.plot_elec_bands(eps_k)

        return eps_k, psi_k


    def plot_elec_bands(self, eps_k, **kwargs):
        """
            Function to plot the electronic bands of the supercell

            Parameters
            ----------
                eps_k : array of shape (N_atoms,N_k) containing floats
                    Array with the values of the energy for each band and k-point (in eV)
        """

        N_atoms = self.N_atoms
        supercell_length = self.supercell_length

        N_k = len(eps_k[0,:])
        k_min = -np.pi/supercell_length
        k_max = np.pi/supercell_length
        k = np.linspace(k_min,k_max,N_k)

        fig = plt.figure()
        ax = plt.gca()
        for i in range(N_atoms):
            if i<N_atoms*0.5:
                ax.plot(k,eps_k[i,:],c='b',marker='.')
            else:
                ax.plot(k,eps_k[i,:],c='r',marker='.')
        plt.hlines(0.0,k_min,k_max,linestyles='dashed',color='k')


    def E_gap(self, eps_k, *args, **kwargs):
        """
            Function to compute the electronic band-gap

            Parameters
            ----------
                eps_k : array of shape (N_atoms,N_k) containing floats
                    Array with the values of the energy for each band and k-point (in eV)

            Return
            ------
                E_gap : float
                    Value of the electronic band-gap (in eV)
        """

        N_atoms = self.N_atoms
        N_occ = int(N_atoms*0.5)
        # E_gap = 0.

        E_gap = np.amin( eps_k[N_occ,:] - eps_k[N_occ-1,:] )

        return E_gap


    def E_tot(self, eps_k, **kwargs):
        """
            To compute the total energy of the system

            Parameters
            ----------
                eps_k : array of shape (N_atoms,N_k) containing floats
                    Array with the values of the energy for each band and k-point (in eV)

            Return
            ------
                E_tot : float
                    Value of the total energy of the supercell (in eV)
        """

        N_atoms = self.N_atoms

        # ion-ion interaction as harmonic
        k_ela = self.k_ela  # elastic constant in units of eV/A^2
        E_ion = 0.5*k_ela*self.cell_length**2*np.sum([(self.atom[(i+1)%N_atoms].displacement-self.atom[i].displacement)**2 for i in range(N_atoms)])

        # to obtain the electronic contribution we integrate over the occupied bands
        N_occ = int(N_atoms*0.5)
        eps_k = eps_k
        N_k = len(eps_k[0,:])
        E_elec = 2*np.sum(eps_k[:N_occ,:])/N_k  # the 2 is for the spin
        E_tot = E_ion + E_elec

        return E_tot


    def dH_ddispl(self,alpha,k):
        """
            This function calculate the derivative of the electronic Hamiltonian w.r.t. the dispclacement of an atom

            Parameters
            ----------
                alpha : int
                    The index of the atom with respect to which the derivative is taken
                k : float
                    It is the k-point on which the derivative is computed

            Return
            ------
                dH_ddispl : array with shape (N_atoms,N_atoms) with complex_
                    Derivative of the electronic Hamiltonian w.r.t displacement of atom alpha at point k (in eV/Angstrom)
        """

        N_atoms = self.N_atoms
        cell_length = self.cell_length
        t = self.hopping
        dt = self.dt
        beta = self.el_ph
        displacement = self.displacement
        dH_ddispl = np.zeros((N_atoms,N_atoms),dtype=np.complex_)

        if N_atoms!=2:
            dH_ddispl[alpha-1,alpha] = -((t+dt[alpha-1])*1j*k - beta)*np.exp(1j*k*cell_length*(displacement[alpha] - displacement[alpha-1] + 0.5))
            dH_ddispl[alpha,(alpha+1)%N_atoms] = ((t+dt[alpha])*1j*k - beta)*np.exp(1j*k*cell_length*(displacement[(alpha+1)%N_atoms] - displacement[alpha] + 0.5))
        else:
            if alpha==0:
                dH_ddispl[0,1] =  ((t+dt[0])*1j*k - beta)*np.exp(1j*k*cell_length*(displacement[1] - displacement[0] + 0.5))
                dH_ddispl[1,0] = -((t+dt[1])*1j*k - beta)*np.exp(1j*k*cell_length*(displacement[0] - displacement[1] + 0.5))
            elif alpha==1:
                dH_ddispl[0,1] = -((t+dt[0])*1j*k - beta)*np.exp(1j*k*cell_length*(displacement[1] - displacement[0] + 0.5))
                dH_ddispl[1,0] =  ((t+dt[1])*1j*k - beta)*np.exp(1j*k*cell_length*(displacement[0] - displacement[1] + 0.5))

        return dH_ddispl + np.matrix.getH(dH_ddispl)


    def compute_forces(self, psi_k, **kwargs):
        """
            This function computes the forces acting on each atom of the supercell
            They are computed analitically for the ionic part and with perturbation theory for the electronic part

            Parameters
            ----------
                psi_k : array of shape (N_atoms,N_atoms,N_k) containing complex_
                    Array with the eigenvectors for each band and for each k-point

            Return
            ------
                forces : array of length N_atoms containing floats
                    Array with the forces acting on each atom of the supercell (in eV/Angstrom)
        """

        N_atoms = self.N_atoms
        cell_length = self.cell_length
        supercell_length = self.supercell_length
        psi_k = psi_k
        k_ela = self.k_ela

        # component i is the derivative of electronic energy wrt the displacement of atom i
        force_elec = np.zeros(N_atoms)
        # component i is the derivative of ionic energy wrt the displacement of atom i
        force_ion = np.zeros(N_atoms)

        N_occ = int(N_atoms*0.5)
        N_k = len(psi_k[0,0,:])
        k_min = -np.pi/supercell_length
        k_max = np.pi/supercell_length
        k = np.linspace(k_min,k_max,N_k)

        for alpha in range(N_atoms):
            # ionic contribution
            # derivative of the ionic energy w.r.t. the displacement of atom alpha
            dEion_ddispl_alpha = 2*self.atom[alpha].displacement-self.atom[(alpha+1)%N_atoms].displacement-self.atom[alpha-1].displacement
            force_ion[alpha] = -k_ela*dEion_ddispl_alpha*cell_length
            # electronic contribution
            for i_k in range(N_k):
                H_alpha = self.dH_ddispl(alpha,k[i_k])
                for i_occ in range(N_occ):
                    force_elec[alpha] -= 2*np.real(np.vdot(psi_k[:,i_occ,i_k],H_alpha.dot(psi_k[:,i_occ,i_k])))

        # print("f_elec",force_elec/(N_k*cell_length))
        # print("f_ion", force_ion)

        forces = force_elec/N_k + force_ion

        return forces

    def check_forces(self, f, tol=1e-8):
        """
            Function to check that the sum of the forces is always zero

            Parameters
            ----------
                f : array of floats
                    Array with the forces acting on each atom of the supercell

            Return
            ------
                check : bool
                    If True the sum rule is respected
        """
        check = (np.abs(np.sum(f)) < tol)
        return check


    def d2H_ddispl2(self, alpha, gamma, k):
        """
            Function to calculate the second derivative of the electronic Hamiltonian w.r.t. atomic displacements

            Parameters
            ----------
                alpha : int
                    Index of one of the two atoms with respect to which the derivative is taken
                gamma : int
                    Index of one of the two atoms with respect to which the derivative is taken
                k : float
                    It is the k-point on which we compute the derivatives

            Return
            ------
                d2H_ddispl2 : array with shape (N_atoms,N_atoms) with complex_
                    Second derivative of the electronic Hamiltonian w.r.t displacement of atom alpha and gamma at point k (in eV/Angstrom^2)
        """

        N_atoms = self.N_atoms
        cell_length = self.cell_length
        t = self.hopping
        dt = self.dt
        displacement = self.displacement
        d2H_ddispl2 = np.zeros((N_atoms,N_atoms),dtype=np.complex_)

        if N_atoms!=2:
            if gamma==alpha:
                d2H_ddispl2[alpha-1,alpha] = (t+dt[alpha-1])*k**2*np.exp(1j*k*cell_length*(displacement[alpha] - displacement[alpha-1] + 0.5))
                d2H_ddispl2[alpha,(alpha+1)%N_atoms] = (t+dt[alpha])*k**2*np.exp(1j*k*cell_length*(displacement[(alpha+1)%N_atoms] - displacement[alpha] + 0.5))
            elif gamma==alpha+1:
                d2H_ddispl2[alpha,(alpha+1)%N_atoms] = -(t+dt[alpha])*k**2*np.exp(1j*k*cell_length*(displacement[(alpha+1)%N_atoms] - displacement[alpha] + 0.5))
            elif gamma==alpha-1:
                d2H_ddispl2[alpha-1,alpha] = -(t+dt[alpha-1])*k**2*np.exp(1j*k*cell_length*(displacement[alpha] - displacement[alpha-1] + 0.5))
        else:
            if alpha==0 and gamma==0:
                d2H_ddispl2[0,1] = (t+dt[0])*k**2*np.exp(1j*k*cell_length*(displacement[1] - displacement[0] + 0.5))
                d2H_ddispl2[1,0] = (t+dt[1])*k**2*np.exp(1j*k*cell_length*(displacement[0] - displacement[1] + 0.5))
            elif alpha==0 and gamma==1:
                d2H_ddispl2[0,1] = -(t+dt[0])*k**2*np.exp(1j*k*cell_length*(displacement[1] - displacement[0] + 0.5))
                d2H_ddispl2[1,0] = -(t+dt[1])*k**2*np.exp(1j*k*cell_length*(displacement[0] - displacement[1] + 0.5))
            elif alpha==1 and gamma==1:
                d2H_ddispl2[0,1] = (t+dt[0])*k**2*np.exp(1j*k*cell_length*(displacement[1] - displacement[0] + 0.5))
                d2H_ddispl2[1,0] = (t+dt[1])*k**2*np.exp(1j*k*cell_length*(displacement[0] - displacement[1] + 0.5))

        return d2H_ddispl2 + np.matrix.getH(d2H_ddispl2)


    def compute_force_constants(self, **kwargs):
        """
            Function to compute the force constant matrix of the supercell.
            They are computed as second derivatives of the total energy w.r.t. atomic displacements

            Parameters
            ----------
                eps_k : array of shape (N_atoms,N_k) containing floats
                    Array with the eigenenergy for each band and k-point (in eV)
                psi_k : array of shape (N_atoms,N_atoms,N_k) containing complex_
                    Array with the eigenvectors for each band and for each k-point

            Return
            ------
                force_constants : matrix of floats with shape (N_atoms,N_atoms)
                    Force constant matrix of the supercell (in eV/Angstrom^2)
        """

        N_atoms = self.N_atoms
        cell_length = self.cell_length
        supercell_length = self.supercell_length
        k_ela = self.k_ela
        eps_k = kwargs['eps_k']
        psi_k = kwargs['psi_k']

        N_occ = int(N_atoms*0.5)
        N_k = len(psi_k[0,0,:])
        k_min = -np.pi/supercell_length
        k_max = np.pi/supercell_length
        k = np.linspace(k_min,k_max,N_k)

        # ionic contribute
        d2Eion_d2displ = np.zeros((N_atoms,N_atoms))
        for i in range(N_atoms):
            for j in range(i,N_atoms):
                if j==i:
                    d2Eion_d2displ[i,j] = 2*0.5
                elif j==i+1:
                    d2Eion_d2displ[i,j] = -1
                elif j==N_atoms-1 and i==0:
                    d2Eion_d2displ[i,j] = -1
        d2Eion_d2displ += np.matrix.getT(d2Eion_d2displ)
        d2Eion_d2displ *= k_ela

        # electronic contribute
        d2Eelec_d2displ = np.zeros((N_atoms,N_atoms))
        for i_k in range(N_k):
            for alpha in range(N_atoms):
                dH_dalpha = self.dH_ddispl(alpha,k[i_k])
                for gamma in range(N_atoms):
                    dH_dgamma = self.dH_ddispl(gamma,k[i_k])
                    dH_dalpha_dgamma = self.d2H_ddispl2(alpha,gamma,k[i_k])
                    # print(dH_dalpha_dgamma)
                    for i_occ in range(N_occ):
                        d2Eelec_d2displ[alpha,gamma] += 2*np.real(np.vdot(psi_k[:,i_occ,i_k],dH_dalpha_dgamma.dot(psi_k[:,i_occ,i_k])))
                        for j_emp in range(N_occ,N_atoms):
                            d2Eelec_d2displ[alpha,gamma] += 4*np.real(np.vdot(psi_k[:,i_occ,i_k],dH_dalpha.dot(psi_k[:,j_emp,i_k]))
                                                                      *np.vdot(psi_k[:,j_emp,i_k],dH_dgamma.dot(psi_k[:,i_occ,i_k]))
                                                                      /(eps_k[i_occ,i_k]-eps_k[j_emp,i_k]))

        force_constants = d2Eelec_d2displ/N_k + d2Eion_d2displ

        return force_constants


    def check_force_constants(self, FC, tol=1e-8):
        """
            Function to check if the force constant matrix of the supercell is invariant for discrete translations

            Parameters
            ----------
                FC : matrix of floats with shape (N_atoms,N_atoms)
                    Force constant matrix of the supercell

            Return
            ------
                check : bool
                    If True the force constants matrix of the supercell is invariant for discrete translations
        """

        N_atoms = len(FC[0,:])
        symm = np.all(np.abs(FC-FC.T) < tol)
        transl_inv = np.all(np.array([FC[0,i]-FC[j,(i+j)%N_atoms] for i in range(N_atoms) for j in range(N_atoms)]) < tol )
        check = (symm and transl_inv)

        return check


    def compute_dyn_matrix(self, force_constants, plot=False, **kwargs):
        """
            Function to compute the dynamical matrix of the system

            Parameters
            ----------
                force_constants : matrix of floats with shape (N_atoms,N_atoms)
                    Force constants matrix of the supercell (in eV/Angstrom^2)
                plot : bool
                    If True, the phonon bands are plotted

            Return
            ------
                dyn_matr : matrix of complex_ with shape (atoms_per_cell,atoms_per_cell,N_q)
                    The dynamical matrix for each q-point (in eV/Angstrom^2*Ry)
                omega_q : matrix of complex_ with shape (atoms_per_cell,N_q)
                    Matrix with phonon dispersion for each band (in cm^-1)
                v_q : matrix of complex_ with shape (atoms_per_cell,atoms_per_cell,N_q)
                    Eigenvectors of the dynamical matrix for each phonon band and for each q point
        """

        force_constants = force_constants

        N_cells = self.N_cells
        cell_length = self.cell_length
        N_atoms = self.N_atoms
        atoms_per_cell = self.atoms_per_cell
        mass = self.mass

        # we need N_cells+1 q-points because the first and the last are the same point (they are separated by 2pi/cell_length)
        N_q = N_cells+1
        q_min = 0
        q_max = 2*np.pi/cell_length
        q = np.linspace(q_min,q_max,N_q)
        # if (N_cells % 2) == 0:
        #     q -= np.pi/cell_length

        dyn_matr = np.zeros((atoms_per_cell,atoms_per_cell,N_q),dtype=np.complex_) # dynamical matrix for each q
        omega2_q = np.zeros((atoms_per_cell,N_q),dtype=np.double) # eigenvalues of dynamical matrix
        omega_q = np.zeros_like(omega2_q)
        v_q = np.zeros((atoms_per_cell,atoms_per_cell,N_q),dtype=np.complex_) # eigenvectors of dynamical matrix

        from ase.units import Ry, Bohr

        for i_q in range(N_q):
            for alpha in range(atoms_per_cell):
                for gamma in range(atoms_per_cell):
                    tmp = 0*1j
                    for i in range(N_cells):
                        tmp += (np.exp(1j*q[i_q]*self.atom[i*atoms_per_cell+gamma].cell_label*cell_length)
                                *force_constants[alpha,i*atoms_per_cell+gamma])
                    dyn_matr[alpha,gamma,i_q] = tmp/np.sqrt(mass[alpha]*mass[gamma])
            dyn_matr[:,:,i_q] *= (Bohr**2/Ry) # to convert dynamical matrix in atomic units
            omega2_q[:,i_q], v_q[:,:,i_q] = np.linalg.eigh(dyn_matr[:,:,i_q])

        # eV_to_Hartee = 0.037 # (Hartree/eV)
        # Angstrom_to_Bohr = 1.88973 # (Bohr/A)
        # atomic_units_of_freq_to_THz = 4.13394e+4 # (THz/a.u.)

        from cellconstructor.Units import RY_TO_CM
        omega_q = np.sign(omega2_q)*np.sqrt(np.abs(omega2_q))*RY_TO_CM # frequencies converted to cm^-1

        if plot:
            self.plot_phonon_band(omega_q)

        return dyn_matr, omega_q, v_q


    def check_dyn_matr(self, D, tol=1e-8):
        """
            Function to check if the dynamical matrices are hermitian for each q point

            Parameters
            ----------
                D : matrix of complex_ with shape (atoms_per_cell,atoms_per_cell,N_q)
                    The dynamical matrix for each q-point
            Return
                check : bool
                    If True, the dynamical matrices are hermitian for each q point
        """

        N = len(D[0,0,:]) # number of q-points
        check = True
        for i in range(N):
            check = check and np.all(np.abs(D[:,:,i]-np.matrix.getH(D[:,:,i])) < tol)
        return check


    def plot_phonon_band(self, omega_q, **kwargs):
        """
            Function to plot the phonon dispersion

            Parameters
            ----------
                omega_q : matrix of complex_ with shape (atoms_per_cell,N_q)
                    Matrix with phonon dispersion for each band
        """

        atoms_per_cell = self.atoms_per_cell
        N_cells = self.N_cells
        cell_length = self.cell_length
        N_q = N_cells+1
        q_min = 0
        q_max = 2*np.pi/cell_length
        q = np.linspace(q_min,q_max,N_q)
        # if (N_cells % 2) == 0:
        #     q -= np.pi/cell_length

        fig = plt.figure()
        ax = plt.gca()
        colors = ['b','r']
        for i in range(int(atoms_per_cell*0.5)):
            ax.plot(q,omega_q[i,:],c=colors[0],marker='v',linewidth=2,markersize=12)
            ax.plot(q,omega_q[i+int(atoms_per_cell*0.5),:],c=colors[1],marker='v',linewidth=2,markersize=12)
        ax.set_xlabel('q  ($\AA^{-1}$)')# ,labelpad=15,fontsize=25)
        ax.set_ylabel('omega  (a.u.)')# ,labelpad=20,fontsize=25)


    def dH_dk(self, k):
        """
            Function to compute the derivative of the electronic Hamiltonian w.r.t. the quasi-momentum k

            Parameters
            ----------
                k : float
                    It is the k-point on which we compute the derivatives

            Return
            ------
                dH_dk : matrix of complex_ with shape (N_atoms,N_atoms)
                    Derivative of the electronic Hamiltonian w.r.t. the quasi-momentum
        """

        N_atoms = self.N_atoms
        cell_length = self.cell_length
        t = self.hopping
        dt = self.dt
        displacement = self.displacement

        dH_dk = np.zeros((N_atoms,N_atoms),dtype=np.complex_)

        if N_atoms!=2:
            for i in range(N_atoms):
                for j in range(i,N_atoms):
                    if j==i+1:
                        dH_dk[i,j]=-(t+dt[i])*np.exp(1j*k*cell_length*(displacement[j] - displacement[i] + 0.5))*1j*cell_length*(displacement[j] - displacement[i] + 0.5)
                    elif i==0 and j==(N_atoms-1):
                        dH_dk[i,j]=-(t+dt[j])*np.exp(-1j*k*cell_length*(displacement[i] - displacement[j] + 0.5))*(-1j)*cell_length*(displacement[i] - displacement[j] + 0.5)
        else:
            dH_dk[0,1] = -(t+dt[0])*np.exp(1j*k*cell_length*(displacement[1] - displacement[0] + 0.5))*1j*cell_length*(displacement[1] - displacement[0] + 0.5)
            dH_dk[1,0] = -(t+dt[1])*np.exp(1j*k*cell_length*(displacement[0] - displacement[1] + 0.5))*1j*cell_length*(displacement[0] - displacement[1] + 0.5)

        return dH_dk + np.matrix.getH(dH_dk)


    def Zeff_alpha(self, **kwargs):
        """
            Function to compute the effective charge of a specific atom in the supercell

            Parameters
            ----------
                eps_k : array of shape (N_atoms,N_k) containing floats
                    Array with the eigenenergy for each band and k-point
                psi_k : array of shape (N_atoms,N_atoms,N_k) containing complex_
                    Array with the eigenvectors for each band and for each k-point
                alpha : int
                    Index of the atom whose effective charge we want to compute

            Return
            ------
                Zeff : float
                    effective charge of atom alpha in the supercell
        """

        N_atoms = self.N_atoms
        supercell_length = self.supercell_length
        eps_k = kwargs['eps_k']
        psi_k = kwargs['psi_k']
        alpha = kwargs['alpha']

        N_occ = int(N_atoms*0.5)
        N_k = len(psi_k[0,0,:])
        k_min = -np.pi/supercell_length
        k_max = np.pi/supercell_length
        k = np.linspace(k_min,k_max,N_k)

        Zeff = 1j*0.

        for i_k in range(N_k):
            H_alpha = self.dH_ddispl(alpha,k[i_k])
            H_k = self.dH_dk(k[i_k])
            for i_occ in range(N_occ):
                for j_emp in range(N_occ,N_atoms):
                    Zeff -= 1j*(np.vdot(psi_k[:,i_occ,i_k],H_alpha.dot(psi_k[:,j_emp,i_k]))
                                *np.vdot(psi_k[:,j_emp,i_k],H_k.dot(psi_k[:,i_occ,i_k]))
                                /(eps_k[i_occ,i_k] - eps_k[j_emp,i_k])**2)

        return 4*np.real(Zeff)/N_k


class toy_model():
    def __init__(self, name=None, cell_length=1., N_cells=1., atoms_per_cell=2., onsite=0., hopping=1., mass=1., el_ph=1., k_ela=1., N_k=1):
        """
            This class represents a particular system that can be described by the supercell class, for example carbyne or polyacetylene.
            It can be used to fit the parameters from experiments or other simulations.

            Parameters
            ----------
                name : string
                    Name of the model (e.g. carbyne, polyacetylene,...)
                cell_length : float
                    The length of the unit cell (in Angstrom)
                N_cells : int
                    Number of cells in the supercell
                atoms_per_cell : int
                    Number of atoms per single cell
                onsite : float or array of floats
                    Onsite energy of each atom in the supercell (in eV)
                    If float, an array is created with alternating +onsite -onsite
                hopping : float
                    Hopping energy in the case of equidistant atoms (in eV)
                mass : float or array of floats
                    Mass of each atom in the supercell (in Ry)
                    If float, an array of ones*mass is created
                el_ph : float
                    Electron-phonon coupling parameter (in eV/Angstrom)
                k_ela : float
                    Elastic constant to account for ion-ion harmonic interaction (in eV/Angstrom^2)
                N_k : int
                    Number of kpoints to be used in the electronic calculations
        """

        self.name=name
        self.cell_length = cell_length
        self.N_cells = N_cells
        self.atoms_per_cell = atoms_per_cell
        self.N_atoms = self.N_cells*self.atoms_per_cell # total number of atoms in the supercell
        self.supercell_length = self.N_cells*self.cell_length # length of the supercell
        self.onsite = onsite
        self.hopping = hopping
        self.el_ph = el_ph
        self.k_ela = k_ela
        self.mass = mass
        self.N_k = N_k


    def total_energy(self, displ):
        """
            Function which computes the total energy per unit cell of the 1D chain given a displacement.
            It is used by the scipy.optimize.minimize() method, which takes the free parameters in input as a list

            Parameters
            ----------
                displ : list with only one float
                    A single float is given because one atom every two will be displaced of this quantity

            Return
            ------
                E_tot : float
                    Total energy per unit cell
        """

        this_supercell = supercell(N_cells=self.N_cells, atoms_per_cell=self.atoms_per_cell, cell_length=self.cell_length, displacement=displ[0],
                                    default=False, onsite=self.onsite, hopping=self.hopping, el_ph=self.el_ph, k_ela=self.k_ela, mass=self.mass)

        eps_k, psi_k = this_supercell.compute_elec_bands(self.N_k)

        E_tot = this_supercell.E_tot(k_ela=self.k_ela, eps_k=eps_k)

        return E_tot/self.N_cells


    def BLA(self):
        """
            Function to compute the BLA of the system, given all the other parameters.
            It is obtained by minimising the total energy as a functions of the displacement

            Return
            ------
                BLA : float
                    BLA of the system (in Angstrom)
        """

        result = minimize(self.total_energy, [0.01])
        BLA = 2*self.cell_length*result.x[0]
        # print("Results of minimisation of Etot vs displ", result)
        # print("BLA : ",BLA)

        return BLA


    def omega_LO(self):
        """
            Function to obtain the frequency of the longitudinal optical mode at the Gamma point.

            Return
                omega_LO : float
                    Value of the frequency of the longitudinal optical mode at the Gamma point (in cm^-1)
        """

        displ_min = self.BLA()*0.5/self.cell_length

        this_supercell = supercell(self, N_cells=self.N_cells, atoms_per_cell=self.atoms_per_cell, cell_length=self.cell_length, displacement=displ_min,
                                       default=False, onsite=self.onsite, hopping=self.hopping, el_ph=self.el_ph, k_ela=self.k_ela, mass=self.mass)

        eps_k, psi_k = this_supercell.compute_elec_bands(self.N_k)

        force_constants = this_supercell.compute_force_constants(psi_k=psi_k,eps_k=eps_k)

        dyn_matrix, omega_q, v_q = this_supercell.compute_dyn_matrix(force_constants)

        omega_LO = omega_q[1,0]

        return omega_LO


    def E_gap(self):
        """
            Function to compute the energy gap of the electronic bands. It uses the E_gap() function of the supercell class

            Return
            ------
            E_gap : float
                Energy gap of the electronic bands (in eV)
        """

        displ_min = self.BLA()*0.5/self.cell_length

        this_supercell = supercell(self, N_cells=self.N_cells, atoms_per_cell=self.atoms_per_cell, cell_length=self.cell_length, displacement=displ_min,
                                       default=False, onsite=self.onsite, hopping=self.hopping, el_ph=self.el_ph, k_ela=self.k_ela, mass=self.mass)

        eps_k, psi_k = this_supercell.compute_elec_bands(self.N_k)

        E_gap = this_supercell.E_gap(eps_k)

        return E_gap


    def E_gain(self):
        """
            Function which returns the energy gain, calculated as the depth of the occupied band

            Returns
            -------
                E_gain : float
                    Depth of the occupied band
        """

        displ_min = self.BLA()*0.5/self.cell_length

        this_supercell = supercell(self, N_cells=self.N_cells, atoms_per_cell=self.atoms_per_cell, cell_length=self.cell_length, displacement=displ_min,
                                       default=False, onsite=self.onsite, hopping=self.hopping, el_ph=self.el_ph, k_ela=self.k_ela, mass=self.mass)

        eps_k, psi_k = this_supercell.compute_elec_bands(self.N_k)

        N_occ = int(self.N_atoms*0.5)

        E_gain = np.amax(eps_k[N_occ-1,:]) - np.amin(eps_k[N_occ-1,:])

        return E_gain

    def energy_profile(self, displ_min=-0.5, displ_max=0.5, displ_step=0.05, plot_vs_displ=False, plot_vs_BLA=False):
        """
            Function to obtain the profile of the total energy as a function of the displacement

            Parameters
            ----------
                displ_min : float
                    Minimum value of the displacement, in fractional units
                displ_max : float
                    Maximum value of the displacement, in fractional units
                displ_step : float
                    Increment of the displacement at each iteration, in fractional units
                plot_vs_displ : bool
                    If True, a plot of E_tot vs displacement is done
                plot_vs_BLA : bool
                    If True, a plot of E_tot vs BLA is done

            Return
            ------
                E_tot : list of floats of length (N_displ)
                    List with the values of total energy for each displacement
                DISPLACEMENTS : nparray of floats of length (N_displ)
                    Array with the values of the displacements in fractional units
        """

        N_displ = int((displ_max - displ_min)/displ_step + 1)
        DISPLACEMENTS = np.arange(displ_min, displ_max + displ_step, displ_step)

        E_tot = []

        for displ in DISPLACEMENTS:

            this_supercell = supercell(N_cells=self.N_cells, atoms_per_cell=self.atoms_per_cell, cell_length=self.cell_length, displacement=displ,
                                        default=False, onsite=self.onsite, hopping=self.hopping, el_ph=self.el_ph, k_ela=self.k_ela, mass=self.mass)

            eps_k, psi_k = this_supercell.compute_elec_bands(N_k=self.N_k)

            E_tot.append(this_supercell.E_tot(k_ela=self.k_ela, eps_k=eps_k)/self.N_cells)

        E_tot = np.array(E_tot)/self.N_cells

        if plot_vs_BLA:

            fig = plt.figure()
            ax = plt.gca()

            BLA = 2*self.cell_length*DISPLACEMENTS
            ax.plot(BLA, E_tot, c='r', marker='.', label='$E_{tot}$')

            plt.title("$E_{tot}$ vs BLA", fontsize=20)
            plt.xlabel("BLA  [$\AA$]", fontsize=15)
            plt.ylabel("Energy [eV]", fontsize=15)
            plt.legend(loc='best', fontsize=15)
            plt.tight_layout()

        if plot_vs_displ:

            fig = plt.figure()
            ax = plt.gca()

            ax.plot(DISPLACEMENTS, E_tot, c='r', marker='.', label='$E_{tot}$')
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

            plt.title("$E_{tot}$ vs displacement", fontsize=20)
            plt.xlabel("displacement  [fractional units]", fontsize=15, )
            plt.ylabel("Energy [eV]", fontsize=15)
            plt.legend(loc='best', fontsize=15)
            plt.tight_layout()

        return E_tot, DISPLACEMENTS
