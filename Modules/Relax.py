# -*- coding: utf-8 -*-

"""
This module performs the relax over more
SCHA minimization. It can be both a constant temperature and a 
constant pressure relax. 
"""
import numpy as np
import sscha, sscha.Ensemble, sscha.SchaMinimizer
import sscha.Optimizer
import cellconstructor as CC
import cellconstructor.symmetries

__EPSILON__ = 1e-5

class SSCHA:
    def __init__(self, minimizer, ase_calculator, N_configs, max_pop = 20, save_ensemble = True):
        """
        This module initialize the relaxer. It may perform
        constant volume or pressure relaxation using fully anharmonic potentials.
        
        Parameters
        ----------
            minimizer : SSCHA_Minimizer
                An initialized SCHA minimizer. Note that its ensemble must be initialized
                with the correct dynamical matrix.
            ase_calculator : ase.calculators...
                An initialized ASE calculator. This will be used to compute energies and forces
                for the relaxation of the SSCHA.
            N_configs : int
                The number of configuration to be used for each population
            max_pop: int, optional
                The maximum number of iteration (The code will stop)
            save_ensemble : bool, optional
                If True (default) the ensemble is saved after each energy and forces calculation.
        """
        
        self.minim = minimizer
        self.calc = ase_calculator
        self.N_configs = N_configs
        self.max_pop = max_pop
        
        # If the ensemble must be saved at each iteration.
        # 
        self.save_ensemble = save_ensemble

        self.__cfpre__ = None
        self.__cfpost__ = None
        self.__cfg__ = None
        
    def setup_custom_functions(self, custom_function_pre = None,
                               custom_function_post = None,
                               custom_function_gradient = None):
        """
        This subroutine setup which custom functions should be called during the minimization.
        Look for the SCHA_Minimizer.run() method for other details.
        """
        
        self.__cfpre__ = custom_function_pre
        self.__cfpost__ = custom_function_post
        self.__cfg__ = custom_function_gradient
        
        
    def relax(self, restart_from_ens = False, get_stress = False,
              ensemble_loc = ".", start_pop = 1):
        """
        COSTANT VOLUME RELAX
        ====================
        
        This function performs the costant volume SCHA relaxation, by submitting several populations
        until the minimization converges (or the maximum number of population is reached)
        
        Parameters
        ----------
            restart_from_ens : bool, optional
                If True the ensemble is used to start the first population, without recomputing
                energies and forces. If False (default) the first ensemble is overwritten with
                a new one, and the minimization starts.
            get_stress : bool, optional
                If true the stress tensor is calculated. This may increase the computational
                cost, as it will be computed for each ab-initio configuration (it may be not available
                with some ase calculator)
            ensemble_loc : string
                Where the ensemble of each population is saved on the disk. You can specify None
                if you do not want to save the ensemble (useful to avoid disk I/O for force fields)
            start_pop : int, optional
                The starting index for the population, used only for saving the ensemble and the dynamical 
                matrix.
            
        Returns
        -------
            status : bool
                True if the minimization converged, False if the maximum number of 
                populations has been reached.
        """
        
        pop = start_pop
                
        running = True
        while running:
            # Generate the ensemble
            self.minim.ensemble.dyn_0 = self.minim.dyn.Copy()
            self.minim.ensemble.generate(self.N_configs)
            
            # Compute energies and forces
            self.minim.ensemble.get_energy_forces(self.calc, get_stress)
            
            if ensemble_loc is not None and self.save_ensemble:
                self.minim.ensemble.save_bin(ensemble_loc, pop)
            
            self.minim.population = pop
            self.minim.init()

            self.minim.print_info()
        
            self.minim.run(custom_function_pre = self.__cfpre__,
                           custom_function_post = self.__cfpost__,
                           custom_function_gradient = self.__cfg__)
        
            
            self.minim.finalize()

            # Save the dynamical matrix
            if self.save_ensemble:
                self.minim.dyn.save_qe("dyn_pop%d_" % pop)
        
            # Check if it is converged
            running = not self.minim.is_converged()
            pop += 1
            
            
            if pop > self.max_pop:
                running = False
                
        return self.minim.is_converged()
    
    
    def vc_relax(self, target_press = 0, static_bulk_modulus = "recalc",
                 restart_from_ens = False,
                 ensemble_loc = ".", start_pop = 1, stress_numerical = False):
        """
        VARIABLE CELL RELAX
        ====================
        
        This function performs a variable cell SCHA relaxation at constant pressure,
        It is similar to the relax calculation, but the unit cell is updated according
        to the anharmonic stress tensor at each new population. 
        The cell optimization is performed with the BFGS algorithm. 
        
        NOTE: 
            remember to setup the stress_offset variable of the SCHA_Minimizer,
            because in ab-initio calculation the stress tensor converges porly with the cutoff, 
            but stress tensor differences converges much quicker. Therefore, setup the
            stress tensor difference between a single very high-cutoff calculation and a
            single low-cutoff (the one you use), this difference will be added at the final
            stress tensor to get a better estimation of the true stress.
        
        Parameters
        ----------
            target_press : float, optional
                The target pressure of the minimization (in GPa). The minimization is stopped if the 
                target pressure is the stress tensor is the identity matrix multiplied by the
                target pressure, with a tollerance equal to the stochastic noise. By default 
                it is 0 (ambient pressure)
            static_bulk_modulus : float, or (9x9) or string, optional
                The static bulk modulus, expressed in GPa. It is used to initialize the
                hessian matrix on the BFGS cell relaxation, to guess the volume deformation caused
                by the anharmonic stress tensor in the first steps. By default is 1000 GPa (higher value
                are safer, since they means a lower change in the cell shape).
                It can be also the whole non isotropic matrix. If you specify a string "recalc", then
                it is recomputed using finite differences at each population recal.
            restart_from_ens : bool, optional
                If True the ensemble is used to start the first population, without recomputing
                energies and forces. If False (default) the first ensemble is overwritten with
                a new one, and the minimization starts.
            get_stress : bool, optional
                If true the stress tensor is calculated. This may increase the computational
                cost, as it will be computed for each ab-initio configuration (it may be not available
                with some ase calculator)
            ensemble_loc : string
                Where the ensemble of each population is saved on the disk. You can specify None
                if you do not want to save the ensemble (useful to avoid disk I/O for force fields)
            start_pop : int, optional
                The starting index for the population, used only for saving the ensemble and the dynamical 
                matrix.
            stress_numerical : bool
                If True the stress is computed by finite difference (usefull for calculators that 
                does not support it by default)
            
        Returns
        -------
            status : bool
                True if the minimization converged, False if the maximum number of 
                populations has been reached.
        """
        # Rescale the target pressure in eV / A^3
        target_press_evA3 = target_press / sscha.SchaMinimizer.__evA3_to_GPa__

        # Read the bulk modulus
        kind_minimizer = "SD"
        if type(static_bulk_modulus) == type(""):
            if static_bulk_modulus == "recalc":
                kind_minimizer = "RPSD"
            elif static_bulk_modulus == "none":
                kind_minimizer = "SD"
                static_bulk_modulus = 100
            else:
                raise ValueError("Error, value '%s' not supported for bulk modulus." % static_bulk_modulus)
        elif len(np.shape(static_bulk_modulus)) == 0:
            kind_minimizer = "SD"
        elif len(np.shape(static_bulk_modulus)) == 2:
            kind_minimizer = "PSD"
        else:
            raise ValueError("Error, the given value not supported as a bulk modulus.")
        

        if static_bulk_modulus is not "recalc":
            # Rescale the static bulk modulus in eV / A^3
            static_bulk_modulus /= sscha.SchaMinimizer.__evA3_to_GPa__ 

        # initilaize the cell minimizer
        #BFGS = sscha.Optimizer.BFGS_UC(self.minim.dyn.structure.unit_cell, static_bulk_modulus)
        if kind_minimizer == "SD":
            BFGS = sscha.Optimizer.UC_OPTIMIZER(self.minim.dyn.structure.unit_cell)
            BFGS.alpha = 1 / (3 * static_bulk_modulus * np.linalg.det(self.minim.dyn.structure.unit_cell))
        elif kind_minimizer == "PSD":
            BFGS = sscha.Optimizer.SD_PREC_UC(self.minim.dyn.structure.unit_cell, static_bulk_modulus)


        # Initialize the bulk modulus
        # The gradient (stress) is in eV/A^3, we have the cell in Angstrom so the Hessian must be
        # in eV / A^6

        pop = start_pop
                
        running = True
        while running:
            # Compute the static bulk modulus if required
            if kind_minimizer == "RPSD":
                # Compute the static bulk modulus
                sbm = GetStaticBulkModulus(self.minim.dyn.structure, self.calc)
                print "BM:"
                print sbm
                BFGS = sscha.Optimizer.SD_PREC_UC(self.minim.dyn.structure.unit_cell, sbm)

            # Generate the ensemble
            self.minim.ensemble.dyn_0 = self.minim.dyn.Copy()
            self.minim.ensemble.generate(self.N_configs)
            
            # Compute energies and forces
            self.minim.ensemble.get_energy_forces(self.calc, True, stress_numerical = stress_numerical)
            
            if ensemble_loc is not None and self.save_ensemble:
                self.minim.ensemble.save_bin(ensemble_loc, pop)
            
            self.minim.population = pop
            self.minim.init()
        
            self.minim.run(custom_function_pre = self.__cfpre__,
                           custom_function_post = self.__cfpost__,
                           custom_function_gradient = self.__cfg__)
        
            
            self.minim.finalize()
            
            # Get the stress tensor [ev/A^3]
            stress_tensor, stress_err = self.minim.get_stress_tensor() 
            stress_tensor *= sscha.SchaMinimizer.__RyBohr3_to_evA3__
            stress_err *=  sscha.SchaMinimizer.__RyBohr3_to_evA3__

            # Get the pressure
            Press = np.trace(stress_tensor) / 3
            
            # Get the volume
            Vol = np.linalg.det(self.minim.dyn.structure.unit_cell)
            
            # Get the Gibbs free energy
            gibbs = self.minim.get_free_energy() * sscha.SchaMinimizer.__RyToev__ + target_press_evA3 * Vol - self.minim.eq_energy
            
            
            # Print the enthalpic contribution
            print ""
            print " ENTHALPIC CONTRIBUTION "
            print " ====================== "
            print ""
            print "  P = %.4f GPa    V = %.4f A^3" % (target_press * sscha.SchaMinimizer.__evA3_to_GPa__, Vol)
            print ""
            print "  P V = %.8e eV " % (target_press * Vol)
            print ""
            print " Gibbs Free energy = %.8e eV " % gibbs
            print " (Zero energy = %.8e eV) " % self.minim.eq_energy
            print ""
        
            # Perform the cell step
            cell_gradient = (stress_tensor - np.eye(3, dtype = np.float64) *target_press_evA3)
            
            new_uc = self.minim.dyn.structure.unit_cell.copy()
            BFGS.UpdateCell(new_uc,  cell_gradient)
            
            # Strain the structure preserving the symmetries
            self.minim.dyn.structure.change_unit_cell(new_uc)
            

            print " New unit cell:"
            print " v1 [A] = (%16.8f %16.8f %16.8f)" % (new_uc[0,0], new_uc[0,1], new_uc[0,2])
            print " v2 [A] = (%16.8f %16.8f %16.8f)" % (new_uc[1,0], new_uc[1,1], new_uc[1,2])
            print " v3 [A] = (%16.8f %16.8f %16.8f)" % (new_uc[2,0], new_uc[2,1], new_uc[2,2])
            
            print ""
            print "Check the symmetries in the new cell:"
            qe_sym = CC.symmetries.QE_Symmetry(self.minim.dyn.structure)
            qe_sym.SetupQPoint(verbose = True)

            # Save the dynamical matrix
            self.minim.dyn.save_qe("dyn_pop%d_" % pop)

            # Check if the constant volume calculation is converged
            running1 = not self.minim.is_converged()

            # Check if the cell variation is converged
            running2 = True
            not_zero_mask = stress_err != 0
            if np.max(np.abs(cell_gradient[not_zero_mask]) / stress_err[not_zero_mask]) <= 1:
                running2 = False

            running = running1 or running2

            pop += 1
            
            if pop > self.max_pop:
                running = False
                
        return (not running1) and (not running2)


def GetStaticBulkModulus(structure, ase_calculator, eps = 1e-3):
    """
    GET STATIC BULK MODULUS
    =======================

    This method uses finite differences on the cell to compute
    the static bulk modulus. The cell is strained into several volumes,
    and the stress tensor is computed in orther to obtain the bulk modulus.
    Only the symmmetry relevant terms are computed.

    Parameters
    ----------
        structure : CC.Structure.Structure()
            The structure on which you want to compute the static bulk modulus
        ase_calculator : ase.calculators.calculator.Calculator()
            One of the ase calculators to get the stress tensor in several strained
            cells.
        eps : float
            The strain module

    Results
    -------
        bk_mod : ndarray (9x9)
            The bulk modulus as a 9x9 matrix, expressed in eV / A^3
    """

    # Initialize the symmetries
    qe_sym = CC.symmetries.QE_Symmetry(structure)

    # Perform the firts calculation
    atm_center = structure.get_ase_atoms()
    atm_center.set_calculator(ase_calculator)
    V_0 = np.linalg.det(structure.unit_cell)
    stress_0 = atm_center.get_stress(False)
    I = np.eye(3, dtype = np.float64)
    

    qe_sym.ApplySymmetryToMatrix(stress_0)

    bk_mod = np.zeros( (9,9), dtype = np.float64)
    # Get the non zero elements
    for i in range(3):
        for j in range(i, 3):
            trials = np.zeros((3,3), dtype = np.float64)
            trials[i,j] = 1
            qe_sym.ApplySymmetryToMatrix(trials)


            if trials[i,j] == 0:
                continue


            mask = (np.abs(trials) < __EPSILON__).astype(np.float64)

            strain = eps * mask
            
            dstrain = eps * np.sqrt(np.sum(mask))

            # Strain the cell and perform the calculation
            new_structure = structure.copy()
            new_structure.change_unit_cell(structure.unit_cell.dot(I + strain))

            atm = new_structure.get_ase_atoms()
            atm.set_calculator(ase_calculator)
            stress_1 = atm.get_stress(False)
            V_1 = np.linalg.det(new_structure.unit_cell)


            qe_sym.ApplySymmetryToMatrix(stress_1)

            bk_mod[3*i + j, :] = -(stress_1 - stress_0).ravel() / dstrain
            if j!= i:
                bk_mod[3*j + i, :] = -(stress_1 - stress_0).ravel() / dstrain
            
            # Apply hermitianity
            bk_mod = 0.5 * (bk_mod.transpose() + bk_mod)

    return bk_mod




