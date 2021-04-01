from __future__ import print_function
from __future__ import division

import sscha.DynamicalLanczos

import sys, os
import time
import warnings, difflib
import numpy as np

from timeit import default_timer as timer

# Import the scipy sparse modules
import scipy, scipy.sparse.linalg

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

import sscha.Ensemble as Ensemble
import sscha.Tools
import sscha_HP_odd

# Override the print function to print in parallel only from the master
import cellconstructor.Settings as Parallel
from sscha.Parallel import pprint as print




class StaticHessian(object):
    def __init__(self, ensemble = None, verbose = False):
        """
        STATIC HESSIAN
        ==============

        This class is for the advanced computation of the static hessian matrix.
        This exploit the inversion of the auxiliary systems, which allows for including the
        fourth order contribution exploiting sparse linear algebra to speedup the calculation.

        You can either initialize directly the object passing the ensemble with the configurations,
        or call the init function after the object has been defined.
        """


        # The minimization variables
        self.Ginv = None 
        self.W = None
        self.lanczos = None
        self.step = 0
        self.verbose = False

        if ensemble is not None:
            self.init(ensemble, verbose)

        # Setup the attribute control
        # Every attribute introduce after this line will raise an exception
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        self.fixed_attributes = True # This must be the last attribute to be setted


    def __setattr__(self, name, value):
        """
        This method is used to set an attribute.
        It will raise an exception if the attribute does not exists (with a suggestion of similar entries)
        """
        
        if "fixed_attributes" in self.__dict__:
            if name in self.__total_attributes__:
                super(StaticHessian, self).__setattr__(name, value)
            elif self.fixed_attributes:
                similar_objects = str( difflib.get_close_matches(name, self.__total_attributes__))
                ERROR_MSG = """
        Error, the attribute '{}' is not a member of '{}'.
        Suggested similar attributes: {} ?
        """.format(name, type(self).__name__,  similar_objects)

                raise AttributeError(ERROR_MSG)
        else:
            super(StaticHessian, self).__setattr__(name, value)


    def init(self, ensemble, verbose = True):
        """
        Initialize the StaticHessian with a given ensemble

        Parameters
        ----------
            ensemble : sscha.Ensemble.Ensemble
                The object that contains the configurations
            verbose : bool
                If true prints the memory occupied for the calculation
        """

        self.lanczos = sscha.DynamicalLanczos.Lanczos(ensemble)

        self.Ginv = np.zeros( (self.lanczos.n_modes, self.lanczos.n_modes), dtype = sscha.DynamicalLanczos.TYPE_DP)
        self.W = np.zeros( (self.lanczos.n_modes, self.lanczos.n_modes*self.lanczos.n_modes), dtype = sscha.DynamicalLanczos.TYPE_DP)

        # Initialize Ginv with the initial guess (the SSCHA matrix)
        self.Ginv[:,:] = np.diag(1 / self.lanczos.w**2)
        self.verbose = verbose


        if verbose:
            print("Memory of StaticHessian initialized.")
            # The seven comes from all the auxiliary varialbes necessary in the gradient computation and the CG
            print("     memory requested: {} Gb of RAM per process".format((self.Ginv.nbytes + self.W.nbytes) * 7 / 1024**3))
            print("     (excluding memory occupied to store the ensemble)")
        

    def run(self, n_steps, save_dir = None, threshold = 1e-8):
        """
        RUN THE HESSIAN MATRIX CALCULATION
        ==================================

        This subroutine runs the algorithm that computes the hessian matrix.

        After this subroutine finished, the result are stored in the
        self.Ginv and selfW.W variables.
        You can retrive the Hessian matrix as a CC.Phonons.Phonons object
        with the retrive_hessian() subroutine.

        The algorithm is a generalized conjugate gradient minimization
        as the minimum residual algorithm, to optimize also non positive definite hessians.

        Parameters
        ----------
            n_steps : int
                Number of steps to converge the calculation
            save_dir : string
                Path to the directory in which the results are saved.
                Each step the status of the algorithm is saved and can be restored.
                TODO
            thr : np.double
                Threshold for the convergence of the algorithm. 
                If the gradient is lower than this threshold, the algorithm is 
                converged.
        """

        # Prepare all the variable for the minimization
        pG = np.zeros(self.Ginv.shape, dtype = sscha.DynamicalLanczos.TYPE_DP)
        pG_bar = np.zeros(self.Ginv.shape, dtype = sscha.DynamicalLanczos.TYPE_DP)

        pW = np.zeros(self.W.shape, dtype = sscha.DynamicalLanczos.TYPE_DP)
        pW_bar = np.zeros(self.W.shape, dtype = sscha.DynamicalLanczos.TYPE_DP)


        # Perform the first application
        rG, rW = self.get_gradient(self.Ginv, self.W)
        rG_bar, rW_bar = self.apply_L(rG, rW)

        # Setup the initial
        pG[:,:] = rG
        pG_bar[:,:] = rG_bar[:,:]

        pW[:,:] = rW
        pW_bar[:,:] = rW_bar

        while self.step < n_steps:
            if self.verbose:
                print("Hessian calculation step {} / {}".format(self.step + 1, n_steps))
            
            ApG = pG_bar
            ApW = pW_bar

            ApG_bar, ApW_bar = self.apply_L(pG_bar, pW_bar)

            rbar_dot_r = np.einsum("ab, ab -> a", rG_bar, rG) + np.einsum("ab, ab ->a", rW_bar, rW)

            alpha = rbar_dot_r
            alpha /= np.einsum("ab, ab ->a", pG_bar, ApG) + np.einsum("ab, ab ->a", pW_bar, ApW)

            # Update the solution
            self.Ginv[:,:] += np.einsum("a, ab ->ab", alpha, pG)
            self.W[:,:] += np.einsum("a, ab ->ab", alpha, pW)

            # Update r and r_bar
            rG[:,:] -= np.einsum("a, ab -> ab", alpha, ApG)
            rW[:,:] -= np.einsum("a, ab -> ab", alpha, ApW)

            rG_bar[:,:] -= np.einsum("a, ab -> ab", alpha, ApG_bar)
            rW_bar[:,:] -= np.einsum("a, ab -> ab", alpha, ApW_bar)

            rbar_dot_r_new = np.einsum("ab, ab -> a", rG_bar, rG) + np.einsum("ab, ab ->a", rW_bar, rW)
            beta = rbar_dot_r / rbar_dot_r_new

            # Update p and p_bar
            pG[:,:] = rG[:,:] + np.einsum("a, ab -> ab", beta, pG)
            pW[:,:] = rW[:,:] + np.einsum("a, ab -> ab", beta, pW)
            pG_bar[:,:] = rG_bar[:,:] + np.einsum("a, ab -> ab", beta, pG_bar)
            pW_bar[:,:] = rW_bar[:,:] + np.einsum("a, ab -> ab", beta, pW_bar)

            self.step += 1

            # Check the residual
            thr = np.max(np.abs(rG))
            if self.verbose:
                print("   residual = {} (The threshold is {})".format(thr, threshold))
            if thr < threshold:
                if self.verbose:
                    print()
                    print("CONVERGED!")
                break

    def retrive_hessian(self):
        """
        Return the Hessian matrix as a CC.Phonons.Phonons object.

        Note that you need to run the Hessian calculation (run method), otherwise this
        method returns the SSCHA dynamical matrix.
        """

        G = np.linalg.inv(self.Ginv)
        dyn = self.lanczos.pols.dot(G.dot(self.lanczos.pols.T))
        dyn[:,:] *= np.outer(np.sqrt(self.lanczos.m, self.lanczos.m))

        # We copy the q points from the SSCHA dyn
        q_points = np.array(self.lanczos.dyn.q_tot)
        uc_structure = self.lanczos.dyn.structure.copy()
        ss_structure = self.lanczos.dyn.structure.generate_supercell(self.lanczos.dyn.GetSupercell())

        # Compute the dynamical matrix
        dynq = CC.Phonons.GetDynQFromFCSupercell(dyn, q_points, uc_structure, ss_structure)

        # Create the CellConstructor Object.
        hessian_matrix = self.lanczos.dyn.Copy()
        for i in range(q_points.shape[0]):
            hessian_matrix.dynmats[i] = dynq[i, :, :]

        return hessian_matrix


    def apply_L(self, Ginv, W):
        """
        Apply the system matrix to the full array.
        """

        
        if self.verbose:
            t1 = time.time()
            print("Applying the L matrix (this takes some time...)")

        lenv = self.lanczos.n_modes
        lenv += (self.lanczos.n_modes * (self.lanczos.n_modes + 1)) // 2

        Ginv_out = np.zeros(self.Ginv.shape, dtype = sscha.DynamicalLanczos.TYPE_DP)
        W_out = np.zeros(self.W.shape, dtype = sscha.DynamicalLanczos.TYPE_DP)

        for i in range(self.lanczos.n_modes):
            if self.verbose:
                print("Applying vector {} / {}".format(i +1, self.lanczos.n_modes))
            
            vector = np.zeros(lenv, dtype = sscha.DynamicalLanczos.TYPE_DP)
            vector[:self.lanczos.n_modes] = Ginv[i, :]
            vector[self.lanczos.n_modes:] = W[i, :]

            # Here the L application (TODO: Here eventual preconditioning)
            self.lanczos.psi = vector
            outv = self.lanczos.apply_L1_static(vector)
            outv += self.lanczos.apply_anharmonic_static()

            Ginv_out[i, :] = outv[:self.lanczos.n_modes]
            W_out[i, :] = outv[self.lanczos.n_modes:]

        if self.verbose:
            t2 = time.time()
            print("Total time to apply the L matrix: {} s".format(t2- t1))
            print("Enforcing symmetries...")


        # Enforce the permutation symmetry
        Ginv_out = 0.5 * ( Ginv_out + Ginv_out.T )
        W_aux = np.reshape(W_out, (self.lanczos.n_modes, self.lanczos.n_modes, self.lanczos.n_modes))
        W_aux += np.einsum("abc -> acb", W_aux) 
        W_aux += np.einsum("abc -> bac", W_aux)
        W_aux += np.einsum("abc -> bca", W_aux)
        W_aux += np.einsum("abc -> cab", W_aux)
        W_aux += np.einsum("abc -> cba", W_aux)
        W_aux /= 6

        # TODO: enforce the space group symmetries
        W_out[:,:] = np.reshape(W_aux, (self.lanczos.n_modes, self.lanczos.n_modes * self.lanczos.n_modes))

        if self.verbose:
            t3 = time.time()
            print("Total time to enforce the symmetries: {} s".format(t3-t2))
        
        return Ginv_out, W_out

    def get_gradient(self, Ginv, W):
        """
        Compute the gradient of the function to be minimzied.
        """

        Gout, Wout = self.apply_L(Ginv, W)
        Gout[:,:] -= np.eye(self.lanczos.n_modes)
            
        return Gout, Wout







