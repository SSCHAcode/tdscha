# -*- coding: utf-8 -*-

from __future__ import print_function
"""
This is part of the program python-sscha
Copyright (C) 2018  Lorenzo Monacelli

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>. 
"""


"""
This module contains usefull subroutines to work with
"""
import cellconstructor as CC
import sscha.Parallel
from sscha.Parallel import pprint as print
import scipy, scipy.linalg, scipy.optimize
import numpy as np
import sys, os
import time




def restarted_full_orthogonalization_method(A, b, x0, precond = None, verbose = True, max_iters = 100, conv_thr = 1e-6, callback = None, krylov_dimension = 10):
    """
    Restarted Full Orthogonalization Method
    ---------------------------------------

    This is a projection method on the Krylov subspace to solve a linear symmetric system.

    .. math::

        Ax = b

    
    You must provide the function that performs the matrix multiplication :math:`Ax`, the
    starting vector :math:`x_0`, and the known vector :math:`b`.

    Parameters
    ----------
        A : scipy.sparse.linalg.LinearOperator
            The A matrix that takes the vector (numpy array as x0) as input and returns
            the :math:`Ax` operation.
        b : ndarray
            The known term of the system
        x0 : ndarray
            The starting guess.
        precond : scipy.sparse.linalg.LinearOperator or None
            If it is a LinearOperator, it is used to precondition the minimization.
            This is a raw estimate of :math:`A^{-1}`
        verbose : bool
            If true, print the residual after each iteration
        max_iters : int
            The maximum number of iterations.
        callback : pointer to function 
            This is a callback function that is called each time passing the curret solution
            The function must accept two arguments: the current solution x and the number of iterations
        krylov_dimension : int
            The dimension of the krylov subspace before the algorithm is restarted.
            Big values reduce the number of iterations, but you can run in the lack of orthogonality issue.
    """

    if verbose:
        print()
        print("Restarted full orthogonalization method")
        print("--------------------------")
        print()
        print("Initialization...")
    x_start = x0.copy()
    x_new = x_start.copy()

    iters = 1
    converged = False

    basis = np.zeros( (krylov_dimension + 1, len(x0)), dtype = x0.dtype)
    A_proj = np.zeros( (krylov_dimension + 1, krylov_dimension), dtype = x0.dtype)

    residual = -1
    while iters < max_iters:
        # Compute the residual

        if verbose:
            print(" FOM algorithm, iteration {}, residual = {:.16f}".format(iters, residual))
        

        i_new = iters % krylov_dimension

        # Restarting procedure
        if i_new == 1:
            r1 = b - A.matvec(x_new) 
            residual = np.sqrt(r1.dot(r1))

            # Restart 
            basis[:,:] = 0.0
            A_proj[:,:] = 0.0

            # Fill the first vector
            basis[0, :] = r1 / residual

        # Compute the krilov subspace (next iteration)
        krylov_subspace(A, A_proj, basis, i_new - 1, verbose)

        # GMRES
        def cost_function(yv):
            vect = np.zeros((i_new +1), dtype = x0.dtype)
            print("Y: {} | vect: {} | I: {}".format(yv.shape, vect.shape, i_new))
            vect[0] = residual 
            vect -= A_proj[:i_new + 1, :i_new].dot(yv)
            return np.sqrt(vect.dot(vect))

        # Optimizing the residual
        min_res = scipy.optimize.minimize(cost_function, np.zeros((i_new) , dtype = np.double))
        y_out = min_res.x
        res = np.sqrt(y_out.dot(y_out))

        if res < conv_thr:
            converged = True
            break
        y = y_out.dot(basis[:i_new, :])




        # Invert A in the krilov subspace and update the solution
        #r1_proj = basis.dot(r1)
        #y_proj = np.linalg.inv(A_proj).dot(r1_proj)
        #y = y_proj.dot(basis)
        x_new = x_start + y

        if callback is not None:
            callback(x_new, iters)
        
        iters += 1


    return x_new


def krylov_subspace(A, A_projected, basis, iteration, verbose = True, threshold = 1e-12):
    """
    Perform an iteration to get the Krylov subspace
    Get the Krilov subspace and the A matrix inside the space.

    Parameters
    ----------
        A : scipy.sparse.linalg.LinearOperator
            The sparse linear operator on which you want to compute the operation
        vector : ndarray
            The starting vector to initialize the Krylov subspace
        A_projection : ndarray
            The A matrix projected in the Krylov subspace.
            This method will fill the next row (specified by iteration + 1)
        basis: ndarray
            The basis of the Krylov subspace. 
            This method will fill the next row (specified by iteration + 1)
        iteration : int
            The iteration of the Krylov subspace. 
            Specify which vector is the next one in the basis.
        verbose : bool
            If true prints the status of the iteration
        threshold : float
            If the new vector is contained in the previous old vectors, terminate the iteration.

    Results
    -------
        A_projected : ndarray size = (dimension, dimension)
            The A matrix projected in the krylov subspace.
        basis : ndarray size = (dimension, vector.shape)
            The basis of the Krylov subspace
    """
    i = iteration
    dimension = A_projected.shape[0]
    old_v = basis[i, :].copy()

    if i >= dimension:
        raise ValueError("Error, the iteration must be lower than the dimension of the projected basis and matrix.")


    new_v = A.matvec(basis[i, :])
    print("Element:", new_v.dot(basis[i, :]))
    print("A dot |e_{:d}> = {}".format(i, new_v))
    print("|e_{:d}> = {}".format(i, basis[i, :]))

    for k in range(i + 1):
        element = new_v.dot(basis[k, :])
        print("Filling elements {} {} with : {}".format(i, k, element))
        A_projected[i,k] = element
        A_projected[k,i] = A_projected[i, k]

    # Normalize the vector
    norm = np.sqrt(new_v.dot(new_v))
    new_v /= norm

    # Gram-Shmidt
    converged = False
    for double in range(2):
        for k in range(i+1):
            component =  new_v.dot(basis[k, :])
            new_v -= component * basis[k, :]

            print("ORHTO {} | V_{} component along {}: {}".format(double, i, k, component))

        print("Scalar products after {} orthogonalization:".format(double))
        print(new_v.dot(basis.T))

        norm = np.sqrt(new_v.dot(new_v))
        print("Residual norm: {}".format(norm))

        if double == 0:
            A_projected[i+1, i] = norm
        
        # check primitive convergence
        if norm < threshold:
            print("Linear dependency: NORM:", norm)
            converged = True
            break
    
            
    
        new_v /= norm 

    if converged:
        return

    basis[i+1, :] = new_v 
    old_v[:] = new_v

    print("A matrix:")
    print(A_projected)
    print("Scalar product:")
    print(basis.dot(basis.T))
    print("Eigenvalues:")
    print(np.linalg.eigvalsh(A_projected[:i, :i]))

    # print()
    # print("A new:")
    # M = np.zeros(A_projected.shape, dtype = np.double)
    # for i in range(dimension):
    #     for j in range(dimension):
    #         M[i, j] = basis[i, :].dot(A.matvec(basis[j, :]))
    # print(M)

    # Symmetrize A
    #A_projected += A_projected.T
    #A_projected /= 2
    





def minimum_residual_algorithm(A, b, x0, precond = None, verbose = True, max_iters = 100, conv_thr = 1e-6, callback = None):
    """
    Minimum residual algorithm
    --------------------------

    This implement an algorithm to solve the linear system
    
    .. math::

        Ax = b

    You must provide the function that performs the matrix multiplication :math:`Ax`, the
    starting vector :math:`x_0`, and the known vector :math:`b`.

    Parameters
    ----------
        A : scipy.sparse.linalg.LinearOperator
            The A matrix that takes the vector (numpy array as x0) as input and returns
            the :math:`Ax` operation.
        b : ndarray
            The known term of the system
        x0 : ndarray
            The starting guess.
        precond : scipy.sparse.linalg.LinearOperator or None
            If it is a LinearOperator, it is used to precondition the minimization.
            This is a raw estimate of :math:`A^{-1}`
        verbose : bool
            If true, print the residual after each iteration
        max_iters : int
            The maximum number of iterations.
        callback : pointer to function 
            This is a callback function that is called each time passing the curret solution
            The function must accept two arguments: the current solution x and the number of iterations

    Results
    -------
        x : ndarray
            The solution of the inversion A^-1 b = x obtained in the last step
    """

    # Setup the starting condition
    # Compute the residual
    if verbose:
        print()
        print("Minimal residual algorithm")
        print("--------------------------")
        print()
        print("Initialization...")
    r1 = b - A.matvec(x0) 
    if precond:
        r1[:] = precond.matvec(r1)

    b_prec = b.copy()
    if precond:
        b_prec = precond.matvec(b)
    
    if verbose:
        print("Starting residual: {}".format(np.sqrt(r1.dot(r1))))

    r1_bar = A.matvec(r1)
    if precond:
        r1_bar = precond.matvec(r1_bar)

    # Get the zbar
    z1 = r1.copy()
    z1_bar = r1_bar.copy()

    if False:# precond:
        z1[:] = precond.matvec(r1)
        z1_bar[:] = precond.matvec(r1_bar)

    p1 = r1.copy()
    p1_bar = r1_bar.copy()
    x = x0.copy()

    iters = 1
    converged = False
    while iters < max_iters and not converged:
        t1 = time.time()
        if verbose:
            print("Iteration: {}".format(iters))
        
        Ap = p1_bar #A.matvec(p1)
        Ap_bar = A.matvec(p1_bar)
        if precond:
            Ap_bar[:] = precond.dot(Ap_bar)

        alpha = r1_bar.dot(z1) / (p1_bar.dot(Ap))
        x[:] += alpha * p1 # Update the solution

        if callback is not None:
            callback(x, iters)

        r = r1 - alpha * Ap
        rbar = r1_bar -alpha * Ap_bar


        r_norm = np.sqrt(r.dot(r)) / np.sqrt(b_prec.dot(b_prec))
        # if not precond:
        #     pass
        # else:
        #     num = precond.matvec(r)
        #     den = precond.matvec(b)
        #     r_norm = np.sqrt(num.dot(num)) / np.sqrt(den.dot(den))

        t2 = time.time()
        if verbose:
            print("    residual = {}".format(r_norm))
            print("    time to perform the step: {} s".format(t2-t1))
            if sscha.Parallel.am_i_the_master():
                sys.stdout.flush()

            # print("Check:")
            # print("r mine = {}".format(r))
            # print("r with x = {}".format(A.matvec(x) - b))

        z = r.copy()
        zbar = rbar.copy()
        
        if False:
            z[:] = precond.matvec(r)
            zbar[:] = precond.matvec(rbar)

        # if verbose:
        #     print("Check:")
        #     print("r = {}".format(r))
        #     print("z = {}".format(z))


        beta = rbar.dot(z) / (r1_bar.dot(z1))

        p1[:] = z + beta * p1
        p1_bar[:] = zbar + beta* p1_bar

        r1[:] = r
        r1_bar[:] = rbar
        z1[:] = z 
        z1_bar[:] = zbar

        # check convergency
        converged = r_norm < conv_thr
        iters += 1

    if not converged:
        print("WARNING: cg did not converge after {} iterations.".format(max_iters))
        print("         residual = {}".format(r_norm))

    return x




def minimum_residual_algorithm_precond(A, b, precond_half, **kwargs):
    """
    Minimum residual algorithm
    --------------------------

    This implement an algorithm to solve the linear system
    
    .. math::

        Ax = b

    You must provide the function that performs the matrix multiplication :math:`Ax`, the
    starting vector :math:`x_0`, and the known vector :math:`b`.

    Parameters
    ----------
        A : scipy.sparse.linalg.LinearOperator
            The A matrix that takes the vector (numpy array as x0) as input and returns
            the :math:`Ax` operation.
        b : ndarray
            The known term of the system
        x0 : ndarray
            The starting guess.
        precond_half : scipy.sparse.linalg.LinearOperator or None
            If it is a LinearOperator, it is used to precondition the minimization.
            This is a raw estimate of :math:`A^{-1/2}`
        **kwargs : The same arguments as minimum_residual_algorithm
    """

    # Setup the starting condition

    # This is the standard gradient, but now we rescale all the quantities.
    def apply_new_A(x):
        # M^-1/2 A M^-1/2
        x1 = precond_half.matvec(x)
        x2 = A.matvec(x1)
        return precond_half.matvec(x2)

    if precond_half is not None:
        A_tilde = scipy.sparse.linalg.LinearOperator(A.shape, matvec = apply_new_A)

        b_tilde = precond_half.matvec(b)

        if not "x0" in kwargs:
            kwargs["x0"] = b_tilde.copy()
            

        x_tilde = minimum_residual_algorithm(A_tilde, b_tilde, precond = None, **kwargs)

        # Convert back in the real system
        x = precond_half.matvec(x_tilde)
    else:
        x = minimum_residual_algorithm(A, b, b.copy(), precond = None, **kwargs)

    return x


    return x


def get_matrix_from_sparse_linop(A):
    """
    Transform the linear operator A into a numpy matrix.
    This is used for debugging purpouses.
    """

    n, m = A.shape
    L = np.zeros((n,m), dtype=A.dtype)
    for i in range(m):
        v1 = np.zeros(m, dtype = A.dtype)
        v1[i] = 1
        L[:, i] = A.dot(v1) 

    return L




# Here some usefull functions to solve linear systems
def BiconjugateVector(R, Rbar, P, Pbar, Xold, ApplyMatrix, ApplyTranspose):
    """
    BICONJUGATE STEP
    ================

    This is the single step of the biconjugate algorithm for finding
    the solution of a linear system.

    Parameters
    ----------
        R, Rbar, P, Pbar : vectors
            The parameters of the Bicojugate, they will be updated at each step.
        Xold : vector
            The guess of the system solution, will be updated
        ApplyMatrix: function of vector
            Takes in input the vector, and computes A*x 
        ApplyTranspose: function of the vector
            Takes in input the vector and computes A.T * x
    """

    Ap = ApplyMatrix(P)
    Apbar = ApplyTranspose(Pbar)

    # Get the alpha step
    alpha = Rbar.dot(R) / (Pbar.dot(Ap))

    # Get the rest
    newR = R - alpha * Ap
    newRbar = Rbar - alpha * Apbar

    # Get the beta step
    beta = newRbar.dot(newR) / Rbar.dot(R)
    newP = newR + beta *P
    newPbar = newRbar + beta*Pbar

    # Update all
    Xold += alpha*P
    R[:] = newR
    Rbar[:] = newRbar
    P[:] = newP
    Pbar[:] = newPbar


# ------------------ GENERATORS ---------------------------
# Here we work with the old sscha generators, to enable them
# to represent any given matrix

class Generators:
    """
    This is a class that allows for the interface with the old
    Fortran sscha code, that is interely based on the generators.
    """
    
    def __init__(self):
        """
        Initialize the generator class
        """
        
        # The number of q points represented by the generators
        self.nq = 0
        self.nat = 0
        self.wyck_gen = None
        self.wyck_ncoeff = 0
        
        self.dyn_ncoeff = []
        self.dyn_gen = []
        
    def LoadFromFileWyck(self, filename, natoms):
        """
        LOAD THE WYCKOFF GENERATORS FROM FILE
        =====================================
        
        This subroutine initializes and loads the wyckoff
        generator from the given filename
        
        Parameters
        ----------
            filename : string
                Path to the file that contains the generators
            natoms : int
                The number of atoms in the structure
        
        """
        # Check if the file exists
        if not os.path.exists(filename):
            raise IOError("Error while loading %s, file not found." % filename)
        
        
        # Load the file
        f = open(filename, "r")
        flines = [l.strip() for l in f.readlines()]
        f.close()
        
        
        ngen = 0
        current_i = 0
        current_at = 0
        for i, line in enumerate(flines):
            # Get the number of generators
            if i == 0:
                ngen = int(flines[0])
                self.wyck_gen = np.zeros((ngen, natoms, 3), dtype = np.float64)
                continue
            
            # Get the value of the generators
            l_number = np.array([np.float64(x) for x in line.split()])
            
            # Setup the generators
            self.wyck_gen[current_i, current_at, :] = l_number
            
            current_at += 1
            
            # Check if the atoms are endend
            if current_at == natoms:
                current_at = 0
                current_i += 1
        
        # Check if the number of line read matched the correct one
        if current_i != ngen or current_at != 0:
            raise IOError("Error, the specified file %s do not match with the given number of atoms" % filename)
                
        self.wyck_ncoeff = ngen
        
    
    def LoadFromFileFC(self, filename, natoms, nqirr):
        """
        This subroutine loads the list of generators from a file.
        The FC means that the generators are expected to be on the force constant
        matrix, not the wyckoff positions.
        The generators are written in the q space
        
        
        Parameters
        ----------
            filename : string
                The path to the file that contains all the generators
            natoms : int
                The number of atoms in the current structure (unit_cell)
            nqirr : int
                The number of irreducible q points
        """
        
        # Check if the file exists
        if not os.path.exists(filename):
            raise IOError("Error while loading %s, file not found." % filename)
            
            
        f = open(filename, "r")
        flines = [l.strip() for l in f.readlines()]
        
        self.dyn_ncoeff = []
        self.nat = natoms
        self.dyn_gen = np.zeros( (nqirr, 3*natoms*3*natoms, 3*natoms, 3*natoms), 
                                dtype = np.complex128)
        # Read how many generator are for this particular q point
        n_gen = int(flines[0])
        current_i = 0
        
        ghrs = np.zeros( (3*natoms*3*natoms, 3*natoms, 3*natoms), dtype = np.complex128)
        fc = np.zeros( (3*natoms, 3*natoms), dtype = np.complex128) 
        new_gen = False
        
        na = 0
        nb = 0
        index  = 0
        iq = 0
        
        #print "NGEN:", n_gen
        for i, line in enumerate(flines):
            # Skip the first line
            if i == 0:
                continue
            
            #print current_i, line
            if new_gen:
                # Append the generator
                ghrs[current_i, :, :] = fc.copy()
                
                if current_i+1 == n_gen:
                    #print "NEW GEN LINE:", line
                    
                    self.dyn_ncoeff.append(n_gen)
                    n_gen = int(line)
                    current_i = -1
                    self.dyn_gen[iq, :,:,:] =ghrs.copy()
                    iq += 1
                    continue
            
                
                fc = np.zeros( (3*natoms, 3*natoms), dtype = np.complex128)
                new_gen = False
                current_i += 1
            
            # Polish the line
            line = line.replace(",", " ")
            line = line.replace("(", " ")
            line = line.replace(")", " ")
            
            line_list = [np.float64(x) for x in line.split()]
            
            # Select the atomic indexs
            if len(line_list) == 2:
                na = int(line_list[0] -1)
                nb = int(line_list[1] -1)
                index = 0
                continue
#            
#            if na == 0 and nb ==0 and current_i == 0:
#                print na, nb, index, line, line_list
            
            #print "INDEX:", na, nb, index, "GEN:", iq, current_i
            
            fc[3 * na + index, 3 * nb] = line_list[0] + 1j*line_list[1]
            fc[3 * na + index, 3 * nb + 1] = line_list[2] + 1j*line_list[3]
            fc[3 * na + index, 3 * nb + 2] = line_list[4] + 1j*line_list[5]
            
            index += 1
            
            if (na+1  == natoms ) and (nb+1 == natoms) and index == 3:
                #print na, nb, index, natoms, "NEW GEN", current_i
                new_gen = True
            
                
        # Append also the last generators
        ghrs[current_i, :, :] = fc.copy()
        self.dyn_gen[iq, :, :, :] =  ghrs.copy()
        self.dyn_ncoeff.append(n_gen)
        
        
    def ProjectWyck(self, coords):
        """
        PROJECT THE COORDINATES IN THE WYCKOFF GENERATORS
        =================================================
        
        The following method project the given atomic displacement into
        the generators of the wyckoff positions.
        
        Parameters
        ----------
            coords : n_at x 3
                The coordinates of the atomic displacement to be projected
        
        Results
        -------
            coeffs : ndarray
                The coefficients of the wyckoff positions
        """
        # Check the coords shape
        s = np.shape(coords)
        if len(s) != 2:
            raise ValueError("The given array must be 2 dimensional")
        
        nat = np.shape(self.wyck_gen)[1]
        if s[0] != nat:
            raise ValueError("The number of atoms in the coords does not match the one in the wyckoff positions")

        if s[1] != 3:
            raise ValueError("Error, the vectors must be 3d-cartesian for each atom")
            
        return np.einsum("ijk, jk", self.wyck_gen, coords)
    
    def GenWyck(self, coeffs):
        """
        GENERATE A WYCKOFF DISPLACEMENT
        ===============================
        
        Generate the wyckoff displacement from the coefficients of the generator.
        
        Parameters
        ----------
            coeffs : ndarray
                The coefficients of the wyckoff generators
        
        Returns
        -------
            coords : nat x 3
                The 2d array containing the cartesian displacement for each atom.
        """
        
        # Check the length of the coeffs
        if len(coeffs) != self.wyck_ncoeff:
            raise ValueError("Error, the coefficients must have the same length of the generators")
        
        return  np.einsum("ijk, i", self.wyck_gen, coeffs)
        
    def ProjectDyn(self, fc, iq = -1):
        """
        Project the force constant matrix in the supercell in the
        basis of the generators
            
        
        Parameters
        ----------
            fc : ndarray ((iq) x 3n x 3n)
                The force constant matrix to be projected on the generator subspace.
                This must be in the supercell. If iq is not specified (or negative) then
                the fc supercell must be passed as 3nx3n and it is multiplied only for that specific irreducible q point
            iq : integer, optional
                the index of the irreducible q point. If negative, all the q
                point are used.
                
        Results
        -------
            ndarray :
                The coefficients of the generators that decompose the number.
        """
        
        if iq >= len(self.dyn_ncoeff):
            raise ValueError("Error, the given iq (%d) must be negative or lower than the number of irreducible points (%d)" % (iq, len(self.dyn_ncoeff)))
        
        
        total_index = 0
        if iq < 0:
            res = np.zeros(np.prod(self.dyn_ncoeff), dtype = np.float64)
            for iq in range(len(self.dyn_ncoeff)):
                res[total_index : total_index + self.dyn_ncoeff[iq]] = np.real(np.einsum("ijk, kj", self.dyn_gen[iq, :,:,:], fc[iq, :, :]))
                total_index += self.dyn_ncoeff[iq]
        else:
            res = np.real(np.einsum("ijk, kj", self.dyn_gen[iq, :,:,:], fc))
            total_index = self.dyn_ncoeff[iq]
            
        return res[:total_index]
    
    
    def GetDynFromCoeff(self, coeffs, iq=0):
        """
        This subroutine generate the dynamical matrix starting from the generator coefficients
        
        
        Parameters
        ----------
            coeffs : ndarray
                The coefficients that represent the dynamical matrix. 
                Must be of the correct dimension.
            iq : int
                The index of the q point
                
                
        Result
        ------
            ndarray 3N x 3N
                The fc generated by the coefficients.
        """
        
        # Check if the coeff are of the correct length
        if len(coeffs) != self.dyn_ncoeff[iq]:
            raise ValueError("Error, the number of coeff %d does not match the number of generator %d. (iq=%d)" % (len(coeffs), self.dyn_ncoeff[iq], iq))
        
        new_coeffs = coeffs
        k_len = len(self.dyn_gen[iq,:, 0,0])
        if len(coeffs != k_len):
            new_coeffs = np.zeros( (k_len), dtype = np.float64)
            new_coeffs[: len(coeffs)] = coeffs
        
        fc = np.einsum("ijk, i", self.dyn_gen[iq, :,:,:], new_coeffs)
        return fc
    
    def GetNCoeffDyn(self):
        """
        Returns the total number of coefficients of the generators for the
        force constant matrix.
        
        Results
        -------
            int
                The number of coefficients (even for each q point)
        """
        return np.sum(self.dyn_ncoeff)
    
    def GetCoeffLimits(self, iq):
        """
        Get the limit index for the coefficients in the specified q index.
        
        Returns
        -------
            int, int
                Start and End index for the generators at the given q point
        """
        
        start_index = 0
        for i in range(iq):
            start_index += self.dyn_ncoeff[i]
        
        return start_index, start_index + self.dyn_ncoeff[iq]
    
    
    def Generate(self, dyn, qe_sym = None):
        """
        GENERATE THE GENERATORS
        =======================
        
        
        The following subroutine generate the generators for the given dynamical
        matrix and the given symmetries.
        
        NOTE: this subroutine must be test for supercells, in particular complex generator should be added
        NOTE: This must be tested in general
        Parameters
        ----------
            dyn : CC.Phonons.Phonons()
                The dynamical matrix represented by the generators.
            qe_sym : CC.symmetries.QE_Symmetry()
                If given, the selected symmetries will be used to generate
                the generators. Otherwise symmetries will be generated from the 
                dynamical matrix using the default parameters.
        """
        
        # Check if the symmetries must be initialize
        if qe_sym is None:
            qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
            
        
        # Get the number of irreducible q points from the matrix
        self.nq = dyn.nqirr
        self.nat = dyn.structure.N_atoms
        
        # Initialize the symmetries at q = 0
        qe_sym.SetupQPoint()
        
        # Prepare the wyckoff basis
        tmp_wyck_gen = np.zeros((3 * self.nat, self.nat, 3), dtype = np.float64)
        
        for i in range( 3 * self.nat):
            x = i % 3
            n = i / 3
            tmp_wyck_gen[i, n, x] = 1
            
            # Symmetrize the vector
            qe_sym.SymmetrizeVector(tmp_wyck_gen[i, :, :])
        
        # Apply the gram-schmidt
        new_gen = tmp_wyck_gen.reshape((3 * self.nat, 3 * self.nat)).transpose()
        new_gen = scipy.linalg.orth(new_gen).transpose()
        
        # Get the number of wyckoff coefficients
        self.wyck_ncoeff = new_gen.shape()[0]
        
        # Reshape the array and get the coefficients
        self.wyck_gen = new_gen.reshape((self.wyck_ncoeff, self.nat, 3))
        
        r = np.arange(3 * self.nat)
        
        self.dyn_ncoeff = np.zeros(self.nq, dtype = int)
        self.dyn_gen = []
        
        # Cycle for each irreducible q point of the matrix
        for iq in range(self.nq):
            q = dyn.q_stars[iq][0]
            # Setup the symmetries for this q point
            qe_sym.SetupQPoint(q)
            
            gh = []
            
            for i in range(self.nat * 3):
                for j in range(i, self.nat * 3):
                    # Take the generator
                    fc = np.zeros((3 * self.nat, 3 * self.nat), dtype = np.complex128)
                    fc[i, j] = 1
                    
                    # Apply the symmetry
                    qe_sym.SymmetrizeDynQ(q, fc)
                    
                    # Check if the generator has already be generated
                    is_new = True
                    for k in range(i+1):
                        mask = fc[k, :] != 0
                        first_value = r[mask]
                        if len(first_value):
                            if k == i:
                                if first_value[0] < j:
                                    is_new = False
                                    break
                        else:
                            is_new = False
                            break
                    
                    # If the generator is new
                    if is_new:
                        qe_sym.ImposeSumRule(fc, "simple")
                        
                        # Check if the sum rule makes this generator desappearing
                        if np.sum ((fc != 0).as_type(int)) != 0:
                            gh.append(fc / np.sqrt(np.trace(fc.dot(fc))))
        
            dim = len(gh)
        
            # Prepare the gram-shmidt
            gh = np.array(gh, dtype = np.complex128)
        
            gh_new = np.reshape((dim, 9 * self.nat**2)).transpose()
            gh_new = scipy.linalg.orth(gh_new).transpose()
        
            self.dyn_ncoeff = np.shape(gh_new)[0]
        
            self.dyn_gen.append(np.reshape(gh_new, (self.dyn_ncoeff, 3*self.nat, 3*self.nat)))
            
                            
                    
                    
                    
                
            