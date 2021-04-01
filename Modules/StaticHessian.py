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
        self.W = np.zeros( (self.lanczos.n_modes, self.lanczos.n_modes, self.lanczos.n_modes), dtype = sscha.DynamicalLanczos.TYPE_DP)

        if verbose:
            print("Memory of StaticHessian initialized.")
            print("     memory requested: {} Gb of RAM per process".format((self.Ginv.nbytes + self.W.nbytes) / 1024**3))
            print("     (excluding memory occupied to store the ensemble)")
            print("     (during the Hessian optimization, three times this memory is required.)")
        
