import sys, os
import numpy as np

import cellconstructor as CC, cellconstructor.Phonons
from cellconstructor.Settings import ParallelPrint as print

import sscha, sscha.Ensemble, sscha.SchaMinimizer
import tdscha
import tdscha.DynamicalLanczos as DL

import matplotlib.pyplot as plt


W_START = 0.0
W_END = 200.0 # cm-1
NW = 2000 # number of frequency points in the IR spectrum

SMEARING = 2.5 # smearing in cm-1

IR_FILE = "ir_polx_qz.npz"  # Results file from compute_ir.py

def plot_results():
    # Load the IR file 
    qspace_lanczos = DL.Lanczos()
    qspace_lanczos.load_status(IR_FILE)

    # Get the frequency array
    w_cm = np.linspace(W_START, W_END, NW)
    w_ry = w_cm / CC.Units.RY_TO_CM
    smearing_ry = SMEARING / CC.Units.RY_TO_CM

    # Get the IR Green function
    gf = qspace_lanczos.get_green_function_continued_fraction(w_ry, smearing=smearing_ry, use_terminator=False)

    # Plot the IR spectrum (i.e. the imaginary part of the Green function)
    plt.figure(figsize=(8, 6))
    plt.plot(w_cm, -np.imag(gf), label="IR Spectrum")
    plt.xlabel("Frequency (cm$^{-1}$)")
    plt.ylabel("-Im(GF)")
    plt.title("IR Spectrum")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()
