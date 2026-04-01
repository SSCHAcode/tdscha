#!python
from __future__ import print_function
from __future__ import division

import sys, os

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import cellconstructor as CC
import cellconstructor.Phonons

import sscha
import tdscha, tdscha.DynamicalLanczos as DL
import tdscha.QSpaceKPM as QKPM
import sscha.Ensemble

HEADER = """

TDSCHA
------

"""


if __name__ == "__main__":
    print(HEADER)

    if len(sys.argv) not in [2, 4, 5]:
        print("""
Plot the spectrum of a TDSCHA calculation.

Usage: tdscha-plot.py <file> [w_start w_end [smearing]]

Pass a .abc, .npz, or .kpm file resulting from a linear response calculation.
- .abc / .npz : use Lanczos continued fraction
- .kpm        : use KPM spectral function
Optionally you can pass a range of frequencies (cm-1) and the smearing.
""")
        exit()
    
    fname = sys.argv[1]
    assert os.path.exists(fname), "Error, file {} does not exist".format(fname)
    
    use_kpm = fname.endswith(".kpm")

    if use_kpm:
        print("Loading KPM file {}".format(fname))
        kpm = QKPM.QSpaceKPM(None)
        kpm.load_kpm(fname)
        lanc = None
    else:
        print("Loading file {}".format(fname))
        lanc = DL.Lanczos()
        if fname.endswith(".abc"):
            lanc.load_abc(fname)
        elif fname.endswith(".npz"):
            lanc.load_status(fname)
        else:
            print("ERROR, the specified file must be a .abc, .npz, or .kpm file.")
            exit()

    w_start = 0
    w_end = 5000
    n_w = 50000
    smearing = 5

    if len(sys.argv) > 2:
        w_start = float(sys.argv[2])
        w_end = float(sys.argv[3])
    if len(sys.argv) == 5:
        smearing = float(sys.argv[4])
    
    w = np.linspace(w_start, w_end, n_w)
    w_ry = w / CC.Units.RY_TO_CM
    smearing /= CC.Units.RY_TO_CM

    if use_kpm:
        # KPM spectral function does not use smearing parameter
        spectrum = kpm.get_spectral_function_KPM(w_ry, regularization="jackson")
    else:
        gf = lanc.get_green_function_continued_fraction(w_ry, smearing = smearing, use_terminator=False)
        spectrum = - np.imag(gf)

    # Print some info about the calculation
    print()
    if use_kpm:
        print("Number of KPM moments: {}".format(kpm.kpm_n_moments))
    else:
        print("Number of poles: {}".format(len(lanc.a_coeffs)))
    
    plt.plot(w, spectrum)
    plt.xlabel("Frequency [cm-1]")
    plt.ylabel("Spectrum [a.u.]")
    plt.tight_layout()
    plt.show()


