from __future__ import print_function

import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble

import tdscha, tdscha.DynamicalLanczos as DL
import time, numpy as np

import matplotlib.pyplot as plt


# INPUT VARIABLES -------------------
FREQ_START = 0   # cm-1
FREQ_END = 1000 # cm-1
N_FREQS = 9000

USE_WINGER = False

# Lanczos status file
if not USE_WINGER:
    DATA_FILE = "data/tdscha_lanczos_STEP100.npz"
else:
    DATA_FILE = "data_wigner/tdscha_lanczos_STEP100.npz"

# If you submitted with the tdscha-lanczos.x
# Copy all the files inside the directory
LOAD_FROM_EXECUTABLE = False
DIRECTORY = "submit_on_cluster"
PREFIX = "tdscha_lanczos"

# HERE THE SCRIPT

# Get the frequency array for plotting
w = np.linspace(FREQ_START, FREQ_END, N_FREQS)

# Convert in Ry
w_ry = w / CC.Units.RY_TO_CM

# Load the lanczos
lanczos = DL.Lanczos()

if not LOAD_FROM_EXECUTABLE:
    lanczos.load_status(DATA_FILE)
else:
    lanczos.load_from_input_files(PREFIX, DIRECTORY)
    
# Get the dynamical green function
green_function = lanczos.get_green_function_continued_fraction(w_ry)


# Print on the screen the static value of the free energy hessian
# (In this way we show how to compute the static free energy hessian).
w_static  = np.array([0])
gf_static = lanczos.get_green_function_continued_fraction(w_static)

# The free energy hessian is the inverse of the static green function (real part)
# computed at omega = 0
hessian = 1 / np.real(gf_static[0])

# Get the frequency (the square root with negative sign for imaginary values)
static_frequency = np.sign(hessian) * np.sqrt(np.abs(hessian))

# Convert from Ry to cm-1
static_frequency *= CC.Units.RY_TO_CM

print("The static frequency of the simualated mode is: {} cm-1".format(static_frequency))


# Plot the imaginary part
plt.figure(dpi = 150)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("- Im(G) / Re(G)")
plt.title("Spectral function \n Static freqeuncy = {} cm-1".format(static_frequency))
plt.plot(w, -np.imag(green_function)/np.abs(np.imag(green_function)).max(), label = '-Im(G)')
plt.plot(w, +np.real(green_function)/np.abs(np.real(green_function)).max(),'--', label = '+Re(G)')
plt.legend()

plt.tight_layout()
plt.show()



