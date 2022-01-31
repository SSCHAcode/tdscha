from __future__ import print_function

import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble

import tdscha, tdscha.DynamicalLanczos as DL
import time, numpy as np

import matplotlib.pyplot as plt


# INPUT VARIABLES -------------------
FREQ_START = 0   # cm-1
FREQ_END = 5000 # cm-1
N_FREQS = 20000

# Smearing in cm-1 -> Ry
delta = 10. / CC.Units.RY_TO_CM

MODE_PERTURBATION_ID = 0

step = 95

USE_THIRD_ORDER = True  # Use the third order in the calculation
USE_FOURTH_ORDER = True # Use the fourth order (2x computational cost)
SAVE_EACH = 5 # Save the result each tot steps (for restart)

print('D3 = {} D4 = {}'.format(USE_THIRD_ORDER, USE_FOURTH_ORDER))

if USE_FOURTH_ORDER and USE_FOURTH_ORDER:
    DATA_FILE = "data_md_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, step)
    DATA_FILE_WIGNER = "data_wigner_md_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, step)
elif not USE_FOURTH_ORDER:
    DATA_FILE = "data_no_D4_md_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, step) # The folder in which the data are saved
    DATA_FILE_WIGNER = "data_wigner_no_D4_md_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, step)
else:
    DATA_FILE = "data_harm_md_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, step) # The folder in which the data are saved
    DATA_FILE_WIGNER = "data_wigner_harm_md_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, step)


# If you submitted with the tdscha-lanczos.x
# Copy all the files inside the directory
LOAD_FROM_EXECUTABLE = False
DIRECTORY = "submit_on_cluster"
PREFIX = "tdscha_lanczos"

# Get the frequency array for plotting in cm-1
w = np.linspace(FREQ_START, FREQ_END, N_FREQS)

# Convert in Ry
w_ry = w / CC.Units.RY_TO_CM
w_static  = np.array([0])



##########
# NORMAL #
##########

# Load the lanczos
lanczos = DL.Lanczos()

if not LOAD_FROM_EXECUTABLE:
    lanczos.load_status(DATA_FILE)
else:
    lanczos.load_from_input_files(PREFIX, DIRECTORY)
    
# Get the dynamical green function
green_function = lanczos.get_green_function_continued_fraction(w_ry, use_terminator = False, smearing = delta)

gf_static = lanczos.get_green_function_continued_fraction(w_static)

# The free energy hessian is the inverse of the static green function (real part)
# computed at omega = 0
hessian = 1 / np.real(gf_static[0])

# Get the frequency (the square root with negative sign for imaginary values)
static_frequency = np.sign(hessian) * np.sqrt(np.abs(hessian))

# Convert from Ry to cm-1
static_frequency *= CC.Units.RY_TO_CM


##########
# WIGNER #
##########

# Load the lanczos
lanczos = DL.Lanczos()

if not LOAD_FROM_EXECUTABLE:
    lanczos.load_status(DATA_FILE_WIGNER)
else:
    lanczos.load_from_input_files(PREFIX, DIRECTORY)
    
# lanczos.c_coeffs = lanczos.b_coeffs   
# print('Diff between coeff = {}'.format(np.asarray(lanczos.c_coeffs - lanczos.b_coeffs)))
print('Max diff between coeff = {}'.format(np.asarray(lanczos.c_coeffs - lanczos.b_coeffs).max()))
    
# Get the dynamical green function
green_function_wigner = lanczos.get_green_function_continued_fraction(w_ry, use_terminator = False, smearing = delta)

gf_static = lanczos.get_green_function_continued_fraction(w_static)

# The free energy hessian is the inverse of the static green function (real part)
# computed at omega = 0
hessian = 1 / np.real(gf_static[0])

# Get the frequency (the square root with negative sign for imaginary values)
static_frequency_wigner = np.sign(hessian) * np.sqrt(np.abs(hessian))

# Convert from Ry to cm-1
static_frequency_wigner *= CC.Units.RY_TO_CM


########
# PLOT #
########

TITLE = """[AUXILIARY] $\omega$ = {:.2f} cm-1
[NORMAL] $\omega$ = {:.2f} cm-1  [WIGNER] $\omega$ = {:.2f} cm-1""".format(lanczos.w[MODE_PERTURBATION_ID] * CC.Units.RY_TO_CM, static_frequency, static_frequency_wigner)

# Plot the imaginary part
plt.figure(dpi = 150)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("- $\\omega$Im(G)")
plt.title(TITLE)
plt.axvline(x=lanczos.w[MODE_PERTURBATION_ID] * CC.Units.RY_TO_CM, ymin=-10, ymax=+10, color = 'red', ls = 'dotted', label = 'Auxiliary SCHA freq.')
plt.plot(w, -w * np.imag(green_function_wigner), label = 'Wigner -$\\omega$Im(G)')
plt.plot(w, -w * np.imag(green_function), '--', label = 'Normal -$\\omega$Im(G)')
plt.legend()

plt.tight_layout()
plt.show()


# Plot the real part
plt.figure(dpi = 150)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Re(G)")
plt.title(TITLE)
plt.axvline(x=lanczos.w[MODE_PERTURBATION_ID] * CC.Units.RY_TO_CM, ymin=-10, ymax=+10, color = 'red', ls = 'dotted', label = 'Auxiliary SCHA freq.')
plt.plot(w, +np.real(green_function_wigner), label = 'Wigner +Re(G)')
plt.plot(w, +np.real(green_function),'--', label = 'Normal +Re(G)')
plt.legend()

plt.tight_layout()
plt.show()


#  Real and Imaginary part
plt.figure(dpi = 150)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Re(G)/- Im(G)")
plt.title('Wigner')
plt.axvline(x=static_frequency_wigner, ymin=-10, ymax=+10, color = 'purple', ls = 'dotted', label = 'Wigner static freq.')
plt.plot(w, +np.real(green_function_wigner), label = 'Wigner +Re(G)')
plt.plot(w, -np.imag(green_function_wigner), '--', label = 'Wigner -Im(G)')
plt.legend()

plt.tight_layout()
plt.show()


# Real and Imaginary part
plt.figure(dpi = 150)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Re(G)/- Im(G)")
plt.title('Normal')
plt.axvline(x=static_frequency, ymin=-10, ymax=+10, color = 'green', ls = 'dotted', label = 'Static freq.')
plt.plot(w, +np.real(green_function), label = 'Normal +Re(G)')
plt.plot(w, -np.imag(green_function), '--', label = 'Normal -Im(G)')
plt.legend()

plt.tight_layout()
plt.show()

delta = static_frequency - static_frequency_wigner
print('\nStatic freq with Lanczos: diff = {} cm-1 = {} meV'.format(delta, delta * CC.Units.RY_TO_EV * 1000/CC.Units.RY_TO_CM))

# # Static freq
# plt.figure(dpi = 150)
# plt.xlabel("MODE")
# plt.ylabel("Frequency [cm-1]")
# plt.title('Normal')
# plt.plot(np.arange(0, len(lanczos.w), 1), lanczos.w * CC.Units.RY_TO_CM, 'rx', label = 'Auxiliary freq')
# plt.legend()

# plt.tight_layout()
# plt.show()

# np.savetxt('static_freq.txt', lanczos.w * CC.Units.RY_TO_CM)