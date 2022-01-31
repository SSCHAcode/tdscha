from __future__ import print_function

import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble

import tdscha, tdscha.DynamicalLanczos as DL
import time, numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# INPUT VARIABLES -------------------
FREQ_START = 1000   # cm-1
FREQ_END = 4000 # cm-1
N_FREQS = 10000

# Smearing in cm-1 -> Ry
delta = 10. / CC.Units.RY_TO_CM

MODE_PERTURBATION_ID = 25

step_range = np.array([100, 200])

LOAD_FROM_EXECUTABLE = False
DIRECTORY = "submit_on_cluster"
PREFIX = "tdscha_lanczos"

# Get the frequency array for plotting
w = np.linspace(FREQ_START, FREQ_END, N_FREQS)

# Convert in Ry
w_ry = w / CC.Units.RY_TO_CM

w_static  = np.array([0])

##########
# NORMAL #
##########

static_freq = []
RE_G = []
IM_G = []

for i in range(len(step_range)):
    print(i)
    # Load the lanczos
    lanczos = DL.Lanczos()
    # Lanczos status file
    DATA_FILE = "data_md_{}/tdscha_lanczos_STEP{}.npz".format(MODE_PERTURBATION_ID, step_range[i])

    if not LOAD_FROM_EXECUTABLE:
        lanczos.load_status(DATA_FILE)
    else:
        lanczos.load_from_input_files(PREFIX, DIRECTORY)

    G = lanczos.get_green_function_continued_fraction(w_ry, use_terminator = False, smearing = delta)
    
    RE_G.append(np.real(G))
    IM_G.append(-np.imag(G))

#     gf_static = lanczos.get_green_function_continued_fraction(w_static)
#     hessian = 1 / np.real(gf_static[0])
#     w_static = np.sign(hessian) * np.sqrt(np.abs(hessian)) * CC.Units.RY_TO_CM
    
#     static_freq.append(w_static)

RE_G = np.asarray(RE_G).reshape(len(step_range), len(w))
IM_G = np.asarray(IM_G).reshape(len(step_range), len(w))

plt.figure(dpi = 150)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Re(G)")
colors = ['red', 'blue', 'gree', 'purple']
shift = 0
for i in range(len(step_range)):
#     plt.plot(w, -np.imag(green_function_wigner), label = 'Wigner -Im(G)')
    plt.plot(w, RE_G[i,:] + shift * i, '--', color = colors[i], label = 'Nsteps = {}'.format(step_range[i]))
    plt.legend()

plt.tight_layout()
plt.show()


# plt.figure(dpi = 150)
# plt.xlabel("Frequency [cm-1]")
# plt.ylabel("-Im(G)")
# colors = ['red', 'blue', 'gree', 'purple']
# shift = 10
# for i in range(len(step_range)):
# #     plt.plot(w, -np.imag(green_function_wigner), label = 'Wigner -Im(G)')
#     plt.plot(w, IM_G[i,:] + shift * i, '--', color = colors[i], label = 'Nsteps = {}'.format(step_range[i]))
#     plt.legend()

# plt.tight_layout()
# plt.show()
# plt.pause(0.1)