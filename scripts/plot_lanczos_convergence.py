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
import sscha.Ensemble

# Get from command line the last lanczos step
if not len(sys.argv) in [2, 3]:
    print("Error, I require the Lanczos .npz file to be analyzed (the last), and (optionally) the smearing [cm-1]")
    raise ValueError("Error, I require the Lanczos .npz file to be analyzed (the last)")

fname = sys.argv[1]

smearing = 1
if len(sys.argv) == 3:
    smearing = float(sys.argv[2])

# Check if the file exists
if not os.path.exists(fname):
    raise IOError("Error, the specified file '{}' does not exist.".format(fname))

# Check how many files are here
#files = [x for x in  os.listdir(".") if "LANCZOS_STEP" in x and ".npz" in x]


data = []
nat = 0
print ("Reading the lanczos file...")
data = DL.Lanczos()

if fname.endswith('.abc'):
    data.load_abc(fname)
else:
    data.load_status(fname)
nat = data.nat
N_iters = len(data.a_coeffs) - 1

# Now get the static converge
W_START = 0
W_END = 10000 / CC.Units.RY_TO_CM
NW = 10000
SMEARING = smearing / CC.Units.RY_TO_CM

print ("Computing the static responce...")
freqs = np.zeros((N_iters, 3*nat))
dynamical = np.zeros((N_iters, NW))
dynamical_noterm = np.zeros((N_iters, NW))
w_array = np.linspace(W_START, W_END, NW)
w_static = np.zeros((N_iters), dtype = np.double)

a_coeffs = np.copy(data.a_coeffs)
b_coeffs = np.copy(data.b_coeffs)
c_coeffs = np.copy(data.c_coeffs)

for i in range(N_iters):
    data.a_coeffs = a_coeffs[:i+1]
    data.b_coeffs = b_coeffs[:i]
    data.c_coeffs = c_coeffs[:i]
    
    #fc_odd = data.get_static_odd_fc(False)
    #fc_odd /= np.sqrt(np.outer(data.m, data.m))
    #w, p = np.linalg.eigh(fc_odd)
    #sign_mask = np.sign(w)
    #w = sign_mask * np.sqrt(np.abs(w))
    #w *= CC.Phonons.RY_TO_CM
    #freqs[i, :] = w 

    gf = data.get_green_function_continued_fraction(w_array, use_terminator = False, smearing = SMEARING)

    dynamical_noterm[i, :] = -np.imag( gf)
    w2 = 1 / np.real(gf[0])
    w_static[i] = np.sign(w2) * np.sqrt(np.abs(w2)) * CC.Units.RY_TO_CM
    dynamical[i, :] = -np.imag( data.get_green_function_continued_fraction(w_array, use_terminator = True, smearing = SMEARING))
    



    if i % 10 == 0:
        sys.stderr.write("\rProgress %% %d" % (i * 100 / N_iters))
        sys.stderr.flush()
sys.stderr.write("\n")

print ("Plotting the results...")

plt.figure(dpi = 160)
plt.title("Static Frequency")
plt.plot(np.arange(N_iters), w_static, marker = "o")
plt.xlabel("Lanczos step")
plt.ylabel("Static frequency [cm-1]")
plt.tight_layout()


plt.figure(dpi = 160)
plt.title("Green function without terminator")
plt.imshow(dynamical_noterm, aspect = "auto", origin = "lower", extent = [W_START*CC.Phonons.RY_TO_CM, W_END*CC.Phonons.RY_TO_CM, 1, N_iters])
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Steps")
plt.colorbar()
plt.tight_layout()


plt.figure(dpi = 160)
plt.title("Green function with terminator")
plt.imshow(dynamical, aspect = "auto", origin = "lower", extent = [W_START*CC.Phonons.RY_TO_CM, W_END*CC.Phonons.RY_TO_CM, 1, N_iters])
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Steps")
plt.colorbar()
plt.tight_layout()



# get colormap
cmap=plt.cm.viridis
# build cycler with 5 equally spaced colors from that colormap
c = cycler('color', cmap(np.linspace(0,1,N_iters)) )
# supply cycler to the rcParam
plt.rcParams["axes.prop_cycle"] = c

plt.figure(dpi = 160)
plt.title("Green function with terminator")
for i in range(N_iters):
    plt.plot(w_array * CC.Phonons.RY_TO_CM, dynamical[i, :])

# The last one is plotted in red underlined
plt.plot(w_array * CC.Phonons.RY_TO_CM, dynamical[-1, :], color = "r", linewidth = 3.5)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Steps")
plt.tight_layout()

plt.show()
