import numpy as np
import matplotlib.pyplot as plt
import os
import time, numpy as np
import matplotlib.pyplot as plt


RE_G = np.load('real_G.npy')
IM_G = np.load('imag_G.npy')
stat_freqs = np.loadtxt('freq_stat.txt')

RE_G[1,:,:] = -RE_G[1,:,:]
stat_freqs[1,:] = -stat_freqs[1,:]

print(np.abs(stat_freqs[0,:] - stat_freqs[1,:]))

FREQ_START = 0 # cm-1
FREQ_END = 6000 # cm-1
N_FREQS = 10000
# Get the frequency array for plotting
w = np.linspace(FREQ_START, FREQ_END, N_FREQS)

# Plot the imaginary part
plt.figure(dpi = 150)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("-Im(G)")
plt.title("-Im(G)")
for i in range(3):
    plt.plot(w, IM_G[0,i,:] /np.max(IM_G[0,i,:]), label = 'Normal mode = {}'.format(i))
    plt.plot(w, IM_G[1,i,:] /np.max(IM_G[1,i,:]), '--', label = 'Wigner mode = {}'.format(i))
plt.legend()
plt.show()


# Plot the real part
plt.figure(dpi = 150)
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Re(G)")
plt.title("Re(G)")
for i in range(3):
    plt.plot(w, RE_G[0,i,:] /np.max(RE_G[0,i,:]), label = 'Normal mode = {}'.format(i))
    plt.plot(w, RE_G[1,i,:] /np.max(RE_G[1,i,:]), '--', label = 'Wigner mode = {}'.format(i))
plt.legend()
plt.show()

# Plot the freq
plt.figure(dpi = 150)
plt.ylabel("Frequency (cm-1)")
plt.xlabel("Mode number")
plt.title("Static frequencies")
y = np.arange(3)
plt.plot(y, stat_freqs[0,:], 'x', label = 'Normal: freq mode')
plt.plot(y, stat_freqs[1,:], 's', label = 'Wigner: freq mode')
plt.legend()
plt.show()


