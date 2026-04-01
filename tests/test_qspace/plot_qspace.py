import sys, os
import matplotlib.pyplot as plt
import numpy as np


import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble
import tdscha, tdscha.QSpaceKPM

N_STEPS = 256
NQIRR = 3
T = 250
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'test_julia', 'data')
IGNORE_V3 = True
IGNORE_V4 = True

def plot_qspace(show=False):
    dyn = CC.Phonons.Phonons("{}/dyn_gen_pop1_".format(DATA_DIR), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin(DATA_DIR, 1)

    kpm = tdscha.QSpaceKPM.QSpaceKPM(ens, lo_to_split=None)
    kpm.ignore_v3 = IGNORE_V3
    kpm.ignore_v4 = IGNORE_V4
    kpm.init()

    kpm.prepare_mode_q(0, 5)
    
    # Estimate KPM steps for 2 cm⁻¹ precision
    n_steps_estimated = kpm.estimate_kpm_steps(2.0, bound_factor=1.2)
    print(f"Estimated KPM steps for 2 cm⁻¹ precision: {n_steps_estimated}")
    
    # Use the estimated steps or the fixed N_STEPS, whichever is larger
    n_steps = max(N_STEPS, n_steps_estimated)
    print(f"Using {n_steps} KPM steps")
    
    kpm.run_KPM(n_steps, verbose=False, bound_factor=1.2)  # Use tighter bounds for better resolution
    
    w = np.linspace(0, 200, 1000)
    w_ry = w / CC.Units.RY_TO_CM
    spectral = kpm.get_spectral_function_KPM(w_ry, regularization="jackson")

    # Do the same with the standard lanczos
    lanc = tdscha.QSpaceLanczos.QSpaceLanczos(ens, lo_to_split=None)
    lanc.ignore_v3 = IGNORE_V3  
    lanc.ignore_v4 = IGNORE_V4
    lanc.init()
    lanc.prepare_mode_q(0, 5)
    lanc.run_FT(N_STEPS, verbose=False)

    gf_lanc = lanc.get_green_function_continued_fraction(w_ry, use_terminator=False, smearing=0.05 * kpm.w_q[5, 0])
    spectral_lanc = -np.imag(gf_lanc)


    peak_pos = w[np.argmax(spectral)]
    peak_pos_lanc = w[np.argmax(spectral_lanc)]
    print("Peak position KPM: {:.2f} cm-1".format(peak_pos))
    print("Peak position Lanczos: {:.2f} cm-1".format(peak_pos_lanc))
    print("Error: {:.2f} cm-1 | {} %".format(abs(peak_pos - peak_pos_lanc), 100 * abs(peak_pos - peak_pos_lanc) / peak_pos_lanc))

    # Plot the resunt
    fig = plt.figure()
    plt.plot(w, spectral, label="KPM")
    plt.plot(w, spectral_lanc, label="Lanczos")
    plt.axvline(kpm.w_q[5, 0] * CC.Units.RY_TO_CM, color="C0", linestyle="--", label="Mode freq")
    plt.xlabel("Frequency (cm$^{-1}$)")
    plt.ylabel("Spectral function")
    plt.legend()
    plt.savefig("qspace_kpm_spectral.png")
    if show:
        plt.show()


if __name__ == "__main__":
    show = sys.argv[-1] == "--show"
    plot_qspace(show=show)

    
