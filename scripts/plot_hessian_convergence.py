#!python

import sys, os
import sscha
import sscha.DynamicalLanczos, sscha.StaticHessian
import numpy as np
import matplotlib.pyplot as plt

import cellconstructor as CC 
import cellconstructor.Units

def print_info():
    print("The program requires 2 or 3 arguments:")
    print()
    print("1) The directory in which the calculations are stored")
    print("2) The PREFIX of the calculation")
    print("3) [Optional] The status to initialize the system (must be the prefix of two files: a json and an npz)")
    print("4) [Optional] The dynamical matrix for reference phonon modes")
    print()
    print("The last two arguments are optional, however if you want to specify the fourth parameter, you need also the third one")
    print()


def plot_convergence(directory, prefix, sh, reference = None):
    """
    Plot the convergence
    """

    # Get all the files from the minimization
    all_files = [x for x in os.listdir(directory) if x.startswith(prefix) and x.endswith(".dat") and len(x) == len(prefix) + 14]
    all_files.sort()
    finalfile = os.path.join(directory, prefix)
    if os.path.exists(finalfile):
        all_files.append(prefix)
        print("The algorithm converged!")
    else:
        print("The algorithm did not converged!")

    ws = np.zeros((sh.lanczos.pols.shape[0], len(all_files)))
    
    print()
    for i, fname in enumerate(all_files):
        sh.vector = np.loadtxt(os.path.join(directory, fname))

        G = sh.retrieve_hessian(noq = True) / np.sqrt(np.outer(sh.lanczos.m, sh.lanczos.m))

        w2 = np.linalg.eigvalsh(G)
        ws[:, i] = np.sign(w2) * np.sqrt(np.abs(w2))

        if i % 10 == 0:
            sys.stdout.write("\rProgress {:d} %".format((i * 100) // len(all_files)))
            sys.stdout.flush()
    print()
    
    # Plot
    plt.rcParams["font.family"] = "Liberation Serif"
    plt.figure(dpi = 120)
    
    for i in range(ws.shape[0]):
        plt.plot(ws[i, :] * CC.Units.RY_TO_CM)

    if reference is not None:
        w, p = reference.DiagonalizeSupercell()
        trans = CC.Methods.get_translations(p, reference.structure.generate_supercell(reference.GetSupercell()).get_masses_array())
        w = w[~trans]

        for wx in w:
            plt.axhline(wx * CC.Units.RY_TO_CM, 0, 1, ls = "--", color = "k")

    plt.xlabel("Optimization steps")
    plt.ylabel("Hessian frequencies [cm-1]")
    plt.title("Hessian convergence")
    plt.tight_layout()
    plt.show()
    
    

if __name__ == "__main__":
    # Check the arguments

    nargs = len(sys.argv)

    if nargs != 3 and nargs != 4 and nargs != 5:
        print_info()
        exit(1)

    datadir = sys.argv[1]
    prefix = sys.argv[2]

    static_hessian = sscha.StaticHessian.StaticHessian()
    
    if nargs >= 4:
        if not os.path.exists(sys.argv[3] + ".json"):
            static_hessian.lanczos.load_status(sys.argv[3])
            static_hessian.preconitioned = False
            print("Neglecting json file")
        else:
            static_hessian.load_status(sys.argv[3])
            print("Loading from json file")

    else:
        static_hessian.lanczos.load_status(os.path.join(datadir, prefix))

    reference = None

    if nargs == 5:
        dname = os.path.dirname(sys.argv[4])
        fname = os.path.split(sys.argv[4])[-1]
        nqirr = len([x for x in os.listdir(dname) if x.startswith(fname)])
        reference = CC.Phonons.Phonons(sys.argv[4], nqirr)
    
    plot_convergence(datadir, prefix, static_hessian, reference)
    

        
    

