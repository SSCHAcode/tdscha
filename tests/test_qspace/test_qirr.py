import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble

import sys, os
import tdscha, tdscha.QSpaceHessian as QH

T = 420
NQIRR = 18


def test_hessian_nqirr():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    dyn = CC.Phonons.Phonons(os.path.join("qirr_hessian", "final_dyn_T420_"), NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.generate(10) # No forces, ok just test the nqirr for hessian

    # Prepare the hessian calculation
    qh = QH.QSpaceHessian(ens, verbose=True, lo_to_split=None, ignore_v3 = True, ignore_v4 = True)
    qh.init(use_symmetries=True)
    hessian = qh.compute_full_hessian(tol=1e-2, max_iters=10)

    # Check the number of q-points in the hessian
    nqirr = len(qh.irr_qpoints)
    print(f"Number of q-points in the Hessian: {nqirr}")
    print(f"Expected number of q-points (NQIRR): {NQIRR}")

    assert nqirr == NQIRR, f"Expected {NQIRR} q-points, but got {nqirr}"


if __name__ == "__main__":
    test_hessian_nqirr()

 
