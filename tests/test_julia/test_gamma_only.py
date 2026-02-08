from __future__ import print_function

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons

import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL

from tdscha.Parallel import pprint as print

import sys, os


def test_gamma_only():
    """
    Test that gamma_only=True produces the same Lanczos coefficients
    as the standard full-symmetry approach for a Gamma-point IR perturbation.

    Uses fake effective charges: +I for atom 1, -I for atom 2 (acoustic sum rule satisfied).
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    T = 250
    NQIRR = 3
    N_STEPS = 5

    dyn = CC.Phonons.Phonons("data/dyn_gen_pop1_", NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin("data", 1)

    # Fake effective charges: +1*I for atom 1, -1*I for atom 2 (ASR satisfied)
    nat_uc = dyn.structure.N_atoms  # 2
    ec = np.zeros((nat_uc, 3, 3))
    ec[0] = np.eye(3)
    ec[1] = -np.eye(3)

    # Run with full symmetries (reference)
    lanc_full = DL.Lanczos(ens)
    lanc_full.mode = DL.MODE_FAST_JULIA
    lanc_full.init(use_symmetries=True)
    n_syms_full = lanc_full.n_syms
    lanc_full.prepare_ir(effective_charges=ec, pol_vec=np.array([1., 0., 0.]))
    lanc_full.run_FT(N_STEPS)

    # Run with gamma_only optimization
    lanc_gamma = DL.Lanczos(ens)
    lanc_gamma.gamma_only = True
    lanc_gamma.mode = DL.MODE_FAST_JULIA
    lanc_gamma.init(use_symmetries=True)
    n_syms_gamma = lanc_gamma.n_syms
    lanc_gamma.prepare_ir(effective_charges=ec, pol_vec=np.array([1., 0., 0.]))
    lanc_gamma.run_FT(N_STEPS)

    # Verify fewer symmetries used
    print("Full syms: {}, Gamma-only syms: {}".format(n_syms_full, n_syms_gamma))
    assert n_syms_gamma < n_syms_full, \
        "gamma_only should use fewer syms: {} vs {}".format(n_syms_gamma, n_syms_full)

    # Verify Lanczos a_coeffs match
    for i in range(len(lanc_full.a_coeffs)):
        diff = abs(lanc_full.a_coeffs[i] - lanc_gamma.a_coeffs[i])
        denom = max(abs(lanc_full.a_coeffs[i]), 1e-15)
        assert diff / denom < 1e-6, \
            "a_coeffs[{}] mismatch: full={} gamma_only={}".format(
                i, lanc_full.a_coeffs[i], lanc_gamma.a_coeffs[i])

    # Verify Lanczos c_coeffs match
    for i in range(len(lanc_full.c_coeffs)):
        diff = abs(lanc_full.c_coeffs[i] - lanc_gamma.c_coeffs[i])
        denom = max(abs(lanc_full.c_coeffs[i]), 1e-15)
        assert diff / denom < 1e-6, \
            "c_coeffs[{}] mismatch: full={} gamma_only={}".format(
                i, lanc_full.c_coeffs[i], lanc_gamma.c_coeffs[i])

    print("PASSED: gamma_only produces same results with {} fewer replicas".format(
        n_syms_full - n_syms_gamma))


if __name__ == "__main__":
    test_gamma_only()
