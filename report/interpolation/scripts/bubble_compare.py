"""Example application for the report: one-phonon spectral function at an
INTERPOLATED q-point from the fine-mesh TDSCHA Lanczos, compared with the
standard d3 dynamic bubble of cellconstructor.Spectral (exact Phi3 tensor,
centered + ASR, same k-integration mesh).

Toy: anharmonic diatomic chain (g3 only, weak coupling), coarse (1,1,4)
ensemble, fine mesh (1,1,16).
Writes bubble_compare.json with the curves.
"""
import os, sys, json, time
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, "os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "test_interpolation")")
import numpy as np
import _toy_chain as TC
import cellconstructor as CC
import cellconstructor.ForceTensor
import cellconstructor.Spectral
import tdscha.QSpaceInterpolation as QI

T = 300.0
N_CONF = 8000
G3 = 0.1
LC = 4
LF = 16
N_STEPS = 120
SMEAR_RY = 2.0e-5          # ~4.4 cm-1
RY_TO_CM = CC.Units.RY_TO_CM

dyn = TC.build_dyn(LC)
unit = dyn.structure
sc_struct = unit.generate_supercell(dyn.GetSupercell())
nat_sc = sc_struct.N_atoms

# ---------------------------------------------------------------------
# exact third-order tensor of the bond model in the coarse supercell
# ---------------------------------------------------------------------
bonds = TC.get_bonds(sc_struct, unit, LC)
phi3 = np.zeros((3 * nat_sc, 3 * nat_sc, 3 * nat_sc))
for (i, j, k_spring) in bonds:
    for alpha in range(3):
        for (a, sa) in ((i, 1.0), (j, -1.0)):
            for (b, sb) in ((i, 1.0), (j, -1.0)):
                for (c, sc_) in ((i, 1.0), (j, -1.0)):
                    phi3[3 * a + alpha, 3 * b + alpha, 3 * c + alpha] += \
                        2.0 * G3 * sa * sb * sc_
# units: G3 in Ry/Bohr^3 -> Spectral expects Ry/A^3? tensors from phonons are
# in the dyn units (Ry/A^2 style mixed). The dyn was built in Ry/Bohr^2 with
# positions in A; forces = -dE/du with u in Bohr in the toy. Convert phi3 to
# the same convention as the dyn (energy per displacement^3 with u in the
# SAME units the tensor2 uses). tensor2 comes from dynmats (Ry/Bohr^2), so
# use Ry/Bohr^3 consistently: frequencies come out in Ry both ways ONLY if
# masses are in Ry units and displacements consistent. The bubble divides
# phi3 by sqrt(m)^3 and phi2 by m: [phi2/m] = w^2 -> phi2 in Ry/Bohr^2 with
# m in Ry units gives w in Ry (as in the toy dyn). For phi3 the mode
# matrix elements are phi3/sqrt(m^3) [Ry/Bohr^3 * Bohr^3 ...]: consistent
# with phi2 in Ry/Bohr^2 IF displacements are measured in Bohr in both.
# CC tensors interpolate phases with positions in A but that only affects
# q-space phases, not units. => keep phi3 in Ry/Bohr^3, matching phi2.

t2 = CC.ForceTensor.Tensor2(unit, sc_struct, dyn.GetSupercell())
t2.SetupFromPhonons(dyn)
t2.Center()
t2.Apply_ASR()

t3 = CC.ForceTensor.Tensor3(unit, sc_struct, dyn.GetSupercell())
t3.SetupFromTensor(phi3)
t3.Center()
t3.Apply_ASR()

# ---------------------------------------------------------------------
# Lanczos side: coarse ensemble interpolated to the fine mesh
# ---------------------------------------------------------------------
ens = TC.make_ensemble(dyn, T, N_CONF, seed=2024, g3=G3)
lanc = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, LF),
                              window_design="minimal_image",
                              window_origins=2)
lanc.ignore_v4 = True     # pure-cubic toy: match the bubble diagram content
lanc.init(use_symmetries=True)

# energy grid
w_max = 1.25 * np.max(lanc.w_q)
energies = np.linspace(1e-6, w_max, 1200)

# probe: interpolated q-points (not on the coarse mesh) and bands
probes = [(2, 4), (2, 5), (6, 4), (6, 5)]   # (n_z on the fine mesh, band)

results = {"energies_cm": (energies * RY_TO_CM).tolist(), "probes": []}
bg = unit.get_reciprocal_vectors() / (2 * np.pi)

for n_z, band in probes:
    iq = int(np.where((lanc._fine_idx == [0, 0, n_z]).all(axis=1))[0][0])
    q = lanc.q_points[iq]

    # --- TDSCHA interpolated Lanczos ---
    lanc.prepare_mode_q(iq, band)
    lanc.run_FT(N_STEPS, verbose=False)
    gf = lanc.get_green_function_continued_fraction(
        energies, use_terminator=False, smearing=SMEAR_RY)
    a_lanc = -np.imag(gf) / np.pi

    # --- standard d3 bubble (Spectral.py) on the same k mesh ---
    spectralf, z, z_pert, w_q_bub = CC.Spectral.get_diag_dynamic_bubble(
        t2, t3, k_grid=(1, 1, LF), q=np.asarray(q),
        smear_id=np.array([SMEAR_RY]), smear=np.array([SMEAR_RY]),
        energies=energies, T=T)
    # mode matching by frequency (both sorted ascending by eigh)
    a_bub = spectralf[:, band, 0]

    # peak positions & widths
    def peak_info(a):
        i0 = int(np.argmax(a))
        return energies[i0] * RY_TO_CM
    results["probes"].append({
        "n_z": n_z, "band": band,
        "q_frac": float(n_z) / LF,
        "w_sscha_cm": float(lanc.w_q[band, iq] * RY_TO_CM),
        "w_bub_cm": float(w_q_bub[band] * RY_TO_CM),
        "lanczos": a_lanc.tolist(),
        "bubble": a_bub.tolist(),
        "peak_lanczos_cm": peak_info(a_lanc),
        "peak_bubble_cm": peak_info(a_bub),
    })
    print("PROBE q=%d/%d band=%d: w_SSCHA=%.2f  peak_lanc=%.2f  peak_bub=%.2f cm-1"
          % (n_z, LF, band, lanc.w_q[band, iq] * RY_TO_CM,
             results["probes"][-1]["peak_lanczos_cm"],
             results["probes"][-1]["peak_bubble_cm"]))
    sys.stdout.flush()

with open("bubble_compare.json", "w") as f:
    json.dump(results, f)
print("DONE")
