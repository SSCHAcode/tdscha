"""Report benchmark: one-phonon SPECTRAL FUNCTION in a regime where the
fourth order (D4) is fundamental, interpolated from a coarse q-mesh and
compared against the DIRECT calculation on a finer q-mesh.

Both sides are the full q-space TDSCHA Lanczos (continued-fraction Green
function, D3 + D4); the analytic d3 bubble is NOT usable here because the
whole point is a regime dominated by the quartic vertex.  The coarse
(1,1,LC) ensemble is interpolated to the fine (1,1,LF) mesh; the direct
reference is an independent (1,1,LF) ensemble of the SAME anharmonic chain.

For each interpolated q-point we store four spectra:
  direct  full  (D3+D4)  -- the reference,
  direct  D3-only        -- shows the D4-induced reshaping of the reference,
  interp  full  (D3+D4)  -- the interpolated prediction,
  interp  D3-only        -- what one would get if D4 were dropped.

The validation: interp-full overlays direct-full (D4 interpolates
correctly), while interp-D3only is displaced by ~the D4 shift (D4 is
fundamental, not a small correction).

Writes d4_spectral.json.
"""
import os, sys, json, time
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "tests", "test_interpolation"))
import numpy as np
import cellconstructor as CC
import _toy_chain as TC
import tdscha.QSpaceLanczos as QL
import tdscha.QSpaceInterpolation as QI

RY_TO_CM = CC.Units.RY_TO_CM
T = 300.0
LC, LF = 3, 9
N_CONF = 6000
N_STEPS = 110
SMEAR_RY = 3.0e-5                # ~6.6 cm-1
G3, G4 = 0.15, 2.0              # cubic softens, strong quartic hardens
# interpolated (non-commensurate with the LC=3 mesh) fine-mesh probes
PROBES = [(1, 5), (2, 5), (4, 4)]

trapz = getattr(np, "trapezoid", None) or np.trapz

t0 = time.time()
ensc = TC.make_ensemble(TC.build_dyn(LC), T, N_CONF, seed=11, g3=G3, g4=G4)
ensf = TC.make_ensemble(TC.build_dyn(LF), T, N_CONF, seed=77, g3=G3, g4=G4)

li = QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, LF),
                            window_design="minimal_image", window_origins=2)
li.init(use_symmetries=True)
ld = QL.QSpaceLanczos(ensf, lo_to_split=None)
ld.init(use_symmetries=True)

w_max = 1.3 * np.max(li.w_q)
energies = np.linspace(1e-6, w_max, 1600)


def spec(lanc, iq, band):
    lanc.prepare_mode_q(iq, band)
    lanc.run_FT(N_STEPS, verbose=False)
    gf = lanc.get_green_function_continued_fraction(
        energies, use_terminator=False, smearing=SMEAR_RY)
    return -np.imag(gf) / np.pi


def peak_cm(a):
    return float(energies[int(np.argmax(a))] * RY_TO_CM)


out = {"energies_cm": (energies * RY_TO_CM).tolist(),
       "meta": {"LC": LC, "LF": LF, "N_CONF": N_CONF, "N_STEPS": N_STEPS,
                "G3": G3, "G4": G4, "smear_cm": SMEAR_RY * RY_TO_CM},
       "probes": []}

for n_z, band in PROBES:
    iq_f = int(np.where((li._fine_idx == [0, 0, n_z]).all(axis=1))[0][0])
    iq_d = next(jq for jq in range(ld.n_q)
                if li.find_fine_q(ld.q_points[jq]) == iq_f)

    ld.ignore_v4 = False; a_dir = spec(ld, iq_d, band)
    ld.ignore_v4 = True;  a_dir3 = spec(ld, iq_d, band)
    ld.ignore_v4 = False
    li.ignore_v4 = False; a_int = spec(li, iq_f, band)
    li.ignore_v4 = True;  a_int3 = spec(li, iq_f, band)
    li.ignore_v4 = False

    def norm(a):
        return a / trapz(a, energies)

    def l1(a, b):
        return float(trapz(np.abs(norm(a) - norm(b)), energies))

    rec = {
        "n_z": n_z, "band": band, "q_frac": n_z / LF,
        "w_sscha_cm": float(li.w_q[band, iq_f] * RY_TO_CM),
        "direct_full": a_dir.tolist(),
        "direct_d3only": a_dir3.tolist(),
        "interp_full": a_int.tolist(),
        "interp_d3only": a_int3.tolist(),
        "peak_direct_full": peak_cm(a_dir),
        "peak_direct_d3only": peak_cm(a_dir3),
        "peak_interp_full": peak_cm(a_int),
        "peak_interp_d3only": peak_cm(a_int3),
        "l1_interp_vs_direct": l1(a_int, a_dir),
        "l1_d3only_vs_direct": l1(a_int3, a_dir),
    }
    out["probes"].append(rec)
    print("q=%d/%d b=%d | wSSCHA %.1f | direct[D3 %.1f -> D3+D4 %.1f] "
          "interp[D3 %.1f -> D3+D4 %.1f] | D4shift(dir) %.1f  "
          "interp-dir %.1f cm-1 | L1 full %.3f  D3only %.3f"
          % (n_z, LF, band, rec["w_sscha_cm"],
             rec["peak_direct_d3only"], rec["peak_direct_full"],
             rec["peak_interp_d3only"], rec["peak_interp_full"],
             rec["peak_direct_full"] - rec["peak_direct_d3only"],
             rec["peak_interp_full"] - rec["peak_direct_full"],
             rec["l1_interp_vs_direct"], rec["l1_d3only_vs_direct"]))
    sys.stdout.flush()

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "..", "data", "d4_spectral.json"), "w") as f:
    json.dump(out, f)
print("DONE in %.1f s" % (time.time() - t0))
