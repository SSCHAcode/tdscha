"""asr-design renorm error on the ORIGINAL pairwise toy (M3 protocol):
Lc=3 -> Lf=6, N=4000, g3=0.1, no three-body term. Compares directly with
the report table: floor 0.103, plain 0.299, mimg+3origins 0.122."""
import os, sys, json
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "test_interpolation"))
import numpy as np
import _toy_chain as TC
import cellconstructor as CC
import tdscha.QSpaceLanczos as QL
import tdscha.QSpaceInterpolation as QI

T, N, G3, Lc, Lf, N_STEPS = 300.0, 4000, 0.1, 3, 6, 35
RY_TO_CM = CC.Units.RY_TO_CM

objs = {
    "directA": QL.QSpaceLanczos(
        TC.make_ensemble(TC.build_dyn(Lf), T, N, seed=101, g3=G3),
        lo_to_split=None),
    "asr3": QI.QSpaceLanczosInterp(
        TC.make_ensemble(TC.build_dyn(Lc), T, N, seed=303, g3=G3),
        fine_mesh=(1, 1, Lf), window_design="asr", window_origins=3),
    "asr_decay3": QI.QSpaceLanczosInterp(
        TC.make_ensemble(TC.build_dyn(Lc), T, N, seed=303, g3=G3),
        fine_mesh=(1, 1, Lf), window_design="asr", window_decay=0.4,
        window_origins=3),
}
for o in objs.values():
    o.init(use_symmetries=True)

def renorms(lanc):
    out = {}
    for iq in range(lanc.n_q):
        for band in range(lanc.n_bands):
            if not lanc.valid_modes_q[band, iq]:
                continue
            lanc.prepare_mode_q(iq, band)
            lanc.run_FT(N_STEPS, verbose=False)
            key = "%.6f_%d" % (round(lanc.q_points[iq][2], 6), band)
            try:
                out[key] = (TC.lanczos_effective_freq(lanc)
                            - lanc.w_q[band, iq]) * RY_TO_CM
            except AssertionError:
                out[key] = None
    return out

data = {}
for name, o in objs.items():
    data[name] = renorms(o)
    print("done", name); sys.stdout.flush()

ref = data["directA"]
for name in ("asr3", "asr_decay3"):
    errs = [data[name][k] - ref[k] for k in ref
            if data[name].get(k) is not None]
    nf = sum(1 for k in ref if data[name].get(k) is None)
    print("%-11s aggregate %.4f cm^-1  (%d failed)"
          % (name, float(np.sqrt(np.mean(np.array(errs) ** 2))), nf))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench_asr_pairwise.json"), "w") as f:
    json.dump(data, f, indent=1)
print("DONE")
