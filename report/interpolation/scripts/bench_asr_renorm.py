"""Renorm-error benchmark including the ASR window design.

Same protocol as report renorm_data.py: per-mode anharmonic renormalization
(Lanczos - SSCHA) on the Lc=3 -> Lf=6 chain, error vs a direct fine-mesh
ensemble, aggregated over all valid fine modes. Adds: asr, asr+3origins,
asr+decay. Writes bench_asr_renorm.json.
"""
import os, sys, json
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "test_interpolation"))
import numpy as np
import _toy_chain as TC
import cellconstructor as CC
import tdscha.QSpaceLanczos as QL
import tdscha.QSpaceInterpolation as QI

T, N, G3, Lc, Lf, N_STEPS = 300.0, 4000, 0.1, 3, 6, 35
G3B = 0.08
RY_TO_CM = CC.Units.RY_TO_CM

dync, dynf = TC.build_dyn(Lc), TC.build_dyn(Lf)

def coarse_ens():
    return TC.make_ensemble(TC.build_dyn(Lc), T, N, seed=303, g3=G3, g3b=G3B)

objs = {
    "directA": QL.QSpaceLanczos(TC.make_ensemble(dynf, T, N, seed=101, g3=G3,
                                                 g3b=G3B), lo_to_split=None),
    "directB": QL.QSpaceLanczos(TC.make_ensemble(dynf, T, N, seed=202, g3=G3,
                                                 g3b=G3B), lo_to_split=None),
    "plain": QI.QSpaceLanczosInterp(coarse_ens(), fine_mesh=(1, 1, Lf),
                                    window_design="plain"),
    "mimg3": QI.QSpaceLanczosInterp(coarse_ens(), fine_mesh=(1, 1, Lf),
                                    window_design="minimal_image",
                                    window_origins=3),
    "asr": QI.QSpaceLanczosInterp(coarse_ens(), fine_mesh=(1, 1, Lf),
                                  window_design="asr"),
    "asr3": QI.QSpaceLanczosInterp(coarse_ens(), fine_mesh=(1, 1, Lf),
                                   window_design="asr", window_origins=3),
    "asr_decay3": QI.QSpaceLanczosInterp(coarse_ens(), fine_mesh=(1, 1, Lf),
                                         window_design="asr",
                                         window_decay=0.4,
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
            except AssertionError as e:
                print("  FAILED mode", key, ":", e)
                out[key] = None
    return out


data = {"meta": {"T": T, "N": N, "g3": G3, "Lc": Lc, "Lf": Lf}}
for name, o in objs.items():
    data[name] = renorms(o)
    print("done", name)
    sys.stdout.flush()

# aggregate errors vs directA (and floor = directB vs directA)
ref = data["directA"]
print("\naggregate RMS error vs directA over", len(ref), "modes:")
for name in ("directB", "plain", "mimg3", "asr", "asr3", "asr_decay3"):
    errs = [data[name][k] - ref[k] for k in ref
            if data[name].get(k) is not None and ref[k] is not None]
    n_fail = sum(1 for k in ref if data[name].get(k) is None)
    agg = float(np.sqrt(np.mean(np.array(errs) ** 2)))
    data.setdefault("aggregate", {})[name] = agg
    data.setdefault("n_failed", {})[name] = n_fail
    print("  %-11s %.4f cm^-1   (%d failed modes)" % (name, agg, n_fail))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "bench_asr_renorm.json")
with open(out, "w") as f:
    json.dump(data, f, indent=1)
print("DONE ->", out)
