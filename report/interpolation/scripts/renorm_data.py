"""Per-mode renormalization data for the report figure:
direct (two seeds), plain interp, minimal-image interp (3 origins).
Writes renorm_data.json."""
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

dync, dynf = TC.build_dyn(Lc), TC.build_dyn(Lf)
ensA = TC.make_ensemble(dynf, T, N, seed=101, g3=G3)
ensB = TC.make_ensemble(dynf, T, N, seed=202, g3=G3)
ensc = TC.make_ensemble(dync, T, N, seed=303, g3=G3)

objs = {
    "directA": QL.QSpaceLanczos(ensA, lo_to_split=None),
    "directB": QL.QSpaceLanczos(ensB, lo_to_split=None),
    "plain": QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, Lf), window_design="plain"),
    "mimg": QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, Lf),
                                   window_design="minimal_image", window_origins=3),
}
for o in objs.values():
    o.init(use_symmetries=True)

ref = objs["plain"]  # fine-mesh q indexing/labels

def renorms(lanc):
    out = {}
    for iq in range(lanc.n_q):
        for band in range(lanc.n_bands):
            if not lanc.valid_modes_q[band, iq]:
                continue
            lanc.prepare_mode_q(iq, band)
            lanc.run_FT(N_STEPS, verbose=False)
            key = "%.6f_%d" % (round(lanc.q_points[iq][2], 6), band)
            out[key] = (TC.lanczos_effective_freq(lanc) - lanc.w_q[band, iq]) * RY_TO_CM
    return out

data = {"meta": {"T": T, "N": N, "g3": G3, "Lc": Lc, "Lf": Lf}}
for name, o in objs.items():
    data[name] = renorms(o)
    print("done", name)
    sys.stdout.flush()

# mode labels: q_frac (n_z/Lf) per key on the fine mesh
labels = {}
for iq in range(ref.n_q):
    key_q = "%.6f" % round(ref.q_points[iq][2], 6)
    labels[key_q] = int(ref._fine_idx[iq][2])
data["nz_of_q"] = labels
data["on_coarse"] = [int(nz * Lc) % Lf == 0 for nz in range(Lf)]

with open("renorm_data.json", "w") as f:
    json.dump(data, f, indent=1)
print("DONE")
