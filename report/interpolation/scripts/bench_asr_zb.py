"""Acoustic leak probe with FINITE q_pert (zone boundary).

The deterministic kernel curves show the mimg ASR violation vanishes when
both pair legs go to Gamma together (q_pert = 0) but is O(1) when the
acoustic leg q2 -> 0 at fixed finite q1 = q_pert - q2. Probe exactly that:
q_pert = 1/2 (fine point, incommensurate with Lc=3), pairs (q1, q2=n/Lf).
"""
import os, sys, json
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "test_interpolation"))
import numpy as np
import _toy_chain as TC
import tdscha.QSpaceInterpolation as QI

T, N, G3, G3B, Lc, Lf = 300.0, 4000, 0.02, 0.4, 3, 48
NP = Lf // 2                      # q_pert = 1/2

def make(design, **kw):
    ens = TC.make_ensemble(TC.build_dyn(Lc), T, N, seed=303, g3=G3, g3b=G3B)
    lanc = QI.QSpaceLanczosInterp(ens, fine_mesh=(1, 1, Lf),
                                  window_design=design, **kw)
    lanc.init(use_symmetries=True)
    return lanc

probe_n = [1, 2, 3, 4, 6, 8, 12, 16]

def acoustic_rows(lanc):
    iq_pert = int(lanc._q_lookup[(0, 0, NP)])
    band = int(np.argmax(np.where(lanc.valid_modes_q[:, iq_pert],
                                  lanc.w_q[:, iq_pert], -np.inf)))
    lanc.prepare_mode_q(iq_pert, band)
    R1 = lanc.get_R1_q()
    nb = lanc.n_bands
    alpha1 = lanc._flatten_blocks(
        [np.zeros((nb, nb), dtype=np.complex128) for _ in lanc.unique_pairs])
    _, d2v = lanc._call_julia_qspace(R1, alpha1)
    rows = {}
    for n in probe_n:
        iq2 = int(lanc._q_lookup[(0, 0, n % Lf)])          # acoustic leg
        iq1 = int(lanc.q_pair_map[iq2])                    # partner
        lo, hi = min(iq1, iq2), max(iq1, iq2)
        blk = d2v[lanc.unique_pairs.index((lo, hi))]
        w2 = np.where(lanc.valid_modes_q[:, iq2],
                      np.abs(lanc.w_q[:, iq2]), np.inf)
        ac = int(np.argmin(w2))
        rows[n] = (blk[ac, :] if lo == iq2 else blk[:, ac]).copy()
    return rows

rows = {}
for name, design, kw in [("plain", "plain", {}),
                         ("mimg", "minimal_image", {}),
                         ("asr", "asr", {}),
                         ("asr_decay", "asr", {"window_decay": 0.4})]:
    rows[name] = acoustic_rows(make(design, **kw))
    print("done", name); sys.stdout.flush()

print("\n%-4s %8s %12s %12s %12s %12s" % ("n", "q2", "plain", "mimg",
                                          "asr", "asr_decay"))
out = {}
for n in probe_n:
    vals = {k: float(np.linalg.norm(rows[k][n])) for k in rows}
    out[n] = dict(q2=n / Lf, **vals)
    print("%-4d %8.4f %12.4e %12.4e %12.4e %12.4e"
          % (n, n / Lf, vals["plain"], vals["mimg"], vals["asr"],
             vals["asr_decay"]))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench_asr_zb.json"), "w") as f:
    json.dump({"meta": dict(T=T, N=N, g3=G3, g3b=G3B, Lc=Lc, Lf=Lf, np=NP),
               "rows": out}, f, indent=1)
print("DONE")
