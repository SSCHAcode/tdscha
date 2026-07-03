"""Control: Lc=5 -> Lf=10 with the three-body toy.

At Lc=5 the spread-2 three-body Phi3 is interior to the WS cell (no tie
ambiguity), so centering is resolvable. If the near-Gamma static-response
failures at Lc=3 were (mimg) ASR leak and (asr) WS-boundary
under-resolution, then here mimg should still fail / be badly wrong while
asr matches the direct reference. Near-Gamma modes only.
"""
import os, sys, json
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "test_interpolation"))
import numpy as np
import _toy_chain as TC
import cellconstructor as CC
import tdscha.QSpaceLanczos as QL
import tdscha.QSpaceInterpolation as QI

T, N, G3, G3B, Lc, Lf, N_STEPS = 300.0, 4000, 0.1, 0.08, 5, 10, 35
RY_TO_CM = CC.Units.RY_TO_CM

objs = {
    "direct": QL.QSpaceLanczos(
        TC.make_ensemble(TC.build_dyn(Lf), T, N, seed=101, g3=G3, g3b=G3B),
        lo_to_split=None),
    "plain": QI.QSpaceLanczosInterp(
        TC.make_ensemble(TC.build_dyn(Lc), T, N, seed=303, g3=G3, g3b=G3B),
        fine_mesh=(1, 1, Lf), window_design="plain"),
    "mimg": QI.QSpaceLanczosInterp(
        TC.make_ensemble(TC.build_dyn(Lc), T, N, seed=303, g3=G3, g3b=G3B),
        fine_mesh=(1, 1, Lf), window_design="minimal_image"),
    "asr": QI.QSpaceLanczosInterp(
        TC.make_ensemble(TC.build_dyn(Lc), T, N, seed=303, g3=G3, g3b=G3B),
        fine_mesh=(1, 1, Lf), window_design="asr"),
}
for o in objs.values():
    o.init(use_symmetries=True)

# near-Gamma interpolated modes: n_z = 1 (incommensurate), 2 (commensurate),
# 3 (incommensurate); acoustic and top optical bands
probes = [(1, 0), (1, 5), (3, 0), (3, 5)]

data = {"meta": dict(T=T, N=N, g3=G3, g3b=G3B, Lc=Lc, Lf=Lf)}
for name, lanc in objs.items():
    out = {}
    for nz, band in probes:
        if name == "direct":
            q_ref = objs["plain"].q_points[
                int(objs["plain"]._q_lookup[(0, 0, nz)])]
            cand = [i for i in range(lanc.n_q)
                    if np.linalg.norm(lanc.q_points[i] - q_ref) < 1e-6
                    or np.linalg.norm(lanc.q_points[i] + q_ref) < 1e-6]
            iq = cand[0]
        else:
            iq = int(lanc._q_lookup[(0, 0, nz)])
        lanc.prepare_mode_q(iq, band)
        lanc.run_FT(N_STEPS, verbose=False)
        try:
            r = (TC.lanczos_effective_freq(lanc) - lanc.w_q[band, iq]) * RY_TO_CM
        except AssertionError as e:
            r = None
            print("  FAILED", name, nz, band, ":", e)
        out["%d_%d" % (nz, band)] = r
    data[name] = out
    print("done", name, out)
    sys.stdout.flush()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench_asr_L5.json"), "w") as f:
    json.dump(data, f, indent=1)
print("DONE")
