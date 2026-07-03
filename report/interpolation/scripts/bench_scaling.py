"""M4 benchmark: wall time of one L application vs fine mesh size N_f.
Also: batched vs scalar kernel speedup at fixed size.
Writes JSON for the report figures."""
import os, sys, time, json
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, "os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "test_interpolation")")
import numpy as np
import _toy_chain as TC
import tdscha.QSpaceInterpolation as QI
import tdscha.JuliaExt as JuliaExt

T, N, G3, Lc = 300.0, 1000, 0.1, 4
dync = TC.build_dyn(Lc)
ensc = TC.make_ensemble(dync, T, N, seed=42, g3=G3)

out = {"N_conf": N, "Lc": Lc, "plain": [], "mimg": []}

for Lf in (8, 16, 32, 64, 128):
    for design in ("plain", "minimal_image"):
        if design == "minimal_image" and Lf > 64:
            pass  # keep, it's linear too
        lm = QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, Lf),
                                    window_design=design)
        lm.init(use_symmetries=True)
        lm.prepare_mode_q(1, 4)
        # warm up (JIT) then time
        lm.apply_full_L()
        ts = []
        for _ in range(3):
            t0 = time.time()
            lm.apply_full_L()
            ts.append(time.time() - t0)
        key = "plain" if design == "plain" else "mimg"
        out[key].append({"Lf": Lf, "n_pairs": len(lm.unique_pairs),
                         "psi_size": lm.get_psi_size(),
                         "t_L": min(ts)})
        print("BENCH %s Lf=%3d n_pairs=%3d psi=%6d  t_L=%.3fs" % (
            design, Lf, len(lm.unique_pairs), lm.get_psi_size(), min(ts)))
        sys.stdout.flush()

# batched vs scalar at Lf=32 (plain, direct kernel calls)
lm = QI.QSpaceLanczosInterp(ensc, fine_mesh=(1, 1, 32), window_design="plain")
lm.init(use_symmetries=True)
lm.prepare_mode_q(1, 4)
jl = JuliaExt.get_main()
rs = np.random.RandomState(0)
R1 = rs.randn(lm.n_bands) + 0j
alpha1 = rs.randn(len(lm.unique_pairs) * lm.n_bands ** 2) + 0j
up = np.array(lm.unique_pairs, dtype=np.int32) + 1
vm = np.array(lm.valid_modes_q, dtype=np.bool_)
plain = (lm.X_q, lm.Y_q)
args = (plain[0], plain[1], plain[0], plain[1], plain[0], plain[1],
        lm.w_q, lm.rho, R1, alpha1, float(lm.T), True, True,
        int(lm.iq_pert) + 1, up, 1, int(lm.n_syms_qspace * lm.N), vm,
        1.0, 1.0, True)
for batched in (True, False):
    jl.get_perturb_averages_qspace_slots(*args, batched)  # warm
    t0 = time.time()
    jl.get_perturb_averages_qspace_slots(*args, batched)
    dt = time.time() - t0
    out["kernel_batched" if batched else "kernel_scalar"] = dt
    print("BENCH kernel batched=%s: %.3fs" % (batched, dt))

with open("bench_scaling.json", "w") as f:
    json.dump(out, f, indent=1)
print("BENCH speedup: %.1fx" % (out["kernel_scalar"] / out["kernel_batched"]))
