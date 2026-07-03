"""Deterministic kernel-level ASR data for the report (no ensembles).

Writes asr_data.json with:
  - projected-target distances (irreducible ASR cost) for L=3,4,6 and
    supports 2L,3L,4L;
  - realization RMS of the fitted asr designs;
  - deterministic acoustic-leak curves |T(q)| for plain / minimal-image /
    asr / asr+decay kernels contracted with the exact three-body test
    tensor (V = g s1^2 s2 on an A-atom triplet), acoustic leg q2 -> 0 at
    fixed q1 = -q (the q_pert = 0 pair line is leak-free by phase
    cancellation; the generic-q1 line is the dangerous one);
  - the kernels themselves for the heatmap figure (L=3).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "Modules"))
import QSpaceInterpolation as QI

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "data", "asr_data.json")
data = {}

# ---- irreducible ASR cost vs support --------------------------------
proj = {}
for L in (3, 4, 6):
    for mult in (2, 3):
        S = mult * L
        tgt = QI.embed_minimal_image_target_1d(L, S)
        p = QI.asr_projected_target_1d(L, S)
        proj["L%d_S%d" % (L, S)] = float(np.sqrt(np.mean((p - tgt) ** 2)))
data["proj_rms"] = proj

# ---- realization RMS of the fitted designs --------------------------
real = {}
for L in (3, 4):
    S = 2 * L
    passes = QI.get_window_design_asr(L, K=3)
    Kt = sum(QI._kernel_sym(*p) for p in passes)
    tgt = QI.asr_projected_target_1d(L, S)
    real["L%d" % L] = float(np.sqrt(np.mean((Kt - tgt) ** 2)))
data["realization_rms"] = real

# ---- three-body test tensor (L = 3) ---------------------------------
L = 3
Phi = np.zeros((L, L, L))
d1v = np.array([-1.0, 1.0, 0.0])
d2v = np.array([0.0, -1.0, 1.0])
for n in range(L):
    idx = [(n + m) % L for m in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Phi[idx[i], idx[j], idx[k]] += 2.0 * (
                    d1v[i] * d1v[j] * d2v[k] + d1v[i] * d2v[j] * d1v[k]
                    + d2v[i] * d1v[j] * d1v[k])
phi = np.zeros((L, L))
for d1 in range(L):
    for d2 in range(L):
        phi[d1, d2] = Phi[0, d1, d2]

S = 2 * L
plain6 = np.zeros(S); plain6[:L] = 1.0
K_plain = QI._kernel_sym(plain6, plain6, plain6)

def embed(K):
    n = 2 * S - 1
    out = np.zeros((n, n)); m = K.shape[0]
    off = (n - m) // 2
    out[off:off + m, off:off + m] = K
    return out

K_mimg = embed(sum(QI._kernel_sym(*p) for p in QI.get_window_design(L, K=3)))
K_asr = sum(QI._kernel_sym(*p) for p in QI.get_window_design_asr(L, K=3))
K_asrd = sum(QI._kernel_sym(*p)
             for p in QI.get_window_design_asr(L, K=3, decay_weight=0.4))

def leak_curve(Ktot):
    """|T(q)|: v-leg translation contraction of the effective tensor at
    q1 = -q (acoustic partner q2 -> 0 of a perturbation at finite q)."""
    n = 2 * S - 1
    qs = np.linspace(0.0, 0.5, 101)
    out = []
    for q in qs:
        t = 0.0 + 0j
        for i in range(n):
            for j in range(n):
                d1 = i - (S - 1)
                t += (Ktot[i, j] * phi[d1 % L, (j - (S - 1)) % L]
                      * np.exp(-2j * np.pi * q * d1))
        out.append(abs(t) / L)
    return qs.tolist(), out

curves = {}
for name, K in (("plain", K_plain), ("mimg", K_mimg), ("asr", K_asr),
                ("asr_decay", K_asrd)):
    qs, T = leak_curve(K)
    curves[name] = T
data["leak_q"] = qs
data["leak"] = curves
data["phi_scale"] = float(np.max(np.abs(phi)))

# ---- kernels for the heatmap (L=3) ----------------------------------
data["kernels_L3"] = {
    "target_mi": QI.embed_minimal_image_target_1d(L, S).tolist(),
    "target_proj": QI.asr_projected_target_1d(L, S).tolist(),
    "fitted_asr": K_asr.tolist(),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(data, f)
print("DONE ->", OUT)
