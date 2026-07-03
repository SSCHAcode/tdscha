"""ASR-design figures. Palette (validated, extends make_figs.py): plain
yellow #eda100, minimal-image aqua #1baf7a, asr rose #c93a63, reference
blue #2a78d6. Identity never color-alone: direct labels + line styles."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")
DATA = os.path.join(HERE, "..", "data")

C_PLAIN = "#eda100"
C_MIMG = "#1baf7a"
C_ASR = "#c93a63"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e0", "grid.linewidth": 0.5,
    "legend.frameon": False,
})

d = json.load(open(os.path.join(DATA, "asr_data.json")))

# ------------------------------------------------- kernel heatmaps (L=3)
L, S = 3, 6
tgt_mi = np.array(d["kernels_L3"]["target_mi"]) / L
tgt_pr = np.array(d["kernels_L3"]["target_proj"]) / L
fitted = np.array(d["kernels_L3"]["fitted_asr"]) / L

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.85), constrained_layout=True)
ext = [-(S - 1) - .5, S - 1 + .5, -(S - 1) - .5, S - 1 + .5]
vmax = np.max(np.abs(tgt_mi))
for ax, K, title in zip(
        axes, (tgt_mi, tgt_pr, fitted),
        ("minimal-image target\n(violates ASR)",
         "ASR-projected target\n(one-shot projection)",
         "fitted, uniform class sums\n($K=3$, support $2L$)")):
    ax.imshow(K.T, origin="lower", extent=ext, cmap="RdBu_r",
              vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(r"$\delta_1$ (cells)")
    ax.grid(False)
    for i in range(2 * S - 1):
        for j in range(2 * S - 1):
            if tgt_mi[i, j] > 1e-10:
                ax.plot(i - (S - 1), j - (S - 1), "x", ms=3.0,
                        color="#666666", mew=0.7)
axes[0].set_ylabel(r"$\delta_2$ (cells)")
fig.savefig(os.path.join(FIGS, "asr_kernel.pdf"))
plt.close(fig)

# --------------------------------------- leak: deterministic + stochastic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                               constrained_layout=True)

qs = np.array(d["leak_q"])
scale = d["phi_scale"]
ax1.plot(qs, np.array(d["leak"]["mimg"]) / scale, color=C_MIMG, lw=1.8)
ax1.plot(qs, np.array(d["leak"]["plain"]) / scale, color=C_PLAIN, lw=1.8,
         ls="--")
ax1.plot(qs, np.array(d["leak"]["asr"]) / scale, color=C_ASR, lw=1.8,
         ls=":")
ax1.text(0.155, 0.82, "minimal-image", color=C_MIMG, fontsize=8.5,
         ha="center")
ax1.text(0.405, 0.07, "plain and ASR design:\nexactly 0", color=C_ASR,
         fontsize=8.0, ha="center")
ax1.set_xlabel(r"$\tilde q_1$ (r.l.u.)")
ax1.set_ylabel(r"$|T(\tilde q_1)| / \max|\phi|$")
ax1.set_title("kernel-level leak (deterministic)")

zb = json.load(open(os.path.join(DATA, "asr_zb.json")))
rows = zb["rows"]
ns = sorted(rows, key=lambda k: int(k))
q2 = np.array([rows[n]["q2"] for n in ns])
keep = [n for n in ns if rows[n]["q2"] <= 0.19]
q2 = np.array([rows[n]["q2"] for n in keep])
for key, c, ls in (("plain", C_PLAIN, "--"), ("mimg", C_MIMG, "-"),
                   ("asr", C_ASR, ":")):
    v = np.array([rows[n][key] for n in keep]) * 1e6
    ax2.plot(q2, v, color=c, ls=ls, lw=1.8, marker="o", ms=3.5)
ax2.text(0.021, 8.3, "minimal-image\n(plateau: ASR leak)", color=C_MIMG,
         fontsize=8.0)
ax2.text(0.055, 3.1, "ASR design", color=C_ASR, fontsize=8.5,
         rotation=38, rotation_mode="anchor")
ax2.text(0.083, 3.6, "plain", color=C_PLAIN, fontsize=8.5,
         rotation=33, rotation_mode="anchor")
ax2.set_xlabel(r"acoustic $\tilde q_2$ (r.l.u.)")
ax2.set_ylabel(r"$|\langle \partial^2 V\rangle|$ acoustic row ($10^{-6}$)")
ax2.set_title(r"stochastic estimator, $q_p = 1/2$ (three-body toy)")
ax2.set_xlim(0, 0.185)
ax2.set_ylim(0, None)
fig.savefig(os.path.join(FIGS, "asr_leak.pdf"))
plt.close(fig)
print("DONE")
