"""Report figures. Palette (validated): direct/reference blue #2a78d6,
plain window yellow #eda100, minimal-image aqua #1baf7a, bubble violet
#4a3aa7. Identity is never color-alone: direct labels/markers everywhere."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")

C_DIR = "#2a78d6"   # direct reference
C_PLAIN = "#eda100" # plain window
C_MIMG = "#1baf7a"  # minimal-image windows
C_BUB = "#4a3aa7"   # d3 bubble
GRID = dict(color="#dddddd", lw=0.6)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e0", "grid.linewidth": 0.5,
    "legend.frameon": False,
})

# ----------------------------------------------------------------- kernels
import tdscha.QSpaceInterpolation as QI
L = 4
target = QI.minimal_image_target_1d(L) / L
plain = QI._kernel_sym(np.ones(L), np.ones(L), np.ones(L)) / L
passes = QI.get_window_design(L, K=2)
designed = sum(QI._kernel_sym(*p) for p in passes) / L

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
vmax = 1.0
ext = [-(L - 1) - .5, L - 1 + .5, -(L - 1) - .5, L - 1 + .5]
for ax, K, title in zip(axes, (plain, target, designed),
                        ("plain (tent)", "minimal-image target",
                         "fitted, $K=2$")):
    im = ax.imshow(K.T, origin="lower", extent=ext, cmap="Blues",
                   vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(r"$\delta_1$ (cells)")
    ax.grid(False)
    # mark nonzero target weights
    for i in range(2 * L - 1):
        for j in range(2 * L - 1):
            if target[i, j] > 1e-10:
                ax.plot(i - (L - 1), j - (L - 1), "x", ms=3.5,
                        color="#666666", mew=0.8)
axes[0].set_ylabel(r"$\delta_2$ (cells)")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label("kernel weight $S/L$")
fig.savefig(os.path.join(FIGS, "kernel.pdf"))
plt.close(fig)
print("kernel.pdf done: designed vs target RMS =",
      float(np.sqrt(np.mean((designed - target) ** 2))))

# ----------------------------------------------------------------- scaling
with open("bench_scaling.json") as f:
    bench = json.load(f)
fig, ax = plt.subplots(figsize=(3.9, 3.0), constrained_layout=True)
for key, color, label, marker in (("plain", C_PLAIN, "plain window", "o"),
                                  ("mimg", C_MIMG, "minimal-image ($K{=}2$)", "s")):
    Lfs = np.array([d["Lf"] for d in bench[key]])
    ts = np.array([d["t_L"] for d in bench[key]])
    ax.plot(Lfs, ts, marker=marker, ms=4.5, lw=1.6, color=color, label=label)
    ax.annotate(label, xy=(Lfs[-1], ts[-1]), xytext=(-4, 6),
                textcoords="offset points", ha="right", color="#333333")
# ideal linear guide through the first plain point
L0, t0 = bench["plain"][0]["Lf"], bench["plain"][0]["t_L"]
xs = np.array([8, 128.0])
ax.plot(xs, t0 * xs / L0, ls=":", lw=1.0, color="#999999")
ax.annotate("linear", xy=(xs[-1], t0 * xs[-1] / L0), xytext=(-2, -11),
            textcoords="offset points", ha="right", color="#999999")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel(r"fine mesh size $N_f$")
ax.set_ylabel(r"wall time per $\mathcal{L}$ application (s)")
ax.legend(loc="upper left")
fig.savefig(os.path.join(FIGS, "scaling.pdf"))
plt.close(fig)
print("scaling.pdf done")

# ----------------------------------------------------------------- renorm
with open("renorm_data.json") as f:
    rd = json.load(f)
Lf = rd["meta"]["Lf"]; Lc = rd["meta"]["Lc"]
nz_of_q = rd["nz_of_q"]

def nz_of_key(k):
    return int(nz_of_q["%.6f" % float(k.split("_")[0])])

keys = sorted(rd["directA"].keys(),
              key=lambda k: (nz_of_key(k), int(k.split("_")[1])))
xs = np.arange(len(keys))
dA = np.array([rd["directA"][k] for k in keys])
dB = np.array([rd["directB"][k] for k in keys])
pl = np.array([rd["plain"][k] for k in keys])
mi = np.array([rd["mimg"][k] for k in keys])
nzs = np.array([nz_of_key(k) for k in keys])

fig, ax = plt.subplots(figsize=(7.0, 3.1), constrained_layout=True)
ticks, ticklabels = [], []
for nz in sorted(set(nzs)):
    sel = np.where(nzs == nz)[0]
    if (nz * Lc) % Lf != 0:   # interpolated (non-commensurate) q: shade
        ax.axvspan(sel.min() - 0.5, sel.max() + 0.5, color="#eeeeea", zorder=0)
    ticks.append(0.5 * (sel.min() + sel.max()))
    ticklabels.append("$%d/%d$" % (nz, Lf) if nz else "$\\Gamma$")
ax.fill_between(xs, np.minimum(dA, dB), np.maximum(dA, dB),
                color=C_DIR, alpha=0.30, lw=0,
                label="direct fine supercell (seed spread)")
ax.plot(xs, 0.5 * (dA + dB), color=C_DIR, lw=1.6)
ax.plot(xs, pl, "o", ms=4, color=C_PLAIN, label="interp, plain window")
ax.plot(xs, mi, "s", ms=4, color=C_MIMG,
        label="interp, minimal-image (3 origins)")
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)
ax.set_xlabel("wavevector $q$ (bands grouped per $q$; shaded = interpolated $q$)")
ax.set_ylabel(r"renormalization $\omega_{\rm eff}-\omega_{\rm SSCHA}$ (cm$^{-1}$)")
ax.legend(loc="lower right", ncol=1)
fig.savefig(os.path.join(FIGS, "renorm.pdf"))
plt.close(fig)
print("renorm.pdf done")

# ----------------------------------------------------------------- bubble
with open("bubble_compare.json") as f:
    bb = json.load(f)
en = np.array(bb["energies_cm"])
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=False,
                         constrained_layout=True)
for ax, pr in zip(axes.ravel(), bb["probes"]):
    al = np.array(pr["lanczos"]); ab = np.array(pr["bubble"])
    # normalize to unit maximum for shape comparison
    al = al / np.max(np.abs(al)); ab = ab / np.max(np.abs(ab))
    ax.plot(en, ab, color=C_BUB, lw=2.4, alpha=0.55,
            label="$d_3$ bubble (Spectral.py)")
    ax.plot(en, al, color=C_MIMG, lw=1.3,
            label="TDSCHA Lanczos (interp)")
    ax.axvline(pr["w_sscha_cm"], color="#999999", lw=0.9, ls="--")
    ax.annotate("SSCHA", xy=(pr["w_sscha_cm"], 0.97), rotation=90,
                ha="right", va="top", color="#777777", fontsize=7.5)
    w0 = pr["w_sscha_cm"]
    ax.set_xlim(w0 - 90, w0 + 45)
    ax.set_title(r"$q = %d/16$, band %d" % (pr["n_z"], pr["band"]),
                 fontsize=9)
    ax.set_ylabel("spectral function (norm.)")
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
axes[0, 0].legend(loc="upper left", fontsize=7.5)
fig.savefig(os.path.join(FIGS, "bubble.pdf"))
plt.close(fig)
print("bubble.pdf done")

# ------------------------------------------------------ D4 spectral function
with open("d4_spectral.json") as f:
    d4 = json.load(f)
en = np.array(d4["energies_cm"])
probes = d4["probes"]
LF = d4["meta"]["LF"]
fig, axes = plt.subplots(1, len(probes), figsize=(7.0, 2.7),
                         constrained_layout=True)
for ax, pr in zip(np.atleast_1d(axes), probes):
    a_dir = np.array(pr["direct_full"])
    a_int = np.array(pr["interp_full"])
    a_i3 = np.array(pr["interp_d3only"])
    s = 1.0 / np.max(a_dir)
    # reference: direct fine supercell, full D3+D4
    ax.plot(en, a_dir * s, color=C_DIR, lw=2.6, alpha=0.5,
            label="direct fine ($D_3{+}D_4$)")
    # interpolated prediction, full
    ax.plot(en, a_int * s, color=C_MIMG, lw=1.3,
            label="interp ($D_3{+}D_4$)")
    # interpolated but with D4 dropped -> displaced (D4 is fundamental)
    ax.plot(en, a_i3 * s, color="#b0392b", lw=1.1, ls=(0, (4, 2)),
            label="interp ($D_3$ only)")
    ax.axvline(pr["w_sscha_cm"], color="#999999", lw=0.9, ls="--")
    ax.annotate("SSCHA", xy=(pr["w_sscha_cm"], 0.98), rotation=90,
                ha="right", va="top", color="#777777", fontsize=7.5)
    p3, pf = pr["peak_direct_d3only"], pr["peak_direct_full"]
    lo = min(p3, pf, pr["w_sscha_cm"])
    ax.set_xlim(lo - 60, pr["w_sscha_cm"] + 40)
    ax.set_title(r"$q=%d/%d$, band %d" % (pr["n_z"], LF, pr["band"]),
                 fontsize=9)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
np.atleast_1d(axes)[0].set_ylabel("spectral function (norm.)")
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7.0)
fig.savefig(os.path.join(FIGS, "d4_spectral.pdf"))
plt.close(fig)
print("d4_spectral.pdf done")
