"""SnTe benchmark figure for the report: spectral function comparison.

Data sources:
  report/interpolation/data/snte_reference_4x4x4_converged_sscha_lanczos_N4000_s250.dat
      — converged 4x4x4 SSCHA native q-Lanczos (reference)
  SnTe_FF/Spectral/TDSCHA_Interpolate/direct_444.dat
      — matched 4x4x4 control (interpolated 2³ auxiliary, NOT converged)
  fix_tensor_d3.dat           — tensor-D3 oracle (interp 2x2x2->4x4x4)
  atomic_delta_perm_asr_N1000_s30.dat  — atomic_delta (THIS WORK)
  atomic_perm_asr_N1000_s30.dat        — atomic raw permutation-correct
  plain_asr_N500_s30.dat               — plain (broken baseline)
  old_gamma_noLOTO.dat                 — old d3 bubble (k=20^3, sm=1.5)

Palette (validated from make_figs.py):
  converged 4³ ref  #111111  black (true reference)
  direct/control    #2a78d6  blue (matched control)
  atomic_delta      #d9361e  red (the new validated method)
  tensor-D3         #1baf7a  green-aqua
  atomic raw        #4a3aa7  violet (permutation-correct, no ASR)
  plain             #eda100  yellow
  old bubble        #999999  grey dash
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/darth-vader/Dropbox/MEGA/Research/Simulations/SnTe_FF/Spectral/TDSCHA_Interpolate"
REPO_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")

C_CONV = "#111111"   # converged 4³ SSCHA reference
C_DIR = "#2a78d6"    # matched 4x4x4 control
C_DELTA = "#d9361e"  # atomic_delta (new validated method)
C_TENSOR = "#1baf7a"  # tensor-D3 oracle
C_ATOMIC = "#4a3aa7"  # atomic raw
C_PLAIN = "#eda100"   # plain broken
C_BUB = "#999999"     # old d3 bubble

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e0", "grid.linewidth": 0.5,
    "legend.frameon": False,
    "text.usetex": False,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(filename, data_dir=None):
    if data_dir is None:
        data_dir = ROOT
    d = np.loadtxt(os.path.join(data_dir, filename))
    return d[:, 0], d[:, 1]  # energy_cm, spectral

# Converged 4³ SSCHA reference (true physical reference)
e_conv, s_conv = load_data(
    "snte_reference_4x4x4_converged_sscha_lanczos_N4000_s250.dat",
    REPO_DATA)

# Matched 4³ control (interpolated 2³ auxiliary, NOT converged)
e_dir, s_dir = load_data("direct_444.dat")

e_fix, s_fix = load_data("fix_tensor_d3.dat")
e_ad, s_ad = load_data("atomic_delta_perm_asr_N1000_s30.dat")
e_at, s_at = load_data("atomic_perm_asr_N1000_s30.dat")
e_pl, s_pl = load_data("plain_asr_N500_s30.dat")
# old_gamma_noLOTO.dat has 3 columns: energy, TO_spectral, total_spectral
d_bub = np.loadtxt(os.path.join(ROOT, "old_gamma_noLOTO.dat"))
e_bub = d_bub[:, 0]
s_bub = d_bub[:, 1]  # TO spectral

def peak(e, s, lo=15, hi=70):
    m = (e > lo) & (e < hi)
    return e[m][np.argmax(s[m])]

print("Peaks (cm-1):")
print("  converged 4³ SSCHA ref: %.2f" % peak(e_conv, s_conv))
print("  matched 4³ control:     %.2f" % peak(e_dir, s_dir))
print("  tensor-D3 oracle:       %.2f" % peak(e_fix, s_fix))
print("  atomic_delta:           %.2f" % peak(e_ad, s_ad))
print("  atomic raw:             %.2f" % peak(e_at, s_at))
print("  plain broken:           %.2f" % peak(e_pl, s_pl))
print("  old bubble:             %.2f" % peak(e_bub, s_bub))

# ---------------------------------------------------------------------------
# Main figure: all spectral functions
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)

# Area-normalize everything
def norm(s, e):
    return s / np.trapz(s, e)

# Converged 4³ SSCHA reference (primary, shaded background)
ax.fill_between(e_conv, norm(s_conv, e_conv), alpha=0.22, color=C_CONV, lw=0,
                label=r"converged $4^3$ SSCHA, native q-Lanczos (reference)")
ax.plot(e_conv, norm(s_conv, e_conv), color=C_CONV, lw=2.8, alpha=0.7)

# Matched 4³ control (interpolated auxiliary -- algorithm validation only)
ax.plot(e_dir, norm(s_dir, e_dir), color=C_DIR, lw=1.6, ls="--",
        label=r"matched $4^3$ control (interp.\ auxiliary)")

# atomic_delta (the new validated method - thick, prominent)
ax.plot(e_ad, norm(s_ad, e_ad), color=C_DELTA, lw=2.2,
        label=r"\texttt{atomic\_delta} (this work, $N_I{=}10^3$, 30 steps)")

# tensor-D3 oracle
ax.plot(e_fix, norm(s_fix, e_fix), color=C_TENSOR, lw=1.6, ls=(0, (5, 2)),
        label=r"tensor-D3 oracle (same $\Phi^{(3)}$ as bubble)")

# atomic raw (dotted, for comparison)
ax.plot(e_at, norm(s_at, e_at), color=C_ATOMIC, lw=1.2, ls=(0, (3, 2)),
        label=r"\texttt{atomic} raw (same, no ASR projector)")

# plain broken
ax.plot(e_pl, norm(s_pl, e_pl), color=C_PLAIN, lw=1.4, ls=":",
        label=r"\texttt{plain} window (broken interpolation)")

# old bubble
ax.plot(e_bub, norm(s_bub, e_bub), color=C_BUB, lw=1.2, ls=(0, (5, 3)),
        label=r"old $d_3$ bubble ($k{=}20^3$, $F_{\rm far}{=}3$)")

# SSCHA harmonic position (2³ auxiliary, the starting point for all methods)
sscha_2 = 54.0
ax.axvline(sscha_2, color="#777777", lw=1.0, ls="--")
ax.annotate(r"SSCHA $\omega^{\rm TO}_\Gamma$ ($2^3$) = 54.0 cm$^{-1}$",
            xy=(sscha_2, 0.92), xytext=(57.0, 0.95),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
            color="#666666", fontsize=8.0, ha="left")

ax.set_xlim(15, 65)
ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax.set_ylabel("spectral function (area-normalized)")
ax.legend(loc="upper left", fontsize=6.5, ncol=1)

# Title
ax.set_title(
    r"SnTe $\Gamma$ TO spectral function --- $2^3\rightarrow4^3$ interpolation, $D_3$-only, "
    r"$T=280$ K, $\sigma=1.5$ cm$^{-1}$",
    fontsize=9.0, pad=6)

fig.savefig(os.path.join(FIGS, "snte_spectral.pdf"), dpi=150)
plt.close(fig)
print("\nsnte_spectral.pdf done")

# ---------------------------------------------------------------------------
# Second figure: zoom on the TO peak region (30-46 cm-1), normalized
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)

# Converged 4³ SSCHA reference
ax2.fill_between(e_conv, norm(s_conv, e_conv), alpha=0.25, color=C_CONV, lw=0,
                 label=r"converged $4^3$ SSCHA (reference)")
ax2.plot(e_conv, norm(s_conv, e_conv), color=C_CONV, lw=2.4, alpha=0.7)

# Matched 4³ control
ax2.plot(e_dir, norm(s_dir, e_dir), color=C_DIR, lw=1.6, ls="--",
         label=r"matched $4^3$ control")

# atomic_delta
ax2.plot(e_ad, norm(s_ad, e_ad), color=C_DELTA, lw=2.2,
         label=r"\texttt{atomic\_delta}")

# tensor-D3 oracle
ax2.plot(e_fix, norm(s_fix, e_fix), color=C_TENSOR, lw=1.6, ls=(0, (5, 2)),
         label=r"tensor-D3 oracle")

# atomic raw
ax2.plot(e_at, norm(s_at, e_at), color=C_ATOMIC, lw=1.2, ls=(0, (3, 2)),
         label=r"\texttt{atomic} raw")

# old bubble
ax2.plot(e_bub, norm(s_bub, e_bub), color=C_BUB, lw=1.2, ls=(0, (5, 3)),
         label=r"old $d_3$ bubble")

# Mark peak positions
peaks_to_mark = [
    (e_conv, s_conv, "34.79", C_CONV, 0.88),
    (e_dir, s_dir, "37.56", C_DIR, 0.78),
    (e_ad, s_ad, "37.52", C_DELTA, 0.68),
    (e_fix, s_fix, "37.65", C_TENSOR, 0.58),
]
for ee, ss, label, color, yoff in peaks_to_mark:
    pk = peak(ee, ss, 30, 46)
    ax2.axvline(pk, color=color, lw=0.7, ls=":", alpha=0.6)
    ax2.annotate(label, xy=(pk, 0), xytext=(pk + 0.15, yoff),
                 textcoords=("data", "axes fraction"),
                 color=color, fontsize=7.0, rotation=90, va="bottom")

ax2.set_xlim(30, 46)
ax2.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax2.set_ylabel("spectral function (norm.)")
ax2.legend(loc="upper left", fontsize=6.0, ncol=1)
ax2.set_title("SnTe $\\Gamma$ TO --- zoom on the non-Lorentzian peak", fontsize=9.0)

fig2.savefig(os.path.join(FIGS, "snte_zoom.pdf"), dpi=150)
plt.close(fig2)
print("snte_zoom.pdf done")

# ---------------------------------------------------------------------------
# Third figure: ASR diagnostic -- near-Gamma acoustic D3 vertex decay
# ---------------------------------------------------------------------------
asr_at = np.loadtxt(os.path.join(ROOT, "atomic_perm_asr_N1000_s30_asr.dat"))
asr_ad = np.loadtxt(os.path.join(ROOT, "atomic_delta_perm_asr_N1000_s30_asr.dat"))

fig3, ax3 = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)

# Filter out near-zero q (q=0, machine precision)
mask_at = asr_at[:, 0] > 1e-8
mask_ad = asr_ad[:, 0] > 1e-8

ax3.semilogy(asr_at[mask_at, 0], asr_at[mask_at, 1], "o-", ms=5, lw=1.4,
             color=C_ATOMIC, label=r"\texttt{atomic} raw (no ASR projector)")
ax3.semilogy(asr_ad[mask_ad, 0], asr_ad[mask_ad, 1], "s-", ms=5, lw=1.4,
             color=C_DELTA, label=r"\texttt{atomic\_delta} (with ASR projector)")

# Annotate the improvement factor
improvement = np.median(asr_at[mask_at, 1] / (asr_ad[mask_ad, 1] + 1e-40))
ax3.annotate(r"$\sim {} \times$ smaller".format(int(improvement)),
             xy=(0.5, 0.5), xycoords="axes fraction",
             ha="center", va="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffdd",
                        edgecolor="#ccccaa", alpha=0.9))

ax3.set_xlabel(r"$|\mathbf{q}_{\rm ac}|$ (\AA$^{-1}$)")
ax3.set_ylabel(r"$\|D_3^{\rm ac}\|$ (acoustic-band row norm)")
ax3.legend(loc="upper left", fontsize=8.5)
ax3.set_title(
    r"Acoustic $D_3$ vertex decay near $\Gamma$ "
    r"--- $4^3$ mesh, $q_{\rm pert}=\Gamma$",
    fontsize=9.0)

fig3.savefig(os.path.join(FIGS, "snte_asr.pdf"), dpi=150)
plt.close(fig3)
print("snte_asr.pdf done")
print("ALL DONE")
