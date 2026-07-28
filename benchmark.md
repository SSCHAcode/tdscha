# SnTe trilinear q-space interpolation benchmark

All runs probe the Γ TO band at 280 K, D3-only, with LO-TO stripped and
`ignore_v4=True`. The interpolation test is 2×2×2→4×4×4 with the same
seed-1, N=1000 ensemble, 30 Lanczos steps, 1.5 cm⁻¹ broadening, and
2000-point 0–130 cm⁻¹ energy grid.

## Matched Upsilon implementation control

The saved N=4000 4×4×4 calculation used in this first diagnostic is **not an
independently converged 4³ SSCHA reference**. Its auxiliary dynamical matrix
was made with `dyn_T_280_.Interpolate((2,2,2),(4,4,4))`, and the ensemble was
then generated around that interpolated matrix. It is retained below only as
a matched algorithmic control for the Upsilon variants. It must not be used
to assess convergence with source mesh. Commands were run after installing
this checkout with the `sscha` micromamba environment.

| calculation | Upsilon / Fourier convention | TO peak (cm⁻¹) | difference vs control |
|---|---|---:|---:|
| 4³ control around interpolated 2³ auxiliary matrix | no q-space interpolation in response | **37.56** | — |
| trilinear identity 2×2×2→2×2×2 | any retained option | exact to 5–6×10⁻¹⁴ | — |
| trilinear 2×2×2→4×4×4 | coarse corner Upsilon (default) | **46.11** | +8.55 |
| trilinear 2×2×2→4×4×4 | coarse Upsilon + atom-position phase | **45.98** | +8.42 |
| trilinear 2×2×2→4×4×4 | Cartesian Upsilon Fourier-interpolated directly (removed) | **49.88** | +12.32 |
| trilinear 2×2×2→4×4×4 | direct Fourier Upsilon + atom-position phase (removed) | **49.94** | +12.38 |
| trilinear 2×2×2→4×4×4 | Upsilon rebuilt from Fourier-interpolated harmonic D (removed) | **51.25** | +13.69 |
| trilinear 4×4×4→4×4×4 identity (N=2000) | coarse corner Upsilon | 37.46 | −0.10 |
| plain windows 2×2×2→4×4×4 (old) | cell window | 52.3 | +14.7 |
| atomic windows / tensor-D3 (old fixes) | atom-centered FC3 | 37.7 / 37.6 | +0.1 / ≈0 |
| bare SSCHA | no D3 self-energy | 54.0 | +16.4 |

## Correct benchmark against a converged 4×4×4 SSCHA reference

A new 4³ SSCHA relaxation was performed at 280 K with the same SnTe toy
force field (`pbtex`, `p3=0.036475`, `p4=-0.022`, `p4x=-0.014`), 4000
configurations per population, and the standard preconditioned SSCHA
minimizer. The stopping criterion was a force-constant gradient smaller than
0.2 times its Monte-Carlo uncertainty. The run reports `converged=true`.
The interpolated 2³ starting guess was used only to initialize the
relaxation: its Γ auxiliary TO frequency is 53.985 cm⁻¹, whereas the
converged 4³ value is 47.463 cm⁻¹.

The reference spectral function was then computed with native
`QSpaceLanczos` on the converged 4³ ensemble, N=4000, seed 1, 250 Lanczos
steps, and `ignore_v4=True`. It peaks at **34.79 cm⁻¹**. An independent
standard centered-FC3 bubble calculation using the same converged auxiliary
matrix peaks at 34.99 cm⁻¹, a 0.20 cm⁻¹ cross-check of the reference.

| calculation | TO peak (cm⁻¹) | error vs converged 4³ | normalized L1 vs converged 4³ |
|---|---:|---:|---:|
| native 4³ TDSCHA Lanczos, converged 4³ SSCHA reference | **34.79** | — | — |
| standard-centered oracle 2³→4³ | **37.65** | +2.86 | 0.698 |
| original 2³, no interpolation | **44.03** | +9.23 | 1.446 |
| trilinear 2³→4³ | **46.11** | +11.32 | 1.516 |

The corrected reference confirms the systematic hardening: trilinear
interpolation moves the peak upward by 2.08 cm⁻¹ relative to the original
2³ result and therefore farther from the converged 4³ result. Standard
atom-resolved centering instead removes 6.37 cm⁻¹ of the original peak error
and lies within 2.86 cm⁻¹ of the converged reference.

![Corrected SnTe 4x4x4 spectral comparison](report/interpolation/figs/snte_trilinear4_converged_reference.png)

### Why trilinear interpolation hardens instead of softening

There is no sign reversal in the D3 self-energy. The auxiliary Γ TO
frequency is 53.985 cm⁻¹ for the converged 2³ matrix used by the
interpolation and 47.463 cm⁻¹ for the independently converged 4³ matrix.
The dynamic peak shifts are therefore

- native 2³: 53.985→44.03 cm⁻¹ (−9.96 cm⁻¹);
- trilinear 2³→4³: 53.985→46.11 cm⁻¹ (−7.88 cm⁻¹);
- native converged 4³: 47.463→34.79 cm⁻¹ (−12.67 cm⁻¹).

Thus trilinear D3 still softens the mode, but by too little. Of its
11.32 cm⁻¹ error against native 4³, 6.52 cm⁻¹ is already the change in the
mesh-converged auxiliary frequency and the remaining 4.80 cm⁻¹ is the
weaker anharmonic shift. The standard-centered 2³→4³ oracle reaches
37.65 cm⁻¹, showing that the refined two-phonon phase space would soften the
mode if the off-grid vertex were reconstructed correctly.

The vertex-level audit isolates the cause. For a coarse vertex field `V`,
the trilinear return map has the form
`V_fine = sqrt(Nc/Nf) P V`, where `P` contains the corner weights. The
reverse channel uses `P†`, so Hermiticity is exact, but `P` is not unitary.
It averages complex/sign-changing vertex amplitudes before the response
contains their squared magnitude. Constant fields retain their norm;
short-wavelength fields do not. In the simple 1D 2→4 case, the alternating
coarse component retains only one half of its squared norm. The attenuation
multiplies across the three axes.

For the actual SnTe Γ-TO vertex:

| vertex diagnostic | result |
|---|---:|
| coarse map vs native map × `sqrt(8/64)` | 1.83×10⁻¹⁴ relative error |
| fitted commensurate amplitude | +0.9979 |
| commensurate block error after fit | 3.00% |
| off-grid block error after fit | 113.4% |
| weighted `|V|²` ratio, all fine pairs | 35.45% |
| weighted `|V|²` ratio, off-grid pairs | 29.20% |

This explains the apparent wrong-sign mesh correction: standard centering
makes the 2³→4³ peak move 6.38 cm⁻¹ downward, whereas trilinear corner
averaging removes enough vertex norm that it instead moves 2.08 cm⁻¹
upward. The exact coarse-map equality also rules out a missing supercell
factor or a D3 sign error in the implementation.

[CORRECTED 2026-07-18 — see "Per-atom phase gauge" section below. The
singular-value argument was wrong (it constrains norms, not alignment),
and the tested atomic-phase implementation had a branch bug. The
corrected per-atom phase DOES repair the vertex phases (rel err² 1.12 →
0.178) but cannot repair amplitudes, and this Γ D3-only observable is
phase-blind, so the spectral conclusion below is unchanged.] A global
rescaling is also not a sound fix: it could restore one integrated norm
but not the 113% off-grid block error. A precision-preserving formulation must Fourier-pad a
properly centered atom-triplet FC3 field (the tensor oracle) or expose all
three atom slots and apply the atom-triplet centering before the stochastic
contraction.

## Upsilon and atomic-phase result

Fourier interpolation of Upsilon does not improve SnTe; it weakens the
interpolated D3 self-energy and moves the peak toward the bare SSCHA value.
Those implementation paths have therefore been removed.
The best trilinear variant remains the original coarse-corner Upsilon
prescription at 46.11 cm⁻¹. Adding the atomic-position Bloch phase changes it
by only −0.13 cm⁻¹, far smaller than the 11.32 cm⁻¹ discrepancy from the
converged 4³ reference.

The atomic phase was applied to the complete rank-2 fold/unfold map, not only
to Upsilon. With `P_a(q)=exp(-2π i q·tau_a)`, fine→coarse transport uses
`P(k)†P(q)` on both tensor legs and the return path uses the inverse adjoint.
The centered-BZ q representative is used because this gauge is not periodic
under `q→q+G` for a non-Bravais basis. This construction preserves both
`A(Q-k)=A(k)^T` and the masked Hermiticity of the full D3+D4 operator.
[SUPERSEDED 2026-07-18: the centered-BZ representative choice is exactly
the branch bug described in the "Per-atom phase gauge" section below; the
implementation now transports with the local displacement delta =
corner − q, which is branch-consistent and phase-exact for SnTe. The
−0.13 cm⁻¹ number itself is essentially unchanged by the fix because the
Γ D3-only observable is phase-blind.]

The phase does not make the interpolation atom-resolved. It is a unitary
Bloch-gauge change: it includes the intracell coordinate `tau_a`, but does
not select a supercell replica `delta+L m`. The trilinear corner weights
remain identical for every atom pair. At L=2 they therefore retain the same
`+1/-1` cell-image tie.

Exact atom-resolved FC3 centering instead needs a kernel indexed by all
three atoms and both folded cell differences. Its image translations must
minimize the complete triplet perimeter, including basis positions. The
current rank-2 fold cannot do this because the force-leg atom remains hidden
inside the coarse Julia kernel. Recovering Tensor3-like precision requires
either the centered Tensor3 vertex (37.65 cm⁻¹ in the corrected comparison)
or a slot-resolved stochastic kernel split by force atom/channel. The older
37.6–37.7 cm⁻¹ identity/control values used the interpolated auxiliary matrix
and are not physical 4³ references.

The conclusion is therefore unchanged but sharper: the error is not caused
by the harmonic/Upsilon interpolation or by omission of the common
atom-position Fourier phase. Trilinear q interpolation is a cell-basis
Bartlett centering and cannot reconstruct SnTe's atom-resolved, long-ranged
FC3 vertex from the 2×2×2 cell-corner representation. The atom-centered
Tensor3/FC3 route remains necessary to reproduce the direct result.

## 8×8×8 coarse-mesh convergence audit — previous results withdrawn

The previously reported 4³-source and “full 8³” curves did not satisfy the
required provenance and are withdrawn. Specifically:

- the 4³ auxiliary matrix was interpolated from the converged 2³ SSCHA
  matrix, rather than obtained from an independent 4³ SSCHA relaxation;
- the associated 4³ ensemble was generated around that interpolated matrix;
- the curve previously called “full 8³” used the converged 2³ auxiliary
  matrix and 2³ `FC3`, evaluated on an 8³ integration grid. It tested the
  integration grid, not convergence of an 8³ SSCHA reference.

Consequently, the old 38.24, 41.43, and 48.19 cm⁻¹ values cannot establish
2³→4³→8³ source-mesh convergence, and the generated comparison figure is
not a valid convergence figure. Fresh, independently converged 4³ and 8³
SSCHA auxiliary matrices and mesh-matched ensembles are required. The 4³
reference has now been supplied by the converged calculation above; the
8³ calculation and the requested 4³→8³ audit are intentionally deferred.
The replacement 2×2 figure will be produced only after a converged 8³
reference is available.

## Decisive oracle test: the hardening is the trilinear map, not a bug

Requested experiment: apply **trilinear interpolation to the centered FC3
oracle itself** (evaluate the deterministic tensor vertex at the fine pairs
by trilinear corner-summing its coarse-pair values, instead of Fourier
evaluation of the centered real-space tensor), and push it through the same
tensor-D3 Lanczos pipeline. If the TDSCHA trilinear code is correct, this
must reproduce the TDSCHA trilinear spectrum; if it instead reproduces the
Fourier oracle, the code has a bug.
Script: `report/interpolation/scripts/oracle_trilinear_snte.py`; results:
`report/interpolation/data/snte_oracle_trilinear_test.json`.

**A. Code audit (bug hunt).**

| check | result |
|---|---:|
| TDSCHA fine vertex vs independently re-coded trilinear map (fresh corner weights from q coordinates, fresh Cartesian rotations, applied to the native coarse map) | 1.89×10⁻¹⁴ |
| masked fine modes | 3 (exactly the Γ acoustic translations) |
| smallest valid interpolated frequency | 3.7×10⁻⁴ Ry (≈ 81 cm⁻¹·10⁻¹… no anomaly) |

No spurious zeros at interpolated q points: the off-grid vertex is smoothly
*attenuated* (29–35 % weighted |V|², previous section), never zeroed, and
the fold/unfold is bit-exact trilinear interpolation.

**B. Vertex level.** Trilinear-oracle ≡ Fourier-oracle at commensurate
pairs to 6×10⁻¹⁷ (corner sum is the identity there). Off-grid,
trilinear-oracle vs Fourier-oracle differ by 113.4 % — identical to the
TDSCHA-vs-Fourier discrepancy. The stochastic TDSCHA trilinear vertex
matches the **trilinear-oracle** to 3.0 % (commensurate) and 3.1 %
(off-grid) after one global amplitude fit — pure stochastic noise, uniform
on/off grid.

**C. Spectrum level (Γ TO, D3-only, 30 steps, sm 1.5).**

| pipeline | TO peak (cm⁻¹) |
|---|---:|
| centered FC3, **Fourier** evaluation (oracle) | **37.65** |
| centered FC3, **trilinear** evaluation (this test) | **45.78** |
| stochastic TDSCHA trilinear (N=4000) | **45.65** |

The predicted equivalence holds: trilinear applied to the exact,
noise-free, atom-centered tensor gives the same washed-out spectrum as the
TDSCHA trilinear machinery. **Conclusion: there is no bug.** Trilinear
interpolation uses only the vertex values at the 8 coarse q-points, and all
centerings agree on those samples — the atom-centered information that
distinguishes the Fourier oracle lives entirely in the choice of
real-space images, which is invisible to any interpolation built purely on
the coarse q-samples. Corner-averaging the complex amplitudes then cancels
the short-wavelength components before the response squares them, which
attenuates the off-grid vertex (~29 % weighted |V|²) and produces the more
Lorentzian, harder lineshape.

## Reproduction

```bash
micromamba run -n sscha python -m pip install -e . --no-build-isolation
micromamba run -n sscha python -m pytest tests/test_trilinear -q
micromamba run -n sscha python \
  report/interpolation/scripts/converge_snte_sscha_reference.py \
  --mesh 4 --n-config 4000 --max-pop 10 --tag N4000
micromamba run -n sscha python \
  report/interpolation/scripts/bench_snte_converged4_lanczos.py
micromamba run -n sscha python \
  report/interpolation/scripts/diagnose_snte_trilinear_vertex.py
micromamba run -n sscha python \
  report/interpolation/scripts/plot_snte_converged4_comparison.py
```

The converged SSCHA dynamical matrices and provenance are under
`report/interpolation/data/sscha_references/mesh4_N4000/`. The native
Lanczos spectrum is
`report/interpolation/data/snte_reference_4x4x4_converged_sscha_lanczos_N4000_s250.dat`.

`QSpaceTrilinearLanczos` now uses only coarse corner Upsilon. The remaining
diagnostic switch is `atomic_phase=True|False`, defaulting to `False`.

## Decisive oracle test: the hardening is the trilinear map itself, not a code bug

Requested experiment (`report/interpolation/scripts/oracle_trilinear_snte.py`,
JSON in `report/interpolation/data/snte_oracle_trilinear_test.json`): the
deterministic tensor-D3 pipeline is run twice on the same centered FC3 —
once evaluating the vertex at the fine pairs by Fourier interpolation (the
standard oracle), once by TRILINEAR corner-summing of the same tensor
evaluated only at the coarse pairs. All at 2³→4³, Γ TO, D3-only, N=4000,
30 steps, sm 1.5.

| check | result |
|---|---:|
| A. TDSCHA fold/unfold vs independently coded trilinear map | 1.9×10⁻¹⁴ |
| A. masked fine modes (spurious-zero hypothesis) | 3 = Γ acoustics only; min valid ω = 3.7×10⁻⁴ Ry |
| B. trilinear-oracle vs Fourier-oracle, commensurate blocks | 6×10⁻¹⁷ (identity) |
| B. trilinear-oracle vs Fourier-oracle, off-grid blocks | 113.4% |
| B. stochastic TDSCHA vertex vs trilinear-oracle (amplitude-fitted) | 3.0% on-grid / 3.1% off-grid |
| C. Fourier-oracle spectrum peak | **37.65 cm⁻¹** |
| C. trilinear-oracle spectrum peak | **45.78 cm⁻¹** |
| C. stochastic trilinear spectrum peak (N=4000) | **45.65 cm⁻¹** |

The predicted equivalence holds to 0.13 cm⁻¹: trilinear interpolation
applied to the exact centered FC3 reproduces the TDSCHA trilinear result,
and the TDSCHA stochastic vertex equals the trilinear-oracle vertex to the
stochastic noise level uniformly on and off grid. There is no bug and no
spurious zeroing. The mechanism is structural: trilinear interpolation
uses only the tensor's values at the 8 coarse q-points, and those
commensurate samples are identical for the centered and uncentered
(aliased) tensor — the atom-resolved centering information lives entirely
in how the vertex is continued *between* coarse samples, which the Fourier
evaluation of the centered real-space representation supplies and a
corner-weighted average cannot. At L=2 all coarse points are also
time-reversal invariant (V real-structured), so the complex phase winding
of the true vertex at e.g. q=1/4 is unrepresentable and the averaged
amplitudes cancel (113% off-grid error, 29–35% surviving weighted |V|²).

Practical consequence: to use this class on SnTe-like vertices, the D3
channel must evaluate the centered tensor by Fourier interpolation
(the `d3_mode="tensor"` hybrid — deterministic and cheap) while the
trilinear machinery serves the D4 channel; equivalently, the paper's
scheme should state that the trilinear formula applies to the *centered*
vertex continuation, not to the raw coarse Bloch samples.

## Phase-winding topology: why the trilinear map zeroes the vertex

Requested analysis: the phase of Phi3 appears to rotate in q space on the
Te atoms, and the trilinear interpolation zeroes the tensor at cube
centres while the centered continuation stays finite. Is there a
singularity, and is there topology behind it?
Script: `report/interpolation/scripts/snte_phase_topology.py`; data:
`report/interpolation/data/snte_phase_topology.{json,npz}`; figures:
`figs/snte_phase_winding.pdf`, `figs/snte_winding_scheme.pdf`; report
section `sec:topology` ("Why the phase of Phi3 rotates in q space").

| measurement | result |
|---|---:|
| Te–Sn block norm vs Te–Te / Sn–Sn blocks | 0.28 vs ~1e-4 |
| lattice harmonics of the dominant Te–Sn element | c=+0.2208 at R=(0,−2,1), −0.2208 at (1,−1,0), rest <1e-3 (3D fit; corrected 2026-07-18) |
| centre of the two images x_c | (1/2,−3/2,1/2) = τ_Te−τ_Sn — Te–Sn bond (corrected 2026-07-18) |
| phase advance across one BZ crossing | exactly pi (linear to 6e-7 deg) |
| coarse 2^3 samples | all real (Im/Re <1e-12), signs (0,−,+,0,+,0,0,−) |
| trilinear fine vertex | identically real (max Im 8e-17; true Im up to 0.44) |
| amplitude kept at (1/4,0,0) | 0.706 = cos(pi/4) (chord of the phase arc) |
| cube centre (1/4,1/4,1/4) | ‖V‖ 0.442 → 0.0007 (factor 1.6e-3); true norm 71% of max |
| interior phase vortices in (q1,q2,0) plane | none (41×41 plaquette winding scan) |
| exact zeros of the true vertex | only parity zeros at Gamma |

Interpretation. There is no singularity of the physical vertex: it is
smooth and vanishes only at symmetry-forced points (Gamma). The rotation
is the Fourier shift phase e^{2 pi i q·x_c} of a coupling centred at half
a lattice vector — on the Te–Sn bond. With Sn at the origin the factor
attaches to the Te legs, which is the observed "rotation on the Te
atoms". Inversion symmetry quantizes the centre (on-site or mid-bond,
nothing between): the harmonics obey c_R = −c_{2x_c−R} with 2x_c integer,
a Z2 invariant nu = 2x_c mod 2 per direction. SnTe is in the nontrivial
class nu = (1,1,1) — the Zak-phase-pi / SSH mid-bond class. This forces
(i) V(Gamma)=0, (ii) real coarse samples with alternating signs, and
(iii) protection: the rotation is invisible inside the coarse samples
(all real), so it must be supplied from outside — a branch-consistent
per-atom Bloch phase does supply it (see the "Per-atom phase gauge"
section below, which corrects the earlier blanket no-gauge claim),
while amplitudes remain unrecoverable by any phase. The zeroing "in the centre" is manufactured
by the reconstruction: a real convex corner average connecting
opposite-sign samples must cross zero on surfaces through the cube
centres, while the true vertex escapes through the complex plane.

Practical corollary: corner-based q interpolation is accurate exactly for
nu = 0 (site-centred) coupling channels. The invariant is readable
without any fit: track the vertex phase along one BZ crossing per axis,
or compare vertex signs between the q=0 and q=1/2 TRI planes. For
SnTe-like bond-centred anharmonicity the centering information
(half-lattice-vector offset) must be supplied explicitly, as the
tensor/atomic-window routes do.

## Per-atom phase gauge: the report's dismissal was wrong, and here is what a phase can and cannot cure

User challenge (2026-07-18): can a different phase per atom cure the
trilinear failure?  Scripts:
`report/interpolation/scripts/snte_phase_gauge_test.py` (vertex level),
`report/interpolation/scripts/oracle_gauge_spectrum.py` (spectra); data
`snte_phase_gauge_test.json`, `snte_phase_gauge_paths.npz`,
`snte_oracle_gauge_spectrum.json`; report Sec. `sec:topo-gauge`.

Two errors were found and fixed:

1. The report's "unitary gauge does not change the singular values of P"
   argument is invalid: it constrains norms, not the alignment between
   the interpolated function and the truth. A q-dependent phase changes
   WHICH function is interpolated.
2. The implemented `atomic_phase=True` had a branch inconsistency: corner
   phases were evaluated at BZ-wrapped representatives, which flips signs
   for half-integer tau differences. FIXED in `Modules/QSpaceTrilinear.py`:
   the corner cache now stores the local displacement delta = corner − q
   (unwrapped, |delta| ≤ 1/Nc) and the gauge uses exp(±2πi delta·tau_a).
   The fixed class matches an independently coded gauged map to 0.0 and
   all 14 trilinear tests pass.

Vertex-level metrics (Γ TO row, all 4³ fine pairs, Frobenius):

| reconstruction | rel err² | \|V\|² survival | ‖V‖ at cube centre |
|---|---:|---:|---:|
| plain trilinear | 1.12 | 0.375 | 0.0007 |
| atomic phase, wrapped (old, buggy) | 0.78 | 0.422 | 0.156 |
| atomic phase, branch-consistent | **0.178** | 0.422 | 0.156 |
| best integer branch (scan of 125) | 0.178 (m=0) | 0.422 | 0.156 |
| per-channel oracle phase strip | 0.178 | 0.422 | 0.156 |
| truth | 0 | 1 | 0.442 |

Key structural fact: fitting the antisymmetry centre of every Te–Sn
coupling channel independently gives x_c = tau_Te − tau_Sn (raw stored
coords) for ALL nine Cartesian elements — for SnTe the factorized
per-atom phase IS the optimal per-channel phase (they agree to machine
precision). This also corrects the earlier report section: the earlier
claimed centre (½,−½,−½) from path FFTs was under-determined; the 3D
fit gives (½,−3/2,½) = Δτ_raw. [Fixed in the report.]

Spectral test (deterministic tensor-D3 oracle, N=400 scaffold, Γ TO,
D3-only, 2³→4³): Fourier oracle 37.65 (control, exact match) | plain
trilinear 45.78 (exact match) | phase-corrected trilinear **45.59**.
The optimal phase recovers 0.19 of the 8.13 cm⁻¹ gap.

Why so little: (i) amplitude is phase-inaccessible — |Σ w e^{iφ} V| ≤
Σ w |V| for any phases, and half the coarse samples sit at the parity
nodes of the sine envelope (survival saturates at 0.42; cube centre at
best ~50% even with perfect alignment); (ii) the D3-only Γ response
contracts the two-phonon sector as |V|²χ — vertex phases cancel
identically, which also retroactively explains the old −0.13 cm⁻¹
atomic-phase result (the bug corrupted phases, but the observable never
looked at them).

Practical consequences: the corrected `atomic_phase=True` is now
strictly preferable (exact phases; matters for D3×D4 cross terms,
off-diagonal responses, non-Γ perturbations), but for the spectral
benchmark the conclusion is unchanged — the missing information is the
amplitude envelope between samples, which requires the real-space image
assignment (centering), not a gauge.

## 11. FT-level phase vs phase-in-the-weights (2026-07-19, user question)

Question: can the per-atom phase exp(2πi q·τ_a) be moved out of the
interpolation coefficients and into the Fourier transform of
displacements/forces (eq:q of the Overleaf paper), leaving a *standard*
interpolation?  Script:
`report/interpolation/scripts/verify_ft_phase_equivalence.py`
(vertex-level, seconds; output data/snte_ft_phase_equivalence.json).

Results (SnTe Γ-TO vertex, 2³→4³, rel err² over the fine mesh):

| scheme | rel err² | identity check |
|---|---|---|
| plain trilinear, cell gauge | 1.125 | — |
| gauged FT + plain trilinear (consistent chart) | 0.178 | == delta-phase trilinear to 2.2e-16 |
| gauged FT + trilinear, periodic table (no branch factor) | 0.737 | ≈ old wrapped bug (0.782) |
| gauged FT + smallest-frequency Fourier interp | 2.9e-5 | == atom_fourier cardinal P_ab to 3.2e-16 |

Conclusions:
1. YES — the phase belongs in the FT.  Both the delta-phase trilinear
   AND the atom-pair cardinal kernel are exactly (machine-precision)
   equal to *phase-free* standard rules applied to the gauged fields.
2. The gauged fields are NOT periodic in q: Ṽ(q+G) =
   e^{−2πi G·(τ_a−τ_b)} Ṽ(q).  The single extra ingredient any standard
   rule needs is the boundary branch factor (a sign for SnTe half-integer
   Δτ); it can be baked into a padded (N+1)³ table.  Ignoring it is the
   old branch bug (0.18 → 0.74).
3. Plain trilinear on the gauged fields is NOT sufficient (0.178, peak
   45.59): the amplitude envelope sin(πq) is not piecewise linear.  Full
   quality needs Fourier interpolation — but in the atom-centred gauge it
   is the *textbook* rule: FFT the samples, keep in each aliasing class
   the smallest frequency, which is automatically R+τ_a−τ_b = the
   shortest physical pair separation (= q-space pairwise centering).
   The P_ab cardinal kernel is just this rule written in the cell gauge.

Overleaf paper (6a4ba34cdbe5288bb39378e3) updated: the obscure first
audit block (argmin/S_s/w_R/P_ab notation) replaced by the atomic
Fourier series A_ab(q) = Σ_R c_ab(R) e^{2πi q·(R+τ_a−τ_b)}
(eq:atomic-series) + smallest-distance rule + the sin(πq) undergrad
example; all other P_ab references rewritten accordingly.

## 12. Fourth order (D4) in the atom-centred Fourier scheme (2026-07-21)

The atom-centred Fourier continuation (`atom_fourier=True`, the phase moved
into the FT) that repairs D3 also repairs D4, by the same mechanism, with
NO new code path.  Scripts: `d4_atom_fourier_oracle.py` (noise-free vertex),
`bench_d4_atom_fourier.py` (spectra), `d4_seed_noise_control.py` (noise
floor), `make_d4_atom_fourier_fig.py` (figure `figs/d4_atom_fourier.pdf`);
tests `tests/test_trilinear/test_d4_atom_fourier.py` (10 tests); report
Sec. `sec:d4-atom-fourier`.

### Why it works (momentum count)

The D4 term maps the incoming two-phonon block at `(q', Q-q')` to the
outgoing block at `(q, Q-q)`.  With conjugated fields the four legs carry
`(+q, -q, -q', +q')`, so the two internal momenta are INDEPENDENT and each
couples to ONE pair offset: q to R2 (~ tau_a-tau_b), q' to R3-R4
(~ -(tau_c-tau_d)).  The correct four-leg continuation is the tensor
product of the pairwise kernel with itself -- `P_ab(q,k)` on the unfold,
`conj(P_cd(q',k'))` on the fold -- exactly what the existing fold/unfold
does.  This is the decisive contrast with the real-space D4 centering of
`sec:d4center` (`leg` wrong-sign, `reference` 50%): there the minimal image
is a genuine four-body geometry; in the atom-centred q-space gauge it
factorizes into two pair problems.

### Noise-free vertex oracle (machine-exact)

Closed-form quartic vertex of the chain, all fine (q,q') pairs, rel err^2:

| coarse -> fine | atom_fourier | trilinear (off-grid) |
|---|---:|---:|
| 3 -> 9 | 1.4e-31 | 5.55e-2 (6.25e-2) |
| 2 -> 4 | 8.7e-33 | 1.75e-1 (2.33e-1) |
| 2 -> 6 | 3.1e-32 | 1.81e-1 (2.03e-1) |
| 4 -> 8 | 4.3e-32 | 1.91e-2 (2.55e-2) |

Exact to machine zero, including L=2 where every off-grid pair is a
Nyquist tie (split 50/50) -- which also confirms the sign/branch
convention (one wrong sign kills the cancellation).  Range stress test
with a second-neighbour quartic bond (|R|=2): aliased and NOT recovered at
L=2 (rel err^2 0.25), exact at L=4,5 (< 1e-31).  The limit is coarse-mesh
range, one order up from D3 -- not a defect of the map.

### Spectral benchmark (D4-dominated, g3=0.15 g4=2.0, N=6000, 110 steps)

Metric = D4-induced peak shift (the renormalization).  Error bars = 3-seed
spread of the direct D4 shift.

| probe | direct D4 shift (cm-1) | atom_fourier capture / L1 | trilinear capture / L1 |
|---|---:|---:|---:|
| Gamma band 5 | +52.9 +-1.9 | 109% / 0.38 | 72% / 0.67 |
| Gamma band 4 | +14.7 +-3.7 | 139% / 0.80 | -16% / 0.51 |
| q=3/9 band 5 | +22.6 +-1.8 | 112% / 0.37 | 90% / 0.35 |
| q=3/9 band 4 | +21.0 +-4.7 | 89% / 0.21 | 61% / 0.41 |

atom_fourier captures the fourth-order renormalization to within the
stochastic noise floor at every probe; trilinear corners under-capture and
flip sign for the weak Gamma band-4 line (the four-leg analogue of the D3
hardening).  Probes are coarse Q because this class refines the INTERNAL
loop q', not the external Q -- the SnTe Gamma-TO setup.

### Tests (10, all green)

`test_d4_atom_fourier.py`: exact vertex reconstruction; range-limit is
aliasing not a defect; kernel mirror identity `P_ab(q,k)=P_ba(-q,-k)`; the
three leg exchanges of Phi4 at the vertex level; ASR preserved on all four
legs with non-cardinal + atom-pinned control maps proving the test has
teeth; isolated-D4 block Hermiticity at TRI/non-TRI Q; unfold mirror
`D(Q-q)=D(q)^T`; commensurate identity to the parent (D4 shown non-vacuous).
Note: `ignore_v3=True` does NOT give a Hermitian operator (it zeroes the D3
output channel but not the D3 input channel; measured 2.9e-2 asymmetry in
the PARENT QSpaceLanczos too) -- D4 is isolated by projecting the R sector
instead.

## 13. AUDIT: Nyquist-tie handling is metric-blind (2026-07-21, user-requested)

Question: are the Nyquist ties correctly implemented, including >2-image
3D ties (e.g. a bcc central atom equidistant from 8 replicas)?

**Finding: NO for non-orthogonal cells.** `_nearest_alias_images` +
`_build_atom_fourier_kernel` select the minimal image and split ties
SEPARABLY PER CARTESIAN AXIS in FRACTIONAL coordinates (metric-free), then
take a product over axes (matching the report's product form
eq:atom-fourier-kernel). This is exact ONLY when the coarse Wigner-Seitz
cell is an axis-aligned box (orthorhombic + diagonal metric). The true
minimal image minimizes the Cartesian length |(R-d).A|, coupling the axes
through the metric A.A^T; its ties live on non-axis-aligned WS faces.

Audit: `report/interpolation/scripts/audit_nyquist_ties.py`, data
`data/audit_nyquist_ties.json`.

| cell | Nc | separations | kernel disagrees | note |
|---|---|---:|---:|---|
| cubic (control) | 2^3, 3^3 | 6 | 0 | exact, incl. 8-fold WS-corner tie -> 1/8 |
| fcc (SnTe) | 2^3 | 5 | 5 | kernel differs; observable impact = only excited classes |
| bcc primitive | 2^3 | 3 | 2 | |
| hexagonal | 3x3x2 | 3 | 3 | |

("disagrees" = the kernel's class assignment differs from the metric; this
is NOT the observable error -- see the correction below.)

These numbers count aliasing-class assignments; the "disagree" column is a
KERNEL property. Kernel-level norm impact (2^3->4^3, relative Frobenius,
separable vs metric): cubic 0.0 (exact) | fcc SnTe bond centre 23% | fcc
d=(1/2,0,0) 36%.

### CRITICAL CORRECTION: the kernel-norm mismatch is NOT the observable error

(User challenge: "the SnTe test works very well, why the big mismatch?"
The challenge is correct.) The 23-36% figure is the difference of the FULL
kernel matrix, but a physical coupling only excites the aliasing classes
where it has non-zero coarse weight. A short-ranged coupling excites only
classes whose physically-shortest image is UNIQUE -- and on those classes
separable and metric AGREE. Verified directly:

- SnTe dominant Gamma-TO vertex = two nn Te-Sn harmonics R1=(0,-2,1)
  [class (0,0,1)], R2=(1,-1,0) [class (1,1,0)]. Each is the UNIQUE shortest
  image of its class (3.28 A; next image 7.34 A, no tie). Both rules place
  them identically.
- Reconstructing the ACTUAL two-image SnTe coupling 2^3->4^3: separable
  rel err^2 = 2.0e-32, metric rel err^2 = 2.2e-32 -- BOTH machine-exact.
- So the 2.9e-5 SnTe Gamma-TO result is CORRECT and is NOT compromised.
  The 23-36% lives in unexcited classes (e.g. class (0,0,0), coarse
  coefficient = 0) and is multiplied by zero.

The bug bites ONLY a coupling with weight in a class whose shortest image
the separable rule mis-assigns -- i.e. a coupling pinned to a genuine
metric degeneracy (a WS face/edge/corner of a non-orthogonal cell) or a
separably-manufactured spurious tie. Minimal demonstration (primitive bcc,
d=(1/2,1/2,0), coupling on the metric-shortest image of class (0,0,1) where
separable splits (0,0,+-1) at 1/2 but metric picks (0,0,-1) alone):
**separable rel err^2 = 0.50, metric rel err^2 = 1.1e-32.**

Note on the user's canonical bcc example: the central atom equidistant to 8
nn, d=(1/2,1/2,1/2), is NOT a failing case even in the primitive cell --
its excited classes each reduce to a unique or separably-tied image, so the
separable rule already handles it. The failure is specific to couplings on
a NON-axis-aligned WS degeneracy, e.g. d=(1/2,1/2,0), not to "every
non-orthogonal cell always".

**Scope.** (1) chain D4 tests: UNAFFECTED ((1,1,L) mesh = single aliasing
class per transverse axis = trivial factor 1). (2) SnTe Gamma-TO: CORRECT
as reported (excited classes have unique images). (3) The metric-blind rule
is nonetheless wrong as a GENERAL kernel and will corrupt any observable
whose coupling excites a mis-assigned degenerate class. The fix is worth
doing for correctness/robustness, NOT because it changes any result
obtained so far.

**Fix (implemented 2026-07-21):** replace the separable per-axis product by
the true 3D metric minimum-image assignment eq:atom-fourier-metric
(`tie_metric=True`, default). See section 14 below for validation.

## 14. Metric-aware tie fix: implementation + validation (2026-07-21)

Fix: `QSpaceTrilinearLanczos(..., atom_fourier=True, tie_metric=True)`
(default). New static `_metric_alias_images(d, Nc, metric)` returns the true
3D minimum-image set of every aliasing class (metric = A A^T), with equal
tie splitting; `_build_atom_fourier_kernel_metric` assembles the
non-separable kernel P_ab(q,k) = (1/prodNc) Σ_class Σ_{R in minimg} w_R
e^{2πi(q-k/Nc).R}. The legacy separable product is kept as
`tie_metric=False` (orthorhombic fast path + control).

Preserved properties (proved + tested): commensurate identity (cardinal for
ANY image choice), mirror identity P_ab(q,k)=P_ba(-q,-k) [algebra:
img(κ;-d)=-img(-κ;d) with symmetric tie weights], hence Hermiticity; ASR
(cardinal at Γ, independent of the other pair's atoms).

Validation:
- `_metric_alias_images` == brute-force 3D min image on cubic/fcc/bcc,
  incl. the 4-fold non-separable tie; cubic reduces to the separable
  product (8-fold WS corner -> 1/8).
- Non-orthogonal (bcc-primitive) class instance: a coupling pinned to the
  mis-assigned class (0,0,1) of d=(1/2,1/2,0) -> metric rel err^2 < 1e-20,
  separable rel err^2 ~ 0.50.
- Commensurate identity, mirror identity, operator Hermiticity all hold for
  the metric kernel on the non-orthogonal cell.

Tests: `tests/test_trilinear/test_nyquist_ties.py` (8 tests) +
`_toy_crystal3d.py` (minimal non-orthogonal 3D crystal). Full trilinear
suite: 34 passed (was 26). Chain D4 tests unchanged (metric == separable on
(1,1,L)). SnTe Gamma-TO unchanged (excited classes have unique images).

Bottom line: the fix corrects a real correctness bug for couplings on
non-axis-aligned WS degeneracies of non-orthogonal cells; it changes NO
result obtained so far (SnTe Gamma-TO, chain D3/D4), which were already in
the regime where separable and metric agree.

## 15. CsSnI3 production stress test + performance optimization (2026-07-22)

Production run: CsSnI3 P4/mbm (10-atom cell, tetragonal/DIAGONAL metric,
supercell 4x4x4 = 640 atoms), T=500 K, native 4x4x4 -> interpolated 8x8x8
unpolarized Raman (7 components, D3+D4, atom_fourier). The ensemble has
12288 configs; subsampled for tractability. Scripts:
`report/interpolation/scripts/cssni3_raman_interp.py` (run),
`make_cssni3_raman_fig.py` (figure `figs/cssni3_raman_interp.pdf`).
Q = Gamma is on the coarse mesh, so the interpolation refines the internal
two-phonon loop 4^3 -> 8^3 while reusing the coarse stochastic kernel:
8^3 spectral resolution at 4^3 sampling cost, no independent 8^3 ensemble.

### Bottleneck analysis (8x8x8, 10-atom cell, profiled)

Per L-application (apply_anharmonic_FT), 500 configs:
- Julia coarse kernel (get_perturb_averages_qspace): ~1.75s DOMINANT --
  SAME cost as native 4x4x4 (runs on cn_q=64 coarse pairs), scales with
  N_configs x N_sym. This is the irreducible cost; interpolation does NOT
  inflate it (the whole point: coarse kernel, fine resolution).
- fold + unfold (my code): ~0.3s AFTER optimization (was the target).
Per-step overhead of interpolation vs native ~= 0.27s/step (measured:
native 17s/pol vs interp 21s/pol at 300 configs / 15 steps).

Construction (one-time): ~45s, dominated by CellConstructor
`ForceTensor.Interpolate` (28s, harmonic dyn -> 512 fine q-points, called
per non-commensurate q) + `SetupFromPhonons`/`SetupFromTensor` (~25s). This
is reused CC code, amortized over the whole Lanczos run.

### Optimizations applied (my implementation)

1. **atom_fourier kernel build**: was a Python triple loop
   iq x ik x nat^2 (for 8^3 CsSnI3: 512x64x100 = 3.3M iterations, ~10s+).
   The phase exp(2pi i (q-x).R) factorizes as exp(2pi i q.R) *
   conj(exp(2pi i x.R)), separable in the fine index q and coarse index x,
   so each atom-pair kernel is ONE matmul over the images R:
   K[:,:,a,b] = (1/Nc)(E_fine * w) @ E_coarse.conj().T. -> milliseconds.
2. **fold/unfold**: were Python double loops over iq x ik doing
   `np.repeat` of the (nat,nat) kernel to (nb,nb) then elementwise multiply
   (197k np.repeat calls/application). Replaced by a single einsum on the
   ATOM-BLOCK kernel:
   fold  A[k,a,i,b,j] = sum_q conj(K[q,k,a,b]) blk[q,a,i,b,j]
                        ('qkab,qaibj->kaibj'),
   unfold B[p,a,i,b,j] = sum_k K[iq1(p),k,a,b] D[k,a,i,b,j]
                        ('pkab,kaibj->paibj').
   fold 0.22s->~0.09s, unfold 0.07s->~0.02s; eliminated all np.repeat.
   34/34 trilinear tests still green (correctness preserved).

Residual one-time bottleneck is CC's harmonic-dyn Fourier interpolation
(per-q ForceTensor.Interpolate); a batched multi-q DFT of the centered FC2
would cut it further but it is amortized and left as a future micro-opt.

Note: this cell is tetragonal (DIAGONAL metric), so tie_metric metric==
separable here; the run does not stress the non-orthogonal tie path (that
is covered by the bcc tests of section 14).

### CsSnI3 run result (2026-07-22)

Completed: native 4^3 3829s (7 pols x ~547s), interpolated 8^3 4042s
(7 pols x ~577s) + 45s construct. **Only +5.6% wall time for 8^3 two-phonon
resolution** (the Julia kernel ~5.5s/step is shared; interpolation adds
~0.3s/step fold/unfold). Figure `figs/cssni3_raman_interp.pdf`.

Correctness: identity control fine=(4,4,4) reproduces native to 3.2e-15 per
polarization on the real material. Interpolated 8^3 harmonic dyn stable
except 10 near-Gamma acoustic dips (-6..-12 cm-1, 0.1% of modes, masked);
Gamma optical modes pinned bit-identical.

Result: native 4^3 has one band ~90 cm-1; interpolated 8^3 splits it into
~125 and ~155 cm-1. This is refinement of the under-resolved 4^3 two-phonon
continuum (64 -> 512 internal momenta), large here because CsSnI3 is
strongly anharmonic. NO native 8^3 reference exists (interpolation is the
estimate, not ground truth); the identity control + machine-precision chain
D4 benchmarks certify the interpolation machinery. Report sec:cssni3.

### Files / reproduction (2026-07-22)

Production scripts + results archived in the simulation tree:
`.../Perovskites/CsSnI3/SCHA/P4_mbm/4x4x4/qspace_interpolation_8x8x8/`
(subdirs scripts/, results/, diagnostics/, plus README.md). Repo copies:
`report/interpolation/scripts/cssni3_raman_interp.py`,
`make_cssni3_raman_fig.py`, `cssni3_raman_analysis.py`,
`cssni3_raman_convergence.py`; data `report/interpolation/data/
cssni3_raman_interp.npz`; figs `figs/cssni3_raman_interp.pdf`,
`cssni3_raman_analysis.pdf`.

## 16. Deep investigation of the CsSnI3 4^3->8^3 shift + save_abc replot (2026-07-22)

User challenged the massive peak shift (native ~90 -> interp ~125/155).
Deep diagnostics (scripts in the qspace_interpolation_8x8x8/diagnostics dir):

- **Identity control**: fine=(4,4,4) atom_fourier reproduces native 4^3 to
  3.2e-15 per polarization on the REAL material. Machinery correct. (The
  identity control does NOT test scale3/scale4 since they=1 at fine=coarse;
  the scale factors are separately validated by the chain physics test vs a
  direct fine-supercell ensemble.)
- **Incommensurate anharmonicity (the user's hypothesis)**: the two-phonon
  coupling |d2v| at OFF-GRID fine pairs vs COMMENSURATE pairs has per-pair
  ratio = **1.00** (off-grid = 86% of pairs -> 86% of weight). So there is
  NO lowering of anharmonicity at incommensurate points -- the opposite of
  the SnTe trilinear failure. atom_fourier reconstructs the vertex at full
  magnitude off-grid.
- **Bare SSCHA auxiliary Raman**: main optical band ~80 cm-1. Native 4^3
  (~90) sits essentially AT the bare position; interp 8^3 (~125/155) moves
  strongly away (harder).
- **Two-phonon JDOS**: nearly identical 4^3 vs 8^3 (mean 107 cm-1). Gross
  continuum already converged.
- Masked modes: only 10 near-Gamma acoustic dips (-6..-12 cm-1, 0.1%), not
  the cause.

Interpretation: the shift is a DYNAMIC resonance effect. The Raman mode is
embedded in the two-phonon continuum; at only 64 momenta the 4^3 continuum
is under-sampled (discrete states spaced wider than the 8 cm-1 smearing ->
spurious structure pinning the peak near the bare ~90), while 8^3 (512
momenta) resolves the |V|^2-weighted continuum near the pole. Vertex mag
(1.00) and gross JDOS agree; the fine resonant structure is what moves the
peak. Convergence 4^3/8^3/12^3 (running) tests whether 8^3 is settled. No
native 8^3 reference exists.

**save_abc replot workflow (user-requested):** the Lanczos coefficients are
now saved per mesh/polarization with the built-in `Lanczos.save_abc`
(+ perturbation_modulus in meta.npz, which save_abc omits) by
`cssni3_raman_convergence.py` and `cssni3_raman_interp.py`. Replot at any
smearing/grid in <1s with `cssni3_raman_replot.py` -- reconstruction exact
(max|live - from_abc| = 0). Never redo the multi-hour Lanczos to retune the
smearing again.

## 17. RETRACTION of the CsSnI3 spectral results (2026-07-22)

**Sections 15-16 physical conclusions are RETRACTED.** User challenged the
"massive" 4^3->8^3 peak shift; deep investigation showed it was a bug, not
physics. The user's instinct was correct.

Root cause: for this soft, strongly anharmonic perovskite the NATIVE
QSpaceLanczos (not the interpolation) develops EXPONENTIALLY GROWING
tridiagonal coefficients: b*c grows ~4x/step to ~1e41 by step 80, while a_n
stays at the physical omega^2 scale. The terminator (averaging the last
coefficients) is then meaningless and the line shape is unstable to the step
count (global max jumps between the ~11 cm-1 soft continuum and a spurious
~90 cm-1 optical peak). The "90 (4^3) -> 125/155 (8^3)" was an artifact of
this instability, NOT a mesh effect.

The CORRECT spectrum: reconstructed from the user's own saved June-2026
production (raman_pop200_unpol0..6.npz, BOUNDED coefficients |b|~1e-6): it is
SOFT-MODE DOMINATED, peaking near **29 cm-1**, with only weak optical
structure. Script `cssni3_correct_vs_buggy.py`, fig
`figs/cssni3_correct_vs_buggy.pdf`.

Not an interpolation error:
- growth reproduced in bare QSpaceLanczos with the standard recipe on the
  FULL 12288-config ensemble (no interpolation);
- the validated toy chain has BOUNDED b*c (ratio 0.99) under identical code
  -> pathology is system-specific (CsSnI3) in the current code, likely a
  regression since the June production (native path changed: e.g. the
  q-space conjugation-convention commit). Tracked as a separate native-code
  issue.
- interpolation machinery still validated: (4,4,4) identity 3.2e-15 on this
  ensemble; incommensurate vertex ratio off/comm = 1.00 (NO anharmonicity
  loss at incommensurate points -- the user's other hypothesis, refuted);
  chain/SnTe vertex machine-exact; 34 tests green.

Also fixed two real bugs in MY scripts (necessary but not sufficient given
the native growth): (1) missing ens.init() after ens.split() -> stale
forces_qspace -> coefficients ~100x wrong; (2) should build the ensemble
around dyn_gen and update_weights(final), as load_distributed_tdscha does.

What STANDS: the performance result (+~6% wall time for 8^3 resolution) and
the fold/unfold + kernel-build optimizations are independent of spectral
correctness. What is retracted is only the CsSnI3 line-shape interpretation.
A trustworthy CsSnI3 convergence study awaits the native coefficient-growth
fix.

## 18. CsSnI3 CORRECTED — supersedes sections 15-17 (2026-07-22)

The retraction in section 17 was itself too pessimistic. The user was right:
with correct ensemble initialization, standard TD-SCHA works and the method
is fine. Final corrected picture:

**The bug was ensemble initialization in MY driver, not native code.** Two
fixes (both now in cssni3_raman_interp.py / cssni3_raman_convergence.py):
(1) call `ens.init()` AFTER `ens.split()` (QSpaceLanczos reuses stale
forces_qspace otherwise -> coefficients ~100x wrong); (2) build the ensemble
around `dyn_gen` and `update_weights(final)` as load_distributed_tdscha does.

**The b*c coefficient growth is BENIGN.** After the fix, unpol0 4^3 peak =
14.0 cm-1 and the full unpolarized peak = 33.1 cm-1, both STEP-STABLE at
40/60/80/100 steps (identical), despite b*c still growing to ~1e24. The
terminated continued fraction is insensitive to the growth. My earlier
"fatal native growth" claim (section 17) was WRONG: the step-instability was
caused by the init bug (100x wrong coeffs), not the growth.

**Corrected result (rerun, 1500 cfg, 100 steps, save_abc):** native 4^3 and
interp 8^3 both peak at 33.1 cm-1 (soft-mode dominated; weak optical band at
93 cm-1), and they AGREE: normalized L1 = 0.0012, 0.06% max pointwise. This
is physically correct -- the dominant one-phonon soft mode (pinned Gamma
frequency) is insensitive to the internal two-phonon mesh, so 4^3 is already
converged for this observable and the interpolation faithfully reproduces it
(not distorts it). Fig figs/cssni3_raman_interp.pdf (regenerated).

**Cost:** native 4^3 3884s | interp 8^3 4085s + 45s construct = +5% wall
time for 8^3 internal resolution. Stands. Incommensurate vertex ratio
off/comm = 1.00 (no anharmonicity loss off-grid). Interpolation machinery
validated (identity 3e-15, 34 tests green).

Net: sections 15-16 numbers (90->125 "shift") were an init bug; section 17's
"native regression/fatal growth" diagnosis was wrong; THIS section is the
final word. The method works; 4^3 and 8^3 agree for CsSnI3 Raman.

## 19. ROOT CAUSE of the CsSnI3 8^3 imaginary modes: inherited effective charges (2026-07-23)

The twelve imaginary modes on the interpolated CsSnI3 8^3 auxiliary dynamical
matrix (report sec:cssni3, Tab. tab:cssni3-unstable) are **an interpolation
artifact, not physics and not an under-converged SSCHA**. They come from
applying a long-range dipole model to an ensemble generated by a short-range
ML force field. Removing that model removes all twelve.

**Mechanism.** `ForceTensor.Tensor2.SetupFromPhonons` checks
`dyn.effective_charges`. If present it SUBTRACTS the Ewald dipole-dipole term
(`symph.rgd_blk`, sign -1) from every commensurate block before `Center()` and
`Apply_ASR()` run, and `Interpolate` ADDS it back (sign +1). Passing
`lo_to_splitting=False` does NOT disable this -- it only kills the extra
nonanalytic term exactly at Gamma. So:

    Phi_interp(q) = F[ C( F^-1( Phi(qc) - Phi_LR(qc) ) ) ](q) + Phi_LR(q)

At q = qc the two Phi_LR cancel identically => **the cycle is exactly the
identity on the coarse mesh**. Every regression test in this repo is
commensurate, so none of them can ever see this. It acts only off-grid.

For a genuinely polar potential this is correct (the 1/r^3 tail cannot fit in
a 4^3 WS cell). For CsSnI3 the forces come from a short-range MLIP with no
electrostatics: the FCs already decay inside the 4^3 cell, and the stored
Z*/eps are inherited DFT metadata about the MATERIAL, not the potential.
Subtracting a tail that is not in the data leaves a remainder that is
long-ranged (short-ranged minus long-ranged), centering truncates it, and the
ASR is then imposed on the truncated object.

**Audit** (`report/interpolation/scripts/cssni3_longrange_audit.py`, data
`report/interpolation/data/cssni3_longrange_audit.json`), 8^3 = 512 q, 30
bands, from the same converged 4^3 dyn:

| Z*     | ASR | imag modes  | \|dFC_ASR\|/\|FC\| | commensurate residual |
|--------|-----|-------------|--------------------|-----------------------|
| used   | yes | 12 (at 10 q)| 3.61e-3            | **19.5 cm-1**         |
| used   | no  | 4 (at 4 q)  | --                 | 6.4e-4 cm-1           |
| ignore | yes | **0**       | 2.08e-10           | 1.8e-6 cm-1           |
| ignore | no  | **0**       | --                 | 5.4e-4 cm-1           |

Three conclusions:

1. **Imaginary modes vanish completely.** With Z* ignored the min signed freq
   over all 512 q x 30 bands is +7.8e-7 cm-1 (the Gamma acoustic zero), with
   or without ASR. Both hypotheses left open in sec:cssni3 ("soft branch
   under-resolved" / "genuine instability between q points") are EXCLUDED.
2. **The 4^3 dyn was well converged and the ASR pass was repairing
   self-inflicted damage.** Without Z* the centered FCs already satisfy the
   ASR to 2.1e-10 (projection is a no-op). With Z* it is a 3.6e-3 relative
   perturbation. The "1.98% FC change" quoted in the report was never a
   statement about the SSCHA data -- it measured the truncation error of a
   dipole tail the potential does not have. This also explains why turning
   ASR OFF made 8 of 12 modes stable again: with Z* used, both options are
   wrong and ASR stacks a second error on the first.
3. **The Z*+ASR continuation does not reproduce its own input**: worst
   commensurate frequency residual 19.5 cm-1, i.e. evaluated at the coarse q
   points it was built from it returns frequencies off by up to 19.5 cm-1.
   (Eq. above is the identity only when C is pure centering; the ASR
   projection breaks the cancellation, violently, because it acts on a
   truncated long-range remainder.) In production this is hidden by
   `reuse_commensurate=True`, which copies the measured blocks verbatim at
   coarse points -- so the error is invisible exactly where it could have been
   caught and fully expressed on the off-grid shell where nothing checks it.

Off-grid displacement matches: over the 504 off-grid q, max |Z* - noZ*| freq
difference = 17.7 cm-1, rms 0.98 cm-1; over the 8 coarse q it is identically 0.

**Also contaminated:** `ForceTensor.get_phonons_in_qpath` builds a Tensor2
from dyn, so the "native 4^3 reference" curve in fig:cssni3-dispersion had the
same subtract/re-add cycle at every off-symmetry path point. The corrected
figure recomputes the reference under each convention rather than sharing one
curve. Script `cssni3_dispersion_longrange.py`, fig
`figs/cssni3_dispersion_longrange.pdf`, data `.npz`.

**API (opt-in, interpolation-scoped).**

    QSpaceTrilinearLanczos(ens, fine_mesh=(8,8,8), ignore_effective_charges=True)

forwards to `interpolate_dyn_fine(..., ignore_effective_charges=True)`.
Default stays False: for a polar potential the existing behaviour is correct
and silently dropping the dipole term would be a regression. Scope is narrow
on purpose -- Z*/eps are stripped on a LOCAL COPY of the dyn used to build the
Tensor2, so `ensemble.dyn` is untouched and IR intensities / IR response /
LO-TO at Gamma still see them. The flag disables a long-range *interpolation
model*, not the long-range physics of the observable.

**Error message.** Because this is silent on every commensurate check, the
diagnosis now lives in the error path. `QSpaceTrilinearLanczos` imaginary-mode
errors (`_unstable_message`) now report the on-grid vs off-grid split of the
affected q points (on-grid points are pinned to the SSCHA matrix and CANNOT be
an interpolation artifact -- that split alone localises the cause), state
whether Z* were used, ask if that was intended, explain the subtract/re-add
cycle, and name `ignore_effective_charges=True` for short-range MLIP forces.
When Z* were already ignored it lists the remaining candidates instead.
`allow_unstable=True` survives with a strengthened warning.

**Drive-by bug fixed:** `QSpaceInterpolation.py` had a function-local
`import warnings` inside the `d4_center == "leg"` branch, which shadowed the
module-level import and made every later `warnings.warn()` in that constructor
raise `UnboundLocalError` (test_d3_tensor_mode.py::test_forces_plain_windows
was failing on it).

**Status of the spectra.** sec:cssni3's 8^3 spectral comparisons are
superseded: they used a masked Hilbert space (12 spurious channels removed)
and an auxiliary dyn displaced by up to 17.7 cm-1 off-grid. Native 4^3 results
stand unchanged (they never touch the interpolation), including the
NumPy-downgrade runtime-regression diagnosis.

## 20. CORRECTED CsSnI3 8^3 production spectrum (2026-07-23)

Reran the full 12288-config 8^3 production Lanczos with
`ignore_effective_charges=True` AND `allow_unstable=False` (NumPy 2.4.6). It
did NOT raise -> **0 imaginary modes** over the 512 q x 30 fine-mesh
eigenvalues, so the full two-phonon space is available for the first time.
All 7 polarization chains: 100 steps, finite, `max|b| ~ 2.9e-6`,
`max|b-c| <= 1e-15`, and `b0 = 1.12e-7` matching the native 4^3 reference.

Comparison (`cssni3_full_lowenergy_compare.py` with the new env knobs
`CSSNI3_NEW_ABC` / `CSSNI3_ARCHIVE_NPZ` / `CSSNI3_COMPARE_STEM`; the old
defaults still reproduce the retracted masked figure). Fig
`figs/cssni3_full_lowenergy_noEC.pdf`, report fig:cssni3-lowenergy-noEC:

| smearing cm-1 | L1(4 vs 8) | W8/W4  | L1(8, 80 vs 100) |
|---------------|-----------|--------|-------------------|
| 2.0           | 0.028     | 1.005  | 0.048             |
| 1.0           | 0.078     | 1.007  | 0.170             |
| 0.5           | 0.160     | 1.008  | 0.354             |
| 0.25          | 0.249     | 1.009  | 0.533             |

Key points:
- **W8/W4 >= 1 now**: weight CONSERVED (and slightly enhanced by finer
  resolution). The masked run had W8/W4 = 0.984-0.9999, i.e. it LOST weight
  because 12 channels were deleted. Weight conservation is the sharpest
  signature that the fix is physical.
- Dominant soft-mode peak ~31.5 cm-1 agrees between 4^3 and 8^3 to 0.05 cm-1.
- 8^3 resolves low-energy structure at ~4.3, 8.5, 12.3 cm-1 that 4^3 merges
  -- the intended payoff (8^3 resolution at 4^3 cost).
- Lanczos-length caveat (sec 18, independent of the EC bug) SURVIVES on the
  corrected coefficients: L1(80 vs 100) still ~0.53 at 0.25 cm-1. Valid
  mesh-refinement comparison at/above 1 cm-1 smearing; narrowest features
  are still resolution-limited.

**IMPORTANT env trap (see numpy1_python314_qspace_issue.md, updated).** The
FIRST rerun was INVALID: it inherited the env's NumPy 1.26.4, which aliases
the masked metric products and returns `b0 = 4.5e-4` with `b - c` EXACTLY 0.
`b == c` does NOT detect the corruption; the coefficient magnitude does
(4.5e-4 >> two-phonon ceiling (440 cm-1)^2 = 4e-6 Ry^2). Fixes:
`QSpaceLanczos.check_numpy_version()` raises from `run_FT` under NumPy < 2
(covers the trilinear class by inheritance; test
`tests/test_qspace/test_numpy_guard.py`); the submit script prepends the
cached 2.4.6 to PYTHONPATH and forwards it with `mpirun -x PYTHONPATH`; the
driver calls `check_numpy()` too. Invalid outputs kept under `INVALID_numpy1_`
prefix in `data/`. The EC flag and the NumPy bug are ORTHOGONAL: with the
same ensemble/env, IGNORE_EC on vs off give b0 agreeing to 4 digits.
