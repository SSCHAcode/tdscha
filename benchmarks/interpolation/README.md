# Fourth order interpolation validation

## Paper example with a visible spectral effect

Rechecked on 2026-10-15 after merging main at `17a0dd6e`: **62 tests
passed**, including the stronger-model production checks and four static
oracle regressions. Eleven existing CellConstructor complex-casting warnings
were emitted. The analytic oracle now handles exactly zero frequency and
degenerate pairs, including the thermal Bose derivative in the static limit.

`paper_model.py` retains the same finite-range filters and sets the cubic
coefficient to 0.5 and quartic coefficients to 0.72 and 0.24. Run:

    OPENBLAS_NUM_THREADS=1 python benchmarks/interpolation/paper_model.py

The generated [spectrum](paper_model_spectrum.png) and
[measurements](paper_model_results.json) show a 37.7% relative Euclidean-norm
change in the spectral curve on the stated frequency window at Q=0, T=0,
Nf=72 and broadening 0.015. Both bubble and RPA interpolation errors are
below 1e-14. These are finite-mesh results, not converged lifetimes.
[The paper model derivation](paper_model.tex) states the potential,
counterterms, support condition, and precise exactness claim.

The current production method refines **internal momenta only**; external Q
must lie on the coarse mesh. For this three-cell cluster span, Nc > 6 is a
sufficient centered-Fourier resolution condition. Keep the auxiliary matrix
and potential coefficients fixed across meshes. This excludes the change in
the quartic loop with mesh size, rather than removing the reference loop
renormalization already contained in the auxiliary matrix.

## Original validation

**Validated:** the production atom Fourier operator reproduces an independent
analytic fourth-order response when the coarse mesh resolves the interaction
range. All **56 tests pass**, including 16 new deterministic regressions.

The benchmark uses a bounded cubic-plus-quartic polynomial in molecular internal
displacements, with a fixed three-cell cutoff. Its quartic term includes four
distinct cells, not just onsite or bond interactions. The optical dispersion
omega(q)^2 = 1 + 0.8 cos(2 pi q) admits a two-phonon decay channel at Q=0.
The exact coefficients and energy/force implementation are in
`polynomial_reference.py`.

Gaussian counterterms cancel the mean anharmonic force and curvature. The
potential and harmonic reference remain fixed during mesh refinement: there
is **no separate SSCHA minimization**. Degree-seven Gaussian quadrature removes
sampling error while exercising the actual production Fourier fold, Julia
force contraction, reconstruction, and full operator. Fine reference vertices
come directly from the same polynomial's analytic derivatives.

Tests compare cubic and quartic operators separately, bubble and resummed
self-energies, one-phonon response, and the isolated quartic pair resolvent.
They cover nonzero momentum, finite-temperature sum/difference channels,
9→9/18/36/72 and 8→24 meshes, unequal masses, rebased atoms, dense complex mode
mixing, and acoustic masks. Vertex and resolvent errors satisfy 1e-10 tolerances;
self-energy errors satisfy 1e-8.

**Why interpolation works:** at fixed external momentum, the finite-range quartic
vertex is a trigonometric polynomial in two independent internal momenta.
Production interpolates both legs, with normalization Nc/Nf, reconstructing
the resolved tensor before response inversion. Wrong normalization, wrong
conjugation, and insufficient coarse resolution all fail the tests. Quartic
resummation changes the reference self-energy by 7–14%. Interpolating an
already computed response instead gives 31% error in the selected off-grid
control: the response itself is not finite range.

**Fixed:** `ignore_v3` previously removed only one direction of cubic coupling.
Both q-space classes now remove both; regression tests verify pure-quartic
isolation. No defect was found in the resolved-support fourth-order
interpolation. Thermodynamic lifetime extraction still requires mesh and
broadening convergence, separately from this operator validation.

Reproduce with this checkout installed and Julia available:

    python -m pytest -q tests/test_atom_fourier tests/test_interpolation tests/test_qspace/test_hessian_v3v4_flags.py

Illustrated report: [LaTeX source](fourth_order_validation.tex) and
[compiled PDF](../../output/pdf/fourth_order_validation.pdf).
Regenerate the figures with `python benchmarks/interpolation/make_report_figures.py`;
the plotted arrays and production-source hashes are saved in `figures/`.

Undergraduate explanation with explicit equations and no auxiliary shorthand:
[LaTeX source](fourth_order_validation_explained.tex) and
[expanded PDF](../../output/pdf/fourth_order_validation_explained.pdf).
