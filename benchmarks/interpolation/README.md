# Fourth order interpolation validation

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
