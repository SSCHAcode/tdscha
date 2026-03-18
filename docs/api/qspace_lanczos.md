# QSpaceLanczos Module

::: tdscha.QSpaceLanczos
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      members_order: source

## QSpaceLanczos Class

::: tdscha.QSpaceLanczos.QSpaceLanczos
    options:
      heading_level: 2
      show_source: true
      members_order: source
      filters: ["!^_", "!^__"]

### Key Differences from Lanczos

| Feature | `Lanczos` (real-space) | `QSpaceLanczos` (q-space) |
|---------|----------------------|--------------------------|
| Psi vector | Real (`float64`) | Complex (`complex128`) |
| Inner product | Standard dot product | Hermitian: $\langle p | q \rangle = \bar{p}^T (q \cdot m)$ |
| Lanczos type | Bi-conjugate or symmetric | Hermitian symmetric ($b = c$, real coefficients) |
| Two-phonon pairs | All supercell mode pairs | $(q_1, q_2)$ pairs with $q_1 + q_2 = q_\text{pert}$ |
| Symmetries | Full space group | Point group only (translations analytic) |
| Backend | C / MPI / Julia | Julia only |

### Utility Functions

::: tdscha.QSpaceLanczos.find_q_index
    options:
      heading_level: 3
      show_source: true
