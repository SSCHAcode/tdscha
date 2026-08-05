# QSpaceAtomFourier Module

Atom-centred Fourier interpolation of the q-space TD-SCHA Lanczos d3 and d4
operators.

The constructor accepts the same `lo_to_split=None`, `"random"`, or explicit
three-vector convention as `QSpaceLanczos`. Commensurate fine points,
including Gamma, are pinned to that parent basis; noncommensurate points keep
CellConstructor's tensorial dipole--dipole interpolation. With
`ignore_effective_charges=True`, the harmonic interpolation omits both that
tail and its Gamma LO--TO limit, but the caller's dynamical matrix retains its
effective charges for IR response vertices.

::: tdscha.QSpaceAtomFourier
    options:
      show_root_heading: true
      members:
        - QSpaceAtomFourierLanczos
        - load_distributed_atom_fourier_tdscha
