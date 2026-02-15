# DynamicalLanczos Module

::: tdscha.DynamicalLanczos
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      members_order: source

## Computation Mode Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MODE_SLOW_SERIAL` | 0 | Pure Python implementation (testing only) |
| `MODE_FAST_SERIAL` | 1 | C extension with OpenMP |
| `MODE_FAST_MPI` | 2 | C extension with MPI parallelization |
| `MODE_FAST_JULIA` | 3 | Julia extension (fastest - default if julia is present) |

## Lanczos Class

::: tdscha.DynamicalLanczos.Lanczos
    options:
      heading_level: 2
      show_source: true
      members_order: source
      filters: ["!^_", "!^__"]


### Performance Considerations

- Use `mode=MODE_FAST_JULIA` if Julia is available (2-10× speedup)
- For large systems, use MPI parallelization with `mode=MODE_FAST_MPI`
- Enable `gamma_only=True` for Γ-point-only calculations
- Use `select_modes` to exclude high-frequency modes if not needed
