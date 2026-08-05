# TD-SCHA spectroscopy implementation report

This is the standalone Raman/IR report. It is intentionally separate from
`report/interpolation/` and has its own source, benchmark data, and figures,
including a configuration-scaling plot for the symmetry-reduced kernel.

Regenerate the evidence:

```bash
micromamba run -n sscha python spectroscopy_report/benchmark_spectroscopy.py
```

Compile from this directory:

```bash
lualatex -interaction=nonstopmode -halt-on-error tdscha_spectroscopy_report.tex
lualatex -interaction=nonstopmode -halt-on-error tdscha_spectroscopy_report.tex
```
