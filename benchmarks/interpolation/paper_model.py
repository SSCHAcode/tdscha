"""Stronger finite-range paper example, using the independent analytic oracle.

Run: OPENBLAS_NUM_THREADS=1 python benchmarks/interpolation/paper_model.py
Outputs: paper_model_results.json and paper_model_spectrum.png.
These finite-mesh spectra demonstrate interpolation, not converged lifetimes.
"""
import json
from pathlib import Path

import numpy as np
import polynomial_reference as ref


def main():
    # Keep the previously validated filters; increase interaction strengths.
    ref.CUBIC = [(0.5, np.array([1., .35]))]
    ref.QUARTIC = [(0.72, np.array([1., .37, -.29, .21])),
                   (0.24, np.array([1., -.23]))]
    nc, nf, Q = 9, 72, 0.
    z = np.linspace(.6, 1.65, 421)+.015j
    v, W = ref.vertices(nf, Q)
    vi, Wi = ref.interpolate_vertices(nc, nf, Q)
    result = dict(coarse_cells=nc, fine_cells=nf, Q=Q, eta=.015,
                  cubic=.5, quartic=[.72, .24],
                  D3_error=ref.relative(vi, v), D4_error=ref.relative(Wi, W))
    spectra, sigmas = {}, {}
    for quartic, name in [(False, 'bubble'), (True, 'RPA')]:
        direct = np.array([ref.response(nf, Q, zz, quartic=quartic) for zz in z])
        interpolated = np.array([ref.response(nf, Q, zz, vi, Wi, quartic)
                                 for zz in z])
        result[name+'_interpolation_error'] = ref.relative(interpolated, direct)
        assert result[name+'_interpolation_error'] < 1e-11
        spectra[name] = -direct[:, 1].imag/np.pi
        sigmas[name] = direct[:, 0]
        result[name+'_peak_frequency'] = float(z[spectra[name].argmax()].real)
    result['spectral_relative_change'] = ref.relative(spectra['RPA'], spectra['bubble'])
    result['self_energy_relative_change'] = ref.relative(sigmas['RPA'], sigmas['bubble'])
    # Discrete static stability diagnostic, not a claim of global equilibrium.
    for quartic, name in [(False, 'bubble'), (True, 'RPA')]:
        result[name+'_minimum_sampled_static_curvature'] = float(min(
            ref.frequency(q)**2 + ref.response(nf, q, 0j, quartic=quartic)[0].real
            for q in np.arange(nf)/nf))
        assert result[name+'_minimum_sampled_static_curvature'] > 0
    assert result['spectral_relative_change'] > .2
    out = Path(__file__).resolve().parent
    (out/'paper_model_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for name, curve in spectra.items():
        ax.plot(z.real, curve, label=name)
    sample = np.arange(0, len(z), 12)
    interp = np.array([ref.response(nf, Q, zz, vi, Wi)[1] for zz in z[sample]])
    ax.plot(z.real[sample], -interp.imag/np.pi, 'o', ms=3, fillstyle='none',
            label='RPA interpolated 9 to 72')
    ax.set(xlabel='Frequency (reference units)', ylabel=r'$-\mathrm{Im}\,G/\pi$',
           title=r'Finite-range optical chain: $Q=0$, $T=0$, $\eta=0.015$')
    ax.legend()
    fig.savefig(out/'paper_model_spectrum.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
