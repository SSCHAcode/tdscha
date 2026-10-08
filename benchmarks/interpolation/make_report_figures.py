"""Recompute report figures from production operators; no cached test results.

Run from the checkout with its Python/Julia environment. Outputs vector PDF
figures, the plotted numerical data, and compact measured error summaries.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import polynomial_reference as ref
import production_validation as production

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'figures'


def check_sources():
    installed = Path(production.AF.__file__).resolve().parent
    hashes = {}
    for name in ('QSpaceAtomFourier.py', 'QSpaceLanczos.py', 'tdscha_qspace.jl'):
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[name] = digest(ROOT / 'Modules' / name)
        assert digest(installed / name) == hashes[name], 'Imported a different checkout'
    return hashes


def spectra(matrix, z, Q):
    eye = np.eye(len(matrix))
    sigma, pair = [], []
    for value in z:
        pair_inverse = np.linalg.inv(value**2*eye[1:, 1:]+matrix[1:, 1:])
        sigma.append(matrix[0, 1:] @ pair_inverse @ matrix[1:, 0])
        pair.append(-np.trace(pair_inverse).imag/np.pi/(len(matrix)-1))
    return np.array(sigma), np.array(pair)


def main():
    hashes = check_sources()
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'legend.fontsize': 8, 'axes.labelsize': 10})
    blue, orange, grey = '#17639b', '#ce6520', '#5a6269'
    z = np.linspace(.7, 2.9, 441)+.04j
    full, exact, errors = production.probe(9, 72, 0., embedded=True)
    bubble, _, _ = production.probe(9, 72, 0., embedded=True, ignore_v4=True)
    sigma, _ = spectra(full, z, 0.)
    sigma_exact = np.array([ref.response(72, 0., value)[0] for value in z])
    sigma_bubble, _ = spectra(bubble, z, 0.)
    assert ref.relative(sigma, sigma_exact) < 1e-10
    quartic, quartic_exact, pair_errors = production.probe(
        9, 36, 1/3, embedded=True, thermal=.3, ignore_v3=True)
    free, _, _ = production.probe(9, 36, 1/3, embedded=True, thermal=.3,
                                 ignore_v3=True, ignore_v4=True)
    _, pair = spectra(quartic, z, 1/3)
    _, pair_exact = spectra(quartic_exact, z, 1/3)
    _, pair_free = spectra(free, z, 1/3)
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.65), layout='constrained')
    ax[0].plot(z.real, -sigma_exact.imag, color=blue, label='Analytic fine: cubic + quartic')
    ax[0].plot(z.real[::14], -sigma.imag[::14], 'o', mfc='none', ms=4,
               color=orange, label='Production interpolation')
    ax[0].plot(z.real, -sigma_bubble.imag, '--', color=grey, label='Bubble only')
    ax[0].set(xlabel=r'Frequency $\omega$', ylabel=r'$-\mathrm{Im}\,\Sigma(\omega+i\eta)$',
              title=r'(a) Decay self-energy: $Q=0$, $N_f=72$')
    ax[0].legend(loc='upper left', frameon=False)
    ax[1].plot(z.real, pair_exact-pair_free, color=blue, label='Analytic fine')
    ax[1].plot(z.real[::14], (pair-pair_free)[::14], 'o', mfc='none', ms=4, color=orange,
               label='Production')
    ax[1].axhline(0, ls='--', color=grey, lw=1, label='Quartic off')
    ax[1].set(xlabel=r'Frequency $\omega$', ylabel=r'Quartic pair correction $\Delta A_4$',
              title=r'(b) Pure quartic: $Q=1/3$, $N_f=36$')
    ax[1].legend(loc='upper right', frameon=False)
    fig.savefig(OUT / 'response.pdf')
    plt.close(fig)

    sizes = [18, 36, 72]
    mesh_errors = []
    for nf in sizes:
        _, _, row = production.probe(9, nf, 1/3, embedded=True, thermal=.3)
        mesh_errors.append(row)
    controls = {}
    _, _, correct_control = production.probe(9, 18, 1/3, embedded=True)
    for mutation in ('missing_scale4', 'conjugate_kernel'):
        _, _, row = production.probe(9, 18, 1/3, embedded=True, check=False,
                                     mutation=mutation)
        controls[mutation] = row['D4']
    _, _, row = production.probe(3, 18, 0., check=False)
    controls['underresolved'] = row['D4']
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.65), layout='constrained')
    for key, label, marker, color in (
            ('D4', 'Quartic operator', 'o', blue),
            ('self_energy', 'Self-energy', 's', orange),
            ('pair_resolvent', 'Pair resolvent', '^', grey)):
        ax[0].semilogy(sizes, [r[key] for r in mesh_errors], marker=marker,
                       color=color, label=label)
    ax[0].axhline(1e-10, color='#777777', ls=':', lw=1)
    ax[0].text(19, 1.3e-10, r'$10^{-10}$ target', fontsize=8)
    ax[0].set(xlabel=r'Fine mesh size $N_f$ ($N_c=9$)', ylabel='Relative error',
              title=r'(a) Resolved-mesh accuracy', xticks=sizes,
              ylim=(1e-15, 1e-8))
    ax[0].legend(loc='center right', bbox_to_anchor=(1, .71), frameon=False)
    values = [correct_control['D4'], *controls.values()]
    ax[1].bar(range(4), values, color=[blue, orange, orange, orange])
    ax[1].set(yscale='log', ylabel='Quartic operator relative error',
              title='(b) Deliberately incorrect controls', ylim=(1e-15, 10),
              xticks=range(4), xticklabels=['Correct', 'No scale',
                                          'Wrong\nphase', 'Too small\nmesh'])
    ax[1].tick_params(axis='x', labelsize=9)
    for i, value in enumerate(values):
        ax[1].text(i, value*1.6, f'{value:.1e}', ha='center', fontsize=8)
    fig.savefig(OUT / 'accuracy.pdf')
    plt.close(fig)
    np.savez_compressed(OUT / 'simulation_data.npz', frequency=z.real, eta=z.imag[0],
                        sigma_production=sigma, sigma_analytic=sigma_exact,
                        sigma_bubble=sigma_bubble, pair_production=pair,
                        pair_analytic=pair_exact, pair_free=pair_free,
                        mesh_sizes=sizes,
                        mesh_errors=np.array([[r[k] for k in ('D4', 'self_energy', 'pair_resolvent')]
                                              for r in mesh_errors]),
                        negative_controls=values)
    summary = {'production_source_sha256': hashes, 'mesh_errors': mesh_errors,
               'negative_controls': controls,
               'full_self_energy_relative_error': ref.relative(sigma, sigma_exact),
               'quartic_effect_relative_to_bubble': ref.relative(sigma, sigma_bubble),
               'pure_quartic_response_error': ref.relative(pair, pair_exact)}
    (OUT / 'validation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'production_source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
