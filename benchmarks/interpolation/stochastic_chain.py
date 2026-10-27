"""Stochastic TD-SCHA on the anharmonic chain through the public tdscha API.

The chain of polynomial_reference.py is embedded in a real crystal: two atoms
per cell (masses M and 2M), supercell (1, 1, Nc) along z, and the scalar chain
coordinate is the relative displacement along x of the M atom of cell l and
its nearest 2M atom (assigned to cell l-1). The potential depends on nothing
else, so the other five bands have exactly zero frequency; they are removed
by the standard mode exclusion (ensemble.ignore_small_w): locked in the
ensemble generation and masked in the q-space Lanczos.

Everything below goes through CellConstructor, sscha.Ensemble.generate and
QSpaceAtomFourierLanczos (prepare_mode_q / run_FT / continued fraction).
The only model-specific code is the force evaluation. The analytic oracle
is polynomial_reference.response, i.e. the bubble and RPA self-energies on
the fine mesh, in units hbar = M_r = omega_0 = 1.
"""
import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Structure
import cellconstructor.symmetries
import cellconstructor.Units
import sscha.Ensemble

import polynomial_reference as ref

CELL = 3.0                      # Angstrom
TAU2 = np.array([0.13, 0.21, 1.37])
MASS = 2000.0                   # Ry atomic units; the second atom has 2 * MASS
OMEGA0 = 1e-3                   # Ry
J1 = 0.4
STRONG = dict(cubic=[(0.5, np.array([1., .35]))],
              quartic=[(0.72, np.array([1., .37, -.29, .21])),
                       (0.24, np.array([1., -.23]))])


def reduced_mass():
    return 2 * MASS / 3


def length_unit():
    """x_0 = sqrt(hbar / (M_r omega_0)) in Bohr."""
    return 1 / np.sqrt(reduced_mass() * OMEGA0)


def unit_structure():
    s = CC.Structure.Structure(2)
    s.unit_cell = CELL * np.eye(3)
    s.coords[0] = 0.
    s.coords[1] = TAU2 * CELL
    s.atoms = ["A", "B"]
    s.masses = {"A": MASS, "B": 2 * MASS}
    s.has_unit_cell = True
    return s


def chain_projector(unit, supercell):
    """B[l, 3 i + a] such that x_l = sum B u (u in Bohr, along x)."""
    nc = supercell[2]
    big = unit.generate_supercell(supercell)
    itau = big.get_itau(unit) - 1
    frac = np.linalg.solve(unit.unit_cell.T, (big.coords - unit.coords[itau]).T).T
    cell = np.round(frac[:, 2]).astype(int) % nc
    index = {(k, n): i for i, (k, n) in enumerate(zip(itau, cell))}
    B = np.zeros((nc, 3 * big.N_atoms))
    for n in range(nc):
        B[n, 3 * index[0, n]] = 1.
        B[n, 3 * index[1, (n - 1) % nc]] = -1.
    return big, B


def build_dyn(nc):
    """Harmonic part of V on the (1, 1, nc) supercell."""
    unit = unit_structure()
    supercell = (1, 1, nc)
    big, B = chain_projector(unit, supercell)
    K = np.eye(nc) + J1 * (np.roll(np.eye(nc), 1, 0) + np.roll(np.eye(nc), -1, 0))
    fc = reduced_mass() * OMEGA0**2 * B.T @ K @ B
    q_tot = [np.array(q) for q in CC.symmetries.GetQGrid(unit.unit_cell, supercell)]
    dynq = CC.Phonons.GetDynQFromFCSupercell(fc, np.array(q_tot), unit, big)
    dyn = CC.Phonons.Phonons(unit, nqirr=len(q_tot))
    dyn.q_tot = q_tot
    dyn.dynmats = [dynq[i] for i in range(len(q_tot))]
    dyn.q_stars = [[q] for q in q_tot]
    dyn.AdjustQStar()
    return dyn


def chain_forces(x, sigma2):
    """polynomial_reference.energy_force forces, vectorized over configurations."""
    force = -(x + J1 * (np.roll(x, 1, -1) + np.roll(x, -1, -1)))
    for i, (g, h) in enumerate(ref.CUBIC + ref.QUARTIC):
        s = sum(hr * np.roll(x, -r, -1) for r, hr in enumerate(h))
        ds = g/2*(s*s - sigma2[i]) if i < len(ref.CUBIC) else g/6*(s**3 - 3*sigma2[i]*s)
        for r, hr in enumerate(h):
            force -= hr * np.roll(ds, r, -1)
    return force


def make_ensemble(dyn, n_configs, seed=0, couplings=STRONG):
    """T = 0 SSCHA ensemble of the auxiliary dyn, with forces from V."""
    ref.CUBIC, ref.QUARTIC = couplings['cubic'], couplings['quartic']
    nc = dyn.GetSupercell()[2]
    _, B = chain_projector(dyn.structure, dyn.GetSupercell())
    np.random.seed(seed)
    ens = sscha.Ensemble.Ensemble(dyn, 0.)
    ens.ignore_small_w = True
    ens.generate(n_configs)
    x0 = length_unit()
    u = ens.u_disps * CC.Units.A_TO_BOHR                        # Bohr
    # Fixed potential parameters: cluster variances in the coarse T = 0 Gaussian.
    sigma2 = ref.counterterms(master_size=nc)
    force_x = chain_forces((u @ B.T) / x0, sigma2)
    forces = (force_x @ B) * (OMEGA0 / x0) * CC.Units.A_TO_BOHR  # Ry/Angstrom
    ens.forces = forces.reshape(n_configs, -1, 3)
    ens.energies = np.zeros(n_configs)
    ens.force_computed = np.ones(n_configs, dtype=bool)
    ens.init()
    return ens


def load_or_make_ensemble(dyn, n_configs, seed, directory, couplings=STRONG):
    """Load the ensemble saved by sscha (save_bin) in directory, or make and save it."""
    from pathlib import Path
    if (Path(directory) / 'energies_pop1.npy').is_file():
        ref.CUBIC, ref.QUARTIC = couplings['cubic'], couplings['quartic']
        ens = sscha.Ensemble.Ensemble(dyn, 0.)
        ens.ignore_small_w = True
        ens.load_bin(str(directory), 1)
        assert ens.N == n_configs, (ens.N, n_configs)
        return ens
    ens = make_ensemble(dyn, n_configs, seed, couplings)
    Path(directory).mkdir(parents=True, exist_ok=True)
    ens.save_bin(str(directory), 1)
    return ens


def spectral_function(ens, fine, frequencies, smearing, quartic=True, steps=None):
    """-Im G(omega + i eta)/pi of the Gamma chain phonon, in units of omega_0."""
    import tdscha.QSpaceAtomFourier as AF
    lanczos = AF.QSpaceAtomFourierLanczos(ens, fine_mesh=(1, 1, fine))
    lanczos.ignore_v4 = not quartic
    lanczos.init(use_symmetries=False)
    band = int(np.argmax(lanczos.w_q[:, 0]))
    lanczos.prepare_mode_q(0, band)
    # Pair space at Q = 0 has at most (fine // 2 + 1) pairs per sector.
    lanczos.run_FT(steps or 2 * (fine // 2 + 1) + 1, verbose=False)
    G = lanczos.get_green_function_continued_fraction(
        frequencies * OMEGA0, use_terminator=False, smearing=smearing * OMEGA0)
    return -np.imag(G) * OMEGA0**2 / np.pi


def analytic_spectral_function(fine, frequencies, smearing, quartic=True, couplings=STRONG):
    ref.CUBIC, ref.QUARTIC = couplings['cubic'], couplings['quartic']
    z = frequencies + 1j * smearing
    return np.array([-ref.response(fine, 0., zz, quartic=quartic)[1].imag / np.pi for zz in z])
