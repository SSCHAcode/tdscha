"""Minimal non-orthogonal 3D crystal for the Nyquist-tie tests.

The atom-centred Fourier kernel is built from geometry alone (atom
positions + cell metric) at construction time, so these tests only need a
class instance on a genuinely non-orthogonal cell; the ensemble forces are
irrelevant to the kernel.  The model is a two-atom cell on a primitive
bcc-like (non-orthogonal, 109.47-degree) lattice with an isotropic
nearest-neighbour spring network, which gives a stable harmonic dynamical
matrix (three acoustic modes at Gamma, everything else positive).
"""
from __future__ import print_function

import numpy as np

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.symmetries
import cellconstructor.Units

import sscha
import sscha.Ensemble


A_LAT = 3.0
MASS0 = 1200.0
MASS1 = 2600.0
# Atom-1 fractional offset: d = tau0 - tau1 = -(0.5, 0.5, 0.0), whose
# aliasing class (0,0,1) is the metric/separable disagreement case on this
# primitive bcc cell (separable manufactures a (0,0,+-1) tie; the metric
# selects a single image).
TAU1 = np.array([0.5, 0.5, 0.0])
K_SPRING = np.array([0.16, 0.13, 0.10])   # per-Cartesian isotropic-ish


def build_unit_structure():
    s = CC.Structure.Structure(2)
    s.unit_cell = A_LAT * np.array([[-1.0, 1.0, 1.0],
                                    [1.0, -1.0, 1.0],
                                    [1.0, 1.0, -1.0]])
    s.coords[0] = np.zeros(3)
    s.coords[1] = TAU1 @ s.unit_cell
    s.atoms = ["A", "B"]
    s.masses = {"A": MASS0, "B": MASS1}
    s.has_unit_cell = True
    return s


def _nearest_bonds(super_struct, n_neigh=8):
    """Shortest inter-atomic bonds (minimum image) of the supercell."""
    coords = super_struct.coords
    cell = super_struct.unit_cell
    nat = super_struct.N_atoms
    inv = np.linalg.inv(cell)
    bonds = []
    for i in range(nat):
        d2 = []
        for j in range(nat):
            if i == j:
                continue
            frac = (coords[j] - coords[i]) @ inv
            frac -= np.round(frac)
            cart = frac @ cell
            d2.append((float(cart @ cart), j))
        d2.sort()
        for _, j in d2[:n_neigh]:
            if i < j:
                bonds.append((i, j))
    return sorted(set(bonds))


def build_dyn(supercell=(2, 2, 2)):
    unit = build_unit_structure()
    super_struct = unit.generate_supercell(supercell)
    nat = super_struct.N_atoms
    bonds = _nearest_bonds(super_struct)

    fc = np.zeros((3 * nat, 3 * nat))
    for i, j in bonds:
        for a in range(3):
            k = K_SPRING[a]
            fc[3 * i + a, 3 * j + a] += -k
            fc[3 * j + a, 3 * i + a] += -k
            fc[3 * i + a, 3 * i + a] += k
            fc[3 * j + a, 3 * j + a] += k

    q_tot = CC.symmetries.GetQGrid(unit.unit_cell, supercell)
    q_tot = [np.array(q) for q in q_tot]
    dynq = CC.Phonons.GetDynQFromFCSupercell(
        fc, np.array(q_tot), unit, super_struct)

    dyn = CC.Phonons.Phonons(unit, nqirr=len(q_tot))
    dyn.q_tot = q_tot
    dyn.dynmats = [dynq[i] for i in range(len(q_tot))]
    dyn.q_stars = [[np.array(q)] for q in q_tot]
    dyn.AdjustQStar()
    return dyn


def make_ensemble(dyn, T=250.0, N=40, seed=0):
    """A trivial ensemble (harmonic forces): only geometry matters here."""
    np.random.seed(seed)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.generate(N)
    ens.forces = np.zeros_like(ens.forces)
    ens.energies = np.zeros(N)
    ens.force_computed = np.ones(N, dtype=bool)
    ens.init()
    return ens
