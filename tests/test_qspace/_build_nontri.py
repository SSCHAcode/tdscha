"""
Builder for a small NON-time-reversal-invariant test system.

A diatomic chain with a 1x1x3 supercell gives q = 0, 1/3, 2/3 along z.
The points 1/3 and 2/3 = -1/3 are *distinct* (non-TRI), which is the path the
all-TRI 2x2x2 dataset never exercises in the anharmonic q-space code.

The model is an isotropic diatomic spring chain (guaranteed positive definite,
exact acoustic sum rule).  We then build a SSCHA ensemble and give it forces
with a deterministic cubic+quartic anharmonicity so that D3/D4 averages are
non-zero.
"""
from __future__ import print_function
import numpy as np

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.symmetries
import cellconstructor.Units

import sscha, sscha.Ensemble

SUPERCELL = (1, 1, 3)
# anisotropic spring constants (Ry/Bohr^2) per Cartesian component
# (distinct per direction -> non-degenerate modes, clean mode matching).
K1 = np.array([0.18, 0.225, 0.15])   # intracell
K2 = np.array([0.10, 0.07, 0.16])    # intercell
MASS_A = 1000.0   # in Ry atomic units (m_e); distinct masses -> optical gap
MASS_B = 2500.0


def build_unit_structure():
    s = CC.Structure.Structure(2)
    # Triclinic cell with generic atomic positions -> P1 (point group = identity
    # only).  With no point-group symmetry the anharmonic force model cannot
    # accidentally break a symmetry the codes assume; the only symmetry left is
    # translational (momentum conservation), which both codes must reproduce.
    s.unit_cell = np.array([[5.0, 0.0, 0.0],
                            [0.7, 6.0, 0.0],
                            [0.9, 0.6, 7.0]])
    s.coords[0] = np.array([0.10, 0.15, 0.00])
    s.coords[1] = np.array([0.37, 0.21, 2.60])  # generic, along the chain (z)
    s.atoms = ["A", "B"]
    s.masses = {"A": MASS_A, "B": MASS_B}
    s.has_unit_cell = True
    return s


def get_bonds(super_struct, itau):
    """Return the list of (i, j, k_vec) bonds of the diatomic chain.

    Each A atom bonds to its nearest B (k1, intracell) and second-nearest B
    (k2, intercell), using the minimum image along the chain.
    """
    nat = super_struct.N_atoms
    coords = super_struct.coords
    cell = super_struct.unit_cell

    def mindist(i, j):
        d = coords[j] - coords[i]
        frac = CC.Methods.covariant_coordinates(cell, d[None, :])[0]
        frac -= np.round(frac)
        return frac @ cell

    A_atoms = [i for i in range(nat) if itau[i] == 0]
    B_atoms = [i for i in range(nat) if itau[i] == 1]
    bonds = []
    for i in A_atoms:
        order = sorted(B_atoms, key=lambda j: np.linalg.norm(mindist(i, j)))
        bonds.append((i, order[0], K1))
        bonds.append((i, order[1], K2))
    return bonds


def build_supercell_fc(super_struct, itau):
    """Diatomic anisotropic spring chain force constants in the supercell."""
    nat = super_struct.N_atoms
    fc = np.zeros((3 * nat, 3 * nat))
    for i, j, k in get_bonds(super_struct, itau):
        for a in range(3):
            fc[3 * i + a, 3 * j + a] += -k[a]
            fc[3 * j + a, 3 * i + a] += -k[a]
            fc[3 * i + a, 3 * i + a] += k[a]
            fc[3 * j + a, 3 * j + a] += k[a]
    return fc


def build_dyn():
    unit = build_unit_structure()
    super_struct = unit.generate_supercell(SUPERCELL)
    itau = super_struct.get_itau(unit) - 1

    fc_sc = build_supercell_fc(super_struct, itau)

    q_tot = CC.symmetries.GetQGrid(unit.unit_cell, SUPERCELL)
    q_tot = [np.array(q) for q in q_tot]

    dynq = CC.Phonons.GetDynQFromFCSupercell(
        fc_sc, np.array(q_tot), unit, super_struct)

    dyn = CC.Phonons.Phonons(unit, nqirr=len(q_tot))
    dyn.q_tot = [np.array(q) for q in q_tot]
    dyn.dynmats = [dynq[i] for i in range(len(q_tot))]
    # Treat every q as its own star (avoid symmetry q-star machinery).
    dyn.q_stars = [[np.array(q)] for q in q_tot]
    dyn.AdjustQStar()
    return dyn


def tri_status(dyn):
    q = np.array(dyn.q_tot)
    bg = dyn.structure.get_reciprocal_vectors() / (2 * np.pi)
    out = []
    for i, qq in enumerate(q):
        d = CC.Methods.get_min_dist_into_cell(bg, qq, -qq)
        out.append((i, qq, d < 1e-6))
    return out


def make_ensemble(dyn, T, N, seed=0, g3=0.6, g4=1.5):
    """Generate a SSCHA ensemble and assign BOND-based anharmonic forces.

    Each bond (a, b) gets an anharmonic energy g3/3 s^3 + g4/4 s^4 in the bond
    stretch s = u_a - u_b (per Cartesian component).  Because the anharmonicity
    lives on the bonds (like the harmonic springs), the residual force
    forces - sscha_forces has weight at ALL q-points, including q = +-1/3.
    This is essential to exercise the (q, -q) off-diagonal anharmonic blocks.
    """
    np.random.seed(seed)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.generate(N)

    u = ens.u_disps.copy()                      # Angstrom, (N, 3*nat_sc)
    super_struct = dyn.structure.generate_supercell(dyn.GetSupercell())
    itau = super_struct.get_itau(dyn.structure) - 1
    bonds = get_bonds(super_struct, itau)

    u_bohr = u * CC.Units.A_TO_BOHR             # Bohr
    nat = super_struct.N_atoms
    f_harm = np.zeros_like(u_bohr)              # harmonic bond force, Ry/Bohr
    f_anh = np.zeros_like(u_bohr)               # anharmonic bond force, Ry/Bohr

    for (i, j, k) in bonds:
        for a in range(3):
            ia, ja = 3 * i + a, 3 * j + a
            s = u_bohr[:, ia] - u_bohr[:, ja]   # bond stretch
            fh = k[a] * s                       # harmonic
            fa = g3 * s ** 2 + g4 * s ** 3      # anharmonic part of dE/ds
            f_harm[:, ia] += -fh
            f_harm[:, ja] += +fh
            f_anh[:, ia] += -fa
            f_anh[:, ja] += +fa

    # Centre the anharmonic force so <forces - sscha_forces> = 0, i.e. mimic a
    # SSCHA stationary point.  Otherwise the (large) net force exercises the
    # Gamma-only mean-force subtraction, which is irrelevant to converged runs.
    f_anh -= np.mean(f_anh, axis=0, keepdims=True)
    f_tot = (f_harm + f_anh) * CC.Units.A_TO_BOHR  # Ry/Angstrom
    ens.forces = f_tot.reshape(N, nat, 3)
    ens.energies = np.zeros(N)
    ens.force_computed = np.ones(N, dtype=bool)

    # CRITICAL: refresh the q-space arrays. ens.generate() precomputes
    # u_disps_qspace and leaves forces_qspace = 0; assigning ens.forces
    # afterwards does NOT update them, and QSpaceLanczos skips ens.init()
    # when u_disps_qspace is already present. Without this call the q-space
    # code silently uses STALE ZERO forces (this was the origin of the
    # "force-Parseval artifact" previously attributed to the hand-built dyn).
    ens.init()
    return ens


if __name__ == "__main__":
    dyn = build_dyn()
    w, p, wq, pq = dyn.DiagonalizeSupercell(return_qmodes=True)
    print("supercell", dyn.GetSupercell(), "nq", len(dyn.q_tot))
    for i, q, tri in tri_status(dyn):
        print("  iq={} q={} TRI={}  freqs[cm-1]={}".format(
            i, np.round(q, 4), tri, np.round(wq[:, i] * CC.Units.RY_TO_CM, 2)))
    nontri = sum(1 for _, _, tri in tri_status(dyn) if not tri)
    print("NON-TRI q-points:", nontri)
    print("min supercell freq cm-1:", np.min(w) * CC.Units.RY_TO_CM)
