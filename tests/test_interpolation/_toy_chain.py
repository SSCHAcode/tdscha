"""
Anharmonic diatomic-chain toy model, parametrized by the supercell length.

The model is defined by its TOPOLOGY (not by distances), so that every
supercell length L represents exactly the same infinite chain:

    ... A(n) --K1-- B(n) --K2-- A(n+1) ...      (chain along z)

Each bond carries, per Cartesian component, the energy

    E(s) = 1/2 k s^2 + g3/3 s^3 + g4/4 s^4,     s = u_first - u_second

with the bond ends ordered consistently (A(n), B(n)) and (B(n), A(n+1)),
so the cubic term has the same sign in every cell and supercell.

Because the harmonic dynamical matrix is built from the same springs, the
force residual (forces - sscha_forces) is EXACTLY the anharmonic bond force:
no harmonic sampling noise enters the D3/D4 estimators.

The unit cell is triclinic with generic positions (P1: identity is the only
point-group operation), so no symmetry can mask convention errors.
"""
from __future__ import print_function
import numpy as np

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.symmetries
import cellconstructor.Units

import sscha, sscha.Ensemble

# anisotropic spring constants (Ry/Bohr^2) per Cartesian component
K1 = np.array([0.18, 0.225, 0.15])   # A(n) - B(n)
K2 = np.array([0.10, 0.07, 0.16])    # B(n) - A(n+1)
MASS_A = 1000.0   # Ry atomic units
MASS_B = 2500.0


def build_unit_structure():
    s = CC.Structure.Structure(2)
    s.unit_cell = np.array([[5.0, 0.0, 0.0],
                            [0.7, 6.0, 0.0],
                            [0.9, 0.6, 7.0]])
    s.coords[0] = np.array([0.10, 0.15, 0.00])
    s.coords[1] = np.array([0.37, 0.21, 2.60])
    s.atoms = ["A", "B"]
    s.masses = {"A": MASS_A, "B": MASS_B}
    s.has_unit_cell = True
    return s


def get_bonds(super_struct, unit, L):
    """Chain bonds by topology: (index_A(n), index_B(n), K1) and
    (index_B(n), index_A(n+1 mod L), K2).

    The supercell atom of unit-cell atom a in cell n is identified through
    itau and the integer cell index along z.
    """
    itau = super_struct.get_itau(unit) - 1
    nat_sc = super_struct.N_atoms
    # integer cell index of each supercell atom (fractional coords along z
    # of the cell origin)
    r_lat = super_struct.coords - unit.coords[itau]
    frac = np.linalg.solve(unit.unit_cell.T, r_lat.T).T  # cell indices (float)
    n_z = np.round(frac[:, 2]).astype(int) % L

    idx_A = {}
    idx_B = {}
    for k in range(nat_sc):
        if itau[k] == 0:
            idx_A[n_z[k]] = k
        else:
            idx_B[n_z[k]] = k

    bonds = []
    for n in range(L):
        bonds.append((idx_A[n], idx_B[n], K1))
        bonds.append((idx_B[n], idx_A[(n + 1) % L], K2))
    return bonds


def get_triplets(super_struct, unit, L):
    """A-atom triplets (A(n), A(n+1), A(n+2)) for the three-body term."""
    itau = super_struct.get_itau(unit) - 1
    r_lat = super_struct.coords - unit.coords[itau]
    frac = np.linalg.solve(unit.unit_cell.T, r_lat.T).T
    n_z = np.round(frac[:, 2]).astype(int) % L
    idx_A = {}
    for k in range(super_struct.N_atoms):
        if itau[k] == 0:
            idx_A[n_z[k]] = k
    return [(idx_A[n], idx_A[(n + 1) % L], idx_A[(n + 2) % L])
            for n in range(L)]


def build_dyn(L):
    """Harmonic spring-chain dyn on the (1, 1, L) supercell."""
    unit = build_unit_structure()
    supercell = (1, 1, L)
    super_struct = unit.generate_supercell(supercell)
    bonds = get_bonds(super_struct, unit, L)

    nat = super_struct.N_atoms
    fc = np.zeros((3 * nat, 3 * nat))
    for i, j, k in bonds:
        for a in range(3):
            fc[3 * i + a, 3 * j + a] += -k[a]
            fc[3 * j + a, 3 * i + a] += -k[a]
            fc[3 * i + a, 3 * i + a] += k[a]
            fc[3 * j + a, 3 * j + a] += k[a]

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


def make_ensemble(dyn, T, N, seed=0, g3=0.6, g4=0.0, g3b=0.0):
    """SSCHA ensemble with deterministic bond anharmonicity.

    forces = harmonic bond force + anharmonic bond force. The harmonic part
    coincides with the SSCHA force by construction, so the residual is the
    pure anharmonic force.

    With g4=0 (default) the model is purely cubic: <d^2 V_anh> = D3 <u> = 0,
    so the SSCHA stationarity assumed by the vertex rescaling holds exactly
    in expectation.

    g3b adds a THREE-BODY cubic term per cell and Cartesian component,
        V_3b = g3b * s1^2 * s2,
        s1 = uA(n+1) - uA(n),  s2 = uA(n+2) - uA(n+1),
    whose Phi3 entries span three distinct cells.
    """
    np.random.seed(seed)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.generate(N)

    unit = dyn.structure
    L = dyn.GetSupercell()[2]
    super_struct = unit.generate_supercell(dyn.GetSupercell())
    bonds = get_bonds(super_struct, unit, L)

    u_bohr = ens.u_disps.copy() * CC.Units.A_TO_BOHR   # Bohr
    nat = super_struct.N_atoms
    f_harm = np.zeros_like(u_bohr)   # Ry/Bohr
    f_anh = np.zeros_like(u_bohr)

    for (i, j, k) in bonds:
        for a in range(3):
            ia, ja = 3 * i + a, 3 * j + a
            s = u_bohr[:, ia] - u_bohr[:, ja]
            fh = k[a] * s
            fa = g3 * s ** 2 + g4 * s ** 3
            f_harm[:, ia] += -fh
            f_harm[:, ja] += +fh
            f_anh[:, ia] += -fa
            f_anh[:, ja] += +fa

    if g3b != 0.0:
        for (i, j, k) in get_triplets(super_struct, unit, L):
            for a in range(3):
                ia, ja, ka = 3 * i + a, 3 * j + a, 3 * k + a
                s1 = u_bohr[:, ja] - u_bohr[:, ia]
                s2 = u_bohr[:, ka] - u_bohr[:, ja]
                dv1 = 2.0 * g3b * s1 * s2       # dV/ds1
                dv2 = g3b * s1 ** 2             # dV/ds2
                f_anh[:, ia] += dv1
                f_anh[:, ja] += dv2 - dv1
                f_anh[:, ka] += -dv2

    # Remove the ensemble-average anharmonic force (mimic SSCHA stationarity)
    f_anh -= np.mean(f_anh, axis=0, keepdims=True)

    f_tot = (f_harm + f_anh) * CC.Units.A_TO_BOHR      # Ry/Angstrom
    ens.forces = f_tot.reshape(N, nat, 3)
    ens.energies = np.zeros(N)
    ens.force_computed = np.ones(N, dtype=bool)

    # CRITICAL: refresh the q-space arrays. ens.generate() precomputes
    # u_disps_qspace and leaves forces_qspace = 0; assigning ens.forces
    # afterwards does NOT update them, and QSpaceLanczos skips ens.init()
    # when u_disps_qspace is already present.
    ens.init()
    return ens


def lanczos_effective_freq(lanc):
    """Static effective frequency (Ry) of the perturbed mode.

    The Wigner L has eigenvalues -w^2. The static response of the initial
    perturbation is g = [M^-1]_00 with M the Lanczos tridiagonal matrix;
    the renormalized frequency is w_eff = sqrt(-1/g). This is the standard
    static-Hessian-from-Lanczos observable: it includes all decay channels
    with their proper spectral weight.
    """
    a = np.array(lanc.a_coeffs)
    b = np.array(lanc.b_coeffs)
    n = len(a)
    M = np.diag(a)
    if n > 1:
        M += np.diag(b[:n - 1], 1) + np.diag(b[:n - 1], -1)
    e1 = np.zeros(n)
    e1[0] = 1.0
    g = np.linalg.solve(M, e1)[0]
    assert g < 0, "Static response is not negative definite (g = {})".format(g)
    return np.sqrt(-1.0 / g)
