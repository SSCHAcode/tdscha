"""
Q-Space Lanczos Interpolation Module
====================================

Run the q-space TDSCHA Lanczos on a FINE uniform q-mesh that is not
commensurate with the supercell of the stochastic ensemble, while the
ensemble stays on the COARSE (commensurate) mesh.

The method (see Interpolation_plan.md for the full derivation):

1. The anharmonic operator of the q-space Lanczos is a stochastic estimator
   built from per-configuration Bloch fields x(q), y(q); no high-order force
   constant tensor is ever formed. The Bloch sums over the supercell lattice
   can be evaluated at ANY q (a non-uniform DFT of the configuration): the
   ensemble averages of the off-grid products are trigonometric
   interpolations of the anharmonic correlation functions.
2. The SSCHA dynamical matrix is Fourier-interpolated (2nd order tensor,
   centered, with the acoustic sum rule) to obtain w(q), e(q) on the fine
   mesh, feeding the harmonic part of L and the chi/f_Y/f_psi factors.
3. The mode-space vertices scale as d3 ~ N^-1/2 and d4 ~ N^-1 with the
   number of cells N: the coarse-ensemble averages must be rescaled by
   scale3 = sqrt(N_c/N_f) (D3 terms) and scale4 = N_c/N_f (D4 terms) to
   represent the fine-mesh (N_f-cell) Lanczos operator.
4. Acoustic sum rule: the per-configuration translation zero-modes are
   projected out of displacements and force residuals before the transform,
   so that every leg of the effective D3/D4 kernels vanishes on acoustic
   modes at q -> 0 (Interpolation_plan.md section 5.5). With the plain
   (full-period) window this only removes the exact per-configuration
   zero-mode components: it is a no-op at commensurate q-points and on
   the (masked) Gamma translations.

This module implements the PLAIN-WINDOW estimator (milestone M2 of the
plan): exact at commensurate q, tent-kernel interpolation between them.
The designed multitaper windows ("stochastic centering", plan section 5)
are milestone M3 and will plug into the same field-construction step.

Limitations (phase 1):
- LO-TO splitting / effective charges are not applied to the interpolated
  dynamical matrix (lo_to_split must be None).
- prepare_ir / prepare_raman inherit the coarse sqrt(N_c) Gamma-amplitude
  prefactor; only the overall intensity scale is affected, not the
  spectral shape.
"""

from __future__ import print_function
from __future__ import division

import itertools
import warnings

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Methods
import cellconstructor.symmetries
import cellconstructor.ForceTensor
import cellconstructor.Units

from cellconstructor.Settings import ParallelPrint as print

import tdscha.QSpaceLanczos as QL


__EPSILON__ = 1e-12


# =========================================================================
# Fine mesh utilities
# =========================================================================

def generate_fine_mesh(uc_structure, mesh):
    """Generate a Gamma-centered uniform q-mesh (Gamma first).

    Parameters
    ----------
    uc_structure : CC.Structure.Structure
        The unit cell structure.
    mesh : tuple(3) of int
        The mesh dimensions (m1, m2, m3).

    Returns
    -------
    q_points : ndarray(N_f, 3)
        Cartesian q-points in the same units as dyn.q_tot (2pi/A free).
        The first point is Gamma; ordering is lexicographic in the integer
        mesh indices.
    idx : ndarray(N_f, 3), int
        The integer mesh indices n of each point (q_frac = n / mesh, folded
        to (-1/2, 1/2]).
    """
    mesh = np.asarray(mesh, dtype=int)
    assert np.all(mesh > 0), "Invalid mesh {}".format(mesh)

    bg = uc_structure.get_reciprocal_vectors() / (2 * np.pi)  # rows b_i, b_i . a_j = delta_ij

    idx = np.array(list(itertools.product(range(mesh[0]),
                                          range(mesh[1]),
                                          range(mesh[2]))), dtype=int)
    frac = idx / mesh[None, :]
    # Fold into (-1/2, 1/2] for aesthetics (does not affect any Bloch phase:
    # shifts are reciprocal lattice vectors)
    frac = frac - np.floor(frac + 0.5)

    q_points = frac @ bg
    return q_points, idx


def build_q_index_lookup(q_points, uc_structure, mesh, tol=1e-6):
    """Build an O(1) lookup {mesh index tuple -> position in q_points}.

    Works for any list of q-points lying on the uniform mesh (modulo G).
    """
    mesh = np.asarray(mesh, dtype=int)
    lookup = {}
    for iq, q in enumerate(q_points):
        key = _mesh_key(q, uc_structure, mesh, tol)
        lookup[key] = iq
    return lookup


def _mesh_key(q, uc_structure, mesh, tol=1e-6):
    """Integer mesh-index key of a q-point (modulo reciprocal lattice)."""
    frac = uc_structure.unit_cell @ np.asarray(q)  # frac_i = q . a_i
    n = frac * mesh
    n_round = np.round(n)
    if np.max(np.abs(n - n_round)) > tol * np.max(mesh):
        raise ValueError(
            "q-point {} is not on the {} mesh (frac*mesh = {})".format(q, mesh, n))
    return tuple((n_round.astype(int)) % mesh)


# =========================================================================
# Dynamical matrix interpolation
# =========================================================================

def interpolate_dyn_fine(dyn, q_points, use_asr=True, reuse_commensurate=True,
                         verbose=False):
    """Fourier-interpolate the dynamical matrix on a list of q-points.

    Uses ForceTensor.Tensor2 with real-space centering and (optionally) the
    iterative acoustic sum rule, exactly like the standard force-constant
    interpolation. Frequencies/polarizations are computed with the same
    mass convention as CC.Phonons.DyagDinQ, and the time-reversal gauge
    e(-q) = conj(e(q)) is enforced between +-q partners on the mesh.

    Parameters
    ----------
    dyn : CC.Phonons.Phonons
        The (coarse) SSCHA dynamical matrix.
    q_points : ndarray(N_f, 3)
        Target q-points (same units as dyn.q_tot).
    use_asr : bool
        Apply Tensor2.Apply_ASR() after centering.
    reuse_commensurate : bool
        For q-points matching dyn.q_tot, use dyn.dynmats directly instead of
        the centered/ASR-projected interpolation (guarantees exact
        reproduction of the coarse calculation on-grid).

    Returns
    -------
    w_q : ndarray(n_bands, N_f)
        Frequencies in Ry.
    pols_q : ndarray(3*nat, n_bands, N_f), complex128
        Polarization vectors.
    """
    uc = dyn.structure
    supercell = dyn.GetSupercell()
    super_structure = uc.generate_supercell(supercell)

    if dyn.effective_charges is not None:
        warnings.warn("Effective charges present: the interpolated dynamical "
                      "matrix neglects the nonanalytic LO-TO term (phase 1).")

    t2 = CC.ForceTensor.Tensor2(uc, super_structure, supercell)
    t2.SetupFromPhonons(dyn)
    t2.Center()
    if use_asr:
        t2.Apply_ASR()

    n_q = len(q_points)
    nat = uc.N_atoms
    nb = 3 * nat

    m3 = np.repeat(uc.get_masses_array(), 3)
    inv_sqrt_mm = 1.0 / np.sqrt(np.outer(m3, m3))

    bg = uc.get_reciprocal_vectors() / (2 * np.pi)

    # Match commensurate points
    commensurate_of = np.full(n_q, -1, dtype=int)
    if reuse_commensurate:
        for iq, q in enumerate(q_points):
            for jq, qc in enumerate(dyn.q_tot):
                if CC.Methods.get_min_dist_into_cell(bg, np.asarray(q), np.asarray(qc)) < 1e-6:
                    commensurate_of[iq] = jq
                    break

    # Identify TRI partners on the list: iq -> index of -q (or -1)
    minus_of = np.full(n_q, -1, dtype=int)
    for iq, q in enumerate(q_points):
        for jq, q2 in enumerate(q_points):
            if CC.Methods.get_min_dist_into_cell(bg, -np.asarray(q), np.asarray(q2)) < 1e-6:
                minus_of[iq] = jq
                break

    w_q = np.zeros((nb, n_q), dtype=np.float64)
    pols_q = np.zeros((nb, nb, n_q), dtype=np.complex128)
    done = np.zeros(n_q, dtype=bool)

    for iq in range(n_q):
        if done[iq]:
            continue
        q = np.asarray(q_points[iq], dtype=np.float64)

        if commensurate_of[iq] >= 0:
            fc = np.array(dyn.dynmats[commensurate_of[iq]], dtype=np.complex128)
        else:
            # NOTE the minus sign: Tensor2.Interpolate uses the phase
            # e^{-2 pi i q.r} while the CC dynmats convention corresponds to
            # the opposite sign; Interpolate(-q) == dyn.dynmats[q] at
            # commensurate q to machine precision (regression-tested).
            fc = t2.Interpolate(-q, asr=False, lo_to_splitting=False)

        D = fc * inv_sqrt_mm
        D = 0.5 * (D + np.conj(D.T))

        # At time-reversal-invariant points (q = -q + G) the matrix is real
        if minus_of[iq] == iq:
            D = np.real(D)

        eigvals, eigvects = np.linalg.eigh(D)
        w_q[:, iq] = np.sign(eigvals) * np.sqrt(np.abs(eigvals))
        pols_q[:, :, iq] = eigvects
        done[iq] = True

        # Enforce the time-reversal gauge on the -q partner
        jq = minus_of[iq]
        if jq >= 0 and jq != iq and not done[jq]:
            w_q[:, jq] = w_q[:, iq]
            pols_q[:, :, jq] = np.conj(pols_q[:, :, iq])
            done[jq] = True

    if verbose:
        print("Interpolated dynamical matrix on {} q-points "
              "({} commensurate reused)".format(n_q, np.sum(commensurate_of >= 0)))

    return w_q, pols_q


# =========================================================================
# The interpolated Lanczos
# =========================================================================

class QSpaceLanczosInterp(QL.QSpaceLanczos):
    """Q-space Lanczos on a fine q-mesh, interpolated from a coarse ensemble.

    Usage
    -----
    >>> lanczos = QSpaceLanczosInterp(ensemble, fine_mesh=(4, 4, 4))
    >>> lanczos.init(use_symmetries=True)
    >>> lanczos.prepare_mode_q(iq, band)   # iq indexes the FINE mesh
    >>> lanczos.run_FT(100)

    The two-phonon (a'/b') sector lives on the fine mesh: the perturbation
    at q_pert can decay into pairs of interpolated phonons. The perturbation
    q-point must be a point of the fine mesh.

    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble
        The SSCHA ensemble (on the coarse supercell).
    fine_mesh : tuple(3) of int
        The fine uniform Gamma-centered q-mesh. Does NOT need to be a
        multiple of the coarse mesh, but the coarse points are reproduced
        exactly only when it is a superset of the coarse mesh.
    use_asr_dyn : bool
        Apply the acoustic sum rule to the interpolated dynamical matrix.
    reuse_commensurate : bool
        Use the coarse dyn matrices directly at commensurate fine points.
    w_min_guard : float
        Frequencies (in Ry) below this threshold at q != Gamma are masked
        with a warning (guards against numerically-zero interpolated
        frequencies; genuine acoustic modes near Gamma are far above it).
    allow_unstable : bool
        If False (default), raise if the interpolated dyn has imaginary
        frequencies away from Gamma.
    """

    def __init__(self, ensemble, fine_mesh=None, use_asr_dyn=True,
                 reuse_commensurate=True, w_min_guard=1e-8,
                 allow_unstable=False, prefilter=True,
                 lo_to_split=None, **kwargs):

        if lo_to_split is not None:
            raise NotImplementedError(
                "LO-TO splitting is not supported by the interpolated "
                "q-space Lanczos (phase 1).")

        super().__init__(ensemble, lo_to_split=None, **kwargs)

        interp_attrs = ['fine_mesh', '_fine_idx', '_q_lookup', '_fpsi_fine']
        self.__total_attributes__.extend(interp_attrs)

        self.fine_mesh = None
        self._fine_idx = None
        self._q_lookup = None
        self._fpsi_fine = None

        # Bare initialization (used by the distributed loader)
        if ensemble is None:
            return

        if fine_mesh is None:
            raise ValueError("QSpaceLanczosInterp requires fine_mesh=(m1, m2, m3)")

        self.fine_mesh = np.asarray(fine_mesh, dtype=int)
        coarse_mesh = np.asarray(self.dyn.GetSupercell(), dtype=int)
        n_c = int(np.prod(coarse_mesh))
        n_f = int(np.prod(self.fine_mesh))

        # Stash the coarse-grid quantities computed by the parent before we
        # overwrite them: they are needed to build the pre-filtered fields.
        coarse_data = {
            'q_points': np.array(self.q_points),
            'w_q': np.array(self.w_q),
            'pols_q': np.array(self.pols_q),
            'valid_modes_q': np.array(self.valid_modes_q),
            'X_q': self.X_q,
        }

        # == 1. Fine mesh and lookup ==
        q_fine, idx_fine = generate_fine_mesh(self.uci_structure, self.fine_mesh)
        self.q_points = q_fine
        self.n_q = n_f
        self._fine_idx = idx_fine
        self._q_lookup = build_q_index_lookup(q_fine, self.uci_structure,
                                              self.fine_mesh)

        # == 2. Interpolated dynamical matrix (harmonic sector) ==
        self.w_q, self.pols_q = interpolate_dyn_fine(
            self.dyn, q_fine, use_asr=use_asr_dyn,
            reuse_commensurate=reuse_commensurate)

        # == 3. Mode validity masks on the fine mesh ==
        masses_uc = self.dyn.structure.get_masses_array()
        self.valid_modes_q = np.ones((self.n_bands, self.n_q), dtype=bool)
        trans_mask = CC.Methods.get_translations(
            np.real(self.pols_q[:, :, 0]), masses_uc)
        self.valid_modes_q[:, 0] = ~trans_mask

        unstable = (self.w_q < -w_min_guard)
        unstable[:, 0] = unstable[:, 0] & self.valid_modes_q[:, 0]
        if np.any(unstable):
            bad = np.unique(np.where(unstable)[1])
            msg = ("Interpolated dynamical matrix has imaginary frequencies "
                   "at fine q-points {} (the SSCHA dyn must be positive "
                   "definite for the TDSCHA response)".format(bad))
            if allow_unstable:
                warnings.warn(msg + " -- masking those modes.")
                self.valid_modes_q &= ~unstable
            else:
                raise ValueError(msg)

        small = (np.abs(self.w_q) < w_min_guard) & self.valid_modes_q
        small[:, 0] = False  # Gamma translations already masked
        if np.any(small):
            warnings.warn("Masking {} interpolated modes with |w| < {} Ry "
                          "away from Gamma.".format(np.sum(small), w_min_guard))
            self.valid_modes_q &= ~small

        if ensemble.ignore_small_w:
            small_freq = np.abs(self.w_q) < CC.Phonons.__EPSILON_W__
            self.valid_modes_q &= ~small_freq

        # == 4. Bloch fields on the fine mesh (NUDFT of the ensemble) ==
        self._bloch_transform_ensemble_fine(
            coarse_data if prefilter else None)

        # Fine-side f_psi table (folded into alpha1 in prefiltered mode)
        self.qspace_prefiltered = bool(prefilter)
        if prefilter:
            self._fpsi_fine = self._get_fpsi_table()

        # == 5. Vertex renormalization N_c -> N_f ==
        self.qspace_scale3 = np.sqrt(n_c / n_f)
        self.qspace_scale4 = n_c / n_f

        # Reset the pair-map state (was initialized on the coarse mesh)
        self.iq_pert = None
        self.q_pair_map = None
        self.unique_pairs = None
        self._psi_size = None

    # ---------------------------------------------------------------
    def _get_fpsi_table(self):
        """Fine-mesh f_psi = (1+2n)/(2w) with masked modes set to zero."""
        fpsi = np.zeros((self.n_bands, self.n_q), dtype=np.float64)
        for iq in range(self.n_q):
            n_bose, valid = self._safe_bose_and_mask(iq)
            w = self.w_q[:, iq]
            fpsi[valid, iq] = (1.0 + 2.0 * n_bose[valid]) / (2.0 * w[valid])
        return fpsi

    def _get_fy_table_coarse(self, coarse_data):
        """Coarse-grid f_Y = 2w/(1+2n) with masked modes set to zero."""
        w_c = coarse_data['w_q']
        valid_c = coarse_data['valid_modes_q']
        n_qc = w_c.shape[1]
        fy = np.zeros_like(w_c)
        for iq in range(n_qc):
            valid = valid_c[:, iq]
            w = w_c[valid, iq]
            if self.T > __EPSILON__:
                n_bose = 1.0 / (np.exp(w * QL.__RyToK__ / self.T) - 1.0)
            else:
                n_bose = np.zeros_like(w)
            fy[valid, iq] = 2.0 * w / (1.0 + 2.0 * n_bose)
        return fy

    # ---------------------------------------------------------------
    def _bloch_transform_ensemble_fine(self, coarse_data=None):
        """Bloch transform the ensemble on the fine mesh (plain window NUDFT).

        Replicates the conventions of sscha's vector_r2q (phase
        exp(-2 pi i q . R) on the cell origin R of each supercell atom,
        normalization 1/sqrt(N_c) with N_c the number of COARSE cells)
        evaluated at the fine q-points, with:

        - the rho-weighted symmetrized average force subtracted in real
          space (kills its window leakage at incommensurate q, reduces to
          the parent Gamma subtraction at commensurate q);
        - the per-configuration translation zero-modes projected out
          (acoustic sum rule, plan section 5.5): displacements lose their
          mass-weighted center of mass, force residuals lose a rigid
          mass-proportional redistribution of the net force. Both
          subtracted patterns are exactly the translation modes in the
          mass-scaled representation: optical components are untouched.

        If coarse_data is given (pre-filter mode, plan section 5.6), the
        displacement fields are filtered with f_Y = 2w/(1+2n) ON THE COARSE
        GRID before the fine transform. By the Gaussian integration-by-parts
        identity, (f_Y x) legs carry no phonon-propagator dressing: the
        interpolated D3/D4 correlations then decay with the range of the
        anharmonic force constants themselves instead of the (much longer)
        propagator range, drastically reducing the interpolation error.
        The Julia kernel is informed via qspace_prefiltered and the exact
        fine-side f_psi factors are folded into alpha1.
        """
        ens = self.ensemble
        uc = self.dyn.structure
        sc = self.super_structure
        nat_uc = uc.N_atoms
        nat_sc = sc.N_atoms
        n_c = nat_sc // nat_uc
        N = self.N

        itau = sc.get_itau(uc) - 1                     # (nat_sc,)
        r_lat = sc.coords - uc.coords[itau]            # Angstrom, cell origins

        # --- Real-space data (match parent unit handling) ---
        u_conv = 1.0
        f_conv = 1.0
        if ens.units == "default":
            u_conv = CC.Units.A_TO_BOHR
            f_conv = 1.0 / CC.Units.A_TO_BOHR
        elif ens.units == "hartree":
            f_conv = 2.0

        u_sc = np.array(ens.u_disps, dtype=np.float64).reshape(N, nat_sc, 3)

        # The real-space sscha_forces may be empty when the ensemble runs in
        # Fourier-gradient mode (only *_qspace arrays are filled). Rebuild the
        # real-space SSCHA forces from the q-space array with the inverse
        # Bloch transform (exact on the coarse grid):
        #   f(R, a) = 1/sqrt(N_c) sum_q e^{+2 pi i q . R} f_q(a)
        q_coarse = np.array(self.dyn.q_tot)                       # (n_qc, 3), 2pi/A
        phases_c = np.exp(-2j * np.pi * (q_coarse @ r_lat.T))     # (n_qc, nat_sc)
        fsq = np.array(ens.sscha_forces_qspace)                   # (N, 3*nat_uc, n_qc)
        f_sscha = np.zeros((N, nat_sc, 3), dtype=np.float64)
        for a in range(nat_uc):
            sel = np.where(itau == a)[0]
            # inverse transform: conj phases, same 1/sqrt(N_c) normalization
            f_sscha[:, sel, :] = np.real(np.einsum(
                'qk,iaq->ika', np.conj(phases_c[:, sel]),
                fsq[:, 3 * a:3 * a + 3, :], optimize=True)) / np.sqrt(n_c)

        delta_f = np.array(ens.forces, dtype=np.float64).reshape(N, nat_sc, 3) - f_sscha

        # --- Average force subtraction (real space, tiled over cells) ---
        f_mean_uc = ens.get_average_forces(get_error=False)   # (nat_uc, 3)
        qe_sym = CC.symmetries.QE_Symmetry(uc)
        qe_sym.SetupQPoint()
        qe_sym.SymmetrizeVector(f_mean_uc)
        delta_f -= f_mean_uc[itau, :][None, :, :]

        # --- ASR zero-mode projection (per configuration) ---
        m_sc = sc.get_masses_array()                   # (nat_sc,)
        M_tot = np.sum(m_sc)
        # displacements: remove the mass-weighted center of mass
        com = np.einsum('k,ika->ia', m_sc, u_sc) / M_tot        # (N, 3)
        u_sc = u_sc - com[:, None, :]
        # forces: remove the net force, redistributed proportionally to the
        # masses (the translation mode in the mass-scaled metric)
        f_net = np.sum(delta_f, axis=1)                         # (N, 3)
        delta_f = delta_f - (m_sc / M_tot)[None, :, None] * f_net[:, None, :]

        # --- NUDFT at the fine q-points ---
        # phase(q, k) = exp(-2 pi i q . R_k); normalization 1/sqrt(N_c)
        phases = np.exp(-2j * np.pi * (self.q_points @ r_lat.T))  # (n_q, nat_sc)

        def nudft(field_sc):
            """(N, nat_sc, 3) real/complex -> (n_q, N, 3*nat_uc)."""
            out = np.zeros((self.n_q, N, 3 * nat_uc), dtype=np.complex128)
            for a in range(nat_uc):
                sel = np.where(itau == a)[0]
                out[:, :, 3 * a:3 * a + 3] = np.einsum(
                    'qk,ika->qia', phases[:, sel], field_sc[:, sel, :],
                    optimize=True)
            return out / np.sqrt(n_c)

        m_uc = uc.get_masses_array()
        sqrt_m3 = np.sqrt(np.repeat(m_uc, 3))

        # --- Displacement channel ---
        if coarse_data is None:
            # Raw fields: NUDFT then unit conversion + mass scaling
            u_tilde = nudft(u_sc)
            u_scale = u_conv * sqrt_m3
        else:
            # Pre-filtered fields: apply f_Y on the coarse grid in mode
            # space (this is exact there), rebuild the filtered real-space
            # configuration, and transform THAT. The coarse X_q already
            # includes units and mass scaling.
            fy_c = self._get_fy_table_coarse(coarse_data)   # (nb, n_qc)
            X_c = coarse_data['X_q']                        # (n_qc, N, nb)
            pols_c = coarse_data['pols_q']
            q_c = coarse_data['q_points']
            n_qc = X_c.shape[0]

            phases_cc = np.exp(-2j * np.pi * (q_c @ r_lat.T))  # (n_qc, nat_sc)
            uf_sc = np.zeros((N, nat_sc, 3), dtype=np.float64)
            for iq in range(n_qc):
                # back to Cartesian (mass-scaled) unit-cell components:
                # x = u_mass . conj(P)  =>  u_mass = x . P^T (P unitary)
                u_mass_q = (X_c[iq] * fy_c[:, iq][None, :]) @ pols_c[:, :, iq].T
                # inverse Bloch transform (conjugate phases):
                # uf[i, k, alpha] += conj(phase[k]) * u_mass_q[i, 3*itau[k]+alpha]
                idx = 3 * itau[:, None] + np.arange(3)[None, :]   # (nat_sc, 3)
                uf_sc += np.real(np.conj(phases_cc[iq])[None, :, None]
                                 * u_mass_q[:, idx])
            uf_sc /= np.sqrt(n_qc)

            u_tilde = nudft(uf_sc)
            u_scale = np.ones_like(sqrt_m3)  # already scaled

        # --- Force channel (always raw: y legs carry no propagator) ---
        f_tilde = nudft(delta_f)

        # --- Mode projection ---
        self.X_q = np.zeros((self.n_q, N, self.n_bands), dtype=np.complex128)
        self.Y_q = np.zeros((self.n_q, N, self.n_bands), dtype=np.complex128)
        for iq in range(self.n_q):
            u_mass = u_tilde[iq] * u_scale[None, :]
            f_mass = f_tilde[iq] * (f_conv / sqrt_m3[None, :])
            pol_iq = self.pols_q[:, :, iq]
            self.X_q[iq] = u_mass @ np.conj(pol_iq)
            self.Y_q[iq] = f_mass @ np.conj(pol_iq)

    # ---------------------------------------------------------------
    def _call_julia_qspace(self, R1, alpha1_flat):
        """Fold the exact fine-side f_psi factors into alpha1 (prefiltered).

        In prefiltered mode the kernel contracts alpha1 with the FILTERED
        fields (f_Y x), so alpha1 must carry the compensating f_psi factors:
        alpha1 : x* x* = (alpha1 o f_psi x f_psi) : (f_Y x)* (f_Y x)*.
        The buffer_u-based sums then use f_psi = 1 in the kernel (the factor
        is already inside the modified alpha1).
        """
        if self.qspace_prefiltered:
            blocks = self._unflatten_blocks(np.array(alpha1_flat))
            folded = []
            for pair_idx, (iq1, iq2) in enumerate(self.unique_pairs):
                fold = np.outer(self._fpsi_fine[:, iq1], self._fpsi_fine[:, iq2])
                folded.append(blocks[pair_idx] * fold)
            alpha1_flat = self._flatten_blocks(folded)
        return super()._call_julia_qspace(R1, alpha1_flat)

    # ---------------------------------------------------------------
    def build_q_pair_map(self, iq_pert):
        """O(N_f) pair map via integer mesh-index arithmetic."""
        if self._fine_idx is None:
            # Bare/distributed instance: fall back to the parent search
            return super().build_q_pair_map(iq_pert)

        mesh = self.fine_mesh
        self.iq_pert = iq_pert
        n_pert = self._fine_idx[iq_pert]

        self.q_pair_map = np.zeros(self.n_q, dtype=np.int32)
        for iq1 in range(self.n_q):
            n2 = tuple((n_pert - self._fine_idx[iq1]) % mesh)
            self.q_pair_map[iq1] = self._q_lookup[n2]

        self.unique_pairs = []
        for iq1 in range(self.n_q):
            iq2 = int(self.q_pair_map[iq1])
            if iq1 <= iq2:
                self.unique_pairs.append((iq1, iq2))

        self._compute_block_layout()

    # ---------------------------------------------------------------
    def find_fine_q(self, q):
        """Index of a (Cartesian) q-vector in the fine mesh, O(1)."""
        return self._q_lookup[_mesh_key(q, self.uci_structure, self.fine_mesh)]
