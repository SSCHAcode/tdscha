"""
Q-Space Lanczos Module
======================

This module implements the Lanczos algorithm in q-space (Bloch basis) to exploit
momentum conservation and block structure from Bloch's theorem. This gives a
speedup of ~N_cell over the real-space implementation.

Key differences from the real-space DynamicalLanczos.Lanczos:
- Psi vector is complex128 (Hermitian Lanczos with sesquilinear inner product)
- Two-phonon sector uses (q1, q2) pairs constrained by q1+q2 = q_pert + G
- Symmetries are point-group only (translations handled by Fourier transform)
- Requires Julia extension (tdscha_qspace.jl)

References:
    Implementation plan: implementation_plan.md
    Parent class: DynamicalLanczos.py
"""

from __future__ import print_function
from __future__ import division

import sys, os
import time
import warnings
import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries
import cellconstructor.Methods

import tdscha.DynamicalLanczos as DL
import cellconstructor.Settings as Parallel
from cellconstructor.Settings import ParallelPrint as print

# Try to import the julia module
__JULIA_EXT__ = False
try:
    import julia, julia.Main
    julia.Main.include(os.path.join(os.path.dirname(__file__), 
        "tdscha_qspace.jl"))
    __JULIA_EXT__ = True
except:
    try:
        import julia
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        import julia.Main
        try:
            julia.Main.include(os.path.join(os.path.dirname(__file__),
                "tdscha_qspace.jl"))
            __JULIA_EXT__ = True
        except:
            # Install the required modules
            julia.Main.eval("""
using Pkg
Pkg.add("SparseArrays")
Pkg.add("InteractiveUtils")
""")
            try:
                julia.Main.include(os.path.join(os.path.dirname(__file__),
                    "tdscha_qspace.jl"))
                __JULIA_EXT__ = True
            except Exception as e:
                warnings.warn("Julia extension not available.\nError: {}".format(e))
    except Exception as e:
        warnings.warn("Julia extension not available.\nError: {}".format(e))
    pass

try:
    import spglib
    __SPGLIB__ = True
except ImportError:
    __SPGLIB__ = False


# Constants
__EPSILON__ = 1e-12
__RyToK__ = 157887.32400374097
TYPE_DP = np.double


def find_q_index(q_target, q_points, bg, tol=1e-6):
    """Find the index of q_target in q_points up to a reciprocal lattice vector.

    Parameters
    ----------
    q_target : ndarray(3,)
        The q-point to find (Cartesian coordinates).
    q_points : ndarray(n_q, 3)
        Array of q-points.
    bg : ndarray(3, 3)
        Reciprocal lattice vectors / (2*pi), rows are vectors.
    tol : float
        Tolerance for matching.

    Returns
    -------
    int
        Index of the matching q-point.
    """
    for iq, q in enumerate(q_points):
        dist = CC.Methods.get_min_dist_into_cell(bg, q_target, q)
        if dist < tol:
            return iq
    raise ValueError("Could not find q-point {} in the q-point list".format(q_target))


class QSpaceLanczos(DL.Lanczos):
    """Q-space Lanczos for spectral calculations exploiting Bloch's theorem.

    This class works in the q-space mode basis to exploit momentum conservation,
    reducing the psi vector size by ~N_cell and the anharmonic computation by ~N_cell.

    Only Wigner formalism is supported. Requires Julia extension.
    """

    def __init__(self, ensemble, **kwargs):
        """Initialize the Q-Space Lanczos.

        Parameters
        ----------
        ensemble : sscha.Ensemble.Ensemble
            The SSCHA ensemble.
        **kwargs
            Additional keyword arguments passed to the parent Lanczos class.
        """
        # Force Wigner and Julia mode
        kwargs['mode'] = DL.MODE_FAST_JULIA
        kwargs['use_wigner'] = True

        self.ensemble = ensemble
        super().__init__(ensemble, unwrap_symmetries=False, **kwargs)

        if not __JULIA_EXT__:
            raise ImportError(
                "QSpaceLanczos requires Julia. Install with: pip install julia"
            )
        self.use_wigner = True

        # -- Add the q-space attributes --
        qspace_attrs = [
            'q_points', 'n_q', 'n_bands', 'w_q', 'pols_q',
            'acoustic_eps', 'X_q', 'Y_q',
            'iq_pert', 'q_pair_map', 'unique_pairs',
            '_psi_size', '_block_offsets_a', '_block_offsets_b', '_block_sizes',
            '_qspace_sym_data', '_qspace_sym_q_map', 'n_syms_qspace',
        ]
        self.__total_attributes__.extend(qspace_attrs)

        # == 1. Get q-space eigenmodes ==
        ws_sc, pols_sc, w_q, pols_q = self.dyn.DiagonalizeSupercell(return_qmodes=True)

        self.q_points = np.array(self.dyn.q_tot)  # (n_q, 3)
        self.n_q = len(self.q_points)
        self.n_bands = 3 * self.uci_structure.N_atoms  # uniform band count
        self.w_q = w_q        # (n_bands, n_q) from DiagonalizeSupercell
        self.pols_q = pols_q  # (3*n_at, n_bands, n_q) complex eigenvectors

        # Small frequency threshold for acoustic mode masking
        self.acoustic_eps = 1e-6

        # == 2. Bloch transform ensemble data ==
        self._bloch_transform_ensemble()

        # Q-space specific state (set by build_q_pair_map)
        self.iq_pert = None
        self.q_pair_map = None
        self.unique_pairs = None
        self._psi_size = None
        self._block_offsets_a = None
        self._block_offsets_b = None
        self._block_sizes = None

        # Symmetry data for q-space
        self._qspace_sym_data = None
        self._qspace_sym_q_map = None
        self.n_syms_qspace = 0

    def _bloch_transform_ensemble(self):
        """Bloch transform the ensemble displacements and forces into q-space mode basis.

        Computes X_q and Y_q from the ensemble u_disps_qspace and forces_qspace.
        Uses sscha.Ensemble implementation for the Fourier transform via Julia.

        The forces are the anharmonic residual: f - f_SSCHA - <f - f_SSCHA>,
        matching the preprocessing done in DynamicalLanczos.__init__.
        """
        # Ensure the ensemble has computed q-space quantities
        # Check if fourier_gradient is active or force it
        if self.ensemble.u_disps_qspace is None:
             # Force fourier gradient initialization in the ensemble
             if not self.ensemble.fourier_gradient:
                 print("Ensemble checking: computing Fourier transform of displacements and forces...")
                 self.ensemble.fourier_gradient = True
             self.ensemble.init()

        # Unit conversion factors
        # Target: Bohr (u) and Ry/Bohr (f)
        u_conv = 1.0
        f_conv = 1.0
        if self.ensemble.units == "default":
            u_conv = CC.Units.A_TO_BOHR
            f_conv = 1.0 / CC.Units.A_TO_BOHR
        elif self.ensemble.units == "hartree":
            f_conv = 2.0 # Ha -> Ry

        # Mass scaling factors (sqrt(m) for u, 1/sqrt(m) for f)
        # We use self.dyn.structure corresponding to unit cell
        m_uc = self.dyn.structure.get_masses_array()
        sqrt_m = np.sqrt(m_uc)
        sqrt_m_3 = np.repeat(sqrt_m, 3) # (3*nat_uc,)

        # Compute the average anharmonic force (matching parent DynamicalLanczos)
        # get_average_forces returns rho-weighted <f - f_SSCHA> in unit cell, Ry/Angstrom
        f_mean_uc = self.ensemble.get_average_forces(get_error=False)  # (nat_uc, 3)
        # Symmetrize the average force
        qe_sym = CC.symmetries.QE_Symmetry(self.dyn.structure)
        qe_sym.SetupQPoint()
        qe_sym.SymmetrizeVector(f_mean_uc)
        f_mean_flat = f_mean_uc.ravel()  # (3*nat_uc,) in Ry/Angstrom

        # Fourier transform of the constant (tiled) average force:
        # At Gamma: f_mean_q[Gamma] = sqrt(n_cell) * f_mean_uc
        # At q != Gamma: f_mean_q[q] = 0
        n_cell = np.prod(self.dyn.GetSupercell())
        f_mean_q_gamma = np.sqrt(n_cell) * f_mean_flat  # (3*nat_uc,) complex

        # Allocate q-space arrays
        self.X_q = np.zeros((self.n_q, self.N, self.n_bands), dtype=np.complex128)
        self.Y_q = np.zeros((self.n_q, self.N, self.n_bands), dtype=np.complex128)

        # Projection loop
        for iq in range(self.n_q):
            # Retrieve ensemble q-space data (N, 3*nat_uc)
            # Apply conversions and mass scaling
            u_tilde_q = self.ensemble.u_disps_qspace[:, :, iq] * (u_conv * sqrt_m_3[None, :])

            # Anharmonic force residual: f - f_SSCHA (both in Ry/Angstrom in q-space)
            delta_f_q = (self.ensemble.forces_qspace[:, :, iq]
                         - self.ensemble.sscha_forces_qspace[:, :, iq])

            # Subtract the average force (only non-zero at Gamma)
            if iq == 0:
                delta_f_q = delta_f_q - f_mean_q_gamma[None, :]

            # Convert to Ry/Bohr and mass-scale
            f_tilde_q = delta_f_q * (f_conv / sqrt_m_3[None, :])

            # Project onto eigenvector basis: X_q[iq, config, nu] = sum_a conj(pols_q[a, nu, iq]) * u_tilde_q[config, a]
            # pols_q shape: (3*nat_uc, n_bands, n_q)
            pol_iq = self.pols_q[:, :, iq]  # (3*nat_uc, n_bands)

            self.X_q[iq, :, :] = u_tilde_q @ np.conj(pol_iq)
            self.Y_q[iq, :, :] = f_tilde_q @ np.conj(pol_iq)

    def build_q_pair_map(self, iq_pert):
        """Find all (iq1, iq2) pairs satisfying q1 + q2 = q_pert + G.

        Parameters
        ----------
        iq_pert : int
            Index of the perturbation q-point.
        """
        bg = self.uci_structure.get_reciprocal_vectors() / (2 * np.pi)
        q_pert = self.q_points[iq_pert]

        self.iq_pert = iq_pert
        self.q_pair_map = np.zeros(self.n_q, dtype=np.int32)

        for iq1 in range(self.n_q):
            q_target = q_pert - self.q_points[iq1]
            found = False
            for iq2 in range(self.n_q):
                if CC.Methods.get_min_dist_into_cell(bg, q_target, self.q_points[iq2]) < 1e-6:
                    self.q_pair_map[iq1] = iq2
                    found = True
                    break
            if not found:
                raise ValueError(
                    "Could not find partner for q1={} with q_pert={}".format(
                        self.q_points[iq1], q_pert))

        # Unique pairs: iq1 <= iq2 (avoids double-counting)
        self.unique_pairs = []
        for iq1 in range(self.n_q):
            iq2 = self.q_pair_map[iq1]
            if iq1 <= iq2:
                self.unique_pairs.append((iq1, iq2))

        # Pre-compute block layout
        self._compute_block_layout()

    def _compute_block_layout(self):
        """Pre-compute the block offsets and sizes for the psi vector."""
        nb = self.n_bands
        n_pairs = len(self.unique_pairs)

        # Compute block sizes
        self._block_sizes = []
        for iq1, iq2 in self.unique_pairs:
            if iq1 < iq2:
                self._block_sizes.append(nb * nb)  # full block
            else:  # iq1 == iq2
                self._block_sizes.append(nb * (nb + 1) // 2)  # upper triangle

        # R sector: n_bands entries
        r_size = nb

        # a' sector offsets
        self._block_offsets_a = []
        offset = r_size
        for size in self._block_sizes:
            self._block_offsets_a.append(offset)
            offset += size

        # b' sector offsets
        self._block_offsets_b = []
        for size in self._block_sizes:
            self._block_offsets_b.append(offset)
            offset += size

        self._psi_size = offset

    def get_psi_size(self):
        """Return the total size of the psi vector."""
        if self._psi_size is None:
            raise ValueError("Must call build_q_pair_map first")
        return self._psi_size

    def get_block_offset(self, pair_idx, sector='a'):
        """Get the offset into psi for a given pair index.

        Parameters
        ----------
        pair_idx : int
            Index into self.unique_pairs.
        sector : str
            'a' for a' sector, 'b' for b' sector.
        """
        if sector == 'a':
            return self._block_offsets_a[pair_idx]
        else:
            return self._block_offsets_b[pair_idx]

    def get_block_size(self, pair_idx):
        """Get the number of entries for this pair."""
        return self._block_sizes[pair_idx]

    def get_R1_q(self):
        """Extract R^(1) from psi (n_bands complex entries at q_pert)."""
        return self.psi[:self.n_bands].copy()

    def _unpack_upper_triangle(self, flat_data, n):
        """Unpack upper triangle storage into a full (n, n) matrix.

        Storage order: for i in range(n): M[i, i:] stored contiguously.
        Off-diagonal: M[j, i] = conj(M[i, j]) for Hermitian matrix.
        """
        mat = np.zeros((n, n), dtype=np.complex128)
        idx = 0
        for i in range(n):
            length = n - i
            mat[i, i:] = flat_data[idx:idx + length]
            idx += length
        # Fill lower triangle (Hermitian)
        for i in range(n):
            mat[i + 1:, i] = np.conj(mat[i, i + 1:])
        return mat

    def _pack_upper_triangle(self, mat, n):
        """Pack a full (n, n) matrix into upper triangle storage."""
        size = n * (n + 1) // 2
        flat = np.zeros(size, dtype=np.complex128)
        idx = 0
        for i in range(n):
            length = n - i
            flat[idx:idx + length] = mat[i, i:]
            idx += length
        return flat

    def get_block(self, pair_idx, sector='a'):
        """Reconstruct full (n_bands, n_bands) matrix from psi storage.

        Parameters
        ----------
        pair_idx : int
            Index into self.unique_pairs.
        sector : str
            'a' or 'b'.

        Returns
        -------
        ndarray(n_bands, n_bands), complex128
        """
        nb = self.n_bands
        iq1, iq2 = self.unique_pairs[pair_idx]
        offset = self.get_block_offset(pair_idx, sector)
        size = self.get_block_size(pair_idx)

        raw = self.psi[offset:offset + size]

        if iq1 < iq2:
            # Full block, row-major
            return raw.reshape(nb, nb).copy()
        else:
            # Upper triangle storage
            return self._unpack_upper_triangle(raw, nb)

    def get_a1_block(self, pair_idx):
        """Get the a'(1) block for pair_idx."""
        return self.get_block(pair_idx, 'a')

    def get_b1_block(self, pair_idx):
        """Get the b'(1) block for pair_idx."""
        return self.get_block(pair_idx, 'b')

    def set_block_in_psi(self, pair_idx, matrix, sector, target_psi):
        """Write a (n_bands, n_bands) block into the target psi vector.

        Parameters
        ----------
        pair_idx : int
        matrix : ndarray(n_bands, n_bands)
        sector : str ('a' or 'b')
        target_psi : ndarray — the psi vector to write into
        """
        nb = self.n_bands
        iq1, iq2 = self.unique_pairs[pair_idx]
        offset = self.get_block_offset(pair_idx, sector)

        if iq1 < iq2:
            # Full block
            target_psi[offset:offset + nb * nb] = matrix.ravel()
        else:
            # Upper triangle
            target_psi[offset:offset + self.get_block_size(pair_idx)] = \
                self._pack_upper_triangle(matrix, nb)

    # ====================================================================
    # Step 4: Mask for Hermitian inner product
    # ====================================================================
    def mask_dot_wigner(self, debug=False):
        """Build the mask for Hermitian inner product with upper-triangle storage.

        For full blocks (iq1 < iq2): factor 2 for the conjugate block (iq2, iq1).
        For diagonal blocks (iq1 == iq2): off-diagonal factor 2, diagonal factor 1.

        Returns
        -------
        ndarray(psi_size,), float64
        """
        mask = np.ones(self.get_psi_size(), dtype=np.float64)

        for pair_idx, (iq1, iq2) in enumerate(self.unique_pairs):
            offset_a = self.get_block_offset(pair_idx, 'a')
            offset_b = self.get_block_offset(pair_idx, 'b')
            size = self.get_block_size(pair_idx)

            if iq1 < iq2:
                # Full block: factor 2 for the conjugate block
                mask[offset_a:offset_a + size] = 2
                mask[offset_b:offset_b + size] = 2
            else:  # iq1 == iq2, upper triangle storage
                nb = self.n_bands
                idx_a = offset_a
                idx_b = offset_b
                for i in range(nb):
                    # Diagonal element: factor 1
                    mask[idx_a] = 1
                    mask[idx_b] = 1
                    idx_a += 1
                    idx_b += 1
                    # Off-diagonal elements: factor 2
                    for j in range(i + 1, nb):
                        mask[idx_a] = 2
                        mask[idx_b] = 2
                        idx_a += 1
                        idx_b += 1

        return mask

    # ====================================================================
    # Step 5: Harmonic operator
    # ====================================================================
    def apply_L1_FT(self, transpose=False):
        """Apply the harmonic part of L in q-space (Wigner formalism).

        L_harm is block-diagonal:
          R sector: -(w_q_pert[nu])^2 * R[nu]
          a' sector: -(w1 - w2)^2 * a'
          b' sector: -(w1 + w2)^2 * b'

        Returns
        -------
        ndarray(psi_size,), complex128
        """
        out = np.zeros(self.get_psi_size(), dtype=np.complex128)

        if self.ignore_harmonic:
            return out

        # R sector
        w_qp = self.w_q[:, self.iq_pert]  # (n_bands,)
        out[:self.n_bands] = -(w_qp ** 2) * self.psi[:self.n_bands]

        # a' and b' sectors
        for pair_idx, (iq1, iq2) in enumerate(self.unique_pairs):
            w1 = self.w_q[:, iq1]  # (n_bands,)
            w2 = self.w_q[:, iq2]  # (n_bands,)
            a_block = self.get_a1_block(pair_idx)
            b_block = self.get_b1_block(pair_idx)

            w_minus2 = np.subtract.outer(w1, w2) ** 2  # (n_bands, n_bands)
            w_plus2 = np.add.outer(w1, w2) ** 2

            self.set_block_in_psi(pair_idx, -w_minus2 * a_block, 'a', out)
            self.set_block_in_psi(pair_idx, -w_plus2 * b_block, 'b', out)

        return out

    # ====================================================================
    # Step 6: Anharmonic operator
    # ====================================================================
    def _safe_bose_and_mask(self, iq):
        """Compute Bose-Einstein occupation for bands at iq, masking acoustic modes.

        Returns
        -------
        n : ndarray(n_bands,)
            Bose-Einstein occupation; 0 for acoustic modes.
        valid : ndarray(n_bands,), bool
            True for non-acoustic bands.
        """
        w = self.w_q[:, iq]
        valid = w > self.acoustic_eps
        n = np.zeros_like(w)
        if self.T > __EPSILON__:
            n[valid] = 1.0 / (np.exp(w[valid] * __RyToK__ / self.T) - 1.0)
        return n, valid

    def get_chi_minus_q(self):
        """Get chi^- for each unique pair as a list of (n_bands, n_bands) matrices.

        chi^-_{nu1, nu2} = (w1 - w2)(n1 - n2) / (2 * w1 * w2)
        Entries involving acoustic modes (w < acoustic_eps) are set to 0.
        """
        chi_list = []
        for iq1, iq2 in self.unique_pairs:
            w1 = self.w_q[:, iq1]
            w2 = self.w_q[:, iq2]
            n1, v1 = self._safe_bose_and_mask(iq1)
            n2, v2 = self._safe_bose_and_mask(iq2)
            # Outer mask: both bands must be non-acoustic
            valid_mask = np.outer(v1, v2)

            w1_mat = np.tile(w1, (self.n_bands, 1)).T
            w2_mat = np.tile(w2, (self.n_bands, 1))
            n1_mat = np.tile(n1, (self.n_bands, 1)).T
            n2_mat = np.tile(n2, (self.n_bands, 1))

            # Safe division: use np.where to avoid 0/0
            denom = 2.0 * w1_mat * w2_mat
            chi = np.where(valid_mask,
                           (w1_mat - w2_mat) * (n1_mat - n2_mat) / np.where(valid_mask, denom, 1.0),
                           0.0)
            chi_list.append(chi)
        return chi_list

    def get_chi_plus_q(self):
        """Get chi^+ for each unique pair as a list of (n_bands, n_bands) matrices.

        chi^+_{nu1, nu2} = (w1 + w2)(1 + n1 + n2) / (2 * w1 * w2)
        Entries involving acoustic modes (w < acoustic_eps) are set to 0.
        """
        chi_list = []
        for iq1, iq2 in self.unique_pairs:
            w1 = self.w_q[:, iq1]
            w2 = self.w_q[:, iq2]
            n1, v1 = self._safe_bose_and_mask(iq1)
            n2, v2 = self._safe_bose_and_mask(iq2)
            valid_mask = np.outer(v1, v2)

            w1_mat = np.tile(w1, (self.n_bands, 1)).T
            w2_mat = np.tile(w2, (self.n_bands, 1))
            n1_mat = np.tile(n1, (self.n_bands, 1)).T
            n2_mat = np.tile(n2, (self.n_bands, 1))

            denom = 2.0 * w1_mat * w2_mat
            chi = np.where(valid_mask,
                           (w1_mat + w2_mat) * (1 + n1_mat + n2_mat) / np.where(valid_mask, denom, 1.0),
                           0.0)
            chi_list.append(chi)
        return chi_list

    def get_alpha1_beta1_wigner_q(self, get_alpha=True):
        """Get the perturbation on alpha (Upsilon) from the q-space psi.

        Transforms a'/b' blocks back to the alpha1 perturbation that the
        Julia code needs.

        alpha1[iq1, iq2] = (w1*w2/X) * [sqrt(-0.5*chi_minus)*a' - sqrt(0.5*chi_plus)*b']

        Returns
        -------
        list of ndarray(n_bands, n_bands) — one per unique pair
        """
        chi_minus_list = self.get_chi_minus_q()
        chi_plus_list = self.get_chi_plus_q()

        alpha1_blocks = []
        for pair_idx, (iq1, iq2) in enumerate(self.unique_pairs):
            w1 = self.w_q[:, iq1]
            w2 = self.w_q[:, iq2]
            n1, v1 = self._safe_bose_and_mask(iq1)
            n2, v2 = self._safe_bose_and_mask(iq2)
            valid_mask = np.outer(v1, v2)

            w1_mat = np.tile(w1, (self.n_bands, 1)).T
            w2_mat = np.tile(w2, (self.n_bands, 1))
            n1_mat = np.tile(n1, (self.n_bands, 1)).T
            n2_mat = np.tile(n2, (self.n_bands, 1))

            X = (1 + 2 * n1_mat) * (1 + 2 * n2_mat) / 8
            # Safe division for w2_on_X: when acoustic, X→∞ and w→0, set to 0
            w2_on_X = np.where(valid_mask,
                               (w1_mat * w2_mat) / np.where(valid_mask, X, 1.0),
                               0.0)
            chi_minus = chi_minus_list[pair_idx]
            chi_plus = chi_plus_list[pair_idx]

            a_block = self.get_a1_block(pair_idx)
            b_block = self.get_b1_block(pair_idx)

            if get_alpha:
                new_a = w2_on_X * np.sqrt(-0.5 * chi_minus) * a_block
                new_b = w2_on_X * np.sqrt(+0.5 * chi_plus) * b_block
                alpha1 = new_a - new_b
            else:
                X_safe = np.where(valid_mask, X, 1.0)
                new_a = np.where(valid_mask,
                                 (np.sqrt(-0.5 * chi_minus) / X_safe) * a_block,
                                 0.0)
                new_b = np.where(valid_mask,
                                 (np.sqrt(+0.5 * chi_plus) / X_safe) * b_block,
                                 0.0)
                alpha1 = new_a + new_b

            alpha1_blocks.append(alpha1)
        return alpha1_blocks

    def _flatten_blocks(self, blocks):
        """Flatten a list of (n_bands, n_bands) blocks into a contiguous array."""
        return np.concatenate([b.ravel() for b in blocks])

    def _unflatten_blocks(self, flat):
        """Unflatten a contiguous array back into a list of blocks."""
        nb = self.n_bands
        blocks = []
        offset = 0
        for iq1, iq2 in self.unique_pairs:
            size = nb * nb
            blocks.append(flat[offset:offset + size].reshape(nb, nb))
            offset += size
        return blocks

    def apply_anharmonic_FT(self, transpose=False, **kwargs):
        """Apply the anharmonic part of L in q-space (Wigner formalism).

        Calls the Julia q-space extension to compute the perturbed averages,
        then assembles the output psi vector.

        Returns
        -------
        ndarray(psi_size,), complex128
        """
        # If both D3 and D4 are ignored, return zero
        if self.ignore_v3 and self.ignore_v4:
            return np.zeros(self.get_psi_size(), dtype=np.complex128)

        import julia.Main

        R1 = self.get_R1_q()
        # If D3 is ignored, zero out R1 so that D3 weight is zero
        if self.ignore_v3:
            R1 = np.zeros_like(R1)
        alpha1_blocks = self.get_alpha1_beta1_wigner_q(get_alpha=True)
        alpha1_flat = self._flatten_blocks(alpha1_blocks)

        # Call Julia
        f_pert, d2v_blocks = self._call_julia_qspace(R1, alpha1_flat)

        # Build output psi
        final_psi = np.zeros(self.get_psi_size(), dtype=np.complex128)

        # R sector
        final_psi[:self.n_bands] = f_pert

        # a'/b' sectors
        chi_minus_list = self.get_chi_minus_q()
        chi_plus_list = self.get_chi_plus_q()

        for pair_idx, (iq1, iq2) in enumerate(self.unique_pairs):
            d2v_block = d2v_blocks[pair_idx]
            pert_a = np.sqrt(-0.5 * chi_minus_list[pair_idx]) * d2v_block
            pert_b = -np.sqrt(+0.5 * chi_plus_list[pair_idx]) * d2v_block
            self.set_block_in_psi(pair_idx, pert_a, 'a', final_psi)
            self.set_block_in_psi(pair_idx, pert_b, 'b', final_psi)

        return final_psi

    def _call_julia_qspace(self, R1, alpha1_flat):
        """Call the Julia q-space extension with MPI parallelization.

        Same MPI pattern as DynamicalLanczos.apply_anharmonic_FT.

        Returns
        -------
        f_pert : ndarray(n_bands,), complex128
        d2v_blocks : list of ndarray(n_bands, n_bands), complex128
        """
        import julia.Main

        n_total = self.n_syms_qspace * self.N
        n_processors = Parallel.GetNProc()

        count = n_total // n_processors
        remainer = n_total % n_processors
        indices = []
        for rank in range(n_processors):
            if rank < remainer:
                start = np.int64(rank * (count + 1))
                stop = np.int64(start + count + 1)
            else:
                start = np.int64(rank * count + remainer)
                stop = np.int64(start + count)
            indices.append([start + 1, stop])  # 1-indexed for Julia

        unique_pairs_arr = np.array(self.unique_pairs, dtype=np.int32) + 1  # 1-indexed

        def get_combined(start_end):
            return julia.Main.get_perturb_averages_qspace(
                self.X_q, self.Y_q, self.w_q, self.rho,
                R1, alpha1_flat,
                np.float64(self.T), bool(not self.ignore_v4),
                np.int64(self.iq_pert + 1),
                self.q_pair_map + 1,  # 1-indexed
                unique_pairs_arr,
                int(start_end[0]), int(start_end[1])
            )

        combined = Parallel.GoParallel(get_combined, indices, "+")
        f_pert = combined[:self.n_bands]
        d2v_flat = combined[self.n_bands:]
        d2v_blocks = self._unflatten_blocks(d2v_flat)
        return f_pert, d2v_blocks

    # ====================================================================
    # Step 5+6: Combined L application (override apply_full_L)
    # ====================================================================
    def apply_full_L(self, target=None, force_t_0=False, force_FT=True,
                     transpose=False, fast_lanczos=True):
        """Apply the full L operator in q-space.

        Parameters
        ----------
        target : ndarray or None
            If provided, copy into self.psi first.
        transpose : bool
            Not used for Hermitian Lanczos.

        Returns
        -------
        ndarray(psi_size,), complex128
        """
        if target is not None:
            self.psi = target.copy()

        result = self.apply_L1_FT(transpose=transpose)
        result += self.apply_anharmonic_FT(transpose=transpose)

        return result

    # ====================================================================
    # Step 7: Hermitian Lanczos (run_FT override)
    # ====================================================================
    def run_FT(self, n_iter, save_dir=None, save_each=5, verbose=True,
               n_rep_orth=0, n_ortho=10, flush_output=True, debug=False,
               prefix="LANCZOS", run_simm=None, optimized=False):
        """Run the Hermitian Lanczos algorithm for q-space.

        This is the same structure as the parent run_FT but with:
        1. Forced run_simm = True (Hermitian)
        2. Hermitian dot products: psi.conj().dot(psi * mask).real
        3. Complex128 psi
        4. Real coefficients (guaranteed by Hermitian L)
        """
        self.verbose = verbose

        if not self.initialized:
            if verbose:
                print('Not initialized. Now we symmetrize\n')
            self.prepare_symmetrization()

        ERROR_MSG = """
Error, you must initialize a perturbation to start the Lanczos.
Use prepare_mode_q or prepare_perturbation_q before calling run_FT.
"""
        if self.psi is None:
            raise ValueError(ERROR_MSG)

        mask_dot = self.mask_dot_wigner(debug)
        psi_norm = np.real(np.conj(self.psi).dot(self.psi * mask_dot))
        if np.isnan(psi_norm) or psi_norm == 0:
            raise ValueError(ERROR_MSG)

        # Force symmetric Lanczos
        run_simm = True

        if Parallel.am_i_the_master():
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

        if verbose:
            print('Running the Hermitian Lanczos in q-space')
            print()

        # Get current step
        i_step = len(self.a_coeffs)

        if verbose:
            header = """
<=====================================>
|                                     |
|     Q-SPACE LANCZOS ALGORITHM       |
|                                     |
<=====================================>

Starting the algorithm.
Starting from step %d
""" % i_step
            print(header)

        # Initialize
        if i_step == 0:
            self.basis_Q = []
            self.basis_P = []
            self.s_norm = []
            norm = np.sqrt(np.real(np.conj(self.psi).dot(self.psi * mask_dot)))
            first_vector = self.psi / norm
            self.basis_Q.append(first_vector)
            self.basis_P.append(first_vector)
            self.s_norm.append(1)
        else:
            if verbose:
                print('Restarting the Lanczos')
            self.basis_Q = list(self.basis_Q)
            self.basis_P = list(self.basis_P)
            self.s_norm = list(self.s_norm)
            self.a_coeffs = list(self.a_coeffs)
            self.b_coeffs = list(self.b_coeffs)
            self.c_coeffs = list(self.c_coeffs)

        psi_q = self.basis_Q[-1]
        psi_p = self.basis_P[-1]

        next_converged = False
        converged = False

        for i in range(i_step, i_step + n_iter):
            if verbose:
                print("\n ===== NEW STEP %d =====\n" % (i + 1))
                if flush_output:
                    sys.stdout.flush()

            # Apply L (Hermitian => p_L = L_q)
            t1 = time.time()
            L_q = self.apply_full_L(psi_q)
            p_L = np.copy(L_q)
            t2 = time.time()

            # p normalization
            c_old = 1
            if len(self.c_coeffs) > 0:
                c_old = self.c_coeffs[-1]
            p_norm = self.s_norm[-1] / c_old

            # a coefficient (real for Hermitian L)
            a_coeff = np.real(np.conj(psi_p).dot(L_q * mask_dot)) * p_norm

            if np.isnan(a_coeff):
                raise ValueError("Invalid value in Lanczos. Check frequencies/initialization.")

            # Residuals
            rk = L_q - a_coeff * psi_q
            if len(self.basis_Q) > 1:
                rk -= self.c_coeffs[-1] * self.basis_Q[-2]

            sk = p_L - a_coeff * psi_p
            if len(self.basis_P) > 1:
                old_p_norm = self.s_norm[-2]
                if len(self.c_coeffs) >= 2:
                    old_p_norm = self.s_norm[-2] / self.c_coeffs[-2]
                sk -= self.b_coeffs[-1] * self.basis_P[-2] * (old_p_norm / p_norm)

            # s_norm
            s_norm = np.sqrt(np.real(np.conj(sk).dot(sk * mask_dot)))
            sk_tilde = sk / s_norm
            s_norm *= p_norm

            # b and c coefficients (real, should be equal for Hermitian L)
            b_coeff = np.sqrt(np.real(np.conj(rk).dot(rk * mask_dot)))
            c_coeff = np.real(np.conj(sk_tilde).dot((rk / b_coeff) * mask_dot)) * s_norm

            self.a_coeffs.append(a_coeff)

            if np.abs(b_coeff) < __EPSILON__ or next_converged:
                if verbose:
                    print("Converged (b = {})".format(b_coeff))
                converged = True
                break
            if np.abs(c_coeff) < __EPSILON__:
                if verbose:
                    print("Converged (c = {})".format(c_coeff))
                converged = True
                break

            psi_q = rk / b_coeff
            psi_p = sk_tilde.copy()

            # Gram-Schmidt
            new_q = psi_q.copy()
            new_p = psi_p.copy()

            for k_orth in range(n_rep_orth):
                start = max(0, len(self.basis_P) - (n_ortho or len(self.basis_P)))

                for j in range(start, len(self.basis_P)):
                    coeff1 = np.real(np.conj(self.basis_P[j]).dot(new_q * mask_dot))
                    coeff2 = np.real(np.conj(self.basis_Q[j]).dot(new_p * mask_dot))
                    new_q -= coeff1 * self.basis_P[j]
                    new_p -= coeff2 * self.basis_Q[j]

                normq = np.sqrt(np.real(np.conj(new_q).dot(new_q * mask_dot)))
                if normq < __EPSILON__:
                    next_converged = True
                new_q /= normq

                normp = np.real(np.conj(new_p).dot(new_p * mask_dot))
                if np.abs(normp) < __EPSILON__:
                    next_converged = True
                new_p /= normp

                s_norm = c_coeff / np.real(np.conj(new_p).dot(new_q * mask_dot))

            if not converged:
                self.basis_Q.append(new_q)
                self.basis_P.append(new_p)
                psi_q = new_q.copy()
                psi_p = new_p.copy()

                self.b_coeffs.append(b_coeff)
                self.c_coeffs.append(c_coeff)
                self.s_norm.append(s_norm)

            if verbose:
                print("Time for L application: %d s" % (t2 - t1))
                print("a_%d = %.8e" % (i, self.a_coeffs[-1]))
                print("b_%d = %.8e" % (i, self.b_coeffs[-1]))
                print("c_%d = %.8e" % (i, self.c_coeffs[-1]))
                print("|b-c| = %.8e" % np.abs(self.b_coeffs[-1] - self.c_coeffs[-1]))

            if save_dir is not None:
                if (i + 1) % save_each == 0:
                    self.save_status("%s/%s_STEP%d" % (save_dir, prefix, i + 1))

            if verbose:
                print("Lanczos step %d completed." % (i + 1))

        if converged and verbose:
            print("   last a coeff = {}".format(self.a_coeffs[-1]))

    # ====================================================================
    # Step 8: Perturbation setup
    # ====================================================================
    def prepare_mode_q(self, iq, band_index):
        """Prepare perturbation for mode (q, nu).

        Parameters
        ----------
        iq : int
            Index of the q-point.
        band_index : int
            Band index (0-based).
        """
        self.build_q_pair_map(iq)
        self.reset_q()
        self.psi[band_index] = 1.0 + 0j
        self.perturbation_modulus = 1.0

    def prepare_perturbation_q(self, iq, vector):
        """Prepare perturbation at q from a real-space vector (3*n_at_uc,).

        Projects the vector onto q-space eigenmodes at iq.

        Parameters
        ----------
        iq : int
            Index of the q-point.
        vector : ndarray(3*n_at_uc,)
            Perturbation vector in Cartesian real space.
        """
        self.build_q_pair_map(iq)
        self.reset_q()
        m = np.tile(self.uci_structure.get_masses_array(), (3, 1)).T.ravel()
        v_scaled = vector / np.sqrt(m)
        R1 = np.conj(self.pols_q[:, :, iq]).T @ v_scaled  # (n_bands,) complex
        self.psi[:self.n_bands] = R1
        self.perturbation_modulus = np.real(np.conj(R1) @ R1)

    def reset_q(self):
        """Reset the Lanczos state for q-space."""
        n = self.get_psi_size()
        self.psi = np.zeros(n, dtype=np.complex128)

        self.eigvals = None
        self.eigvects = None

        self.a_coeffs = []
        self.b_coeffs = []
        self.c_coeffs = []
        self.krilov_basis = []
        self.basis_P = []
        self.basis_Q = []
        self.s_norm = []

    # ====================================================================
    # Step 11: Q-space symmetry matrix construction
    # ====================================================================
    def prepare_symmetrization(self, no_sym=False, verbose=True, symmetries=None):
        """Build q-space symmetry matrices and cache them in Julia.

        Overrides the parent to build sparse complex symmetry matrices
        in the q-space mode basis.

        Uses spglib on the unit cell (not supercell) to get correct
        fractional-coordinate rotations and translations, then converts
        to Cartesian for the representation matrices.
        """
        self.initialized = True

        if no_sym:
            # Identity only
            self.n_syms_qspace = 1
            n_total = self.n_q * self.n_bands
            # Build identity sparse matrix
            import julia.Main
            julia.Main.eval("""
            function init_identity_qspace(n_total::Int64)
                I_sparse = SparseArrays.sparse(
                    Int32.(1:n_total), Int32.(1:n_total),
                    ComplexF64.(ones(n_total)), n_total, n_total)
                _cached_qspace_symmetries[] = [I_sparse]
                return nothing
            end
            """)
            julia.Main.init_identity_qspace(np.int64(n_total))
            return

        if not __SPGLIB__:
            raise ImportError("spglib required for symmetrization")

        # Get symmetries from the UNIT CELL directly via spglib.
        # spglib returns rotations and translations in fractional
        # (crystal) coordinates. We convert rotations to Cartesian via
        # R_cart = M @ R_frac @ M^{-1} where M = unit_cell.T.
        spg_data = spglib.get_symmetry(self.uci_structure.get_spglib_cell())
        rot_frac_all = spg_data['rotations']   # (n_sym, 3, 3) integer
        trans_frac_all = spg_data['translations']  # (n_sym, 3) fractional

        M = self.uci_structure.unit_cell.T       # columns = lattice vectors
        Minv = np.linalg.inv(M)

        # Extract unique point-group rotations (keep first occurrence)
        unique_pg = {}
        for i in range(len(rot_frac_all)):
            key = rot_frac_all[i].tobytes()
            if key not in unique_pg:
                unique_pg[key] = i
        pg_indices = list(unique_pg.values())

        if verbose:
            print("Q-space: {} PG symmetries from {} total unit cell symmetries".format(
                len(pg_indices), len(rot_frac_all)))

        self._build_qspace_symmetries(
            rot_frac_all, trans_frac_all, pg_indices, M, Minv,
            verbose=verbose)

    @staticmethod
    def _get_atom_perm(structure, R_cart, t_cart, M, Minv, tol=0.1):
        """Find atom permutation under symmetry {R|t}.

        Returns irt such that R @ tau[kappa] + t ≡ tau[irt[kappa]] mod lattice.
        """
        nat = structure.N_atoms
        irt = np.zeros(nat, dtype=int)
        for kappa in range(nat):
            tau = structure.coords[kappa]
            mapped = R_cart @ tau + t_cart
            for kp in range(nat):
                diff = mapped - structure.coords[kp]
                diff_frac = Minv @ diff
                diff_frac -= np.round(diff_frac)
                if np.linalg.norm(M @ diff_frac) < tol:
                    irt[kappa] = kp
                    break
        return irt

    def _build_qspace_symmetries(self, rot_frac_all, trans_frac_all,
                                  pg_indices, M, Minv, verbose=True):
        """Build sparse complex symmetry matrices for q-space modes.

        For each PG symmetry {R|t}:
          - Maps q -> Rq (permutes q-points)
          - Rotates bands within each q-block via
            D_{nu',nu}(iq'<-iq) = conj(pols_q[:,nu',iq']).T @ P_uc(q') @ pols_q[:,nu,iq]
          - P_uc includes Cartesian rotation, atom permutation, and Bloch phase:
            P_uc[3*kp:3*kp+3, 3*k:3*k+3] = exp(-2*pi*i * q' . L_k) * R_cart
            where L_k = R_cart @ tau_k + t_cart - tau_kp is a lattice vector.
        """
        import julia.Main

        nat_uc = self.uci_structure.N_atoms
        bg = self.uci_structure.get_reciprocal_vectors() / (2 * np.pi)
        n_total = self.n_q * self.n_bands
        nb = self.n_bands

        n_syms = len(pg_indices)
        self.n_syms_qspace = n_syms

        # Build all sparse matrices in Python, then pass to Julia
        all_rows = []
        all_cols = []
        all_vals = []

        for i_sym_idx in pg_indices:
            R_frac = rot_frac_all[i_sym_idx].astype(float)
            t_frac = trans_frac_all[i_sym_idx]

            # Convert rotation and translation to Cartesian
            R_cart = M @ R_frac @ Minv
            t_cart = M @ t_frac

            # Get atom permutation
            irt = self._get_atom_perm(
                self.uci_structure, R_cart, t_cart, M, Minv)

            rows, cols, vals = [], [], []

            for iq in range(self.n_q):
                q = self.q_points[iq]
                Rq = R_cart @ q

                # Find iq' matching Rq
                iq_prime = find_q_index(Rq, self.q_points, bg)
                q_prime = self.q_points[iq_prime]

                # Build P_uc with Bloch phase factor
                P_uc = np.zeros((3 * nat_uc, 3 * nat_uc), dtype=np.complex128)
                for kappa in range(nat_uc):
                    kp = irt[kappa]
                    tau_k = self.uci_structure.coords[kappa]
                    tau_kp = self.uci_structure.coords[kp]
                    # L is the lattice vector: R@tau + t - tau'
                    L = R_cart @ tau_k + t_cart - tau_kp
                    phase = np.exp(-2j * np.pi * q_prime @ L)
                    P_uc[3 * kp:3 * kp + 3,
                         3 * kappa:3 * kappa + 3] = phase * R_cart

                # D block: representation matrix in eigenvector basis
                D = np.conj(self.pols_q[:, :, iq_prime]).T @ P_uc @ self.pols_q[:, :, iq]

                # Add to sparse entries
                for nu1 in range(nb):
                    for nu2 in range(nb):
                        if abs(D[nu1, nu2]) > 1e-12:
                            rows.append(iq_prime * nb + nu1)
                            cols.append(iq * nb + nu2)
                            vals.append(D[nu1, nu2])

            all_rows.append(np.array(rows, dtype=np.int32))
            all_cols.append(np.array(cols, dtype=np.int32))
            all_vals.append(np.array(vals, dtype=np.complex128))

        # Pass to Julia for caching (convert to 1-indexed)
        for i in range(n_syms):
            all_rows[i] += 1
            all_cols[i] += 1

        julia.Main.eval("""
        function init_sparse_symmetries_qspace(
            all_rows::Vector{Vector{Int32}},
            all_cols::Vector{Vector{Int32}},
            all_vals::Vector{Vector{ComplexF64}},
            n_total::Int64
        )
            n_syms = length(all_rows)
            mats = Vector{SparseMatrixCSC{ComplexF64,Int32}}(undef, n_syms)
            for i in 1:n_syms
                mats[i] = sparse(
                    all_rows[i], all_cols[i], all_vals[i], n_total, n_total)
            end
            _cached_qspace_symmetries[] = mats
            return nothing
        end
        """)

        julia.Main.init_sparse_symmetries_qspace(
            all_rows, all_cols, all_vals, np.int64(n_total))

        if verbose:
            print("Q-space symmetry matrices ({} x {}), {} symmetries cached in Julia".format(
                n_total, n_total, n_syms))

    # Override init to use q-space symmetrization
    def init(self, use_symmetries=True):
        """Initialize the q-space Lanczos calculation."""
        self.prepare_symmetrization(no_sym=not use_symmetries)
        self.initialized = True
