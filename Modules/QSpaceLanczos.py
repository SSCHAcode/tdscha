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

import sscha.Ensemble

# MPI support for distributed mode
__MPI4PY__ = False
try:
    import mpi4py.MPI
    __MPI4PY__ = True
except ImportError:
    pass

# The Julia runtime is booted lazily by JuliaExt at the first actual use
# (tdscha_qspace.jl is included by JuliaExt.get_main()), so that importing
# tdscha.QSpaceLanczos stays fast.
import tdscha.JuliaExt as JuliaExt

# Deprecated alias kept for backward compatibility: it only tells whether a
# Julia backend is installed, the runtime is not initialized at import time.
__JULIA_EXT__ = JuliaExt.available()

try:
    import spglib
    __SPGLIB__ = True
except ImportError:
    __SPGLIB__ = False


# Constants
__EPSILON__ = 1e-12
__RyToK__ = 157887.32400374097
TYPE_DP = np.double


def check_numpy_version():
    """Refuse to run the q-space Lanczos under a NumPy that corrupts it.

    NumPy 1.26.4 on Python 3.14 lets a masked product alias its input and
    silently mutates the Krylov vectors (see
    ``numpy1_python314_qspace_issue.md``).  The corruption is NOT detectable
    from the Hermiticity invariant: on a full 12288-configuration CsSnI3
    ensemble the failing environment returns ``b - c == 0`` exactly and
    still gives ``b[0] = 4.53e-4`` instead of the correct ``1.13e-7`` --
    two orders of magnitude above the largest possible two-phonon
    eigenvalue.  The run completes, the coefficients are finite, and the
    spectrum is wrong.

    Raised at the start of every Lanczos recursion rather than at import, so
    that analysis code which only reads stored coefficients keeps working
    under any NumPy.

    Raises
    ------
    RuntimeError
        If the NumPy major version is below 2.
    """
    major = int(np.__version__.split(".")[0])
    if major < 2:
        raise RuntimeError(
            "The q-space Lanczos cannot be run with NumPy %s.\n\n"
            "NumPy 1.x on this interpreter aliases the masked metric "
            "products and silently corrupts the Krylov vectors: the "
            "recursion completes with finite coefficients and with b == c "
            "to machine precision, but the coefficients -- and therefore "
            "the spectrum -- are wrong (see "
            "numpy1_python314_qspace_issue.md).\n\n"
            "Install NumPy >= 2 in this environment, or prepend one to "
            "PYTHONPATH (and pass '-x PYTHONPATH' to mpirun so that every "
            "rank inherits it)." % np.__version__)


# Number of Krylov vectors retained by run_FT(optimized=True).  The
# non-reorthogonalized three-term recurrence only ever reads basis_Q[-1],
# basis_Q[-2] (and the matching P/s_norm entries), so three is already one
# more than it needs; the extra slot keeps the restart path -- which resumes
# from [-1]/[-2] after load_status -- comfortably inside the window.  Same
# convention as DynamicalLanczos.run_FT(optimized=True).
_KEEP_BASIS_OPTIMIZED = 3


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

    # Attributes that ``load_distributed_tdscha`` must carry from the master
    # to the worker ranks on top of the common q-space state.  Subclasses that
    # add their own structure (see QSpaceAtomFourierLanczos) list it here so
    # the distributed loader stays a single code path: the master builds these
    # once and broadcasts them, rather than every rank rebuilding them and
    # risking a degenerate-subspace gauge mismatch between ranks.
    _DISTRIBUTED_EXTRA_ATTRS = ()

    @classmethod
    def prepare_distributed_construction(cls, dyn, **kwargs):
        """Run the collective part of the construction on every MPI rank.

        ``load_distributed_tdscha`` reads the ensemble on the master alone,
        so anything the constructor does that involves an MPI collective
        would leave the workers -- parked in the metadata broadcast -- in a
        different collective.  MPI does not diagnose the mismatch: the run
        either hangs or silently delivers one collective's payload to the
        other's receiver.

        The loader therefore calls this classmethod on **all** ranks before
        the master/worker split.  Whatever it returns is merged into the
        constructor keyword arguments, so the collective work is already
        done by the time the master builds the object alone.

        The plain q-space construction has no collective step, so the base
        implementation returns an empty mapping.  Subclasses whose
        constructor calls into CellConstructor's ``ForceTensor`` -- see
        :class:`~tdscha.QSpaceAtomFourier.QSpaceAtomFourierLanczos` --
        override it.

        Parameters
        ----------
        dyn : CC.Phonons.Phonons
            The dynamical matrix the object will be built on: the ensemble's
            ``current_dyn``, i.e. ``final_dyn`` when the loader reweights.
        **kwargs
            The constructor keyword arguments of the loader call.

        Returns
        -------
        dict
            Extra keyword arguments for the constructor.
        """
        return {}

    def __init__(self, ensemble, lo_to_split=None, **kwargs):
        """Initialize the Q-Space Lanczos.

        Parameters
        ----------
        ensemble : sscha.Ensemble.Ensemble
            The SSCHA ensemble.
        lo_to_split : string, ndarray, or None
            LO-TO splitting mode. If None (default), LO-TO splitting correction
            is neglected. If "random", a random direction is used. If an ndarray,
            it specifies the q-direction for the LO-TO splitting correction.
        **kwargs
            Additional keyword arguments passed to the parent Lanczos class.
        """
        # Force Wigner and Julia mode
        kwargs['mode'] = DL.MODE_FAST_JULIA
        kwargs['use_wigner'] = True

        self.ensemble = ensemble
        super().__init__(ensemble, unwrap_symmetries=False, lo_to_split=lo_to_split, **kwargs)

        if not JuliaExt.available():
            raise ImportError(
                "QSpaceLanczos requires Julia. Install with: pip install juliacall"
            )
        self.use_wigner = True

        # -- Add the q-space attributes --
        qspace_attrs = [
            'q_points', 'n_q', 'n_bands', 'w_q', 'pols_q',
            'valid_modes_q', 'X_q', 'Y_q',
            'iq_pert', 'q_pair_map', 'unique_pairs',
            '_psi_size', '_block_offsets_a', '_block_offsets_b', '_block_sizes',
            '_qspace_sym_data', '_qspace_sym_q_map', 'n_syms_qspace',
            # Vertex renormalization for q-mesh interpolation (1.0 = no interp)
            'qspace_scale3', 'qspace_scale4', 'qspace_prefiltered',
            # Distributed mode attributes
            '_distributed', '_N_global', '_N_eff_global', '_N_local',
        ]
        self.__total_attributes__.extend(qspace_attrs)

        # D3/D4 vertex rescaling factors passed to the Julia kernel.
        # They stay 1.0 for a standard (commensurate) calculation; the
        # interpolated subclass sets sqrt(N_c/N_f) and N_c/N_f respectively
        # (see Interpolation_plan.md, section 6).
        self.qspace_scale3 = 1.0
        self.qspace_scale4 = 1.0
        # True when X_q already carries the f_Y filter and the f_psi factors
        # are folded into alpha1 (set by the interpolated subclass).
        self.qspace_prefiltered = False

        # If ensemble is None, perform a bare initialization like the parent
        if ensemble is None:
            return

        # == 1. Get q-space eigenmodes ==
        ws_sc, pols_sc, w_q, pols_q = self.dyn.DiagonalizeSupercell(
            return_qmodes=True, lo_to_split=lo_to_split)

        self.q_points = np.array(self.dyn.q_tot)  # (n_q, 3)
        self.n_q = len(self.q_points)
        self.n_bands = 3 * self.uci_structure.N_atoms  # uniform band count
        self.w_q = w_q        # (n_bands, n_q) from DiagonalizeSupercell
        self.pols_q = pols_q  # (3*n_at, n_bands, n_q) complex eigenvectors

        # The masses needs to be restricted to the primitive cell only
        self.m = self.m[:self.n_bands]

        # Build valid_modes_q mask for acoustic mode exclusion
        # Always apply translation-based mask at Gamma (iq=0)
        masses_uc = self.dyn.structure.get_masses_array()
        self.valid_modes_q = np.ones((self.n_bands, self.n_q), dtype=bool)
        trans_mask = CC.Methods.get_translations(np.real(self.pols_q[:, :, 0]), masses_uc)
        self.valid_modes_q[:, 0] = ~trans_mask

        # If ignore_small_w is True, also mask small frequencies at ALL q-points
        if ensemble.ignore_small_w:
            for iq in range(self.n_q):
                small_freq_mask = np.abs(self.w_q[:, iq]) < CC.Phonons.__EPSILON_W__
                self.valid_modes_q[:, iq] &= ~small_freq_mask

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

        # Distributed mode attributes (for MPI configuration distribution)
        self._distributed = False
        self._N_global = self.N
        self._N_eff_global = self.N_eff
        self._N_local = self.N

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

    def get_static_psi_size(self):
        """Return psi size for the static layout: [R, one W sector].

        This equals the end of the a' sector, i.e. the start of b'.
        """
        return self._block_offsets_b[0]

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
        Off-diagonal: M[j, i] = M[i, j] — the diagonal-pair (q, q) blocks are
        complex SYMMETRIC in the bilinear convention (q1 + q2 = q_pert), not
        Hermitian: the implied reverse block is the transpose.
        """
        mat = np.zeros((n, n), dtype=np.complex128)
        idx = 0
        for i in range(n):
            length = n - i
            mat[i, i:] = flat_data[idx:idx + length]
            idx += length
        # Fill lower triangle (symmetric, no conjugation)
        for i in range(n):
            mat[i + 1:, i] = mat[i, i + 1:]
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

    def get_block(self, pair_idx, sector='a', source=None):
        """Reconstruct full (n_bands, n_bands) matrix from psi storage.

        Parameters
        ----------
        pair_idx : int
            Index into self.unique_pairs.
        sector : str
            'a' or 'b'.
        source : ndarray or None
            If provided, read from this array instead of self.psi.

        Returns
        -------
        ndarray(n_bands, n_bands), complex128
        """
        nb = self.n_bands
        iq1, iq2 = self.unique_pairs[pair_idx]
        offset = self.get_block_offset(pair_idx, sector)
        size = self.get_block_size(pair_idx)

        data = self.psi if source is None else source
        raw = data[offset:offset + size]

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

        # R sector — mask acoustic modes
        w_qp = self.w_q[:, self.iq_pert]  # (n_bands,)
        v_pert = self.valid_modes_q[:, self.iq_pert]
        out[:self.n_bands] = -(w_qp ** 2) * self.psi[:self.n_bands]
        out[:self.n_bands] *= v_pert  # zero out acoustic

        # a' and b' sectors — mask pairs involving acoustic modes
        for pair_idx, (iq1, iq2) in enumerate(self.unique_pairs):
            w1 = self.w_q[:, iq1]  # (n_bands,)
            w2 = self.w_q[:, iq2]  # (n_bands,)
            v1 = self.valid_modes_q[:, iq1]
            v2 = self.valid_modes_q[:, iq2]
            valid_mask = np.outer(v1, v2)

            a_block = self.get_a1_block(pair_idx)
            b_block = self.get_b1_block(pair_idx)

            w_minus2 = np.subtract.outer(w1, w2) ** 2  # (n_bands, n_bands)
            w_plus2 = np.add.outer(w1, w2) ** 2

            self.set_block_in_psi(pair_idx, -w_minus2 * a_block * valid_mask, 'a', out)
            self.set_block_in_psi(pair_idx, -w_plus2 * b_block * valid_mask, 'b', out)

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
        valid = self.valid_modes_q[:, iq]
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
        """Flatten a list of (n_bands, n_bands) blocks into a contiguous array.

        Uses Fortran (column-major) order to match Julia's column-major storage.
        """
        return np.concatenate([b.ravel(order='F') for b in blocks])

    def _unflatten_blocks(self, flat):
        """Unflatten a contiguous array back into a list of blocks.

        Uses Fortran (column-major) order to match Julia's column-major storage.
        """
        nb = self.n_bands
        blocks = []
        offset = 0
        for iq1, iq2 in self.unique_pairs:
            size = nb * nb
            blocks.append(flat[offset:offset + size].reshape(nb, nb, order='F'))
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

        In distributed mode, dispatches to _call_julia_qspace_distributed.

        Returns
        -------
        f_pert : ndarray(n_bands,), complex128
        d2v_blocks : list of ndarray(n_bands, n_bands), complex128
        """
        # Dispatch to distributed version if in distributed mode
        if self._distributed:
            return self._call_julia_qspace_distributed(R1, alpha1_flat)

        jl = JuliaExt.get_main()

        n_active_syms = self._spectroscopy_symmetry_count(
            self.n_syms_qspace)
        reduction_args = self._spectroscopy_reduction_arguments()
        n_total = n_active_syms * self.N
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
        valid_modes = np.array(self.valid_modes_q, dtype=np.bool_)
        iq_pert_jl = int(self.iq_pert) + 1
        q_pair_map_jl = np.array(self.q_pair_map, dtype=np.int32) + 1

        def get_combined(start_end):
            return jl.get_perturb_averages_qspace(
                self.X_q, self.Y_q, self.w_q, self.rho,
                R1, alpha1_flat,
                float(self.T), bool(not self.ignore_v4),
                iq_pert_jl,
                q_pair_map_jl,  # 1-indexed
                unique_pairs_arr,
                int(start_end[0]), int(start_end[1]),
                valid_modes,  # Pass mask to Julia
                float(self.qspace_scale3), float(self.qspace_scale4),
                bool(self.qspace_prefiltered), *reduction_args
            )

        combined = Parallel.GoParallel(get_combined, indices, "+")
        f_pert = combined[:self.n_bands]
        d2v_flat = combined[self.n_bands:]
        d2v_blocks = self._unflatten_blocks(d2v_flat)
        return f_pert, d2v_blocks

    def _call_julia_qspace_distributed(self, R1, alpha1_flat):
        """Call Julia with distributed configurations across MPI ranks.

        In distributed mode, each process holds only N_local configs.
        This method:
        1. Calls Julia with all n_syms * N_local (config, sym) pairs on local data
        2. Un-normalizes the Julia result by multiplying by N_eff_local
           (Julia divides by n_syms * sum(rho_local) internally)
        3. MPI Allreduce(SUM) across all processes
        4. Divides by N_eff_global to get the correctly normalized average

        Normalization proof:
        - Julia returns partial_k / (n_syms * N_eff_local_k)
        - Multiply by N_eff_local_k -> partial_k / n_syms
        - Allreduce(SUM) -> sum_k(partial_k) / n_syms
        - Divide by N_eff_global -> total / (n_syms * N_eff_global) ✓

        Returns
        -------
        f_pert : ndarray(n_bands,), complex128
        d2v_blocks : list of ndarray(n_bands, n_bands), complex128
        """
        jl = JuliaExt.get_main()

        if not __MPI4PY__:
            raise RuntimeError(
                "Distributed mode requires MPI (mpi4py). "
                "Use load_distributed_tdscha with MPI or disable distribution."
            )

        comm = mpi4py.MPI.COMM_WORLD

        # Compute local config slice
        # N_local may differ from original N due to distribution
        N_local = self.N
        N_eff_local = self.N_eff
        N_eff_global = self._N_eff_global

        # Handle edge case: this proc has 0 configs
        if N_local == 0:
            # Contribute zero result
            f_pert = np.zeros(self.n_bands, dtype=np.complex128)
            d2v_flat = np.zeros(len(self.unique_pairs) * self.n_bands * self.n_bands,
                               dtype=np.complex128)
            # Allreduce to get correct shape on all procs (all zeros)
            f_pert_global = np.zeros_like(f_pert)
            d2v_flat_global = np.zeros_like(d2v_flat)
            comm.Allreduce(f_pert, f_pert_global, op=mpi4py.MPI.SUM)
            comm.Allreduce(d2v_flat, d2v_flat_global, op=mpi4py.MPI.SUM)
            d2v_blocks = self._unflatten_blocks(d2v_flat_global)
            return f_pert_global, d2v_blocks

        # Total number of (config, sym) pairs for this proc
        n_active_syms = self._spectroscopy_symmetry_count(
            self.n_syms_qspace)
        reduction_args = self._spectroscopy_reduction_arguments()
        n_total_local = n_active_syms * N_local

        # Build indices for this proc (1-indexed for Julia)
        indices = [[1, n_total_local]]  # Single element list for local range

        unique_pairs_arr = np.array(self.unique_pairs, dtype=np.int32) + 1  # 1-indexed

        def get_combined_local(start_end):
            # Julia will process self.N configs (which is now N_local)
            # Use local rho slice
            rho_local = self.rho[:N_local]
            # Convert types explicitly for Julia
            valid_modes = np.array(self.valid_modes_q, dtype=np.bool_)
            iq_pert_jl = int(self.iq_pert) + 1
            q_pair_map_jl = np.array(self.q_pair_map, dtype=np.int32) + 1
            return jl.get_perturb_averages_qspace(
                self.X_q, self.Y_q, self.w_q, rho_local,
                R1, alpha1_flat,
                float(self.T), bool(not self.ignore_v4),
                iq_pert_jl,
                q_pair_map_jl,  # 1-indexed
                unique_pairs_arr,
                int(start_end[0]), int(start_end[1]),
                valid_modes,  # Pass mask to Julia
                float(self.qspace_scale3), float(self.qspace_scale4),
                bool(self.qspace_prefiltered), *reduction_args
            )

        # Call Julia (serial call, local configs only)
        combined_local = get_combined_local(indices[0])

        # Un-normalize: Julia divides by n_syms * sum(rho_local)
        # We want to divide by n_syms * N_eff_local to get partial / n_syms
        if N_eff_local > 0:
            combined_local = combined_local * N_eff_local
        # If N_eff_local == 0, keep zero (avoid division by zero)

        # Split into f_pert and d2v components
        f_pert_local = combined_local[:self.n_bands]
        d2v_flat_local = combined_local[self.n_bands:]

        # Allreduce to sum across all processes
        f_pert_global = np.zeros_like(f_pert_local)
        d2v_flat_global = np.zeros_like(d2v_flat_local)

        comm.Allreduce(f_pert_local, f_pert_global, op=mpi4py.MPI.SUM)
        comm.Allreduce(d2v_flat_local, d2v_flat_global, op=mpi4py.MPI.SUM)

        # Final normalization: divide by N_eff_global
        if N_eff_global > 0:
            f_pert_global = f_pert_global / N_eff_global
            d2v_flat_global = d2v_flat_global / N_eff_global

        d2v_blocks = self._unflatten_blocks(d2v_flat_global)
        return f_pert_global, d2v_blocks

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
               prefix="LANCZOS", run_simm=None, optimized=False,
               reorthogonalize=False):
        """Run the Hermitian Lanczos algorithm for q-space.

        This is the same structure as the parent run_FT but with:
        1. Forced run_simm = True (Hermitian)
        2. Hermitian dot products: psi.conj().dot(psi * mask).real
        3. Complex128 psi
        4. Real coefficients (guaranteed by Hermitian L)
        """
        # Before anything else: a NumPy that corrupts the recursion must stop
        # the run here, not after symmetrization has already been done.
        check_numpy_version()

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

        def metric_dot(left, right):
            """Hermitian product without allowing ufunc buffer reuse."""
            weighted_right = np.empty_like(right)
            np.multiply(right, mask_dot, out=weighted_right)
            return np.vdot(left, weighted_right)

        psi_norm = np.real(metric_dot(self.psi, self.psi))
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

        # `optimized` keeps only the tail of the Krylov basis.  That is exact
        # for the bare three-term recurrence, but silently wrong for anything
        # that re-reads older vectors, so refuse those combinations rather
        # than quietly changing the result.
        if optimized:
            if reorthogonalize:
                raise ValueError(
                    "optimized=True keeps only the last %d Krylov vectors, "
                    "while reorthogonalize=True re-orthogonalizes against the "
                    "whole basis. Use one or the other."
                    % _KEEP_BASIS_OPTIMIZED)
            if n_rep_orth > 0 and (not n_ortho
                                   or n_ortho > _KEEP_BASIS_OPTIMIZED):
                raise ValueError(
                    "optimized=True keeps only the last %d Krylov vectors, "
                    "but n_rep_orth=%d requests re-orthogonalization against "
                    "%s of them."
                    % (_KEEP_BASIS_OPTIMIZED, n_rep_orth,
                       "all" if not n_ortho else str(n_ortho)))

        # A basis that was truncated by a previous optimized run cannot be
        # reorthogonalized against: len(basis) < i_step + 1 is the signature.
        if reorthogonalize and i_step > 0 and len(self.basis_Q) < i_step + 1:
            raise ValueError(
                "Cannot continue with reorthogonalize=True: the stored Krylov "
                "basis holds %d vectors but %d steps were run, so it was "
                "truncated by an earlier optimized=True run and the older "
                "vectors are gone." % (len(self.basis_Q), i_step))

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
            norm = np.sqrt(np.real(metric_dot(self.psi, self.psi)))
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
            a_coeff = np.real(metric_dot(psi_p, L_q)) * p_norm

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
            s_norm = np.sqrt(np.real(metric_dot(sk, sk)))
            sk_tilde = sk / s_norm
            s_norm *= p_norm

            # b and c coefficients (real, should be equal for Hermitian L)
            b_coeff = np.sqrt(np.real(metric_dot(rk, rk)))
            c_coeff = np.real(metric_dot(
                sk_tilde, rk / b_coeff)) * s_norm

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

            # Gram-Schmidt reorthogonalization
            if reorthogonalize:
                new_q = psi_q.copy()

                for j in range(len(self.basis_Q)):
                    coeff = np.real(metric_dot(self.basis_Q[j], new_q))
                    new_q -= coeff * self.basis_Q[j]

                normq = np.sqrt(np.real(metric_dot(new_q, new_q)))
                if normq < __EPSILON__:
                    next_converged = True
                new_q /= normq

                # Hermitian L: P = Q, s_norm = c_coeff (since <P,Q> = <Q,Q> = 1)
                new_p = new_q.copy()
                s_norm = c_coeff
            else:
                # Existing biconjugate GS (for backward compatibility)
                new_q = psi_q.copy()
                new_p = psi_p.copy()

                for k_orth in range(n_rep_orth):
                    start = max(0, len(self.basis_P) - (n_ortho or len(self.basis_P)))

                    for j in range(start, len(self.basis_P)):
                        coeff1 = np.real(metric_dot(self.basis_P[j], new_q))
                        coeff2 = np.real(metric_dot(self.basis_Q[j], new_p))
                        new_q -= coeff1 * self.basis_P[j]
                        new_p -= coeff2 * self.basis_Q[j]

                    normq = np.sqrt(np.real(metric_dot(new_q, new_q)))
                    if normq < __EPSILON__:
                        next_converged = True
                    new_q /= normq

                    normp = np.real(metric_dot(new_p, new_p))
                    if np.abs(normp) < __EPSILON__:
                        next_converged = True
                    new_p /= normp

                    s_norm = c_coeff / np.real(metric_dot(new_p, new_q))

            if not converged:
                self.basis_Q.append(new_q)
                self.basis_P.append(new_p)
                psi_q = new_q.copy()
                psi_p = new_p.copy()

                self.b_coeffs.append(b_coeff)
                self.c_coeffs.append(c_coeff)
                self.s_norm.append(s_norm)

                # Drop the Krylov vectors that will never be read again.
                # Without reorthogonalization the recurrence only touches
                # [-1] and [-2], so retaining _KEEP_BASIS_OPTIMIZED vectors
                # leaves the coefficients bit-identical while the memory
                # stops growing with the step count -- the difference between
                # 25 MB and 10 GB per rank on a 12^3 fine mesh.  The
                # compatibility of `optimized` with the reorthogonalization
                # options was checked once before the loop.
                if optimized:
                    while len(self.basis_Q) > _KEEP_BASIS_OPTIMIZED:
                        self.basis_Q.pop(0)
                    while len(self.basis_P) > _KEEP_BASIS_OPTIMIZED:
                        self.basis_P.pop(0)
                    while len(self.s_norm) > _KEEP_BASIS_OPTIMIZED:
                        self.s_norm.pop(0)

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
        if band_index < 0 or band_index >= self.n_bands:
            raise ValueError("Invalid band index for perturbation: {}".format(band_index))

        self._clear_spectroscopy_symmetry()
        self.build_q_pair_map(iq)
        self.reset_q()
        self.psi[band_index] = 1.0 + 0j
        self.perturbation_modulus = 1.0

    def _prepare_gamma_cartesian_perturbation(self, vector):
        """Prepare a unit-cell Cartesian Gamma perturbation in q space."""
        n_cell = np.prod(self.dyn.GetSupercell())
        gamma_vector = np.asarray(vector).ravel() * np.sqrt(n_cell)
        self.prepare_perturbation_q(0, gamma_vector)

    def prepare_perturbation_q(self, iq, vector, add=False):
        """Prepare perturbation at q from a real-space vector (3*n_at_uc,).

        Projects the vector onto q-space eigenmodes at iq.

        Parameters
        ----------
        iq : int
            Index of the q-point.
        vector : ndarray(3*n_at_uc,)
            Perturbation vector in Cartesian real space.
        add : bool
            If true, the perturbation is added on top of the one already setup.
            Calling add does not cause a reset of the Lanczos.
        """
        self._clear_spectroscopy_symmetry()
        if not add:
            self.build_q_pair_map(iq)
            self.reset_q()
        
        m = np.tile(self.uci_structure.get_masses_array(), (3, 1)).T.ravel()
        v_scaled = vector / np.sqrt(m)
        R1 = np.conj(self.pols_q[:, :, iq]).T @ v_scaled  # (n_bands,) complex
        self.psi[:self.n_bands] += R1
        perturbation = self.psi[:self.n_bands]
        self.perturbation_modulus = np.real(
            np.conj(perturbation) @ perturbation)

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
        self._clear_spectroscopy_symmetry()

        if no_sym:
            # Identity only
            self.n_syms_qspace = 1
            self.n_syms = 1
            self._spectroscopy_symmetry_rotations = (np.eye(3),)
            n_total = self.n_q * self.n_bands
            indices = np.arange(n_total, dtype=np.int32)
            self._qspace_sym_data = ((
                indices.copy(), indices.copy(),
                np.ones(n_total, dtype=np.complex128)),)
            self._qspace_sym_q_map = np.arange(
                self.n_q, dtype=np.int32)[None, :]
            # Build identity sparse matrix
            jl = JuliaExt.get_main()
            jl.eval("""
            function init_identity_qspace(n_total::Int64)
                I_sparse = SparseArrays.sparse(
                    Int32.(1:n_total), Int32.(1:n_total),
                    ComplexF64.(ones(n_total)), n_total, n_total)
                _cached_qspace_symmetries[] = [I_sparse]
                return nothing
            end
            """)
            jl.init_identity_qspace(int(n_total))
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
        supercell_matrix = np.diag(
            np.asarray(self.dyn.GetSupercell(), dtype=float))
        inverse_supercell = np.linalg.inv(supercell_matrix)
        for i in range(len(rot_frac_all)):
            mesh_rotation = (
                inverse_supercell @ rot_frac_all[i] @ supercell_matrix)
            if not np.allclose(mesh_rotation, np.rint(mesh_rotation),
                               atol=1e-8, rtol=0):
                continue
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
        return DL.Spectroscopy.find_atom_permutation(
            structure, R_cart, t_cart, tolerance=tol)

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
        jl = JuliaExt.get_main()

        nat_uc = self.uci_structure.N_atoms
        bg = self.uci_structure.get_reciprocal_vectors() / (2 * np.pi)
        n_total = self.n_q * self.n_bands
        nb = self.n_bands

        n_syms = len(pg_indices)
        self.n_syms_qspace = n_syms
        self.n_syms = n_syms
        self._spectroscopy_symmetry_rotations = tuple(
            M @ rot_frac_all[index].astype(float) @ Minv
            for index in pg_indices)

        # Build all sparse matrices in Python, then pass to Julia
        all_rows = []
        all_cols = []
        all_vals = []
        all_q_maps = []

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
            q_map = np.empty(self.n_q, dtype=np.int32)

            for iq in range(self.n_q):
                q = self.q_points[iq]
                Rq = R_cart @ q

                # Find iq' matching Rq
                iq_prime = find_q_index(Rq, self.q_points, bg)
                q_map[iq] = iq_prime
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
            all_q_maps.append(q_map)

        # Keep the exact representation used by Julia available to Python.
        # This is required for finite-q little-group detection and is also the
        # portable representation sent by the distributed loader.
        self._qspace_sym_data = tuple(
            (rows.copy(), cols.copy(), vals.copy())
            for rows, cols, vals in zip(all_rows, all_cols, all_vals))
        self._qspace_sym_q_map = np.asarray(all_q_maps, dtype=np.int32)

        # Pass to Julia for caching (convert to 1-indexed)
        for i in range(n_syms):
            all_rows[i] += 1
            all_cols[i] += 1

        jl.eval("""
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

        jl.init_sparse_symmetries_qspace(
            all_rows, all_cols, all_vals, int(n_total))

        if verbose:
            print("Q-space symmetry matrices ({} x {}), {} symmetries cached in Julia".format(
                n_total, n_total, n_syms))

    def _apply_qspace_symmetry(self, symmetry_index, vector):
        """Apply the exact cached Bloch-mode representation in Python."""
        if self._qspace_sym_data is None:
            raise RuntimeError("Call init(use_symmetries=True) first")
        vector = np.asarray(vector, dtype=np.complex128)
        expected = self.n_q * self.n_bands
        if vector.shape != (expected,):
            raise ValueError(
                "q-space vector must have shape ({},)".format(expected))
        rows, cols, values = self._qspace_sym_data[int(symmetry_index)]
        result = np.zeros(expected, dtype=np.complex128)
        np.add.at(result, rows, values * vector[cols])
        return result

    @staticmethod
    def _line_phase(reference, candidate, tolerance):
        """Unit phase when ``candidate`` spans the line of ``reference``."""
        denominator = np.vdot(reference, reference)
        if abs(denominator) <= np.finfo(float).tiny:
            return None
        phase = np.vdot(reference, candidate) / denominator
        scale = max(np.linalg.norm(reference), np.linalg.norm(candidate),
                    np.finfo(float).tiny)
        if (abs(abs(phase) - 1.0) > tolerance or
                np.linalg.norm(candidate - phase * reference) >
                tolerance * scale):
            return None
        return phase / abs(phase)

    def configure_qspace_perturbation_symmetry(
            self, vector=None, tolerance=1e-8):
        """Configure coset reduction for the current finite-q perturbation.

        Unlike :meth:`configure_spectroscopy_symmetry`, which accepts the
        real Cartesian Gamma representation used by optical requests, this
        method detects the stabilizer directly in the complex Bloch-mode
        representation cached by :class:`QSpaceLanczos`.  It is therefore
        valid at non-time-reversal-invariant q points as well.

        Parameters
        ----------
        vector : array-like, optional
            A one-phonon vector of length ``n_bands`` at ``iq_pert``, or a
            full vector of length ``n_q * n_bands``.  The current R sector is
            used by default.
        tolerance : float
            Relative line-invariance tolerance.

        Returns
        -------
        dict
            Full order, stabilizer order, coset count, and projector phases.
        """
        if self.iq_pert is None or self.psi is None:
            raise RuntimeError(
                "Prepare a q-space perturbation before symmetry reduction")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be a positive finite number")
        if self._qspace_sym_data is None:
            raise RuntimeError("Call init(use_symmetries=True) first")

        if vector is None:
            vector = self.get_R1_q()
        vector = np.asarray(vector, dtype=np.complex128).ravel()
        full_size = self.n_q * self.n_bands
        if vector.size == self.n_bands:
            full_vector = np.zeros(full_size, dtype=np.complex128)
            start = self.iq_pert * self.n_bands
            full_vector[start:start + self.n_bands] = vector
        elif vector.size == full_size:
            full_vector = vector.copy()
        else:
            raise ValueError(
                "vector must have length n_bands or n_q * n_bands")
        if np.linalg.norm(full_vector) <= np.finfo(float).tiny:
            raise ValueError("the q-space perturbation must not be zero")

        # The point-group multiplication table and the Bloch matrices have
        # exactly the same ordering (both were built from pg_indices).
        from tdscha.Spectroscopy import SymmetryGroup
        group = SymmetryGroup.from_matrices(
            self._spectroscopy_symmetry_rotations,
            tolerance=max(float(tolerance), 1e-7))
        stabilizer = []
        eigenphases = []
        for index in range(self.n_syms_qspace):
            candidate = self._apply_qspace_symmetry(index, full_vector)
            phase = self._line_phase(full_vector, candidate, tolerance)
            if phase is not None:
                stabilizer.append(index)
                eigenphases.append(phase)

        cosets = group.right_cosets(stabilizer)
        if len(cosets) >= self.n_syms_qspace:
            self._clear_spectroscopy_symmetry()
        else:
            self._spectroscopy_coset_indices = np.asarray(
                [coset[0] + 1 for coset in cosets], dtype=np.int32)
            self._spectroscopy_stabilizer_indices = np.asarray(
                [index + 1 for index in stabilizer], dtype=np.int32)
            # P_chi = |H|^-1 sum_h conj(chi_h) D(h).
            self._spectroscopy_characters = np.asarray(
                np.conj(eigenphases), dtype=np.complex128)

        return {
            "full_group_order": int(self.n_syms_qspace),
            "stabilizer_order": len(stabilizer),
            "coset_representatives": len(cosets),
            "eigenphases": tuple(complex(value) for value in eigenphases),
        }

    def _spectroscopy_reduction_arguments(self):
        """Return complex projector characters for the q-space Julia API."""
        cosets, stabilizer, characters = super()._spectroscopy_reduction_arguments()
        return (cosets, stabilizer,
                np.asarray(characters, dtype=np.complex128))

    # Override init to use q-space symmetrization
    def init(self, use_symmetries=True):
        """Initialize the q-space Lanczos calculation."""
        self.prepare_symmetrization(no_sym=not use_symmetries)
        self.initialized = True


# =============================================================================
# Distributed Configuration Loading
# =============================================================================

# Sentinel carried by the distributed loader's metadata broadcast.  It is
# not decoration: the failure this catches is the one that made the previous
# interpolated loader unusable.  When the master enters a collective the
# workers are not in -- CellConstructor's ForceTensor broadcasts, say -- MPI
# matches the two by arrival order and the workers' ``bcast`` returns
# *something else*, with no error anywhere.  Checking the payload turns that
# into an immediate, explicit failure instead of a silently wrong spectrum.
_DISTRIBUTED_METADATA_TAG = "__tdscha_distributed_metadata__"
_DISTRIBUTED_METADATA_VERSION = 1


def _check_distributed_metadata(metadata):
    """Fail loudly unless the broadcast delivered the loader's own metadata.

    Once a collective has been mismatched the communicator is unusable and
    no rank can make progress: the master is blocked in a collective nobody
    will complete.  Waiting for a scheduler to time out is the worst of the
    available outcomes, so this reports the diagnosis and aborts the job.
    """
    if (isinstance(metadata, dict)
            and metadata.get(_DISTRIBUTED_METADATA_TAG)
            == _DISTRIBUTED_METADATA_VERSION):
        return
    message = (
        "The distributed loader received something other than its own "
        "metadata from the master.\n\n"
        "The master entered an MPI collective that the other ranks did "
        "not: MPI matched them by arrival order and delivered the wrong "
        "payload here.  The usual cause is construction work that "
        "broadcasts internally -- CellConstructor's ForceTensor does, "
        "while centering the force constants and imposing the acoustic sum "
        "rule.  Such work belongs in the Lanczos class's "
        "prepare_distributed_construction(), which every rank runs "
        "together, not in the master-only branch.\n")
    sys.stderr.write("\nTD-SCHA distributed loader: " + message)
    sys.stderr.flush()
    if __MPI4PY__ and mpi4py.MPI.COMM_WORLD.Get_size() > 1:
        mpi4py.MPI.COMM_WORLD.Abort(1)
    raise RuntimeError(message)


def _distributed_slice(rank, n_procs, N_global):
    """Contiguous [start, end) block of configurations owned by ``rank``."""
    per_proc = N_global // n_procs
    remainder = N_global % n_procs
    if rank < remainder:
        start = rank * (per_proc + 1)
        end = start + per_proc + 1
    else:
        start = rank * per_proc + remainder
        end = start + per_proc
    return start, end


def _load_distributed_build_everywhere(cls, data_dir, population_id, dyn, T,
                                       lo_to_split=None, use_symmetries=True,
                                       n_configs=None, final_dyn=None,
                                       final_T=None, **kwargs):
    """Distribute the configurations by building on every rank, then slicing.

    A diagnostic oracle for the master-builds-and-scatters path, not a
    production loader: every rank reads the whole ensemble, so peak memory is
    the replicated one and the very cost this module exists to remove is paid
    in full.  It survives because it makes no assumption at all about which
    rank computed what -- the ranks run the same deterministic code on the
    same input and are never compared -- which makes it a clean reference for
    checking that the master-only path returns the same coefficients.

    A constructor performing MPI collectives is *not* a reason to prefer this
    path; those collectives belong in
    ``QSpaceLanczos.prepare_distributed_construction``, which the master-only
    loader runs on every rank.
    """
    rank = Parallel.get_rank() if hasattr(Parallel, "get_rank") else \
        mpi4py.MPI.COMM_WORLD.Get_rank()
    n_procs = Parallel.GetNProc()

    ensemble = sscha.Ensemble.Ensemble(dyn, T)
    if n_configs is not None:
        ensemble.load_bin(data_dir, population_id, n_configs=n_configs)
    else:
        ensemble.load_bin(data_dir, population_id)
    if final_dyn is not None:
        ensemble.update_weights(final_dyn,
                                final_T if final_T is not None else T)

    qlanc = cls(ensemble, lo_to_split=lo_to_split, **kwargs)
    qlanc.init(use_symmetries=use_symmetries)
    del ensemble

    N_global = qlanc.N
    N_eff_global = float(np.sum(qlanc.rho))

    start, end = _distributed_slice(rank, n_procs, N_global)
    qlanc.X_q = qlanc.X_q[:, start:end, :].copy()
    qlanc.Y_q = qlanc.Y_q[:, start:end, :].copy()
    qlanc.rho = qlanc.rho[start:end].copy()

    qlanc._distributed = True
    qlanc._N_global = N_global
    qlanc._N_eff_global = N_eff_global
    qlanc._N_local = end - start
    qlanc.N = end - start
    # float, not int: Julia normalizes by the exact sum(rho) of this slice.
    qlanc.N_eff = float(np.sum(qlanc.rho))

    if getattr(qlanc, "X", None) is not None:
        qlanc.X = None
    if getattr(qlanc, "Y", None) is not None:
        qlanc.Y = None

    return qlanc


def load_distributed_tdscha(data_dir, population_id, dyn, T, lo_to_split=None,
                           use_symmetries=True, n_configs=None,
                           final_dyn=None, final_T=None,
                           lanczos_class=None, build_on_all_ranks=False,
                           **kwargs):
    """Load QSpaceLanczos with distributed configurations across MPI ranks.

    Loads the ensemble on master rank only, then distributes configuration data
    across all ranks. Each process stores only N/n_procs configs instead of all N.
    This avoids having the full ensemble replicated in memory on all ranks.

    Parameters
    ----------
    data_dir : str
        Directory containing the ensemble data files.
    population_id : int
        Population ID of the ensemble to load.
    dyn : CC.Phonons.Phonons
        Dynamical matrix object (used to create the ensemble).
    T : float
        Temperature in Kelvin.
    lo_to_split : str, ndarray, or None
        LO-TO splitting mode.
    use_symmetries : bool
        If True, use q-space symmetries.
    n_configs : int or None
        Number of configs to load. If None, loads all available.
    final_dyn : CC.Phonons.Phonons, optional
        Final dynamical matrix from the SSCHA calculation. If provided,
        the ensemble weights are updated using update_weights(final_dyn, final_T).
        This is recommended for production calculations where the ensemble
        was generated with a preliminary dynamical matrix.
    final_T : float, optional
        Temperature for weight updates. Defaults to T if not specified.
        Use this if the final temperature differs from the ensemble temperature.
    lanczos_class : type, optional
        The QSpaceLanczos subclass to build. Defaults to QSpaceLanczos. Pass
        ``QSpaceAtomFourierLanczos`` (or use
        ``QSpaceAtomFourier.load_distributed_atom_fourier_tdscha``) to
        distribute an *interpolated* calculation; the subclass declares the
        extra state to broadcast through ``_DISTRIBUTED_EXTRA_ATTRS`` and the
        collective part of its construction through
        ``prepare_distributed_construction``.
    build_on_all_ranks : bool
        Diagnostic path.  Every rank loads the whole ensemble, builds the
        whole object, and only then drops the configurations it does not
        own.  It replicates the ensemble during construction, which is
        exactly what this loader exists to avoid, and it is kept only as an
        oracle the master-only path can be compared against on small
        systems.  Leave it False for production.
    **kwargs
        Additional arguments passed to the Lanczos class (e.g. ``fine_mesh``
        and ``ignore_effective_charges`` for interpolation).

    Returns
    -------
    QSpaceLanczos
        QSpaceLanczos with distributed configs. Each rank has N_local = N/n_procs configs.
        With n_procs=1, rank 0 holds all configs (N_local = N).

    Usage
    -----
    mpirun -np 8 python your_script.py

    Flow:
    - All ranks: run ``cls.prepare_distributed_construction`` together, so
                 that any MPI collective inside the construction (the
                 harmonic interpolation of the interpolated subclasses) is
                 matched across ranks before the master goes on alone
    - Rank 0: loads ensemble, optionally updates weights, creates QSpaceLanczos,
              broadcasts metadata, sends slices
    - Ranks 1..n-1: receive metadata, build bare QSpaceLanczos, receive slices
    - All: have only their local slice, _distributed=True
    """
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = Parallel.GetNProc()

    cls = QSpaceLanczos if lanczos_class is None else lanczos_class

    if build_on_all_ranks:
        return _load_distributed_build_everywhere(
            cls, data_dir, population_id, dyn, T, lo_to_split=lo_to_split,
            use_symmetries=use_symmetries, n_configs=n_configs,
            final_dyn=final_dyn, final_T=final_T, **kwargs)

    # Collective, on every rank: the object the master is about to build
    # lives on the ensemble's current_dyn, which is final_dyn whenever the
    # weights are updated.
    kwargs = dict(kwargs)
    kwargs.update(cls.prepare_distributed_construction(
        dyn if final_dyn is None else final_dyn,
        lo_to_split=lo_to_split, **kwargs))

    if Parallel.am_i_the_master():
        # ========== MASTER (RANK 0) ==========
        ensemble = sscha.Ensemble.Ensemble(dyn, T)
        if n_configs is not None:
            ensemble.load_bin(data_dir, population_id, n_configs=n_configs)
        else:
            ensemble.load_bin(data_dir, population_id)

        # Update weights if final_dyn is provided
        if final_dyn is not None:
            T_for_update = final_T if final_T is not None else T
            ensemble.update_weights(final_dyn, T_for_update)

        qlanc = cls(ensemble, lo_to_split=lo_to_split, **kwargs)
        qlanc.init(use_symmetries=use_symmetries)

        # Free ensemble - we only need QSpaceLanczos arrays
        del ensemble

        # Extract global info
        N_global = qlanc.N
        N_eff_global = qlanc.N_eff

        # Broadcast metadata (structure arrays, NO config data)
        metadata = {
            _DISTRIBUTED_METADATA_TAG: _DISTRIBUTED_METADATA_VERSION,
            'T': qlanc.T, 'dyn': qlanc.dyn,
            'uci_structure': qlanc.uci_structure,
            'super_structure': qlanc.super_structure,
            'q_points': qlanc.q_points, 'n_q': qlanc.n_q,
            'n_bands': qlanc.n_bands, 'w_q': qlanc.w_q,
            'pols_q': qlanc.pols_q, 'valid_modes_q': qlanc.valid_modes_q,
            'm': qlanc.m,
            'ignore_v3': qlanc.ignore_v3, 'ignore_v4': qlanc.ignore_v4,
            'N_global': N_global, 'N_eff_global': N_eff_global,
            'n_syms_qspace': qlanc.n_syms_qspace,
            '_qspace_sym_data': qlanc._qspace_sym_data,
            '_qspace_sym_q_map': qlanc._qspace_sym_q_map,
            # The ensemble Bloch fields are NOT necessarily indexed by n_q:
            # the interpolated subclass keeps X_q/Y_q on the coarse mesh while
            # n_q counts the fine one. Send the real leading dimension so the
            # workers allocate receive buffers that match what is sent.
            'xq_nq': qlanc.X_q.shape[0],
        }
        # Whatever extra structure the subclass needs to be functional.
        for attr in cls._DISTRIBUTED_EXTRA_ATTRS:
            metadata[attr] = getattr(qlanc, attr, None)
        comm.bcast(metadata, root=0)

        # Barrier to ensure all ranks have received metadata before we start sending slices
        comm.barrier()

        # Distribute configs to other ranks
        configs_per_proc = N_global // n_procs
        remainder = N_global % n_procs

        for target in range(1, n_procs):
            if target < remainder:
                start = target * (configs_per_proc + 1)
                end = start + configs_per_proc + 1
            else:
                start = target * configs_per_proc + remainder
                end = start + configs_per_proc

            N_local = end - start
            comm.Send(np.array([N_local], dtype=np.int64), dest=target, tag=0)
            comm.Send(qlanc.X_q[:, start:end, :].copy(), dest=target, tag=1)
            comm.Send(qlanc.Y_q[:, start:end, :].copy(), dest=target, tag=2)
            comm.Send(qlanc.rho[start:end].copy(), dest=target, tag=3)

        # Master keeps its slice (rank 0)
        if remainder > 0:
            start, end = 0, configs_per_proc + 1
        else:
            start, end = 0, configs_per_proc

        N_local = end - start
        qlanc.X_q = qlanc.X_q[:, start:end, :].copy()
        qlanc.Y_q = qlanc.Y_q[:, start:end, :].copy()
        qlanc.rho = qlanc.rho[start:end].copy()

        # Set distributed state
        qlanc._distributed = True
        qlanc._N_global = N_global
        qlanc._N_eff_global = N_eff_global
        qlanc._N_local = N_local
        qlanc.N = N_local
        # float, not int: Julia normalizes by the exact sum(rho) of this
        # rank's slice, so truncating here would leave a systematic
        # mis-normalization on any reweighted ensemble (rho != 1).
        qlanc.N_eff = float(np.sum(qlanc.rho))

        # Free unused arrays
        if hasattr(qlanc, 'X') and qlanc.X is not None:
            qlanc.X = None
        if hasattr(qlanc, 'Y') and qlanc.Y is not None:
            qlanc.Y = None

        return qlanc

    else:
        # ========== OTHER RANKS ==========
        metadata = comm.bcast(None, root=0)
        _check_distributed_metadata(metadata)

        # Barrier to ensure all ranks have received metadata before slices are sent
        comm.barrier()

        # Create bare Lanczos object and populate from metadata
        qlanc = cls(ensemble=None, lo_to_split=lo_to_split, **kwargs)
        qlanc.T = metadata['T']
        qlanc.dyn = metadata['dyn']
        qlanc.uci_structure = metadata['uci_structure']
        qlanc.super_structure = metadata['super_structure']
        qlanc.q_points = metadata['q_points']
        qlanc.n_q = metadata['n_q']
        qlanc.n_bands = metadata['n_bands']
        qlanc.w_q = metadata['w_q']
        qlanc.pols_q = metadata['pols_q']
        qlanc.valid_modes_q = metadata['valid_modes_q']
        qlanc.m = metadata['m']
        qlanc.ignore_v3 = metadata['ignore_v3']
        qlanc.ignore_v4 = metadata['ignore_v4']
        qlanc.n_syms_qspace = metadata['n_syms_qspace']
        qlanc._qspace_sym_data = metadata['_qspace_sym_data']
        qlanc._qspace_sym_q_map = metadata['_qspace_sym_q_map']
        for attr in cls._DISTRIBUTED_EXTRA_ATTRS:
            setattr(qlanc, attr, metadata[attr])

        # Receive local config slice
        N_local_arr = np.array([0], dtype=np.int64)
        comm.Recv(N_local_arr, source=0, tag=0)
        N_local = int(N_local_arr[0])

        xq_nq = metadata['xq_nq']
        qlanc.X_q = np.zeros((xq_nq, N_local, qlanc.n_bands), dtype=np.complex128)
        qlanc.Y_q = np.zeros((xq_nq, N_local, qlanc.n_bands), dtype=np.complex128)
        qlanc.rho = np.zeros(N_local, dtype=np.float64)

        comm.Recv(qlanc.X_q, source=0, tag=1)
        comm.Recv(qlanc.Y_q, source=0, tag=2)
        comm.Recv(qlanc.rho, source=0, tag=3)

        # Set distributed state
        qlanc._distributed = True
        qlanc._N_global = metadata['N_global']
        qlanc._N_eff_global = metadata['N_eff_global']
        qlanc._N_local = N_local
        qlanc.N = N_local
        # float, not int: Julia normalizes by the exact sum(rho) of this
        # rank's slice, so truncating here would leave a systematic
        # mis-normalization on any reweighted ensemble (rho != 1).
        qlanc.N_eff = float(np.sum(qlanc.rho))

        # Build Julia symmetry cache
        qlanc.prepare_symmetrization(no_sym=not use_symmetries)
        qlanc.initialized = True

        return qlanc
