"""
Q-Space Free Energy Hessian
============================

Computes the free energy Hessian d²F/dRdR in q-space (Bloch basis).

Instead of building the full D4 matrix explicitly (O(N⁴) memory, O(N⁶) time),
solves L_static(q) x = e_i for each band at each irreducible q-point, where
L_static is the static Liouvillian operator. The Hessian is then
H(q) = inv(G(q)), where G(q) is the static susceptibility.

The static L operator is DIFFERENT from the spectral L used in Lanczos:
  - Static:   R sector = +w²,  W sector = 1/Lambda (one 2-phonon sector)
  - Spectral: R sector = -w²,  a'/b' sectors = -(w1∓w2)² (two sectors)

Both share the same anharmonic core (ensemble averages of D3/D4).

The q-space block-diagonal structure gives a speedup of ~N_cell³ / N_q_irr
over the real-space approach.

References:
    Monacelli & Mauri 2021 (Phys. Rev. B)
"""

from __future__ import print_function, division

import sys
import time
import warnings
import numpy as np
import scipy.sparse.linalg

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Methods

import cellconstructor.Settings as Parallel
from cellconstructor.Settings import ParallelPrint as print

try:
    import spglib
    __SPGLIB__ = True
except ImportError:
    __SPGLIB__ = False

# Constants
__EPSILON__ = 1e-12
__RyToK__ = 157887.32400374097


class QSpaceHessian:
    """Compute the free energy Hessian in q-space via iterative linear solves.

    Uses the static Liouvillian operator L_static, which has the structure:
      - R sector: w² * R  (positive, unlike spectral -w²)
      - W sector: (1/Lambda) * W  (static 2-phonon propagator)
      - Anharmonic coupling via ensemble averages (same Julia core)

    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble
        The SSCHA ensemble.
    verbose : bool
        If True, print progress information.
    **kwargs
        Additional keyword arguments passed to QSpaceLanczos.
    """

    def __init__(self, ensemble, verbose=True, **kwargs):
        from tdscha.QSpaceLanczos import QSpaceLanczos

        self.qlanc = QSpaceLanczos(ensemble, **kwargs)
        self.ensemble = ensemble
        self.verbose = verbose

        # Shortcuts
        self.n_q = self.qlanc.n_q
        self.n_bands = self.qlanc.n_bands
        self.q_points = self.qlanc.q_points
        self.w_q = self.qlanc.w_q
        self.pols_q = self.qlanc.pols_q

        # Results storage: iq -> H_q matrix (n_bands x n_bands)
        self.H_q_dict = {}

        # Irreducible q-point data (set by init)
        self.irr_qpoints = None
        self.q_star_map = None

        # Static psi layout (set by _compute_static_block_layout)
        self._static_psi_size = None
        self._static_block_offsets = None
        self._static_block_sizes = None

    def init(self, use_symmetries=True):
        """Initialize the Lanczos engine and find irreducible q-points.

        Parameters
        ----------
        use_symmetries : bool
            If True, use symmetries to reduce q-points.
        """
        self.qlanc.init(use_symmetries=use_symmetries)
        if use_symmetries and __SPGLIB__:
            self._find_irreducible_qpoints()
        else:
            self.irr_qpoints = list(range(self.n_q))
            self.q_star_map = {iq: [(iq, np.eye(3), np.zeros(3))]
                               for iq in range(self.n_q)}

    def _find_irreducible_qpoints(self):
        """Find irreducible q-points under point-group symmetries."""
        from tdscha.QSpaceLanczos import find_q_index

        uci = self.qlanc.uci_structure
        spg_data = spglib.get_symmetry(uci.get_spglib_cell())
        rot_frac_all = spg_data['rotations']
        trans_frac_all = spg_data['translations']

        M = uci.unit_cell.T
        Minv = np.linalg.inv(M)

        unique_pg = {}
        for i in range(len(rot_frac_all)):
            key = rot_frac_all[i].tobytes()
            if key not in unique_pg:
                unique_pg[key] = i
        pg_indices = list(unique_pg.values())

        bg = uci.get_reciprocal_vectors() / (2 * np.pi)

        visited = set()
        self.irr_qpoints = []
        self.q_star_map = {}

        for iq in range(self.n_q):
            if iq in visited:
                continue

            self.irr_qpoints.append(iq)
            star = []

            for idx in pg_indices:
                R_frac = rot_frac_all[idx].astype(float)
                t_frac = trans_frac_all[idx]
                R_cart = M @ R_frac @ Minv
                t_cart = M @ t_frac

                Rq = R_cart @ self.q_points[iq]
                try:
                    iq_rot = find_q_index(Rq, self.q_points, bg)
                except ValueError:
                    continue

                if iq_rot not in visited:
                    visited.add(iq_rot)
                star.append((iq_rot, R_cart.copy(), t_cart.copy()))

            seen_iqs = set()
            unique_star = []
            for entry in star:
                if entry[0] not in seen_iqs:
                    seen_iqs.add(entry[0])
                    unique_star.append(entry)

            self.q_star_map[iq] = unique_star

        if self.verbose:
            print("Q-space Hessian: {} irreducible q-points out of {}".format(
                len(self.irr_qpoints), self.n_q))

    def _build_D_matrix(self, R_cart, t_cart, iq_from, iq_to):
        """Compute the mode representation matrix D for symmetry {R|t}."""
        from tdscha.QSpaceLanczos import QSpaceLanczos

        uci = self.qlanc.uci_structure
        nat_uc = uci.N_atoms
        M = uci.unit_cell.T
        Minv = np.linalg.inv(M)

        irt = QSpaceLanczos._get_atom_perm(uci, R_cart, t_cart, M, Minv)

        q_to = self.q_points[iq_to]
        P_uc = np.zeros((3 * nat_uc, 3 * nat_uc), dtype=np.complex128)
        for kappa in range(nat_uc):
            kp = irt[kappa]
            tau_k = uci.coords[kappa]
            tau_kp = uci.coords[kp]
            L = R_cart @ tau_k + t_cart - tau_kp
            phase = np.exp(-2j * np.pi * q_to @ L)
            P_uc[3 * kp:3 * kp + 3, 3 * kappa:3 * kappa + 3] = phase * R_cart

        D = np.conj(self.pols_q[:, :, iq_to]).T @ P_uc @ self.pols_q[:, :, iq_from]
        return D

    # ==================================================================
    # Static psi layout: (R[nb], W[blocks])
    # Only ONE 2-phonon sector (unlike spectral a'/b')
    # ==================================================================
    def _compute_static_block_layout(self):
        """Pre-compute block layout for the static psi vector.

        Layout: [R sector (nb)] [W blocks (one per unique pair)]
        """
        nb = self.n_bands
        sizes = []
        for iq1, iq2 in self.qlanc.unique_pairs:
            if iq1 < iq2:
                sizes.append(nb * nb)
            else:
                sizes.append(nb * (nb + 1) // 2)
        self._static_block_sizes = sizes

        offsets = []
        offset = nb  # R sector first
        for s in sizes:
            offsets.append(offset)
            offset += s
        self._static_block_offsets = offsets
        self._static_psi_size = offset

    def _get_static_psi_size(self):
        return self._static_psi_size

    def _get_W_block(self, psi, pair_idx):
        """Extract full (nb, nb) W block from static psi."""
        nb = self.n_bands
        iq1, iq2 = self.qlanc.unique_pairs[pair_idx]
        offset = self._static_block_offsets[pair_idx]
        size = self._static_block_sizes[pair_idx]
        raw = psi[offset:offset + size]
        if iq1 < iq2:
            return raw.reshape(nb, nb).copy()
        else:
            return self.qlanc._unpack_upper_triangle(raw, nb)

    def _set_W_block(self, psi, pair_idx, matrix):
        """Write full (nb, nb) matrix into W sector of static psi."""
        nb = self.n_bands
        iq1, iq2 = self.qlanc.unique_pairs[pair_idx]
        offset = self._static_block_offsets[pair_idx]
        if iq1 < iq2:
            psi[offset:offset + nb * nb] = matrix.ravel()
        else:
            size = self._static_block_sizes[pair_idx]
            psi[offset:offset + size] = \
                self.qlanc._pack_upper_triangle(matrix, nb)

    def _static_mask(self):
        """Build mask for static psi inner product.

        Same structure as FT mask but only one W sector (not a' + b').
        """
        psi_size = self._get_static_psi_size()
        mask = np.ones(psi_size, dtype=np.float64)
        nb = self.n_bands

        for pair_idx, (iq1, iq2) in enumerate(self.qlanc.unique_pairs):
            offset = self._static_block_offsets[pair_idx]
            size = self._static_block_sizes[pair_idx]

            if iq1 < iq2:
                mask[offset:offset + size] = 2
            else:
                idx = offset
                for i in range(nb):
                    mask[idx] = 1  # diagonal
                    idx += 1
                    for j in range(i + 1, nb):
                        mask[idx] = 2  # off-diagonal
                        idx += 1

        return mask

    # ==================================================================
    # Lambda: static 2-phonon propagator
    # ==================================================================
    def _compute_lambda_q(self, iq1, iq2):
        """Compute the static 2-phonon propagator Lambda for pair (iq1, iq2).

        Lambda[nu1, nu2] = ((n1+n2+1)/(w1+w2) - dn12/dw) / (4*w1*w2)

        where dn12/dw = (n1-n2)/(w1-w2) for w1 != w2,
                       = -beta * exp(beta*w) * n^2 for w1 == w2.

        Returns
        -------
        Lambda : ndarray(n_bands, n_bands), float64
        """
        w1 = self.w_q[:, iq1]
        w2 = self.w_q[:, iq2]
        T = self.qlanc.T

        w1_mat = np.tile(w1, (self.n_bands, 1)).T
        w2_mat = np.tile(w2, (self.n_bands, 1))

        n1 = np.zeros_like(w1)
        n2 = np.zeros_like(w2)
        if T > __EPSILON__:
            beta = __RyToK__ / T
            valid1 = w1 > self.qlanc.acoustic_eps
            valid2 = w2 > self.qlanc.acoustic_eps
            n1[valid1] = 1.0 / (np.exp(w1[valid1] * beta) - 1.0)
            n2[valid2] = 1.0 / (np.exp(w2[valid2] * beta) - 1.0)

        n1_mat = np.tile(n1, (self.n_bands, 1)).T
        n2_mat = np.tile(n2, (self.n_bands, 1))

        valid_mask = np.outer(w1 > self.qlanc.acoustic_eps,
                              w2 > self.qlanc.acoustic_eps)

        # (n1 - n2) / (w1 - w2), regularized for w1 ≈ w2
        diff_n = np.zeros_like(w1_mat)
        w_diff = w1_mat - w2_mat
        w_equal = np.abs(w_diff) < 1e-8

        if T > __EPSILON__:
            beta = __RyToK__ / T
            # Normal case: w1 != w2
            safe_diff = np.where(w_equal, 1.0, w_diff)
            diff_n = np.where(w_equal, 0.0, (n1_mat - n2_mat) / safe_diff)
            # Degenerate case: w1 == w2
            w_eq_vals = np.where(w_equal & valid_mask, w1_mat, 0.0)
            n_eq_vals = np.where(w_equal & valid_mask, n1_mat, 0.0)
            diff_n_deg = np.where(
                w_equal & valid_mask,
                -beta * np.exp(w_eq_vals * beta) * n_eq_vals ** 2,
                0.0)
            diff_n = np.where(w_equal, diff_n_deg, diff_n)

        # Lambda = ((n1+n2+1)/(w1+w2) - diff_n) / (4*w1*w2)
        w_sum = w1_mat + w2_mat
        safe_wsum = np.where(valid_mask, w_sum, 1.0)
        safe_wprod = np.where(valid_mask, w1_mat * w2_mat, 1.0)

        Lambda = np.where(
            valid_mask,
            ((n1_mat + n2_mat + 1) / safe_wsum - diff_n) / (4 * safe_wprod),
            0.0)

        return Lambda

    # ==================================================================
    # Static L operator
    # ==================================================================
    def _apply_L_static_q(self, psi):
        """Apply the static Liouvillian L_static to a psi vector.

        L_static has:
          R sector: w² * R
          W sector: (1/Lambda) * W
          Anharmonic coupling from ensemble averages

        Parameters
        ----------
        psi : ndarray(static_psi_size,), complex128

        Returns
        -------
        out : ndarray(static_psi_size,), complex128
        """
        nb = self.n_bands
        out = np.zeros_like(psi)

        # --- Harmonic part ---
        # R sector: w² * R
        w_qp = self.w_q[:, self.qlanc.iq_pert]
        out[:nb] = (w_qp ** 2) * psi[:nb]

        # W sector: (1/Lambda) * W
        for pair_idx, (iq1, iq2) in enumerate(self.qlanc.unique_pairs):
            Lambda = self._compute_lambda_q(iq1, iq2)
            W_block = self._get_W_block(psi, pair_idx)

            # 1/Lambda * W, with Lambda=0 for acoustic modes → set to 0
            safe_Lambda = np.where(np.abs(Lambda) > 1e-30, Lambda, 1.0)
            inv_Lambda_W = np.where(
                np.abs(Lambda) > 1e-30,
                W_block / safe_Lambda,
                0.0)

            self._set_W_block(out, pair_idx, inv_Lambda_W)

        # --- Anharmonic part ---
        anh_out = self._apply_anharmonic_static_q(psi)
        out += anh_out

        return out

    def _apply_anharmonic_static_q(self, psi):
        """Apply the anharmonic part of the static L operator.

        1. Extract R1 from R sector
        2. Transform W blocks → Y1 (Upsilon perturbation)
        3. Call Julia q-space extension
        4. R output = -f_pert, W output = d2v blocks

        Parameters
        ----------
        psi : ndarray(static_psi_size,), complex128

        Returns
        -------
        out : ndarray(static_psi_size,), complex128
        """
        nb = self.n_bands
        out = np.zeros_like(psi)

        R1 = psi[:nb].copy()

        # Build Y1 blocks from W: Y1 = -2 * Y_w1 * Y_w2 * W
        # Y_w[nu] = 2*w/(2*n+1) for each q-point
        Y1_blocks = []
        for pair_idx, (iq1, iq2) in enumerate(self.qlanc.unique_pairs):
            W_block = self._get_W_block(psi, pair_idx)

            Y_w1 = self._compute_Y_w(iq1)  # (nb,)
            Y_w2 = self._compute_Y_w(iq2)  # (nb,)

            Y1_block = -2.0 * np.outer(Y_w1, Y_w2) * W_block
            Y1_blocks.append(Y1_block)

        Y1_flat = self.qlanc._flatten_blocks(Y1_blocks)

        # Call Julia via the same interface as the spectral case
        f_pert, d2v_blocks = self.qlanc._call_julia_qspace(R1, Y1_flat)

        # Output: R ← -f_pert (note the sign!)
        out[:nb] = -f_pert

        # Output: W ← d2v (direct, no chi± transformation)
        for pair_idx in range(len(self.qlanc.unique_pairs)):
            self._set_W_block(out, pair_idx, d2v_blocks[pair_idx])

        return out

    def _compute_Y_w(self, iq):
        """Compute Y_w = 2*w/(2*n+1) for all bands at q-point iq.

        Acoustic modes (w < eps) get Y_w = 0.
        """
        w = self.w_q[:, iq]
        T = self.qlanc.T

        n = np.zeros_like(w)
        if T > __EPSILON__:
            valid = w > self.qlanc.acoustic_eps
            n[valid] = 1.0 / (np.exp(w[valid] * __RyToK__ / T) - 1.0)

        Y_w = np.zeros_like(w)
        valid = w > self.qlanc.acoustic_eps
        Y_w[valid] = 2.0 * w[valid] / (2.0 * n[valid] + 1.0)
        return Y_w

    # ==================================================================
    # Preconditioner: inverse of harmonic L_static
    # ==================================================================
    def _apply_harmonic_preconditioner(self, psi_tilde, sqrt_mask,
                                       inv_sqrt_mask):
        """Apply harmonic preconditioner M = L_harm^{-1} in transformed basis.

        For L_static_harm:
          R sector: eigenvalue = w², M = 1/w²
          W sector: eigenvalue = 1/Lambda, M = Lambda
        """
        nb = self.n_bands
        psi = psi_tilde * inv_sqrt_mask
        result = np.zeros_like(psi)

        # R sector: 1/w²
        w_qp = self.w_q[:, self.qlanc.iq_pert]
        for nu in range(nb):
            if w_qp[nu] > self.qlanc.acoustic_eps:
                result[nu] = psi[nu] / (w_qp[nu] ** 2)

        # W sector: Lambda (inverse of 1/Lambda)
        for pair_idx, (iq1, iq2) in enumerate(self.qlanc.unique_pairs):
            Lambda = self._compute_lambda_q(iq1, iq2)
            W_block = self._get_W_block(psi, pair_idx)
            result_block = Lambda * W_block
            self._set_W_block(result, pair_idx, result_block)

        return result * sqrt_mask

    # ==================================================================
    # Core solver
    # ==================================================================
    def compute_hessian_at_q(self, iq, tol=1e-6, max_iters=500,
                             use_preconditioner=True):
        """Compute the free energy Hessian at a single q-point.

        Solves L_static(q) x_i = e_i for each non-acoustic band.
        G_q[j,i] = x_i[j] (R-sector), H_q = inv(G_q).

        Parameters
        ----------
        iq : int
            Q-point index.
        tol : float
            Convergence tolerance for the iterative solver.
        max_iters : int
            Maximum number of iterations.
        use_preconditioner : bool
            If True, use harmonic preconditioner.

        Returns
        -------
        H_q : ndarray(n_bands, n_bands), complex128
            The Hessian matrix in the mode basis at q-point iq.
        """
        nb = self.n_bands

        # 1. Setup pair map and block layout
        self.qlanc.build_q_pair_map(iq)
        self._compute_static_block_layout()
        psi_size = self._get_static_psi_size()
        mask = self._static_mask()

        # Similarity transform arrays
        sqrt_mask = np.sqrt(mask)
        inv_sqrt_mask = np.zeros_like(sqrt_mask)
        nonzero = mask > 0
        inv_sqrt_mask[nonzero] = 1.0 / sqrt_mask[nonzero]

        # 2. Define transformed operator: L_tilde = D^{1/2} L_static D^{-1/2}
        def apply_L_tilde(x_tilde):
            x = x_tilde * inv_sqrt_mask
            Lx = self._apply_L_static_q(x)
            return Lx * sqrt_mask

        L_op = scipy.sparse.linalg.LinearOperator(
            (psi_size, psi_size), matvec=apply_L_tilde, dtype=np.complex128)

        # 3. Preconditioner
        M_op = None
        if use_preconditioner:
            def apply_M_tilde(x_tilde):
                return self._apply_harmonic_preconditioner(
                    x_tilde, sqrt_mask, inv_sqrt_mask)
            M_op = scipy.sparse.linalg.LinearOperator(
                (psi_size, psi_size), matvec=apply_M_tilde,
                dtype=np.complex128)

        # 4. Identify non-acoustic bands
        w_qp = self.w_q[:, iq]
        non_acoustic = [nu for nu in range(nb)
                        if w_qp[nu] > self.qlanc.acoustic_eps]

        if self.verbose:
            print("  Solving at q={} ({} non-acoustic bands out of {})".format(
                iq, len(non_acoustic), nb))

        # 5. Solve L_static x_i = e_i for each non-acoustic band
        G_q = np.zeros((nb, nb), dtype=np.complex128)
        total_iters = 0

        for band_i in non_acoustic:
            rhs = np.zeros(psi_size, dtype=np.complex128)
            rhs[band_i] = 1.0
            rhs_tilde = rhs * sqrt_mask

            # Initial guess from preconditioner
            x0 = None
            if use_preconditioner:
                x0 = apply_M_tilde(rhs_tilde)

            n_iters = [0]
            def _count(xk):
                n_iters[0] += 1

            t1 = time.time()

            # Use GMRES since L_static can be indefinite for unstable systems
            x_tilde, info = scipy.sparse.linalg.gmres(
                L_op, rhs_tilde, x0=x0, rtol=tol, maxiter=max_iters,
                M=M_op, callback=_count, callback_type='legacy')

            if info != 0:
                if self.verbose:
                    print("    GMRES did not converge for band {} (info={}), "
                          "trying BiCGSTAB...".format(band_i, info))
                x_tilde, info = scipy.sparse.linalg.bicgstab(
                    L_op, rhs_tilde, x0=x_tilde, rtol=tol, maxiter=max_iters,
                    M=M_op)
                if info != 0 and self.verbose:
                    print("    WARNING: BiCGSTAB also did not converge "
                          "for band {} (info={})".format(band_i, info))

            t2 = time.time()
            total_iters += n_iters[0]

            # Un-transform
            x = x_tilde * inv_sqrt_mask

            # Extract R-sector
            G_q[:, band_i] = x[:nb]

            if self.verbose:
                print("    Band {}: {} iters, {:.2f}s".format(
                    band_i, n_iters[0], t2 - t1))

        # 6. Symmetrize G_q (should be Hermitian)
        G_q = (G_q + G_q.conj().T) / 2

        # 7. Invert to get H_q
        H_q = np.zeros((nb, nb), dtype=np.complex128)
        if len(non_acoustic) > 0:
            na = np.array(non_acoustic)
            G_sub = G_q[np.ix_(na, na)]

            cond = np.linalg.cond(G_sub)
            if cond > 1e12 and self.verbose:
                print("    WARNING: G_q ill-conditioned (cond={:.2e})".format(
                    cond))

            H_sub = np.linalg.inv(G_sub)
            H_q[np.ix_(na, na)] = H_sub

        H_q = (H_q + H_q.conj().T) / 2

        if self.verbose:
            eigvals = np.sort(np.real(np.linalg.eigvalsh(H_q)))
            non_zero = eigvals[np.abs(eigvals) > 1e-10]
            if len(non_zero) > 0:
                print("  H_q eigenvalue range: [{:.6e}, {:.6e}]".format(
                    non_zero[0], non_zero[-1]))
            print("  Total L applications: {}".format(total_iters))

        self.H_q_dict[iq] = H_q
        return H_q

    def compute_full_hessian(self, tol=1e-6, max_iters=500,
                             use_preconditioner=True):
        """Compute the Hessian at all q-points and return as CC.Phonons.

        Parameters
        ----------
        tol : float
            Convergence tolerance for iterative solver.
        max_iters : int
            Maximum iterations per linear solve.
        use_preconditioner : bool
            If True, use harmonic preconditioner.

        Returns
        -------
        hessian : CC.Phonons.Phonons
            The free energy Hessian as a Phonons object (Ry/bohr²).
        """
        if self.verbose:
            print()
            print("=" * 50)
            print("  Q-SPACE FREE ENERGY HESSIAN")
            print("=" * 50)
            print()

        t_start = time.time()

        for iq_irr in self.irr_qpoints:
            if self.verbose:
                print("Irreducible q-point {} / {}".format(
                    self.irr_qpoints.index(iq_irr) + 1,
                    len(self.irr_qpoints)))
            self.compute_hessian_at_q(
                iq_irr, tol=tol, max_iters=max_iters,
                use_preconditioner=use_preconditioner)

        # Unfold to full BZ
        for iq_irr in self.irr_qpoints:
            H_irr = self.H_q_dict[iq_irr]
            for iq_rot, R_cart, t_cart in self.q_star_map[iq_irr]:
                if iq_rot == iq_irr:
                    continue
                D = self._build_D_matrix(R_cart, t_cart, iq_irr, iq_rot)
                self.H_q_dict[iq_rot] = D @ H_irr @ D.conj().T

        t_end = time.time()
        if self.verbose:
            print()
            print("Total time: {:.1f}s".format(t_end - t_start))

        return self._build_phonons()

    def _build_phonons(self):
        """Convert mode-basis H_q to Cartesian and create CC.Phonons.Phonons."""
        nq = self.n_q
        q_points = np.array(self.qlanc.dyn.q_tot)
        uc_structure = self.qlanc.dyn.structure.copy()

        hessian = CC.Phonons.Phonons(uc_structure, nqirr=nq)

        for iq in range(nq):
            H_q = self.H_q_dict[iq]
            pol = self.pols_q[:, :, iq]
            pol_m = np.einsum("a, ab -> ab", np.sqrt(self.qlanc.m), pol)
            Phi_q = pol_m @ H_q @ np.conj(pol_m).T
            hessian.dynmats[iq] = Phi_q
            hessian.q_tot[iq] = q_points[iq]

        hessian.AdjustQStar()
        return hessian
