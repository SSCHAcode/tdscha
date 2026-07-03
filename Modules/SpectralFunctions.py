"""
Q-Point Spectral Function Module
=================================

This module provides the :class:`QPointSpectralFunction` class for computing
anharmonic phonon spectral functions at a selected q-point, exploiting
symmetry reduction and the Q-space Lanczos algorithm.

Features
--------
- Identifies and computes only symmetry-independent phonon modes
- Stores Lanczos ``a, b, c`` coefficients, SSCHA auxiliary frequencies,
  and polarization vectors for each mode
- Caches the dielectric tensor, Raman tensor and effective charges
  (Born charges) when available
- Unpolarized Raman spectrum with powder averaging via Lebedev
  quadrature, including LO-TO splitting through the non-analytic
  dynamical matrix (``symph.nonanal``)

Efficiency notes
----------------
- Lanczos is run once per **symmetry-independent** mode. Degenerate
  modes (same irreducible representation) share coefficients.
- The LO-TO correction is applied as a scalar post-processing step on
  the continued-fraction Green function, avoiding re-runs of the
  expensive Lanczos iteration for each Lebedev direction.
- For broadband spectra the :class:`QSpaceKPM` may be preferable;
  this class is optimised for high-resolution, mode-resolved analysis
  at a single q-point.
"""

from __future__ import print_function
from __future__ import division

import os
import sys
import warnings
import numpy as np

# -- cellconstructor -----------------------------------------------------------
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries
import cellconstructor.Methods
import cellconstructor.Units
import symph  # Fortran extension (always available with cellconstructor)

# -- TD-SCHA modules ----------------------------------------------------------
import tdscha.DynamicalLanczos as DL
import tdscha.QSpaceLanczos as QL

# -- SSCHA --------------------------------------------------------------------
import sscha.Ensemble

# -- SciPy (Lebedev quadrature) -----------------------------------------------
from scipy.integrate import lebedev_rule

__all__ = ["QPointSpectralFunction"]


# =============================================================================
#  Standalone continued-fraction Green function
# =============================================================================

def _gf_from_abc(a_coeffs, b_coeffs, c_coeffs, w_array,
                 perturbation_modulus=1.0, use_terminator=True,
                 last_average=1, smearing=0.0, smooth_ramp=0,
                 shift_value=0.0, verbose=False):
    r"""Compute the Wigner Green function from Lanczos coefficients.

    In Wigner formalism we invert :math:`(L + \omega^2)`, so the
    continued fraction is built as:

    .. math::

        G_n = 0, \qquad
        G_i = \frac{1}{a_i + \omega^2 + 2i\omega\eta - b_i c_i G_{i+1}}

    The returned value is :math:`G(\omega) \times \mathrm{modulus}`
    following the same convention as
    :meth:`Lanczos.get_green_function_continued_fraction`.

    Parameters
    ----------
    a_coeffs : list[float]
    b_coeffs : list[float]
    c_coeffs : list[float]
    w_array : ndarray
        Frequencies in Rydberg.
    perturbation_modulus : float
    use_terminator : bool
    last_average : int
    smearing : float
    smooth_ramp : int
    shift_value : float
    verbose : bool

    Returns
    -------
    gf : ndarray, complex128
        The Green function (Wigner convention).
    """
    n_iters = len(a_coeffs)
    if n_iters == 0:
        raise ValueError("Empty Lanczos coefficients.")
    gf = np.zeros(np.shape(w_array), dtype=np.complex128)

    if use_terminator:
        a_av = np.mean(a_coeffs[-last_average:])
        b_av = np.mean(b_coeffs[-last_average:])
        c_av = b_av
        if len(c_coeffs) == len(b_coeffs):
            c_list = c_coeffs if isinstance(c_coeffs, list) else list(c_coeffs)
            c_av = np.mean(c_list[-last_average:])

        # Guard against degenerate terminator coefficients
        bc_prod = b_av * c_av
        if abs(bc_prod) < 1e-30:
            # Fall back to zero-order continued fraction
            use_terminator = False
        else:
            a_mean = a_av - shift_value
            b_mean = b_av
            c_mean = c_av

            d = a_mean + w_array ** 2
            disc = d ** 2 - 4 * b_mean * c_mean
            sqrt_disc = np.sqrt(disc + 0j)
            denominator = 2 * b_mean * c_mean
            gf_plus = (d + sqrt_disc) / denominator
            gf_minus = (d - sqrt_disc) / denominator
            gf[:] = np.where(np.abs(gf_plus) < np.abs(gf_minus),
                             gf_plus, gf_minus)

    if not use_terminator:
        a = a_coeffs[-1] - shift_value
        gf[:] = 1.0 / (a + w_array ** 2 + 2j * w_array * smearing)

    for i in range(n_iters - 2, -1, -1):
        a = a_coeffs[i] - shift_value
        b = b_coeffs[i]
        c = b
        if len(c_coeffs) == len(b_coeffs):
            c_val = c_coeffs[i]
            if isinstance(c_val, (list, np.ndarray)):
                c = float(c_val) if np.ndim(c_val) == 0 else c_val
            else:
                c = c_val

        if use_terminator and smooth_ramp > 0 and i >= n_iters - smooth_ramp:
            alpha = (i - n_iters + smooth_ramp + 1) / smooth_ramp
            a = (1 - alpha) * a + alpha * a_mean
            b = (1 - alpha) * b + alpha * b_mean
            c = (1 - alpha) * c + alpha * c_mean

        denominator = a + w_array ** 2 + 2j * w_array * smearing - b * c * gf
        # Guard against degenerate coefficients
        with np.errstate(divide='ignore', invalid='ignore'):
            gf = 1.0 / denominator
            gf[np.isnan(gf)] = 0.0
            gf[np.isinf(gf)] = 0.0

    return (-np.real(gf) + 1j * np.imag(gf)) * perturbation_modulus


# =============================================================================
#  Main class
# =============================================================================

class QPointSpectralFunction:
    """Anharmonic spectral function at a selected q-point.

    The class wraps a :class:`QSpaceLanczos` instance, runs the Lanczos
    algorithm for every symmetry-independent mode at the chosen q-point,
    and caches the results.  Post-processing methods give access to
    mode-resolved spectral functions and the unpolarised Raman spectrum
    (the latter always at :math:`\Gamma`, where first-order Raman is
    defined).

    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble
        The SSCHA ensemble.
    select_q : int or ndarray, optional
        Target q-point.  Pass an integer index (into the q-point list)
        or a 3-component Cartesian coordinate in :math:`2\pi/a` units.
        Default is ``0`` (the :math:`\Gamma` point).
    n_lebedev : int, optional
        Lebedev quadrature **order** for powder averaging (default 29,
        which yields 302 points on the unit sphere).
    use_symmetries : bool, optional
        Enable q-space point-group symmetry reduction (default ``True``).
    lo_to_split : str or ndarray or None, optional
        Passed to the underlying :class:`QSpaceLanczos` for the initial
        diagonalization.  For pure mode-resolved spectral functions
        *without* LO-TO (the default internal behaviour) this should be
        ``None``; the LO-TO splitting is applied analytically during
        the Raman powder average.
    **qlanc_kwargs
        Extra keyword arguments forwarded to :class:`QSpaceLanczos`.
    """

    # -- Placzek prefactors (same values as
    #    DynamicalLanczos.get_prefactors_unpolarized_raman) ----------------
    _PLACZEK_PREFACTORS = {
        0: 45.0 / 9.0,    # alpha^2  = (xx+yy+zz)^2
        1: 7.0 / 2.0,     # beta_1^2 = (xx-yy)^2
        2: 7.0 / 2.0,     # beta_2^2 = (xx-zz)^2
        3: 7.0 / 2.0,     # beta_3^2 = (yy-zz)^2
        4: 7.0 * 3.0,     # beta_4^2 = 3*(xy)^2
        5: 7.0 * 3.0,     # beta_5^2 = 3*(yz)^2
        6: 7.0 * 3.0,     # beta_6^2 = 3*(xz)^2
    }

    # -------------------------------------------------------------------
    def __init__(self, ensemble, select_q=0, n_lebedev=29,
                 use_symmetries=True, lo_to_split=None, **qlanc_kwargs):
        if ensemble is None:
            raise ValueError("ensemble must be a valid SSCHA Ensemble.")

        # -- Build the underlying q-space Lanczos ------------------------
        self.qlanc = QL.QSpaceLanczos(ensemble, lo_to_split=lo_to_split,
                                      **qlanc_kwargs)
        self.qlanc.init(use_symmetries=use_symmetries)

        # -- Resolve target q-point --------------------------------------
        if isinstance(select_q, np.ndarray) and select_q.shape == (3,):
            bg = (self.qlanc.uci_structure
                  .get_reciprocal_vectors() / (2 * np.pi))
            self.iq_target = QL.find_q_index(select_q, self.qlanc.q_points, bg)
        elif isinstance(select_q, (int, np.integer)):
            if select_q < 0 or select_q >= self.qlanc.n_q:
                raise IndexError("select_q={} out of range (n_q={})"
                                 .format(select_q, self.qlanc.n_q))
            self.iq_target = int(select_q)
        else:
            raise TypeError("select_q must be int or ndarray(3,), got {}"
                            .format(type(select_q)))

        self._q_point_coords = self.qlanc.q_points[self.iq_target].copy()

        # -- Symmetry-independent modes ----------------------------------
        self.independent_modes = self._find_independent_modes()
        self._n_independent = len(self.independent_modes)

        # -- Lebedev order -----------------------------------------------
        self.n_lebedev = n_lebedev

        # -- Copy available tensors from the dynamical matrix ------------
        dyn = self.qlanc.dyn
        self.effective_charges = (dyn.effective_charges.copy()
                                  if dyn.effective_charges is not None
                                  else None)
        self.dielectric_tensor = (dyn.dielectric_tensor.copy()
                                  if dyn.dielectric_tensor is not None
                                  else None)
        self.raman_tensor = (dyn.raman_tensor.copy()
                             if dyn.raman_tensor is not None
                             else None)

        # -- Unit-cell volume in Bohr^3 ----------------------------------
        self.volume_bohr3 = (self.qlanc.uci_structure.get_volume()
                             * CC.Units.A_TO_BOHR ** 3)

        # -- Internal storage (populated by run_all_modes) ---------------
        self._mode_data = {}
        self._is_run = False

    # ====================================================================
    #  Symmetry reduction helpers
    # ====================================================================

    def _find_independent_modes(self):
        """Identify symmetry-independent (or irreducible) phonon modes.

        Modes are grouped by frequency degeneracy at the target q-point.
        For each degenerate block only **one** representative is kept.
        Acoustic modes (masked by ``valid_modes_q``) are skipped.

        Returns
        -------
        list[tuple[int, int]]
            Sorted list of ``(iq, band)`` pairs.
        """
        iq = self.iq_target
        w_qp = self.qlanc.w_q[:, iq]
        valid = self.qlanc.valid_modes_q[:, iq]
        deg_lists = CC.symmetries.get_degeneracies(w_qp)

        seen = set()
        independent = []
        for nu in range(self.qlanc.n_bands):
            if nu in seen or not valid[nu]:
                continue
            block = sorted(deg_lists[nu].tolist())
            block = [b for b in block if valid[b]]
            for b in block:
                seen.add(b)
            if block:
                independent.append((iq, block[0]))
        return sorted(independent)

    # ====================================================================
    #  Main Lanczos run
    # ====================================================================

    def run_all_modes(self, n_steps, save_dir=None, prefix="QPSF",
                      verbose=True, reorthogonalize=True, **run_kwargs):
        """Run the Hermitian Lanczos for every symmetry-independent mode.

        Parameters
        ----------
        n_steps : int
            Number of Lanczos iterations per mode.
        save_dir : str or None
            Directory for checkpoints (passed to
            :meth:`QSpaceLanczos.run_FT`).
        prefix : str
            Prefix for checkpoint files.
        verbose : bool
        reorthogonalize : bool
        **run_kwargs
            Passed through to :meth:`QSpaceLanczos.run_FT`.
        """
        for idx, (_iq, band) in enumerate(self.independent_modes):
            freq_cm = (self.qlanc.w_q[band, _iq]
                       * CC.Units.RY_TO_CM)

            if verbose:
                print("\n[{}/{}] Band {} : {:.2f} cm-1".format(
                    idx + 1, self._n_independent, band, freq_cm))

            self.qlanc.prepare_mode_q(_iq, band)
            self.qlanc.run_FT(n_steps, save_dir=save_dir,
                              prefix="{}_{}_band{}".format(prefix, idx, band),
                              verbose=verbose,
                              reorthogonalize=reorthogonalize,
                              **run_kwargs)
            self._store(band)

        self._is_run = True

    def _store(self, band):
        """Copy current Lanczos state into ``_mode_data``."""
        data = {
            'a_coeffs': list(self.qlanc.a_coeffs),
            'b_coeffs': list(self.qlanc.b_coeffs),
            'c_coeffs': (list(self.qlanc.c_coeffs)
                         if self.qlanc.c_coeffs
                         else None),
            'w_harmonic': self.qlanc.w_q[band, self.iq_target],
            'polarization': self.qlanc.pols_q[:, band,
                                              self.iq_target].copy(),
            'perturbation_modulus': self.qlanc.perturbation_modulus,
            'n_steps': len(self.qlanc.a_coeffs),
        }
        self._mode_data[(self.iq_target, band)] = data

    # ====================================================================
    #  Mode-resolved spectral function
    # ====================================================================

    def get_spectral_function(self, band, w_array, smearing=0.0,
                              use_terminator=True, smooth_ramp=0,
                              lo_to_direction=None, lo_to_sign=+1.0):
        r"""Compute the anharmonic spectral function of one mode.

        .. math::

            A(\omega) = -\operatorname{Im}\,G_\mu(\omega)

        where :math:`G_\mu` is the continued-fraction Green function.

        Parameters
        ----------
        band : int
            Band index.
        w_array : ndarray
            Frequency grid in **Rydberg**.
        smearing : float
            Broadening in Rydberg.
        use_terminator : bool
        smooth_ramp : int
        lo_to_direction : ndarray(3,) or None
            If given and the target q-point is :math:`\Gamma`, apply the
            LO-TO splitting correction with the non-analytic dynamical
            matrix along this Cartesian direction.
        lo_to_sign : float
            Sign of the LO-TO self-energy term (default +1).  Verify
            against known physical systems.

        Returns
        -------
        spectral : ndarray
            :math:`A(\omega)`.
        """
        gf = self._get_gf_for_mode(band, w_array,
                                   smearing=smearing,
                                   use_terminator=use_terminator,
                                   smooth_ramp=smooth_ramp)

        if lo_to_direction is not None:
            self._check_lo_to_available()
            if self.iq_target != 0:
                warnings.warn(
                    "LO-TO correction requested at q != Gamma; "
                    "the non-analytic correction is physically "
                    "meaningful only at q→0.  Applying anyway.")
            gf = self._apply_lo_to_correction(
                gf, band, lo_to_direction, sign=lo_to_sign)

        return -np.imag(gf)

    # ====================================================================
    #  Unpolarized Raman with powder average (Gamma only)
    # ====================================================================

    def get_unpolarized_raman_powder(self, w_array, smearing=0.0,
                                     use_terminator=True, smooth_ramp=0,
                                     lo_to_sign=+1.0, verbose=True):
        """Unpolarized Raman spectrum with powder-averaged LO-TO.

        Uses the Placzek isotropic invariants for the powder
        intensity, averaged over LO-TO directions by Lebedev
        quadrature on the unit sphere.

        .. note::
            This method **requires** ``iq_target == 0`` (:math:`\Gamma`).
            First-order Raman is defined only at the zone centre.

        Parameters
        ----------
        w_array : ndarray
            Frequency grid in **Rydberg**.
        smearing : float
        use_terminator : bool
        smooth_ramp : int
        lo_to_sign : float
        verbose : bool

        Returns
        -------
        raman : ndarray
            Raman spectrum (arbitrary units).
        """
        if self.iq_target != 0:
            raise RuntimeError(
                "Unpolarized Raman powder spectrum is only available "
                "at Gamma (iq=0).  The current target is iq={}.".format(
                    self.iq_target))
        if self.raman_tensor is None:
            raise RuntimeError(
                "No Raman tensor found in the dynamical matrix.")
        if not self._is_run:
            raise RuntimeError(
                "Call run_all_modes() before computing spectra.")

        has_lo_to = (self.effective_charges is not None
                     and self.dielectric_tensor is not None)

        # -- Lebedev grid -------------------------------------------------
        q_dirs, leb_weights = lebedev_rule(self.n_lebedev)
        # q_dirs: (3, n_points) — columns are unit vectors
        # leb_weights sum to 4π; normalise
        leb_weights /= leb_weights.sum()
        n_leb = len(leb_weights)

        if verbose:
            print("Lebedev grid: {} points (order {})".format(
                n_leb, self.n_lebedev))
            print("LO-TO available:", has_lo_to)

        # -- Accumulate spectrum over Placzek components and modes -------
        raman_total = np.zeros_like(w_array)

        for component in range(7):
            pref = self._PLACZEK_PREFACTORS[component]

            # Raman perturbation vector projected onto each mode
            R1_dict = self._compute_raman_R1_all_modes(component)

            raman_component = np.zeros_like(w_array)

            for (_iq, band), R1_mu in R1_dict.items():
                mode_gf = self._get_gf_for_mode(
                    band, w_array,
                    smearing=smearing,
                    use_terminator=use_terminator,
                    smooth_ramp=smooth_ramp)

                if has_lo_to:
                    intensity = np.zeros_like(w_array)
                    for j in range(n_leb):
                        gf_corr = self._apply_lo_to_correction(
                            mode_gf, band, q_dirs[:, j], sign=lo_to_sign)
                        intensity += leb_weights[j] * (-np.imag(gf_corr))
                else:
                    intensity = -np.imag(mode_gf)

                raman_component += np.abs(R1_mu) ** 2 * intensity

            raman_total += pref * raman_component

        return raman_total

    # ====================================================================
    #  D_LR helpers (delegate to Fortran symph.nonanal)
    # ====================================================================

    def _check_lo_to_available(self):
        if self.effective_charges is None:
            raise RuntimeError("No effective charges available for LO-TO.")
        if self.dielectric_tensor is None:
            raise RuntimeError("No dielectric tensor available for LO-TO.")

    def _compute_D_LR_projection(self, band, q_direction):
        """Compute :math:`\\langle e_\\mu | D^{\\rm LR}(\\hat q) | e_\\mu \\rangle`.

        Uses the Fortran subroutine ``symph.nonanal`` (avoiding code
        duplication with cellconstructor).
        """
        self._check_lo_to_available()

        nat = self.qlanc.uci_structure.N_atoms
        itau = np.arange(nat, dtype=np.int32) + 1       # 1-indexed

        # zeu must be (3, 3, nat) in Fortran order
        zeu = np.asfortranarray(
            np.einsum('sij->ijs', self.effective_charges))

        # dynq is intent(inout) — start from zero
        dynq = np.zeros((3, 3, nat, nat), dtype=np.complex128, order='F')

        # Normalize to unit vector
        q_dir = q_direction / np.linalg.norm(q_direction)

        symph.nonanal(itau, self.dielectric_tensor, q_dir, zeu,
                      self.volume_bohr3, dynq, nat, nat)

        # Unpack (3,3,nat,nat) → (3*nat, 3*nat)
        D_LR = _dynq_to_full(dynq, nat)

        # Mass-weight and project
        e_mu = self._mode_data[(self.iq_target, band)]['polarization']
        return _project_dynmat(D_LR, e_mu,
                               self.qlanc.uci_structure.get_masses_array())

    def _apply_lo_to_correction(self, gf_array, band, q_direction,
                                 sign=+1.0):
        """Apply LO-TO correction to a mode-resolved Green function.

        .. math::

            G'(\\omega) = \\left[G^{-1}(\\omega)
                           + s \\, \\langle e|D^{\\rm LR}|e\\rangle
                           \\right]^{-1}
        """
        proj = self._compute_D_LR_projection(band, q_direction)
        if abs(proj) < 1e-16:
            return gf_array
        # Guard division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            gf_inv = 1.0 / gf_array
            result = 1.0 / (gf_inv + sign * proj)
            # Where gf_array was ~0, the corrected gf is also ~0
            result[~np.isfinite(result)] = 0.0
        return result

    # ====================================================================
    #  Green-function reconstruction from stored coefficients
    # ====================================================================

    def _get_gf_for_mode(self, band, w_array, **kwargs):
        """Return the Wigner Green function for one mode."""
        key = (self.iq_target, band)
        if key not in self._mode_data:
            raise KeyError("Mode (iq={}, band={}) has not been computed. "
                           "Call run_all_modes() first.".format(
                               self.iq_target, band))
        data = self._mode_data[key]
        c_coeffs = data['c_coeffs'] if data['c_coeffs'] is not None else []
        return _gf_from_abc(
            data['a_coeffs'], data['b_coeffs'], c_coeffs,
            w_array,
            perturbation_modulus=data['perturbation_modulus'],
            **kwargs)

    # ====================================================================
    #  Raman tensor contractions
    # ====================================================================

    def _get_raman_cartesian_vector(self, component):
        """Raw Cartesian Raman vector for a Placzek component.

        Returns ``v`` of shape ``(3*nat,)`` **without** mass scaling
        or supercell tiling.
        """
        raman = self.raman_tensor  # (3, 3, 3*nat)
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])

        if component == 0:      # alpha: (xx + yy + zz)
            v = np.einsum('ija,i,j->a', raman, ex, ex)
            v += np.einsum('ija,i,j->a', raman, ey, ey)
            v += np.einsum('ija,i,j->a', raman, ez, ez)
        elif component == 1:    # beta_1: (xx - yy)
            v = np.einsum('ija,i,j->a', raman, ex, ex)
            v -= np.einsum('ija,i,j->a', raman, ey, ey)
        elif component == 2:    # beta_2: (xx - zz)
            v = np.einsum('ija,i,j->a', raman, ex, ex)
            v -= np.einsum('ija,i,j->a', raman, ez, ez)
        elif component == 3:    # beta_3: (yy - zz)
            v = np.einsum('ija,i,j->a', raman, ey, ey)
            v -= np.einsum('ija,i,j->a', raman, ez, ez)
        elif component == 4:    # beta_4: (xy)
            v = np.einsum('ija,i,j->a', raman, ex, ey)
        elif component == 5:    # beta_5: (yz)
            v = np.einsum('ija,i,j->a', raman, ey, ez)
        elif component == 6:    # beta_6: (xz)
            v = np.einsum('ija,i,j->a', raman, ex, ez)
        else:
            raise ValueError("Invalid component index {}"
                             .format(component))
        return v

    def _compute_raman_R1_all_modes(self, component):
        """Project Raman vector onto each mode, returning {key: R1_mu}.

        Mimics :meth:`QSpaceLanczos.prepare_perturbation_q` for
        Gamma (iq=0) without actually modifying the Lanczos state.
        """
        raman_v = self._get_raman_cartesian_vector(component)   # (3*nat,)

        # Same scaling as prepare_perturbation_q (Gamma point)
        n_cell = np.prod(self.qlanc.dyn.GetSupercell())
        # mass weighting (unit cell only)
        m_uc = np.tile(self.qlanc.uci_structure.get_masses_array(), 3)
        v_scaled = raman_v * np.sqrt(n_cell) / np.sqrt(m_uc)

        # pols_q[:, :, 0] is (3*nat, n_bands) at Gamma
        pols_gamma = self.qlanc.pols_q[:, :, 0]

        R1_all = {}
        for (_iq, band) in self.independent_modes:
            e_mu = pols_gamma[:, band]
            R1_mu = np.conj(e_mu) @ v_scaled
            R1_all[(_iq, band)] = R1_mu

        # Extend to degenerate partners (same R1 in the mode basis
        # because the Raman vector respects the symmetries of the
        # irreducible representation — the projection is identical
        # for degenerate modes sharing the same block).
        # Recompute with full degeneracy grouping:
        deg_dict = self._get_degenerate_partners()
        for (_iq, band), partners in deg_dict.items():
            for partner in partners:
                if partner != band and (_iq, partner) not in R1_all:
                    R1_all[(_iq, partner)] = R1_all[(_iq, band)]

        return R1_all

    def _get_degenerate_partners(self):
        """Map each independent mode to its degenerate partners."""
        iq = self.iq_target
        w_qp = self.qlanc.w_q[:, iq]
        valid = self.qlanc.valid_modes_q[:, iq]
        deg_lists = CC.symmetries.get_degeneracies(w_qp)

        partners = {}
        for (_iq, band) in self.independent_modes:
            block = sorted(deg_lists[band].tolist())
            block = [b for b in block if valid[b]]
            partners[(_iq, band)] = block
        return partners

    # ====================================================================
    #  Save / Load
    # ====================================================================

    def save(self, filepath):
        """Save all computed data to a compressed ``.npz`` file."""
        if not self._is_run:
            raise RuntimeError("No data to save — call run_all_modes() first.")

        # Flatten mode_data into savable arrays
        keys = sorted(self._mode_data.keys())
        iq_list = np.array([k[0] for k in keys], dtype=np.int32)
        band_list = np.array([k[1] for k in keys], dtype=np.int32)

        # Store variable-length arrays as object arrays
        a_list = np.empty(len(keys), dtype=object)
        b_list = np.empty(len(keys), dtype=object)
        c_list = np.empty(len(keys), dtype=object)
        w_harmonic = np.empty(len(keys), dtype=np.float64)
        pols = np.empty(len(keys), dtype=object)
        pert_mod = np.empty(len(keys), dtype=np.float64)
        n_steps = np.empty(len(keys), dtype=np.int32)

        for i, k in enumerate(keys):
            d = self._mode_data[k]
            a_list[i] = np.array(d['a_coeffs'])
            b_list[i] = np.array(d['b_coeffs'])
            c_list[i] = (np.array(d['c_coeffs'])
                         if d['c_coeffs'] is not None
                         else np.array([]))
            w_harmonic[i] = d['w_harmonic']
            pols[i] = d['polarization']
            pert_mod[i] = d['perturbation_modulus']
            n_steps[i] = d['n_steps']

        save_dict = dict(
            iq_target=self.iq_target,
            n_lebedev=self.n_lebedev,
            has_raman=self.raman_tensor is not None,
            has_effective_charges=self.effective_charges is not None,
            has_dielectric=self.dielectric_tensor is not None,
            volume_bohr3=self.volume_bohr3,
            iq_list=iq_list,
            band_list=band_list,
            a_list=a_list,
            b_list=b_list,
            c_list=c_list,
            w_harmonic=w_harmonic,
            pols=pols,
            perturbation_modulus=pert_mod,
            n_steps=n_steps,
            independent_modes=np.array(self.independent_modes,
                                       dtype=np.int32),
        )

        if self.raman_tensor is not None:
            save_dict['raman_tensor'] = self.raman_tensor
        if self.effective_charges is not None:
            save_dict['effective_charges'] = self.effective_charges
        if self.dielectric_tensor is not None:
            save_dict['dielectric_tensor'] = self.dielectric_tensor

        np.savez_compressed(filepath, **save_dict)

    @classmethod
    def load(cls, filepath, ensemble):
        """Load a previously saved :class:`QPointSpectralFunction`.

        Parameters
        ----------
        filepath : str
            Path to the ``.npz`` file written by :meth:`save`.
        ensemble : sscha.Ensemble.Ensemble
            A fresh ensemble (must match the one used when saving).

        Returns
        -------
        QPointSpectralFunction
        """
        data = np.load(filepath, allow_pickle=True)

        obj = cls(ensemble, select_q=int(data['iq_target']),
                  n_lebedev=int(data['n_lebedev']))

        # Restore tensors
        if data['has_raman']:
            obj.raman_tensor = data['raman_tensor']
        if data['has_effective_charges']:
            obj.effective_charges = data['effective_charges']
        if data['has_dielectric']:
            obj.dielectric_tensor = data['dielectric_tensor']
        obj.volume_bohr3 = float(data['volume_bohr3'])

        # Restore mode data
        iq_list = data['iq_list']
        band_list = data['band_list']
        a_arr = data['a_list']
        b_arr = data['b_list']
        c_arr = data['c_list']
        w_arr = data['w_harmonic']
        pols_arr = data['pols']
        pmod_arr = data['perturbation_modulus']
        nstep_arr = data['n_steps']

        for i in range(len(iq_list)):
            c_c = list(c_arr[i])
            if len(c_c) == 0 or np.allclose(c_c, 0):
                c_c = None
            else:
                c_c = c_c

            obj._mode_data[(int(iq_list[i]), int(band_list[i]))] = {
                'a_coeffs': list(a_arr[i]),
                'b_coeffs': list(b_arr[i]),
                'c_coeffs': c_c,
                'w_harmonic': float(w_arr[i]),
                'polarization': pols_arr[i],
                'perturbation_modulus': float(pmod_arr[i]),
                'n_steps': int(nstep_arr[i]),
            }

        obj.independent_modes = [tuple(x) for x in data['independent_modes']]
        obj._n_independent = len(obj.independent_modes)
        obj._is_run = True

        return obj

    # ====================================================================
    #  Pretty printing
    # ====================================================================

    def __repr__(self):
        q_str = ("({:8.5f}, {:8.5f}, {:8.5f})"
                 .format(*self._q_point_coords))
        return ("QPointSpectralFunction(q={}, iq_target={}, "
                "n_independent_modes={}, is_run={})"
                .format(q_str, self.iq_target,
                        self._n_independent, self._is_run))


# =============================================================================
#  Pure utility functions (not class members)
# =============================================================================

def _dynq_to_full(dynq, nat):
    """Convert (3,3,nat,nat) Fortran-order dynq → (3*nat, 3*nat) C-order."""
    full = np.zeros((3 * nat, 3 * nat), dtype=np.complex128)
    for i in range(nat):
        for j in range(nat):
            full[3 * i:3 * i + 3, 3 * j:3 * j + 3] = dynq[:, :, i, j]
    return full


def _project_dynmat(dyn_mat, e_vector, masses):
    """Mass-weighted projection ``<e| D_mat / √(mm) |e>``.

    Parameters
    ----------
    dyn_mat : ndarray(3*nat, 3*nat)
        Force-constant matrix (units of Ry/Bohr²).
    e_vector : ndarray(3*nat,)
        Complex eigenvector (ortho-normal, from ``DiagonalizeSupercell``).
    masses : ndarray(nat,)
        Atomic masses in AMU.

    Returns
    -------
    float
        The real part of the mass-weighted projection.
    """
    sqrt_m = np.sqrt(np.tile(masses, 3))
    D_mass = dyn_mat / np.outer(sqrt_m, sqrt_m)
    return np.real(np.conj(e_vector) @ D_mass @ e_vector)
