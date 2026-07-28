"""
Trilinear Q-Space Interpolated Lanczos
======================================

Implementation of the trilinear q-space interpolation of the TDSCHA
anharmonic operator (Overleaf: "Interpolation of high-rank phonon
interactions within the TD-SCHA", with the sign/normalization corrections
listed in Trilinear_interpolation_plan.md section 1).

Scheme
------
The perturbation momentum Q is constrained to the coarse mesh; the
two-phonon sector of psi runs over FINE pairs (q', Q - q') with q' on a
fine mesh that is an integer multiple of the coarse one. Every
per-configuration Bloch field of a fine pair is evaluated at the 8
trilinear corners of q' on the coarse mesh (the partner leg Q - k_eps is
then also coarse, so momentum conservation holds term by term), with the
corner mixing performed in primitive-cell CARTESIAN coordinates -- never
across q in the polarization basis.

Key algorithmic property (exact "folding"): the per-configuration weights
do not depend on q', so the whole fine-pair kernel folds exactly onto the
coarse mesh,

    A_cart(k) = sum_{q' in fine BZ} I_{eps: k_eps(q') = k}(q')
                    * alpha_cart(q', Q - q'),

and the fine d2v outputs are the trilinear interpolation (the exact
adjoint of the fold) of the coarse Cartesian d2v blocks. The config x
symmetry loop therefore runs on the UNMODIFIED coarse Julia kernel
(get_perturb_averages_qspace) at coarse cost; the O(N_f) fold/unfold is
cheap numpy work. Hermiticity of L is preserved because unfold = adjoint
of fold and the mesh measure is split symmetrically through the existing
scale3 = sqrt(N_c/N_f), scale4 = N_c/N_f vertex factors.

Permutation symmetry: for coarse Q the corner set of Q - q' is the mirror
of the corner set of q' with identical weights, so folding both
orientations of each stored pair block (reverse orientation = transpose,
bilinear convention) yields A(Q - k) = A(k)^T exactly -- the transpose
symmetry the Julia kernel assumes. The permutations that move the Q leg
are handled by the kernel's force-channel symmetrization, unchanged from
the commensurate calculation.

Storage: the psi two-phonon blocks stay in the per-fine-pair Wigner mode
basis (harmonic propagator and chi factors are diagonal there); the
per-pair mode <-> Cartesian rotations are exact unitaries using each q's
own eigenvectors, so no eigenvector-phase/degeneracy ambiguity can enter
the interpolation.

Note on the estimator: the coarse kernel's f_Y filter puts the coarse
Upsilon on the interpolated legs (r1 = f_Y x = mode representation of
Upsilon u), so the interpolated object is the correct rank-3 combination
<(Upsilon u)(Upsilon u) f> of the paper -- the fine Psi enters only the
per-fine-pair chi factors. This is NOT the raw-correlation interpolation
that showed the 2x bias in the windowed scheme.

The optional ``atomic_phase`` applies the full-Bloch
exp(-2 pi i q.tau_a) gauge on both legs of the piecewise-linear fold and its
inverse adjoint on unfold.  This is a gauge-transport diagnostic; it does not
perform an atom-dependent real-space image assignment.

The optional ``atom_fourier`` replaces the local tent weights by an
atom-pair-resolved trigonometric cardinal kernel.  For pair (a,b), every
aliased lattice harmonic R is represented by the image for which
|R-(tau_a-tau_b)| is shortest (exact ties are split).  It is the Fourier
continuation naturally selected by the atom-centred Bloch convention.  It
still uses only coarse q-space blocks and its adjoint, but it is not a
piecewise-trilinear map.

The shortest-image assignment and its Nyquist ties are resolved with the
true 3D cell metric (``tie_metric=True``, default): on a non-orthogonal
cell the minimal image minimizes the Cartesian length |(R-d).A| and its
degeneracies can be genuine >2-fold and non-separable (e.g. a coupling on a
Wigner-Seitz face of an fcc/bcc/hexagonal cell).  The legacy separable
per-Cartesian-axis rule (``tie_metric=False``) is metric-free, exact only
for orthorhombic coarse lattices, and kept as a fast path / control.  The
distinction only changes an observable when a coupling carries weight in an
aliasing class the separable rule mis-assigns; short-ranged couplings whose
excited classes have a unique shortest image (e.g. the SnTe Gamma-TO
vertex) are reconstructed identically by both.  See
report/interpolation/scripts/audit_nyquist_ties.py.
"""
from __future__ import print_function
from __future__ import division

import itertools
import warnings
import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Methods

import cellconstructor.Settings as Parallel
from cellconstructor.Settings import ParallelPrint as print

import tdscha.QSpaceLanczos as QL
import tdscha.JuliaExt as JuliaExt
from tdscha.QSpaceInterpolation import (
    generate_fine_mesh, build_q_index_lookup, _mesh_key,
    interpolate_dyn_fine)


class QSpaceTrilinearLanczos(QL.QSpaceLanczos):
    """Q-space Lanczos with trilinear interpolation of the anharmonic vertex.

    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble
        The (coarse-supercell) SSCHA ensemble.
    fine_mesh : tuple(3) of int
        The fine q-mesh; each dimension must be an integer multiple of the
        corresponding coarse supercell dimension.
    use_asr_dyn : bool
        Apply the acoustic sum rule to the centered force constants before
        Fourier-interpolating the dynamical matrix.
    ignore_effective_charges : bool
        Opt-in.  If True, the Born effective charges and dielectric tensor
        stored in the dynamical matrix are hidden from the *dynamical-matrix
        interpolation only*.  Default False, which keeps the standard
        cellconstructor behaviour.

        The scope really is only the interpolation: the ensemble's ``dyn``
        is not modified, so IR intensities, Raman/IR response functions and
        anything else that legitimately needs Z*/eps keep seeing them.

        Turn it on when the forces come from a potential with no intrinsic
        long-range electrostatics -- typically a short-range machine-learning
        interatomic potential -- where the Z*/eps in the dynamical-matrix
        header were inherited from a DFT reference and do not describe the
        potential that generated the ensemble.  In that case ``ForceTensor``
        subtracts a dipole tail that is not in the data, leaving a remainder
        that is not short-ranged, so centering truncates it and the ASR is
        imposed on the wrong object.  The damage is invisible on the coarse
        mesh (the subtract/re-add cycle is exactly the identity there) and
        shows up between coarse points, typically as imaginary modes on the
        off-grid shell of the fine mesh.  See ``ignore_effective_charges``
        in ``QSpaceInterpolation.interpolate_dyn_fine``.
    w_min_guard : float
        Interpolated modes with |w| below this (Ry), away from Gamma, are
        masked (chi factors diverge as 1/w).
    allow_unstable : bool
        Diagnostic escape hatch only.  If True, imaginary interpolated
        frequencies are excluded from the one- and two-phonon Hilbert space
        with a warning instead of raising.  The resulting spectrum is
        incomplete and must not be treated as a physical interpolation.
    atomic_phase : bool
        Use the full-Bloch exp(-2 pi i q.tau_a) gauge for the rank-2
        fold/unfold interpolation. The inverse phase is used on the adjoint
        return path.
    atom_fourier : bool
        Use the atom-centred trigonometric cardinal continuation instead of
        piecewise trilinear weights. Mutually exclusive with
        ``atomic_phase``.
    tie_metric : bool
        Only relevant with ``atom_fourier=True``. If True (default), the
        minimal image of each aliasing class and its Nyquist ties are
        resolved with the true 3D cell metric ``A A^T`` (the assignment
        used by tensor centering), which is correct on any cell. If False,
        the legacy separable per-Cartesian-axis, metric-free rule is used;
        this is exact only for orthorhombic coarse lattices and is retained
        as a fast path and as the control that exposes the discrepancy on
        non-orthogonal cells (see ``audit_nyquist_ties.py``).
    """

    # The interpolation state layered on top of the coarse QSpaceLanczos.
    _TRI_ATTRS = (
        'fine_mesh', 'coarse_mesh', '_fine_idx',
        '_q_lookup', '_coarse_lookup', '_coarse_idx',
        '_corners', '_fine_pair_of',
        '_fine_of_coarse', '_coarse_of_fine',
        'cq_points', 'cn_q', 'cw_q', 'cpols_q',
        'cvalid_modes_q',
        'c_iq_pert', 'c_q_pair_map', 'c_unique_pairs',
        'atomic_phase', 'atom_fourier', 'tie_metric',
        'ignore_effective_charges',
        '_interp_used_effective_charges',
        '_atom_fourier_kernel')

    # What the distributed loader must carry to the worker ranks: the
    # interpolation state above, plus the mesh-measure scale factors that
    # __init__ sets on the parent (section 2.3 of the plan).  These are
    # broadcast rather than recomputed per rank: interpolate_dyn_fine
    # diagonalises the fine dynamical matrix, and two diagonalisations of a
    # degenerate block need not agree on a gauge -- ranks disagreeing on the
    # polarization vectors would silently corrupt every reduced dot product.
    _DISTRIBUTED_EXTRA_ATTRS = _TRI_ATTRS + (
        'qspace_scale3', 'qspace_scale4', 'qspace_prefiltered')

    def __init__(self, ensemble, fine_mesh=None, use_asr_dyn=True,
                 ignore_effective_charges=False,
                 w_min_guard=1e-8, allow_unstable=False,
                 atomic_phase=False,
                 atom_fourier=False, tie_metric=True,
                 lo_to_split=None, **kwargs):

        if lo_to_split is not None:
            raise NotImplementedError(
                "LO-TO splitting is not supported by the trilinear "
                "interpolated q-space Lanczos (phase 1).")

        super().__init__(ensemble, lo_to_split=None, **kwargs)

        self.__total_attributes__.extend(self._TRI_ATTRS)

        self.atomic_phase = bool(atomic_phase)
        self.atom_fourier = bool(atom_fourier)
        self.tie_metric = bool(tie_metric)
        if self.atomic_phase and self.atom_fourier:
            raise ValueError("atomic_phase and atom_fourier are mutually "
                             "exclusive interpolation choices")

        self.fine_mesh = None
        self._fine_idx = None
        self._q_lookup = None
        self._coarse_lookup = None
        self._coarse_idx = None
        self._corners = None
        self._fine_pair_of = None
        self._atom_fourier_kernel = None
        self.c_iq_pert = None
        self.c_q_pair_map = None
        self.c_unique_pairs = None

        # Bare initialization (distributed loader path of the parent)
        if ensemble is None:
            return

        if fine_mesh is None:
            raise ValueError(
                "QSpaceTrilinearLanczos requires fine_mesh=(m1, m2, m3)")

        self.fine_mesh = np.asarray(fine_mesh, dtype=int)
        self.coarse_mesh = np.asarray(self.dyn.GetSupercell(), dtype=int)
        if np.any(self.fine_mesh % self.coarse_mesh != 0):
            raise ValueError(
                "fine_mesh {} must be an integer multiple of the coarse "
                "mesh {} (the perturbation Q and the pair partners must "
                "live on both meshes)".format(tuple(self.fine_mesh),
                                              tuple(self.coarse_mesh)))

        # == 1. Stash the coarse-side arrays (Julia kernel operates here) ==
        self.cq_points = np.array(self.q_points)
        self.cn_q = self.n_q
        self.cw_q = np.array(self.w_q)
        self.cpols_q = np.array(self.pols_q)
        self.cvalid_modes_q = np.array(self.valid_modes_q)
        # X_q, Y_q remain the coarse ensemble Bloch fields (untouched).

        self._coarse_lookup = build_q_index_lookup(
            self.cq_points, self.uci_structure, self.coarse_mesh)
        self._coarse_idx = [
            _mesh_key(q, self.uci_structure, self.coarse_mesh)
            for q in self.cq_points]

        # == 2. Fine mesh and interpolated dynamical matrix ==
        q_fine, idx_fine = generate_fine_mesh(self.uci_structure,
                                              self.fine_mesh)
        self._fine_idx = idx_fine
        self._q_lookup = build_q_index_lookup(q_fine, self.uci_structure,
                                              self.fine_mesh)
        self.ignore_effective_charges = bool(ignore_effective_charges)
        self._interp_used_effective_charges = (
            self.dyn.effective_charges is not None
            and not self.ignore_effective_charges)
        w_f, pols_f = interpolate_dyn_fine(
            self.dyn, q_fine, use_asr=use_asr_dyn,
            ignore_effective_charges=self.ignore_effective_charges,
            reuse_commensurate=True)

        # Pin the commensurate fine points to the parent's
        # DiagonalizeSupercell output: the R sector and the kernel exchange
        # R1/f_pert in the mode basis at Q, which must be the SAME basis on
        # both sides (degenerate-subspace gauges of two diagonalizations of
        # the same matrix need not coincide).
        self._fine_of_coarse = np.full(self.cn_q, -1, dtype=int)
        self._coarse_of_fine = np.full(len(q_fine), -1, dtype=int)
        for jq in range(self.cn_q):
            key_f = _mesh_key(self.cq_points[jq], self.uci_structure,
                              self.fine_mesh)
            iq = self._q_lookup[key_f]
            self._fine_of_coarse[jq] = iq
            self._coarse_of_fine[iq] = jq
            w_f[:, iq] = self.cw_q[:, jq]
            pols_f[:, :, iq] = self.cpols_q[:, :, jq]

        self.q_points = q_fine
        self.n_q = len(q_fine)
        self.w_q = w_f
        self.pols_q = pols_f

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
            n_bad = int(np.sum(unstable))
            msg = self._unstable_message(unstable, bad, n_bad)
            if allow_unstable:
                warnings.warn(
                    msg + "\n\nallow_unstable=True: EXCLUDING these modes "
                    "from the Lanczos Hilbert space. Every two-phonon "
                    "channel containing one of them is dropped, so the "
                    "resulting spectrum is INCOMPLETE and is for "
                    "diagnostics only -- do not publish it as an "
                    "interpolated result.")
                self.valid_modes_q &= ~unstable
            else:
                raise ValueError(msg)

        small = (np.abs(self.w_q) < w_min_guard) & self.valid_modes_q
        small[:, 0] = False
        if np.any(small):
            warnings.warn("Masking {} interpolated modes with |w| < {} Ry "
                          "away from Gamma.".format(np.sum(small),
                                                    w_min_guard))
            self.valid_modes_q &= ~small

        if ensemble.ignore_small_w:
            small_freq = np.abs(self.w_q) < CC.Phonons.__EPSILON_W__
            self.valid_modes_q &= ~small_freq

        # == 4. Trilinear corner cache (fine mesh geometry, Q-independent) ==
        self._build_corner_cache()
        if self.atom_fourier:
            self._build_atom_fourier_kernel()

        # == 5. Hermitian-symmetric mesh measure (plan section 2.3) ==
        ratio = float(self.cn_q) / float(self.n_q)
        self.qspace_scale3 = np.sqrt(ratio)
        self.qspace_scale4 = ratio
        self.qspace_prefiltered = False

        # Reset pair-map state (was initialized on the coarse mesh)
        self.iq_pert = None
        self.q_pair_map = None
        self.unique_pairs = None
        self._psi_size = None

    # ================================================================
    # Diagnostics
    # ================================================================
    def _unstable_message(self, unstable, bad, n_bad):
        """Explain imaginary interpolated modes and what to do about them.

        The single most useful discriminator is *where* the unstable points
        sit.  Fine q-points that are also coarse points are pinned to the
        SSCHA dynamical matrix itself, so an instability there is a property
        of the input, not of the interpolation.  Instabilities confined to
        the off-grid points are produced by the continuation, and the
        commonest cause is the effective-charge subtract/re-add cycle
        applied to an ensemble whose potential has no long-range part.
        """
        w_cm = self.w_q * CC.Units.RY_TO_CM
        on_grid = [int(iq) for iq in bad if self._coarse_of_fine[iq] >= 0]
        off_grid = [int(iq) for iq in bad if self._coarse_of_fine[iq] < 0]

        lines = [
            "Interpolated dynamical matrix has {} imaginary modes at {} of "
            "the {} fine q-points.".format(n_bad, len(bad), self.n_q),
            "  most negative signed frequency: {:.4f} cm-1".format(
                float(w_cm[unstable].min())),
            "  on coarse-mesh q-points : {} {}".format(
                len(on_grid), on_grid if on_grid else ""),
            "  on off-grid q-points    : {} {}".format(
                len(off_grid), off_grid if off_grid else ""),
        ]

        if on_grid:
            lines += [
                "",
                "Some unstable points are commensurate with the ensemble "
                "supercell. Those are pinned to the SSCHA dynamical matrix "
                "and are NOT an interpolation artifact: the auxiliary "
                "reference itself is unstable there, and the "
                "stable-reference TD-SCHA expansion does not apply. "
                "Re-converge the SSCHA minimization before interpolating.",
            ]

        if self._interp_used_effective_charges:
            lines += [
                "",
                "The interpolation USED the Born effective charges and "
                "dielectric tensor stored in the dynamical matrix (the "
                "default). Is that what you want for this ensemble?",
                "",
                "ForceTensor subtracts the Ewald dipole-dipole term before "
                "centering and adds it back at every interpolated q. That "
                "cycle is exactly the identity on the coarse mesh, so it "
                "cannot be detected by any commensurate test, but off-grid "
                "it replaces the Fourier continuation of your measured "
                "force constants with an analytic dipole continuation. If "
                "the potential that generated the ensemble has no intrinsic "
                "long-range electrostatics -- in particular a SHORT-RANGE "
                "MACHINE-LEARNING INTERATOMIC POTENTIAL, whose dynamical "
                "matrix may still carry Z*/eps inherited from a DFT "
                "reference -- then the subtracted tail is not in the data, "
                "the centered remainder is no longer short-ranged, and the "
                "ASR is imposed on the wrong object. Spurious imaginary "
                "modes on the off-grid shell are the typical symptom.",
                "",
                "  If your forces come from a short-range MLIP, rerun with",
                "      {}(..., ignore_effective_charges=True)".format(
                    type(self).__name__),
                "  which hides Z*/eps from the dynamical-matrix "
                "interpolation ONLY. The ensemble's dynamical matrix is not "
                "modified, so IR intensities and any other response "
                "function that needs the effective charges are unaffected.",
                "  If the potential really is polar, keep the default and "
                "treat the instability as physical or as under-resolution "
                "of the coarse mesh.",
            ]
        else:
            lines += [
                "",
                "The interpolation ran with ignore_effective_charges=True, "
                "so the long-range subtract/re-add cycle is already "
                "excluded as the cause. The remaining candidates are a "
                "genuinely unstable auxiliary reference between the sampled "
                "q-points, an under-converged or under-resolved coarse "
                "dynamical matrix, or the centering/ASR prescription "
                "(try use_asr_dyn=False to separate the last one).",
            ]

        lines += [
            "",
            "Set allow_unstable=True to mask these modes instead of "
            "raising, but note that this REMOVES every two-phonon channel "
            "containing them: the spectrum becomes incomplete and is valid "
            "only as a diagnostic.",
        ]
        return "\n".join(lines)

    # ================================================================
    # Geometry
    # ================================================================
    def _build_corner_cache(self):
        """corners[iq_fine] = [(coarse_index, weight, delta), ...], w > 0.

        Exact integer arithmetic: q'_frac_d = n_d / N_f_d, so
        N_c_d * x_d = j_d + r_d / N_f_d with (j_d, r_d) = divmod(...).
        Commensurate fine points collapse to a single corner of weight 1.

        ``delta`` is the LOCAL fractional displacement from the fine
        point to the (unwrapped) corner, delta_d = (j_d + eps_d)/N_c_d -
        n_d/N_f_d, with |delta_d| <= 1/N_c_d.  The atomic-phase gauge
        must be built from this local displacement: wrapping the corner
        and the fine point into the BZ independently can shift their
        difference by a lattice vector, which flips the sign of
        exp(-2 pi i q.tau_a) phases for half-integer tau differences.
        """
        Nc = self.coarse_mesh
        Nf = self.fine_mesh
        self._corners = []
        for n in self._fine_idx:
            jt = [divmod(int(Nc[d]) * int(n[d]), int(Nf[d]))
                  for d in range(3)]
            entries = []
            for eps in itertools.product((0, 1), repeat=3):
                w = 1.0
                for d in range(3):
                    t = jt[d][1] / float(Nf[d])
                    w *= t if eps[d] else (1.0 - t)
                if w == 0.0:
                    continue
                key = tuple((jt[d][0] + eps[d]) % Nc[d] for d in range(3))
                delta = np.array(
                    [(jt[d][0] + eps[d]) / float(Nc[d])
                     - int(n[d]) / float(Nf[d]) for d in range(3)])
                entries.append((self._coarse_lookup[key], w, delta))
            self._corners.append(entries)

    def find_fine_q(self, q):
        """Index of a (Cartesian) q-vector in the fine mesh, O(1)."""
        return self._q_lookup[_mesh_key(q, self.uci_structure,
                                        self.fine_mesh)]

    def _atomic_delta_phase(self, delta):
        """Diagonal transport phase exp(+2 pi i delta.tau_a) per (atom,
        cart) component, for the LOCAL fine->corner displacement delta.

        This equals pc(x_corner)* pf(q_fine) with p(x) =
        exp(-2 pi i x.tau_a) evaluated on the same (unwrapped) branch,
        which is the only branch-consistent choice: independent BZ
        wrapping of corner and fine point shifts delta by lattice
        vectors and flips the sign for half-integer tau differences.
        """
        tau = np.linalg.solve(self.uci_structure.unit_cell.T,
                              self.uci_structure.coords.T).T
        phase_atom = np.exp(2j * np.pi * (tau @ np.asarray(delta)))
        return np.repeat(phase_atom, 3)

    @staticmethod
    def _nearest_alias_images(alias, centre, mesh, tol=1e-12):
        """Nearest integers R == alias (mod mesh) to an atomic centre.

        The returned list contains ``(R, weight)`` pairs. Exact Nyquist
        ties are retained with equal weights, as in tensor centering.

        NOTE: this is the SEPARABLE, per-axis, metric-FREE rule. It is
        exact only for orthorhombic coarse lattices. For non-orthogonal
        cells use ``_metric_alias_images`` (the ``tie_metric=True``
        default), which resolves the minimal image and its (possibly
        >2-fold, non-separable) ties with the true 3D cell metric. See
        ``report/interpolation/scripts/audit_nyquist_ties.py``.
        """
        m0 = int(np.floor((float(centre) - int(alias)) / int(mesh)))
        candidates = np.array(
            [int(alias) + int(mesh) * m for m in range(m0 - 2, m0 + 4)],
            dtype=np.float64)
        distance = np.abs(candidates - float(centre))
        keep = np.abs(distance - np.min(distance)) < tol
        chosen = candidates[keep]
        return [(float(r), 1.0 / len(chosen)) for r in chosen]

    @staticmethod
    def _metric_alias_images(d, Nc, metric, tol=1e-9, span=2):
        """True 3D minimal-image assignment of every aliasing class.

        For atom separation ``d`` (fractional) and coarse mesh ``Nc``,
        returns a dict mapping each class tuple ``k`` in prod(range(Nc)) to
        a list of ``(R (3-tuple int), weight)`` pairs: the lattice images
        ``R == k (mod Nc)`` minimizing the Cartesian length
        ``sqrt((R-d) . metric . (R-d))``, with exact ties split at equal
        weight (the rule tensor centering uses).

        ``metric = A A^T`` with ``A`` the fractional-to-Cartesian matrix
        (rows = lattice vectors), so a fractional vector ``v`` has squared
        Cartesian length ``v . metric . v``. This couples the three axes
        and, unlike the separable rule, can produce genuine >2-fold ties
        and select non-axis-aligned images.
        """
        d = np.asarray(d, dtype=np.float64)
        Nc = np.asarray(Nc, dtype=int)
        shifts = list(itertools.product(range(-span, span + 1), repeat=3))
        out = {}
        for k in itertools.product(*[range(int(n)) for n in Nc]):
            best = []
            best_d2 = np.inf
            for m in shifts:
                R = np.array([k[ax] + int(Nc[ax]) * m[ax]
                              for ax in range(3)], dtype=np.float64)
                v = R - d
                d2 = float(v @ metric @ v)
                if d2 < best_d2 - tol:
                    best_d2 = d2
                    best = [tuple(int(x) for x in R)]
                elif abs(d2 - best_d2) < tol:
                    best.append(tuple(int(x) for x in R))
            w = 1.0 / len(best)
            out[k] = [(R, w) for R in best]
        return out

    def _build_atom_fourier_kernel(self):
        """Cache atom-pair cardinal weights P(q_fine,k_coarse,a,b).

        For a coarse sample n/N and atom separation d = tau_a - tau_b,

          P_d(q,n) = 1/prod(N) sum_class sum_{R == class (mod N), min image}
                         w_R exp[2 pi i (q-n/N) . R].

        At a commensurate q it is exactly the identity. Between samples it
        selects the aliased Fourier image(s) shortest to d.

        The minimal-image assignment is done with the true 3D cell metric
        (``tie_metric=True``, default), which is correct on any cell and
        represents genuine >2-fold / non-separable Nyquist ties. With
        ``tie_metric=False`` the legacy separable per-axis product is used
        (exact only for orthorhombic coarse lattices).
        """
        tau = np.linalg.solve(self.uci_structure.unit_cell.T,
                              self.uci_structure.coords.T).T
        nat = len(tau)
        kernel = np.empty((self.n_q, self.cn_q, nat, nat),
                          dtype=np.complex128)
        fine_frac = (np.asarray(self._fine_idx, dtype=np.float64)
                     / self.fine_mesh[None, :])
        coarse_frac = (np.asarray(self._coarse_idx, dtype=np.float64)
                       / self.coarse_mesh[None, :])

        if self.tie_metric:
            self._build_atom_fourier_kernel_metric(
                kernel, tau, nat, fine_frac, coarse_frac)
        else:
            self._build_atom_fourier_kernel_separable(
                kernel, tau, nat, fine_frac, coarse_frac)
        self._atom_fourier_kernel = kernel

    def _build_atom_fourier_kernel_separable(self, kernel, tau, nat,
                                             fine_frac, coarse_frac):
        """Legacy separable per-axis, metric-free kernel (orthorhombic)."""
        for iq, q in enumerate(fine_frac):
            for ik, x in enumerate(coarse_frac):
                dx = q - x
                for ia in range(nat):
                    for ib in range(nat):
                        d = tau[ia] - tau[ib]
                        value = 1.0 + 0.0j
                        for axis in range(3):
                            cardinal = 0.0j
                            nmesh = int(self.coarse_mesh[axis])
                            for alias in range(nmesh):
                                harmonic = 0.0j
                                for R, weight in self._nearest_alias_images(
                                        alias, d[axis], nmesh):
                                    harmonic += weight * np.exp(
                                        2j * np.pi * dx[axis] * R)
                                cardinal += harmonic
                            value *= cardinal / float(nmesh)
                        kernel[iq, ik, ia, ib] = value

    def _build_atom_fourier_kernel_metric(self, kernel, tau, nat,
                                          fine_frac, coarse_frac):
        """Non-separable kernel with the true 3D metric minimal image.

        The image assignment depends only on the atom pair (a,b), not on
        q, so it is computed once per pair and reused over all (q,k).
        """
        A = np.asarray(self.uci_structure.unit_cell, dtype=np.float64)
        metric = A @ A.T
        Nc = self.coarse_mesh
        norm = 1.0 / float(np.prod(Nc))

        # For each atom pair the image assignment depends only on (a,b), so
        # the phase exp(2 pi i (q - x).R) factorizes as
        #   exp(2 pi i q.R) * conj(exp(2 pi i x.R)),
        # separable in the fine index q and the coarse index x.  The
        # per-pair kernel is then a single matmul over the images R:
        #   K[iq, ik] = (1/Nc) sum_R w_R E_fine[iq, R] conj(E_coarse[ik, R]).
        for ia in range(nat):
            for ib in range(nat):
                d = tau[ia] - tau[ib]
                imgs = self._metric_alias_images(d, Nc, metric)
                Rs, ws = [], []
                for entries in imgs.values():
                    for R, w in entries:
                        Rs.append(R)
                        ws.append(w)
                Rs = np.asarray(Rs, dtype=np.float64)          # (nR, 3)
                ws = np.asarray(ws, dtype=np.float64)          # (nR,)
                e_fine = np.exp(2j * np.pi * (fine_frac @ Rs.T))    # (n_q, nR)
                e_coarse = np.exp(2j * np.pi * (coarse_frac @ Rs.T))  # (cn_q,)
                kernel[:, :, ia, ib] = norm * (
                    (e_fine * ws[None, :]) @ e_coarse.conj().T)

    def _atom_fourier_cart_kernel(self, iq, ik):
        """Expand the cached per-atom kernel over Cartesian components."""
        return np.repeat(np.repeat(
            self._atom_fourier_kernel[iq, ik], 3, axis=0), 3, axis=1)

    # ================================================================
    # Pair maps (fine for psi, coarse for the kernel)
    # ================================================================
    def build_q_pair_map(self, iq_pert):
        """Fine and coarse pair maps for a perturbation at fine index iq_pert.

        The perturbation momentum must lie on the coarse mesh (paper
        assumption: only the internal loop q' is refined).
        """
        q_pert = self.q_points[iq_pert]
        try:
            key_c = _mesh_key(q_pert, self.uci_structure, self.coarse_mesh)
        except ValueError:
            raise ValueError(
                "The perturbation q-point {} must lie on the coarse mesh "
                "{} (only the internal q' loop is interpolated)".format(
                    q_pert, tuple(self.coarse_mesh)))

        self.iq_pert = iq_pert

        # Fine pair map by integer mesh arithmetic
        mesh = self.fine_mesh
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

        # Fold plan: for each fine q' of the full BZ, the stored unique
        # pair holding it and whether it enters as the transpose (reverse
        # orientation, bilinear convention).
        self._fine_pair_of = [None] * self.n_q
        for p, (i1, i2) in enumerate(self.unique_pairs):
            self._fine_pair_of[i1] = (p, False)
            if i2 != i1:
                self._fine_pair_of[i2] = (p, True)

        # Coarse pair map (kernel side)
        n_pert_c = np.array(key_c, dtype=int)
        self.c_iq_pert = self._coarse_lookup[key_c]
        self.c_q_pair_map = np.zeros(self.cn_q, dtype=np.int32)
        for ik in range(self.cn_q):
            n2 = tuple((n_pert_c - np.array(self._coarse_idx[ik]))
                       % self.coarse_mesh)
            self.c_q_pair_map[ik] = self._coarse_lookup[n2]

        self.c_unique_pairs = []
        for ik1 in range(self.cn_q):
            ik2 = int(self.c_q_pair_map[ik1])
            if ik1 <= ik2:
                self.c_unique_pairs.append((ik1, ik2))

    # ================================================================
    # Symmetrization: the kernel rotates COARSE fields
    # ================================================================
    def prepare_symmetrization(self, no_sym=False, verbose=True,
                               symmetries=None):
        """Build the sparse symmetry matrices in the coarse mode basis.

        The Julia kernel rotates the coarse ensemble fields; the fold
        geometry is fixed under the ensemble symmetrization, exactly like
        the alpha1 blocks in the commensurate calculation.
        """
        fine = (self.q_points, self.n_q, self.w_q, self.pols_q,
                self.valid_modes_q)
        self.q_points, self.n_q = self.cq_points, self.cn_q
        self.w_q, self.pols_q = self.cw_q, self.cpols_q
        self.valid_modes_q = self.cvalid_modes_q
        try:
            super().prepare_symmetrization(no_sym=no_sym, verbose=verbose,
                                           symmetries=symmetries)
        finally:
            (self.q_points, self.n_q, self.w_q, self.pols_q,
             self.valid_modes_q) = fine

    # ================================================================
    # Fold / unfold (all cross-q mixing in Cartesian)
    # ================================================================
    # Basis transforms (E = pols, unitary; from x = u^T conj(E)):
    #   mode -> Cartesian:  M_cart = E1 @ M_mode @ E2.T
    #   Cartesian -> mode:  M_mode = E1.conj().T @ M_cart @ E2.conj()
    # The same pair applies to the alpha kernel (contracted with
    # conjugated fields) and to the d2v dyadics (unconjugated components).

    def _get_alpha1_bare(self):
        """Fine-pair perturbation blocks with the chi dressing but WITHOUT
        the w1*w2/X = 2 f_Y(q1) f_Y(q2) vertex filter.

        The filter is the coarse-Upsilon part of the interpolated vertex
        (w1 w2 / X * conj(x) conj(x) = 2 conj(Upsilon u) conj(Upsilon u)):
        the paper prescribes evaluating the Upsilon u legs AT THE CORNERS
        with the COARSE Upsilon, so the filter is applied to the folded
        kernel (_fold_alpha_to_coarse) with the coarse f_Y tables. Applying
        it here with the fine tables (as the parent's
        get_alpha1_beta1_wigner_q does) breaks exact per-configuration
        Hermiticity: the kernel's output dyadics carry f_Y at the corners,
        and the fine/coarse mismatch makes the D4 element asymmetric.
        """
        chi_minus_list = self.get_chi_minus_q()
        chi_plus_list = self.get_chi_plus_q()
        blocks = []
        for pair_idx in range(len(self.unique_pairs)):
            a_block = self.get_a1_block(pair_idx)
            b_block = self.get_b1_block(pair_idx)
            alpha = 2.0 * (
                np.sqrt(-0.5 * chi_minus_list[pair_idx]) * a_block
                - np.sqrt(+0.5 * chi_plus_list[pair_idx]) * b_block)
            blocks.append(alpha)
        return blocks

    def _get_fy_coarse(self):
        """Coarse f_Y = 2w/(1+2n) table (nb, cn_q); masked modes -> 0.

        Same definition as the Julia kernel's internal table.
        """
        fy = np.zeros((self.n_bands, self.cn_q), dtype=np.float64)
        for ik in range(self.cn_q):
            w = self.cw_q[:, ik]
            valid = self.cvalid_modes_q[:, ik]
            n = np.zeros_like(w)
            if self.T > QL.__EPSILON__:
                n[valid] = 1.0 / (np.exp(
                    w[valid] * QL.__RyToK__ / self.T) - 1.0)
            fy[valid, ik] = 2.0 * w[valid] / (1.0 + 2.0 * n[valid])
        return fy

    def _fold_alpha_cart(self, alpha1_fine):
        """Fold the fine-pair (bare) alpha kernel onto the coarse mesh
        (Cartesian).

        Returns A(cn_q, nb, nb): A[k] is the kernel block whose first leg
        is at coarse k (pairing with Q - k). Enumerates the full fine BZ,
        so A(Q - k) = A(k)^T holds exactly (corner-mirror property).
        """
        nb = self.n_bands
        nat = nb // 3

        if self.atom_fourier:
            # Vectorized fold: build every fine Cartesian block, then
            # contract with the (atom-resolved) cardinal kernel in one
            # einsum.  A[ik][a,i,b,j] = sum_iq conj(P[iq,ik,a,b])
            #                                    * blk_cart[iq][a,i,b,j].
            blk_all = np.empty((self.n_q, nb, nb), dtype=np.complex128)
            for iq in range(self.n_q):
                p, is_transpose = self._fine_pair_of[iq]
                blk = alpha1_fine[p]
                if is_transpose:
                    blk = blk.T
                if2 = int(self.q_pair_map[iq])
                blk_all[iq] = (self.pols_q[:, :, iq] @ blk
                               @ self.pols_q[:, :, if2].T)
            blk_r = blk_all.reshape(self.n_q, nat, 3, nat, 3)
            A_r = np.einsum('qkab,qaibj->kaibj',
                            np.conj(self._atom_fourier_kernel), blk_r,
                            optimize=True)
            return A_r.reshape(self.cn_q, nb, nb)

        A = np.zeros((self.cn_q, nb, nb), dtype=np.complex128)
        for iq in range(self.n_q):
            p, is_transpose = self._fine_pair_of[iq]
            blk = alpha1_fine[p]
            if is_transpose:
                blk = blk.T
            if2 = int(self.q_pair_map[iq])
            blk_cart = self.pols_q[:, :, iq] @ blk @ self.pols_q[:, :, if2].T
            for ik, w, delta in self._corners[iq]:
                mapped = blk_cart
                if self.atomic_phase:
                    # Leg 1 moves by +delta (fine -> corner), leg 2 (at
                    # Q - q) by -delta: the transport phases pc* pf per
                    # leg depend only on the local displacement.
                    left = self._atomic_delta_phase(delta)
                    right = left.conj()
                    mapped = left[:, None] * blk_cart * right[None, :]
                A[ik] += w * mapped
        return A

    def _fold_alpha_to_coarse(self, alpha1_fine):
        """Folded alpha blocks in the coarse mode basis, ordered as
        c_unique_pairs (the Julia kernel input).

        The coarse f_Y filter is applied on both legs AFTER the fold: this is
        the coarse Upsilon on the interpolated Upsilon-u legs of the paper.
        In the commensurate limit f_Y(k1) f_Y(k2) * bare = (w1 w2 / X) *
        dressed, i.e. exactly the parent's alpha1.
        """
        A = self._fold_alpha_cart(alpha1_fine)
        fy = self._get_fy_coarse()
        out = []
        for ik1, ik2 in self.c_unique_pairs:
            E1 = self.cpols_q[:, :, ik1]
            E2 = self.cpols_q[:, :, ik2]
            A_mode = E1.conj().T @ A[ik1] @ E2.conj()
            A_mode = (fy[:, ik1][:, None] * A_mode
                      * fy[:, ik2][None, :])
            out.append(A_mode)
        return out

    def _interp_d2v_to_fine(self, d2v_coarse):
        """Adjoint interpolation of coarse d2v to the fine mode basis.

        The atomic-gauge option applies the exact paired return
        transformation corresponding to _fold_alpha_cart.
        """
        nb = self.n_bands
        nat = nb // 3
        D = np.zeros((self.cn_q, nb, nb), dtype=np.complex128)
        for p, (ik1, ik2) in enumerate(self.c_unique_pairs):
            E1 = self.cpols_q[:, :, ik1]
            E2 = self.cpols_q[:, :, ik2]
            Dc = E1 @ d2v_coarse[p] @ E2.T
            D[ik1] = Dc
            if ik1 != ik2:
                # Reverse orientation: the kernel's dyadic weights are
                # pair-symmetric, so the reversed block is the transpose.
                D[ik2] = Dc.T
        if self.atom_fourier:
            # Vectorized unfold: B[p][a,i,b,j] = sum_ik P[iq1(p),ik,a,b]
            #                                          * D[ik][a,i,b,j].
            iq1_list = np.array([iq1 for iq1, _ in self.unique_pairs])
            D_r = D.reshape(self.cn_q, nat, 3, nat, 3)
            B_r = np.einsum('pkab,kaibj->paibj',
                            self._atom_fourier_kernel[iq1_list], D_r,
                            optimize=True)
            B_all = B_r.reshape(len(self.unique_pairs), nb, nb)
            fine = []
            for p, (iq1, iq2) in enumerate(self.unique_pairs):
                E1 = self.pols_q[:, :, iq1]
                E2 = self.pols_q[:, :, iq2]
                fine.append(E1.conj().T @ B_all[p] @ E2.conj())
            return fine

        fine = []
        for iq1, iq2 in self.unique_pairs:
            B = np.zeros((nb, nb), dtype=np.complex128)
            for ik, w, delta in self._corners[iq1]:
                mapped = D[ik]
                if self.atomic_phase:
                    # Adjoint of the fold transport: conjugate phases.
                    left = self._atomic_delta_phase(delta).conj()
                    right = left.conj()
                    mapped = left[:, None] * D[ik] * right[None, :]
                B += w * mapped
            E1 = self.pols_q[:, :, iq1]
            E2 = self.pols_q[:, :, iq2]
            fine.append(E1.conj().T @ B @ E2.conj())
        return fine

    # ================================================================
    # Anharmonic application
    # ================================================================
    def apply_anharmonic_FT(self, transpose=False, **kwargs):
        """Fold the fine alpha kernel to the coarse mesh, run the coarse
        Julia kernel, and interpolate the outputs back to the fine pairs."""
        if self.ignore_v3 and self.ignore_v4:
            return np.zeros(self.get_psi_size(), dtype=np.complex128)

        R1 = self.get_R1_q()
        if self.ignore_v3:
            R1 = np.zeros_like(R1)

        alpha1_fine = self._get_alpha1_bare()
        alpha_coarse = self._fold_alpha_to_coarse(alpha1_fine)
        alpha_flat = self._flatten_blocks(alpha_coarse)

        f_pert, d2v_coarse = self._call_julia_qspace_coarse(R1, alpha_flat)
        d2v_fine = self._interp_d2v_to_fine(d2v_coarse)

        final_psi = np.zeros(self.get_psi_size(), dtype=np.complex128)
        final_psi[:self.n_bands] = f_pert

        chi_minus_list = self.get_chi_minus_q()
        chi_plus_list = self.get_chi_plus_q()
        for pair_idx in range(len(self.unique_pairs)):
            d2v_block = d2v_fine[pair_idx]
            pert_a = np.sqrt(-0.5 * chi_minus_list[pair_idx]) * d2v_block
            pert_b = -np.sqrt(+0.5 * chi_plus_list[pair_idx]) * d2v_block
            self.set_block_in_psi(pair_idx, pert_a, 'a', final_psi)
            self.set_block_in_psi(pair_idx, pert_b, 'b', final_psi)

        return final_psi

    def _unflatten_blocks_coarse(self, flat):
        """Column-major unflatten over the coarse unique pairs."""
        nb = self.n_bands
        blocks = []
        offset = 0
        for _ in self.c_unique_pairs:
            blocks.append(flat[offset:offset + nb * nb].reshape(
                nb, nb, order='F'))
            offset += nb * nb
        return blocks

    def _call_julia_qspace_coarse(self, R1, alpha1_flat):
        """Parent's kernel call with the coarse-side arrays.

        R1 is in the mode basis at Q, identical on the fine and coarse
        sides by the commensurate pinning of the eigenvectors.
        """
        if self._distributed:
            return self._call_julia_qspace_coarse_distributed(R1, alpha1_flat)

        jl = JuliaExt.get_main()

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

        unique_pairs_arr = np.array(self.c_unique_pairs,
                                    dtype=np.int32) + 1
        valid_modes = np.array(self.cvalid_modes_q, dtype=np.bool_)
        iq_pert_jl = int(self.c_iq_pert) + 1
        q_pair_map_jl = np.array(self.c_q_pair_map, dtype=np.int32) + 1

        def get_combined(start_end):
            return jl.get_perturb_averages_qspace(
                self.X_q, self.Y_q, self.cw_q, self.rho,
                R1, alpha1_flat,
                float(self.T), bool(not self.ignore_v4),
                iq_pert_jl,
                q_pair_map_jl,
                unique_pairs_arr,
                int(start_end[0]), int(start_end[1]),
                valid_modes,
                float(self.qspace_scale3), float(self.qspace_scale4),
                False)

        combined = Parallel.GoParallel(get_combined, indices, "+")
        f_pert = combined[:self.n_bands]
        d2v_blocks = self._unflatten_blocks_coarse(combined[self.n_bands:])
        return f_pert, d2v_blocks

    def _call_julia_qspace_coarse_distributed(self, R1, alpha1_flat):
        """Coarse kernel call when the configurations are split across ranks.

        The coarse counterpart of
        ``QSpaceLanczos._call_julia_qspace_distributed``: instead of every rank
        walking a slice of the *same* replicated ensemble through
        ``GoParallel``, each rank owns a disjoint block of configurations and
        the partial averages are summed with an Allreduce.

        Normalization (identical to the parent, restated because it is the
        only subtle part): Julia returns ``partial_k / (n_syms * N_eff_k)``,
        so multiplying by the local ``N_eff_k`` gives ``partial_k / n_syms``;
        summing those over ranks and dividing by the global ``N_eff`` gives
        the correctly weighted average.  Rescaling by the *local* weight and
        dividing by the *global* one is what makes unequal rank loads and
        unequal weight sums come out right.
        """
        jl = JuliaExt.get_main()

        if not QL.__MPI4PY__:
            raise RuntimeError(
                "Distributed mode requires MPI (mpi4py). Use "
                "load_distributed_trilinear_tdscha under mpirun, or build the "
                "Lanczos from an ensemble for a replicated run.")

        comm = QL.mpi4py.MPI.COMM_WORLD

        N_local = self.N
        N_eff_local = self.N_eff
        N_eff_global = self._N_eff_global

        n_blocks = len(self.c_unique_pairs) * self.n_bands * self.n_bands
        if N_local == 0:
            # A rank with no configurations still has to take part in the
            # reduction, otherwise the others block forever.
            combined_local = np.zeros(self.n_bands + n_blocks,
                                      dtype=np.complex128)
        else:
            unique_pairs_arr = np.array(self.c_unique_pairs,
                                        dtype=np.int32) + 1
            valid_modes = np.array(self.cvalid_modes_q, dtype=np.bool_)
            iq_pert_jl = int(self.c_iq_pert) + 1
            q_pair_map_jl = np.array(self.c_q_pair_map, dtype=np.int32) + 1

            combined_local = jl.get_perturb_averages_qspace(
                self.X_q, self.Y_q, self.cw_q, self.rho[:N_local],
                R1, alpha1_flat,
                float(self.T), bool(not self.ignore_v4),
                iq_pert_jl,
                q_pair_map_jl,
                unique_pairs_arr,
                1, int(self.n_syms_qspace * N_local),
                valid_modes,
                float(self.qspace_scale3), float(self.qspace_scale4),
                False)
            if N_eff_local > 0:
                combined_local = combined_local * N_eff_local

        combined_global = np.zeros_like(combined_local)
        comm.Allreduce(np.ascontiguousarray(combined_local), combined_global,
                       op=QL.mpi4py.MPI.SUM)
        if N_eff_global > 0:
            combined_global = combined_global / N_eff_global

        f_pert = combined_global[:self.n_bands]
        d2v_blocks = self._unflatten_blocks_coarse(combined_global[self.n_bands:])
        return f_pert, d2v_blocks


def load_distributed_trilinear_tdscha(data_dir, population_id, dyn, T,
                                      fine_mesh=None, use_symmetries=True,
                                      n_configs=None, final_dyn=None,
                                      final_T=None, **kwargs):
    """Distributed-ensemble loader for the *interpolated* q-space Lanczos.

    Same contract as ``QSpaceLanczos.load_distributed_tdscha`` -- the ensemble
    is read on the master and the configurations are scattered, so each rank
    holds only N/n_procs of them instead of a full replica -- but the object
    returned is a :class:`QSpaceTrilinearLanczos`.

    The interpolation is built redundantly on every rank
    (``build_on_all_ranks=True``) and only then are the configurations split.
    That is not an optimisation choice: ``interpolate_dyn_fine`` calls
    ``ForceTensor.Apply_ASR``, which broadcasts, so a master-only construction
    deadlocks against the workers waiting in the metadata broadcast.  Building
    everywhere keeps every collective matched, and the ensemble is replicated
    only during construction -- the steady state each rank carries into the
    Lanczos is its own N/n_procs slice of X_q/Y_q.

    Parameters
    ----------
    fine_mesh : tuple(3) of int
        The interpolation mesh, e.g. ``(12, 12, 12)``.
    **kwargs
        Forwarded to :class:`QSpaceTrilinearLanczos` (``atom_fourier``,
        ``ignore_effective_charges``, ``allow_unstable``, ...).

    Usage
    -----
        mpirun -np 4 python driver.py
    """
    return QL.load_distributed_tdscha(
        data_dir, population_id, dyn, T,
        lo_to_split=None, use_symmetries=use_symmetries,
        n_configs=n_configs, final_dyn=final_dyn, final_T=final_T,
        lanczos_class=QSpaceTrilinearLanczos, build_on_all_ranks=True,
        fine_mesh=fine_mesh, **kwargs)
