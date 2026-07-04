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
3. Vertex renormalization: mode-space d3 ~ N^-1/2 and d4 ~ N^-1, so the
   coarse-ensemble averages are rescaled by scale3 = sqrt(N_c/N_f) (D3) and
   scale4 = N_c/N_f (D4) to represent the fine-mesh Lanczos operator.
4. Field PRE-FILTERING (essential, plan section 5.6): f_Y = 2w/(1+2n) is
   applied on the coarse grid per configuration, which exactly strips the
   phonon-propagator dressing of every displacement leg (Gaussian
   integration by parts); the interpolated correlations then decay with the
   range of the anharmonic force constants themselves.
5. Acoustic sum rule (plan section 5.5): the (window-weighted) translation
   zero-modes are projected out of the fields, so every leg of the
   effective D3/D4 kernels vanishes on acoustic modes as q -> 0.
6. Designed multitaper windows ("stochastic centering", plan sections
   5.2-5.5, window_design="minimal_image"): the D3 estimator is evaluated
   on a small set of windowed field passes whose per-dimension window
   triples are fitted (ALS with partition-of-unity constraints) so that the
   effective interpolation kernel approximates the minimal-image
   (zero-padded) centering of ForceTensor -- without ever forming tensors.
   The D4 terms use the plain full-period window (shorter range, smaller
   N_c/N_f weight).

Limitations (phase 1):
- LO-TO splitting / effective charges are not applied to the interpolated
  dynamical matrix (lo_to_split must be None).
- prepare_ir / prepare_raman inherit the coarse sqrt(N_c) Gamma-amplitude
  prefactor; only the overall intensity scale is affected, not the shape.
- The distributed (MPI config-sliced) loader is not yet wired for this
  class; standard GoParallel parallelism over (config, sym) works.
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

import cellconstructor.Settings as Parallel
from cellconstructor.Settings import ParallelPrint as print

import tdscha.QSpaceLanczos as QL
import tdscha.JuliaExt as JuliaExt


__EPSILON__ = 1e-12


# =========================================================================
# Fine mesh utilities
# =========================================================================

def generate_fine_mesh(uc_structure, mesh):
    """Generate a Gamma-centered uniform q-mesh (Gamma first).

    Returns
    -------
    q_points : ndarray(N_f, 3)
        Cartesian q-points in the same units as dyn.q_tot (2pi/A free).
    idx : ndarray(N_f, 3), int
        Integer mesh indices n of each point (q_frac = n / mesh).
    """
    mesh = np.asarray(mesh, dtype=int)
    assert np.all(mesh > 0), "Invalid mesh {}".format(mesh)

    bg = uc_structure.get_reciprocal_vectors() / (2 * np.pi)

    idx = np.array(list(itertools.product(range(mesh[0]),
                                          range(mesh[1]),
                                          range(mesh[2]))), dtype=int)
    frac = idx / mesh[None, :]
    frac = frac - np.floor(frac + 0.5)

    q_points = frac @ bg
    return q_points, idx


def build_q_index_lookup(q_points, uc_structure, mesh, tol=1e-6):
    """Build an O(1) lookup {mesh index tuple -> position in q_points}."""
    mesh = np.asarray(mesh, dtype=int)
    lookup = {}
    for iq, q in enumerate(q_points):
        lookup[_mesh_key(q, uc_structure, mesh, tol)] = iq
    return lookup


def _mesh_key(q, uc_structure, mesh, tol=1e-6):
    """Integer mesh-index key of a q-point (modulo reciprocal lattice)."""
    frac = uc_structure.unit_cell @ np.asarray(q)
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

    ForceTensor.Tensor2 with real-space centering and (optionally) the
    iterative acoustic sum rule; DyagDinQ mass conventions; time-reversal
    gauge e(-q) = conj(e(q)) enforced between +-q partners.

    Returns (w_q(nb, N_f) [Ry], pols_q(3nat, nb, N_f) complex).
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

    commensurate_of = np.full(n_q, -1, dtype=int)
    if reuse_commensurate:
        for iq, q in enumerate(q_points):
            for jq, qc in enumerate(dyn.q_tot):
                if CC.Methods.get_min_dist_into_cell(bg, np.asarray(q), np.asarray(qc)) < 1e-6:
                    commensurate_of[iq] = jq
                    break

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
        if minus_of[iq] == iq:
            D = np.real(D)

        eigvals, eigvects = np.linalg.eigh(D)
        w_q[:, iq] = np.sign(eigvals) * np.sqrt(np.abs(eigvals))
        pols_q[:, :, iq] = eigvects
        done[iq] = True

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
# Window design toolbox ("stochastic centering", plan sections 5.2-5.5)
# =========================================================================
#
# Per lattice dimension of length L, the D3 estimator pass with windows
# (z, w, v) on the three field slots has the effective interpolation kernel
#
#   S_{z,w,v}(d1, d2) = sum_s z(s) w(s+d1) v(s+d2)        (zero-padded)
#
# on the correlation differences d1, d2 in [-(L-1), L-1]. The PLAIN
# estimator is the single pass z = w = v = ones(L), whose kernel is the
# "tent": weight L - spread on each image, summing to L over every
# difference class (partition of unity => exact at commensurate q).
# The design below fits K passes so that  sum_r S_r  approximates
# L * M(d1, d2), where M is the minimal-image indicator with tie splits
# (the separable analogue of ForceTensor centering), under the hard
# partition constraint  sum_images sum_r S_r = L  (exact commensurate
# limit preserved). w <-> v symmetry is built into the fitted kernel and
# realized at runtime by averaging the two pass orientations.

def minimal_image_target_1d(L):
    """Target kernel L * M on the (2L-1)^2 difference grid.

    M assigns each difference class (d1, d2) mod L to its minimal-spread
    image(s) (spread = max(0,d1,d2) - min(0,d1,d2)), splitting ties."""
    n = 2 * L - 1
    target = np.zeros((n, n))

    def spread(d1, d2):
        return max(0, d1, d2) - min(0, d1, d2)

    for d1 in range(L):
        for d2 in range(L):
            imgs = [(d1 + a * L, d2 + b * L) for a in (-1, 0) for b in (-1, 0)]
            sp = [spread(*im) for im in imgs]
            mn = min(sp)
            winners = [im for im, s in zip(imgs, sp) if s == mn]
            for im in winners:
                target[im[0] + L - 1, im[1] + L - 1] = float(L) / len(winners)
    return target


def triple_kernel_1d(z, w, v):
    """S(d1,d2) = sum_s z(s) w(s+d1) v(s+d2), windows zero-padded outside
    their length-L support. Returns ndarray(2L-1, 2L-1)."""
    L = len(z)
    emb = 3 * L
    Z = np.zeros(emb); Z[L:2 * L] = z
    W = np.zeros(emb); W[L:2 * L] = w
    V = np.zeros(emb); V[L:2 * L] = v
    n = 2 * L - 1
    S = np.zeros((n, n))
    for i, d1 in enumerate(range(-(L - 1), L)):
        for j, d2 in enumerate(range(-(L - 1), L)):
            s0 = max(0, -d1, -d2)
            s1 = min(emb, emb - d1, emb - d2)
            S[i, j] = np.dot(Z[s0:s1] * W[s0 + d1:s1 + d1], V[s0 + d2:s1 + d2])
    return S


def _kernel_sym(z, w, v):
    """w <-> v symmetrized kernel (matches the runtime orientation average)."""
    S = triple_kernel_1d(z, w, v)
    return 0.5 * (S + triple_kernel_1d(z, v, w).T)


def _partition_residual(K_tot, L):
    """Deviations of the per-class image sums from L."""
    out = []
    for d1 in range(L):
        for d2 in range(L):
            tot = 0.0
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    i, j = d1 + a * L + L - 1, d2 + b * L + L - 1
                    if 0 <= i < 2 * L - 1 and 0 <= j < 2 * L - 1:
                        tot += K_tot[i, j]
            out.append(tot - L)
    return np.array(out)


def design_windows_1d(L, K=3, iters=300, n_restarts=4, seed=0,
                      partition_weight=1.0, verbose=False):
    """Fit K window passes (z_r, w_r, v_r) per dimension by ALS.

    Minimizes || sum_r S_sym(z_r, w_r, v_r) - L*M ||^2 with a strong
    penalty on the partition-of-unity residuals. Pass 0 is seeded with the
    plain window (guaranteed feasible point: the fit can only improve on
    the tent kernel). Returns a list of (z, w, v) float arrays; the design
    coefficient is folded into z.
    """
    if L == 1:
        return [(np.ones(1), np.ones(1), np.ones(1))]
    if L == 2:
        # All nonzero differences are Wigner-Seitz ties: the tent kernel IS
        # the minimal-image kernel. Plain window is exact.
        return [(np.ones(2), np.ones(2), np.ones(2))]

    target = minimal_image_target_1d(L)
    n_pts = (2 * L - 1) ** 2

    def model(passes, skip=None):
        M = np.zeros_like(target)
        for r, (z, w, v) in enumerate(passes):
            if r == skip:
                continue
            M += _kernel_sym(z, w, v)
        return M

    def fit_slot(passes, r, slot):
        """LSQ over the slot values (kernel is linear in each slot)."""
        resid = target - model(passes, skip=r)
        A = np.zeros((n_pts + L * L, L))
        b = np.zeros(n_pts + L * L)
        b[:n_pts] = resid.ravel()
        # partition rows: target value is L minus other passes' class sums
        part_other = _partition_residual(model(passes, skip=r), L) + L
        for k in range(L):
            trial = [np.array(x) for x in passes[r]]
            trial[slot] = np.zeros(L)
            trial[slot][k] = 1.0
            Sk = _kernel_sym(*trial)
            A[:n_pts, k] = Sk.ravel()
            # class sums of the basis kernel (partition_residual returns
            # classsum - L, which is only meaningful for a TOTAL kernel)
            A[n_pts:, k] = partition_weight * (_partition_residual(Sk, L) + L)
        b[n_pts:] = partition_weight * (L - part_other)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        new = [np.array(x) for x in passes[r]]
        new[slot] = sol
        return tuple(new)

    best = None
    best_err = np.inf
    for trial in range(n_restarts):
        rs = np.random.RandomState(1000 * L + 17 * K + trial + seed)
        passes = [tuple(rs.randn(L) * 0.4 for _ in range(3)) for _ in range(K)]
        # seed pass 0 with the plain window
        passes[0] = (np.ones(L), np.ones(L), np.ones(L))
        # ALS with a penalty ramp: fit freely first (the target satisfies the
        # partition, so a good fit inherits it), then polish the partition.
        schedule = [(iters, partition_weight),
                    (iters // 4, 10 * partition_weight),
                    (iters // 4, 100 * partition_weight)]
        pw_backup = partition_weight
        for n_it, pw in schedule:
            partition_weight = pw
            for it in range(n_it):
                for r in range(K):
                    for slot in range(3):
                        passes[r] = fit_slot(passes, r, slot)
        partition_weight = pw_backup
        Ktot = model(passes)
        err = np.sqrt(np.mean((Ktot - target) ** 2))
        perr = np.max(np.abs(_partition_residual(Ktot, L)))
        score = err + 10.0 * perr
        if score < best_err:
            best_err = score
            best = ([tuple(np.array(x) for x in p) for p in passes], err, perr)

    passes, err, perr = best
    if verbose:
        print("window design L={} K={}: kernel RMS err {:.2e}, "
              "partition dev {:.2e}".format(L, K, err, perr))
    if perr > 1e-6 * L:
        warnings.warn("Window design (L={}, K={}) violates partition of "
                      "unity by {:.2e}: the commensurate limit is only "
                      "approximate. Increase K or iters.".format(L, K, perr))
    return passes


# Module-level cache of designs
_WINDOW_DESIGN_CACHE = {}


def get_window_design(L, K=3, max_K=5):
    """Cached design; escalates K until the kernel is accurate enough."""
    key = (int(L), int(K))
    if key not in _WINDOW_DESIGN_CACHE:
        target = minimal_image_target_1d(L)
        best = None
        for K_try in range(K, max_K + 1):
            passes = design_windows_1d(L, K=K_try)
            Ktot = sum(_kernel_sym(*p) for p in passes)
            err = np.sqrt(np.mean((Ktot - target) ** 2)) / max(L, 1)
            best = passes
            if err < 1e-6:
                break
        _WINDOW_DESIGN_CACHE[key] = best
    return _WINDOW_DESIGN_CACHE[key]


# =========================================================================
# ASR-exact window design (plan section 5.7)
#
# Doubled-support (2L) windows constrained to EXACTLY uniform class sums,
#     w(d) + w(d + L) = c_w        for all d,
# fitted to the projection of the minimal-image target onto the acoustic-
# sum-rule subspace. On this manifold all three ASR conditions of the pair
# kernel (image-sum constancy on the w, v and z legs) and the partition of
# unity (which collapses to the single scalar sum_r c_z c_w c_v = 1,
# enforced as a hard KKT constraint) hold at machine precision, so the
# centering, the ASR, the commensurate limit and the w<->v symmetrization
# are obtained in ONE constrained fit -- no iterative ASR/symmetrization
# alternation as in ForceTensor.Tensor3.Apply_ASR.
# =========================================================================

def _spread_ext(d1, d2):
    """Three-leg spread of an extended difference pair (z leg at 0)."""
    return max(0, d1, d2) - min(0, d1, d2)


def embed_minimal_image_target_1d(L, S):
    """Minimal-image target L*M embedded in the (2S-1)^2 difference grid."""
    tgt_small = minimal_image_target_1d(L)
    n = 2 * S - 1
    tgt = np.zeros((n, n))
    off = S - L
    tgt[off:off + 2 * L - 1, off:off + 2 * L - 1] = tgt_small
    return tgt


def _class_images_gen(L, S):
    """Flat indices of the periodic images of each class (d1, d2) in [0,L)^2
    on the (2S-1)^2 extended difference grid."""
    n = 2 * S - 1
    rng = range(-(S - 1) // L - 1, (S - 1) // L + 2)
    out = {}
    for d1 in range(L):
        for d2 in range(L):
            idx = []
            for a in rng:
                for b in rng:
                    i, j = d1 + a * L + S - 1, d2 + b * L + S - 1
                    if 0 <= i < n and 0 <= j < n:
                        idx.append(i * n + j)
            out[(d1, d2)] = idx
    return out


def _asr_constraint_rows(L, S):
    """Linear ASR + partition constraints C k = d on the flattened kernel.

    ASR-v : for each extended row delta1, the image sums over the L
            d2-classes are equal (difference rows);
    ASR-w : transpose of ASR-v;
    ASR-z : per-diagonal-class image sums equal (the sum over the first
            tensor index runs along diagonals in difference coordinates);
    partition: every class total equals L (commensurate exactness).

    These are exactly the conditions under which the effective centered
    kernel inherits the acoustic sum rules from ANY periodic tensor that
    satisfies them (plan section 5.7a).
    """
    n = 2 * S - 1
    N = n * n
    n_img = (S - 1) // L + 2
    rows, rhs = [], []

    def img_v(i, d2):
        return [i * n + j for j in
                (d2 + b * L + S - 1 for b in range(-n_img, n_img + 1))
                if 0 <= j < n]

    for i in range(n):
        base = img_v(i, 0)
        for d2 in range(1, L):
            r = np.zeros(N)
            for c in base:
                r[c] += 1.0
            for c in img_v(i, d2):
                r[c] -= 1.0
            rows.append(r); rhs.append(0.0)
    for j in range(n):
        def img_w(d1):
            return [i * n + j for i in
                    (d1 + a * L + S - 1 for a in range(-n_img, n_img + 1))
                    if 0 <= i < n]
        base = img_w(0)
        for d1 in range(1, L):
            r = np.zeros(N)
            for c in base:
                r[c] += 1.0
            for c in img_w(d1):
                r[c] -= 1.0
            rows.append(r); rhs.append(0.0)
    for Delta in range(-(n - 1), n):
        def img_diag(d):
            out = []
            for b in range(-n_img, n_img + 1):
                i, j = d + b * L + S - 1, d + Delta + b * L + S - 1
                if 0 <= i < n and 0 <= j < n:
                    out.append(i * n + j)
            return out
        classes = [c for c in (img_diag(d) for d in range(L)) if c]
        for k in range(1, len(classes)):
            r = np.zeros(N)
            for c in classes[0]:
                r[c] += 1.0
            for c in classes[k]:
                r[c] -= 1.0
            rows.append(r); rhs.append(0.0)
    cls = _class_images_gen(L, S)
    for key in sorted(cls):
        r = np.zeros(N)
        for c in cls[key]:
            r[c] += 1.0
        rows.append(r); rhs.append(float(L))
    return np.array(rows), np.array(rhs)


def asr_projected_target_1d(L, S, decay_weight=None):
    """Projection of the minimal-image target onto the ASR subspace.

    The exact minimal-image kernel VIOLATES the acoustic sum rules (the
    kernel-space restatement of 'centering destroys the sum rule', which
    is why ForceTensor must run Apply_ASR after Center). The closest
    ASR-compatible kernel is this affine projection; its distance from
    L*M is the irreducible price of an exact ASR at compact support.

    decay_weight : float in (0, 1] or None
        Metric weight rho**spread(d1, d2) for the projection. Weighting
        concentrates kernel fidelity at small spreads (where the physical
        Phi3 is large), pushing the irreducible residual to differences
        where the tensor has decayed -- the analogue of the `power`
        parameter of Tensor3.Apply_ASR. None or 1.0 = unweighted.
    """
    tgt = embed_minimal_image_target_1d(L, S).ravel()
    C, d = _asr_constraint_rows(L, S)
    if decay_weight is None or decay_weight >= 1.0:
        G = C @ C.T
        lam = np.linalg.lstsq(G, d - C @ tgt, rcond=None)[0]
        k = tgt + C.T @ lam
    else:
        n = 2 * S - 1
        rho = float(decay_weight)
        wvec = np.array([rho ** _spread_ext(i - (S - 1), j - (S - 1))
                         for i in range(n) for j in range(n)])
        winv2 = 1.0 / np.maximum(wvec, 1e-8) ** 2
        Cw = C * winv2[None, :]
        G = Cw @ C.T
        lam = np.linalg.lstsq(G, d - C @ tgt, rcond=None)[0]
        k = tgt + winv2 * (C.T @ lam)
    n = 2 * S - 1
    return k.reshape(n, n)


def _asr_basis_matrix(L):
    """Parameterization of the uniform-class-sum manifold:
    w[0:L] = h, w[L:2L] = c - h  ->  w = B @ [h; c], B is (2L, L+1)."""
    B = np.zeros((2 * L, L + 1))
    B[:L, :L] = np.eye(L)
    B[L:, :L] = -np.eye(L)
    B[L:, L] = 1.0
    return B


def design_windows_asr_1d(L, K=3, iters=200, n_restarts=4, seed=0,
                          decay_weight=None, verbose=False):
    """ALS fit of K doubled-support passes to the ASR-projected target.

    Every window lives on the uniform-class-sum manifold (exact ASR on all
    three legs, structurally); the partition of unity collapses to the
    single scalar constraint sum_r c_z c_w c_v = 1, enforced as a hard KKT
    row in every slot fit (machine-precision commensurate limit). Returns
    a list of (z, w, v) arrays of length 2L.
    """
    S = 2 * L
    plain = np.zeros(S)
    plain[:L] = 1.0
    if L <= 2:
        # L=1: single class, plain is exact. L=2: the tent IS the
        # minimal-image kernel and the plain window already has exactly
        # uniform class sums (second period zero): nothing to design.
        return [(plain.copy(), plain.copy(), plain.copy())]

    target = asr_projected_target_1d(L, S, decay_weight=decay_weight)
    n = 2 * S - 1
    n_pts = n * n
    B = _asr_basis_matrix(L)
    n_par = L + 1
    if decay_weight is None or decay_weight >= 1.0:
        fit_w = np.ones(n_pts)
    else:
        rho = float(decay_weight)
        fit_w = np.array([rho ** _spread_ext(i - (S - 1), j - (S - 1))
                          for i in range(n) for j in range(n)])

    def model(passes, skip=None):
        M = np.zeros_like(target)
        for r, p in enumerate(passes):
            if r == skip:
                continue
            M += _kernel_sym(*p)
        return M

    def csum(w):
        return float(np.sum(w)) / L        # the class-sum constant c

    def fit_slot(passes, r, slot):
        resid = (target - model(passes, skip=r)).ravel() * fit_w
        A = np.zeros((n_pts, n_par))
        for k in range(n_par):
            trial = [np.array(x) for x in passes[r]]
            trial[slot] = B[:, k].copy()
            A[:, k] = _kernel_sym(*trial).ravel() * fit_w
        # hard partition row: sum_r cz*cw*cv = 1, linear in this slot
        others = sum(np.prod([csum(w) for w in p])
                     for i, p in enumerate(passes) if i != r)
        oth = np.prod([csum(passes[r][s]) for s in range(3) if s != slot])
        g = oth * (B.sum(axis=0) / L)
        AtA = A.T @ A + 1e-12 * np.eye(n_par)
        Atb = A.T @ resid
        kkt = np.zeros((n_par + 1, n_par + 1))
        kkt[:n_par, :n_par] = AtA
        kkt[:n_par, n_par] = g
        kkt[n_par, :n_par] = g
        sol = np.linalg.lstsq(kkt, np.concatenate([Atb, [1.0 - others]]),
                              rcond=None)[0][:n_par]
        new = [np.array(x) for x in passes[r]]
        new[slot] = B @ sol
        return tuple(new)

    best, best_score = None, np.inf
    for trial in range(n_restarts):
        rs = np.random.RandomState(3000 * L + 7 * K + trial + seed)
        passes = [tuple(B @ (rs.randn(n_par) * 0.4) for _ in range(3))
                  for _ in range(K)]
        passes[0] = (plain.copy(), plain.copy(), plain.copy())
        for it in range(iters):
            for r in range(K):
                for slot in range(3):
                    passes[r] = fit_slot(passes, r, slot)
        Kt = model(passes)
        err = np.sqrt(np.mean(((Kt - target).ravel() * fit_w) ** 2))
        if err < best_score:
            best_score = err
            best = ([tuple(np.array(x) for x in p) for p in passes], err)
    passes, err = best
    if verbose:
        print("ASR window design L={} K={}: RMS to projected target "
              "{:.3e}".format(L, K, err))
    return passes


_WINDOW_DESIGN_ASR_CACHE = {}


def get_window_design_asr(L, K=3, max_K=5, decay_weight=None):
    """Cached ASR-exact design; escalates K while it keeps helping."""
    key = (int(L), int(K), decay_weight)
    if key not in _WINDOW_DESIGN_ASR_CACHE:
        best, best_err = None, np.inf
        for K_try in range(K, max_K + 1):
            passes = design_windows_asr_1d(L, K=K_try,
                                           decay_weight=decay_weight)
            target = asr_projected_target_1d(L, 2 * L,
                                             decay_weight=decay_weight) \
                if L > 2 else None
            if target is None:
                best = passes
                break
            Ktot = sum(_kernel_sym(*p) for p in passes)
            err = np.sqrt(np.mean((Ktot - target) ** 2))
            if err < best_err * 0.9:
                best, best_err = passes, err
            else:
                break
            if err < 1e-8:
                break
        _WINDOW_DESIGN_ASR_CACHE[key] = best
    return _WINDOW_DESIGN_ASR_CACHE[key]


# =========================================================================
# Atom-resolved pinned windows (plan section 5.9)
#
# These windows are a tensor-free way to break the L=2 Wigner-Seitz ties that
# pure cell-index kernels cannot break.  The q_pert leg is pinned to one
# primitive atom and one coarse-cell origin; the two pair legs are assigned to
# the atomic-basis minimal images relative to that pinned atom.  Summing over
# all coarse origins restores the missing translation sum, so the commensurate
# limit has the same normalization as the plain full-window estimator.
# =========================================================================

_ATOMIC_WINDOW_CACHE = {}


def _structure_cache_key(structure, supercell, far, pinned_slot=None):
    return (tuple(np.asarray(supercell, dtype=int)),
            int(far),
            None if pinned_slot is None else pinned_slot,
            tuple(np.round(np.asarray(structure.unit_cell).ravel(), 10)),
            tuple(np.round(np.asarray(structure.coords).ravel(), 10)))


def _atomic_leg_images(structure, supercell, far, a, b, d):
    """Atomic-basis minimal images of leg b relative to pinned atom a.

    Returns [(extended_cell, weight), ...] with tie splitting.  The target
    is geometry-only: no force-constant value enters this assignment.
    """
    L = np.asarray(supercell, dtype=int)
    reps = np.array(list(itertools.product(range(-far, far + 1), repeat=3)),
                    dtype=int)
    ext = np.asarray(d, dtype=int)[None, :] + reps * L[None, :]
    vecs = ext @ structure.unit_cell
    dist = np.linalg.norm(structure.coords[b] - structure.coords[a] + vecs,
                          axis=1)
    mn = np.min(dist)
    winners = np.where(dist < mn + 1e-6)[0]
    return [(tuple(ext[i]), 1.0 / len(winners)) for i in winners]


def get_atomic_window_passes(structure, supercell, far=3, pinned_slot=0):
    """Geometry-only pinned atomic window passes.

    One pass is built for each pinned primitive atom a. The selected
    pinned_slot (0=z, 1=w, 2=v) is a delta at the origin; the other two
    slots are atomic minimal-image selectors relative to that pinned atom.

    The returned window maps use keys (primitive_atom, extended_cell_tuple).
    They are converted to per-q complex weights by _window_map_to_qweights.
    """
    pinned_slot = int(pinned_slot)
    if pinned_slot not in (0, 1, 2):
        raise ValueError("pinned_slot must be 0, 1, or 2")
    key = _structure_cache_key(structure, supercell, far, pinned_slot)
    if key in _ATOMIC_WINDOW_CACHE:
        return _ATOMIC_WINDOW_CACHE[key]

    L = np.asarray(supercell, dtype=int)
    cells = np.array(list(itertools.product(range(L[0]), range(L[1]),
                                            range(L[2]))), dtype=int)
    passes = []
    for a in range(structure.N_atoms):
        maps = [None, None, None]
        maps[pinned_slot] = {(a, (0, 0, 0)): 1.0}
        leg = {}
        for b in range(structure.N_atoms):
            for d in cells:
                for ext, w in _atomic_leg_images(structure, L, far, a, b, d):
                    leg[(b, ext)] = leg.get((b, ext), 0.0) + w
        for slot in range(3):
            if slot != pinned_slot:
                maps[slot] = dict(leg)
        passes.append(tuple(maps))

    _ATOMIC_WINDOW_CACHE[key] = passes
    return passes


_ATOMIC_ASR_WINDOW_CACHE = {}


def _plain_atom_window(structure, supercell, coeff=1.0):
    """Uniform atom/cell window on the fundamental supercell."""
    L = np.asarray(supercell, dtype=int)
    out = {}
    for a in range(structure.N_atoms):
        for cell in itertools.product(range(L[0]), range(L[1]), range(L[2])):
            out[(a, tuple(cell))] = float(coeff)
    return out


def _map_to_vector(map_, keys):
    return np.array([map_.get(k, 0.0) for k in keys], dtype=np.float64)


def _vector_to_map(vec, keys, tol=1e-14):
    return {k: float(v) for k, v in zip(keys, vec) if abs(v) > tol}


def get_atomic_asr_window_passes(structure, supercell, far=3, svd_tol=1e-10):
    """ASR-compatible atom-resolved passes.

    For each force channel, average the raw pinned-atom pair kernel over
    the pinned atom and use a uniform pinned-slot window. The non-pinned
    pair kernel is factorized by SVD into window products. Because every
    atomic leg map has unit folded class sums, the factorized averaged
    target has constant image sums on the non-pinned legs; the uniform
    pinned window makes the pinned-leg ASR exact as well.
    """
    key = _structure_cache_key(structure, supercell, far, "asr")
    if key in _ATOMIC_ASR_WINDOW_CACHE:
        return _ATOMIC_ASR_WINDOW_CACHE[key]

    out = []
    nat = structure.N_atoms
    for channel in range(3):
        raw = get_atomic_window_passes(structure, supercell, far=far,
                                       pinned_slot=channel)
        other = [s for s in range(3) if s != channel]
        keys0 = sorted(set().union(*(set(p[other[0]]) for p in raw)))
        keys1 = sorted(set().union(*(set(p[other[1]]) for p in raw)))
        M = np.zeros((len(keys0), len(keys1)), dtype=np.float64)
        for p in raw:
            M += np.outer(_map_to_vector(p[other[0]], keys0),
                          _map_to_vector(p[other[1]], keys1))
        M /= float(nat)

        U, svals, Vh = np.linalg.svd(M, full_matrices=False)
        if len(svals) == 0:
            continue
        cutoff = float(svd_tol) * max(1.0, svals[0])
        for r, sval in enumerate(svals):
            if sval <= cutoff:
                continue
            maps = [None, None, None]
            maps[channel] = _plain_atom_window(structure, supercell,
                                               coeff=sval)
            maps[other[0]] = _vector_to_map(U[:, r], keys0)
            maps[other[1]] = _vector_to_map(Vh[r, :], keys1)
            out.append((channel, tuple(maps)))

    _ATOMIC_ASR_WINDOW_CACHE[key] = out
    return out


def _window_map_to_qweights(window, q_points, structure, itau, cell_idx,
                            supercell):
    """Convert an extended-cell atom window to _build_field_set weights."""
    L = np.asarray(supercell, dtype=int)
    W = np.zeros((len(q_points), len(itau)), dtype=np.complex128)
    lookup = {}
    for k, (a, n) in enumerate(zip(itau, cell_idx)):
        lookup[(int(a),) + tuple((np.asarray(n, dtype=int) % L))] = k

    for (a, ext_cell), value in window.items():
        ext = np.asarray(ext_cell, dtype=int)
        base = ext % L
        shift = ext - base
        k = lookup[(int(a),) + tuple(base)]
        r_shift = shift @ structure.unit_cell
        W[:, k] += value * np.exp(-2j * np.pi * (q_points @ r_shift))
    return W


# =========================================================================
# Third-order tensor helper (hybrid tensor-D3 mode, plan section 5.8)
# =========================================================================

def load_d3_tensor(dyn, d3_realspace, far=3, apply_asr=True):
    """Build a centered cellconstructor Tensor3 for d3_mode="tensor".

    Parameters
    ----------
    dyn : CC.Phonons.Phonons
        The (coarse) SSCHA dynamical matrix defining structure/supercell.
    d3_realspace : ndarray(3*nat_sc, 3*nat_sc, 3*nat_sc) or str
        The real-space third-order tensor in Ry/Bohr^3 on the coarse
        supercell (e.g. the stochastic d3 saved by
        sscha get_free_energy_hessian(..., verbose=True) times 2 for
        Ha -> Ry), or the path of an .npy file containing it.
    far : int
        Replica search range of Tensor3.Center. Must be large enough that
        the minimal-perimeter image assignment is closed under permutation
        symmetry (cellconstructor raises otherwise); SnTe 2x2x2 needs 3.
    apply_asr : bool
        Re-impose the acoustic sum rule after centering (recommended,
        centering destroys it).
    """
    if isinstance(d3_realspace, str):
        d3_realspace = np.load(d3_realspace)
    t3 = CC.ForceTensor.Tensor3(dyn.structure,
                                dyn.structure.generate_supercell(dyn.GetSupercell()),
                                dyn.GetSupercell())
    t3.SetupFromTensor(d3_realspace)
    t3.Center(Far=far)
    if apply_asr:
        t3.Apply_ASR()
    return t3


# =========================================================================
# The interpolated Lanczos
# =========================================================================

class QSpaceLanczosInterp(QL.QSpaceLanczos):
    """Q-space Lanczos on a fine q-mesh, interpolated from a coarse ensemble.

    Usage
    -----
    >>> lanczos = QSpaceLanczosInterp(ensemble, fine_mesh=(4, 4, 4),
    ...                               window_design="minimal_image")
    >>> lanczos.init(use_symmetries=True)
    >>> lanczos.prepare_mode_q(iq, band)   # iq indexes the FINE mesh
    >>> lanczos.run_FT(100)

    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble
        The SSCHA ensemble (on the coarse supercell).
    fine_mesh : tuple(3) of int
        The fine uniform Gamma-centered q-mesh.
    window_design : str
        "plain": single full-period window (tent interpolation kernel,
        exact ASR, worst centering);
        "minimal_image": designed multitaper windows approximating the
        zero-padded minimal-image centering for the D3 terms (best
        centering, but the kernel-level acoustic sum rule is violated at
        the same order as an un-Apply_ASR'd centered tensor);
        "asr": doubled-support windows on the uniform-class-sum manifold
        (plan section 5.7): exact ASR on every vertex leg AND exact
        commensurate limit by construction, centering fitted to the
        ASR-projected minimal-image kernel. Recommended whenever the fine
        mesh densely samples the Gamma neighborhood (acoustic two-phonon
        continuum), where an ASR leak is amplified by the diverging
        occupation/propagator factors.
        NOTE: "minimal_image" and "asr" assume the coarse supercell
        resolves the third-order force-constant range in the cell-index
        sense (spread strictly inside the Wigner-Seitz cell). If the range
        reaches the WS boundary the image assignment is Nyquist-ambiguous
        and the oscillating designs degrade catastrophically near Gamma at
        strong coupling. "atomic" is the atom-resolved L=2 remedy when
        basis offsets break those ties.
        "atomic": atom-resolved pinned windows (plan section 5.9): one
        q_pert-slot atom is pinned, the two pair legs use geometry-only
        atomic-basis minimal images relative to it, and all coarse origins
        are summed. This breaks the L=2 basis ties without constructing a
        d3 tensor; D4 still uses the plain pass.
        "atomic_delta": atom-resolved pinned windows plus a conservative
        q-space ASR Delta projector. The raw atomic centering passes are
        kept unchanged; before mode projection each windowed Cartesian field
        has only its uniform acoustic component subtracted with a
        Tensor3.Interpolate(asr=True)-style sinc factor. This realizes an
        additive correction to the raw atomic estimator, not a replacement
        of its pinned-atom geometry.
        "atomic_asr": experimental diagnostic prototype: the force slot is
        uniform and the atom-averaged pair kernel is SVD-factorized into
        slot windows. This is not a production mode; on the SnTe L=2 case
        it removes the pinned-atom basis-offset information needed for good
        centering.
    window_K : int
        Number of window passes per lattice dimension for the design.
    window_far : int
        Replica search range for window_design="atomic"; analogous to the
        Far argument of Tensor3.Center, but used only in geometry-only
        atomic minimal-image assignments.
    window_decay : float in (0, 1] or None
        Only for window_design="asr": metric weight rho**spread used in
        the kernel projection/fit (the analogue of Apply_ASR's `power`);
        concentrates centering fidelity at short range where the physical
        Phi3 is large. None = uniform metric.
    d3_mode : str
        "stochastic" (default): the D3 channel uses the windowed
        per-configuration estimator (tensor-free).
        "tensor": the D3 vertex is interpolated DETERMINISTICALLY from a
        centered third-order force-constant tensor -- by construction the
        same interpolation as the cellconstructor.Spectral d3 bubble
        (plan section 5.8). It remains useful as an oracle/control or when
        an explicit FC3 is already available. D4 stays on the stochastic
        plain pass.
    d3_tensor : CC.ForceTensor.Tensor3
        Required for d3_mode="tensor": the third-order tensor on the
        COARSE supercell, already Center()ed (+ Apply_ASR()), in Ry/Bohr^3
        consistent with the dyn -- the exact object the Spectral bubble
        consumes (see load_d3_tensor). Not serialized on save/load.
    prefilter : bool
        Apply the f_Y pre-filter to the displacement fields (recommended;
        see module docstring).
    use_asr_dyn, reuse_commensurate, w_min_guard, allow_unstable :
        See interpolate_dyn_fine and the plan document.
    """

    def __init__(self, ensemble, fine_mesh=None, use_asr_dyn=True,
                 reuse_commensurate=True, w_min_guard=1e-8,
                 allow_unstable=False, prefilter=True,
                 window_design="plain", window_K=3, window_origins=1,
                 asr_fields=True, window_decay=None, window_far=3,
                 d3_mode="stochastic", d3_tensor=None,
                 lo_to_split=None, **kwargs):

        if lo_to_split is not None:
            raise NotImplementedError(
                "LO-TO splitting is not supported by the interpolated "
                "q-space Lanczos (phase 1).")

        super().__init__(ensemble, lo_to_split=None, **kwargs)

        interp_attrs = ['fine_mesh', '_fine_idx', '_q_lookup', '_fpsi_fine',
                        'window_design', 'window_K', 'window_origins', 'asr_fields',
                        'window_decay', 'window_far',
                        'd3_mode', '_d3_tensor', '_d3_blocks',
                        '_field_sets', '_window_passes', '_channels']
        self.__total_attributes__.extend(interp_attrs)

        self.fine_mesh = None
        self._fine_idx = None
        self._q_lookup = None
        self._fpsi_fine = None
        self.window_design = window_design
        self.window_K = window_K
        self.window_origins = int(window_origins)
        self.asr_fields = bool(asr_fields)
        self.window_decay = window_decay
        self.window_far = int(window_far)
        self.d3_mode = d3_mode
        self._d3_tensor = d3_tensor      # centered CC Tensor3 (transient)
        self._d3_blocks = None           # per-pair D3 blocks (built per q_pert)
        self._field_sets = None      # list of (X_q, Y_q) per distinct window
        self._window_passes = None   # list of (c=1-folded, iz, iw, iv)
        self._channels = None

        # Bare initialization (used by the distributed loader)
        if ensemble is None:
            return

        if fine_mesh is None:
            raise ValueError("QSpaceLanczosInterp requires fine_mesh=(m1, m2, m3)")
        if window_design not in ("plain", "minimal_image", "asr",
                                 "atomic", "atomic_delta", "atomic_asr"):
            raise ValueError("window_design must be 'plain', 'minimal_image' "
                             "'asr', 'atomic', 'atomic_delta', or "
                             "'atomic_asr'")
        if window_design == "atomic_asr":
            warnings.warn(
                "window_design='atomic_asr' is an experimental diagnostic "
                "prototype and is not validated for production interpolation; "
                "use window_design='atomic' for the SnTe L=2 tensor-free "
                "centering fix.",
                RuntimeWarning)
        if d3_mode not in ("stochastic", "tensor"):
            raise ValueError("d3_mode must be 'stochastic' or 'tensor'")
        if d3_mode == "tensor":
            if d3_tensor is None:
                raise ValueError(
                    "d3_mode='tensor' requires d3_tensor (a centered "
                    "cellconstructor Tensor3; see load_d3_tensor)")
            if window_design != "plain":
                warnings.warn(
                    "d3_mode='tensor': the designed windows only serve the "
                    "stochastic D3 channel; forcing window_design='plain'.")
                window_design = "plain"
                self.window_design = "plain"

        self.fine_mesh = np.asarray(fine_mesh, dtype=int)
        coarse_mesh = np.asarray(self.dyn.GetSupercell(), dtype=int)
        n_c = int(np.prod(coarse_mesh))
        n_f = int(np.prod(self.fine_mesh))

        # Stash the coarse-grid quantities computed by the parent before we
        # overwrite them (needed for the pre-filtered fields).
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
        small[:, 0] = False
        if np.any(small):
            warnings.warn("Masking {} interpolated modes with |w| < {} Ry "
                          "away from Gamma.".format(np.sum(small), w_min_guard))
            self.valid_modes_q &= ~small

        if ensemble.ignore_small_w:
            small_freq = np.abs(self.w_q) < CC.Phonons.__EPSILON_W__
            self.valid_modes_q &= ~small_freq

        # == 4. Real-space channels and Bloch fields on the fine mesh ==
        self.qspace_prefiltered = bool(prefilter)
        self._channels = self._prepare_realspace_channels(
            coarse_data if prefilter else None)

        # Plain fields (always built: used directly in "plain" mode, and as
        # the D4 pass and back-compat X_q/Y_q otherwise)
        self.X_q, self.Y_q = self._build_field_set(weights=None)

        if window_design in ("minimal_image", "asr", "atomic",
                             "atomic_delta", "atomic_asr"):
            self._setup_window_passes()

        if prefilter or d3_mode == "tensor":
            # (also needed by the deterministic tensor-D3 f_pert contraction)
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
        fy = np.zeros_like(w_c)
        for iq in range(w_c.shape[1]):
            valid = valid_c[:, iq]
            w = w_c[valid, iq]
            if self.T > __EPSILON__:
                n_bose = 1.0 / (np.exp(w * QL.__RyToK__ / self.T) - 1.0)
            else:
                n_bose = np.zeros_like(w)
            fy[valid, iq] = 2.0 * w / (1.0 + 2.0 * n_bose)
        return fy

    # ---------------------------------------------------------------
    def _prepare_realspace_channels(self, coarse_data):
        """Assemble the (unprojected) real-space channels once.

        Returns a dict with:
          u_ch    : displacement channel (N, nat_sc, 3). If coarse_data is
                    given this is the f_Y-PREFILTERED, mass-scaled field
                    (Bohr sqrt(mass) units, no further scaling needed);
                    otherwise the raw displacements (Angstrom).
          f_ch    : force residual channel (N, nat_sc, 3), average force
                    subtracted, Ry/Angstrom.
          itau, r_lat, cell_idx, n_c, and the unit/mass scale vectors.
        """
        ens = self.ensemble
        uc = self.dyn.structure
        sc = self.super_structure
        nat_uc = uc.N_atoms
        nat_sc = sc.N_atoms
        n_c = nat_sc // nat_uc
        N = self.N

        itau = sc.get_itau(uc) - 1
        r_lat = sc.coords - uc.coords[itau]          # Angstrom, cell origins

        # integer cell indices of each supercell atom (for windows)
        frac = np.linalg.solve(uc.unit_cell.T, r_lat.T).T
        cell_idx = np.round(frac).astype(int) % np.asarray(self.dyn.GetSupercell())

        u_conv = 1.0
        f_conv = 1.0
        if ens.units == "default":
            u_conv = CC.Units.A_TO_BOHR
            f_conv = 1.0 / CC.Units.A_TO_BOHR
        elif ens.units == "hartree":
            f_conv = 2.0

        # ---- force residual channel ----
        # Real-space sscha_forces may be empty in Fourier-gradient mode:
        # rebuild from the q-space array (exact inverse on the coarse grid).
        q_coarse = np.array(self.dyn.q_tot)
        phases_c = np.exp(-2j * np.pi * (q_coarse @ r_lat.T))   # (n_qc, nat_sc)
        fsq = np.array(ens.sscha_forces_qspace)                 # (N, 3nat_uc, n_qc)
        f_sscha = np.zeros((N, nat_sc, 3), dtype=np.float64)
        for a in range(nat_uc):
            sel = np.where(itau == a)[0]
            f_sscha[:, sel, :] = np.real(np.einsum(
                'qk,iaq->ika', np.conj(phases_c[:, sel]),
                fsq[:, 3 * a:3 * a + 3, :], optimize=True)) / np.sqrt(n_c)

        delta_f = np.array(ens.forces, dtype=np.float64).reshape(N, nat_sc, 3) - f_sscha

        f_mean_uc = ens.get_average_forces(get_error=False)
        qe_sym = CC.symmetries.QE_Symmetry(uc)
        qe_sym.SetupQPoint()
        qe_sym.SymmetrizeVector(f_mean_uc)
        delta_f -= f_mean_uc[itau, :][None, :, :]

        # ---- displacement channel ----
        if coarse_data is None:
            u_ch = np.array(ens.u_disps, dtype=np.float64).reshape(N, nat_sc, 3)
            u_prefiltered = False
        else:
            fy_c = self._get_fy_table_coarse(coarse_data)
            X_c = coarse_data['X_q']
            pols_c = coarse_data['pols_q']
            q_c = coarse_data['q_points']
            phases_cc = np.exp(-2j * np.pi * (q_c @ r_lat.T))
            uf_sc = np.zeros((N, nat_sc, 3), dtype=np.float64)
            idx3 = 3 * itau[:, None] + np.arange(3)[None, :]
            for iq in range(X_c.shape[0]):
                u_mass_q = (X_c[iq] * fy_c[:, iq][None, :]) @ pols_c[:, :, iq].T
                uf_sc += np.real(np.conj(phases_cc[iq])[None, :, None]
                                 * u_mass_q[:, idx3])
            uf_sc /= np.sqrt(X_c.shape[0])
            u_ch = uf_sc
            u_prefiltered = True

        m_uc = uc.get_masses_array()
        return {
            'u_ch': u_ch, 'f_ch': delta_f, 'u_prefiltered': u_prefiltered,
            'itau': itau, 'r_lat': r_lat, 'cell_idx': cell_idx, 'n_c': n_c,
            'm_sc': sc.get_masses_array(),
            'u_conv': u_conv, 'f_conv': f_conv,
            'sqrt_m3': np.sqrt(np.repeat(m_uc, 3)),
        }

    # ---------------------------------------------------------------
    def _build_field_set(self, weights=None, perm=None):
        """Bloch fields (X_q, Y_q) on the fine mesh for one window.

        Parameters
        ----------
        weights : ndarray(nat_sc,), ndarray(n_q, nat_sc) complex, or None
            Per-supercell-atom window weights W(n(k)) (None = plain window,
            all ones). The window support and the Bloch phases live on the
            FIXED fundamental domain. A 2D complex array gives PER-Q
            effective weights: this is how a doubled-support (2L) window is
            applied to the L-periodic data (the second period contributes
            the same atoms with an extra Bloch phase across the supercell,
            plan section 5.7e).
        perm : ndarray(nat_sc,) int or None
            Atom permutation applied to the real-space data BEFORE
            windowing: rolling the configuration by a lattice vector o and
            keeping window+phases fixed realizes a window at origin o (the
            leftover constant phase e^{-i q.o} cancels in every
            momentum-conserving product). Ensemble translation invariance
            makes the expected kernel origin-independent, so origin
            averaging is a pure variance reduction.

        Applies the window-weighted acoustic-sum-rule projections
        (plan section 5.5) before the transform:
          - raw displacements lose the W-weighted mass-weighted COM;
          - prefiltered (mass-scaled) fields lose the W-weighted projection
            on the sqrt(m) translation pattern;
          - force residuals lose the W-weighted net force redistributed
            proportionally to the masses.
        """
        ch = self._channels
        N = self.N
        itau = ch['itau']
        nat_uc = len(np.unique(itau))
        nat_sc = len(itau)
        m_sc = ch['m_sc']

        if weights is None:
            W = np.ones(nat_sc)
            per_q = False
        else:
            W = np.asarray(weights)
            per_q = (W.ndim == 2)
            if not per_q:
                W = W.astype(float)

        if perm is None:
            u = ch['u_ch'].copy()
            f = ch['f_ch'].copy()
        else:
            u = ch['u_ch'][:, perm, :].copy()
            f = ch['f_ch'][:, perm, :].copy()

        # --- zero-mode (ASR) projections ---
        # Applied only to (near-)uniform windows: there the per-configuration
        # sum rules (Newton's third law, zero COM) make the projection exact
        # and essentially free of statistical cost. For sign-oscillating
        # one-period windows the W-weighted zero-mode is a rank-one
        # modification of the effective window: projecting the FIELD would
        # silently change the interpolation kernel away from the designed
        # one (verified: it introduces an O(30%) systematic bias on the toy
        # model; plan section 5.7b states this as a no-go theorem). Per-q
        # complex weights come from the doubled-support "asr" design, whose
        # uniform class sums make the q->0 acoustic contraction vanish per
        # configuration AUTOMATICALLY (the effective weight at commensurate
        # q is the constant class sum): no projection is needed or applied.
        is_uniform = (not per_q) and (
            np.max(W) - np.min(W) < 1e-12 * max(1.0, np.max(np.abs(W))))
        if self.asr_fields and is_uniform:
            WM = np.sum(W * m_sc)
            if ch['u_prefiltered']:
                sqm = np.sqrt(m_sc)
                coeff = np.einsum('k,k,ika->ia', W, sqm, u) / WM      # (N, 3)
                u -= sqm[None, :, None] * coeff[:, None, :]
            else:
                com = np.einsum('k,k,ika->ia', W, m_sc, u) / WM
                u -= com[:, None, :]
            f_net = np.einsum('k,ika->ia', W, f)
            f -= (m_sc / WM)[None, :, None] * f_net[:, None, :]

        # --- windowed NUDFT ---
        phases = np.exp(-2j * np.pi * (self.q_points @ ch['r_lat'].T))
        phases_w = phases * (W if per_q else W[None, :])

        def nudft(field_sc):
            out = np.zeros((self.n_q, N, 3 * nat_uc), dtype=np.complex128)
            for a in range(nat_uc):
                sel = np.where(itau == a)[0]
                out[:, :, 3 * a:3 * a + 3] = np.einsum(
                    'qk,ika->qia', phases_w[:, sel], field_sc[:, sel, :],
                    optimize=True)
            return out / np.sqrt(ch['n_c'])

        u_tilde = nudft(u)
        f_tilde = nudft(f)

        if ch['u_prefiltered']:
            u_scale = np.ones_like(ch['sqrt_m3'])
        else:
            u_scale = ch['u_conv'] * ch['sqrt_m3']
        f_scale = ch['f_conv'] / ch['sqrt_m3']

        u_cart = u_tilde * u_scale[None, None, :]
        f_cart = f_tilde * f_scale[None, None, :]

        if self.window_design == "atomic_delta":
            self._apply_atomic_delta_projector(u_cart, f_cart)

        X = np.zeros((self.n_q, N, self.n_bands), dtype=np.complex128)
        Y = np.zeros((self.n_q, N, self.n_bands), dtype=np.complex128)
        for iq in range(self.n_q):
            pol_iq = self.pols_q[:, :, iq]
            X[iq] = u_cart[iq] @ np.conj(pol_iq)
            Y[iq] = f_cart[iq] @ np.conj(pol_iq)
        return X, Y

    # ---------------------------------------------------------------
    def _apply_atomic_delta_projector(self, u_cart, f_cart):
        """Tensor3-style q-space ASR Delta for atom-resolved windows.

        This is deliberately conservative: it does not alter any atomic
        window or pinned-atom image assignment.  It subtracts only the
        uniform acoustic Cartesian component from the already-windowed
        fields, multiplied by the same sinc-like support factor used by
        Tensor3.Interpolate(asr=True).  The product of projected fields is
        the raw atomic estimator plus an implicit additive Delta.
        """
        ch = self._channels
        nat = self.dyn.structure.N_atoms
        sqrt_m = np.sqrt(self.dyn.structure.get_masses_array())
        trans = np.zeros((3, 3 * nat), dtype=np.float64)
        for cart in range(3):
            trans[cart, 3 * np.arange(nat) + cart] = sqrt_m
        norm = float(np.sum(sqrt_m ** 2))

        # Odd support size analogous to Tensor3.Interpolate(asr=True).
        # Atomic centering searches images d + m L with |m| <= Far.  The
        # corresponding centered support spans -Far*L ... +Far*L, hence
        # length 2*Far*L + 1.  It is odd, so the sinc kernel is real and
        # equals one at Gamma.
        L = np.asarray(self.dyn.GetSupercell(), dtype=int)
        N_i = 2 * int(self.window_far) * L + 1
        at = self.dyn.structure.unit_cell
        tol = 1e-8
        fq = np.ones(self.n_q, dtype=np.float64)
        for iq, q in enumerate(self.q_points):
            factors = np.ones(3, dtype=np.float64)
            aq = at @ q
            mask = np.abs(np.sin(np.pi * aq)) > tol
            factors[mask] = (np.sin(N_i[mask] * np.pi * aq[mask]) /
                             (N_i[mask] * np.sin(np.pi * aq[mask])))
            fq[iq] = np.prod(factors)

        for iq in range(self.n_q):
            if abs(fq[iq]) < 1e-14:
                continue
            for basis in trans:
                cu = (u_cart[iq] @ basis) / norm
                cf = (f_cart[iq] @ basis) / norm
                u_cart[iq] -= fq[iq] * cu[:, None] * basis[None, :]
                f_cart[iq] -= fq[iq] * cf[:, None] * basis[None, :]

    # ---------------------------------------------------------------
    def _setup_window_passes(self):
        """Build the windowed field sets for the designed-window modes.

        3D windows are per-dimension products; passes are the product of
        the per-dimension pass lists, replicated over `window_origins`
        rigid origin shifts of the window (the ensemble average is
        origin-independent -- translation invariance -- so origin
        averaging is a pure variance reduction; the pass weight 1/n_origins
        is folded into the z-slot window). Field sets are deduplicated
        across passes/slots/origins by hashing the window values.

        For window_design == "asr" the per-dimension windows have doubled
        support (2 L_d, plan section 5.7): on the L-periodic data the
        second period contributes the same atoms with the extra Bloch
        phase exp(-2 pi i q . (L_d a_d)) across the supercell, so each
        3D window becomes a PER-Q complex effective weight
            W_q(s) = prod_d [ wA_d(n_d(s)) + wB_d(n_d(s)) ph_d(q) ].
        At commensurate q the phase is 1 and the weight collapses to the
        constant class-sum product: every pass is a multiple of the plain
        estimator and their sum reproduces it exactly (hard partition
        constraint sum_r c_z c_w c_v = 1), configuration by configuration.
        """
        coarse = np.asarray(self.dyn.GetSupercell(), dtype=int)
        cell_idx = self._channels['cell_idx']    # (nat_sc, 3)

        # Origin shifts: realized by PERMUTING THE DATA (rolling the
        # configuration by a lattice vector) while the window support and
        # the Bloch phases stay on the fixed fundamental domain. Shifting
        # the window cyclically instead would wrap its support across the
        # domain boundary with inconsistent absolute phases and destroy the
        # designed kernel (verified).
        if self.window_design in ("atomic", "atomic_delta", "atomic_asr"):
            shifts = [np.array(s, dtype=int)
                      for s in itertools.product(range(coarse[0]),
                                                 range(coarse[1]),
                                                 range(coarse[2]))]
            if self.window_design == "atomic_asr":
                shifts = [np.zeros(3, dtype=int)]
        else:
            n_orig = max(1, int(self.window_origins))
            d_max = int(np.argmax(coarse))
            shifts = []
            for j in range(n_orig):
                s = np.zeros(3, dtype=int)
                s[d_max] = (j * coarse[d_max]) // n_orig
                if not any(np.array_equal(s, t) for t in shifts):
                    shifts.append(s)

        # atom permutation per shift: perm[k] = atom at cell n(k) + shift
        cell_lookup = {}
        itau = self._channels['itau']
        for k in range(len(itau)):
            cell_lookup[(itau[k],) + tuple(cell_idx[k] % coarse)] = k
        perms = []
        for s in shifts:
            perm = np.array([cell_lookup[(itau[k],) + tuple((cell_idx[k] + s) % coarse)]
                             for k in range(len(itau))], dtype=int)
            perms.append(perm)

        field_cache = {}     # (window-bytes, origin) -> index in _field_sets
        self._field_sets = []

        def get_set(w3d, i_orig):
            key = (np.round(w3d, 12).tobytes(), i_orig)
            if key not in field_cache:
                field_cache[key] = len(self._field_sets)
                self._field_sets.append(self._build_field_set(
                    weights=w3d, perm=perms[i_orig] if i_orig > 0 else None))
            return field_cache[key]

        if self.window_design in ("atomic", "atomic_delta"):
            self._window_passes = []
            for channel in range(3):
                geom_passes = get_atomic_window_passes(
                    self.dyn.structure, coarse, far=self.window_far,
                    pinned_slot=channel)
                for maps in geom_passes:
                    weights = [_window_map_to_qweights(
                        m, self.q_points, self.dyn.structure, itau, cell_idx,
                        coarse) for m in maps]
                    for i_orig in range(len(shifts)):
                        self._window_passes.append((channel,) + tuple(
                            get_set(w, i_orig) for w in weights))

            if Parallel.am_i_the_master():
                label = "atomic-delta" if self.window_design == "atomic_delta" \
                    else "atomic-perm"
                extra = " + q-space ASR Delta projector" \
                    if self.window_design == "atomic_delta" else ""
                print("Stochastic centering [{}]: {} window passes "
                      "({} full origins), {} distinct field sets{}".format(
                          label, len(self._window_passes), len(shifts),
                          len(self._field_sets), extra))
            return

        if self.window_design == "atomic_asr":
            self._window_passes = []
            geom_passes = get_atomic_asr_window_passes(
                self.dyn.structure, coarse, far=self.window_far)
            for channel, maps in geom_passes:
                weights = [_window_map_to_qweights(
                    m, self.q_points, self.dyn.structure, itau, cell_idx,
                    coarse) for m in maps]
                self._window_passes.append((channel,) + tuple(
                    get_set(w, 0) for w in weights))

            if Parallel.am_i_the_master():
                print("Stochastic centering [atomic-asr]: {} SVD/channel "
                      "passes, {} distinct field sets".format(
                          len(self._window_passes), len(self._field_sets)))
            return

        if self.window_design == "asr":
            designs = [get_window_design_asr(coarse[d], K=self.window_K,
                                             decay_weight=self.window_decay)
                       for d in range(3)]
            # per-dimension second-period Bloch phases, (n_q,) each
            uc_cell = self.dyn.structure.unit_cell
            ph = [np.exp(-2j * np.pi *
                         (self.q_points @ (coarse[d] * uc_cell[d])))
                  for d in range(3)]

            def make_weight(p0, p1, p2, slot):
                wq = np.ones((self.n_q, len(itau)), dtype=np.complex128)
                for d, p in enumerate((p0, p1, p2)):
                    Ld = coarse[d]
                    wA = p[slot][:Ld][cell_idx[:, d]]
                    wB = p[slot][Ld:2 * Ld][cell_idx[:, d]]
                    wq *= wA[None, :] + wB[None, :] * ph[d][:, None]
                return wq
        else:
            designs = [get_window_design(coarse[d], K=self.window_K)
                       for d in range(3)]

            def make_weight(p0, p1, p2, slot):
                return (p0[slot][cell_idx[:, 0]]
                        * p1[slot][cell_idx[:, 1]]
                        * p2[slot][cell_idx[:, 2]])

        self._window_passes = []
        for p0 in designs[0]:
            for p1 in designs[1]:
                for p2 in designs[2]:
                    for i_orig in range(len(shifts)):
                        per_slot = []
                        for slot in range(3):
                            w3d = make_weight(p0, p1, p2, slot)
                            if slot == 0:
                                w3d = w3d / len(shifts)
                            per_slot.append(get_set(w3d, i_orig))
                        self._window_passes.append(tuple(per_slot))

        if Parallel.am_i_the_master():
            print("Stochastic centering [{}]: {} window passes ({} origins), "
                  "{} distinct field sets".format(self.window_design,
                                                  len(self._window_passes),
                                                  len(shifts),
                                                  len(self._field_sets)))

    # ---------------------------------------------------------------
    # Hybrid tensor-D3 mode (plan section 5.8)
    # ---------------------------------------------------------------
    def _build_d3_blocks(self):
        """Mode-space D3 vertex blocks at the fine pairs of this q_pert.

        For each unique pair (q1, q2) with q1 + q2 = q_pert + G:

          D3[nu, nu1, nu2] = N_f^{-1/2} sum_abc e_nu^a(q_pert)
                             conj(e_nu1^b(q1)) conj(e_nu2^c(q2))
                             (m_a m_b m_c)^{-1/2} Phi3_hat_abc(-q1, -q2)

        with Phi3_hat(-q1, -q2) = Tensor3.Interpolate(q1, q2) in the
        cellconstructor phase convention e^{-2 pi i q.r} — the SAME centered
        interpolation the Spectral d3 bubble uses. Leg a (cell 0) carries the
        implied momentum q1 + q2 = q_pert; legs b, c pair with the
        displacement fields, whence the conjugated polarization vectors.
        Mode-validity masks are applied to every leg. Storage
        O(n_pairs * nb^3): no tensor ever lives on the fine mesh.
        """
        if self._d3_tensor is None:
            raise ValueError(
                "d3_mode='tensor' but no d3_tensor is attached (it is not "
                "serialized on save/load: re-assign lanczos._d3_tensor).")

        nb = self.n_bands
        inv_sqrt_m3 = 1.0 / np.sqrt(
            np.repeat(self.dyn.structure.get_masses_array(), 3))
        norm = 1.0 / np.sqrt(self.n_q)

        E_pert = (self.pols_q[:, :, self.iq_pert]
                  * self.valid_modes_q[None, :, self.iq_pert]) \
            * inv_sqrt_m3[:, None]

        self._d3_blocks = []
        for (iq1, iq2) in self.unique_pairs:
            phi3 = self._d3_tensor.Interpolate(
                np.asarray(self.q_points[iq1], dtype=np.float64),
                np.asarray(self.q_points[iq2], dtype=np.float64), asr=False)
            E1 = np.conj(self.pols_q[:, :, iq1]
                         * self.valid_modes_q[None, :, iq1]) \
                * inv_sqrt_m3[:, None]
            E2 = np.conj(self.pols_q[:, :, iq2]
                         * self.valid_modes_q[None, :, iq2]) \
                * inv_sqrt_m3[:, None]
            block = np.einsum('abc,an,bp,cq->npq', phi3, E_pert, E1, E2,
                              optimize=True) * norm
            self._d3_blocks.append(block)

    def _apply_d3_tensor(self, R1, alpha1_blocks):
        """Deterministic D3 action from the interpolated vertex blocks.

        Exact infinite-N limit of the stochastic D3 terms (weight_R,
        weight_Rf -> d2v; w1, w2 -> f_pert), mutually adjoint by
        construction (same blocks on both maps -> L stays Hermitian):

          d2v(q1, q2)[nu1, nu2] = + sum_nu R1[nu] D3[nu, nu1, nu2]
          f_pert[nu] = + 1/2 sum_ordered-pairs conj(D3[nu, nu1, nu2])
                         f_psi(q1, nu1) f_psi(q2, nu2) alpha1[nu1, nu2]

        (off-diagonal pairs counted twice, matching total_sum; the
        transpose-symmetric alpha1 blocks make the reversed orientation
        identical). No n_syms average (the centered tensor is already
        symmetric) and no scale3 (N_f^{-1/2} is direct in the blocks).
        """
        nb = self.n_bands
        fpsi = self._fpsi_fine
        f_pert = np.zeros(nb, dtype=np.complex128)
        d2v_blocks = []
        for p, (iq1, iq2) in enumerate(self.unique_pairs):
            D3 = self._d3_blocks[p]
            d2v_blocks.append(np.einsum('n,npq->pq', R1, D3))
            fold = np.outer(fpsi[:, iq1], fpsi[:, iq2])
            w = 1.0 if iq1 == iq2 else 2.0
            f_pert += 0.5 * w * np.einsum(
                'npq,pq->n', np.conj(D3), fold * alpha1_blocks[p],
                optimize=True)
        return f_pert, d2v_blocks

    # ---------------------------------------------------------------
    def _fold_alpha1(self, alpha1_flat):
        """Fold the fine-side f_psi factors into alpha1 (prefiltered mode)."""
        blocks = self._unflatten_blocks(np.array(alpha1_flat))
        folded = []
        for pair_idx, (iq1, iq2) in enumerate(self.unique_pairs):
            fold = np.outer(self._fpsi_fine[:, iq1], self._fpsi_fine[:, iq2])
            folded.append(blocks[pair_idx] * fold)
        return self._flatten_blocks(folded)

    def _call_slots(self, fields_z, fields_w, fields_v, R1, alpha1_flat,
                    compute_d3, compute_d4, d3_channels=(True, True, True)):
        """One slot-resolved kernel call, parallel over (config, sym)."""
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
            indices.append([start + 1, stop])

        unique_pairs_arr = np.array(self.unique_pairs, dtype=np.int32) + 1
        valid_modes = np.array(self.valid_modes_q, dtype=np.bool_)
        iq_pert_jl = int(self.iq_pert) + 1

        Xz, Yz = fields_z
        Xw, Yw = fields_w
        Xv, Yv = fields_v

        def get_combined(start_end):
            return jl.get_perturb_averages_qspace_slots(
                Xz, Yz, Xw, Yw, Xv, Yv,
                self.w_q, self.rho, R1, alpha1_flat,
                float(self.T), bool(compute_d3), bool(compute_d4),
                iq_pert_jl, unique_pairs_arr,
                int(start_end[0]), int(start_end[1]),
                valid_modes,
                float(self.qspace_scale3), float(self.qspace_scale4),
                bool(self.qspace_prefiltered),
                True,
                bool(d3_channels[0]), bool(d3_channels[1]),
                bool(d3_channels[2]))

        return Parallel.GoParallel(get_combined, indices, "+")

    def _call_julia_qspace(self, R1, alpha1_flat):
        """Anharmonic averages: windowed multi-pass or single plain pass.

        In prefiltered mode the exact fine-side f_psi factors are folded
        into alpha1 (the kernel contracts alpha1 with the FILTERED fields).
        """
        # Hybrid tensor-D3 mode (plan section 5.8): the D3 channel is a
        # deterministic contraction of the centered-tensor vertex blocks
        # (with EXPLICIT f_psi factors, so it uses the unfolded alpha1);
        # only the D4 terms go through the stochastic kernel.
        if self.d3_mode == "tensor":
            if self._distributed:
                raise NotImplementedError(
                    "d3_mode='tensor' with the distributed loader is not "
                    "supported yet.")
            alpha1_blocks = self._unflatten_blocks(np.array(alpha1_flat))
            f_pert, d2v_blocks = self._apply_d3_tensor(R1, alpha1_blocks)
            if not self.ignore_v4:
                a1_kernel = self._fold_alpha1(alpha1_flat) \
                    if self.qspace_prefiltered else alpha1_flat
                plain = (self.X_q, self.Y_q)
                combined = self._call_slots(plain, plain, plain, R1,
                                            a1_kernel, False, True)
                f_pert = f_pert + combined[:self.n_bands]
                d4_blocks = self._unflatten_blocks(combined[self.n_bands:])
                d2v_blocks = [a + b for a, b in zip(d2v_blocks, d4_blocks)]
            return f_pert, d2v_blocks

        if self.qspace_prefiltered:
            alpha1_flat = self._fold_alpha1(alpha1_flat)

        plain = (self.X_q, self.Y_q)

        if (self.window_design not in ("minimal_image", "asr", "atomic",
                                       "atomic_delta", "atomic_asr")
                or self._window_passes is None):
            if self._distributed:
                return super()._call_julia_qspace(R1, alpha1_flat)
            # single plain pass through the batched slot kernel
            combined = self._call_slots(plain, plain, plain, R1, alpha1_flat,
                                        True, not self.ignore_v4)
            f_pert = combined[:self.n_bands]
            return f_pert, self._unflatten_blocks(combined[self.n_bands:])

        # D3 terms: sum of windowed passes, both (w,v) orientations averaged
        # (preserves the transpose symmetry of the pair blocks). The plain
        # estimator normalization corresponds to sum_r S_r = L*M with the
        # design windows unnormalized (plain pass = all-ones), so no extra
        # prefactor appears here.
        combined = None
        for pinfo in self._window_passes:
            if self.window_design in ("atomic", "atomic_delta",
                                      "atomic_asr"):
                channel, iz, iw, iv = pinfo
                fz = self._field_sets[iz]
                fw = self._field_sets[iw]
                fv = self._field_sets[iv]
                d3_channels = (channel == 0, channel == 1, channel == 2)
                part = self._call_slots(fz, fw, fv, R1, alpha1_flat,
                                        True, False,
                                        d3_channels=d3_channels)
            else:
                iz, iw, iv = pinfo
                fz = self._field_sets[iz]
                fw = self._field_sets[iw]
                fv = self._field_sets[iv]
                r1 = self._call_slots(fz, fw, fv, R1, alpha1_flat,
                                      True, False)
                if iw == iv:
                    part = r1
                else:
                    r2 = self._call_slots(fz, fv, fw, R1, alpha1_flat,
                                          True, False)
                    part = 0.5 * (r1 + r2)
            combined = part if combined is None else combined + part

        # D4 terms: single plain-window pass (plan section 5.4)
        if not self.ignore_v4:
            combined = combined + self._call_slots(
                plain, plain, plain, R1, alpha1_flat, False, True)

        f_pert = combined[:self.n_bands]
        d2v_blocks = self._unflatten_blocks(combined[self.n_bands:])
        return f_pert, d2v_blocks

    # ---------------------------------------------------------------
    def build_q_pair_map(self, iq_pert):
        """O(N_f) pair map via integer mesh-index arithmetic."""
        if self._fine_idx is None:
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

        # tensor-D3 mode: the vertex blocks depend on q_pert via the pairs
        if self.d3_mode == "tensor":
            self._build_d3_blocks()

    # ---------------------------------------------------------------
    def find_fine_q(self, q):
        """Index of a (Cartesian) q-vector in the fine mesh, O(1)."""
        return self._q_lookup[_mesh_key(q, self.uci_structure, self.fine_mesh)]
