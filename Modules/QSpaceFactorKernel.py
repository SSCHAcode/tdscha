"""
Symmetric-power factorization of the q-space centering kernel
=============================================================

Implements the constrained geometry-kernel fit of
report/interpolation/new_plan.tex for the tensor-free interpolation of the
D3 (rank-3) and D4 (rank-4) anharmonic vertices of the q-space TDSCHA
Lanczos.

The object being fitted is the image-assignment kernel K^(n) that
distributes each periodic force-constant entry among its supercell
replicas ("centering").  It is represented as

    K^(n)  =  K0  +  sum_xi c_xi * u_xi,

where K0 is the plain (tent) kernel of the unwindowed estimator and each
correction candidate u_xi is built from ONE one-leg factor window
A_xi(a, u) (primitive atom a, extended integer cell u) used on EVERY leg:

    type-1:  u_xi = corr_n(A_xi) - corr_n(A0)        (A0 = plain window)
    type-0:  u_xi = corr_n(A_xi)

with the translation-orbit-summed effective kernel

    corr_n(A)(a1; (a2,d2), ..., (an,dn))
        = sum_u  A(a1, u) * prod_s A(a_s, u + d_s).

KEY SIMPLIFICATION w.r.t. the new_plan.tex fitting pipeline
-----------------------------------------------------------
Every factor window is constrained to the UNIFORM-CLASS-SUM manifold:

    sum_p A(a, d + L p) = s_A   for every folded class (a, d),

with s_A = 1 (type-1) or s_A = 0 (type-0).  On this manifold ALL the hard
linear constraints of the tex hold structurally, candidate by candidate:

 *  partition of unity:  sum_images u_xi = s_A^n - s_A0^n = 0 for every
    folded tuple  =>  the correction never changes the commensurate
    tensor;
 *  MORE strongly, the commensurate identity holds PER CONFIGURATION:
    at a commensurate q the windowed Bloch field collapses to the
    class-sum-weighted transform, which for s_A = 1 is EXACTLY the plain
    field, so (A-pass - plain-pass) = 0 for every single configuration
    (and s_A = 0 fields vanish identically at commensurate q);
 *  ASR on every leg: the one-leg image sum H_s = prod_{s'!=s} A * s_A is
    independent of the summed (atom, cell)  =>  the extended tensor
    inherits the acoustic sum rule from any periodic tensor satisfying
    it (eq. asr-constant-image-sum of the tex);
 *  permutation symmetry: equal-factor powers are symmetric term by term.

Consequently the constraint matrix C_n U of the tex vanishes identically,
the nullspace basis Z is the identity, and the constrained fit
(eq. variable-projection) reduces to an ORDINARY least-squares problem in
the coefficients c.  The Gram matrix is evaluated matrix-free with the
symmetric-power identity  < corr_n(A), corr_n(B) > = sum_v g_AB(v)^n
(eq. symmetric-power-gram of the tex, generalized to the orbit-summed
kernels), so no dense rank-n object is ever formed.

The geometry target is the complete-graph minimal-image kernel
(eq. cluster-cost): for every folded atom/cell tuple, the image tuples
minimizing  sum_{s<t} |x_s - x_t|^rho  get the whole class weight, exact
ties split evenly.  This is the true four-leg target that the force-star
D4 assignment fails to represent (the documented SnTe TV=0.61 failure).

Everything here is geometry-only: no force-constant value is ever used.

Production constrained fit
--------------------------
``fit_constrained_factors`` implements the general coefficient-space
procedure without materializing T, K, U, or C.  It streams or samples folded
geometry classes, evaluates target minimizers on demand, builds the orbit
Gram from one-leg correlations, and forms the partition/ASR constraint Gram
analytically.  Its default large-cell dictionary uses sparse local windows
minus supercell-translated copies.  These factors have O(1) entries and zero
folded class sums even when N_c is enormous.  The plain kernel is handled by
closed-form or sampled correlations and is never stored.  An optional outer
width search rebuilds the factors, constraint nullspace, and coefficient fit.
"""

from __future__ import print_function
from __future__ import division

import itertools
import hashlib
import json
import os
import warnings

import numpy as np

__all__ = ["FactorWindow", "FactorFit", "GeometryTargetStream",
           "fit_symmetric_factors", "fit_constrained_factors",
           "optimize_constrained_factors",
           "build_target_tuples", "eval_corr_at_tuples",
           "default_dictionary", "local_dictionary",
           "sparse_feasible_dictionary", "get_factor_fit"]


# =========================================================================
# One-leg factor windows
# =========================================================================

class FactorWindow(object):
    """A one-leg factor A(a, u) on extended integer cells.

    Parameters
    ----------
    entries : dict {(atom, (u1, u2, u3)) : float}
        Window values on extended cells (same key convention as the
        window maps consumed by QSpaceInterpolation._window_map_to_qweights).
    class_sum : float or None
        The (uniform) folded-class sum this window is constrained to.
        Must be 0.0 or 1.0 for termwise-feasible windows. ``None`` marks a
        general local factor whose constraints are enforced only after the
        kernel terms are assembled.
    label : str
        Human-readable tag for diagnostics.
    """

    def __init__(self, entries, class_sum, label=""):
        self.entries = dict(entries)
        self.class_sum = None if class_sum is None else float(class_sum)
        self.label = label

    # -----------------------------------------------------------------
    def arrays(self):
        """(atoms(K,), cells(K,3), values(K,)) of the nonzero entries."""
        keys = sorted(self.entries)
        atoms = np.array([k[0] for k in keys], dtype=int)
        cells = np.array([k[1] for k in keys], dtype=int).reshape(-1, 3)
        vals = np.array([self.entries[k] for k in keys], dtype=np.float64)
        return atoms, cells, vals

    # -----------------------------------------------------------------
    def check_class_sums(self, supercell, tol=1e-10, nat=None):
        """Max deviation of the folded class sums from self.class_sum.

        When ``nat`` is given, every one of the ``nat * prod(supercell)``
        folded classes is checked. Missing classes then have sum zero, so a
        type-1 window fails while a type-0 window remains valid. Without
        ``nat`` only atoms present in the entries can be enumerated; callers
        enforcing feasibility must therefore pass it explicitly.
        """
        if self.class_sum is None:
            raise ValueError("A general factor has no uniform class sum")
        L = np.asarray(supercell, dtype=int)
        sums = {}
        for (a, u), v in self.entries.items():
            d = tuple(np.asarray(u, dtype=int) % L)
            sums[(a, d)] = sums.get((a, d), 0.0) + v
        if self.class_sum == 0.0:
            # Missing sparse classes also sum to zero, so type-0 validation
            # depends only on occupied classes and is independent of N_c.
            return max([abs(value) for value in sums.values()] + [0.0])
        atoms = range(int(nat)) if nat is not None \
            else sorted(set(a for a, _ in self.entries))
        classes = itertools.product(
            range(L[0]), range(L[1]), range(L[2]))
        classes = list(classes)
        dev = 0.0
        for a in atoms:
            for d in classes:
                dev = max(dev, abs(sums.get((a, d), 0.0)
                                   - self.class_sum))
        if not sums and nat is None:
            dev = abs(self.class_sum)
        return dev

    def n_classes(self, supercell):
        L = np.asarray(supercell, dtype=int)
        return len(set((a, tuple(np.asarray(u, dtype=int) % L))
                       for (a, u) in self.entries))

    def folded_sums(self, supercell, nat):
        """Dense folded class sums with shape ``(nat, *supercell)``.

        This is a one-leg object of size ``nat * N_c``.  It is the largest
        supercell-sized array needed by the analytic constraint Gram; no
        rank-n array is constructed.
        """
        L = np.asarray(supercell, dtype=int)
        out = np.zeros((int(nat),) + tuple(L), dtype=np.float64)
        for (a, u), value in self.entries.items():
            d = tuple(np.asarray(u, dtype=int) % L)
            out[(int(a),) + d] += value
        return out


def plain_window(nat, supercell):
    """The plain (all-ones fundamental-domain) window A0; class sums = 1."""
    L = np.asarray(supercell, dtype=int)
    entries = {}
    for a in range(nat):
        for d in itertools.product(range(L[0]), range(L[1]), range(L[2])):
            entries[(a, d)] = 1.0
    return FactorWindow(entries, 1.0, label="plain")


# =========================================================================
# Geometry: complete-graph minimal-image target
# =========================================================================

def _leg_image_shortlist(structure, supercell, far, anchor_pos, b, d,
                         margin):
    """Extended images of folded (b, d) sorted by distance to anchor_pos,
    truncated to dist <= min_dist + margin.  Returns (cells(K,3), pos(K,3),
    dist(K,))."""
    L = np.asarray(supercell, dtype=int)
    reps = np.array(list(itertools.product(range(-far, far + 1), repeat=3)),
                    dtype=int)
    ext = np.asarray(d, dtype=int)[None, :] + reps * L[None, :]
    pos = structure.coords[b][None, :] + ext @ structure.unit_cell
    dist = np.linalg.norm(pos - anchor_pos[None, :], axis=1)
    keep = dist <= dist.min() + margin
    order = np.argsort(dist[keep])
    return ext[keep][order], pos[keep][order], dist[keep][order]


def _decode_folded_state(index, nat, supercell):
    """Decode ``atom * N_c + flat_cell`` without a state lookup table."""
    L = tuple(np.asarray(supercell, dtype=int))
    n_c = int(np.prod(L))
    atom = int(index) // n_c
    cell = np.asarray(np.unravel_index(int(index) % n_c, L), dtype=int)
    return atom, cell


def _class_minimizers(structure, supercell, order, a1, legs, far,
                      cost_power, tie_tol, max_combinations=200000):
    """Minimal complete-graph images for one folded class.

    The routine holds only one class in memory.  When the Cartesian product
    of image shortlists is too large, a depth-first branch-and-bound search
    replaces the dense combination array.
    """
    L = np.asarray(supercell, dtype=int)
    n_c = int(np.prod(L))
    n_legs = int(order)
    rho = float(cost_power)
    anchor = structure.coords[int(a1)]

    candidates = []
    nearest_pos = []
    for b, d in legs:
        ext, pos, dist = _leg_image_shortlist(
            structure, L, far, anchor, int(b), np.asarray(d, dtype=int),
            margin=np.inf)
        candidates.append([ext, pos, dist])
        nearest_pos.append(pos[0])

    # A valid nearest-to-anchor tuple gives an upper bound.  Any image whose
    # anchor edge already exceeds it cannot occur in a minimizing cluster.
    upper = 0.0
    for pos in nearest_pos:
        value = np.linalg.norm(pos - anchor)
        upper += value ** rho if rho != 1 else value
    for i in range(len(nearest_pos)):
        for j in range(i + 1, len(nearest_pos)):
            value = np.linalg.norm(nearest_pos[i] - nearest_pos[j])
            upper += value ** rho if rho != 1 else value
    for item in candidates:
        edge = item[2] ** rho if rho != 1 else item[2]
        keep = edge <= upper + tie_tol
        item[0], item[1], item[2] = item[0][keep], item[1][keep], item[2][keep]

    def edge_cost(distance):
        return distance ** rho if rho != 1 else distance

    sizes = [len(item[0]) for item in candidates]
    n_combinations = int(np.prod(sizes, dtype=object))
    winners = []
    best = float(upper)

    if n_combinations <= int(max_combinations):
        combos = np.asarray(list(itertools.product(
            *[range(size) for size in sizes])), dtype=int)
        positions = np.stack([candidates[j][1][combos[:, j]]
                              for j in range(n_legs - 1)], axis=1)
        costs = np.zeros(len(combos), dtype=np.float64)
        for j in range(n_legs - 1):
            costs += edge_cost(np.linalg.norm(
                positions[:, j] - anchor[None, :], axis=1))
        for j in range(n_legs - 1):
            for k in range(j + 1, n_legs - 1):
                costs += edge_cost(np.linalg.norm(
                    positions[:, j] - positions[:, k], axis=1))
        best = float(costs.min())
        winners = [tuple(row) for row in combos[costs <= best + tie_tol]]
    else:
        chosen = []

        def visit(depth, partial):
            nonlocal best, winners
            if partial > best + tie_tol:
                return
            if depth == n_legs - 1:
                key = tuple(chosen)
                if partial < best - tie_tol:
                    best = partial
                    winners = [key]
                elif partial <= best + tie_tol:
                    winners.append(key)
                return
            for index, pos in enumerate(candidates[depth][1]):
                value = partial + edge_cost(np.linalg.norm(pos - anchor))
                for previous_depth, previous_index in enumerate(chosen):
                    previous = candidates[previous_depth][1][previous_index]
                    value += edge_cost(np.linalg.norm(pos - previous))
                if value <= best + tie_tol:
                    chosen.append(index)
                    visit(depth + 1, value)
                    chosen.pop()

        visit(0, 0.0)

    if not winners:
        raise RuntimeError("Geometry oracle found no image minimizer")
    atoms = np.empty((len(winners), n_legs), dtype=int)
    cells = np.zeros((len(winners), n_legs, 3), dtype=int)
    atoms[:, 0] = int(a1)
    for row, combo in enumerate(winners):
        for leg, ((atom, _), image_index) in enumerate(zip(legs, combo), 1):
            atoms[row, leg] = int(atom)
            cells[row, leg] = candidates[leg - 1][0][image_index]
    weights = np.full(len(winners), float(n_c) / len(winners))
    return atoms, cells, weights


class GeometryTargetStream(object):
    """Re-iterable complete-graph target oracle with bounded memory.

    ``sample_classes=None`` streams every folded class exactly.  A positive
    value draws that many uniformly weighted classes with a reproducible
    seed.  Sampling changes setup time, not peak memory; each sampled class
    represents ``total_classes / sample_classes`` exact classes.
    """

    def __init__(self, structure, supercell, order, far=2, cost_power=1,
                 tie_tol=1e-6, sample_classes=None, seed=0,
                 batch_tuples=4096, max_combinations=200000):
        self.structure = structure
        self.supercell = tuple(np.asarray(supercell, dtype=int))
        self.order = int(order)
        self.far = int(far)
        self.cost_power = float(cost_power)
        self.tie_tol = float(tie_tol)
        self.sample_classes = None if sample_classes is None \
            else int(sample_classes)
        self.seed = int(seed)
        self.batch_tuples = int(batch_tuples)
        self.max_combinations = int(max_combinations)
        self.nat = int(structure.N_atoms)
        self.nc = int(np.prod(self.supercell))
        self.nstates = self.nat * self.nc
        self.total_classes = self.nat * self.nstates ** (self.order - 1)
        if self.sample_classes is not None and self.sample_classes <= 0:
            raise ValueError("sample_classes must be positive or None")

    @property
    def classes_visited(self):
        return self.total_classes if self.sample_classes is None \
            else self.sample_classes

    @property
    def class_scale(self):
        return 1.0 if self.sample_classes is None else \
            float(self.total_classes) / self.sample_classes

    def _iter_classes(self):
        if self.sample_classes is None:
            for a1 in range(self.nat):
                for encoded in itertools.product(
                        range(self.nstates), repeat=self.order - 1):
                    yield a1, [_decode_folded_state(
                        value, self.nat, self.supercell) for value in encoded]
            return
        rng = np.random.default_rng(self.seed)
        anchors = np.arange(self.sample_classes, dtype=int) % self.nat
        rng.shuffle(anchors)
        for a1 in anchors:
            encoded = rng.integers(0, self.nstates, size=self.order - 1)
            yield int(a1), [_decode_folded_state(
                value, self.nat, self.supercell) for value in encoded]

    def iter_batches(self):
        atoms_buffer = []
        cells_buffer = []
        weights_buffer = []
        classes_in_batch = 0
        for a1, legs in self._iter_classes():
            atoms, cells, weights = _class_minimizers(
                self.structure, self.supercell, self.order, a1, legs,
                self.far, self.cost_power, self.tie_tol,
                max_combinations=self.max_combinations)
            atoms_buffer.append(atoms)
            cells_buffer.append(cells)
            weights_buffer.append(weights)
            classes_in_batch += 1
            if sum(len(item) for item in weights_buffer) >= self.batch_tuples:
                yield (np.concatenate(atoms_buffer),
                       np.concatenate(cells_buffer),
                       np.concatenate(weights_buffer),
                       self.class_scale, classes_in_batch)
                atoms_buffer, cells_buffer, weights_buffer = [], [], []
                classes_in_batch = 0
        if weights_buffer:
            yield (np.concatenate(atoms_buffer), np.concatenate(cells_buffer),
                   np.concatenate(weights_buffer), self.class_scale,
                   classes_in_batch)


def build_target_tuples(structure, supercell, order, far=2, cost_power=1,
                        tie_tol=1e-6):
    """Materialize the complete-graph target for small validation cells.

    For every folded class (a1; (a2,d2), ..., (an,dn)) -- leg 1 pinned at
    cell 0 by the translation gauge -- find the image tuples
    (delta_s = d_s + L m_s) minimizing the complete-graph cluster cost

        D_n = sum_{1<=s<t<=n} |x_s - x_t|^cost_power

    with x_1 = tau_{a1}, x_s = tau_{a_s} + delta_s . cell.  Exact ties are
    split evenly.  The class weight is normalized to N_c (the class sum of
    the plain tent kernel K0), so the raw target and K0 live on the same
    scale and satisfy the same partition rows.

    Returns
    -------
    atoms : ndarray(T, order) int      -- atom of each leg (leg 0 = a1)
    cells : ndarray(T, order, 3) int   -- extended cell (leg 0 = 0)
    weights : ndarray(T,) float        -- N_c / n_ties per minimal tuple
    """
    stream = GeometryTargetStream(
        structure, supercell, order, far=far, cost_power=cost_power,
        tie_tol=tie_tol, sample_classes=None)
    batches = list(stream.iter_batches())
    return (np.concatenate([batch[0] for batch in batches]),
            np.concatenate([batch[1] for batch in batches]),
            np.concatenate([batch[2] for batch in batches]))


# =========================================================================
# Matrix-free kernel algebra
# =========================================================================

def window_cross_correlation(win_a, win_b):
    """g_AB(v) = sum_{a,u} A(a,u) B(a,u+v) as {v_tuple: value}."""
    at_a, cl_a, va = win_a.arrays()
    at_b, cl_b, vb = win_b.arrays()
    out = {}
    for a in set(at_a.tolist()) & set(at_b.tolist()):
        ia = np.where(at_a == a)[0]
        ib = np.where(at_b == a)[0]
        # all pairwise differences v = u_b - u_a
        dv = cl_b[ib][None, :, :] - cl_a[ia][:, None, :]     # (Ka, Kb, 3)
        vals = va[ia][:, None] * vb[ib][None, :]
        dv = dv.reshape(-1, 3)
        vals = vals.ravel()
        # accumulate
        keys, inv = np.unique(dv, axis=0, return_inverse=True)
        acc = np.zeros(len(keys))
        np.add.at(acc, inv, vals)
        for k, v in zip(keys, acc):
            t = tuple(k)
            out[t] = out.get(t, 0.0) + v
    return out


def corr_inner_product(win_a, win_b, order):
    """< corr_n(A), corr_n(B) > = sum_v g_AB(v)^n  (orbit-summed kernels)."""
    g = window_cross_correlation(win_a, win_b)
    return float(sum(v ** order for v in g.values()))


def _cyclic_cross_correlation(array_a, array_b):
    """sum_(atom,d) A(atom,d) B(atom,d+v) on the folded quotient."""
    if array_a.shape != array_b.shape:
        raise ValueError("Folded arrays must have identical shapes")
    out = np.zeros(array_a.shape[1:], dtype=np.float64)
    axes = tuple(range(array_a.ndim - 1))
    for atom in range(array_a.shape[0]):
        fa = np.fft.fftn(array_a[atom])
        fb = np.fft.fftn(array_b[atom])
        out += np.fft.ifftn(np.conj(fa) * fb).real
    # Numerical FFT noise in exact zero-sum constraints otherwise creates
    # spurious tiny positive eigenvalues in the constraint Gram.
    scale = max(float(np.max(np.abs(out))), 1.0)
    out[np.abs(out) < 1e-13 * scale] = 0.0
    return out


def _sparse_folded_sums(window, supercell):
    L = np.asarray(supercell, dtype=int)
    out = {}
    for (atom, cell), value in window.entries.items():
        key = (int(atom), tuple(np.asarray(cell, dtype=int) % L))
        out[key] = out.get(key, 0.0) + float(value)
    return {key: value for key, value in out.items() if abs(value) > 1e-15}


def _sparse_cyclic_cross_correlation(folded_a, folded_b, supercell):
    """Sparse quotient correlation keyed only by occupied shifts."""
    L = np.asarray(supercell, dtype=int)
    by_atom_a = {}
    by_atom_b = {}
    for (atom, cell), value in folded_a.items():
        by_atom_a.setdefault(atom, []).append((np.asarray(cell), value))
    for (atom, cell), value in folded_b.items():
        by_atom_b.setdefault(atom, []).append((np.asarray(cell), value))
    out = {}
    for atom in set(by_atom_a) & set(by_atom_b):
        for cell_a, value_a in by_atom_a[atom]:
            for cell_b, value_b in by_atom_b[atom]:
                shift = tuple((cell_b - cell_a) % L)
                out[shift] = out.get(shift, 0.0) + value_a * value_b
    return {key: value for key, value in out.items() if abs(value) > 1e-15}


def constraint_gram(windows, plain, supercell, nat, order,
                    include_partition=True, include_asr=True,
                    reference_weights=None):
    """Analytic ``(C_n U)^T(C_n U)`` for orbit-power corrections.

    Candidate ``i`` is ``corr_n(windows[i]) - r_i corr_n(plain)``. Partition
    rows are orbit powers of the folded class sums.  ASR rows replace one
    factor by its folded sum with the constant component removed.  Their
    pairwise inner products reduce to one-leg correlations, so this routine
    allocates only ``O(R^2 + R * nat * N_c)`` memory.

    Returns the normalized combined Gram and the two unnormalized pieces.
    A coefficient vector is exactly feasible iff it is in the nullspace of
    the combined positive-semidefinite Gram.
    """
    wins = list(windows)
    count = len(wins)
    if reference_weights is None:
        reference_weights = np.ones(count, dtype=np.float64)
    reference_weights = np.asarray(reference_weights, dtype=np.float64)
    if reference_weights.shape != (count,):
        raise ValueError("reference_weights has the wrong shape")
    n_quotient = int(nat) * int(np.prod(np.asarray(supercell, dtype=int)))
    n_shifts = int(np.prod(np.asarray(supercell, dtype=int)))
    folded = [_sparse_folded_sums(w, supercell) for w in wins]
    folded_totals = [float(sum(item.values())) for item in folded]
    partition = np.zeros((count, count), dtype=np.float64)
    asr = np.zeros((count, count), dtype=np.float64)

    quotient_raw = np.zeros((count + 1, count + 1), dtype=np.float64)
    cyclic_cache = {}
    extended_cache = {}
    for i in range(count):
        for j in range(i, count):
            h = _sparse_cyclic_cross_correlation(
                folded[i], folded[j], supercell)
            cyclic_cache[(i, j)] = h
            quotient_raw[i, j] = quotient_raw[j, i] = float(
                sum(value ** int(order) for value in h.values()))
            if include_asr:
                extended_cache[(i, j)] = window_cross_correlation(
                    wins[i], wins[j])
    # The plain folded factor is one on every atom/class.  Its cyclic
    # correlation with a sparse factor is the factor's total folded sum at
    # every quotient shift; no dense all-ones array is needed.
    for i in range(count):
        quotient_raw[i, -1] = quotient_raw[-1, i] = \
            n_shifts * folded_totals[i] ** int(order)
    quotient_raw[-1, -1] = \
        n_shifts * float(n_quotient) ** int(order)

    if include_partition:
        pp = quotient_raw[-1, -1]
        for i in range(count):
            for j in range(count):
                partition[i, j] = (quotient_raw[i, j]
                    - reference_weights[j] * quotient_raw[i, -1]
                    - reference_weights[i] * quotient_raw[j, -1]
                    + reference_weights[i] * reference_weights[j] * pp)

    if include_asr:
        for i in range(count):
            for j in range(i, count):
                h = cyclic_cache[(i, j)]
                mean_product = (folded_totals[i] * folded_totals[j]
                                / float(n_quotient))
                g = extended_cache[(i, j)]
                value = 0.0
                L = np.asarray(supercell, dtype=int)
                for shift, corr in g.items():
                    residue = tuple(np.asarray(shift, dtype=int) % L)
                    h_centered = h.get(residue, 0.0) - mean_product
                    value += h_centered * corr ** (int(order) - 1)
                # All legs give the same residual norm for a symmetric power.
                asr[i, j] = asr[j, i] = float(order) * value

    # A correctly labelled uniform-class-sum candidate is analytically
    # feasible.  Set its row and column to exact zero instead of normalizing
    # roundoff left by correlations of large, nearly cancelling sums.
    individually_feasible = []
    for index, window in enumerate(wins):
        expected = window.class_sum
        feasible = expected in (0.0, 1.0) and \
            abs(reference_weights[index] - expected) < 1e-14 and \
            window.check_class_sums(supercell, nat=nat) < 1e-10
        individually_feasible.append(feasible)
    for index, feasible in enumerate(individually_feasible):
        if feasible:
            partition[index, :] = partition[:, index] = 0.0
            asr[index, :] = asr[:, index] = 0.0

    def normalized(matrix):
        scale = max(float(np.max(np.abs(np.diag(matrix)))), 0.0)
        return matrix / scale if scale > 0 else matrix.copy()

    combined = normalized(partition) + normalized(asr)
    combined = 0.5 * (combined + combined.T)
    return combined, partition, asr


def constraint_nullspace(gram, rtol=1e-10, atol=1e-12):
    """Rank-revealing nullspace of a positive-semidefinite constraint Gram."""
    gram = 0.5 * (np.asarray(gram, dtype=np.float64)
                  + np.asarray(gram, dtype=np.float64).T)
    if gram.size == 0:
        return np.empty((gram.shape[0], 0)), np.empty(0), 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = max(float(np.max(eigenvalues)), 0.0)
    threshold = max(float(atol), float(rtol) * largest)
    keep = eigenvalues <= threshold
    return eigenvectors[:, keep], eigenvalues, threshold


def stream_target_overlaps(windows, stream, nat):
    """Estimate ``<T,P_i>`` and ``<T,T>`` without storing target tuples."""
    wins = list(windows)
    overlaps = np.zeros(len(wins), dtype=np.float64)
    target_norm2 = 0.0
    tuples_visited = 0
    classes_visited = 0
    for atoms, cells, weights, scale, nclasses in stream.iter_batches():
        for i, window in enumerate(wins):
            values = eval_corr_at_tuples(window, atoms, cells, nat)
            overlaps[i] += scale * float(np.dot(weights, values))
        target_norm2 += scale * float(np.dot(weights, weights))
        tuples_visited += len(weights)
        classes_visited += nclasses
    return overlaps, target_norm2, {
        "classes_visited": int(classes_visited),
        "tuples_visited": int(tuples_visited),
        "total_classes": int(stream.total_classes),
        "sampled": stream.sample_classes is not None,
        "seed": int(stream.seed),
    }


def plain_corr_at_tuples(cells, supercell):
    """Orbit correlation of the fundamental-domain window, analytically."""
    cells = np.asarray(cells, dtype=int)
    L = np.asarray(supercell, dtype=int)
    result = np.ones(len(cells), dtype=np.float64)
    for axis in range(3):
        shifts = cells[:, :, axis]
        lower = np.maximum(0, np.max(-shifts, axis=1))
        upper = np.minimum(L[axis] - 1,
                           np.min(L[axis] - 1 - shifts, axis=1))
        result *= np.maximum(upper - lower + 1, 0)
    return result


def plain_power_norm(nat, supercell, order):
    """Exact ``<corr_n(A0),corr_n(A0)>`` in O(1) work."""
    n = int(order)
    if n not in (3, 4):
        raise ValueError("Closed-form plain norm supports order 3 or 4")

    def power_sum(limit):
        m = int(limit)
        if n == 3:
            return (m * (m + 1) // 2) ** 2
        return m * (m + 1) * (2 * m + 1) * \
            (3 * m * m + 3 * m - 1) // 30

    value = float(int(nat) ** n)
    for length in np.asarray(supercell, dtype=int):
        value *= float(int(length) ** n + 2 * power_sum(int(length) - 1))
    return value


def sparse_plain_inner_product(window, supercell, order, samples=20000,
                               seed=0, exact_limit=200000):
    """``<corr_n(A),corr_n(A0)>`` without constructing ``A0``.

    The orbit identity reduces the quantity to ``sum_v g_A0(v)^n``.  The
    shift box is enumerated when small and uniformly sampled otherwise.
    """
    atoms, cells, values = window.arrays()
    L = np.asarray(supercell, dtype=int)
    lo = -cells.max(axis=0)
    hi = L - 1 - cells.min(axis=0)
    widths = hi - lo + 1
    total = int(np.prod(widths, dtype=object))

    def evaluate(shifts):
        answer = np.empty(len(shifts), dtype=np.float64)
        for start in range(0, len(shifts), 2048):
            block = shifts[start:start + 2048]
            placed = cells[None, :, :] + block[:, None, :]
            inside = np.all((placed >= 0) & (placed < L[None, None, :]),
                            axis=2)
            answer[start:start + len(block)] = inside @ values
        return answer ** int(order)

    if total <= int(exact_limit):
        shifts = np.asarray(list(itertools.product(
            range(lo[0], hi[0] + 1), range(lo[1], hi[1] + 1),
            range(lo[2], hi[2] + 1))), dtype=int)
        values_n = evaluate(shifts)
        return float(np.sum(values_n)), {"sampled": False, "shifts": total,
                                         "standard_error": 0.0}
    rng = np.random.default_rng(int(seed))
    count = int(samples)
    shifts = np.column_stack([
        rng.integers(lo[axis], hi[axis] + 1, size=count)
        for axis in range(3)
    ])
    values_n = evaluate(shifts)
    estimate = float(total) * float(np.mean(values_n))
    standard_error = float(total) * float(np.std(values_n, ddof=1)) \
        / np.sqrt(count) if count > 1 else float("inf")
    return estimate, {"sampled": True, "shifts": count,
                      "total_shifts": total, "seed": int(seed),
                      "standard_error": standard_error}


def stream_target_overlaps_implicit_plain(windows, stream, nat, supercell):
    """Target overlaps for sparse factors plus analytic plain values."""
    wins = list(windows)
    overlaps = np.zeros(len(wins) + 1, dtype=np.float64)
    target_norm2 = 0.0
    tuples_visited = 0
    classes_visited = 0
    for atoms, cells, weights, scale, nclasses in stream.iter_batches():
        for i, window in enumerate(wins):
            values = eval_corr_at_tuples(window, atoms, cells, nat)
            overlaps[i] += scale * float(np.dot(weights, values))
        plain_values = plain_corr_at_tuples(cells, supercell)
        overlaps[-1] += scale * float(np.dot(weights, plain_values))
        target_norm2 += scale * float(np.dot(weights, weights))
        tuples_visited += len(weights)
        classes_visited += nclasses
    return overlaps, target_norm2, {
        "classes_visited": int(classes_visited),
        "tuples_visited": int(tuples_visited),
        "total_classes": int(stream.total_classes),
        "sampled": stream.sample_classes is not None,
        "seed": int(stream.seed),
    }


def _window_dense(win, nat):
    """(dense(nat, D0, D1, D2), origin(3,)) lookup array of a window."""
    atoms, cells, vals = win.arrays()
    lo = cells.min(axis=0)
    hi = cells.max(axis=0)
    dims = hi - lo + 1
    dense = np.zeros((nat,) + tuple(dims), dtype=np.float64)
    dense[atoms, cells[:, 0] - lo[0], cells[:, 1] - lo[1],
          cells[:, 2] - lo[2]] = vals
    return dense, lo


def eval_corr_at_tuples(win, atoms, cells, nat):
    """corr_n(A) evaluated at explicit tuples.

    corr(t) = sum_u A(a1, u) prod_{s>=1} A(a_s, u + delta_s)
    with delta_0 = 0 (leg 0 is the gauge anchor, its own factor evaluated
    at u).  atoms(T, n), cells(T, n, 3) -> values (T,).
    """
    dense, lo = _window_dense(win, nat)
    dims = np.array(dense.shape[1:], dtype=int)
    w_at, w_cl, w_val = win.arrays()

    T, n_legs = atoms.shape
    out = np.zeros(T, dtype=np.float64)

    # loop over the nonzero support of the anchor factor: u must be a
    # support point of A for SOME atom; leg-0 weight is A(a1, u).
    u_cells = np.unique(w_cl, axis=0)
    for u in u_cells:
        # leg products: A(a_s, cells_s + u) for every leg incl. leg 0
        prod = np.ones(T, dtype=np.float64)
        for s in range(n_legs):
            idx = cells[:, s, :] + u[None, :] - lo[None, :]
            ok = np.all((idx >= 0) & (idx < dims[None, :]), axis=1)
            vals = np.zeros(T)
            if np.any(ok):
                vals[ok] = dense[atoms[ok, s], idx[ok, 0], idx[ok, 1],
                                 idx[ok, 2]]
            prod *= vals
            if not prod.any():
                break
        out += prod
    return out


# =========================================================================
# Candidate dictionaries
# =========================================================================

def minimal_image_window(structure, supercell, a0, far=3):
    """Atomic minimal-image selector relative to atom a0 (unit class sums).

    For every folded class (b, d), the extended images minimizing
    |tau_b - tau_{a0} + delta . cell| share unit weight.  This is the
    one-leg window of the validated SnTe atomic centering (plan 5.9)."""
    L = np.asarray(supercell, dtype=int)
    entries = {}
    for b in range(structure.N_atoms):
        for d in itertools.product(range(L[0]), range(L[1]), range(L[2])):
            ext, pos, dist = _leg_image_shortlist(
                structure, L, far, structure.coords[a0], b,
                np.array(d, dtype=int), margin=0.0)
            keep = np.where(dist <= dist[0] + 1e-6)[0]
            for i in keep:
                entries[(b, tuple(ext[i]))] = 1.0 / len(keep)
    return FactorWindow(entries, 1.0, label="mimg(a%d,far%d)" % (a0, far))


def gaussian_window(structure, supercell, center, sigma, far=2,
                    rcut_sigma=3.0, label=None):
    """Gaussian-decay window centered on an arbitrary Cartesian point,
    per-class normalized to unit class sums (type-1).  Every folded class
    keeps at least its minimal image, so the normalization is always well
    defined."""
    L = np.asarray(supercell, dtype=int)
    entries = {}
    anchor = np.asarray(center, dtype=np.float64)
    for b in range(structure.N_atoms):
        for d in itertools.product(range(L[0]), range(L[1]), range(L[2])):
            ext, pos, dist = _leg_image_shortlist(
                structure, L, far, anchor, b, np.array(d, dtype=int),
                margin=max(rcut_sigma * sigma - 0.0, 0.0))
            keep = dist <= max(dist[0], rcut_sigma * sigma) + 1e-9
            ext, dist = ext[keep], dist[keep]
            w = np.exp(-0.5 * (dist / sigma) ** 2)
            w /= w.sum()
            for e, v in zip(ext, w):
                entries[(b, tuple(e))] = entries.get((b, tuple(e)), 0.0) + v
    return FactorWindow(entries, 1.0,
                        label=label or "gauss(s%.3g)" % sigma)


def local_gaussian_window(structure, center, sigma, cutoff_sigma=3.0,
                          label=None):
    """A genuinely local, unconstrained one-leg Gaussian factor.

    Unlike :func:`gaussian_window`, this function does not normalize every
    folded class.  Its support and construction cost are independent of the
    supercell size.  The L2 normalization fixes the otherwise arbitrary
    scale of a symmetric power; feasibility is imposed on the assembled
    kernel by :func:`constraint_gram`.
    """
    center = np.asarray(center, dtype=np.float64)
    sigma = float(sigma)
    cutoff = float(cutoff_sigma) * sigma
    if sigma <= 0 or cutoff <= 0:
        raise ValueError("sigma and cutoff_sigma must be positive")
    inverse_cell = np.linalg.inv(np.asarray(structure.unit_cell,
                                            dtype=np.float64))
    frac_radius = cutoff * np.linalg.norm(inverse_cell, axis=0)
    entries = {}
    for atom in range(structure.N_atoms):
        fractional_center = (center - structure.coords[atom]) @ inverse_cell
        lo = np.floor(fractional_center - frac_radius).astype(int) - 1
        hi = np.ceil(fractional_center + frac_radius).astype(int) + 1
        for cell in itertools.product(
                range(lo[0], hi[0] + 1), range(lo[1], hi[1] + 1),
                range(lo[2], hi[2] + 1)):
            position = structure.coords[atom] + np.asarray(cell) @ \
                structure.unit_cell
            distance = np.linalg.norm(position - center)
            if distance <= cutoff + 1e-12:
                entries[(atom, tuple(cell))] = np.exp(
                    -0.5 * (distance / sigma) ** 2)
    if not entries:
        raise ValueError("Local Gaussian support is empty")
    norm = np.sqrt(sum(value * value for value in entries.values()))
    entries = {key: value / norm for key, value in entries.items()}
    return FactorWindow(entries, None,
                        label=label or "local-gauss(s%.3g)" % sigma)


def window_difference(win_a, win_b, label=None):
    """A - B of two equal-class-sum windows: a type-0 (zero-sum) window."""
    if abs(win_a.class_sum - win_b.class_sum) > 1e-12:
        raise ValueError("window_difference needs equal class sums")
    entries = dict(win_a.entries)
    for k, v in win_b.entries.items():
        entries[k] = entries.get(k, 0.0) - v
    entries = {k: v for k, v in entries.items() if abs(v) > 1e-15}
    return FactorWindow(entries, 0.0,
                        label=label or "[%s - %s]" % (win_a.label,
                                                      win_b.label))


def _nn_distance(structure, supercell):
    """Shortest interatomic distance in the crystal."""
    L = np.asarray(supercell, dtype=int)
    d_nn = np.inf
    for a in range(structure.N_atoms):
        for b in range(structure.N_atoms):
            ext, p, dist = _leg_image_shortlist(
                structure, L, 1, structure.coords[a], b,
                np.zeros(3, dtype=int), margin=np.inf)
            d = dist[dist > 1e-6]
            if len(d):
                d_nn = min(d_nn, d.min())
    return d_nn


def _primitive_neighbor_images(structure, atom, other, far=1):
    cells = np.asarray(list(itertools.product(
        range(-far, far + 1), repeat=3)), dtype=int)
    positions = structure.coords[other][None, :] + cells @ structure.unit_cell
    distances = np.linalg.norm(positions - structure.coords[atom][None, :],
                               axis=1)
    order = np.argsort(distances)
    return cells[order], positions[order], distances[order]


def _primitive_nn_distance(structure):
    value = np.inf
    for atom in range(structure.N_atoms):
        for other in range(structure.N_atoms):
            _, _, distances = _primitive_neighbor_images(
                structure, atom, other)
            nonzero = distances[distances > 1e-8]
            if len(nonzero):
                value = min(value, float(nonzero[0]))
    return value


def default_dictionary(structure, supercell, far=3, sigmas=None,
                       bond_centers=True, differences=True):
    """Geometry-informed candidate windows.

    Type-1 (unit class sums): one minimal-image selector per primitive
    atom (the L=2 tie breaker) and Gaussian families centered on atoms
    (and optionally nearest-neighbor bond midpoints) with widths from
    sub-NN to the NN scale.  Type-0 (zero class sums, optional): window
    differences, which contribute genuinely new symmetric-power
    directions through their cross terms."""
    d_nn = _nn_distance(structure, supercell)
    if sigmas is None:
        sigmas = np.array([0.25, 0.4, 0.6, 0.9, 1.35]) * d_nn

    L = np.asarray(supercell, dtype=int)
    centers = [(structure.coords[a0], "a%d" % a0)
               for a0 in range(structure.N_atoms)]
    if bond_centers:
        seen = []
        for a0 in range(structure.N_atoms):
            for b in range(structure.N_atoms):
                ext, pos, dist = _leg_image_shortlist(
                    structure, L, 1, structure.coords[a0], b,
                    np.zeros(3, dtype=int), margin=np.inf)
                sel = np.abs(dist - d_nn) < 1e-6
                for p in pos[sel]:
                    mid = 0.5 * (structure.coords[a0] + p)
                    if not any(np.linalg.norm(mid - m) < 1e-6
                               for m in seen):
                        seen.append(mid)
        for i, m in enumerate(seen):
            centers.append((m, "b%d" % i))

    wins = []
    per_center_gauss = {}
    for a0 in range(structure.N_atoms):
        wins.append(minimal_image_window(structure, supercell, a0, far=far))
    for cpos, cname in centers:
        fam = []
        for s in sigmas:
            w = gaussian_window(structure, supercell, cpos, float(s),
                                far=min(far, 2),
                                label="gauss(%s,s%.3g)" % (cname, s))
            fam.append(w)
            wins.append(w)
        per_center_gauss[cname] = fam

    if differences:
        # zero-sum differences: adjacent widths on the same center, and
        # the atom-centered minimal-image window minus its tightest
        # Gaussian (localizes the tie-breaking information)
        for cname, fam in per_center_gauss.items():
            for i in range(len(fam) - 1):
                wins.append(window_difference(fam[i], fam[i + 1]))
        for a0 in range(structure.N_atoms):
            mimg = wins[a0]
            fam = per_center_gauss.get("a%d" % a0)
            if fam:
                wins.append(window_difference(mimg, fam[0]))
    # drop empty windows (differences that cancelled exactly, e.g. a
    # very tight Gaussian collapsing onto the minimal-image selector)
    wins = [w for w in wins
            if w.entries and max(abs(v) for v in w.entries.values()) > 1e-12]
    return wins


def local_dictionary(structure, supercell, sigmas=None, bond_centers=True,
                     cutoff_sigma=3.0):
    """Supercell-independent dictionary for the joint constrained fit."""
    d_nn = _primitive_nn_distance(structure)
    if sigmas is None:
        sigmas = np.asarray([0.30, 0.45, 0.65, 0.9, 1.25]) * d_nn
    centers = [(structure.coords[atom], "a%d" % atom)
               for atom in range(structure.N_atoms)]
    if bond_centers:
        seen = []
        for atom in range(structure.N_atoms):
            for other in range(structure.N_atoms):
                _, positions, distances = _primitive_neighbor_images(
                    structure, atom, other)
                for position in positions[np.abs(distances - d_nn) < 1e-6]:
                    midpoint = 0.5 * (structure.coords[atom] + position)
                    if not any(np.linalg.norm(midpoint - old) < 1e-6
                               for old in seen):
                        seen.append(midpoint)
        centers.extend((center, "b%d" % index)
                       for index, center in enumerate(seen))
    windows = []
    for center, name in centers:
        for sigma in sigmas:
            windows.append(local_gaussian_window(
                structure, center, float(sigma), cutoff_sigma=cutoff_sigma,
                label="local-gauss(%s,s%.3g)" % (name, sigma)))
    return windows


def shifted_window_difference(window, supercell, quotient_shift, label=None):
    """``window - translated(window)`` with an exact zero folded sum."""
    L = np.asarray(supercell, dtype=int)
    quotient_shift = np.asarray(quotient_shift, dtype=int)
    if quotient_shift.shape != (3,) or not np.any(quotient_shift):
        raise ValueError("quotient_shift must be a nonzero length-three vector")
    translation = L * quotient_shift
    entries = dict(window.entries)
    for (atom, cell), value in window.entries.items():
        shifted = (atom, tuple(np.asarray(cell, dtype=int) + translation))
        entries[shifted] = entries.get(shifted, 0.0) - value
    entries = {key: value for key, value in entries.items()
               if abs(value) > 1e-15}
    norm = np.sqrt(sum(value * value for value in entries.values()))
    entries = {key: value / norm for key, value in entries.items()}
    return FactorWindow(
        entries, 0.0,
        label=label or "deltaL(%s,%s)" % (window.label,
                                          tuple(quotient_shift)))


def sparse_feasible_dictionary(structure, supercell, sigmas=None,
                               bond_centers=False, cutoff_sigma=2.5,
                               quotient_shifts=None):
    """Large-supercell dictionary with O(1) support per exact factor.

    Each candidate is a compact local Gaussian minus a supercell-translated
    copy.  Its folded class sums vanish exactly, while its number of entries
    is independent of ``N_c``.  The support diameter grows with the
    supercell, which is unavoidable for a nonzero image-sum-null correction.
    """
    bases = local_dictionary(
        structure, supercell, sigmas=sigmas, bond_centers=bond_centers,
        cutoff_sigma=cutoff_sigma)
    if quotient_shifts is None:
        quotient_shifts = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    windows = []
    for base in bases:
        for shift in quotient_shifts:
            windows.append(shifted_window_difference(
                base, supercell, shift,
                label="deltaL(%s,%s)" % (base.label, tuple(shift))))
    return windows


# =========================================================================
# The fit
# =========================================================================

def prepare_fit_data(wins, a0w, target, order, nat):
    """Precompute the Gram of raw powers and the target overlaps.

    This is the expensive, dictionary-dependent part of the fit
    (independent of max_rank / ridge), returned in the form accepted by
    fit_symmetric_factors(precomputed=...).
    """
    t_atoms, t_cells, t_w = target
    allw = list(wins) + [a0w]
    R1 = len(allw)
    P = np.zeros((R1, R1))
    for i in range(R1):
        for j in range(i, R1):
            P[i, j] = P[j, i] = corr_inner_product(allw[i], allw[j], order)
    t_dot = np.zeros(R1)
    for i in range(R1):
        vals = eval_corr_at_tuples(allw[i], t_atoms, t_cells, nat)
        t_dot[i] = float(np.dot(t_w, vals))
    tt = float(np.dot(t_w, t_w))
    return P, t_dot, tt


class FactorFit(object):
    """Result of a symmetric-power kernel fit for one tensor order.

    Attributes
    ----------
    order : int                       tensor rank n (3 or 4)
    windows : list of FactorWindow    the retained factor windows
    coeffs : ndarray(R,)              signed coefficients c_xi
    plain_coeff : float               1 - sum(type-1 c_xi)
    diagnostics : dict                fit errors and feasibility residuals
    """

    def __init__(self, order, windows, coeffs, plain_coeff, diagnostics):
        self.order = int(order)
        self.windows = list(windows)
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.plain_coeff = float(plain_coeff)
        self.diagnostics = dict(diagnostics)

    def summary(self):
        d = self.diagnostics
        lines = ["FactorFit order=%d rank=%d plain_coeff=%.6f"
                 % (self.order, len(self.windows), self.plain_coeff),
                 "  |T-K0| = %.6g ; residual = %.6g (rel %.4f)"
                 % (d["norm_t_minus_k0"], d["residual"],
                    d["rel_residual"]),
                 "  target coverage: fit %.4f  (plain %.4f)"
                 % (d.get("coverage", float("nan")),
                    d.get("coverage_plain", float("nan"))),
                 "  correction L1 = %.3f (ridge %.1e)"
                 % (d.get("correction_l1", float("nan")),
                    d.get("ridge_used", float("nan"))),
                 "  max class-sum dev = %.2e" % d["max_class_dev"]]
        for w, c in zip(self.windows, self.coeffs):
            lines.append("    c = %+.6f   %s" % (c, w.label))
        return "\n".join(lines)

    def save(self, path):
        """Persist geometry factors and diagnostics without q-space fields."""
        atoms = []
        cells = []
        values = []
        offsets = [0]
        for window in self.windows:
            at, cl, val = window.arrays()
            atoms.append(at)
            cells.append(cl)
            values.append(val)
            offsets.append(offsets[-1] + len(val))
        np.savez_compressed(
            path, order=np.asarray(self.order), coeffs=self.coeffs,
            plain_coeff=np.asarray(self.plain_coeff),
            atoms=np.concatenate(atoms) if atoms else np.empty(0, dtype=int),
            cells=np.concatenate(cells) if cells else np.empty((0, 3), dtype=int),
            values=np.concatenate(values) if values else np.empty(0),
            offsets=np.asarray(offsets, dtype=int),
            class_sums=np.asarray([
                np.nan if window.class_sum is None else window.class_sum
                for window in self.windows]),
            labels=np.asarray([window.label for window in self.windows]),
            diagnostics=np.asarray(json.dumps(self.diagnostics,
                                               sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            offsets = data["offsets"]
            windows = []
            for index in range(len(offsets) - 1):
                lo, hi = int(offsets[index]), int(offsets[index + 1])
                entries = {
                    (int(atom), tuple(cell)): float(value)
                    for atom, cell, value in zip(
                        data["atoms"][lo:hi], data["cells"][lo:hi],
                        data["values"][lo:hi])
                }
                class_sum = float(data["class_sums"][index])
                if np.isnan(class_sum):
                    class_sum = None
                windows.append(FactorWindow(
                    entries, class_sum, label=str(data["labels"][index])))
            diagnostics = json.loads(str(data["diagnostics"].item()))
            return cls(int(data["order"].item()), windows, data["coeffs"],
                       float(data["plain_coeff"].item()), diagnostics)


def fit_symmetric_factors(structure, supercell, order, far=2, cost_power=1,
                          dictionary=None, ridge=1e-9, coeff_tol=1e-8,
                          max_rank=None, target=None, precomputed=None,
                          l1_max=3.0, verbose=False):
    """Fit K = K0 + sum_xi c_xi (corr_n(A_xi) - corr_n(A0)) to the
    complete-graph geometry target by unconstrained LSQ over c.

    Candidates have either unit class sums (type 1, with the plain power
    subtracted) or zero class sums (type 0).  Both types make every trial
    kernel satisfy partition, per-configuration commensurate identity, ASR,
    and permutation constraints exactly (module docstring).  The fit is
    performed matrix-free via the symmetric-power Gram identity.

    Parameters
    ----------
    target : (atoms, cells, weights) or None
        Precomputed build_target_tuples output (else computed here).

    Returns a FactorFit.
    """
    nat = structure.N_atoms
    n_c = int(np.prod(np.asarray(supercell, dtype=int)))
    n = int(order)

    if dictionary is None:
        dictionary = default_dictionary(structure, supercell, far=max(far, 3))

    a0w = plain_window(nat, supercell)
    wins = list(dictionary)

    # feasibility check (hard gate: the manifold is what makes every
    # trial kernel exact on the commensurate mesh).  Candidates may be
    # type-1 (class sums 1, correction = P_i - P_plain) or type-0 (class
    # sums 0, correction = P_i alone).
    max_dev = 0.0
    s_type = np.zeros(len(wins))
    for i, w in enumerate(wins):
        if w.class_sum not in (0.0, 1.0):
            raise ValueError("Window %s: class_sum must be 0 or 1"
                             % w.label)
        s_type[i] = w.class_sum
        dev = w.check_class_sums(supercell, nat=nat)
        max_dev = max(max_dev, dev)
        if dev > 1e-8:
            raise ValueError("Window %s violates the uniform class-sum "
                             "constraint by %.2e" % (w.label, dev))

    if target is None:
        target = build_target_tuples(structure, supercell, n, far=far,
                                     cost_power=cost_power)
    t_atoms, t_cells, t_w = target

    R = len(wins)
    allw = wins + [a0w]

    if precomputed is not None:
        P, t_dot, tt = precomputed
        assert P.shape == (R + 1, R + 1)
    else:
        # --- Gram of the raw powers (R+1 x R+1), matrix-free ---
        P = np.zeros((R + 1, R + 1))
        for i in range(R + 1):
            for j in range(i, R + 1):
                P[i, j] = P[j, i] = corr_inner_product(allw[i], allw[j], n)

        # --- overlaps with the target ---
        t_dot = np.zeros(R + 1)
        for i in range(R + 1):
            vals = eval_corr_at_tuples(allw[i], t_atoms, t_cells, nat)
            t_dot[i] = float(np.dot(t_w, vals))
        tt = float(np.dot(t_w, t_w))      # <T, T>

    def assemble(idx):
        """Gram/rhs for corrections u_i = P_i - s_i * P_plain."""
        ip = P[-1, -1]                                     # <K0, K0>
        s = s_type[idx]
        Pi = P[np.ix_(idx, idx)]
        Pip = P[idx, -1]
        G = (Pi - s[:, None] * Pip[None, :]
             - s[None, :] * Pip[:, None] + np.outer(s, s) * ip)
        h = (t_dot[idx] - Pip) - s * (t_dot[-1] - ip)      # <u_i, T - K0>
        return G, h

    ip = P[-1, -1]
    norm2 = tt - 2.0 * t_dot[-1] + ip                      # ||T - K0||^2

    def solve(idx, ridge_now=None):
        G, h = assemble(idx)
        scale = np.sqrt(np.maximum(np.diag(G), 1e-300))
        Gs = G / scale[:, None] / scale[None, :]
        rr = ridge if ridge_now is None else ridge_now
        c = np.linalg.lstsq(Gs + rr * np.eye(len(idx)), h / scale,
                            rcond=None)[0] / scale
        res2 = max(norm2 - 2.0 * np.dot(c, h) + c @ G @ c, 0.0)
        return c, res2, np.sqrt(np.diag(G))

    # null candidates (corr_n(A) - s corr_n(A0) == 0, e.g. a minimal-image
    # window whose symmetric power coincides with the tent on this
    # geometry) have zero Gram diagonal: they carry no correction and
    # poison any normalized solve -- mask them out everywhere.
    G_all, h_all = assemble(np.arange(R))
    dG = np.diag(G_all)
    valid = dG > 1e-10 * max(float(np.max(dG)), 1e-300)

    def select(ridge_now):
        if max_rank is None or max_rank >= int(np.sum(valid)):
            idx = np.where(valid)[0]
            coeffs, res2, unorm = solve(idx, ridge_now)
        else:
            # greedy forward selection (OMP on the Gram): the runtime
            # cost is one field set + one kernel pass per retained
            # factor, so a compact model matters more than the last few
            # percent of kernel error.
            un_all = np.sqrt(np.maximum(dG, 1e-300))
            sel = []
            coeffs = np.zeros(0)
            res2 = norm2
            unorm = np.zeros(0)
            for _ in range(int(max_rank)):
                if sel:
                    resid_corr = h_all - G_all[:, sel] @ coeffs
                else:
                    resid_corr = h_all.copy()
                score = np.abs(resid_corr) / un_all
                score[~valid] = -1.0
                score[sel] = -1.0
                inew = int(np.argmax(score))
                if score[inew] <= 0:
                    break
                trial = sel + [inew]
                c_t, res2_t, un_t = solve(np.array(trial, dtype=int),
                                          ridge_now)
                if res2_t >= res2 * (1.0 - 1e-12):
                    break    # no further reduction: stop before damage
                sel = trial
                coeffs, res2, unorm = c_t, res2_t, un_t
            idx = np.array(sel, dtype=int)
        return idx, coeffs, res2, unorm

    # variance guard: the runtime noise added by the correction passes
    # grows with sum_i |c_i| ||u_i|| (each pass is an independent
    # stochastic estimator scaled by c_i).  The normalized-Gram ridge is
    # exactly an L2 penalty on c_i ||u_i||, so escalate it until the L1
    # correction weight is within budget (the kernel-error price is
    # reported in the diagnostics).
    ridge_now = ridge
    for _ in range(40):
        idx, coeffs, res2, unorm = select(ridge_now)
        corr_l1 = float(np.sum(np.abs(coeffs) * unorm)) \
            / max(np.sqrt(norm2), 1e-300)
        if l1_max is None or corr_l1 <= l1_max or ridge_now >= 1.0:
            break
        ridge_now = max(ridge_now * 3.0, 1e-6)

    # prune negligible candidates and refit for a compact runtime model
    keep = np.abs(coeffs) * unorm > coeff_tol * max(np.sqrt(norm2), 1e-300)
    if not np.all(keep):
        idx = idx[keep]
        coeffs, res2, unorm = solve(idx, ridge_now)
    wins = [wins[i] for i in idx]
    s_kept = s_type[idx]
    corr_l1 = float(np.sum(np.abs(coeffs) * unorm)) \
        / max(np.sqrt(norm2), 1e-300)

    # --- coverage diagnostic: overlap with the target support ---
    # coverage = <K, T> / <T, T>; equals 1 when the kernel places exactly
    # the target weight on every minimal tuple.
    cov_fit = (t_dot[-1] + float(np.sum(
        coeffs * (t_dot[idx] - s_kept * t_dot[-1])))) / tt
    cov_plain = t_dot[-1] / tt

    diagnostics = {
        "norm_t_minus_k0": float(np.sqrt(norm2)),
        "residual": float(np.sqrt(res2)),
        "rel_residual": float(np.sqrt(res2 / norm2)) if norm2 > 0 else 0.0,
        "coverage": float(cov_fit),
        "coverage_plain": float(cov_plain),
        "max_class_dev": float(max_dev),
        "n_target_tuples": int(len(t_w)),
        "correction_l1": corr_l1,
        "ridge_used": float(ridge_now),
        "order": n, "far": int(far), "cost_power": float(cost_power),
        "nc": n_c,
    }

    fit = FactorFit(n, wins, coeffs,
                    plain_coeff=1.0 - float(np.sum(s_kept * coeffs)),
                    diagnostics=diagnostics)
    if verbose:
        print(fit.summary())
    return fit


def _raw_power_gram(windows, order):
    windows = list(windows)
    gram = np.zeros((len(windows), len(windows)), dtype=np.float64)
    for i in range(len(windows)):
        for j in range(i, len(windows)):
            gram[i, j] = gram[j, i] = corr_inner_product(
                windows[i], windows[j], order)
    return gram


def fit_constrained_factors(
        structure, supercell, order, far=2, cost_power=1, dictionary=None,
        max_rank=None, ridge=1e-10, coeff_tol=1e-10,
        constraint_rtol=1e-10, constraint_atol=1e-12,
        target_mode="auto", max_exact_classes=20000,
        sample_classes=20000, validation_classes=4000, seed=1729,
        batch_tuples=2048, max_combinations=200000,
        plain_shift_samples=20000, plain_exact_limit=200000,
        include_partition=True, include_asr=True, strict=True,
        verbose=False):
    """Matrix-free jointly constrained fit from the Overleaf formalism.

    The fitted form is ``K = P0 + sum_i c_i (P[A_i] - r_i P0)``. General
    factors use ``r_i=1`` and existing zero-class-sum factors use ``r_i=0``. No target,
    kernel, candidate matrix, or constraint matrix is materialized.  The
    routine constructs only the orbit Gram, streamed target overlaps, and
    the analytic coefficient-space constraint Gram.  ``dictionary`` may
    contain general local factors with ``class_sum=None``.

    ``target_mode='exact'`` streams all folded classes and is intended only
    for validation.  ``'sample'`` and large-cell ``'auto'`` use a fixed-size
    reproducible class sample plus an independent validation sample.
    """
    nat = int(structure.N_atoms)
    L = np.asarray(supercell, dtype=int)
    n = int(order)
    if n not in (3, 4):
        raise ValueError("The constrained factor fit supports order 3 or 4")
    if np.any(L <= 0):
        raise ValueError("supercell entries must be positive")
    if dictionary is None:
        dictionary = sparse_feasible_dictionary(structure, L)
    wins = list(dictionary)
    if not wins:
        raise ValueError("The constrained fit needs a nonempty dictionary")
    reference_weights = np.asarray([
        0.0 if window.class_sum == 0.0 else 1.0 for window in wins
    ], dtype=np.float64)

    # Exact one-leg algebra.  P is the Gram of raw orbit powers; G is the
    # Gram of corrections u_i = P[A_i] - P0.
    P = np.zeros((len(wins) + 1, len(wins) + 1), dtype=np.float64)
    P[:-1, :-1] = _raw_power_gram(wins, n)
    plain_overlap_info = []
    for index, window in enumerate(wins):
        P[index, -1], info = sparse_plain_inner_product(
            window, L, n, samples=plain_shift_samples,
            seed=int(seed) + 1009 * (index + 1),
            exact_limit=plain_exact_limit)
        P[-1, index] = P[index, -1]
        plain_overlap_info.append(info)
    P[-1, -1] = plain_power_norm(nat, L, n)
    pp = P[-1, -1]
    G = (P[:-1, :-1]
         - reference_weights[None, :] * P[:-1, -1][:, None]
         - reference_weights[:, None] * P[:-1, -1][None, :]
         + np.outer(reference_weights, reference_weights) * pp)
    G = 0.5 * (G + G.T)

    Gc, G_partition, G_asr = constraint_gram(
        wins, None, L, nat, n, include_partition=include_partition,
        include_asr=include_asr, reference_weights=reference_weights)

    template = GeometryTargetStream(
        structure, L, n, far=far, cost_power=cost_power, seed=seed,
        batch_tuples=batch_tuples, max_combinations=max_combinations)
    mode = str(target_mode).lower()
    if mode not in ("auto", "exact", "sample"):
        raise ValueError("target_mode must be 'auto', 'exact', or 'sample'")
    if mode == "auto":
        mode = "exact" if template.total_classes <= int(max_exact_classes) \
            else "sample"
    train_count = None if mode == "exact" else int(sample_classes)
    train_stream = GeometryTargetStream(
        structure, L, n, far=far, cost_power=cost_power,
        sample_classes=train_count, seed=seed, batch_tuples=batch_tuples,
        max_combinations=max_combinations)
    t_dot, tt, train_info = stream_target_overlaps_implicit_plain(
        wins, train_stream, nat, L)
    h = ((t_dot[:-1] - reference_weights * t_dot[-1])
         - (P[:-1, -1] - reference_weights * pp))
    norm2 = max(float(tt - 2.0 * t_dot[-1] + pp), 0.0)

    def solve_subset(indices):
        indices = np.asarray(indices, dtype=int)
        if not len(indices):
            return None
        sub_constraint = Gc[np.ix_(indices, indices)]
        Z, eigenvalues, threshold = constraint_nullspace(
            sub_constraint, rtol=constraint_rtol, atol=constraint_atol)
        if Z.shape[1] == 0:
            return None
        sub_gram = G[np.ix_(indices, indices)]
        sub_h = h[indices]
        reduced_gram = Z.T @ sub_gram @ Z
        reduced_h = Z.T @ sub_h
        correction_values, correction_vectors = np.linalg.eigh(
            0.5 * (reduced_gram + reduced_gram.T))
        correction_scale = max(float(np.max(correction_values)), 0.0)
        correction_keep = correction_values > max(
            1e-12, 1e-10 * correction_scale)
        if not np.any(correction_keep):
            return None
        basis = Z @ correction_vectors[:, correction_keep]
        reduced_gram = basis.T @ sub_gram @ basis
        reduced_h = basis.T @ sub_h
        scale = max(float(np.trace(reduced_gram)) /
                    max(len(reduced_gram), 1), 1e-300)
        z = np.linalg.lstsq(
            reduced_gram + float(ridge) * scale * np.eye(basis.shape[1]),
            reduced_h, rcond=None)[0]
        coeffs = basis @ z
        residual2 = max(norm2 - 2.0 * np.dot(coeffs, sub_h)
                        + coeffs @ sub_gram @ coeffs, 0.0)
        return {
            "indices": indices, "coeffs": coeffs, "residual2": residual2,
            "constraint_eigenvalues": eigenvalues,
            "constraint_threshold": threshold,
            "nullity": int(Z.shape[1]),
            "effective_nullity": int(basis.shape[1]),
        }

    selected = np.arange(len(wins), dtype=int)
    individually_constrained = np.max(np.abs(Gc)) < 1e-12
    if (individually_constrained and max_rank is not None
            and int(max_rank) < len(selected)):
        diagonal = np.diag(G)
        valid = diagonal > 1e-12 * max(float(np.max(diagonal)), 1e-300)
        chosen = []
        chosen_coeffs = np.empty(0)
        for _ in range(int(max_rank)):
            residual_overlap = h.copy()
            if chosen:
                residual_overlap -= G[:, chosen] @ chosen_coeffs
            scores = np.abs(residual_overlap) / np.sqrt(
                np.maximum(diagonal, 1e-300))
            scores[~valid] = -1.0
            scores[chosen] = -1.0
            new = int(np.argmax(scores))
            if scores[new] <= 0:
                break
            chosen.append(new)
            sub = G[np.ix_(chosen, chosen)]
            scale = max(float(np.trace(sub)) / len(chosen), 1e-300)
            chosen_coeffs = np.linalg.lstsq(
                sub + float(ridge) * scale * np.eye(len(chosen)),
                h[chosen], rcond=None)[0]
        selected = np.asarray(chosen, dtype=int)
    solved = solve_subset(selected)
    if solved is None:
        message = ("Candidate constraint Gram has an empty nullspace; "
                   "increase or redesign the factor dictionary")
        if strict:
            raise RuntimeError(message)
        solved = {"indices": np.empty(0, dtype=int),
                  "coeffs": np.empty(0), "residual2": norm2,
                  "constraint_eigenvalues": np.linalg.eigvalsh(Gc),
                  "constraint_threshold": 0.0, "nullity": 0,
                  "effective_nullity": 0}
    else:
        # Remove numerically unused factors and recompute feasibility after
        # every deletion.  A coefficient is never pruned without rebuilding
        # the constraint nullspace on the retained dictionary.
        active = np.abs(solved["coeffs"]) > float(coeff_tol)
        if np.any(active) and not np.all(active):
            trial = solve_subset(solved["indices"][active])
            if trial is not None:
                solved = trial
        while max_rank is not None and len(solved["indices"]) > int(max_rank):
            order_remove = np.argsort(np.abs(solved["coeffs"]))
            best = None
            # Trying the smallest coefficients captures the useful sparse
            # deletions without an O(R^4) exhaustive backward search.
            for position in order_remove[:min(12, len(order_remove))]:
                keep = np.delete(solved["indices"], position)
                trial = solve_subset(keep)
                if trial is not None and (best is None or
                        trial["residual2"] < best["residual2"]):
                    best = trial
            if best is None:
                break
            solved = best

    indices = solved["indices"]
    coeffs = solved["coeffs"]
    selected_windows = [wins[index] for index in indices]
    rank_budget_met = max_rank is None or len(indices) <= int(max_rank)
    if strict and not rank_budget_met:
        raise RuntimeError(
            "Exact constraints require %d factors, exceeding max_rank=%d"
            % (len(indices), int(max_rank)))

    constraint_value = float(coeffs @ Gc[np.ix_(indices, indices)] @ coeffs) \
        if len(indices) else 0.0
    constraint_residual = np.sqrt(max(constraint_value, 0.0))

    # Independent sampled validation never changes the fitted coefficients.
    validation_info = None
    validation_residual2 = solved["residual2"]
    validation_norm2 = norm2
    validation_coverage = (t_dot[-1] + np.dot(
        coeffs, t_dot[indices] - reference_weights[indices] * t_dot[-1])) \
        / max(tt, 1e-300) \
        if len(indices) else t_dot[-1] / max(tt, 1e-300)
    if validation_classes and mode == "sample":
        validation_stream = GeometryTargetStream(
            structure, L, n, far=far, cost_power=cost_power,
            sample_classes=int(validation_classes), seed=int(seed) + 1,
            batch_tuples=batch_tuples,
            max_combinations=max_combinations)
        tv, ttv, validation_info = stream_target_overlaps_implicit_plain(
            selected_windows, validation_stream, nat, L)
        if len(indices):
            Psel = P[np.ix_(np.r_[indices, len(wins)],
                            np.r_[indices, len(wins)])]
            refs = reference_weights[indices]
            Gsel = (Psel[:-1, :-1]
                    - refs[None, :] * Psel[:-1, -1][:, None]
                    - refs[:, None] * Psel[:-1, -1][None, :]
                    + np.outer(refs, refs) * Psel[-1, -1])
            hv = ((tv[:-1] - refs * tv[-1])
                  - (Psel[:-1, -1] - refs * Psel[-1, -1]))
            normv = max(float(ttv - 2.0 * tv[-1] + Psel[-1, -1]), 0.0)
            validation_norm2 = normv
            validation_residual2 = max(
                normv - 2.0 * np.dot(coeffs, hv)
                + coeffs @ Gsel @ coeffs, 0.0)
            validation_coverage = (tv[-1] + np.dot(
                coeffs, tv[:-1] - refs * tv[-1])) / max(ttv, 1e-300)
        else:
            normv = max(float(ttv - 2.0 * tv[-1] + P[-1, -1]), 0.0)
            validation_norm2 = normv
            validation_residual2 = normv
            validation_coverage = tv[-1] / max(ttv, 1e-300)

    diagnostics = {
        "norm_t_minus_k0": float(np.sqrt(norm2)),
        "residual": float(np.sqrt(solved["residual2"])),
        "rel_residual": float(np.sqrt(solved["residual2"]
                                      / max(norm2, 1e-300))),
        "coverage": float((t_dot[-1] + np.dot(
            coeffs, t_dot[indices]
            - reference_weights[indices] * t_dot[-1])) / max(tt, 1e-300))
            if len(indices) else float(t_dot[-1] / max(tt, 1e-300)),
        "coverage_plain": float(t_dot[-1] / max(tt, 1e-300)),
        "validation_rel_residual": float(np.sqrt(
            validation_residual2 / max(validation_norm2, 1e-300))),
        "validation_coverage": float(validation_coverage),
        "constraint_residual": float(constraint_residual),
        "constraint_nullity": int(solved["nullity"]),
        "constraint_effective_nullity": int(solved["effective_nullity"]),
        "constraint_threshold": float(solved["constraint_threshold"]),
        "partition_gram_norm": float(np.linalg.norm(G_partition)),
        "asr_gram_norm": float(np.linalg.norm(G_asr)),
        "max_class_dev": float("nan"),
        "n_target_tuples": int(train_info["tuples_visited"]),
        "correction_l1": float("nan"),
        "ridge_used": float(ridge),
        "order": n, "far": int(far), "cost_power": float(cost_power),
        "nc": int(np.prod(L)), "fit_mode": "joint-constrained",
        "target_mode": mode, "rank_budget_met": bool(rank_budget_met),
        "train_stream": train_info, "validation_stream": validation_info,
        "reference_weights": reference_weights[indices].tolist(),
        "plain_overlap_sampling": [plain_overlap_info[index]
                                    for index in indices],
    }
    fit = FactorFit(
        n, selected_windows, coeffs,
        plain_coeff=1.0 - float(np.dot(
            reference_weights[indices], coeffs)), diagnostics=diagnostics)
    if verbose:
        print(fit.summary())
    return fit


def optimize_constrained_factors(
        structure, supercell, order, sweeps=2, scale_trials=(0.75, 1.0, 1.33),
        initial_scale=1.0, base_sigmas=None, bond_centers=False,
        cutoff_sigma=2.5, quotient_shifts=None, verbose=False, **fit_kwargs):
    """Outer factor-parameter refinement with a full constrained refit.

    The optimized parameters are the common dilation of the geometry-based
    Gaussian width grid.  Every trial rebuilds the factors, orbit Gram,
    constraint nullspace, and coefficient solve.  Sparse translated
    differences remain exactly feasible for every width, so no penalty or
    post-fit projection is used.  This bounded coordinate search is robust
    for production; callers needing independent centers/widths can provide
    explicit dictionaries to :func:`fit_constrained_factors`.
    """
    if int(sweeps) < 0:
        raise ValueError("sweeps must be nonnegative")
    if base_sigmas is None:
        distance = _primitive_nn_distance(structure)
        base_sigmas = np.asarray([0.30, 0.45, 0.65, 0.90, 1.25]) * distance
    else:
        base_sigmas = np.asarray(base_sigmas, dtype=np.float64)
    trials = tuple(float(value) for value in scale_trials)
    if not trials or any(value <= 0 for value in trials):
        raise ValueError("scale_trials must contain positive values")

    history = []
    current_scale = float(initial_scale)
    best_fit = None
    best_score = np.inf
    best_scale = current_scale
    seen = {}
    for sweep in range(int(sweeps) + 1):
        multipliers = (1.0,) if sweep == 0 else trials
        candidates = []
        for multiplier in multipliers:
            scale = current_scale * multiplier
            key = round(scale, 12)
            if key not in seen:
                dictionary = sparse_feasible_dictionary(
                    structure, supercell, sigmas=base_sigmas * scale,
                    bond_centers=bond_centers, cutoff_sigma=cutoff_sigma,
                    quotient_shifts=quotient_shifts)
                seen[key] = fit_constrained_factors(
                    structure, supercell, order, dictionary=dictionary,
                    verbose=False, **fit_kwargs)
            fit = seen[key]
            score = float(fit.diagnostics["rel_residual"])
            candidates.append((score, scale, fit))
            history.append({"sweep": sweep, "scale": scale,
                            "train_rel_residual": score,
                            "validation_rel_residual": float(
                                fit.diagnostics["validation_rel_residual"])})
        score, current_scale, candidate = min(candidates,
                                              key=lambda item: item[0])
        if score < best_score:
            best_score, best_scale, best_fit = score, current_scale, candidate
    best_fit.diagnostics["factor_optimization"] = {
        "sweeps": int(sweeps), "selected_scale": best_scale,
        "history": history,
    }
    if verbose:
        print(best_fit.summary())
    return best_fit


# =========================================================================
# Dense validation helpers (small systems only)
# =========================================================================

def dense_kernel(win, nat, supercell, order, box):
    """Materialize corr_n(A) on all tuples within a cell box (leg 0 pinned).

    box : (lo(3,), hi(3,)) inclusive extended-cell bounds for legs >= 1.
    Returns dict {(a_tuple, cells_tuple): value} for validation tests.
    """
    lo, hi = box
    rng = [range(lo[d], hi[d] + 1) for d in range(3)]
    cells_1 = np.array(list(itertools.product(*rng)), dtype=int)
    states = [(a, tuple(c)) for a in range(nat) for c in cells_1]

    n_legs = order
    combos = list(itertools.product(states, repeat=n_legs - 1))
    T = len(combos) * nat
    atoms = np.zeros((T, n_legs), dtype=int)
    cells = np.zeros((T, n_legs, 3), dtype=int)
    k = 0
    for a1 in range(nat):
        for combo in combos:
            atoms[k, 0] = a1
            for s, (b, c) in enumerate(combo):
                atoms[k, s + 1] = b
                cells[k, s + 1] = c
            k += 1
    vals = eval_corr_at_tuples(win, atoms, cells, nat)
    return atoms, cells, vals


# =========================================================================
# Caching
# =========================================================================

_FIT_CACHE = {}


def _structure_fingerprint(structure, supercell, order, far, cost_power,
                           extra=""):
    h = hashlib.sha256()
    h.update(np.round(np.asarray(structure.unit_cell), 10).tobytes())
    h.update(np.round(np.asarray(structure.coords), 10).tobytes())
    h.update(np.asarray(structure.get_masses_array()).tobytes())
    h.update(np.asarray(supercell, dtype=int).tobytes())
    h.update(np.asarray([order, far], dtype=int).tobytes())
    h.update(np.asarray([cost_power], dtype=float).tobytes())
    h.update(extra.encode())
    return h.hexdigest()


def get_factor_fit(structure, supercell, order, far=2, cost_power=1,
                   verbose=False, fit_mode="individual", cache_dir=None,
                   **kwargs):
    """Memory/disk-cached geometry fit reusable across production runs."""
    mode = str(fit_mode).lower()
    if mode not in ("individual", "constrained"):
        raise ValueError("fit_mode must be 'individual' or 'constrained'")
    key = _structure_fingerprint(structure, supercell, order, far,
                                 cost_power,
                                 extra=repr((mode, sorted(kwargs.items()))))
    cache_path = None
    if cache_dir is not None:
        cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir)))
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "factor-fit-%s.npz" % key)
    if key not in _FIT_CACHE:
        if cache_path is not None and os.path.isfile(cache_path):
            _FIT_CACHE[key] = FactorFit.load(cache_path)
            return _FIT_CACHE[key]
        optimize_sweeps = int(kwargs.pop("optimize_sweeps", 0)) \
            if mode == "constrained" else 0
        fitter = fit_symmetric_factors if mode == "individual" else \
            (optimize_constrained_factors if optimize_sweeps > 0
             else fit_constrained_factors)
        if optimize_sweeps > 0:
            kwargs["sweeps"] = optimize_sweeps
        _FIT_CACHE[key] = fitter(structure, supercell, order, far=far,
                                 cost_power=cost_power, verbose=verbose,
                                 **kwargs)
        if cache_path is not None:
            temporary = cache_path + ".tmp-%d.npz" % os.getpid()
            _FIT_CACHE[key].save(temporary)
            os.replace(temporary, cache_path)
    return _FIT_CACHE[key]
