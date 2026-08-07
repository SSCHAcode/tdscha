"""run_FT(optimized=True) must free the Krylov basis without changing anything.

The non-reorthogonalized q-space Lanczos is a three-term recurrence: it only
ever reads basis_Q[-1]/basis_Q[-2] (and the matching P and s_norm entries).
Retaining the whole basis therefore costs O(n_steps) memory for nothing --
about 10 GB per rank at 200 steps on a 12^3 fine mesh, which is what makes the
12^3 interpolation infeasible.  ``optimized=True`` keeps only the last few
vectors.

These tests pin the two properties that make that safe: the coefficients are
bit-identical to a full-basis run, and the basis really does stop growing.
"""
from __future__ import print_function

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_restart import (  # noqa: E402
    _assert_bit_exact, _make_plain, _make_tri, pytestmark)  # noqa: F401

import tdscha.QSpaceLanczos as QL  # noqa: E402


@pytest.mark.parametrize("maker,total", [(_make_plain, 10), (_make_tri, 12)])
def test_optimized_matches_full_basis(maker, total):
    """optimized=True changes memory, not numbers."""
    ref = maker()
    ref.run_FT(total, verbose=False, reorthogonalize=False)

    opt = maker()
    opt.run_FT(total, verbose=False, reorthogonalize=False, optimized=True)

    _assert_bit_exact(ref, opt, "optimized vs full basis")


@pytest.mark.parametrize("maker,total", [(_make_plain, 10), (_make_tri, 12)])
def test_optimized_bounds_the_basis(maker, total):
    """The retained basis stays at the window size instead of growing."""
    opt = maker()
    opt.run_FT(total, verbose=False, reorthogonalize=False, optimized=True)

    keep = QL._KEEP_BASIS_OPTIMIZED
    assert len(opt.basis_Q) <= keep, \
        "basis_Q grew to %d (window %d)" % (len(opt.basis_Q), keep)
    assert len(opt.basis_P) <= keep, \
        "basis_P grew to %d (window %d)" % (len(opt.basis_P), keep)
    assert len(opt.s_norm) <= keep, \
        "s_norm grew to %d (window %d)" % (len(opt.s_norm), keep)
    # and the run really did advance
    assert len(opt.a_coeffs) == total

    # a full-basis run of the same length keeps every vector: this is what
    # guarantees the test above is measuring the truncation and not a
    # recursion that stopped early.
    full = maker()
    full.run_FT(total, verbose=False, reorthogonalize=False)
    assert len(full.basis_Q) == total + 1


@pytest.mark.parametrize("maker,total", [(_make_plain, 10), (_make_tri, 12)])
def test_optimized_in_memory_restart_is_bit_exact(maker, total):
    """Chunked continuation still reproduces the single shot under truncation."""
    ref = maker()
    ref.run_FT(total, verbose=False, reorthogonalize=False, optimized=True)

    split = maker()
    split.run_FT(total // 2, verbose=False, reorthogonalize=False,
                 optimized=True)
    split.run_FT(total - total // 2, verbose=False, reorthogonalize=False,
                 optimized=True)

    _assert_bit_exact(ref, split, "optimized in-memory restart")


def test_optimized_disk_restart_is_bit_exact(tmp_path):
    """save_status/load_status round-trips the truncated basis exactly.

    This is the property the production checkpoints rely on: the saved npz
    holds only the retained window, and resuming from it must still reproduce
    an uninterrupted run.
    """
    total, half = 10, 5
    ref = _make_plain()
    ref.run_FT(total, verbose=False, reorthogonalize=False, optimized=True)

    lanc = _make_plain()
    lanc.run_FT(half, verbose=False, reorthogonalize=False, optimized=True)
    status = str(tmp_path / "opt_status")
    lanc.save_status(status)

    resumed = _make_plain()
    resumed.load_status(status + ".npz")
    resumed.run_FT(total - half, verbose=False, reorthogonalize=False,
                   optimized=True)

    _assert_bit_exact(ref, resumed, "optimized disk restart")


def test_optimized_rejects_reorthogonalization():
    """Silently reorthogonalizing against a truncated basis would be wrong."""
    lanc = _make_plain()
    with pytest.raises(ValueError, match="reorthogonalize"):
        lanc.run_FT(4, verbose=False, optimized=True, reorthogonalize=True)


def test_optimized_rejects_partial_reorthogonalization():
    """n_rep_orth reaches further back than the retained window."""
    lanc = _make_plain()
    with pytest.raises(ValueError, match="n_rep_orth"):
        lanc.run_FT(4, verbose=False, optimized=True, reorthogonalize=False,
                    n_rep_orth=1, n_ortho=10)


def test_reorthogonalize_refuses_to_continue_a_truncated_basis():
    """Switching to full reorthogonalization after truncation must not pass."""
    lanc = _make_plain()
    lanc.run_FT(6, verbose=False, reorthogonalize=False, optimized=True)
    with pytest.raises(ValueError, match="truncated"):
        lanc.run_FT(2, verbose=False, reorthogonalize=True)
