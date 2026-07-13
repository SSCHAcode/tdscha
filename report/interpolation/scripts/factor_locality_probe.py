"""Locality and cancellation diagnostics for symmetric factor kernels."""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

import tdscha.QSpaceFactorKernel as FK


def _factor_extent(structure, window):
    """Bounding-box diameter and absolute-weight RMS radius in Cartesian space."""
    atoms, cells, values = window.arrays()
    points = structure.coords[atoms] + cells @ structure.unit_cell
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    weights = np.abs(values)
    center = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
    rms = float(np.sqrt(np.sum(weights * np.sum((points - center) ** 2,
                                                axis=1)) /
                        np.sum(weights)))
    return extent, rms


def _correction_gram(fit, plain):
    windows = fit.windows
    order = fit.order
    types = np.asarray([w.class_sum for w in windows])
    n = len(windows)
    gram = np.empty((n, n))
    pp = FK.corr_inner_product(plain, plain, order)
    wp = np.asarray([FK.corr_inner_product(w, plain, order)
                     for w in windows])
    for i, wi in enumerate(windows):
        for j in range(i, n):
            raw = FK.corr_inner_product(wi, windows[j], order)
            value = (raw - types[j] * wp[i] - types[i] * wp[j]
                     + types[i] * types[j] * pp)
            gram[i, j] = gram[j, i] = value
    return gram


def summarize_fit(structure, supercell, fit):
    plain = FK.plain_window(structure.N_atoms, supercell)
    gram = _correction_gram(fit, plain)
    coeffs = fit.coeffs
    term_norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    assembled = float(np.sqrt(max(coeffs @ gram @ coeffs, 0.0)))
    numerator = float(np.sum(np.abs(coeffs) * term_norms))
    factors = []
    for coeff, window in zip(coeffs, fit.windows):
        extent, rms = _factor_extent(structure, window)
        factors.append({
            "label": window.label,
            "type": int(window.class_sum),
            "coefficient": float(coeff),
            "entries": len(window.entries),
            "extent": extent,
            "rms_radius": rms,
        })
    return {
        "order": fit.order,
        "rank": len(fit.windows),
        "rel_residual": fit.diagnostics["rel_residual"],
        "coverage": fit.diagnostics["coverage"],
        "plain_coverage": fit.diagnostics["coverage_plain"],
        "plain_coefficient": fit.plain_coeff,
        "correction_condition": numerator / max(assembled, 1e-300),
        "max_factor_extent": max(f["extent"] for f in factors),
        "max_factor_rms_radius": max(f["rms_radius"] for f in factors),
        "factors": factors,
    }


def load_snte():
    import cellconstructor as CC
    import cellconstructor.Phonons
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "..", ".."))
    data = os.path.join(root, "tests", "test_julia", "data")
    dyn = CC.Phonons.Phonons(os.path.join(data, "dyn_gen_pop1_"), 3)
    return dyn.structure, (2, 2, 2)


def load_chain(length):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "..", ".."))
    test_dir = os.path.join(root, "tests", "test_interpolation")
    sys.path.insert(0, test_dir)
    import _toy_chain as toy
    dyn = toy.build_dyn(length)
    return dyn.structure, (1, 1, length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("snte", "chain"),
                        default="snte")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--orders", type=int, nargs="+", default=(3, 4))
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--far", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.system == "snte":
        structure, supercell = load_snte()
    else:
        structure, supercell = load_chain(args.length)
    result = []
    for order in args.orders:
        fit = FK.fit_symmetric_factors(
            structure, supercell, order, far=args.far, max_rank=args.rank)
        result.append(summarize_fit(structure, supercell, fit))
    text = json.dumps({"system": args.system, "supercell": supercell,
                       "fits": result}, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
