#!/usr/bin/env python3
"""Reproduce the numerical evidence used by the spectroscopy report.

The benchmark deliberately uses only data shipped in ``tests/test_julia/data``.
The cubic Raman derivative is synthetic and symmetry-covariant: it isolates a
triply equivalent off-diagonal Raman sector while making the four forbidden
Placzek channels exactly zero.  This gives a transparent test of orbit
reconstruction rather than claiming a material-specific Raman prediction.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
from statistics import median
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cellconstructor as CC
from cellconstructor.Units import RY_TO_CM
import sscha.Ensemble

import tdscha.DynamicalLanczos as DL
import tdscha.Spectroscopy as SP
from tdscha import _SpectroscopyWorkflow as workflow


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "test_julia" / "data"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "data"
FIGURES = HERE / "figures"
# Long enough that symmetry-reduced recurrence work is visible above the
# intentionally conservative atomic-checkpoint overhead, while remaining a
# small two-atom/ten-configuration benchmark.
N_STEPS = 256
N_KERNEL_REPEATS = 7
KERNEL_CONFIGURATION_COUNTS = (10, 40, 160, 640)
SMEARING_RY = 8.0e-5


def load_ensemble():
    dyn = CC.Phonons.Phonons(str(DATA / "dyn_gen_pop1_"), 3)
    ensemble = sscha.Ensemble.Ensemble(dyn, 250.0)
    ensemble.load_bin(str(DATA), 1)
    return ensemble


def cubic_charges(n_atoms):
    """Return a neutral isotropic Born-charge pattern for the two-atom cell."""
    if n_atoms != 2:
        raise ValueError("The bundled benchmark is expected to have two atoms")
    charges = np.zeros((n_atoms, 3, 3))
    charges[0] = np.eye(3)
    charges[1] = -np.eye(3)
    return charges


def cubic_raman_tensor(n_atoms):
    """Return a controlled T2-like Raman derivative in Cartesian coordinates.

    The normalized xy, xz and yz Placzek channels prepare, respectively,
    opposite x, y and z displacements of the two sublattices.  The trace and
    three diagonal-deviatoric channels vanish identically.
    """
    if n_atoms != 2:
        raise ValueError("The bundled benchmark is expected to have two atoms")
    tensor = np.zeros((3, 3, 3 * n_atoms))
    patterns = (
        np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0]),
    )
    for (first, second), pattern in zip(((0, 1), (0, 2), (1, 2)),
                                        patterns):
        # The normalized coefficient has sqrt(3)/2 on each symmetric entry.
        # Equal tensor entries therefore contract to ``pattern``.
        tensor[first, second] = pattern / np.sqrt(3.0)
        tensor[second, first] = pattern / np.sqrt(3.0)
    return tensor


def relative_linf(reference, candidate):
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    return float(np.max(np.abs(candidate - reference)) / scale)


def prepare_direct_engine(ensemble):
    engine = DL.Lanczos(ensemble)
    engine.init(use_symmetries=True)
    return engine


def _tile_engine_configurations(engine, backend, target):
    """Tile a fixed ensemble to isolate kernel scaling with sample count."""
    original = int(engine.N)
    if target % original:
        raise ValueError("target configuration count must divide the fixture")
    factor = target // original
    engine.rho = np.tile(engine.rho, factor)
    if backend == "real":
        engine.X = np.tile(engine.X, (factor, 1))
        engine.Y = np.tile(engine.Y, (factor, 1))
    else:
        engine.X_q = np.tile(engine.X_q, (1, factor, 1))
        engine.Y_q = np.tile(engine.Y_q, (1, factor, 1))
    engine.N = target
    engine.N_eff = float(np.sum(engine.rho))


def timed_kernel(ensemble, backend, run_spec, n_configurations,
                 repeats=N_KERNEL_REPEATS):
    vector = run_spec.as_array()
    full = workflow.create_backend(ensemble, backend, {})
    workflow.prepare_engine(full, vector, True)
    reduced = workflow.create_backend(ensemble, backend, {})
    workflow.prepare_engine(reduced, vector, True, run_spec, 1.0e-8)
    _tile_engine_configurations(full, backend, n_configurations)
    _tile_engine_configurations(reduced, backend, n_configurations)

    sink = io.StringIO()
    with redirect_stdout(sink):
        full_reference = full.apply_anharmonic_FT()
        reduced_reference = reduced.apply_anharmonic_FT()
    full_times = []
    reduced_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with redirect_stdout(sink):
            full_value = full.apply_anharmonic_FT()
        full_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        with redirect_stdout(sink):
            reduced_value = reduced.apply_anharmonic_FT()
        reduced_times.append(time.perf_counter() - start)

    # Compare both the warm-up call and final repeated call to guard against
    # accidental mutation hidden by timing loops.
    error = max(relative_linf(full_reference, reduced_reference),
                relative_linf(full_value, reduced_value))
    active = int(reduced._spectroscopy_symmetry_count(reduced.n_syms))
    return {
        "backend": backend,
        "configurations": n_configurations,
        "full_group_order": int(full.n_syms),
        "coset_representatives": active,
        "stabilizer_order": int(
            len(reduced._spectroscopy_stabilizer_indices)),
        "full_median_seconds": median(full_times),
        "reduced_median_seconds": median(reduced_times),
        "speedup": median(full_times) / median(reduced_times),
        "relative_linf_error": error,
        "repeats": repeats,
    }


def direct_raman_legacy(ensemble, frequencies):
    weights = SP.get_unpolarized_raman_weights("normalized")
    total = np.zeros_like(frequencies)
    active = []
    elapsed = 0.0
    sink = io.StringIO()
    for index in range(7):
        # The old API cannot safely run a zero perturbation.  Detecting and
        # skipping it here is the manual workaround used for the reference.
        probe = SP.build_raman_vector(
            ensemble.current_dyn.raman_tensor,
            SP.get_raman_component(index).coefficients("normalized"))
        if np.linalg.norm(probe) <= 1.0e-12:
            continue
        active.append(index)
        start = time.perf_counter()
        with redirect_stdout(sink):
            engine = prepare_direct_engine(ensemble)
            engine.prepare_raman(unpolarized=index)
            engine.run_FT(N_STEPS, verbose=False)
        elapsed += time.perf_counter() - start
        total += weights[index] * (-np.imag(
            engine.get_green_function_continued_fraction(
                frequencies, use_terminator=False, smearing=SMEARING_RY)))
    return total, elapsed, active


def direct_ir_legacy(ensemble, charges, frequencies):
    total = np.zeros_like(frequencies)
    elapsed = 0.0
    sink = io.StringIO()
    for direction in np.eye(3):
        start = time.perf_counter()
        with redirect_stdout(sink):
            engine = prepare_direct_engine(ensemble)
            engine.prepare_ir(effective_charges=charges, pol_vec=direction)
            engine.run_FT(N_STEPS, verbose=False)
        elapsed += time.perf_counter() - start
        total += (-np.imag(engine.get_green_function_continued_fraction(
            frequencies, use_terminator=False, smearing=SMEARING_RY))) / 3.0
    return total, elapsed


def new_raman(ensemble, frequencies, workdir):
    job = SP.Spectroscopy(
        ensemble, backend="real", workdir=workdir, use_symmetries=True)
    job.add_raman_unpolarized("powder")
    plan = job.plan_calculations()
    sink = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(sink):
        job.run(N_STEPS, save_each=N_STEPS, verbose=False)
    elapsed = time.perf_counter() - start
    spectrum = job.raman_spectrum(
        "powder", frequencies, kind="response", use_terminator=False,
        smearing=SMEARING_RY)
    return spectrum, elapsed, plan


def new_ir(ensemble, charges, frequencies, workdir):
    job = SP.Spectroscopy(
        ensemble, backend="real", workdir=workdir, use_symmetries=True)
    job.add_ir_unpolarized("powder", effective_charges=charges)
    plan = job.plan_calculations()
    sink = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(sink):
        job.run(N_STEPS, save_each=N_STEPS, verbose=False)
    elapsed = time.perf_counter() - start
    spectrum = job.response(
        "powder", frequencies, use_terminator=False,
        smearing=SMEARING_RY)
    return spectrum, elapsed, plan


def save_spectra(frequencies, raman_legacy, raman_new, ir_legacy, ir_new):
    path = OUTPUT / "spectra.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frequency_ry", "frequency_cm-1", "raman_legacy",
                         "raman_new", "ir_legacy", "ir_new"))
        for row in zip(frequencies, frequencies * RY_TO_CM, raman_legacy,
                       raman_new, ir_legacy, ir_new):
            writer.writerow(tuple(float(value) for value in row))


def plot_spectrum(path, frequency_cm, legacy, new, ylabel, title):
    scale = max(float(np.max(np.abs(legacy))), np.finfo(float).tiny)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.plot(frequency_cm, legacy / scale, color="#222222", linewidth=2.2,
              label="legacy independent calculations")
    axis.plot(frequency_cm, new / scale, color="#d95f02", linewidth=1.3,
              linestyle="--", label="symmetry-aware API")
    axis.set_xlabel(r"frequency shift (cm$^{-1}$)")
    axis.set_ylabel(ylabel + " (normalized)")
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_timings(path, summary):
    labels = ("Raman\nworkflow", "IR\nworkflow",
              "real kernel", "q-space kernel")
    legacy = (summary["raman"]["legacy_seconds"],
              summary["ir"]["legacy_seconds"],
              summary["kernels"][0]["full_median_seconds"],
              summary["kernels"][1]["full_median_seconds"])
    reduced = (summary["raman"]["new_seconds"],
               summary["ir"]["new_seconds"],
               summary["kernels"][0]["reduced_median_seconds"],
               summary["kernels"][1]["reduced_median_seconds"])
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(6.8, 3.9))
    axis.bar(positions - width / 2, legacy, width, color="#777777",
             label="legacy/full group")
    axis.bar(positions + width / 2, reduced, width, color="#1b9e77",
             label="new/reduced")
    axis.set_yscale("log")
    axis.set_ylabel("wall time (s, log scale)")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_kernel_scaling(path, scaling):
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for backend, marker in (("real", "o"), ("qspace", "s")):
        rows = [row for row in scaling if row["backend"] == backend]
        axis.plot(
            [row["configurations"] for row in rows],
            [row["speedup"] for row in rows], marker=marker,
            linewidth=1.8, label=backend)
    axis.axhline(16.0, color="#555555", linestyle="--", linewidth=1.0,
                 label=r"representative limit $48/3$")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("configurations (tiled benchmark)")
    axis.set_ylabel("full / reduced kernel time")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def save_latex_macros(summary):
    """Keep numerical statements in the report tied to generated JSON data."""
    raman = summary["raman"]
    ir = summary["ir"]
    real, qspace = summary["kernels"]
    commands = {
        "BenchSteps": summary["system"]["lanczos_steps"],
        "RamanRequested": raman["requested_components"],
        "RamanNonzero": raman["legacy_nonzero_runs"],
        "RamanIndependent": raman["new_independent_runs"],
        "RamanError": "{:.2e}".format(raman["relative_linf_error"]),
        "RamanLegacyTime": "{:.3f}".format(raman["legacy_seconds"]),
        "RamanNewTime": "{:.3f}".format(raman["new_seconds"]),
        "RamanSpeedup": "{:.2f}".format(raman["speedup"]),
        "IRIndependent": ir["new_independent_runs"],
        "IRError": "{:.2e}".format(ir["relative_linf_error"]),
        "IRLegacyTime": "{:.3f}".format(ir["legacy_seconds"]),
        "IRNewTime": "{:.3f}".format(ir["new_seconds"]),
        "IRSpeedup": "{:.2f}".format(ir["speedup"]),
        "GroupOrder": real["full_group_order"],
        "StabilizerOrder": real["stabilizer_order"],
        "CosetCount": real["coset_representatives"],
        "KernelConfigurations": real["configurations"],
        "RealKernelSpeedup": "{:.2f}".format(real["speedup"]),
        "RealKernelError": "{:.2e}".format(real["relative_linf_error"]),
        "QKernelSpeedup": "{:.2f}".format(qspace["speedup"]),
        "QKernelError": "{:.2e}".format(qspace["relative_linf_error"]),
    }
    with (OUTPUT / "benchmark_values.tex").open(
            "w", encoding="utf-8") as stream:
        stream.write("% Generated by benchmark_spectroscopy.py; do not edit.\n")
        for name, value in commands.items():
            stream.write("\\newcommand{{\\{}}}{{{}}}\n".format(name, value))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frequencies = np.linspace(2.0e-4, 8.0e-3, 900)
    ensemble = load_ensemble()
    n_atoms = ensemble.current_dyn.structure.N_atoms
    charges = cubic_charges(n_atoms)
    ensemble.current_dyn.raman_tensor = cubic_raman_tensor(n_atoms)

    # Construct one cubic IR representative for isolated kernel timings.  The
    # real-space run also warms the Julia bridge before end-to-end timings.
    probe = SP.Spectroscopy(ensemble, backend="real", use_symmetries=True)
    probe.add_ir_polarized([1, 0, 0], "ir_x", effective_charges=charges)
    probe.plan_calculations()
    run_spec = next(iter(probe._run_specs.values()))
    kernel_scaling = [
        timed_kernel(ensemble, backend, run_spec, n_configurations)
        for backend in ("real", "qspace")
        for n_configurations in KERNEL_CONFIGURATION_COUNTS]
    kernels = [next(
        row for row in reversed(kernel_scaling)
        if row["backend"] == backend)
        for backend in ("real", "qspace")]

    with tempfile.TemporaryDirectory(prefix="tdscha-spectroscopy-") as temp:
        temp = Path(temp)
        raman_legacy, raman_legacy_time, active_raman = (
            direct_raman_legacy(ensemble, frequencies))
        raman_new, raman_new_time, raman_plan = new_raman(
            ensemble, frequencies, temp / "raman")
        ir_legacy, ir_legacy_time = direct_ir_legacy(
            ensemble, charges, frequencies)
        ir_new, ir_new_time, ir_plan = new_ir(
            ensemble, charges, frequencies, temp / "ir")

    summary = {
        "system": {
            "name": "bundled cubic SnTe ensemble with controlled optical vertices",
            "temperature_K": float(ensemble.current_T),
            "supercell": [int(value) for value in
                          ensemble.current_dyn.GetSupercell()],
            "ensemble_configurations": int(ensemble.N),
            "lanczos_steps": N_STEPS,
            "smearing_ry": SMEARING_RY,
        },
        "raman": {
            "requested_components": int(
                raman_plan["n_requested_components"]),
            "legacy_nonzero_runs": len(active_raman),
            "legacy_active_indices": active_raman,
            "new_independent_runs": int(raman_plan["n_independent_runs"]),
            "zero_components": sum(
                component["run_id"] is None
                for component in raman_plan["request_components"]["powder"]),
            "legacy_seconds": raman_legacy_time,
            "new_seconds": raman_new_time,
            "speedup": raman_legacy_time / raman_new_time,
            "relative_linf_error": relative_linf(raman_legacy, raman_new),
        },
        "ir": {
            "requested_components": int(ir_plan["n_requested_components"]),
            "legacy_runs": 3,
            "new_independent_runs": int(ir_plan["n_independent_runs"]),
            "legacy_seconds": ir_legacy_time,
            "new_seconds": ir_new_time,
            "speedup": ir_legacy_time / ir_new_time,
            "relative_linf_error": relative_linf(ir_legacy, ir_new),
        },
        "kernels": kernels,
        "kernel_scaling": kernel_scaling,
    }
    save_spectra(frequencies, raman_legacy, raman_new,
                 ir_legacy, ir_new)
    with (OUTPUT / "benchmark_summary.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    save_latex_macros(summary)

    plot_spectrum(
        FIGURES / "raman_legacy_vs_symmetry.pdf", frequencies * RY_TO_CM,
        raman_legacy, raman_new, "Raman response",
        "Cubic unpolarized Raman reconstruction")
    plot_spectrum(
        FIGURES / "ir_legacy_vs_symmetry.pdf", frequencies * RY_TO_CM,
        ir_legacy, ir_new, "IR response",
        "Cubic unpolarized IR reconstruction")
    plot_timings(FIGURES / "timings.pdf", summary)
    plot_kernel_scaling(FIGURES / "kernel_scaling.pdf", kernel_scaling)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
