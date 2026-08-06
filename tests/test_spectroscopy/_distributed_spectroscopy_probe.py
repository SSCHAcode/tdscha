"""Worker for test_distributed_spectroscopy.py -- run under mpirun.

Usage:  python _distributed_spectroscopy_probe.py <backend> <data_dir> <workdir>

Drives the public ``Spectroscopy`` API from an ``EnsembleSource`` and writes,
per rank, the assembled responses together with what the engine actually held.
The test compares one rank against several and checks that the several never
replicated the configurations.

A separate process per case because every matrix-vector product is a
collective: a rank stepping an engine the others do not have would hang them.
"""
from __future__ import print_function

import os
import sys

import numpy as np

import cellconstructor.Settings as Parallel

import tdscha.Spectroscopy as SP
from tdscha import _SpectroscopyWorkflow as workflow

T = 250.0
NQIRR = 3
POP = 1
FINE = (2, 2, 4)          # multiple of the 2x2x2 coarse supercell
NSTEPS = 4
FREQUENCIES = np.linspace(1e-4, 5e-3, 20)
ANALYSIS = dict(smearing=2e-4, use_terminator=False)


def build(backend, data_dir, workdir):
    source = SP.EnsembleSource(
        data_dir, POP, os.path.join(data_dir, "dyn_gen_pop%d_" % POP), T,
        nqirr=NQIRR)
    options = {"fine_mesh": FINE} if backend == "atom_fourier" else {}
    job = SP.Spectroscopy(source, backend=backend, workdir=workdir,
                          use_symmetries=True, backend_options=options)
    n_atoms = source.reference_dyn.structure.N_atoms
    # The three Cartesian directions are one symmetry orbit in this cubic
    # cell, so they collapse onto a single Lanczos run; displacing a second
    # atom adds an independent one.  Two runs exercise engine reuse.
    for axis in range(3):
        vector = np.zeros((n_atoms, 3))
        vector[0, axis] = 1.0
        job.add_ir_vector(vector.ravel(), "axis_%d" % axis)
    second = np.zeros((n_atoms, 3))
    second[min(1, n_atoms - 1), 0] = 1.0
    second[0, 1] = 0.5
    job.add_ir_vector(second.ravel(), "mixed")
    return job


def main():
    backend, data_dir, workdir = sys.argv[1], sys.argv[2], sys.argv[3]

    engines = []
    genuine_create_backend = workflow.create_backend

    def spy(*args, **kwargs):
        engine = genuine_create_backend(*args, **kwargs)
        engines.append(engine)
        return engine

    workflow.create_backend = spy
    try:
        job = build(backend, data_dir, workdir)
        plan = job.plan_calculations()
        job.run(NSTEPS, save_each=2, verbose=False)

        # A second, fully satisfied run must not build another engine: the
        # results are restored from the checkpoint and nothing is reloaded.
        n_after_first = len(engines)
        restored = build(backend, data_dir, workdir)
        restored.run(NSTEPS, save_each=2, verbose=False)
    finally:
        workflow.create_backend = genuine_create_backend

    responses = {name: job.response(name, FREQUENCIES, **ANALYSIS)
                 for name in ("axis_0", "axis_1", "axis_2", "mixed")}
    restored_responses = {
        name: restored.response(name, FREQUENCIES, **ANALYSIS)
        for name in responses}

    engine = engines[0]
    np.savez(
        os.path.join(workdir, "rank_%d.npz" % Parallel.get_rank()),
        n_independent_runs=plan["n_independent_runs"],
        n_engines=n_after_first,
        n_engines_total=len(engines),
        distributed=bool(getattr(engine, "_distributed", False)),
        n_local=int(engine.N),
        n_global=int(getattr(engine, "_N_global", engine.N)),
        cls=type(engine).__name__,
        **responses)
    np.savez(os.path.join(workdir, "restored_%d.npz" % Parallel.get_rank()),
             **restored_responses)
    print("%s rank %d done" % (backend, Parallel.get_rank()))


if __name__ == "__main__":
    main()
