# dev_cache.py — Development-only utility: serialize/deserialize a SimulationState to disk so
# downstream logic can be built and debugged against a cached result instead of re-running a
# heavy upstream solve (e.g. relax_initial_state) on every test. Not part of the physics/solver
# pipeline (CLAUDE.md Development Workflow) - never imported by config.py, eos.py, odes.py,
# bvp_solver.py, time_stepper.py, or any other operational module.

# HOUSEKEEPING 2026-08-13 (repository cleanup): moved into run_scripts/ - see main.py's own
# shim comment for why this sys.path prepend is here.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import state


def save_state(s: state.SimulationState, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(s, f)


def load_state(path: str) -> state.SimulationState:
    with open(path, "rb") as f:
        return pickle.load(f)
