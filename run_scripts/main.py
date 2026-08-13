# main.py — Orchestrator: builds the t=0 compact hot start, relaxes it to a genuine solution
# of the full 4-ODE system, and runs the outer Kelvin-Helmholtz contraction time loop
# (time_stepper.run) with adaptive time-stepping (Sub-task 9) and snapshot saving (Sub-task
# 10) wired in.
#
# CURRENT MODE: a capped SANITY-CHECK run (N_STEPS_SANITY_CHECK), not yet the full production
# run to config.R_HALT/config.T_MAX_S - per explicit user request (2026-08-09),
# this exercises the newly-raised config.ADAPTIVE_DT_MAX (now a generous defensive backstop,
# not a validation-scale ceiling - see its own config.py comment), the dual stopping
# condition, live per-step logging, and snapshot I/O end-to-end on a small scale, generating
# a first look at the actual output.py plots, BEFORE committing to the real, much longer run.
# Raise N_STEPS_SANITY_CHECK (or remove the cap) only after that review.

# HOUSEKEEPING 2026-08-13 (repository cleanup): this script now lives in run_scripts/, one
# level below the core physics modules (config.py, bvp_solver.py, ...) - prepend the repo root
# to sys.path so the existing flat `import config` etc. below keep resolving regardless of
# the caller's cwd, without turning the project into an installable package.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bvp_solver
import config
import output
import time_stepper

N_STEPS_SANITY_CHECK = 15   # capped - see module docstring; NOT the full run to R_HALT/T_MAX_S
# ASSUMPTION: same order of magnitude as the relaxation pseudo-timestep (config.
# RELAX_DT_FRACTION*T_KH_TIMESCALE_S) - a reasonable seed for the FIRST real step only; every
# step after that is chosen by time_stepper.select_adaptive_dt (Sub-task 9) instead, once
# config.USE_ADAPTIVE_DT is set below.
DT_SEED = config.RELAX_DT_FRACTION * config.T_KH_TIMESCALE_S


def main():
    config.USE_ADAPTIVE_DT = True   # runtime-only override, matching this project's established pattern for per-run config choices

    print("main: building t=0 compact hot start (solve_static_structure) ...", flush=True)
    state_0 = bvp_solver.solve_static_structure()

    print("main: relaxing to a genuine solution of the full 4-ODE system (relax_initial_state) ...", flush=True)
    state_relaxed = bvp_solver.relax_initial_state(state_0)

    print(f"main: starting sanity-check run, n_steps={N_STEPS_SANITY_CHECK}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"snapshot_dir={output.SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_relaxed, N_STEPS_SANITY_CHECK, DT_SEED,
                                snapshot_dir=output.SNAPSHOT_DIR)

    print("main: generating plots from the saved snapshots (output.py, Sub-task 10) ...", flush=True)
    output.generate_all_plots()

    return history


if __name__ == "__main__":
    main()
