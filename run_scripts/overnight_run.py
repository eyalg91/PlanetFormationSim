# overnight_run.py — Sub-task 9/10 EXTENDED DIAGNOSTIC run, NOT the validated production run.
#
# Purpose (2026-08-09 night, explicit user request): the sanity-check run (main.py) found a
# genuine, dt-correlated r_surface non-monotonicity starting once dt exceeded the previously-
# validated 5e4 yr (PROGRESS.md has the full trail) - an open, unresolved accuracy question,
# not yet chased since the solver itself shows no convergence distress at any of these steps.
# Rather than either (a) blindly trusting an overnight run as the real Stage 3 result, or (b)
# doing nothing while the machine is idle overnight, this generates MORE diagnostic data on
# how that anomaly evolves as dt grows further - to look at tomorrow, not to treat as
# thesis-ready. Uses a SEPARATE snapshot/plot directory from main.py's sanity check so the
# two don't mix.
#
# Same safety nets as main.py: the NaN/positivity guard and the dual R_HALT/AGE_SOLAR_SYSTEM_S
# stopping condition in time_stepper.run() apply unchanged - if something goes wrong, this
# halts with a clear error in the log rather than running indefinitely on bad data.

# HOUSEKEEPING 2026-08-13 (repository cleanup): moved into run_scripts/ - see main.py's own
# shim comment for why this sys.path prepend is here. Snapshot/plot dirs now go through
# output.run_output_dirs (the outputs/ consolidation) instead of bare top-level literals.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bvp_solver
import config
import output
import time_stepper

N_STEPS_MAX = 100   # generous cap, not "no limit" - the dual halt condition is the primary expected stop
SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs("overnight")
DT_SEED = config.RELAX_DT_FRACTION * config.T_KH_TIMESCALE_S


def main():
    config.USE_ADAPTIVE_DT = True

    print("overnight_run: building t=0 compact hot start (solve_static_structure) ...", flush=True)
    state_0 = bvp_solver.solve_static_structure()

    print("overnight_run: relaxing to a genuine solution of the full 4-ODE system (relax_initial_state) ...", flush=True)
    state_relaxed = bvp_solver.relax_initial_state(state_0)

    print(f"overnight_run: starting EXTENDED DIAGNOSTIC run (not yet validated for dt>5e4 yr - "
          f"PROGRESS.md 2026-08-09), n_steps={N_STEPS_MAX}, seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_relaxed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("overnight_run: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    print(f"overnight_run: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR} - REVIEW BEFORE TRUSTING, see module docstring.", flush=True)
    return history


if __name__ == "__main__":
    main()
