# main.py — Orchestrator: builds the t=0 compact hot start, relaxes it to a genuine solution
# of the full 4-ODE system, and runs the outer Kelvin-Helmholtz contraction time loop
# (time_stepper.run) for a small, capped number of steps.
#
# This is PLAN.md Sub-task 8's "dry run" (a strictly-capped run to inspect the exit criterion
# - monotonic r_surface decrease, sustained nonzero L_surface, decreasing T_center - before
# committing to a full run) using the promoted solve_bvp solver (PLAN.md Sub-task 5 update,
# 2026-08-08). NOT yet a full production run to config.R_HALT, which is a separate, much
# longer undertaking (r_surface starts at ~5 R_Jup and R_HALT=1 R_Jup - thousands of steps at
# this dt, not scoped here). Output snapshotting/plotting (output.py, Sub-task 10) is not yet
# built - history is returned in memory only for now.

import bvp_solver
import config
import time_stepper

N_STEPS_DRY_RUN = 10   # capped dry run, not a full run to config.R_HALT - see module docstring
# ASSUMPTION: same order of magnitude as the relaxation pseudo-timestep (config.
# RELAX_DT_FRACTION*T_KH_TIMESCALE_S) - a reasonable first probe for real time evolution, not
# a tuned production value. Sub-task 9 (adaptive dt) remains future work; this dry run uses
# the fixed-dt path PLAN.md §4.5 explicitly retains for exactly this kind of first look.
DT_DRY_RUN = config.RELAX_DT_FRACTION * config.T_KH_TIMESCALE_S


def main():
    print("main: building t=0 compact hot start (solve_static_structure) ...", flush=True)
    state_0 = bvp_solver.solve_static_structure()

    print("main: relaxing to a genuine solution of the full 4-ODE system (relax_initial_state) ...", flush=True)
    state_relaxed = bvp_solver.relax_initial_state(state_0)

    print(f"main: starting capped dry run, n_steps={N_STEPS_DRY_RUN}, "
          f"dt={DT_DRY_RUN / config.SECONDS_PER_YEAR:.4e} yr", flush=True)
    history = time_stepper.run(state_relaxed, N_STEPS_DRY_RUN, DT_DRY_RUN)
    return history


if __name__ == "__main__":
    main()
