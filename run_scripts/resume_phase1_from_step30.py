# resume_phase1_from_step30.py — Third resume of the Phase 1 First Hydrostatic Core run
# (PROGRESS.md has the full report), continuing from "step 30" in original numbering
# (18 original + 3 first-resume + 9 second-resume steps, T_center=1501K, r=249.4 R_Jup).
#
# Unlike the first two resumes, time_stepper.run() now has automatic step-retry
# (config.STEP_RETRY_MAX_ATTEMPTS/SHRINK_FACTOR) - the recurring "growth-capped dt fails,
# several-times-smaller dt at the same state succeeds" pattern hit three times now (T~1300K
# twice, T~1500K) is handled automatically instead of requiring another manual diagnose-and-
# relaunch cycle. This resume is intended to run through to config.PHASE1_T_CENTER_HALT
# unattended.

# HOUSEKEEPING 2026-08-13 (repository cleanup): moved into run_scripts/ - see main.py's own
# shim comment for why this sys.path prepend is here. Snapshot/plot dirs now go through
# output.run_output_dirs (the outputs/ consolidation) instead of bare top-level literals.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import output
import time_stepper

RESUME_SNAPSHOT_DIR, _ = output.run_output_dirs("Phase1_first_core_resumed2")
RESUME_STEP = 9

SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs("Phase1_first_core_resumed3")
N_STEPS_MAX = 400

DT_SEED = 5.0 * config.SECONDS_PER_YEAR   # matches the last successful dt scale from the previous resume


def main():
    config.MU = 2.34
    config.GAMMA = 1.4
    config.USE_H2_RECOMBINATION_PHYSICS = False
    config.USE_ADAPTIVE_DT = True
    config.ADAPTIVE_DT_MAX = 1.0e4 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_MIN = 5.0 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_GROWTH_FACTOR = 1.1
    config.T_MAX_S = 1.0e6 * config.SECONDS_PER_YEAR

    resume_path = f"{RESUME_SNAPSHOT_DIR}/snapshot_{RESUME_STEP:05d}.npz"
    print(f"resume_phase1_from_step30: resuming from {resume_path} ...", flush=True)
    state_resumed, _is_convective = output.load_snapshot(resume_path)
    print(f"resume_phase1_from_step30: resumed state at t={state_resumed.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={state_resumed.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={state_resumed.T[0]:.6e} K", flush=True)

    print(f"resume_phase1_from_step30: starting continuation run, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_resumed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("resume_phase1_from_step30: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"resume_phase1_from_step30: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
