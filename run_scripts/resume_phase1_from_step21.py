# resume_phase1_from_step21.py — Second resume of the Phase 1 First Hydrostatic Core run
# (PROGRESS.md has the full report), continuing from the second resume's last surviving state
# (T_center~1300K, r~293 R_Jup - "step 21" in original numbering: 18 original + 3 first-resume
# steps). The T_center~1250-1300K neighborhood has repeatedly needed a smaller dt than the
# growth-capped adaptive selection was proposing (15-35 yr failed at this point twice; 5-10 yr
# converged directly both times) - gentled further here (smaller seed, gentler growth cap) to
# reduce recurrence, not eliminate it outright (no time for a full root-cause dive at this
# point in the deadline - PI directive is to keep moving, document honestly).

# HOUSEKEEPING 2026-08-13 (repository cleanup): moved into run_scripts/ - see main.py's own
# shim comment for why this sys.path prepend is here. Snapshot/plot dirs now go through
# output.run_output_dirs (the outputs/ consolidation) instead of bare top-level literals.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import output
import time_stepper

RESUME_SNAPSHOT_DIR, _ = output.run_output_dirs("Phase1_first_core_resumed")
RESUME_STEP = 3

SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs("Phase1_first_core_resumed2")
N_STEPS_MAX = 300

DT_SEED = 5.0 * config.SECONDS_PER_YEAR   # verified directly against the actual failure


def main():
    config.MU = 2.34
    config.GAMMA = 1.4
    config.USE_H2_RECOMBINATION_PHYSICS = False
    config.USE_ADAPTIVE_DT = True
    config.ADAPTIVE_DT_MAX = 1.0e4 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_MIN = 5.0 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_GROWTH_FACTOR = 1.1   # gentled further (was 1.3, then 1.15)
    config.T_MAX_S = 1.0e6 * config.SECONDS_PER_YEAR

    resume_path = f"{RESUME_SNAPSHOT_DIR}/snapshot_{RESUME_STEP:05d}.npz"
    print(f"resume_phase1_from_step21: resuming from {resume_path} ...", flush=True)
    state_resumed, _is_convective = output.load_snapshot(resume_path)
    print(f"resume_phase1_from_step21: resumed state at t={state_resumed.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={state_resumed.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={state_resumed.T[0]:.6e} K", flush=True)

    print(f"resume_phase1_from_step21: starting continuation run, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_resumed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("resume_phase1_from_step21: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"resume_phase1_from_step21: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
