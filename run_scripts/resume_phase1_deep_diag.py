# resume_phase1_deep_diag.py — Resumes from the T_center~1561K snapshot with the FULL current
# fix set active (this session's NaN-safe alpha-blend, xtol fix, retry mechanism + the other
# session's latent_heat_capacity/mean_molecular_weight_inv_derivative/_h2_transition_derivatives
# USE_H2_RECOMBINATION_PHYSICS-gating fix, git commit 8531c5f), to reproduce and deep-diagnose
# wherever the run now actually stalls - the user wants real data (location, mechanism), not
# another blind fix attempt.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import output
import time_stepper

RESUME_SNAPSHOT_DIR, _ = output.run_output_dirs("Phase1_first_core_resumed4")
RESUME_STEP = 0

SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs("Phase1_deep_diag")
N_STEPS_MAX = 500

DT_SEED = 5.0 * config.SECONDS_PER_YEAR


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
    print(f"resume_phase1_deep_diag: resuming from {resume_path} ...", flush=True)
    state_resumed, _is_convective = output.load_snapshot(resume_path)
    print(f"resume_phase1_deep_diag: resumed state at t={state_resumed.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={state_resumed.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={state_resumed.T[0]:.6e} K", flush=True)

    print(f"resume_phase1_deep_diag: starting continuation run, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_resumed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("resume_phase1_deep_diag: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"resume_phase1_deep_diag: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
