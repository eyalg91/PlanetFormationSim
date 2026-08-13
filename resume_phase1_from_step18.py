# resume_phase1_from_step18.py — Resumes run_phase1_first_core.py from its last surviving
# snapshot (step 18, PROGRESS.md has the full report) after a new wall at step 19 (the raw
# trial state reached literal inf, a linear-solve overflow, not something the soft clamp alone
# prevents - not chased to full diagnosis given the deadline; same "no more band-aids" honesty
# applies: this is a targeted, VERIFIED recovery, not a guess).
#
# FIX, verified directly against this exact failing (state, dt) before use: the growth-capped
# dt (34.7 yr) fails; so does an intermediate 20 yr (after a pathological 262s hang - same
# "certain intermediate values are numerically pathological, not simply too-big/too-small"
# pattern found for Phase 3's own epsilon sweep); dt=10 yr converges directly, cleanly, fast.
# ADAPTIVE_DT_GROWTH_FACTOR also gentled (1.3 -> 1.15) for the remainder of this run, reducing
# (not eliminating) the chance of the adaptive stepper overshooting into another such pocket -
# more steps, not a different approach; same resume-into-a-new-directory pattern as
# resume_phase3_from_step3.py, so the original run's surviving snapshots stay untouched.

import config
import output
import time_stepper

RESUME_SNAPSHOT_DIR = "snapshots_Phase1_first_core"
RESUME_STEP = 18

SNAPSHOT_DIR = "snapshots_Phase1_first_core_resumed"
PLOT_DIR = "diagnostic_plots/run_Phase1_first_core_resumed"
N_STEPS_MAX = 200

DT_SEED = 10.0 * config.SECONDS_PER_YEAR   # verified directly against the actual step-19 failure


def main():
    config.MU = 2.34
    config.GAMMA = 1.4
    config.USE_H2_RECOMBINATION_PHYSICS = False
    config.USE_ADAPTIVE_DT = True
    config.ADAPTIVE_DT_MAX = 1.0e4 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_MIN = 1.0e1 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_GROWTH_FACTOR = 1.15   # gentled from 1.3 - see module docstring
    config.T_MAX_S = 1.0e6 * config.SECONDS_PER_YEAR

    resume_path = f"{RESUME_SNAPSHOT_DIR}/snapshot_{RESUME_STEP:05d}.npz"
    print(f"resume_phase1_from_step18: resuming from {resume_path} ...", flush=True)
    state_resumed, _is_convective = output.load_snapshot(resume_path)
    print(f"resume_phase1_from_step18: resumed state at t={state_resumed.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={state_resumed.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={state_resumed.T[0]:.6e} K", flush=True)

    print(f"resume_phase1_from_step18: starting continuation run, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_resumed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("resume_phase1_from_step18: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"resume_phase1_from_step18: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
