# resume_phase3_from_step3.py — Resumes the OFFICIAL Phase 3 validation run
# (run_phase3_validation.py) from its last surviving snapshot (step 3), after the 2026-08-11
# solver architecture course correction (soft P/T clamp, its and the smoothed opacity's fixed
# analytic Jacobian, GRAD_EFF_SWITCH_EPSILON_TIMESTEP raised 0.5->2.0 - PROGRESS.md 2026-08-11
# entry has the full diagnosis). Step 4 (the original failure point) is directly confirmed to
# converge with today's fixes before launching this - see that entry for the reconstructed-dt
# verification.
#
# Same resume pattern as extended_run_10gyr.py: writes to a NEW snapshot directory rather than
# renumbering into snapshots_Phase3_recombination/ (which would collide with the existing
# steps 0-3), so the original run's surviving snapshots stay untouched for direct comparison.
#
# Target: unchanged from the original run - config.R_HALT (1 R_Jup) or 4.5 Gyr, whichever
# comes first (run_phase3_validation.py's own override, reproduced here since this script
# doesn't go through that module's main()).

import config
import output
import time_stepper

RESUME_SNAPSHOT_DIR = "snapshots_Phase3_recombination"
RESUME_STEP = 3

SNAPSHOT_DIR = "snapshots_Phase3_recombination_resumed"
PLOT_DIR = "diagnostic_plots/run_Phase3_recombination_resumed"
N_STEPS_MAX = 150   # generous cap, matching run_phase3_validation.py's own budget - the dual
                     # halt condition (R_HALT or T_MAX_S) is the primary expected stop.

# Reconstructed exact dt that produced (the never-saved) step 4 in the original run, chained
# forward from the recorded snapshot t-values via time_stepper.select_adaptive_dt - verified
# to reproduce each intermediate snapshot's own t exactly (PROGRESS.md 2026-08-11 entry).
# Seeding the resume here (not the original RELAX_DT_FRACTION*T_KH_TIMESCALE_S bootstrap
# value) is the physically continuous choice - select_adaptive_dt takes over from the next
# step onward exactly as it would have in the uninterrupted original run.
DT_SEED = 3.661573e11   # s


def main():
    config.USE_ADAPTIVE_DT = True
    config.T_MAX_S = 4.5e9 * config.SECONDS_PER_YEAR   # matches run_phase3_validation.py's own override

    resume_path = f"{RESUME_SNAPSHOT_DIR}/snapshot_{RESUME_STEP:05d}.npz"
    print(f"resume_phase3_from_step3: resuming from {resume_path} ...", flush=True)
    state_resumed, _is_convective = output.load_snapshot(resume_path)
    print(f"resume_phase3_from_step3: resumed state at t={state_resumed.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={state_resumed.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={state_resumed.T[0]:.6e} K", flush=True)

    print(f"resume_phase3_from_step3: starting continuation run, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"R_HALT={config.R_HALT / config.R_JUPITER_CM:.3f} R_Jup, "
          f"T_MAX_S={config.T_MAX_S / config.SECONDS_PER_YEAR:.3e} yr, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_resumed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("resume_phase3_from_step3: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"resume_phase3_from_step3: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
