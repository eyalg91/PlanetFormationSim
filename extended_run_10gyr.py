# extended_run_10gyr.py — Sub-task 9/10 continuation run (2026-08-10, explicit user request).
#
# Purpose: the first full run (overnight_run.py, snapshots in snapshots_overnight/) reached
# only r_surface=4.8661 R_Jup at t=4.5215e9 yr (config.T_MAX_S's PREVIOUS value, 4.5 Gyr, the
# solar system's present-day age) - almost no net contraction from the ~5.1 R_Jup starting
# radius. The user wants to know whether this is genuinely slow Kelvin-Helmholtz contraction,
# or an artificial equilibrium floor from missing physics (MLT, Saha, atomic-only mu(T)).
# config.T_MAX_S is now 10 Gyr, purely as a diagnostic time budget (config.py has the full
# reasoning) - this script extends the trajectory out to that new limit (or R_HALT).
#
# RESUMES rather than re-solves: loads the LAST snapshot of the completed 4.5 Gyr run
# (snapshot_00077.npz, already a converged solve_timestep state) instead of rebuilding
# state_0 -> relax_initial_state -> re-running all 77 steps from scratch. This is both a
# compute-time saving (avoids ~77 redundant BVP solves) and a direct real-world test of
# output.py's snapshot-resumability design (Sub-task 10's .npz I/O was built precisely so a
# run's progress survives independently of the process that produced it).
#
# Seed dt: the last 8 steps of the 4.5 Gyr run were ALL pinned at dt=1.0000e+08 yr
# (config.ADAPTIVE_DT_MAX, the defensive backstop) - i.e. the raw T/P-timescale formula's own
# estimate had already caught up to the ceiling, not the growth cap. Seeding the resume at
# that same value (rather than the original RELAX_DT_FRACTION*T_KH_TIMESCALE_S bootstrap
# value, which was only ever appropriate for the very first step out of relax_initial_state)
# is the physically continuous choice - select_adaptive_dt takes over from step 2 onward
# exactly as before.
#
# Uses SEPARATE snapshot/plot directories from the original run so snapshots_overnight/ and
# diagnostic_plots_overnight/ (the complete, reviewed 0-4.5 Gyr trajectory) are preserved
# untouched for direct before/after comparison - see combine_10gyr_plot.py for the merged
# full-trajectory view.

import config
import output
import time_stepper

RESUME_SNAPSHOT_DIR = "snapshots_overnight"
RESUME_STEP = 77

N_STEPS_MAX = 200   # generous cap - see module docstring; the dual halt condition (R_HALT or
                     # config.T_MAX_S) is the primary expected stop, not this step count.
                     # Reaching 10 Gyr from 4.5 Gyr needs ~55 more steps if dt stays pinned at
                     # the 1e8 yr ADAPTIVE_DT_MAX ceiling throughout - 200 leaves generous
                     # margin in case dt ever drops back down as the structure evolves further.
SNAPSHOT_DIR = "snapshots_10gyr"
PLOT_DIR = "diagnostic_plots/run_10gyr"   # HOUSEKEEPING 2026-08-10: moved under diagnostic_plots/ to stop cluttering the project root
DT_SEED = config.ADAPTIVE_DT_MAX   # matches the last dt actually used to reach step 77 (see module docstring)


def main():
    config.USE_ADAPTIVE_DT = True

    resume_path = f"{RESUME_SNAPSHOT_DIR}/snapshot_{RESUME_STEP:05d}.npz"   # matches output._snapshot_path's naming convention
    print(f"extended_run_10gyr: resuming from {resume_path} ...", flush=True)
    state_resumed, _is_convective = output.load_snapshot(resume_path)
    print(f"extended_run_10gyr: resumed state at t={state_resumed.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={state_resumed.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={state_resumed.T[0]:.6e} K", flush=True)

    print(f"extended_run_10gyr: starting continuation run, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"T_MAX_S={config.T_MAX_S / config.SECONDS_PER_YEAR:.3e} yr, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_resumed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("extended_run_10gyr: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"extended_run_10gyr: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
