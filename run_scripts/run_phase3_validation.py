# run_phase3_validation.py — OFFICIAL Phase 3 engine validation run (2026-08-10), Sub-task 8b
# H<->H2 recombination physics now live throughout the production solver (eos.py/odes.py/
# bvp_solver.py, PROGRESS.md 2026-08-10 evening entry has the full implementation trail).
#
# Unlike the earlier overnight_run.py/extended_run_10gyr.py runs (which used the OLD constant-
# atomic-mu physics throughout, and for the 10gyr run specifically RESUMED from an old
# snapshot), this is a FRESH t=0 start under the FULLY CORRECTED physics from the first step -
# the only way to cleanly see the new physics' CUMULATIVE effect over the whole trajectory,
# not a tail-end continuation mixing old and new physics.
#
# Target: config.R_HALT (1 R_Jup) or 4.5 Gyr (the solar system's age - the physically
# meaningful validation checkpoint for this run, NOT the 10 Gyr diagnostic budget used for the
# earlier asymptotic-behavior investigation - explicit user request 2026-08-10), whichever
# comes first. If this converges cleanly end-to-end with no numerical explosions and shows
# physically sensible contraction, Phase 3 is considered validated and the project pivots to
# building the Phase 1 (First Core) driver next.
#
# Same safety nets as every prior production run: the NaN/positivity guard and the dual
# R_HALT/T_MAX_S stopping condition in time_stepper.run() apply unchanged.

# HOUSEKEEPING 2026-08-13 (repository cleanup): moved into run_scripts/ - see main.py's own
# shim comment for why this sys.path prepend is here.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bvp_solver
import config
import output
import time_stepper

RUN_NAME = "Phase3_recombination"
SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs(RUN_NAME)

N_STEPS_MAX = 150   # generous cap - the dual halt condition is the primary expected stop, not this step count
DT_SEED = config.RELAX_DT_FRACTION * config.T_KH_TIMESCALE_S


def main():
    config.USE_ADAPTIVE_DT = True
    # This run's own validation checkpoint is 4.5 Gyr (solar system age), not the 10 Gyr
    # diagnostic budget config.py currently defaults to (that was set for a DIFFERENT purpose -
    # testing whether the old physics' contraction asymptotes or keeps going, PROGRESS.md
    # 2026-08-10) - overridden here rather than changed in config.py, since 10 Gyr remains the
    # right default for that earlier diagnostic use case.
    config.T_MAX_S = 4.5e9 * config.SECONDS_PER_YEAR

    print("run_phase3_validation: building t=0 compact hot start (solve_static_structure) ...", flush=True)
    state_0 = bvp_solver.solve_static_structure()

    print("run_phase3_validation: relaxing to a genuine solution of the full 4-ODE system "
          "(relax_initial_state) ...", flush=True)
    state_relaxed = bvp_solver.relax_initial_state(state_0)

    print(f"run_phase3_validation: starting OFFICIAL VALIDATION run (Sub-task 8b physics live "
          f"throughout), n_steps={N_STEPS_MAX}, seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"R_HALT={config.R_HALT / config.R_JUPITER_CM:.3f} R_Jup, "
          f"T_MAX_S={config.T_MAX_S / config.SECONDS_PER_YEAR:.3e} yr, "
          f"snapshot_dir={SNAPSHOT_DIR}, plot_dir={PLOT_DIR}", flush=True)
    history = time_stepper.run(state_relaxed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("run_phase3_validation: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"run_phase3_validation: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}.", flush=True)
    print(f"Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
