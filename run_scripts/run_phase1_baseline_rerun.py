# run_phase1_baseline_rerun.py — 2026-08-13. Identical to run_phase1_first_core.py (same
# calibrated seed, same hermetic runtime overrides, same physical picture - see that script's
# docstring for the full rationale) EXCEPT for RUN_NAME: this is a fresh t=0 baseline run under
# the CURRENT codebase (NaN-safe alpha-blend, H2-recombination-gating fix, brentq xtol fix,
# ideal-gas Lane-Emden seed, automatic step-retry all active), used specifically to find the
# exact crash state for fresh troubleshooting. Kept in its own output directory rather than
# reusing Phase1_first_core's, because output.save_snapshot does not clear its directory first -
# reusing that name would leave the OLD run's 33 stale snapshots (a different fix-set, halted at
# T_center=1560.89K) mixed in with this run's files, contaminating both the crash report and
# output.generate_all_plots' glob-based file discovery.
#
# No speculative fixes applied here - this is a clean measurement, not an attempt to push past
# any wall.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bvp_solver
import config
import output
import time_stepper

RUN_NAME = "Phase1_baseline_rerun_20260813"
SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs(RUN_NAME)

N_STEPS_MAX = 200

T_CENTER_INITIAL_PHASE1 = 645.0   # [K] -> r_surface=500.83 R_Jup (2026-08-12 calibration)
DT_SEED = 1.0e1 * config.SECONDS_PER_YEAR
RELAX_DT_FRACTION_PHASE1 = 1.0e-3
ADAPTIVE_DT_MIN_PHASE1 = 1.0e1 * config.SECONDS_PER_YEAR


def main():
    config.MU = 2.34
    config.GAMMA = 1.4
    config.USE_H2_RECOMBINATION_PHYSICS = False
    config.T_CENTER_INITIAL = T_CENTER_INITIAL_PHASE1
    config.USE_ADAPTIVE_DT = True
    config.ADAPTIVE_DT_MAX = 1.0e4 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_MIN = ADAPTIVE_DT_MIN_PHASE1
    config.T_MAX_S = 1.0e6 * config.SECONDS_PER_YEAR
    config.RELAX_DT_FRACTION = RELAX_DT_FRACTION_PHASE1

    print(f"run_phase1_baseline_rerun: building t=0 diffuse molecular first core "
          f"(T_CENTER_INITIAL={config.T_CENTER_INITIAL:.1f}K, MU={config.MU}, GAMMA={config.GAMMA}) ...", flush=True)
    state_0 = bvp_solver.solve_static_structure(use_ideal_gas_seed=True)

    print("run_phase1_baseline_rerun: relaxing to a genuine solution of the full 4-ODE system ...", flush=True)
    state_relaxed = bvp_solver.relax_initial_state(state_0, force_clamp_off_stage1=False)

    print(f"run_phase1_baseline_rerun: starting Phase 1 quasi-static contraction, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_relaxed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("run_phase1_baseline_rerun: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"run_phase1_baseline_rerun: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}. "
          f"Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
