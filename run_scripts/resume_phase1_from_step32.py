# resume_phase1_from_step32.py — Fourth resume of the Phase 1 First Hydrostatic Core run
# (PROGRESS.md has the full report), continuing from the persistent wall at T_center~1561K,
# r~238 R_Jup ("step 32" in original numbering). This resume runs with the 2026-08-12 NaN-safe
# alpha-blend fix active (implicit_rhs_vectorized/implicit_rhs_jacobian) - the wall here did
# NOT resolve via dt-shrinking (down to 28 days) or GRAD_EFF_SWITCH_EPSILON_TIMESTEP widening
# (5.0, 10.0), unlike every smaller pocket recovered earlier in this run - root-caused instead
# to a genuine gap in the alpha-continuation ladder itself (0.0*nan=nan in IEEE float
# arithmetic means alpha=0 does not actually neutralize a pathological grad_rad/opacity
# evaluation on an extreme Newton trial in the outer envelope's narrow Ice-grain-evaporation
# crossing), now fixed and directly verified against finite differences before this run.

# HOUSEKEEPING 2026-08-13 (repository cleanup): moved into run_scripts/ - see main.py's own
# shim comment for why this sys.path prepend is here. Snapshot/plot dirs now go through
# output.run_output_dirs (the outputs/ consolidation) instead of bare top-level literals.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import output
import time_stepper

RESUME_SNAPSHOT_DIR, _ = output.run_output_dirs("Phase1_first_core_resumed3")
RESUME_STEP = 2

SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs("Phase1_first_core_resumed4")
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
    print(f"resume_phase1_from_step32: resuming from {resume_path} (the T_center~1561K wall, "
          f"NaN-safe alpha-blend fix now active) ...", flush=True)
    state_resumed, _is_convective = output.load_snapshot(resume_path)
    print(f"resume_phase1_from_step32: resumed state at t={state_resumed.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={state_resumed.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={state_resumed.T[0]:.6e} K", flush=True)

    print(f"resume_phase1_from_step32: starting continuation run, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K, "
          f"snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_resumed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("resume_phase1_from_step32: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"resume_phase1_from_step32: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
