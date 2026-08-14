# run_phase1_stress_test_past_halt.py — 2026-08-13. DELIBERATE numerics stress test, explicitly
# requested: push past config.py's own PHASE1_T_CENTER_HALT=1900K target halt to find where the
# CURRENT solver (not the physics) actually gives out.
#
# NOT A PHYSICS RUN PAST ~1900-2000K: composition is held artificially fixed throughout
# (USE_H2_RECOMBINATION_PHYSICS=False, same as every other Phase 1 run) - no H2 dissociation is
# modeled, so nothing this script produces above T_center~1900-2000K represents real physics.
# PHASE1_T_CENTER_HALT exists specifically because ~2000K is where Gamma_1 would soften below
# 4/3 in reality and the quasi-static/ideal-gas assumption breaks down (PLAN.md, config.py's own
# comment) - this script intentionally continues past that validity boundary anyway, purely to
# characterize the solver's own numerical failure point (exact T_center/node count/iteration/
# error message), not to produce a trustworthy extended Phase 1 trajectory. Treat everything
# reported above ~1900-2000K as a numerics diagnostic, not a physical result.
#
# Resumes from the just-completed baseline run's own final, successfully-halted state
# (Phase1_baseline_rerun_20260813/snapshot_00035.npz, T_center=1923.684K, t=1518.6 yr) rather
# than re-running the whole contraction from t=0 - PHASE1_T_CENTER_HALT is overridden HIGH
# (effectively disabled) so the ONLY things that can stop this run are R_HALT/T_MAX_S (generous
# backstops, not expected to bind) or a genuine solve_timestep failure that exhausts the
# existing STEP_RETRY_MAX_ATTEMPTS retry budget - i.e. an actual crash, which is what we're
# looking for. Own output directory, so this doesn't touch the successful baseline run's data.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import output
import time_stepper

RESUME_SNAPSHOT_DIR, _ = output.run_output_dirs("Phase1_baseline_rerun_20260813")
RESUME_STEP = 35   # the halted run's final, successfully-converged step (T_center=1923.684K)

SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs("Phase1_stress_test_20260813")
N_STEPS_MAX = 300


def main():
    # Same Phase 1 composition/timescale overrides as run_phase1_baseline_rerun.py - see that
    # script's docstring. config.py's own persisted defaults are untouched by any of this.
    config.MU = 2.34
    config.GAMMA = 1.4
    config.USE_H2_RECOMBINATION_PHYSICS = False
    config.USE_ADAPTIVE_DT = True
    config.ADAPTIVE_DT_MAX = 1.0e4 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_MIN = 1.0e1 * config.SECONDS_PER_YEAR
    config.T_MAX_S = 1.0e6 * config.SECONDS_PER_YEAR

    # THE deliberate override for this stress test: disable the 1900K target halt so the run
    # continues until it either hits R_HALT/T_MAX_S (not expected) or genuinely fails.
    config.PHASE1_T_CENTER_HALT = 10000.0   # [K] effectively disabled - see module docstring

    resume_path = f"{RESUME_SNAPSHOT_DIR}/snapshot_{RESUME_STEP:05d}.npz"
    print(f"run_phase1_stress_test_past_halt: resuming from {resume_path} "
          f"(the baseline run's own successful halt state) ...", flush=True)
    state_resumed, _is_convective = output.load_snapshot(resume_path)
    print(f"run_phase1_stress_test_past_halt: resumed at t={state_resumed.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={state_resumed.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={state_resumed.T[0]:.6e} K - PHASE1_T_CENTER_HALT overridden to "
          f"{config.PHASE1_T_CENTER_HALT:.1f} K (effectively disabled) for this stress test only", flush=True)

    dt_seed = 3.443e1 * config.SECONDS_PER_YEAR   # matches the resumed state's own last-successful dt

    print(f"run_phase1_stress_test_past_halt: continuing past the 1900K target, n_steps={N_STEPS_MAX}, "
          f"seed dt={dt_seed / config.SECONDS_PER_YEAR:.4e} yr, snapshot_dir={SNAPSHOT_DIR}", flush=True)
    history = time_stepper.run(state_resumed, N_STEPS_MAX, dt_seed, snapshot_dir=SNAPSHOT_DIR)

    print("run_phase1_stress_test_past_halt: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"run_phase1_stress_test_past_halt: done (no crash), {len(history)} snapshots saved. "
          f"Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
