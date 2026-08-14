# run_phase1_high_res_overnight.py — 2026-08-14. Dedicated high-resolution overnight Phase 1
# run, ~2 hour compute budget. SAME physical constraints as the successful baseline run
# (run_phase1_baseline_rerun.py): MU=2.34, GAMMA=1.4, USE_H2_RECOMBINATION_PHYSICS=False,
# T_CENTER_INITIAL=645K, PHASE1_T_CENTER_HALT=1900K - no speculative physics changes, only
# resolution/precision numerics are increased. Own output directory - does not touch any
# earlier run's snapshots.
#
# RESOLUTION CHOICES - REVISED after a first attempt genuinely crashed (see PROGRESS.md
# 2026-08-14 entry for the full account, kept here in brief since it directly explains why
# these values look the way they do): the first version also raised BVP_MESH_N_GRID_POINTS
# (2000->3000), BVP_MAX_NODES (80000->120000), and BVP_COLLOCATION_TOL (1e-6->1e-7). That
# combination made relax_initial_state's stage 1 - which had not failed ONCE, at any T_center,
# anywhere in this project's session-long investigation, under the proven default tolerance -
# diverge on its very first alpha=0.5 continuation rung. The iteration trace showed WHY: the
# residual already reached ~6-9e-7 (comfortably under the OLD 1e-6 tolerance) by iteration 8-9,
# then kept refining past that point chasing the NEW, tighter 1e-7 target, and that additional
# refinement is exactly what ran away (node count climbing past 55,000 while the residual grew
# back up to 5e-3). This system has a genuine non-smooth feature near this same alpha=0.5
# transition (the convective-saturation Jacobian degeneracy under active investigation,
# PROGRESS.md's 2026-08-13 entries) that limits how tightly it can actually be resolved before
# mesh refinement starts chasing noise rather than real structure - asking for MORE precision
# than that made things categorically worse, not better.
#
# Reverted BVP_MESH_N_GRID_POINTS/BVP_MAX_NODES/BVP_COLLOCATION_TOL to config.py's own proven
# defaults (2000/80000/1e-6 - the EXACT combination validated twice already tonight, in the
# baseline run and the stress test) by simply not overriding them at all. Kept only the
# resolution increases that are provably free or orthogonal to solve_bvp's own internal
# mesh/tolerance economy:
#   N_GRID_POINTS             200 -> 1000    (OUTPUT/reporting grid only - free, no solver cost;
#                                              purely denser sampling of the converged solution
#                                              for snapshots/plots)
#   GRID_OUTER_REFINEMENT     1e-4 -> 1e-6    (OUTPUT grid's photosphere resolution - reuses
#                                              BVP_MESH_OUTER_REFINEMENT's own already-vetted
#                                              value, not a new untested number)
#   ADAPTIVE_DT_GROWTH_FACTOR 1.3 -> 1.15     (explicitly requested "tighter adaptive dt growth
#                                              cap" - a TIME-stepping choice, doesn't touch
#                                              solve_bvp's internal mesh/tolerance at all - finer
#                                              temporal resolution, more/smaller steps, at the
#                                              cost of needing more steps to cover the same
#                                              physical time span)
# Everything else (composition, T_CENTER_INITIAL, RELAX_DT_FRACTION, ADAPTIVE_DT_MIN/MAX,
# T_MAX_S, PHASE1_T_CENTER_HALT, GRAD_EFF_SWITCH_EPSILON_TIMESTEP, OPACITY_TRANSITION_SMOOTH_
# WIDTH_DEX, the alpha-continuation ladder, and now also BVP_MESH_N_GRID_POINTS/BVP_MAX_NODES/
# BVP_COLLOCATION_TOL) is UNCHANGED from run_phase1_baseline_rerun.py - no speculative physics
# fixes, per explicit instruction, and after tonight's finding, no speculative SOLVER-precision
# increases either.
#
# UNATTENDED-RUN ROBUSTNESS (2026-08-14):
# - time_stepper.run's new max_wall_clock_s parameter (this session's own small, additive,
#   backward-compatible addition - see that function's docstring) caps the main time loop to
#   MAX_WALL_CLOCK_S below, so a retry-storm near a fragile step (2026-08-13 stress-test
#   findings) cannot silently consume the entire overnight budget stuck on one step.
# - If the run fails for ANY reason (including a step that exhausts its own retry budget and
#   raises, matching the 2026-08-13 stress test's own crash mode), the full traceback is
#   printed AND plots are still generated from whatever snapshots converged before the failure -
#   the point is that there is always something to review in the morning, not a bare crash with
#   no output. Exit code reflects whether the run completed cleanly or not.

import os
import sys
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bvp_solver
import config
import output
import time_stepper

RUN_NAME = "Phase1_high_res_overnight_20260813"
SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs(RUN_NAME)

N_STEPS_MAX = 500                 # generous cap - physical/wall-clock halts expected to bind first
MAX_WALL_CLOCK_S = 105.0 * 60.0   # 105 min for the time loop itself, leaving ~15 min of the ~2hr budget for seed/relax/plotting

# Same Phase 1 physical calibration as run_phase1_baseline_rerun.py - unchanged.
T_CENTER_INITIAL_PHASE1 = 645.0   # [K] -> r_surface=500.83 R_Jup
DT_SEED = 1.0e1 * config.SECONDS_PER_YEAR
RELAX_DT_FRACTION_PHASE1 = 1.0e-3
ADAPTIVE_DT_MIN_PHASE1 = 1.0e1 * config.SECONDS_PER_YEAR

# Resolution/precision increases - see module docstring for the reasoning behind each value
# (and why BVP_MESH_N_GRID_POINTS/BVP_MAX_NODES/BVP_COLLOCATION_TOL are deliberately NOT
# overridden here, after the first attempt at raising them crashed relax_initial_state outright).
N_GRID_POINTS_HIRES = 1000
GRID_OUTER_REFINEMENT_HIRES = 1.0e-6
ADAPTIVE_DT_GROWTH_FACTOR_HIRES = 1.15


def main():
    # Physical constraints - identical to run_phase1_baseline_rerun.py.
    config.MU = 2.34
    config.GAMMA = 1.4
    config.USE_H2_RECOMBINATION_PHYSICS = False
    config.T_CENTER_INITIAL = T_CENTER_INITIAL_PHASE1
    config.USE_ADAPTIVE_DT = True
    config.ADAPTIVE_DT_MAX = 1.0e4 * config.SECONDS_PER_YEAR
    config.ADAPTIVE_DT_MIN = ADAPTIVE_DT_MIN_PHASE1
    config.T_MAX_S = 1.0e6 * config.SECONDS_PER_YEAR
    config.RELAX_DT_FRACTION = RELAX_DT_FRACTION_PHASE1
    config.PHASE1_T_CENTER_HALT = 1900.0

    # Resolution/precision - see module docstring. BVP_MESH_N_GRID_POINTS/BVP_MAX_NODES/
    # BVP_COLLOCATION_TOL are deliberately left at config.py's own proven defaults (not
    # overridden at all) after raising them broke relax_initial_state's stage 1 outright.
    config.N_GRID_POINTS = N_GRID_POINTS_HIRES
    config.GRID_OUTER_REFINEMENT = GRID_OUTER_REFINEMENT_HIRES
    config.ADAPTIVE_DT_GROWTH_FACTOR = ADAPTIVE_DT_GROWTH_FACTOR_HIRES

    crashed = False
    history = None
    try:
        print(f"run_phase1_high_res_overnight: building t=0 diffuse molecular first core "
              f"(T_CENTER_INITIAL={config.T_CENTER_INITIAL:.1f}K, MU={config.MU}, GAMMA={config.GAMMA}), "
              f"N_GRID_POINTS={config.N_GRID_POINTS} (output grid), "
              f"BVP_MESH_N_GRID_POINTS={config.BVP_MESH_N_GRID_POINTS}/BVP_MAX_NODES={config.BVP_MAX_NODES}/"
              f"BVP_COLLOCATION_TOL={config.BVP_COLLOCATION_TOL:.1e} (solver internals, left at proven defaults), "
              f"ADAPTIVE_DT_GROWTH_FACTOR={config.ADAPTIVE_DT_GROWTH_FACTOR} ...", flush=True)
        state_0 = bvp_solver.solve_static_structure(use_ideal_gas_seed=True)

        print("run_phase1_high_res_overnight: relaxing to a genuine solution of the full 4-ODE system ...", flush=True)
        state_relaxed = bvp_solver.relax_initial_state(state_0, force_clamp_off_stage1=False)

        print(f"run_phase1_high_res_overnight: starting Phase 1 quasi-static contraction, n_steps={N_STEPS_MAX}, "
              f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
              f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K, "
              f"max_wall_clock={MAX_WALL_CLOCK_S:.0f}s, snapshot_dir={SNAPSHOT_DIR}", flush=True)
        history = time_stepper.run(state_relaxed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR,
                                    max_wall_clock_s=MAX_WALL_CLOCK_S)
    except Exception:
        print("run_phase1_high_res_overnight: *** the run raised - full traceback below. "
              "Attempting to generate plots from whatever snapshots converged before the "
              "failure (if any), so there is something to review regardless. ***", flush=True)
        traceback.print_exc()
        crashed = True

    snapshots_exist = os.path.isdir(SNAPSHOT_DIR) and len(os.listdir(SNAPSHOT_DIR)) > 0
    if snapshots_exist:
        print("run_phase1_high_res_overnight: generating plots from the saved snapshots ...", flush=True)
        output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                                   profile_snapshot_indices=None)
    else:
        print("run_phase1_high_res_overnight: no snapshots were saved before the failure - nothing to plot.", flush=True)

    if crashed:
        print("run_phase1_high_res_overnight: done (with a crash - see traceback above).", flush=True)
        sys.exit(1)

    final = history[-1]
    print(f"run_phase1_high_res_overnight: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
