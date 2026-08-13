# run_phase1_first_core.py — Phase 1 (First Hydrostatic Core) validation run (2026-08-12, PI-
# directed pivot away from Phase 3's unresolved step-5 wall - PROGRESS.md has the full report).
#
# PHYSICAL PICTURE (PLAN.md "Formation Scenario and Scope", Stage 1): a diffuse, fully
# molecular, ideal-gas-supported protoplanet, R~300-1000 R_Jup, contracting quasi-statically
# (Kelvin-Helmholtz) until T_center approaches ~2000K, where H2 dissociation softens Gamma_1
# below 4/3 and triggers the out-of-scope Stage 2 dynamical collapse. Composition held
# STRICTLY constant/molecular throughout (MU=2.34, GAMMA=1.4, no recombination physics) - the
# dissociation transition itself is deliberately never reached (halted just below it), so there
# is no composition drift to model here, unlike Phase 3's post-second-collapse atomic envelope.
#
# HERMETIC ISOLATION FROM PHASE 3 (explicit PI directive): every Phase-1-specific value below
# is a RUNTIME override local to this script's own process, exactly matching how
# run_phase3_validation.py already overrides T_MAX_S - config.py's own persisted defaults
# (atomic MU/GAMMA, USE_H2_RECOMBINATION_PHYSICS=True, T_CENTER_INITIAL=11500K, Gyr-scale
# T_MAX_S/ADAPTIVE_DT_MAX) are never touched, so Phase 3 remains exactly as it was for the
# thesis's return to it later.
#
# TIMESCALE (PI directive, Larson 1969: the First Core phase lasts ~1e4-1e5 yr, not Phase 3's
# Gyr scale): config.T_KH_TIMESCALE_S=1 Myr would give a seed dt of ~1e4 yr (RELAX_DT_FRACTION*
# T_KH_TIMESCALE_S) - already comparable to the WHOLE expected process - and config.
# ADAPTIVE_DT_MAX=1e8 yr is 1,000-10,000x it. Both overridden here to timescales appropriate to
# this phase, so the contraction curve is resolved into many real steps, not a handful of huge
# ones.

import bvp_solver
import config
import output
import time_stepper

RUN_NAME = "Phase1_first_core"
SNAPSHOT_DIR, PLOT_DIR = output.run_output_dirs(RUN_NAME)

N_STEPS_MAX = 200   # generous cap - config.PHASE1_T_CENTER_HALT is the expected stop, not this count

# Calibrated 2026-08-12 via bvp_solver.solve_static_structure(use_ideal_gas_seed=True) directly
# (PROGRESS.md has the full calibration sweep): 645K -> r_surface=500.83 R_Jup, the requested
# R~500 R_Jup target. Found alongside a genuine, dimensionally-wrong brentq xtol bug
# (config.STATIC_STRUCTURE_BRENTQ_XTOL, bvp_solver.py) that was silently limiting
# solve_static_structure's root-find precision in BOTH regimes - fixed as part of this
# calibration, not routed around; residuals dropped from ~1e-2 (marginal) to ~1e-10 (exact) at
# every T_center tested, including Phase 3's own T_CENTER_INITIAL=11500K case.
T_CENTER_INITIAL_PHASE1 = 645.0   # [K]

DT_SEED = 1.0e1 * config.SECONDS_PER_YEAR                  # ~10 yr - revised down from an initial 100 yr guess after Step 6's sterile validation found the very first real solve_timestep call itself needed a gentler seed (100 yr mesh-exploded; 10 yr converged directly, no continuation fallback) - the same "smaller step near a fresh warm-start" lesson relax_initial_state's own dt already needed, empirically confirmed rather than assumed to transfer

# Found during Step 6's sterile validation (PROGRESS.md has the full account), not assumed:
# relax_initial_state's stage 1, run under its DEFAULT settings (clamp forced off, dt_relax=
# RELAX_DT_FRACTION*T_KH_TIMESCALE_S=1e4 yr - both tuned for Phase 3's compact/Gyr-scale
# regime), crashes outright for this diffuse structure - clamp off means a raw np.exp()
# overflow reaches eos.density uncaught. Direct, isolated tests found BOTH a smaller relax
# pseudo-timestep AND leaving the clamp ON (its global default, already fixed/verified today)
# are independently necessary; together, first-attempt convergence, no continuation fallback
# needed. Both overrides local to this script only.
RELAX_DT_FRACTION_PHASE1 = 1.0e-3   # RELAX_DT_FRACTION*T_KH_TIMESCALE_S = 1000 yr (vs Phase 3's 1e4 yr)

# Found immediately after the DT_SEED revision above: config.ADAPTIVE_DT_MIN=100yr (inherited,
# unchanged) sits ABOVE the 10yr seed - select_adaptive_dt's own clip(..., dt_min, dt_max)
# forcibly clamps step 2's dt straight back up to 100yr regardless of the growth cap's 1.3x
# intent, defeating the gentler seed on the very next step (confirmed directly: step 1 at 10yr
# converged cleanly, step 2 immediately jumped to 100yr and failed the same way 100yr did
# before). Overridden to sit at or below the seed.
ADAPTIVE_DT_MIN_PHASE1 = 1.0e1 * config.SECONDS_PER_YEAR   # matches DT_SEED - was 100yr (Phase 3-scale), now the same order as this phase's own fragile-early-step scale


def main():
    # Hermetic runtime overrides - see module docstring. Composition strictly molecular/constant.
    config.MU = 2.34
    config.GAMMA = 1.4
    config.USE_H2_RECOMBINATION_PHYSICS = False
    config.T_CENTER_INITIAL = T_CENTER_INITIAL_PHASE1
    config.USE_ADAPTIVE_DT = True
    config.ADAPTIVE_DT_MAX = 1.0e4 * config.SECONDS_PER_YEAR   # same order as the expected total duration, not Phase 3's Gyr-scale ceiling
    config.ADAPTIVE_DT_MIN = ADAPTIVE_DT_MIN_PHASE1
    config.T_MAX_S = 1.0e6 * config.SECONDS_PER_YEAR           # ~10x headroom above the Larson 1969 upper bound - a meaningful diagnostic ceiling
    config.RELAX_DT_FRACTION = RELAX_DT_FRACTION_PHASE1

    print(f"run_phase1_first_core: building t=0 diffuse molecular first core "
          f"(T_CENTER_INITIAL={config.T_CENTER_INITIAL:.1f}K, MU={config.MU}, GAMMA={config.GAMMA}) ...", flush=True)
    state_0 = bvp_solver.solve_static_structure(use_ideal_gas_seed=True)

    print("run_phase1_first_core: relaxing to a genuine solution of the full 4-ODE system "
          "(relax_initial_state, clamp left at its global default - Phase 1's own trajectory "
          "needs it, unlike Phase 3's) ...", flush=True)
    state_relaxed = bvp_solver.relax_initial_state(state_0, force_clamp_off_stage1=False)

    print(f"run_phase1_first_core: starting Phase 1 quasi-static contraction, n_steps={N_STEPS_MAX}, "
          f"seed dt={DT_SEED / config.SECONDS_PER_YEAR:.4e} yr, "
          f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K, "
          f"T_MAX_S={config.T_MAX_S / config.SECONDS_PER_YEAR:.3e} yr, "
          f"snapshot_dir={SNAPSHOT_DIR}, plot_dir={PLOT_DIR}", flush=True)
    history = time_stepper.run(state_relaxed, N_STEPS_MAX, DT_SEED, snapshot_dir=SNAPSHOT_DIR)

    print("run_phase1_first_core: generating plots from the saved snapshots ...", flush=True)
    output.generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=PLOT_DIR,
                               profile_snapshot_indices=None)

    final = history[-1]
    print(f"run_phase1_first_core: done, {len(history)} snapshots saved to {SNAPSHOT_DIR}, "
          f"plots in {PLOT_DIR}. Final state: t={final.t / config.SECONDS_PER_YEAR:.4e} yr, "
          f"r_surface={final.r[-1] / config.R_JUPITER_CM:.4f} R_Jup, "
          f"T_center={final.T[0]:.6e} K", flush=True)
    return history


if __name__ == "__main__":
    main()
