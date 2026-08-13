# time_stepper.py — Outer Kelvin-Helmholtz contraction time loop (Sub-task 8): repeated
# bvp_solver.solve_timestep calls from a given starting state, down to config.R_HALT or
# config.T_MAX_S (Sub-task 10's dual stopping condition), whichever comes first.
# Also retains compute_time_derivatives as a post-hoc finite-difference diagnostic utility
# (not on solve_timestep's own critical path - bvp_solver._implicit_rhs_logm does its own
# inline differencing), now also select_adaptive_dt's own input (Sub-task 9).

import numpy as np

import bvp_solver
import config
import output

# ==========================================
# SECTION: Finite-Difference Time Derivatives (diagnostic utility)
# ==========================================

def compute_time_derivatives(state_curr, state_prev, dt):
    """(dT_dt, dP_dt) on state_curr.m, finite-differenced against state_prev (interpolated
    onto state_curr.m in case the Lagrangian grid shifted between steps). Previously
    diagnostic-only; now also select_adaptive_dt's own input (Sub-task 9) - solve_timestep
    itself still computes its own inline differencing, unaffected.
    """
    T_prev_interp = np.interp(state_curr.m, state_prev.m, state_prev.T)
    P_prev_interp = np.interp(state_curr.m, state_prev.m, state_prev.P)

    dT_dt = (state_curr.T - T_prev_interp) / dt
    dP_dt = (state_curr.P - P_prev_interp) / dt
    return dT_dt, dP_dt


# ==========================================
# SECTION: Adaptive Time-Stepping (Sub-task 9)
# ==========================================

def select_adaptive_dt(state_curr, state_prev, dt_used):
    """Thermal/pressure-timescale limiter (PLAN.md Sub-task 9, PROGRESS.md has the full
    design discussion):
        dt_raw = ADAPTIVE_DT_SAFETY_FACTOR * min(min_i(T_i/|dT_i/dt|), min_i(P_i/|dP_i/dt|))
    then capped by a per-step growth factor (dt_new <= ADAPTIVE_DT_GROWTH_FACTOR*dt_used -
    growth only; shrinking is never restricted, since a sharp drop in the raw formula is the
    safety mechanism working as intended) and clamped to [ADAPTIVE_DT_MIN, ADAPTIVE_DT_MAX].

    DELIBERATELY dual (T and P), not T alone: P has been measured swinging by ~3 decades over
    a tiny mass range near the photosphere all session - a T-only limiter could stay blind to
    a fast-evolving P profile there.

    DELIBERATELY excludes L: L=0 EXACTLY at the center by construction (the boundary
    condition) and dL/dt there is also ~0 - a literal 0/0 at m=m_min on every step, not an
    edge case; near the photosphere L has been observed crossing zero as normal, expected
    behavior, not a danger signal. T and P already carry the physical signal that matters,
    and neither has L's structural zero (T, P > 0 everywhere by construction), so no
    equivalent 0/0 risk exists for them.

    dT_dt, dP_dt are the REALIZED rates from the just-completed step (state_curr vs
    state_prev, at dt_used) via compute_time_derivatives - a lagged estimate used to select
    the NEXT step's dt, not a predictor-corrector.
    """
    dT_dt, dP_dt = compute_time_derivatives(state_curr, state_prev, dt_used)

    # Points where the rate is exactly zero give T_i/0 = +inf (not NaN - T_i, P_i > 0 always,
    # no structural 0/0 the way L has at the center) - naturally excluded by the min() below,
    # no special-casing needed. Suppress the resulting numpy divide-by-zero warning
    # explicitly rather than letting it print for an expected, harmless case.
    with np.errstate(divide="ignore"):
        T_timescale = state_curr.T / np.abs(dT_dt)
        P_timescale = state_curr.P / np.abs(dP_dt)

    dt_raw = config.ADAPTIVE_DT_SAFETY_FACTOR * min(T_timescale.min(), P_timescale.min())
    dt_growth_capped = min(dt_raw, config.ADAPTIVE_DT_GROWTH_FACTOR * dt_used)
    dt_new = np.clip(dt_growth_capped, config.ADAPTIVE_DT_MIN, config.ADAPTIVE_DT_MAX)
    return float(dt_new)


# ==========================================
# SECTION: Outer Time Loop (Kelvin-Helmholtz Contraction)
# ==========================================

def run(state_prev, n_steps, dt, snapshot_interval=1, snapshot_dir=None):
    """Advance state_prev through up to n_steps of bvp_solver.solve_timestep(state, dt),
    halting early on whichever of THREE physically-motivated conditions triggers first: the
    surface radius reaching config.R_HALT (Stage 3's cooling, degenerate-pressure-supported
    contraction toward a present-day-Jupiter-like state - PLAN.md "Formation Scenario and
    Scope"), the elapsed simulated time reaching config.T_MAX_S (a diagnostic time budget, not
    a claim about the real planet's age - config.py has the full reasoning - a backstop against
    an indefinitely long run if R_HALT is never reached), or (2026-08-12, Phase 1 / First
    Hydrostatic Core pivot) the central temperature reaching config.PHASE1_T_CENTER_HALT - a
    deliberate stop just below where H2 dissociation would soften Gamma_1 below 4/3 and trigger
    the out-of-scope Stage 2 dynamical collapse, before that physical singularity reaches this
    quasi-static solver. No bootstrap/kick step of any kind - uniform from
    the first call. state_prev should already be genuinely self-consistent with
    solve_timestep's equations (bvp_solver.relax_initial_state's output, not
    solve_static_structure's directly).

    dt is always the SEED timestep for step 1. If config.USE_ADAPTIVE_DT is False (default),
    it is also used for every subsequent step, unchanged from before Sub-task 9. If True,
    every step after the first instead uses select_adaptive_dt's thermal/pressure-timescale
    selection, lagged from the JUST-COMPLETED step's realized dT/dt, dP/dt - no prior real-dt
    derivative exists before step 1, hence the fixed seed there regardless of the flag.

    snapshot_dir (Sub-task 10): if given, every snapshot taken is ALSO saved to disk as an
    .npz file (output.save_snapshot) as the run proceeds, not just held in memory - lets a
    long run's progress survive an interruption and feeds output.py's post-processing plots,
    which are built entirely from these files (not from this function's return value).

    Returns the list of snapshots taken: state_prev itself, then every snapshot_interval-th
    step, always including the final step regardless of interval.
    """
    history = [state_prev]
    mode = "ADAPTIVE (Sub-task 9)" if config.USE_ADAPTIVE_DT else "FIXED"
    print(f"time_stepper.run: starting KH-contraction loop, n_steps={n_steps}, dt_mode={mode}, "
          f"seed dt={dt / config.SECONDS_PER_YEAR:.4e} yr, "
          f"R_HALT={config.R_HALT / config.R_JUPITER_CM:.3f} R_Jup, "
          f"t_max={config.T_MAX_S / config.SECONDS_PER_YEAR:.3e} yr, "
          f"PHASE1_T_CENTER_HALT={config.PHASE1_T_CENTER_HALT:.1f} K", flush=True)

    if snapshot_dir is not None:
        output.save_snapshot(state_prev, 0, snapshot_dir)

    state = state_prev
    dt_used = dt
    for step in range(1, n_steps + 1):
        state_before_step = state

        # Step-retry (2026-08-12, Phase 1 pivot - PROGRESS.md has the full report): a failed
        # solve_timestep is retried with a shrunken dt (config.STEP_RETRY_SHRINK_FACTOR) up to
        # config.STEP_RETRY_MAX_ATTEMPTS times before giving up - standard adaptive-integrator
        # step-rejection practice. dt_used is updated to whatever dt actually succeeded, so the
        # NEXT step's growth-capped selection grows from the successful value, not the
        # originally-proposed (failed) one.
        dt_attempt = dt_used
        for retry in range(config.STEP_RETRY_MAX_ATTEMPTS + 1):
            try:
                state = bvp_solver.solve_timestep(state_before_step, dt_attempt)
                dt_used = dt_attempt
                break
            except RuntimeError as exc:
                if retry == config.STEP_RETRY_MAX_ATTEMPTS:
                    raise
                dt_attempt *= config.STEP_RETRY_SHRINK_FACTOR
                print(f"time_stepper.run: step {step} failed at dt={dt_attempt / config.STEP_RETRY_SHRINK_FACTOR / config.SECONDS_PER_YEAR:.4e} yr "
                      f"({exc}) - retrying at dt={dt_attempt / config.SECONDS_PER_YEAR:.4e} yr "
                      f"(attempt {retry + 1}/{config.STEP_RETRY_MAX_ATTEMPTS}) ...", flush=True)

        r_surface = state.r[-1]
        T_center = state.T[0]
        t_yr = state.t / config.SECONDS_PER_YEAR
        dt_used_yr = dt_used / config.SECONDS_PER_YEAR
        L_surface_lsun = state.L[-1] / config.L_SUN_ERG_S

        # Defensive: catch a corrupted state loudly and immediately, rather than letting a
        # NaN or non-physical value silently propagate into further steps or surface as a
        # confusing downstream failure much later - explicit request, so the terminal makes
        # it obvious the run isn't stuck or quietly producing garbage.
        if not (np.isfinite(r_surface) and np.isfinite(T_center) and r_surface > 0.0 and T_center > 0.0):
            raise RuntimeError(
                f"time_stepper.run: state corrupted at step {step} (r_surface={r_surface}, "
                f"T_center={T_center}) - halting immediately rather than continuing on bad data"
            )

        print(f"time_stepper.run: step {step}/{n_steps}, t={t_yr:.4e} yr, dt={dt_used_yr:.4e} yr, "
              f"r_surface={r_surface / config.R_JUPITER_CM:.4f} R_Jup, "
              f"T_center={T_center:.6e} K, L_surface={L_surface_lsun:.4e} L_sun", flush=True)

        halted_radius = r_surface <= config.R_HALT
        halted_time = state.t >= config.T_MAX_S
        halted_T_center = T_center >= config.PHASE1_T_CENTER_HALT
        halted = halted_radius or halted_time or halted_T_center

        if halted or step % snapshot_interval == 0:
            history.append(state)
            if snapshot_dir is not None:
                output.save_snapshot(state, step, snapshot_dir)

        if halted:
            reason = ("R_HALT reached" if halted_radius
                      else "t_max (diagnostic time budget) reached" if halted_time
                      else "PHASE1_T_CENTER_HALT reached (H2-dissociation onset)")
            print(f"time_stepper.run: {reason} (r_surface="
                  f"{r_surface / config.R_JUPITER_CM:.4f} R_Jup, T_center={T_center:.4e} K, "
                  f"t={t_yr:.4e} yr) at step {step} - halting", flush=True)
            break

        if config.USE_ADAPTIVE_DT:
            dt_next = select_adaptive_dt(state, state_before_step, dt_used)
            print(f"time_stepper.run: adaptive dt selected for next step: "
                  f"{dt_next / config.SECONDS_PER_YEAR:.4e} yr (was {dt_used_yr:.4e} yr)", flush=True)
            dt_used = dt_next

    return history
