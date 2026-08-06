# time_stepper.py — Outer Kelvin-Helmholtz contraction time loop (Sub-task 8): repeated
# bvp_solver.solve_timestep calls from a given starting state, down to config.R_HALT. Also
# retains compute_time_derivatives as a post-hoc finite-difference diagnostic utility (not on
# solve_timestep's own critical path - bvp_solver._implicit_rhs_logm does its own inline
# differencing).

import numpy as np

import bvp_solver
import config

# ==========================================
# SECTION: Finite-Difference Time Derivatives (diagnostic utility)
# ==========================================

def compute_time_derivatives(state_curr, state_prev, dt):
    """(dT_dt, dP_dt) on state_curr.m, finite-differenced against state_prev (interpolated
    onto state_curr.m in case the Lagrangian grid shifted between steps). Diagnostic only -
    solve_timestep computes its own inline differencing.
    """
    T_prev_interp = np.interp(state_curr.m, state_prev.m, state_prev.T)
    P_prev_interp = np.interp(state_curr.m, state_prev.m, state_prev.P)

    dT_dt = (state_curr.T - T_prev_interp) / dt
    dP_dt = (state_curr.P - P_prev_interp) / dt
    return dT_dt, dP_dt


# ==========================================
# SECTION: Outer Time Loop (Kelvin-Helmholtz Contraction)
# ==========================================

def run(state_prev, n_steps, dt, snapshot_interval=1):
    """Advance state_prev through up to n_steps of bvp_solver.solve_timestep(state, dt),
    halting early once the surface radius reaches config.R_HALT (Stage 3's cooling,
    degenerate-pressure-supported contraction toward a present-day-Jupiter-like state -
    PLAN.md "Formation Scenario and Scope"). No bootstrap/kick step of any kind - uniform
    from the first call. state_prev should already be genuinely self-consistent with
    solve_timestep's equations (bvp_solver.relax_initial_state's output, not
    solve_static_structure's directly).

    Returns the list of snapshots taken: state_prev itself, then every snapshot_interval-th
    step, always including the final step regardless of interval.
    """
    history = [state_prev]
    dt_yr = dt / config.SECONDS_PER_YEAR
    print(f"time_stepper.run: starting KH-contraction loop, n_steps={n_steps}, dt={dt_yr:.4e} yr, "
          f"R_HALT={config.R_HALT / config.R_JUPITER_CM:.3f} R_Jup")

    state = state_prev
    for step in range(1, n_steps + 1):
        state = bvp_solver.solve_timestep(state, dt)

        r_surface = state.r[-1]
        t_yr = state.t / config.SECONDS_PER_YEAR
        L_surface_lsun = state.L[-1] / config.L_SUN_ERG_S
        print(f"time_stepper.run: step {step}/{n_steps}, t={t_yr:.4e} yr, dt={dt_yr:.4e} yr, "
              f"r_surface={r_surface / config.R_JUPITER_CM:.4f} R_Jup, "
              f"T_center={state.T[0]:.6e} K, L_surface={L_surface_lsun:.4e} L_sun")

        halted = r_surface <= config.R_HALT
        if halted or step % snapshot_interval == 0:
            history.append(state)

        if halted:
            print(f"time_stepper.run: R_HALT reached (r_surface="
                  f"{r_surface / config.R_JUPITER_CM:.4f} R_Jup) at step {step}, "
                  f"t={t_yr:.4e} yr - halting")
            break

    return history
