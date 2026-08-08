# bvp_solver_shooting_archive.py — RETIRED (2026-08-08). Historical archive of the t>0
# shooting-based solve (relax_initial_state, solve_timestep), superseded by
# scipy.integrate.solve_bvp collocation (PLAN_BVP.md Milestone 6, promoted into the live
# bvp_solver.py 2026-08-08). NOT imported by any active module - kept only as a record of
# what was tried and why it was abandoned, per this project's "archive, don't delete"
# convention (see PLAN.md §4.2, PROGRESS.md).
#
# WHY THIS WAS RETIRED: shooting (scipy.integrate.solve_ivp outward integration + root-find
# on the central conditions via scipy.optimize.root(method="lm")) worked at
# T_CENTER_INITIAL=13000K but hit a sequence of non-smooth "kinks" while stabilizing further
# (a hard L>=0 floor, a hard Schwarzschild min() switch, a third unresolved wall) - all
# clustering in the same narrow near-photosphere region, the diagnostic signature of a
# single-long-integration method that cannot contain local non-smoothness. Analytic
# Jacobians (derived to rule out FD-Jacobian imprecision as the cause) reproduced the crash
# IDENTICALLY and revealed the real structural cause instead: the Jacobian is genuinely
# rank-deficient almost everywhere under the infinitely-efficient-convection idealization
# (100% convective saturation makes d(grad_eff)/d(grad_rad)=0, decoupling L from the P-T
# relation). PLAN_BVP.md has the complete milestone-by-milestone trail (Milestones 0-4).
#
# The functions below are otherwise UNMODIFIED from their last working state in bvp_solver.py
# (just before the 2026-08-08 promotion) - this file is a snapshot, not a rewrite. They
# depend on _photosphere_event_adiabatic-style machinery still present in the live
# bvp_solver.py (_build_output_grid) - imported from there for exactly that reason, since
# that adiabatic/t=0 machinery was NOT retired (see bvp_solver.py's own module docstring).

import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

import boundary_conditions
import bvp_solver
import config
import eos
import odes
import state

# ==========================================
# SECTION: Photosphere Event (t>0, implicit 4-ODE system)
# ==========================================

def _photosphere_event_implicit(x, y):
    r, lnP, L, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    return P - boundary_conditions.photospheric_pressure(r, P, T, config.MU, config.MU_E)
_photosphere_event_implicit.terminal = True
_photosphere_event_implicit.direction = -1


# ==========================================
# SECTION: Implicit Per-Timestep Right-Hand Side (shooting, single-point contract)
# ==========================================

def _implicit_rhs_logm(x, y, state_prev, dt, alpha=1.0):
    """dy/dx (x=ln(m)) for the full 4-ODE system. dT_dt, dP_dt are computed from the current
    trial (T, P) differenced against state_prev's (interpolated) profile - the implicit
    (Henyey-style) form: the energy equation's source term is exactly this state difference.

    alpha blends nabla_eff between the pure adiabat (alpha=0, matches solve_static_structure's
    own construction) and the real Schwarzschild-selected value (alpha=1, the default - every
    genuine solve_timestep() call uses this; only relax_initial_state()'s pseudo-steps use
    alpha<1). At alpha=0 this reduces to dlnT_dm = grad_ad*dlnP_dm exactly. dL/dm is NEVER
    scaled by alpha - dL/dm has no other source in this codebase, so scaling it would force
    dL/dm=0 identically at alpha=0.

    State is (r, lnP, L, lnT); P=exp(lnP)>0, T=exp(lnT)>0 hold by construction (module
    docstring) - odes.stellar_odes and eos.density still receive/return linear, physical (P, T).
    """
    m = np.exp(x)
    r, lnP, L, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    T_prev = np.interp(m, state_prev.m, state_prev.T)
    P_prev = np.interp(m, state_prev.m, state_prev.P)
    dT_dt = (T - T_prev) / dt
    dP_dt = (P - P_prev) / dt
    y_full = np.array([[r], [P], [L], [T]])
    dr_dm, dP_dm, dL_dm, dT_dm_real = odes.stellar_odes(
        np.array([m]), y_full, np.array([dT_dt]), np.array([dP_dt])
    )
    dlnP_dm = dP_dm[0] / P
    if alpha == 1.0:
        dlnT_dm = dT_dm_real[0] / T
    else:
        # grad_ad=dlnT/dlnP by definition (eos.grad_adiabatic, single source of truth); dlnP_dm
        # is shared (hydrostatic equilibrium doesn't depend on the temperature gradient), so
        # only the temperature-gradient term needs recombining.
        dlnT_dm_adiabatic = eos.grad_adiabatic(config.GAMMA) * dlnP_dm
        dlnT_dm = (1.0 - alpha) * dlnT_dm_adiabatic + alpha * (dT_dm_real[0] / T)
    return [dr_dm[0] * m, dlnP_dm * m, dL_dm[0] * m, dlnT_dm * m]


def _integrate_timestep_outward(P_center, T_center, x_span, r_start, state_prev, dt, alpha=1.0):
    """Integrate the full 4-ODE system outward for a trial (P_center, T_center), terminating
    at the photosphere event; see _implicit_rhs_logm. L starts at exactly 0 (center BC).

    atol=config.BVP_TOL on the log components gives uniform relative precision on P, T across
    their many-decade range (see bvp_solver._integrate_adiabatic_outward).
    """
    def rhs(x, y):
        return _implicit_rhs_logm(x, y, state_prev, dt, alpha)
    return solve_ivp(
        rhs, x_span, [r_start, np.log(P_center), 0.0, np.log(T_center)], method="Radau",
        dense_output=True, events=_photosphere_event_implicit,
        rtol=config.BVP_TOL, atol=[1.0, config.BVP_TOL, 1.0, config.BVP_TOL],
    )


# ==========================================
# SECTION: Initial-Model Relaxation (bridges solve_static_structure to solve_timestep)
# ==========================================

def relax_initial_state(state_0) -> state.SimulationState:
    """Relax state_0 (solve_static_structure's output - built by forcing the pure ideal-gas
    adiabat, not a genuine solution of the real 4-ODE system's Schwarzschild-selected
    temperature gradient) into a state that IS self-consistent with the same implicit
    equations solve_timestep() uses, via continuation in alpha (_implicit_rhs_logm's
    nabla_eff blend fraction).

    T_CENTER_INITIAL is a prescribed hand-off value, not derived from a previous state, so
    state_0 has no reason to already be a root of solve_timestep's real, time-differenced
    equations. Standard "initial model relaxation" practice (MESA-style pre-main-sequence
    relaxation; classical Henyey-code ZAMS construction): walk a continuous path of
    increasingly real problems from alpha=0 (reproduces state_0's own construction exactly) to
    alpha=1 (the real target), using each step's converged (P_center, T_center) to warm-start
    the next.

    Convergence criteria: each pseudo-step's root-find (scipy.optimize.root, method="lm" -
    see ASSUMPTION below for why not fsolve's default hybrd) must report success AND an
    independently-verified residual below config.RESIDUAL_TOL, or the step is rejected and
    retried at a smaller alpha step (see ADAPTIVE STEPPING below) - a failed step no longer
    raises immediately unless even the minimum step size fails. The achieved [mass, thermal]
    residual is printed for every attempt (successful or not) for a visible audit trail; a
    >50% (P_center, T_center) jump between consecutive steps is flagged as a possible
    solution-branch jump.

    ADAPTIVE STEPPING (2026-08-01, PROGRESS.md has the full motivating trail): replaces the
    original fixed 11-step grid. Two independent root-find algorithms (hybrd and LM)
    converged to the identical non-zero residual attempting the alpha=0.0->0.1 jump directly
    - evidence that a fixed step of 0.1 is genuinely too large for a local Newton/Gauss-
    Newton-type step to bridge at this T_CENTER_INITIAL, not a solver-choice problem.
    Standard numerical-continuation practice (parameter-continuation software; MESA/Henyey-
    code relaxation) for a homotopy whose local difficulty varies along the path: try a
    target step, halve and retry from the last genuinely-converged state on failure, grow
    back toward the target after success so easy stretches aren't crawled through
    unnecessarily. A minimum step floor raises loudly rather than halving forever.
    """
    alpha_step_target = 0.1   # nominal/target step - also the ceiling step recovery grows back toward
    alpha_step_min = 1.0e-6   # safety floor - below this, raise rather than halve indefinitely
    dt_relax = 0.01 * config.T_KH_TIMESCALE_S   # pseudo-timestep; not real elapsed time (t is left unchanged below)

    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    x_span = (np.log(m_min), np.log(50.0 * config.M_TOTAL))
    r_start = state_0.r[0]
    L_scale = config.G * config.M_TOTAL**2 / (state_0.r[-1] * config.T_KH_TIMESCALE_S)

    def residual(u, alpha):
        P_center, T_center = np.exp(u)
        sol = _integrate_timestep_outward(P_center, T_center, x_span, r_start, state_0, dt_relax, alpha)
        if len(sol.t_events[0]) == 0:
            raise RuntimeError(
                f"relax_initial_state: alpha={alpha:.3f} did not reach the photosphere at "
                f"P_center={P_center:.6e}, T_center={T_center:.6e}"
            )
        m_event = np.exp(sol.t_events[0][0])
        yb = sol.y_events[0][0]   # (r, lnP, L, lnT) at the photosphere event
        yb_physical = np.array([yb[0], np.exp(yb[1]), yb[2], np.exp(yb[3])])
        res_full = boundary_conditions.boundary_conditions(np.zeros(4), yb_physical)
        mass_residual = (m_event - config.M_TOTAL) / config.M_TOTAL
        return [mass_residual, res_full[3] / L_scale]

    u = np.array([np.log(state_0.P[0]), np.log(state_0.T[0])]) * (1.0 + 1.0e-6)

    alpha = 0.0
    step = alpha_step_target
    n_attempts = 0
    max_attempts = 1000
    while alpha < 1.0:
        n_attempts += 1
        if n_attempts > max_attempts:
            raise RuntimeError(
                f"relax_initial_state: exceeded {max_attempts} adaptive-step attempts without "
                f"reaching alpha=1.0 (stuck at alpha={alpha:.6f}) - should be structurally "
                f"impossible given the step floor; investigate rather than raise this cap"
            )

        alpha_trial = min(alpha + step, 1.0)
        u_prev = u.copy()
        try:
            opt_result = root(lambda u_trial: residual(u_trial, alpha_trial), u, method="lm",
                               options={"xtol": config.BVP_TOL, "ftol": config.RESIDUAL_TOL})
            res_mass, res_L = opt_result.fun
            max_residual = max(abs(res_mass), abs(res_L))
            step_ok = opt_result.success and max_residual <= config.RESIDUAL_TOL
        except RuntimeError as e:
            step_ok = False
            res_mass = res_L = None
            print(f"bvp_solver_shooting_archive: relaxation step alpha={alpha:.4f}->{alpha_trial:.4f} "
                  f"(step={step:.2e}) raised during root-find: {e}")

        if step_ok:
            u = opt_result.x
            jump = np.max(np.abs(u - u_prev))
            if alpha_trial > 0.0 and jump > np.log(1.5):
                warnings.warn(
                    f"relax_initial_state: (P_center, T_center) jumped >50% between alpha "
                    f"steps (alpha={alpha_trial:.4f}) - possible solution-branch jump, not "
                    f"just smooth continuation; inspect before trusting this relaxation run",
                    RuntimeWarning
                )
            print(f"bvp_solver_shooting_archive: relaxation pseudo-step alpha={alpha_trial:.4f} converged "
                  f"(step={step:.4f}), P_center={np.exp(u[0]):.6e}, T_center={np.exp(u[1]):.6e} K, "
                  f"residuals=[{res_mass:.3e}, {res_L:.3e}]")
            alpha = alpha_trial
            step = min(step * 2.0, alpha_step_target)
        else:
            step /= 2.0
            if step < alpha_step_min:
                raise RuntimeError(
                    f"relax_initial_state: adaptive alpha-step fell below the minimum floor "
                    f"({alpha_step_min:.1e}) trying to advance past alpha={alpha:.6f} - could "
                    f"not find a convergent step even at minimum resolution; the true root may "
                    f"be genuinely unreachable via local continuation from this point"
                )
            if res_mass is not None:
                print(f"bvp_solver_shooting_archive: relaxation step alpha={alpha:.4f}->{alpha_trial:.4f} FAILED "
                      f"(residual [{res_mass:.3e}, {res_L:.3e}] exceeds "
                      f"config.RESIDUAL_TOL={config.RESIDUAL_TOL:.1e}), halving step to {step:.2e}")
            else:
                print(f"bvp_solver_shooting_archive: relaxation step alpha={alpha:.4f}->{alpha_trial:.4f} FAILED "
                      f"(see error above), halving step to {step:.2e}")

    P_center, T_center = np.exp(u)
    sol = _integrate_timestep_outward(P_center, T_center, x_span, r_start, state_0, dt_relax, 1.0)
    m_surface = np.exp(sol.t_events[0][0])

    m = bvp_solver._build_output_grid(m_min, m_surface)
    r, lnP, L, lnT = sol.sol(np.log(m))
    P, T = np.exp(lnP), np.exp(lnT)
    rho = eos.density(P, T, config.MU, config.MU_E)

    print(f"bvp_solver_shooting_archive: initial-state relaxation complete (alpha=1.0, genuine solution of the "
          f"real 4-ODE system), T_center={T_center:.6e} K, r_surface={r[-1]/config.R_JUPITER_CM:.3f} R_Jup, "
          f"m_surface/M_TOTAL={m_surface/config.M_TOTAL:.8f}")

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=state_0.t, prev=state_0)


def solve_timestep(state_prev, dt) -> state.SimulationState:
    """Solve the envelope structure at t = state_prev.t + dt via the implicit shooting scheme.
    Shoots on (ln P_center, ln T_center) to match two conditions at the photosphere event:
    enclosed mass = M_TOTAL (mechanical) and the net radiative flux balance (thermal,
    boundary_conditions.py). The center residuals (r_a=0, L_a=0) are satisfied exactly by
    construction (integration starts at r=r_start, L=0).
    """
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    x_span = (np.log(m_min), np.log(50.0 * config.M_TOTAL))
    r_start = state_prev.r[0]

    u0 = np.array([np.log(state_prev.P[0]), np.log(state_prev.T[0])]) * (1.0 + 1.0e-6)

    L_scale = config.G * config.M_TOTAL**2 / (state_prev.r[-1] * config.T_KH_TIMESCALE_S)

    def residual(u):
        P_center, T_center = np.exp(u)
        sol = _integrate_timestep_outward(P_center, T_center, x_span, r_start, state_prev, dt)
        if len(sol.t_events[0]) == 0:
            raise RuntimeError(
                f"solve_ivp did not reach the photosphere during timestep shooting at "
                f"P_center={P_center:.6e}, T_center={T_center:.6e}"
            )
        m_event = np.exp(sol.t_events[0][0])
        yb = sol.y_events[0][0]   # (r, lnP, L, lnT) at the photosphere event
        yb_physical = np.array([yb[0], np.exp(yb[1]), yb[2], np.exp(yb[3])])
        res_full = boundary_conditions.boundary_conditions(np.zeros(4), yb_physical)
        mass_residual = (m_event - config.M_TOTAL) / config.M_TOTAL
        return [mass_residual, res_full[3] / L_scale]

    opt_result = root(residual, u0, method="lm", options={"xtol": config.BVP_TOL, "ftol": config.RESIDUAL_TOL})
    if not opt_result.success:
        warnings.warn(f"solve_timestep: LM did not fully converge: {opt_result.message}", RuntimeWarning)

    u_sol = opt_result.x
    res_mass, res_L = opt_result.fun
    max_residual = max(abs(res_mass), abs(res_L))
    if max_residual > config.RESIDUAL_TOL:
        raise RuntimeError(
            f"solve_timestep: LM reports success={opt_result.success} but residual "
            f"[{res_mass:.3e}, {res_L:.3e}] exceeds config.RESIDUAL_TOL={config.RESIDUAL_TOL:.1e} "
            f"- not a genuine root"
        )

    P_center, T_center = np.exp(u_sol)
    sol = _integrate_timestep_outward(P_center, T_center, x_span, r_start, state_prev, dt)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("solve_timestep: converged (P_center, T_center) does not reach the photosphere - numerical precision limit near the root")
    m_surface = np.exp(sol.t_events[0][0])

    m = bvp_solver._build_output_grid(m_min, m_surface)
    r, lnP, L, lnT = sol.sol(np.log(m))
    P, T = np.exp(lnP), np.exp(lnT)
    rho = eos.density(P, T, config.MU, config.MU_E)

    print(f"bvp_solver_shooting_archive: timestep converged, t={state_prev.t + dt:.4e} s, P_center={P_center:.6e}, "
          f"T_center={T_center:.6e} K, m_surface/M_TOTAL={m_surface/config.M_TOTAL:.8f}, "
          f"residuals=[{res_mass:.3e}, {res_L:.3e}]")

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=state_prev.t + dt, prev=state_prev)
