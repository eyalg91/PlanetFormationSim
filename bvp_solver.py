# bvp_solver.py — Solves the t=0 compact-protoplanet structure and the per-timestep implicit
# structure for t>0, returning a populated SimulationState in each case.
#
# PHYSICAL PICTURE: t=0 is a hot, compact, fully convective post-collapse protoplanet with a
# PRESCRIBED central temperature (config.T_CENTER_INITIAL - a chosen "hot start" parameter,
# not derived; standard practice for gas-giant "hot start" models, e.g. Marley et al. 2007).
# It is not static (dT_dt=dP_dt=0): it is already contracting at its natural Kelvin-Helmholtz
# rate (T/t_KH, 4P/t_KH, config.T_KH_TIMESCALE_S), evaluated on the current trial profile.
# Compactness (a few R_Jup, not a few hundred) requires non-relativistic electron-degeneracy
# pressure (eos.py), included additively with the ideal-gas term.
#
# OUTER BOUNDARY: the photosphere (Eddington tau=2/3, boundary_conditions.py), not a fixed
# ambient pressure P_neb - a degenerate-supported structure's atmosphere hands off to a
# photosphere at a pressure set by its own surface gravity and opacity, not by the ambient
# nebula. Both shooting routines integrate outward with the photosphere as a solve_ivp EVENT
# and match the ENCLOSED MASS at that event to M_TOTAL, rather than checking a residual at a
# fixed m=M_TOTAL grid endpoint.
#
# SOLVER: shooting (scipy.integrate.solve_ivp, Radau, outward integration + root-finding on
# the central conditions), not scipy.integrate.solve_bvp - the surface pressure boundary
# layer and the wide dynamic range of P, T across the mass grid defeat its collocation
# Jacobian.
#
# STATE REPRESENTATION: both RHS functions below integrate ln P, ln T rather than P, T
# directly, so P=exp(lnP)>0, T=exp(lnT)>0 hold by construction through the stiff solver's
# internal probing - the standard Henyey/MESA-style representation. See PROGRESS.md for the
# full physical/numerical history behind these design choices (photospheric BC, log state
# variables, the L>=0 floor in gradients.grad_radiative).

import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

import boundary_conditions
import config
import eos
import gradients
import odes
import opacity
import state

# ==========================================
# SECTION: Lane-Emden Analytic Bracket Seed
# ==========================================

def _solve_lane_emden(n):
    """Solve the Lane-Emden equation (1/xi^2) d/dxi(xi^2 dtheta/dxi) = -theta^n for polytropic
    index n, stopping at the first surface crossing (theta=0). Returns (xi_1, mass_constant) =
    (dimensionless surface radius, -xi_1^2*theta'(xi_1)) - the two dimensionless constants
    needed to scale a physical (M, T_center) pair to (R, P_center, rho_center) for an ideal-gas,
    fully convective sphere (Kippenhahn & Weigert; Chandrasekhar 1939). Verified against
    tabulated values for n=1.5 (xi_1=3.65375) and n=3.0 (xi_1=6.89685).
    """
    def rhs(xi, y):
        theta, dtheta = y
        theta_pos = max(theta, 0.0)
        # Series solution near xi=0 (theta''(0) = -1/3 for any n) avoids the removable 0/0 in
        # the -(2/xi)*dtheta term at the true center.
        d2theta = -1.0 / 3.0 if xi < 1.0e-8 else -theta_pos**n - (2.0 / xi) * dtheta
        return [dtheta, d2theta]

    def surface_event(xi, y):
        return y[0]
    surface_event.terminal = True
    surface_event.direction = -1

    sol = solve_ivp(rhs, [1.0e-6, 100.0], [1.0, 0.0], events=surface_event, rtol=1.0e-11, atol=1.0e-13, max_step=1.0e-3)
    xi_1 = sol.t_events[0][0]
    _theta_end, dtheta_end = sol.y[:, -1]
    mass_constant = -xi_1**2 * dtheta_end
    return xi_1, mass_constant


def _adiabatic_center_guess():
    """Analytic (Lane-Emden) central pressure and radius bracket SEED for the real shooting
    below - not the final answer (the real shooting includes both the ideal-gas and degenerate
    pressure terms and locates the exact photospheric surface, not this idealized polytrope's
    own P->0 natural surface).

    ASSUMPTION: uses the pure T=0 electron-degeneracy limit (n=3/2 polytrope, P=K1*rho^(5/3) -
    Zapolsky & Salpeter 1969 style), since degeneracy dominates the interior by 2-3 orders of
    magnitude in pressure at the density this seed predicts. There is no closed-form Lane-Emden
    solution for the real, additive combined EOS, so this remains an approximate seed - it pins
    the right order of magnitude to bracket reliably via geometric expansion (see
    solve_static_structure).
    """
    n = 1.5   # non-relativistic electron degeneracy: P=K1*rho^(5/3), gamma=5/3, n=1/(gamma-1)
    xi_1, mass_constant = _solve_lane_emden(n)

    # Polytrope scaling (Kippenhahn & Weigert Ch. 19): alpha^2 = (n+1)*K*rho_c^(1/n-1)/(4*pi*G),
    # R = alpha*xi_1, M = 4*pi*rho_c*alpha^3*mass_constant. K here is the degenerate constant
    # K1 (eos.degenerate_pressure), fixed by fundamental constants alone (unlike the ideal-gas
    # K, it does NOT depend on T_center - the T=0 degenerate limit is temperature-independent
    # by construction). 1/n-1=-1/3 for n=1.5, collapsing alpha^2 to B/rho_c with B independent
    # of rho_c, letting M(rho_c) be inverted in closed form for the rho_c (hence P_center) that
    # gives exactly M_TOTAL.
    K1 = (config.PLANCK_H**2 / (20.0 * config.M_E)) * (3.0 / np.pi) ** (2.0 / 3.0) * (1.0 / (config.MU_E * config.M_H)) ** (5.0 / 3.0)
    B = (n + 1.0) * K1 / (4.0 * np.pi * config.G)
    rho_center = (config.M_TOTAL / (4.0 * np.pi * mass_constant * B**1.5)) ** 2
    P_center = K1 * rho_center ** (5.0 / 3.0)
    R = np.sqrt(B * rho_center ** (-1.0 / 3.0)) * xi_1
    return P_center, R


# ==========================================
# SECTION: Photosphere Event (tau=2/3 surface location)
# ==========================================
# Both shooting routines below terminate their outward integration at the photosphere
# (boundary_conditions.photospheric_pressure) via a solve_ivp EVENT, not a residual at a fixed
# m=M_TOTAL grid endpoint - see that module's docstring for the physical reasoning. The two
# event functions differ only in how they unpack the RHS's state vector.

def _photosphere_event_adiabatic(x, y):
    r, lnP, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    return P - boundary_conditions.photospheric_pressure(r, P, T, config.MU, config.MU_E)
_photosphere_event_adiabatic.terminal = True
_photosphere_event_adiabatic.direction = -1


def _photosphere_event_implicit(x, y):
    r, lnP, L, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    return P - boundary_conditions.photospheric_pressure(r, P, T, config.MU, config.MU_E)
_photosphere_event_implicit.terminal = True
_photosphere_event_implicit.direction = -1


# ==========================================
# SECTION: Adiabatic (Fully Convective) Right-Hand Side
# ==========================================

def _adiabatic_rhs_logm(x, y):
    """dr/dx, dlnP/dx, dlnT/dx (x=ln(m)) for a fully convective sphere.

    ASSUMPTION: dlnT/dm = grad_ad*dlnP/dm identically - a fresh, high-entropy post-collapse
    object is assumed fully convective throughout (standard Hayashi-track picture), not
    derived via the Schwarzschild criterion. Not routed through odes.stellar_odes, which needs
    a self-consistent L unavailable for this purely-convective construction.

    ASSUMPTION: the thermal profile (T vs P) follows the pure ideal-gas adiabat; only the
    mechanical structure (rho, hence dr/dm) uses the combined ideal+degenerate EOS
    (eos.density) - a first-order approximation, not fully self-consistent thermodynamically.

    State is (r, lnP, lnT), not (r, P, T), so P=exp(lnP)>0, T=exp(lnT)>0 hold by construction
    (module docstring). eos.density still receives/returns linear, physical (P, T).
    """
    m = np.exp(x)
    r, lnP, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    rho = eos.density(P, T, config.MU, config.MU_E)
    dr_dm = 1.0 / (4.0 * np.pi * r**2 * rho)
    dP_dm = -config.G * m / (4.0 * np.pi * r**4)
    dlnP_dm = dP_dm / P
    # grad_ad is DEFINED as dlnT/dlnP (eos.grad_adiabatic), so dlnT/dm = grad_ad*dlnP/dm
    # directly - simpler than converting through the linear dT_dm = (T/P)*grad_ad*dP_dm form.
    dlnT_dm = eos.grad_adiabatic(config.GAMMA) * dlnP_dm
    return [dr_dm * m, dlnP_dm * m, dlnT_dm * m]


def _integrate_adiabatic_outward(P_center, x_span, r_start):
    """Integrate (r, lnP, lnT) outward from x_span[0] (r=r_start, T=config.T_CENTER_INITIAL)
    for a trial P_center, terminating at the photosphere event.

    atol=config.BVP_TOL on the log components is approximately a RELATIVE tolerance on the
    physical P, T (d(lnP)=dP/P), giving uniform relative precision across their many-decade
    range - reuses the single tolerance constant already in config.py.
    """
    return solve_ivp(
        _adiabatic_rhs_logm, x_span,
        [r_start, np.log(P_center), np.log(config.T_CENTER_INITIAL)], method="Radau",
        dense_output=True, events=_photosphere_event_adiabatic,
        rtol=config.BVP_TOL, atol=[1.0, config.BVP_TOL, config.BVP_TOL],
    )


# ==========================================
# SECTION: Compact Hot-Start Structure via Shooting (t=0)
# ==========================================

def solve_static_structure() -> state.SimulationState:
    """Solve the t=0 compact-protoplanet structure and return a populated SimulationState.

    Shoots on the central pressure P_center (T_center fixed at config.T_CENTER_INITIAL - a
    prescribed "hot start" parameter, not a shooting unknown, matching how real hot-start models
    treat initial entropy as a chosen input) until the ENCLOSED MASS at the photosphere event
    (boundary_conditions.photospheric_pressure, Eddington tau=2/3) equals M_TOTAL - not a
    residual at a fixed m=M_TOTAL grid endpoint (module docstring explains why), bracketed by
    the analytic Lane-Emden estimate (_adiabatic_center_guess) rather than a blind search.
    """
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    # Generous margin past M_TOTAL for the photosphere event to trigger within - empirically the
    # event triggers within ~1.4x of M_TOTAL even for a substantially too-high P_center, so this
    # is cheap headroom, never expected to bind.
    x_span = (np.log(m_min), np.log(50.0 * config.M_TOTAL))

    P_center_guess, R_guess = _adiabatic_center_guess()
    r_start = R_guess * (m_min / config.M_TOTAL) ** (1.0 / 3.0)

    def mass_error(P_center):
        sol = _integrate_adiabatic_outward(P_center, x_span, r_start)
        if len(sol.t_events[0]) == 0:
            # No photosphere reached within x_span - too little central pressure/density to
            # ever build up enough mass before the adiabat crashes toward P->0, T->0. Treated
            # as an undershoot (needs more mass budget), the signal brentq needs even though
            # no valid event exists here.
            return -config.M_TOTAL
        return np.exp(sol.t_events[0][0]) - config.M_TOTAL

    e_seed = mass_error(P_center_guess)
    if e_seed == 0.0:
        P_low = P_high = P_center_guess
    else:
        # ASSUMPTION: which direction (increasing or decreasing P_center) reduces the
        # mass-matching residual is not fixed - it depends on whether the ideal-gas or
        # degenerate term dominates (the mass-radius relation inverts, R~M^-1/3, for degenerate
        # objects). Expand in both directions simultaneously, taking whichever finds a sign
        # change first, rather than assuming a fixed direction.
        P_up = P_down = P_center_guess
        P_low = P_high = None
        for _ in range(200):
            P_up *= 1.01
            e_up = mass_error(P_up)
            if (e_up < 0.0) != (e_seed < 0.0):
                P_low, P_high = (P_center_guess, P_up) if e_seed > 0.0 else (P_up, P_center_guess)
                break
            P_down /= 1.01
            e_down = mass_error(P_down)
            if (e_down < 0.0) != (e_seed < 0.0):
                P_low, P_high = (P_down, P_center_guess) if e_seed > 0.0 else (P_center_guess, P_down)
                break
        else:
            raise RuntimeError("solve_static_structure: could not bracket the photosphere mass-matching root by geometric expansion in either direction")

    P_center = brentq(mass_error, P_low, P_high, xtol=config.M_TOTAL * 1.0e-12, rtol=config.BVP_TOL)
    sol = _integrate_adiabatic_outward(P_center, x_span, r_start)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("solve_static_structure: brentq converged to a P_center that no longer reaches the photosphere - numerical precision limit near the root")

    m_surface = np.exp(sol.t_events[0][0])
    residual_norm = abs(m_surface - config.M_TOTAL) / config.M_TOTAL

    m = np.logspace(np.log10(m_min), np.log10(m_surface), config.N_GRID_POINTS)
    r, lnP, lnT = sol.sol(np.log(m))
    P, T = np.exp(lnP), np.exp(lnT)
    rho = eos.density(P, T, config.MU, config.MU_E)

    # Diagnostic L(m): marginally-efficient-convection closure (gradients.py), NOT consumed by
    # solve_timestep (which only ever interpolates state_prev.T, .P - see _implicit_rhs_logm) -
    # exists to make state_0 a fully populated, physically meaningful SimulationState for
    # diagnostics/plots. Automatically satisfies the center BC L->0 as m->m_min (L is
    # proportional to m in this formula), no special-casing needed.
    kappa = opacity.bell_lin_opacity(rho, T)
    grad_ad = eos.grad_adiabatic(config.GAMMA)
    L = gradients.marginal_convective_luminosity(m, P, T, kappa, grad_ad)

    print(f"bvp_solver: t=0 compact hot start converged, P_center={P_center:.6e} dyn/cm^2, "
          f"T_center={config.T_CENTER_INITIAL:.1f} K, r_surface={r[-1]/6.9911e9:.3f} R_Jup, "
          f"m_surface/M_TOTAL={m_surface/config.M_TOTAL:.8f}, mass relative residual={residual_norm:.3e}")

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=0.0, prev=None)


# ==========================================
# SECTION: Implicit Per-Timestep Solve
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
    their many-decade range (see _integrate_adiabatic_outward).
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

    Convergence criteria: each pseudo-step's fsolve call must report ier==1 or the function
    raises immediately (a failed step means the alpha spacing should be made finer, not pushed
    through); the achieved [mass, thermal] residual is printed each step for a visible audit
    trail; a >50% (P_center, T_center) jump between consecutive steps is flagged as a possible
    solution-branch jump.
    """
    n_ramp_steps = 11   # alpha = 0.0, 0.1, ..., 1.0 - deliberately simple/auditable fixed spacing
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

    # ASSUMPTION: seeded from state_0's own center values, nudged by a tiny (1e-6) relative
    # amount to avoid catastrophic cancellation in dT_dt=(T-T_prev)/dt: at the exact match
    # point (alpha=0), T_trial and T_prev coincide to near machine precision (both follow the
    # same adiabat by construction), so the difference is floating-point noise, not a genuine
    # solution-boundary failure.
    u = np.array([np.log(state_0.P[0]), np.log(state_0.T[0])]) * (1.0 + 1.0e-6)
    for alpha in np.linspace(0.0, 1.0, n_ramp_steps):
        u_prev = u.copy()
        u, _info, ier, msg = fsolve(lambda u_trial: residual(u_trial, alpha), u, xtol=config.BVP_TOL, full_output=True)
        if ier != 1:
            raise RuntimeError(f"relax_initial_state: pseudo-step alpha={alpha:.3f} failed to converge (ier={ier}): {msg}")

        jump = np.max(np.abs(u - u_prev))   # ln-space, so this is a relative-change measure
        if alpha > 0.0 and jump > np.log(1.5):
            warnings.warn(
                f"relax_initial_state: (P_center, T_center) jumped >50% between alpha steps "
                f"(alpha={alpha:.3f}) - possible solution-branch jump, not just smooth "
                f"continuation; inspect before trusting this relaxation run", RuntimeWarning
            )

        res_mass, res_L = residual(u, alpha)
        print(f"bvp_solver: relaxation pseudo-step alpha={alpha:.3f} converged, "
              f"P_center={np.exp(u[0]):.6e}, T_center={np.exp(u[1]):.6e} K, "
              f"residuals=[{res_mass:.3e}, {res_L:.3e}]")

    P_center, T_center = np.exp(u)
    sol = _integrate_timestep_outward(P_center, T_center, x_span, r_start, state_0, dt_relax, 1.0)
    m_surface = np.exp(sol.t_events[0][0])

    m = np.logspace(np.log10(m_min), np.log10(m_surface), config.N_GRID_POINTS)
    r, lnP, L, lnT = sol.sol(np.log(m))
    P, T = np.exp(lnP), np.exp(lnT)
    rho = eos.density(P, T, config.MU, config.MU_E)

    print(f"bvp_solver: initial-state relaxation complete (alpha=1.0, genuine solution of the "
          f"real 4-ODE system), T_center={T_center:.6e} K, r_surface={r[-1]/6.9911e9:.3f} R_Jup, "
          f"m_surface/M_TOTAL={m_surface/config.M_TOTAL:.8f}")

    # t is left at state_0.t: this is a mathematical relaxation device (pseudo-steps at a fixed
    # pseudo-dt, state_prev held fixed at state_0 throughout), not real elapsed physical time.
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

    # Seed fsolve from state_prev's own (P_center, T_center), nudged by a tiny (1e-6) relative
    # amount - state_prev is itself a converged solution, so without the nudge the trial and
    # state_prev coincide to near machine precision right at the seed, and dT_dt=(T-T_prev)/dt
    # amplifies that floating-point noise into a spurious blow-up (same catastrophic-
    # cancellation mechanism as relax_initial_state's identical seed nudge).
    u0 = np.array([np.log(state_prev.P[0]), np.log(state_prev.T[0])]) * (1.0 + 1.0e-6)

    # Characteristic luminosity scale (Kelvin-Helmholtz estimate for this object) to
    # non-dimensionalize the radiative-flux residual, which is otherwise on a wildly different
    # absolute scale (~1e25+ erg/s) than the O(1) mass residual.
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

    u_sol, _info, ier, msg = fsolve(residual, u0, xtol=config.BVP_TOL, full_output=True)
    if ier != 1:
        warnings.warn(f"solve_timestep: fsolve did not fully converge (ier={ier}): {msg}", RuntimeWarning)

    P_center, T_center = np.exp(u_sol)
    sol = _integrate_timestep_outward(P_center, T_center, x_span, r_start, state_prev, dt)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("solve_timestep: converged (P_center, T_center) does not reach the photosphere - numerical precision limit near the root")
    m_surface = np.exp(sol.t_events[0][0])

    m = np.logspace(np.log10(m_min), np.log10(m_surface), config.N_GRID_POINTS)
    r, lnP, L, lnT = sol.sol(np.log(m))
    P, T = np.exp(lnP), np.exp(lnT)
    rho = eos.density(P, T, config.MU, config.MU_E)

    res_mass, res_L = residual(u_sol)
    print(f"bvp_solver: timestep converged, t={state_prev.t + dt:.4e} s, P_center={P_center:.6e}, "
          f"T_center={T_center:.6e} K, m_surface/M_TOTAL={m_surface/config.M_TOTAL:.8f}, "
          f"residuals=[{res_mass:.3e}, {res_L:.3e}]")

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=state_prev.t + dt, prev=state_prev)
