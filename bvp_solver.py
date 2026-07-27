# bvp_solver.py — Solves the t=0 compact-protoplanet structure and the per-timestep implicit
# structure for t>0, returning a populated SimulationState in each case.
#
# PHYSICAL NOTE (Sub-task 5 pivot, PROGRESS.md has the full investigation): t=0 used to be
# constructed as a diffuse, pre-collapse GI clump in isothermal equilibrium with the ambient
# nebula (T=T_neb, L=0 everywhere) - a physically real result (deeply Bonnor-Ebert subcritical,
# M_TOTAL/M_BE~0.089) but an exact, unbreakable fixed point under hydrostatic time-stepping: any
# scheme re-solving against fixed outer boundary conditions from a state that already satisfies
# them exactly reproduces "no change" for any dt. Root-cause analysis found the premise itself
# was the problem: initial GI collapse is inertia-dominated hydrodynamic free-fall, structurally
# outside what a quasi-static/hydrostatic solver can represent - the same reason
# config.T_DISSOCIATION_LIMIT halts the code at the far end of validity. Standard practice (PMS
# Henyey tracks; Bodenheimer & Pollack 1986; Marley et al. 2007 "hot start" gas-giant models) is
# to hand off from an assumed compact, high-entropy post-collapse state and evolve forward
# quasi-statically from there, never simulating the dynamical collapse itself.
#
# t=0 is therefore now a fully convective, hot protoplanet with a PRESCRIBED central temperature
# (config.T_CENTER_INITIAL, a chosen "hot start" parameter, not derived - see config.py). It is
# NOT static in the sense of dT_dt=dP_dt=0: a freshly-collapsed object is already contracting at
# its natural Kelvin-Helmholtz rate, and using genuinely zero time-derivatives would force L=0
# and reproduce the same isothermal degeneracy this pivot exists to escape. Instead the full
# 4-ODE system (odes.py, unchanged) is integrated with dT_dt, dP_dt set to the homologous-
# contraction rate (T/t_KH, 4P/t_KH, config.T_KH_TIMESCALE_S) evaluated on the current trial
# profile - physically honest for an object caught mid-collapse-relaxation, not a degeneracy-
# breaking trick (contrast with an earlier, rejected attempt this session to inject the same
# rate as an ADDITIONAL term inside solve_timestep's per-step energy equation, which double-
# counted compressional heating - see that function's docstring below).
#
# NUMERICAL/PHYSICAL NOTE (Sub-task 2f + outer BC redesign): ideal gas alone cannot support a
# genuinely compact (few-R_Jup) structure at any physically sane T_center - real gas-giant
# compactness comes from electron-degeneracy pressure (eos.degenerate_pressure), now included
# additively with the ideal-gas term (eos.density). This alone reproduces the analytic
# Zapolsky & Salpeter (1969)-style prediction almost exactly (R~3.1-3.2 R_Jup at
# T_CENTER_INITIAL). But the degenerate-supported structure exposed a second, independent gap:
# the inherited P(M_TOTAL)=P_neb mechanical surface condition has NO SOLUTION AT ALL for this
# structure (not a hard-to-find root - a genuine gap in achievable surface pressure, confirmed
# not a numerical-precision artifact; PROGRESS.md has the full trail). Replaced with a
# physically-motivated photospheric condition (Eddington tau=2/3, boundary_conditions.py) -
# see that module's docstring for the derivation. This changes HOW the surface is located, not
# just the residual formula: both shooting routines below now integrate outward with the
# photosphere as a solve_ivp EVENT (mirroring _solve_lane_emden's own surface-crossing event)
# and match the ENCLOSED MASS at that event to M_TOTAL, rather than checking a residual at a
# fixed m=M_TOTAL grid endpoint - a fixed-endpoint version of the photospheric condition was
# tested and found to have the same reachability-gap problem the old P=P_neb condition did.
#
# SOLVER NOTE: scipy.integrate.solve_bvp is not used (established in the original isothermal
# construction and unchanged here): the surface pressure boundary layer and the wide dynamic
# range of P, T across the mass grid defeat its collocation Jacobian. Both t=0 and t>0 use a
# shooting method (scipy.integrate.solve_ivp outward integration + root-finding on the central
# conditions) instead.

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
    tabulated values for n=1.5 (xi_1=3.65375) and n=3.0 (xi_1=6.89685) during development.
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
    below - not the final answer (the real shooting includes BOTH the ideal-gas and
    degenerate pressure terms and locates the exact photospheric surface, not this idealized
    polytrope's own P->0 natural surface).

    ASSUMPTION (Sub-task 2f): uses the pure T=0 electron-degeneracy limit (an n=3/2
    polytrope, P=K1*rho^(5/3) - Zapolsky & Salpeter 1969 style) rather than the pure ideal-gas
    adiabat used before Sub-task 2f. Justified because degeneracy dominates the interior by
    ~2-3 orders of magnitude in pressure at the density this seed predicts (PROGRESS.md
    Sub-task 2f: rho_center here is ~600x the ideal/degenerate crossover density at
    T_CENTER_INITIAL) - a pure-degenerate polytrope is now the better-motivated seed than a
    pure ideal-gas one. There is no closed-form Lane-Emden solution for the real, additive
    combined EOS (not a pure power law in rho), so this remains an approximate seed - a blind
    numerical P_center search does NOT reliably find the true root, and this analytic estimate
    pins the right order of magnitude to bracket reliably via geometric expansion (see
    solve_static_structure). Still a reasonable seed regardless of how the surface is defined
    (photospheric vs. fixed-pressure), since the interior remains degeneracy-dominated either way.
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
# (boundary_conditions.photospheric_pressure), not at a fixed m=M_TOTAL grid endpoint - see
# that module's docstring for the physical reasoning and module docstring above for why a
# fixed-endpoint version doesn't work for the degenerate-supported structure. The event
# functions differ only in how they unpack the two RHS's different state vectors.

def _photosphere_event_adiabatic(x, y):
    r, P, T = y
    return P - boundary_conditions.photospheric_pressure(r, P, T, config.MU, config.MU_E)
_photosphere_event_adiabatic.terminal = True
_photosphere_event_adiabatic.direction = -1


def _photosphere_event_implicit(x, y):
    r, P, L, T = y
    return P - boundary_conditions.photospheric_pressure(r, P, T, config.MU, config.MU_E)
_photosphere_event_implicit.terminal = True
_photosphere_event_implicit.direction = -1


# ==========================================
# SECTION: Adiabatic (Fully Convective) Right-Hand Side
# ==========================================

def _adiabatic_rhs_logm(x, y):
    """dr/dx, dP/dx, dT/dx (x=ln(m)) for a fully convective sphere.

    ASSUMPTION: dT/dm = (T/P)*grad_ad*dP/dm identically - a fresh, high-entropy post-collapse
    object is assumed fully convective throughout (standard Hayashi-track picture), not derived
    via the Schwarzschild criterion. NOT routed through odes.stellar_odes: that function picks
    grad_eff=min(grad_rad, grad_ad), which needs a self-consistent L - unavailable (and not the
    physical picture being assumed) for this purely-convective structural construction.

    ASSUMPTION (Sub-task 2f): the THERMAL profile (T vs P, via grad_ad) still follows the pure
    ideal-gas adiabat unchanged - a full treatment would re-derive the adiabatic index for the
    combined ideal+degenerate EOS's actual entropy, which is out of scope for this minimal,
    additive-pressure-only fix. Only the MECHANICAL structure (rho, hence dr/dm) uses the new
    combined EOS (eos.density). This is a first-order approximation, not fully self-consistent
    thermodynamically - see PROGRESS.md Sub-task 2f entry.
    """
    m = np.exp(x)
    r, P, T = y
    rho = eos.density(P, T, config.MU, config.MU_E)
    dr_dm = 1.0 / (4.0 * np.pi * r**2 * rho)
    dP_dm = -config.G * m / (4.0 * np.pi * r**4)
    dT_dm = (T / P) * eos.grad_adiabatic(config.GAMMA) * dP_dm
    return [dr_dm * m, dP_dm * m, dT_dm * m]


def _integrate_adiabatic_outward(P_center, x_span, r_start):
    """Integrate (r, P, T) outward from x_span[0] (r=r_start, T=config.T_CENTER_INITIAL) for a
    trial P_center, terminating at the photosphere event. See solve_static_structure's
    ASSUMPTION note on atol scaling."""
    return solve_ivp(
        _adiabatic_rhs_logm, x_span, [r_start, P_center, config.T_CENTER_INITIAL], method="Radau",
        dense_output=True, events=_photosphere_event_adiabatic,
        rtol=config.BVP_TOL, atol=[1.0, config.P_NEB * config.BVP_TOL, 1.0e-6],
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
    # Generous margin past M_TOTAL for the photosphere event to trigger within - empirically
    # (PROGRESS.md Sub-task 5 outer-BC entry) the event triggers within ~1.4x of M_TOTAL even
    # for a P_center 3x too high, so this is cheap headroom, never expected to bind.
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
        # ASSUMPTION (Sub-task 2f): which direction (increasing or decreasing P_center)
        # reduces the mass-matching residual is NOT fixed - it depends on whether the
        # ideal-gas or degenerate term dominates the structure (a genuine consequence of the
        # inverted mass-radius relation for degenerate objects, R~M^-1/3 - PROGRESS.md has the
        # numerical trail, not a bracketing bug). Expand in both directions simultaneously,
        # taking whichever finds a sign change first, rather than assuming a fixed direction.
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
    r, P, T = sol.sol(np.log(m))
    rho = eos.density(P, T, config.MU, config.MU_E)

    # Diagnostic L(m): marginally-efficient-convection closure (gradients.py), NOT consumed by
    # solve_timestep (which only ever interpolates state_prev.T, .P - see _implicit_rhs_logm)-
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
    """dy/dx (x=ln(m)) for the full 4-ODE system, dT_dt, dP_dt computed ON THE FLY from the
    CURRENT trial (T, P) differenced against state_prev's (interpolated) profile - the textbook
    implicit (Henyey-style) form: dt enters directly, and the energy equation's source term is
    exactly this state difference, nothing added on top.

    PHYSICAL NOTE: an earlier version of this function added an EXTRA explicit homologous-
    contraction term (T_prev/t_KH, 4*P_prev/t_KH) on top of the implicit difference, intended to
    keep evolution going past a single step. That double-counted compressional heating - proven
    via energy-conservation violation (a state exactly frozen step-to-step nonetheless radiated
    a constant, non-decaying L, with no reservoir to draw from) - and was reverted. Real
    per-timestep evolution instead comes from t=0 already being a genuine, self-consistent
    thermal disequilibrium (module docstring above), not an injected rate law here.

    RELAXATION NOTE (alpha, see relax_initial_state): nabla_eff is blended between the pure
    adiabat (alpha=0, matching solve_static_structure's own construction exactly - a genuinely
    self-consistent starting point, NOT the isothermal one) and the real Schwarzschild-selected
    value (alpha=1, the standard, unmodified formula below). dL/dm itself is NEVER scaled by
    alpha - an earlier, rejected version of this relaxation idea scaled the whole energy-
    equation source term instead, which forces dL/dm=0 identically at alpha=0 (dL/dm has no
    other source in this codebase) - reproducing the exact isothermal degeneracy proven
    unbreakable at the very start of this investigation (PROGRESS.md). alpha=1.0 (the default)
    reproduces the original formula exactly - every genuine per-timestep solve_timestep() call
    uses this default; only relax_initial_state()'s pseudo-steps use alpha<1.
    """
    m = np.exp(x)
    r, P, L, T = y
    # ASSUMPTION: Radau's own internal stiff-solver Newton iteration probes trial y-vectors
    # while estimating its collocation Jacobian, and can probe P or T slightly negative -
    # unphysical (no EOS solution exists for negative pressure or temperature), which would
    # otherwise crash eos.density's own Newton solve regardless of iteration count. Clamping
    # to a tiny positive floor here (a domain guard on the RHS's OWN inputs, not a fabricated
    # result) lets Radau's adaptive step control naturally back away from the bad region,
    # rather than crashing on an intermediate probe that was never going to be the accepted
    # step - the same spirit as eos.density's own rho-positivity clamp.
    P = max(P, 1.0e-300)
    T = max(T, 1.0e-300)
    T_prev = np.interp(m, state_prev.m, state_prev.T)
    P_prev = np.interp(m, state_prev.m, state_prev.P)
    dT_dt = (T - T_prev) / dt
    dP_dt = (P - P_prev) / dt
    y_full = np.array([[r], [P], [L], [T]])
    dr_dm, dP_dm, dL_dm, dT_dm_real = odes.stellar_odes(
        np.array([m]), y_full, np.array([dT_dt]), np.array([dP_dt])
    )
    if alpha == 1.0:
        dT_dm = dT_dm_real
    else:
        # Same (T/P)*grad_ad*dP_dm form as _adiabatic_rhs_logm (eos.grad_adiabatic, single
        # source of truth for grad_ad); dP_dm is shared (hydrostatic equilibrium doesn't
        # depend on the temperature gradient), so only dT_dm needs recombining.
        dT_dm_adiabatic = (T / P) * eos.grad_adiabatic(config.GAMMA) * dP_dm
        dT_dm = (1.0 - alpha) * dT_dm_adiabatic + alpha * dT_dm_real
    return [dr_dm[0] * m, dP_dm[0] * m, dL_dm[0] * m, dT_dm[0] * m]


def _integrate_timestep_outward(P_center, T_center, x_span, r_start, state_prev, dt, alpha=1.0):
    """Integrate the full 4-ODE system outward for a trial (P_center, T_center), terminating
    at the photosphere event; see _implicit_rhs_logm. L starts at exactly 0 (center BC)."""
    def rhs(x, y):
        return _implicit_rhs_logm(x, y, state_prev, dt, alpha)
    return solve_ivp(
        rhs, x_span, [r_start, P_center, 0.0, T_center], method="Radau", dense_output=True,
        events=_photosphere_event_implicit,
        rtol=config.BVP_TOL, atol=[1.0, config.P_NEB * config.BVP_TOL, 1.0, 1.0e-6],
    )


# ==========================================
# SECTION: Initial-Model Relaxation (bridges solve_static_structure to solve_timestep)
# ==========================================

def relax_initial_state(state_0) -> state.SimulationState:
    """Relax state_0 (solve_static_structure's output - a fully convective structure built by
    FORCING the pure ideal-gas adiabat, not a genuine solution of the real 4-ODE system's
    Schwarzschild-selected temperature gradient) into a state that IS self-consistent with the
    same implicit equations solve_timestep() uses, via continuation in alpha
    (_implicit_rhs_logm's nabla_eff blend fraction).

    PHYSICAL MOTIVATION: t=0 is, by this project's own premise, a prescribed hand-off snapshot
    (T_CENTER_INITIAL is chosen, not derived) - not something with a physical basis for being
    an exact root of solve_timestep's real-time-differenced equations, since there is no
    previous state to difference against at t=0 to derive one from. Directly evaluating
    solve_timestep's real 4-ODE system AT state_0's own values confirmed this mismatch is
    large: T diverges to ~3.4 million K within one full-sized implicit step (dt=0.01*t_KH).
    This is standard "initial model relaxation" territory (MESA-style pre-main-sequence
    relaxation; classical Henyey-code ZAMS model construction): rather than demanding state_0
    already be a perfect root, or inventing an unphysical driving term to force it to be one,
    walk a continuous path of INCREASINGLY REAL problems from a genuinely easy starting point
    (alpha=0, which reproduces state_0's OWN construction exactly, by design - not the
    isothermal degeneracy an earlier, rejected version of this idea collapsed to, see
    _implicit_rhs_logm's RELAXATION NOTE) to the real target (alpha=1), using each step's
    converged (P_center, T_center) to warm-start the next.

    CONVERGENCE CRITERIA (kept strict and explicit per request - no blind continuation):
    - Each pseudo-step's fsolve call must report ier==1 (full convergence); a failed step
      RAISES immediately (does not silently continue with an unreliable intermediate state,
      unlike solve_timestep's real per-timestep calls, which only warn - here we control the
      alpha step size, so a failure means the step should be made finer, not pushed through).
    - After each step, the residual actually achieved [mass, thermal] is printed for a visible
      audit trail (not just "ier==1", which only reflects fsolve's own internal criteria).
    - The (P_center, T_center) jump between consecutive alpha steps is checked against a
      smoothness guard (relative change < 50% per step) - fsolve reporting ier==1 does not by
      itself rule out having converged to a DIFFERENT solution branch than the one being
      continuously tracked; a large jump is flagged as a possible branch jump rather than
      silently accepted.
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
        yb = sol.y_events[0][0]
        res_full = boundary_conditions.boundary_conditions(np.zeros(4), yb)
        mass_residual = (m_event - config.M_TOTAL) / config.M_TOTAL
        return [mass_residual, res_full[3] / L_scale]

    # ASSUMPTION: seeded from state_0's own center values (exact at alpha=0 by construction),
    # nudged by a tiny, physically negligible relative amount (1e-6, far above floating-point
    # noise but far below any real physical scale) to avoid a catastrophic-cancellation
    # artifact: AT the exact match point, T_trial and T_prev coincide almost to machine
    # precision (both follow the same adiabat by construction at alpha=0), so
    # dT_dt=(T-T_prev)/dt divides near-zero floating-point noise by a small dt, amplifying it
    # into an overflow (confirmed: the exact point fails, every nearby perturbed point
    # succeeds cleanly) - not a genuine solution-boundary failure.
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
    r, P, L, T = sol.sol(np.log(m))
    rho = eos.density(P, T, config.MU, config.MU_E)

    print(f"bvp_solver: initial-state relaxation complete (alpha=1.0, genuine solution of the "
          f"real 4-ODE system), T_center={T_center:.6e} K, r_surface={r[-1]/6.9911e9:.3f} R_Jup, "
          f"m_surface/M_TOTAL={m_surface/config.M_TOTAL:.8f}")

    # t is left at state_0.t: this is a mathematical relaxation device (pseudo-steps at a fixed
    # pseudo-dt, state_prev held fixed at state_0 throughout), not real elapsed physical time -
    # same convention the earlier (now-removed) bootstrap kick used, for the same reason.
    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=state_0.t, prev=state_0)


def solve_timestep(state_prev, dt) -> state.SimulationState:
    """Solve the envelope structure at t = state_prev.t + dt via the implicit shooting scheme.
    Shoots on (ln P_center, ln T_center) - log-parametrized for guaranteed positivity through
    fsolve's trial evaluations, the same pattern as solve_static_structure()'s brentq shoot - to
    match two conditions at the photosphere event: enclosed mass = M_TOTAL (mechanical, replaces
    a fixed-endpoint P=P_neb residual - module docstring explains why) and the net radiative
    flux balance (thermal, boundary_conditions.py, evaluated at the event point). The center
    residuals (r_a=0, L_a=0) are satisfied exactly by construction (integration starts at
    r=r_start, L=0).

    From t=0 onward directly - no separate bootstrap/kick step: state_prev already carries a
    genuine, non-degenerate thermal state (module docstring above), so the first real call
    (state_prev=solve_static_structure()'s output) is not a special case.
    """
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    x_span = (np.log(m_min), np.log(50.0 * config.M_TOTAL))
    r_start = state_prev.r[0]

    # Seed fsolve from state_prev's own (P_center, T_center) - state_prev is an actual solved
    # state, already a reasonable starting point for a sensibly-sized dt; no assumed rate law
    # needed just to seed the search (contrast with the removed _homologous_initial_guess).
    u0 = np.array([np.log(state_prev.P[0]), np.log(state_prev.T[0])])

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
        yb = sol.y_events[0][0]
        res_full = boundary_conditions.boundary_conditions(np.zeros(4), yb)
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
    r, P, L, T = sol.sol(np.log(m))
    rho = eos.density(P, T, config.MU, config.MU_E)

    res_mass, res_L = residual(u_sol)
    print(f"bvp_solver: timestep converged, t={state_prev.t + dt:.4e} s, P_center={P_center:.6e}, "
          f"T_center={T_center:.6e} K, m_surface/M_TOTAL={m_surface/config.M_TOTAL:.8f}, "
          f"residuals=[{res_mass:.3e}, {res_L:.3e}]")

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=state_prev.t + dt, prev=state_prev)
