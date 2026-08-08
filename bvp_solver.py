# bvp_solver.py — Solves the t=0 compact-protoplanet structure and the per-timestep implicit
# structure for t>0, returning a populated SimulationState in each case.
#
# PHYSICAL PICTURE: t=0 is a hot, compact, fully convective post-collapse protoplanet with a
# PRESCRIBED central temperature (config.T_CENTER_INITIAL - a chosen "hot start" parameter,
# not derived; standard practice for gas-giant "hot start" models, e.g. Marley et al. 2007).
# It is not static (dT_dt=dP_dt=0): it is already contracting at its natural Kelvin-Helmholtz
# rate, evaluated on the current trial profile via implicit (Henyey-style) time differencing.
# Compactness (a few R_Jup, not a few hundred) requires non-relativistic electron-degeneracy
# pressure (eos.py), included additively with the ideal-gas term.
#
# OUTER BOUNDARY: the photosphere (Eddington tau=2/3, boundary_conditions.py), not a fixed
# ambient pressure P_neb - a degenerate-supported structure's atmosphere hands off to a
# photosphere at a pressure set by its own surface gravity and opacity, not by the ambient
# nebula.
#
# SOLVER, TWO DIFFERENT METHODS FOR TWO DIFFERENT PROBLEMS (2026-08-08 architecture):
#   - t=0 (solve_static_structure): a simple 3-ODE pure-adiabat construction, ONE shooting
#     unknown (P_center). Integrates outward with scipy.integrate.solve_ivp as a solve_ivp
#     EVENT locates the photosphere, `brentq` root-finds on enclosed-mass-at-event=M_TOTAL.
#     This has always been robust (never implicated in the crash investigation below) and is
#     UNCHANGED by the 2026-08-08 pivot.
#   - t>0 (relax_initial_state, solve_timestep): the full 4-ODE system, previously solved by
#     the same shooting strategy (root-find via scipy.optimize.root on two unknowns,
#     P_center/T_center) - RETIRED 2026-08-08, archived verbatim in
#     bvp_solver_shooting_archive.py. PLAN_BVP.md (Milestones 0-4) traced repeated numerical
#     kinks near the photosphere to a genuine Jacobian rank deficiency (100% convective
#     saturation under the infinitely-efficient-convection idealization makes
#     d(nabla_eff)/d(nabla_rad)=0, decoupling L from the P-T relation) - not fixable by
#     patching individual physics/BC terms, confirmed by five independent isolated tests.
#     Now solved via scipy.integrate.solve_bvp (global collocation/relaxation, the same
#     numerical family as Henyey's implicit relaxation used by production stellar-evolution
#     codes): a NONDIMENSIONALIZED state vector (r, L rescaled to be O(1)-comparable with the
#     already-log-transformed P, T), analytic Jacobians (fun_jac/bc_jac, replacing scipy's
#     default finite-difference estimate), and a continuation in alpha (blending the pure
#     adiabat toward the real Schwarzschild-selected gradient) ending just short of the
#     literal alpha=1.0 endpoint (config.BVP_ALPHA_MAX - a small, empirically-required
#     regularizer, PLAN_BVP.md Milestone 6) together attack the Jacobian's CONDITIONING
#     directly, without fixing the underlying rank deficiency itself - a genuine
#     mixing-length-theory convection treatment remains the mathematically complete fix and
#     stays on the roadmap, deliberately deferred (PLAN_BVP.md §6). Proven at
#     T_CENTER_INITIAL=11500K and 12000K in the standalone bvp_experiment.py before being
#     promoted here.
#
# STATE REPRESENTATION: P, T are always log-transformed (lnP, lnT) so P=exp(lnP)>0,
# T=exp(lnT)>0 hold by construction through the solver's internal probing - the standard
# Henyey/MESA-style representation. For t>0, r and L are ADDITIONALLY nondimensionalized
# (r_hat=r/R_JUPITER_CM, L_hat=arcsinh(L/L_KH_SCALE_ERG_S)) - see the State-Vector Scaling
# section below. See PROGRESS.md and PLAN_BVP.md for the full physical/numerical history
# behind every design choice in this file.

import time

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq

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
# SECTION: Composite Output Mass Grid
# ==========================================

def _build_output_grid(m_min, m_surface):
    """Composite Lagrangian mass grid for sampling the dense solve_ivp/solve_bvp solution:
    log-spaced in the core (unchanged), and log-spaced in DISTANCE-TO-SURFACE over the outer
    config.GRID_OUTER_MASS_FRACTION of the mass, so resolution increases smoothly toward the
    photosphere instead of collapsing to ~1-2 points there.

    ASSUMPTION: pure np.logspace(m_min, m_surface) puts the outer 10% of mass (where T, rho,
    P actually change fastest, near the photosphere) into only ~0.05 of the grid's ~6 decades
    in log-space - starving that region of points regardless of the underlying solve's own
    dense, accurate interior interpolant. This produced a visibly jagged, under-resolved drop
    in structure-profile plots even though the underlying physics is smooth (PROGRESS.md
    2026-08-01 entry has the diagnosis).
    """
    n_outer = int(round(config.N_GRID_POINTS * config.GRID_OUTER_POINT_FRACTION))
    n_core = config.N_GRID_POINTS - n_outer

    m_transition = (1.0 - config.GRID_OUTER_MASS_FRACTION) * m_surface
    core = np.logspace(np.log10(m_min), np.log10(m_transition), n_core, endpoint=False)

    delta_max = m_surface - m_transition
    delta_min = delta_max * config.GRID_OUTER_REFINEMENT
    outer_deltas = np.logspace(np.log10(delta_max), np.log10(delta_min), n_outer - 1)
    outer = m_surface - outer_deltas

    return np.concatenate([core, outer, [m_surface]])


# ==========================================
# SECTION: Photosphere Event (tau=2/3 surface location) — t=0 adiabat only
# ==========================================
# solve_static_structure locates the surface via a solve_ivp EVENT (tau=2/3 photosphere,
# boundary_conditions.photospheric_pressure) and matches the ENCLOSED MASS at that event to
# M_TOTAL, rather than checking a residual at a fixed m=M_TOTAL grid endpoint - see that
# module's docstring for the physical reasoning. This is the t=0 problem's OWN solve_ivp
# integration (3-ODE adiabat, `brentq` shooting) - unrelated to the t>0 solve_bvp machinery
# below, where the mass domain [m_min, M_TOTAL] is fixed and known exactly instead (no event
# needed - see the t>0 sections' own docstrings).

def _photosphere_event_adiabatic(x, y):
    r, lnP, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    return P - boundary_conditions.photospheric_pressure(r, P, T, config.MU, config.MU_E)
_photosphere_event_adiabatic.terminal = True
_photosphere_event_adiabatic.direction = -1


# ==========================================
# SECTION: Adiabatic (Fully Convective) Right-Hand Side — t=0 only
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
        # ASSUMPTION: widened 2026-08-07 (PLAN_BVP.md Milestone 5/PROGRESS.md) - the seed
        # (_adiabatic_center_guess) is a fixed T=0-degenerate-limit estimate, independent of
        # config.MU/GAMMA and config.T_CENTER_INITIAL; the true root's distance from that
        # fixed point grows as thermal (ideal-gas) support becomes more significant relative
        # to degeneracy - already flagged as a genuine gap at high T_center (Milestone 5), and
        # confirmed 2026-08-07 to also bind at T_CENTER_INITIAL=13000K once MU dropped from
        # the molecular (2.34) to the atomic (1.278) value: P_ideal~1/mu at fixed rho,T, so a
        # smaller mu increases the ideal-gas contribution and shifts the required P_center
        # enough to exceed the previous ~7.3x (1.01^200) window. 1.03^300~7100x is generous
        # without materially slowing a successful search (brentq's own bisection, not this
        # loop, still does the real root-polishing once any valid bracket is found).
        P_up = P_down = P_center_guess
        P_low = P_high = None
        for _ in range(300):
            P_up *= 1.03
            e_up = mass_error(P_up)
            if (e_up < 0.0) != (e_seed < 0.0):
                P_low, P_high = (P_center_guess, P_up) if e_seed > 0.0 else (P_up, P_center_guess)
                break
            P_down /= 1.03
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
    # brentq is bracket-based (the true root stays trapped between P_low, P_high throughout,
    # unlike fsolve's step-size-only ier==1 - PROGRESS.md 2026-08-01), so this is a lower-risk
    # case, but the same "verify, don't just trust the root-finder's own report" discipline
    # applies - a cheap, direct check on the residual itself.
    assert residual_norm <= config.STATIC_STRUCTURE_RESIDUAL_TOL, (
        f"solve_static_structure: brentq converged but mass residual {residual_norm:.3e} "
        f"exceeds config.STATIC_STRUCTURE_RESIDUAL_TOL={config.STATIC_STRUCTURE_RESIDUAL_TOL:.1e}"
    )

    m = _build_output_grid(m_min, m_surface)
    r, lnP, lnT = sol.sol(np.log(m))
    P, T = np.exp(lnP), np.exp(lnT)
    rho = eos.density(P, T, config.MU, config.MU_E)

    # Diagnostic L(m): marginally-efficient-convection closure (gradients.py), NOT consumed by
    # solve_timestep (which only ever interpolates state_prev.T, .P - see implicit_rhs_scaled) -
    # exists to make state_0 a fully populated, physically meaningful SimulationState for
    # diagnostics/plots. Automatically satisfies the center BC L->0 as m->m_min (L is
    # proportional to m in this formula), no special-casing needed.
    kappa = opacity.bell_lin_opacity(rho, T)
    grad_ad = eos.grad_adiabatic(config.GAMMA)
    L = gradients.marginal_convective_luminosity(m, P, T, kappa, grad_ad)

    print(f"bvp_solver: t=0 compact hot start converged, P_center={P_center:.6e} dyn/cm^2, "
          f"T_center={config.T_CENTER_INITIAL:.1f} K, r_surface={r[-1]/config.R_JUPITER_CM:.3f} R_Jup, "
          f"m_surface/M_TOTAL={m_surface/config.M_TOTAL:.8f}, mass relative residual={residual_norm:.3e}")

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=0.0, prev=None)


# ==========================================
# SECTION: State-Vector Scaling (t>0, PLAN_BVP.md Milestone 6)
# ==========================================
#
# Motivation: the raw physical state y=[r,lnP,L,lnT] has extreme, directly-measured
# heterogeneity - a single Jacobian-verification point showed y=[r=2.9e10, lnP=5.9,
# L=2.6e29, lnT=3.6] - L is 28 ORDERS OF MAGNITUDE larger than lnT in the same vector Newton
# must invert, a textbook cause of an ill-conditioned Jacobian independent of any individual
# physics term. r and L are rescaled to be O(1)-comparable with the already-good lnP/lnT; P
# and T stay log-transformed (unchanged, already well-conditioned and positivity-guaranteed).
#
# New state z = [r_hat, lnP, L_hat, lnT]:
#   r_hat = r / R_SCALE                          (linear rescaling - R_SCALE is a true constant)
#   L_hat = arcsinh(L / L_SCALE)                 (nonlinear, sign-preserving, log-like compression)
#
# arcsinh over a hand-rolled sign*log1p: a single closed-form expression (arcsinh(x)=
# ln(x+sqrt(x^2+1))), smooth (C-infinity) with no piecewise branching, and its own derivative
# (1/sqrt(x^2+1)) is simple and well-conditioned everywhere, including at x=0.
R_SCALE = config.R_JUPITER_CM         # [cm] - true constant, r/R_SCALE is a LINEAR rescaling
L_SCALE = config.L_KH_SCALE_ERG_S     # [erg/s] - already-vetted KH-luminosity reference (config.py)


def _to_physical(z):
    """z=[r_hat, lnP, L_hat, lnT] -> y=[r, lnP, L, lnT] (physical, P/T still log). L=L_SCALE*
    sinh(L_hat) is the exact inverse of L_hat=arcsinh(L/L_SCALE)."""
    r_hat, lnP, L_hat, lnT = z
    return np.array([r_hat * R_SCALE, lnP, L_SCALE * np.sinh(L_hat), lnT])


def _to_scaled(y):
    """y=[r, lnP, L, lnT] (physical, P/T still log) -> z=[r_hat, lnP, L_hat, lnT]."""
    r, lnP, L, lnT = y
    return np.array([r / R_SCALE, lnP, np.arcsinh(L / L_SCALE), lnT])


# ==========================================
# SECTION: Vectorized Physical-Space RHS and Jacobian (t>0)
# ==========================================
# solve_bvp calls fun(x, y) / fun_jac(x, y) with the WHOLE mesh at once (x shape (n,), y shape
# (4, n)) - odes.stellar_odes is already vectorized this way (CLAUDE.md style rule), so these
# call it directly rather than wrapping it point-by-point (contrast the retired shooting
# _implicit_rhs_logm in bvp_solver_shooting_archive.py, written for solve_ivp's single-point
# contract).

def implicit_rhs_vectorized(x, y, state_prev, dt, alpha):
    """dy/dx for the full 4-ODE system (y=[r,lnP,L,lnT], PHYSICAL, not yet nondimensionalized),
    vectorized across solve_bvp's whole mesh at once. dT_dt, dP_dt are the implicit
    (Henyey-style) time derivatives: the current trial (T,P) differenced against state_prev's
    (interpolated) profile - the energy equation's source term is exactly this difference.

    alpha blends nabla_eff between the pure adiabat (alpha=0, matches solve_static_structure's
    own construction) and the real Schwarzschild-selected value (alpha=1). dL/dm is NEVER
    scaled by alpha - it has no other source in this codebase, so scaling it would force
    dL/dm=0 identically at alpha=0.
    """
    m = np.exp(x)
    r, lnP, L, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    # Diagnostic only (does not alter behavior): report WHERE along the mesh a trial (P,T) is
    # extreme, before it potentially crashes eos.density downstream.
    bad = ~np.isfinite(P) | ~np.isfinite(T) | (P <= 0) | (T <= 0) | (np.abs(lnP) > 300) | (np.abs(lnT) > 300)
    if np.any(bad):
        idx = np.where(bad)[0]
        print(f"bvp_solver: [diag] extreme/non-finite (P,T) trial at {len(idx)} mesh point(s), "
              f"m/M_TOTAL in [{(m[idx]/config.M_TOTAL).min():.3e}, {(m[idx]/config.M_TOTAL).max():.3e}], "
              f"lnP range=[{np.nanmin(lnP[idx]):.3e},{np.nanmax(lnP[idx]):.3e}], "
              f"lnT range=[{np.nanmin(lnT[idx]):.3e},{np.nanmax(lnT[idx]):.3e}]", flush=True)
    T_prev = np.interp(m, state_prev.m, state_prev.T)
    P_prev = np.interp(m, state_prev.m, state_prev.P)
    dT_dt = (T - T_prev) / dt
    dP_dt = (P - P_prev) / dt
    y_full = np.array([r, P, L, T])   # shape (4, n) - odes.stellar_odes's native contract
    dr_dm, dP_dm, dL_dm, dT_dm_real = odes.stellar_odes(m, y_full, dT_dt, dP_dt)
    dlnP_dm = dP_dm / P
    if alpha == 1.0:
        dlnT_dm = dT_dm_real / T
    else:
        # grad_ad=dlnT/dlnP by definition (eos.grad_adiabatic).
        dlnT_dm_ad = eos.grad_adiabatic(config.GAMMA) * dlnP_dm
        dlnT_dm = (1.0 - alpha) * dlnT_dm_ad + alpha * (dT_dm_real / T)
    # Chain rule: d/dx = m * d/dm, since x = ln(m)
    return np.array([dr_dm, dlnP_dm, dL_dm, dlnT_dm]) * m


def _eos_density_derivatives(P, T, rho):
    """d(rho)/dP, d(rho)/dT via IMPLICIT differentiation of the EOS's defining equation
    (NOT differentiating through eos.density's own Newton iteration): F(rho,P,T) =
    P_ideal(rho,T) + P_deg(rho) - P = 0.

    Standard implicit-function-theorem result: drho/dP = -(dF/dP)/(dF/drho),
    drho/dT = -(dF/dT)/(dF/drho). dF/drho is exactly the same quantity
    (dP_ideal_drho + dP_deg_drho) eos.density's own Newton loop already computes as its
    convergence-step denominator - reproduced here (eos.py exposes no derivative API by
    design, since gradients.py/odes.py never needed one before this solver).
    """
    dP_ideal_drho = config.K_B * T / (config.MU * config.M_H)              # dF/drho, ideal term
    P_deg = eos.degenerate_pressure(rho, config.MU_E)
    dP_deg_drho = (5.0 / 3.0) * P_deg / rho                                 # dF/drho, degenerate term
    D = dP_ideal_drho + dP_deg_drho                                        # dF/drho total
    drho_dP = 1.0 / D                                                      # dF/dP = -1
    P_ideal = rho * dP_ideal_drho
    drho_dT = -(P_ideal / T) / D                                           # dF/dT = -P_ideal/T
    return drho_dP, drho_dT


def _thermodynamic_delta_derivatives(rho, T, drho_dP, drho_dT, delta):
    """d(delta)/dP, d(delta)/dT for eos.thermodynamic_delta, via logarithmic differentiation
    of delta=P_ideal/(rho*D), D=dP_ideal_drho+dP_deg_drho (same quantities eos.
    thermodynamic_delta itself computes). Both -> 0 as dP_deg_drho -> 0 (pure ideal gas,
    delta=1 exactly, a true constant - zero sensitivity), matching eos.thermodynamic_delta's
    own limiting-case docstring.
    """
    dP_ideal_drho = config.K_B * T / (config.MU * config.M_H)
    P_deg = eos.degenerate_pressure(rho, config.MU_E)
    dP_deg_drho = (5.0 / 3.0) * P_deg / rho
    D = dP_ideal_drho + dP_deg_drho
    ddelta_dP = -delta * (2.0 / 3.0) * (dP_deg_drho / (rho * D)) * drho_dP
    ddelta_dT = delta * (dP_deg_drho / D) * (1.0 / T - (2.0 / 3.0) * drho_dT / rho)
    return ddelta_dP, ddelta_dT


def _opacity_derivatives(rho, T, kappa):
    """d(kappa)/d(rho), d(kappa)/dT from the LOCALLY active Bell & Lin regime's own power
    law, kappa=kappa_i*rho^a*T^b: d(kappa)/d(rho)=a*kappa/rho, d(kappa)/dT=b*kappa/T - exact
    almost everywhere (undefined only exactly AT a regime transition, measure zero)."""
    regime_idx = opacity.determine_regime(rho, T)
    a = np.array([opacity.REGIMES[i].a for i in np.atleast_1d(regime_idx).ravel()]).reshape(np.shape(regime_idx))
    b = np.array([opacity.REGIMES[i].b for i in np.atleast_1d(regime_idx).ravel()]).reshape(np.shape(regime_idx))
    dkappa_drho = a * kappa / rho
    dkappa_dT = b * kappa / T
    return dkappa_drho, dkappa_dT


def _grad_radiative_derivatives(L, m, P, T, kappa, rho, drho_dP, drho_dT, dkappa_drho, dkappa_dT, grad_rad):
    """d(grad_rad)/dL, dP, dT for grad_rad = 3*kappa*L_safe*P/(16*pi*A_RAD*C_LIGHT*G*m*T^4)
    (gradients.grad_radiative). L_safe is the smoothed hyperbolic floor - differentiated here
    from the mathematically-equivalent SIMPLE form L_safe=0.5*(L+sqrt(L^2+eps^2)) (algebraically
    identical to the cancellation-safe form gradients.py evaluates, so the exact derivative is
    identical too). grad_rad depends on P, T both explicitly (T^-4, P^1) AND implicitly through
    kappa(rho(P,T),T) - both channels included via the chain rule.
    """
    eps_L = config.GRAD_RAD_L_FLOOR_EPSILON
    dLsafe_dL = 0.5 * (1.0 + L / np.sqrt(L**2 + eps_L**2))
    C_rad = 3.0 * kappa * P / (16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G * m * T**4)
    dgrad_rad_dL = C_rad * dLsafe_dL
    dgrad_rad_dP = grad_rad * ((dkappa_drho / kappa) * drho_dP + 1.0 / P)
    dgrad_rad_dT = grad_rad * ((dkappa_drho / kappa) * drho_dT + dkappa_dT / kappa - 4.0 / T)
    return dgrad_rad_dL, dgrad_rad_dP, dgrad_rad_dT


def _effective_gradient_derivative(grad_rad, grad_ad):
    """d(grad_eff)/d(grad_rad) for grad_eff=min_smooth(grad_rad,grad_ad)
    (gradients.effective_gradient), differentiated from the simple form
    grad_eff=0.5*(a+b)-0.5*sqrt((a-b)^2+eps^2). grad_ad is a constant (eos.grad_adiabatic),
    so this is the only channel."""
    eps_s = config.GRAD_EFF_SWITCH_EPSILON
    diff = grad_rad - grad_ad
    return 0.5 * (1.0 - diff / np.sqrt(diff**2 + eps_s**2))


def implicit_rhs_jacobian(x, y, state_prev, dt, alpha):
    """d(dy/dx)/dy, shape (4,4,n) - solve_bvp's fun_jac contract, physical-space (y=[r,lnP,
    L,lnT]). Mirrors implicit_rhs_vectorized's physics exactly. Derived by hand; cross-checked
    against finite differences before use (validation.py's Jacobian-correctness check)."""
    m = np.exp(x)
    r, lnP, L, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    n = len(np.atleast_1d(x))

    rho = eos.density(P, T, config.MU, config.MU_E)
    drho_dP, drho_dT = _eos_density_derivatives(P, T, rho)

    dP_dm = -config.G * m / (4.0 * np.pi * r**4)   # f1 numerator, before dividing by P
    f0 = 1.0 / (4.0 * np.pi * r**2 * rho)           # dr_dm
    f1 = dP_dm / P                                   # dlnP_dm

    kappa = opacity.bell_lin_opacity(rho, T)
    dkappa_drho, dkappa_dT = _opacity_derivatives(rho, T, kappa)
    grad_ad = eos.grad_adiabatic(config.GAMMA)
    grad_rad = gradients.grad_radiative(L, m, P, T, kappa)
    dgrad_rad_dL, dgrad_rad_dP, dgrad_rad_dT = _grad_radiative_derivatives(
        L, m, P, T, kappa, rho, drho_dP, drho_dT, dkappa_drho, dkappa_dT, grad_rad)

    J = np.zeros((4, 4, n))

    # Row 0: f0 = dr_dm, depends on r (explicit) and rho(P,T)
    J[0, 0] = -2.0 * f0 / r
    J[0, 1] = -P * f0 / rho * drho_dP                 # d/d(lnP) = P*d/dP
    J[0, 2] = 0.0
    J[0, 3] = -T * f0 / rho * drho_dT                 # d/d(lnT) = T*d/dT

    # Row 1: f1 = dlnP_dm, depends on r (via dP_dm) and P (via the /P) only
    J[1, 0] = -4.0 * f1 / r
    J[1, 1] = -f1                                      # d(dP_dm/P)/d(lnP) = -f1
    J[1, 2] = 0.0
    J[1, 3] = 0.0

    # Row 2: f2 = dL_dm = -c_p*dT_dt + delta*dP_dt/rho, depends on P, T only (not r, not L
    # itself). delta is the genuine EOS-dependent coefficient (eos.thermodynamic_delta),
    # PLAN_BVP.md Milestone 6.
    T_prev = np.interp(m, state_prev.m, state_prev.T)
    P_prev = np.interp(m, state_prev.m, state_prev.P)
    dP_dt = (P - P_prev) / dt
    delta = eos.thermodynamic_delta(rho, T, config.MU, config.MU_E)
    ddelta_dP, ddelta_dT = _thermodynamic_delta_derivatives(rho, T, drho_dP, drho_dT, delta)
    df2_dP = delta / (dt * rho) + dP_dt * ddelta_dP / rho - delta * dP_dt / rho**2 * drho_dP
    c_p = eos.specific_heat_cp(config.GAMMA, config.MU)
    df2_dT = -c_p / dt + dP_dt * ddelta_dT / rho - delta * dP_dt / rho**2 * drho_dT
    J[2, 0] = 0.0
    J[2, 1] = P * df2_dP
    J[2, 2] = 0.0
    J[2, 3] = T * df2_dT

    # Row 3: f3 = dlnT_dm = G_blend*f1, G_blend=(1-alpha)*grad_ad + alpha*grad_eff -
    # dlnT/dm = grad_eff*dlnP/dm identically, so f3 factors through f1 exactly.
    if alpha == 0.0:
        G_blend = np.full(n, grad_ad)
        dGblend_dP = np.zeros(n)
        dGblend_dL = np.zeros(n)
        dGblend_dT = np.zeros(n)
    else:
        grad_eff = gradients.effective_gradient(grad_rad, grad_ad)[0]
        dgeff_dgrad = _effective_gradient_derivative(grad_rad, grad_ad)
        dGblend_dL = alpha * dgeff_dgrad * dgrad_rad_dL
        dGblend_dP = alpha * dgeff_dgrad * dgrad_rad_dP
        dGblend_dT = alpha * dgeff_dgrad * dgrad_rad_dT
        G_blend = (1.0 - alpha) * grad_ad + alpha * grad_eff
    f3 = G_blend * f1
    J[3, 0] = G_blend * J[1, 0]                                    # via f1's r-dependence only
    J[3, 1] = (P * dGblend_dP) * f1 + G_blend * J[1, 1]            # explicit G_blend(P) + f1(lnP) terms
    J[3, 2] = dGblend_dL * f1                                       # f1 has no L-dependence
    J[3, 3] = (T * dGblend_dT) * f1                                 # f1 has no T-dependence

    # Chain rule: d/dx = m * d/dm, so the WHOLE Jacobian (of m*f, not just f) scales by m.
    return J * m


# ==========================================
# SECTION: Nondimensionalized RHS and Jacobian (t>0, the ones solve_bvp actually uses)
# ==========================================

def implicit_rhs_scaled(x, z, state_prev, dt, alpha):
    """dz/dx for the scaled state z=[r_hat,lnP,L_hat,lnT] - converts to physical, calls the
    physical RHS, then applies the OUTPUT-side chain-rule scaling
    (Phi'=diag(1/R_SCALE, 1, 1/sqrt(L^2+L_SCALE^2), 1))."""
    y = _to_physical(z)
    f = implicit_rhs_vectorized(x, y, state_prev, dt, alpha)
    L = y[2]
    g = np.empty_like(f)
    g[0] = f[0] / R_SCALE
    g[1] = f[1]
    g[2] = f[2] / np.sqrt(L**2 + L_SCALE**2)
    g[3] = f[3]
    return g


def implicit_rhs_jacobian_scaled(x, z, state_prev, dt, alpha):
    """d(dz/dx)/dz, shape (4,4,n) - the scaled-state counterpart of implicit_rhs_jacobian.

    J_new[i,j] = row_scale[i] * J_old[i,j] * col_scale[j] + (extra term, row=col=2 only).

    The extra term: because L_hat's own scaling factor Phi'_2(L)=1/sqrt(L^2+L_SCALE^2) is
    itself L-dependent (nonlinear, unlike r_hat's constant 1/R_SCALE), differentiating
    g=Phi'(y)*f(y) a second time picks up a genuine product-rule term from d(Phi')/dy, not
    just the rescaled df/dy: d(Phi'_2)/dL * dL/d(L_hat) = -L/(L^2+L_SCALE^2) exactly - present
    ONLY in the (L_hat row, L_hat column) entry.
    """
    y = _to_physical(z)
    L = y[2]
    f = implicit_rhs_vectorized(x, y, state_prev, dt, alpha)      # physical RHS - needed for the row-2 correction term
    J_old = implicit_rhs_jacobian(x, y, state_prev, dt, alpha)    # (4,4,n), physical-space

    n = J_old.shape[2]
    col_scale = np.array([
        np.full(n, R_SCALE),
        np.ones(n),
        np.sqrt(L**2 + L_SCALE**2),   # dL/d(L_hat) = L_SCALE*cosh(L_hat)
        np.ones(n),
    ])
    row_scale = np.array([
        np.full(n, 1.0 / R_SCALE),
        np.ones(n),
        1.0 / np.sqrt(L**2 + L_SCALE**2),
        np.ones(n),
    ])

    J_new = J_old * row_scale[:, np.newaxis, :] * col_scale[np.newaxis, :, :]
    correction = -L / (L**2 + L_SCALE**2) * f[2]
    J_new[2, 2] += correction
    return J_new


# ==========================================
# SECTION: Boundary Conditions (t>0, scaled state - reuses boundary_conditions.py's
# photospheric_pressure but NOT its packaged boundary_conditions(), which assumes the
# shooting convention of ya=zeros - see make_bc_scaled's own note)
# ==========================================

def make_bc_scaled(m_min):
    """Builds the bc(za, zb) closure for solve_bvp, given m_min = the innermost mesh mass.
    za, zb are SCALED state vectors [r_hat,lnP,L_hat,lnT]; residuals are returned in SCALED
    units throughout (leaving the thermal residual in raw erg/s while everything else is
    O(1)-scaled would reintroduce the scale mismatch this whole nondimensionalization exists
    to remove).

    Center condition: solve_bvp treats za as a genuine solved-for unknown (unlike shooting,
    which always constructs r=r_seed>0, L=0 directly and never checks a residual for it) -
    calling boundary_conditions.boundary_conditions() with za=zeros would force r(m_min)=0
    EXACTLY, a true 1/r^2 singularity in dr_dm at solve_bvp's own m_min mesh point. Instead,
    r(m_min) is tied to the LIVE trial center density via the analytic constant-density-center
    relation r(m)=(3m/(4*pi*rho_c))^(1/3), re-evaluated from (P_a,T_a) at every Newton
    iteration (PLAN_BVP.md Milestone 2) - not a fixed pre-estimate.
    """
    def bc(za, zb):
        y_a, y_b = _to_physical(za), _to_physical(zb)
        r_a, P_a, L_a, T_a = y_a[0], np.exp(y_a[1]), y_a[2], np.exp(y_a[3])
        r_b, P_b, L_b, T_b = y_b[0], np.exp(y_b[1]), y_b[2], np.exp(y_b[3])

        res = np.zeros(4)
        rho_c = eos.density(P_a, T_a, config.MU, config.MU_E)
        # Analytic constant-density-center relation (PLAN_BVP.md Milestone 2):
        # r(m_min) = (3*m_min/(4*pi*rho_c))^(1/3)   [cm]
        r_analytic = (3.0 * m_min / (4.0 * np.pi * rho_c)) ** (1.0 / 3.0)
        res[0] = za[0] - r_analytic / R_SCALE   # r_hat(m_min) = r_analytic/R_SCALE
        res[1] = za[2]                            # L_hat(m_min) = arcsinh(0/L_SCALE) = 0 (no nuclear source)

        # Mechanical (photospheric) residual, log space - consistent with P,T already being
        # log-transformed throughout (PLAN_BVP.md Milestone 3): a linear P_b-P_photo residual
        # sits a ~1e11 dyn/cm^2-scale term in the same vector as center residuals near
        # machine-zero, a scale mismatch the log form removes. zb[1] is already ln(P_b).
        P_photo = boundary_conditions.photospheric_pressure(r_b, P_b, T_b, config.MU, config.MU_E)
        res[2] = zb[1] - np.log(P_photo)

        # Thermal (net-flux radiative) residual, in the SAME arcsinh units as the state
        # vector's own L_hat.
        L_expected = 4.0 * np.pi * r_b**2 * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        res[3] = zb[2] - np.arcsinh(L_expected / L_SCALE)
        return res
    return bc


def make_bc_jacobian_scaled(m_min):
    """Scaled-state counterpart of make_bc_scaled - dbc/d(za), dbc/d(zb), each (4,4). Derived
    by direct differentiation of make_bc_scaled's residuals."""
    def bc_jac(za, zb):
        y_a, y_b = _to_physical(za), _to_physical(zb)
        r_a, P_a, L_a, T_a = y_a[0], np.exp(y_a[1]), y_a[2], np.exp(y_a[3])
        r_b, P_b, L_b, T_b = y_b[0], np.exp(y_b[1]), y_b[2], np.exp(y_b[3])

        rho_a = eos.density(P_a, T_a, config.MU, config.MU_E)
        drho_dP_a, drho_dT_a = _eos_density_derivatives(P_a, T_a, rho_a)
        r_analytic = (3.0 * m_min / (4.0 * np.pi * rho_a)) ** (1.0 / 3.0)
        dr_analytic_drho = -r_analytic / (3.0 * rho_a)

        dbc_dza = np.zeros((4, 4))
        dbc_dza[0, 0] = 1.0
        dbc_dza[0, 1] = -(dr_analytic_drho * drho_dP_a * P_a) / R_SCALE
        dbc_dza[0, 3] = -(dr_analytic_drho * drho_dT_a * T_a) / R_SCALE
        dbc_dza[1, 2] = 1.0

        rho_b = eos.density(P_b, T_b, config.MU, config.MU_E)
        drho_dP_b, drho_dT_b = _eos_density_derivatives(P_b, T_b, rho_b)
        kappa_b = opacity.bell_lin_opacity(rho_b, T_b)
        dkappa_drho_b, dkappa_dT_b = _opacity_derivatives(rho_b, T_b, kappa_b)
        P_photo = boundary_conditions.photospheric_pressure(r_b, P_b, T_b, config.MU, config.MU_E)

        dPphoto_dr = -2.0 * P_photo / r_b
        dPphoto_dP = -P_photo * (dkappa_drho_b / kappa_b) * drho_dP_b
        dPphoto_dT = -P_photo * ((dkappa_drho_b / kappa_b) * drho_dT_b + dkappa_dT_b / kappa_b)

        L_expected = 4.0 * np.pi * r_b**2 * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        dLexp_dr = 8.0 * np.pi * r_b * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        dLexp_dT = 16.0 * np.pi * r_b**2 * config.SIGMA_SB * T_b**3   # plain d(L_expected)/dT_b - see the chain-rule note below
        d_arcsinh = 1.0 / np.sqrt(L_expected**2 + L_SCALE**2)   # d(arcsinh(L_expected/L_SCALE))/d(L_expected)

        dbc_dzb = np.zeros((4, 4))
        dbc_dzb[2, 0] = (-dPphoto_dr / P_photo) * R_SCALE                # d/d(r_hat_b) = R_SCALE * d/d(r_b)
        dbc_dzb[2, 1] = 1.0 - (dPphoto_dP / P_photo) * P_b
        dbc_dzb[2, 3] = -(dPphoto_dT / P_photo) * T_b
        dbc_dzb[3, 0] = -d_arcsinh * dLexp_dr * R_SCALE                  # d/d(r_hat_b) = R_SCALE * d/d(r_b)
        dbc_dzb[3, 2] = 1.0
        dbc_dzb[3, 3] = -d_arcsinh * dLexp_dT * T_b                      # d/d(lnT_b) = T_b * d/d(T_b), chain rule applied once here

        return dbc_dza, dbc_dzb
    return bc_jac


# ==========================================
# SECTION: Mesh and Initial Guess (t>0)
# ==========================================

def _build_mesh_and_guess(guess_state, warm_start_L):
    """Composite log-mass mesh (_build_output_grid, reused, with a solve_bvp-specific point
    count - config.BVP_MESH_N_GRID_POINTS) and an initial y guess interpolated from
    guess_state's own (r,P,T) profile - spanning x=[ln(m_min), ln(M_TOTAL)] exactly, since
    under solve_bvp the domain is fixed and known (unlike shooting's event-determined
    surface).

    warm_start_L distinguishes the two callers' very different guess_state.L reliability:
    - relax_initial_state (warm_start_L=False): guess_state is solve_static_structure's raw
      output, whose L is the diagnostic-only marginal-convection closure - confirmed
      2026-08-06 to be a poor seed for the real, Schwarzschild-selected L(m) (surface
      thermal BC residual ~5e24 at the initial guess). A simple monotonic ramp toward the
      KH-timescale luminosity scale is used instead.
    - solve_timestep (warm_start_L=True): guess_state is itself a GENUINE, already-converged
      solve_bvp solution (from a previous relax_initial_state/solve_timestep call) - its own
      L(m) IS the physically meaningful profile close to the next dt's answer (found
      2026-08-08, PROGRESS.md: reusing the KH-ramp guess here instead - which can be 10+
      orders of magnitude off in both scale AND sign from the true, already-relaxed L, e.g.
      ~+1e29 ramped vs ~-1e23 converged - fed solve_bvp's very first midpoint collocation
      evaluation an unphysical (P,T) trial that crashed eos.density before any Newton
      refinement even began).
    """
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    n_grid_points_orig = config.N_GRID_POINTS
    outer_refinement_orig = config.GRID_OUTER_REFINEMENT
    config.N_GRID_POINTS = config.BVP_MESH_N_GRID_POINTS   # runtime-only override, restored below
    # ASSUMPTION: the deeper BVP_MESH_OUTER_REFINEMENT only helps (and is only needed) for
    # warm_start_L=True (solve_timestep, warm-starting from an already-CONVERGED, full-domain
    # state whose own profile is genuinely steep in the final micro-interval - config.py's own
    # comment). relax_initial_state's guess_state (state_0, the adiabatic seed) never reaches
    # exactly m=M_TOTAL in the first place (np.interp's boundary clamping already avoids the
    # problem there - see the warm_start_L docstring note above), so applying the deeper
    # refinement unconditionally was tried and confirmed 2026-08-08 to be actively HARMFUL for
    # relax_initial_state specifically: it inflates the initial guess mesh enough that solve_
    # bvp's node count balloons across the alpha-continuation ladder and exceeds config.
    # BVP_MAX_NODES before reaching alpha=1, where the original (coarser) mesh converged
    # cleanly. Scoped to warm_start_L only, not a blanket change.
    if warm_start_L:
        config.GRID_OUTER_REFINEMENT = config.BVP_MESH_OUTER_REFINEMENT
    try:
        m_grid = _build_output_grid(m_min, config.M_TOTAL)
    finally:
        config.N_GRID_POINTS = n_grid_points_orig
        config.GRID_OUTER_REFINEMENT = outer_refinement_orig
    x = np.log(m_grid)

    r_guess = np.interp(m_grid, guess_state.m, guess_state.r)
    # ASSUMPTION: interpolate ln(P), ln(T) directly, NOT P, T then log() afterward - found
    # 2026-08-08 (PROGRESS.md) while diagnosing a solve_timestep-only crash: P, T drop by
    # ~3 orders of magnitude over the final ~0.001% of mass near the photosphere (a genuine,
    # extremely steep physical transition, not a numerical artifact), and guess_state's own
    # profile is stored on only config.N_GRID_POINTS=200 output points - far coarser than
    # this new BVP_MESH_N_GRID_POINTS=2000 target mesh. Linearly interpolating P, T (then
    # logging the result) approximates that decade-spanning drop with straight line segments
    # in LINEAR space, producing a badly wrong lnP/lnT guess exactly where the profile is
    # steepest; interpolating ln(P), ln(T) directly (the actual state variables the solver
    # works in) tracks the true profile far better, since P, T are much closer to log-linear
    # in m through the outer radiative layers. This was silently masked for
    # relax_initial_state (guess_state=state_0): solve_static_structure's shooting-event
    # surface never reaches exactly m=M_TOTAL, so np.interp's boundary-clamping quietly
    # avoided ever extrapolating across the worst part of this transition - solve_timestep's
    # guess_state (an already-converged, full-domain solve_bvp state) is the first case that
    # actually exercises it.
    lnP_guess = np.interp(m_grid, guess_state.m, np.log(guess_state.P))
    lnT_guess = np.interp(m_grid, guess_state.m, np.log(guess_state.T))
    if warm_start_L:
        L_guess = np.interp(m_grid, guess_state.m, guess_state.L)
    else:
        L_scale_guess = config.G * config.M_TOTAL**2 / (guess_state.r[-1] * config.T_KH_TIMESCALE_S)
        L_guess = L_scale_guess * (m_grid / config.M_TOTAL)
    y_guess = np.array([r_guess, lnP_guess, L_guess, lnT_guess])
    return x, y_guess


def build_mesh_and_guess_scaled(guess_state, warm_start_L):
    """Scaled-state counterpart of _build_mesh_and_guess - same mesh/guess, converted to
    z=[r_hat,lnP,L_hat,lnT] via _to_scaled."""
    x, y_guess = _build_mesh_and_guess(guess_state, warm_start_L)
    return x, _to_scaled(y_guess)


# ==========================================
# SECTION: Solve Orchestration (t>0) — direct attempt, continuation fallback, crash-safety
# ==========================================

class _CrashedSolve:
    """Stand-in for scipy's solve_bvp OptimizeResult when fun() raises instead of solve_bvp
    itself returning a failed status (e.g. eos.density's own Newton-convergence assertion,
    triggered by solve_bvp's internal Newton step overshooting into an unphysical (P,T)
    region) - lets the SAME fallback logic trigger either way, without silently swallowing
    the crash: the full exception is printed first, never hidden."""
    def __init__(self, x, y, exc):
        self.status = -99
        self.message = f"CRASHED: {type(exc).__name__}: {exc}"
        self.x = x
        self.y = y


def _safe_solve_bvp(fun, bc, x, y, fun_jac, bc_jac):
    import traceback
    try:
        return solve_bvp(fun, bc, x, y, tol=config.BVP_COLLOCATION_TOL, max_nodes=config.BVP_MAX_NODES,
                          verbose=2, fun_jac=fun_jac, bc_jac=bc_jac)
    except Exception as exc:
        print("bvp_solver: *** solve_bvp raised during Newton iteration (not a clean failed "
              "status) - full traceback: ***")
        traceback.print_exc()
        return _CrashedSolve(x, y, exc)


def _smoke_test_vectorization(state_prev, dt, bc, x, y_guess):
    """Cheap sanity check that the RHS/bc closures behave correctly on a small synthetic
    multi-point mesh BEFORE spending time on a real solve_bvp call - catches a shape-mismatch
    bug immediately rather than deep inside a slow, hard-to-diagnose run."""
    x_small, y_small = x[:5], y_guess[:, :5]
    dzdx = implicit_rhs_scaled(x_small, y_small, state_prev, dt, 1.0)
    assert dzdx.shape == y_small.shape, f"RHS shape mismatch: dzdx {dzdx.shape} vs z {y_small.shape}"
    assert np.all(np.isfinite(dzdx)), "RHS produced non-finite values on the smoke-test mesh"
    res = np.asarray(bc(y_guess[:, 0], y_guess[:, -1]))
    assert res.shape == (4,), f"bc() shape mismatch: {res.shape}"
    assert np.all(np.isfinite(res)), "bc() produced non-finite residuals"


def _attempt_direct_solve(state_prev, dt, bc, bc_jac, x, y_guess, alpha):
    def fun(x_, y_):
        return implicit_rhs_scaled(x_, y_, state_prev, dt, alpha)
    def fun_jac(x_, y_):
        return implicit_rhs_jacobian_scaled(x_, y_, state_prev, dt, alpha)
    t0 = time.time()
    sol = _safe_solve_bvp(fun, bc, x, y_guess, fun_jac, bc_jac)
    return sol, time.time() - t0


def _attempt_continuation_solve(state_prev, dt, bc, bc_jac, x, y_guess, alpha_steps):
    """Fallback if the direct alpha=1 solve fails: step alpha 0->1, warm-starting each
    solve_bvp call from the previous alpha's dense solution.

    config.BVP_ALPHA_MAX, not exactly 1.0 (PLAN_BVP.md Milestone 6): continuation converges
    cleanly through alpha=0.9999 but the LITERAL alpha=1.0 diverges via exponentially
    escalating mesh refinement to NaN - the tiny adiabatic admixture at BVP_ALPHA_MAX acts as
    a regularizer, damping a marginal instability in the pure unblended system.
    """
    x_curr, y_curr = x, y_guess
    total_elapsed = 0.0
    sol = None
    for alpha in alpha_steps:
        def fun(x_, y_, alpha=alpha):
            return implicit_rhs_scaled(x_, y_, state_prev, dt, alpha)
        def fun_jac(x_, y_, alpha=alpha):
            return implicit_rhs_jacobian_scaled(x_, y_, state_prev, dt, alpha)
        t0 = time.time()
        sol = _safe_solve_bvp(fun, bc, x_curr, y_curr, fun_jac, bc_jac)
        elapsed = time.time() - t0
        total_elapsed += elapsed
        print(f"bvp_solver: continuation alpha={alpha:.5f}: status={sol.status}, "
              f"message={sol.message}, nodes={sol.x.size}, elapsed={elapsed:.1f}s", flush=True)
        if sol.status != 0:
            return sol, total_elapsed
        x_curr, y_curr = sol.x, sol.y
    return sol, total_elapsed


def _solve_structure_bvp(state_prev, dt, warm_start_L):
    """Shared solve_bvp orchestration for both relax_initial_state (dt=pseudo-relaxation
    timestep) and solve_timestep (dt=real elapsed time) - both warm-start the mesh/initial
    guess from state_prev's own profile (matching the old shooting code's warm-start
    convention: relax_initial_state seeded from state_0 itself, solve_timestep from the
    actual previous converged state). warm_start_L: see _build_mesh_and_guess's docstring -
    False for relax_initial_state (state_0.L is diagnostic-only), True for solve_timestep
    (state_prev.L is a genuine previously-converged solution).

    Attempts a direct alpha=1 solve first (cheap, matches the old shooting solve_timestep's
    behavior of not re-relaxing every step); falls back to the config.BVP_ALPHA_CONTINUATION_
    STEPS ladder only if that fails - bvp_experiment.py's proven strategy (PLAN_BVP.md
    Milestone 6).

    Returns (sol, m_min) - the raw solve_bvp OptimizeResult (scaled-state solution).
    """
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    bc = make_bc_scaled(m_min)
    bc_jac = make_bc_jacobian_scaled(m_min)
    x, y_guess = build_mesh_and_guess_scaled(state_prev, warm_start_L)
    _smoke_test_vectorization(state_prev, dt, bc, x, y_guess)

    print("bvp_solver: attempting direct solve_bvp at alpha=1.0 ...", flush=True)
    sol, elapsed = _attempt_direct_solve(state_prev, dt, bc, bc_jac, x, y_guess, alpha=1.0)
    if sol.status != 0:
        print(f"bvp_solver: direct solve_bvp did not converge (status={sol.status}, "
              f"{sol.message}) after {elapsed:.1f}s - falling back to alpha-continuation", flush=True)
        sol, elapsed = _attempt_continuation_solve(state_prev, dt, bc, bc_jac, x, y_guess,
                                                    config.BVP_ALPHA_CONTINUATION_STEPS)

    if sol.status != 0:
        raise RuntimeError(
            f"bvp_solver: solve_bvp failed to converge even via alpha-continuation "
            f"(status={sol.status}, {sol.message}) after {elapsed:.1f}s - not a genuine solution"
        )
    return sol, m_min


def _bvp_solution_to_state(sol, m_min, state_prev, t) -> state.SimulationState:
    """Converts a converged scaled-state solve_bvp solution into a SimulationState on the
    standard composite output grid. solve_bvp's domain [m_min, M_TOTAL] is FIXED and known
    exactly (unlike shooting's event-determined surface) - a structural simplification of
    this pivot, not an approximation."""
    m = _build_output_grid(m_min, config.M_TOTAL)
    z = sol.sol(np.log(m))
    r, lnP, L, lnT = _to_physical(z)
    P, T = np.exp(lnP), np.exp(lnT)
    rho = eos.density(P, T, config.MU, config.MU_E)

    print(f"bvp_solver: t>0 solve_bvp converged, t={t:.4e} s, nodes={sol.x.size}, "
          f"P_center={P[0]:.6e} dyn/cm^2, T_center={T[0]:.6e} K, "
          f"r_surface={r[-1]/config.R_JUPITER_CM:.4f} R_Jup, L_surface={L[-1]:.4e} erg/s")

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=t, prev=state_prev)


# ==========================================
# SECTION: Public t>0 Solves (relax_initial_state, solve_timestep)
# ==========================================

def relax_initial_state(state_0) -> state.SimulationState:
    """Relax state_0 (solve_static_structure's output - built by forcing the pure ideal-gas
    adiabat, not a genuine solution of the real 4-ODE system's Schwarzschild-selected
    temperature gradient) into a state that IS self-consistent with the same implicit
    equations solve_timestep() uses, via solve_bvp at a small pseudo-timestep
    (config.RELAX_DT_FRACTION*T_KH_TIMESCALE_S - NOT real elapsed time, t is left unchanged).

    2026-08-08: re-platformed from shooting (bvp_solver_shooting_archive.py) onto solve_bvp
    collocation (PLAN_BVP.md Milestone 6) - same physical role, different numerical method.
    """
    dt_relax = config.RELAX_DT_FRACTION * config.T_KH_TIMESCALE_S
    sol, m_min = _solve_structure_bvp(state_0, dt_relax, warm_start_L=False)
    return _bvp_solution_to_state(sol, m_min, state_0, t=state_0.t)


def solve_timestep(state_prev, dt) -> state.SimulationState:
    """Solve the envelope structure at t = state_prev.t + dt via solve_bvp collocation.

    2026-08-08: re-platformed from shooting (bvp_solver_shooting_archive.py) onto solve_bvp
    (PLAN_BVP.md Milestone 6) - same physical role (implicit Henyey-style time differencing,
    photospheric + net-flux-radiative surface conditions), different numerical method.
    """
    sol, m_min = _solve_structure_bvp(state_prev, dt, warm_start_L=True)
    return _bvp_solution_to_state(sol, m_min, state_prev, t=state_prev.t + dt)
