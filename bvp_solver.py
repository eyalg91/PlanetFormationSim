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
from scipy.special import expit

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


def _adiabatic_center_guess_ideal_gas(T_center):
    """Analytic (Lane-Emden) central pressure and radius bracket SEED for a DIFFUSE, degeneracy-
    negligible, ideal-gas-supported structure (Phase 1 / First Hydrostatic Core, PLAN.md
    "Formation Scenario and Scope") - the ideal-gas counterpart to _adiabatic_center_guess's
    degenerate-limit seed above, needed because that seed is a pure function of fundamental
    constants alone (independent of T_center, MU, GAMMA) and, verified directly, anchors
    solve_static_structure's geometric-expansion bracket search onto the WRONG physical root
    (the compact, degenerate branch) for a diffuse molecular composition - not a bracket
    failure, a SILENT wrong answer (PROGRESS.md 2026-08-12 has the full diagnosis).

    Polytrope scaling (Kippenhahn & Weigert Ch. 19), same alpha^2/R/M relations as the
    degenerate seed, but K is now THERMALLY set (K=C*T_center*rho_c^(-1/n), C=k_B/(mu*m_H) -
    the ideal gas law's own P=rho*k_B*T/(mu*m_H) at fixed T_center) rather than a rho_c-
    independent constant:
        alpha^2 = (n+1)*K*rho_c^(1/n-1)/(4*pi*G)
                = (n+1)*C*T_center*rho_c^(-1/n)*rho_c^(1/n-1)/(4*pi*G)
                = (n+1)*C*T_center/(4*pi*G) * rho_c^(-1)              [the rho_c exponents
                                                                        cancel EXACTLY for any n]
    Same closed-form M(rho_c) inversion as the degenerate case follows from this rho_c^(-1)
    collapse. Self-consistency check (verified analytically, not just asserted): substituting
    back gives P_center = C*T_center*rho_center exactly - the plain ideal gas law evaluated at
    the seed's own (rho_center, T_center), as it must.
    """
    n = 1.0 / (config.GAMMA - 1.0)   # n=1/(gamma-1): n=2.5 for GAMMA=1.4 (diatomic molecular H2/He)
    xi_1, mass_constant = _solve_lane_emden(n)

    C = config.K_B / (config.MU * config.M_H)   # ideal gas law constant: P = C*rho*T
    B_prime = (n + 1.0) * C * T_center / (4.0 * np.pi * config.G)
    rho_center = (4.0 * np.pi * mass_constant * B_prime**1.5 / config.M_TOTAL) ** 2
    P_center = C * T_center * rho_center   # = ideal gas law at (rho_center, T_center) - consistency check
    R = np.sqrt(B_prime / rho_center) * xi_1
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

def solve_static_structure(use_ideal_gas_seed=False) -> state.SimulationState:
    """Solve the t=0 compact-protoplanet structure and return a populated SimulationState.

    Shoots on the central pressure P_center (T_center fixed at config.T_CENTER_INITIAL - a
    prescribed "hot start" parameter, not a shooting unknown, matching how real hot-start models
    treat initial entropy as a chosen input) until the ENCLOSED MASS at the photosphere event
    (boundary_conditions.photospheric_pressure, Eddington tau=2/3) equals M_TOTAL - not a
    residual at a fixed m=M_TOTAL grid endpoint (module docstring explains why), bracketed by
    an analytic Lane-Emden estimate rather than a blind search.

    use_ideal_gas_seed (2026-08-12, Phase 1 / First Hydrostatic Core pivot - PROGRESS.md has
    the full diagnosis): selects _adiabatic_center_guess_ideal_gas (thermally-set K, tracks
    config.T_CENTER_INITIAL/MU/GAMMA) instead of the default _adiabatic_center_guess (a fixed
    T=0-degenerate-limit estimate, independent of all three). Defaults False so every EXISTING
    caller (Phase 3's compact, degeneracy-dominated hot start) is byte-for-byte unchanged -
    that seed remains correct and proven for the compact regime it was built for. Set True only
    for a diffuse, degeneracy-negligible structure (verified directly: the degenerate seed
    silently anchors the bracket search onto the wrong physical root there, not a bracket
    failure - a plausible-looking but wrong answer).
    """
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    # Generous margin past M_TOTAL for the photosphere event to trigger within - empirically the
    # event triggers within ~1.4x of M_TOTAL even for a substantially too-high P_center, so this
    # is cheap headroom, never expected to bind.
    x_span = (np.log(m_min), np.log(50.0 * config.M_TOTAL))

    if use_ideal_gas_seed:
        P_center_guess, R_guess = _adiabatic_center_guess_ideal_gas(config.T_CENTER_INITIAL)
    else:
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

    P_center = brentq(mass_error, P_low, P_high, xtol=config.STATIC_STRUCTURE_BRENTQ_XTOL, rtol=config.BVP_TOL)
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


def _interp_state_prev(m, state_prev):
    """(T_prev, P_prev) at mass m, interpolated from state_prev's own (coarser) grid - used
    by both implicit_rhs_vectorized and implicit_rhs_jacobian for the dT_dt=(T-T_prev)/dt,
    dP_dt=(P-P_prev)/dt source terms.

    2026-08-08 (PROGRESS.md): a log-space variant of this interpolation was tried as a
    candidate fix for the step-2 solve_timestep convergence failure (state_prev's own output
    grid is measurably too coarse near T_surface->T_NEB - up to ~1-2% error against the true
    dense solve_bvp interpolant). Confirmed via an isolated test NOT to fix step 2, and to
    make relax_initial_state itself measurably harder (52949 vs 21682 nodes) by shifting
    which nearby equally-valid solution the continuation converges to - reverted to plain
    linear interpolation pending the wide-epsilon Schwarzschild-switch investigation instead
    (the dominant cause: a genuine marginal-convection band, not an interpolation artifact).
    """
    T_prev = np.interp(m, state_prev.m, state_prev.T)
    P_prev = np.interp(m, state_prev.m, state_prev.P)
    return T_prev, P_prev


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

def _softplus(u):
    """log(1+exp(u)), the smooth analog of max(u,0) - computed via the standard numerically
    stable identity max(u,0)+log1p(exp(-|u|)) (avoids exp() overflow for large positive u;
    log1p(exp(-|u|))->0 smoothly for large |u| either sign, never itself overflows)."""
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))


def _soft_clamp(x, lo, hi, width):
    """Smooth two-sided saturation of x into (lo, hi): reproduces x to float64 precision many
    widths inside [lo,hi], asymptotes toward (never exactly reaches) lo/hi outside it. Built
    from two composed one-sided smooth saturations (smooth_max then smooth_min), each a
    scaled/shifted softplus - smooth_max(x,lo,w)=lo+w*softplus((x-lo)/w) -> x for x>>lo, -> lo
    for x<<lo; smooth_min analogously for the upper bound. C-infinity and strictly monotonic
    everywhere (see _soft_clamp_derivative) - replaces a hard np.clip specifically so a Newton
    trial that wanders outside [lo,hi] is always still pulled back, never released onto a flat
    plateau. config.BVP_SOFT_CLAMP_WIDTH's own comment has the full width-choice reasoning.
    """
    x_floored = lo + width * _softplus((x - lo) / width)
    return hi - width * _softplus((hi - x_floored) / width)


def _soft_clamp_derivative(x, lo, hi, width):
    """d(_soft_clamp)/dx - chain rule through the smooth_max -> smooth_min composition. Each
    stage's own derivative is d(softplus(u))/du = sigmoid(u) = expit(u), strictly in (0,1) for
    any finite u (underflows to exactly 0.0 only once u is beyond roughly -745, the float64
    exp() limit - config.BVP_SOFT_CLAMP_WIDTH's own comment quantifies how far past [lo,hi]
    that actually sits). The product of two such factors is itself strictly positive wherever
    both are - THIS strict positivity (never an exact structural zero, unlike np.clip's
    derivative) is the entire point of this construction: see config.py's comment on why a
    hard clamp's zero derivative broke the analytic Jacobian's consistency with its residual.
    """
    x_floored = lo + width * _softplus((x - lo) / width)
    d_floor = expit((x - lo) / width)          # d(smooth_max)/dx
    d_ceil = expit((hi - x_floored) / width)    # d(smooth_min)/d(x_floored)
    return d_ceil * d_floor


def _safe_exp_state(lnP, lnT):
    """P, T = exp(lnP), exp(lnT), with lnP/lnT passed through _soft_clamp first (asymmetric
    bounds - config.LN_P_CLAMP/LN_T_MIN/LN_T_MAX, same VALUES as the original hard clamp, now
    smoothly approached rather than hit at a wall).

    FIX (2026-08-10, Sub-task 8b full-run debugging): solve_bvp's own Newton iteration can
    transiently explore wildly unphysical trial (lnP, lnT) during early, undamped steps before
    converging - implicit_rhs_vectorized's own diagnostic print below already anticipated this
    ("before it potentially crashes eos.density downstream") but provided no actual safeguard,
    since it had never been triggered in practice until Sub-task 8b's mu(T)/gamma_eff(T)
    physics changed the solver's Newton trajectory enough to actually reach lnP=9194 (P=inf via
    exp overflow) during relax_initial_state's t=0 continuation ladder - a genuine, real crash
    (eos.density's Newton-Raphson assertion, unable to converge to P=inf), not a fluke.

    A first attempt clamped lnT symmetrically (down to exp(-100)~4e-44 K) and still crashed -
    NOT from overflow this time, but from eos.density's Newton-Raphson genuinely failing to
    converge at large-but-finite (P,T): at T->0, the ideal-gas-only seed rho=P*mu*m_H/(k_B*T)
    blows up, and once that seed's implied degenerate pressure (~rho^(5/3)) vastly exceeds the
    target P, Newton's correction step overshoots rather than converging (confirmed directly by
    sweeping the (P,T) grid - T>=~1K converges cleanly up to P=1e44 dyn/cm^2 at every P tested;
    T<~0.03K starts failing once P exceeds ~1e40). The T floor here (1K) sits safely above that
    empirical boundary while remaining far below any physically relevant T in this problem.

    SOFT-CLAMP (REVIEWED 2026-08-11, PI-directed architecture course correction - PROGRESS.md
    has the full report): the ORIGINAL hard np.clip prevented overflow but had EXACTLY ZERO
    derivative once saturated - both the reason no Newton correction could ever pull a wayward
    trial value back (no restoring gradient), and the reason implicit_rhs_jacobian/
    make_bc_jacobian_scaled's chain-rule factors (which multiply by the bare P, T value,
    implicitly assuming d(exp(lnX))/d(lnX)=1 always) went actively WRONG in the saturated
    region, not just imprecise - confirmed as the root cause of a step-4 mesh explosion (a
    center-point trial state collapsing to T_a=P_a=0, r_analytic~rho_c^-1/3 correspondingly
    diverging to ~4.2e40 cm, with a genuinely singular collocation Jacobian). _soft_clamp
    replaces the hard clip with a construction whose derivative is strictly nonzero far past
    the boundary (see _soft_clamp_derivative) - callers needing that derivative for the
    analytic Jacobian must use _safe_exp_state_derivatives below, NOT assume d(P)/d(lnP)=P.

    config.BVP_CLAMP_EXTREME_TRIAL_VALUES=False disables the clamp entirely (raw np.exp) - an
    internal escape hatch (config.py's own comment), not expected to be needed now that the
    analytic Jacobian is consistent with the clamp in every regime.
    """
    if not config.BVP_CLAMP_EXTREME_TRIAL_VALUES:
        return np.exp(lnP), np.exp(lnT)
    lnP_safe = _soft_clamp(lnP, -config.LN_P_CLAMP, config.LN_P_CLAMP, config.BVP_SOFT_CLAMP_WIDTH)
    lnT_safe = _soft_clamp(lnT, config.LN_T_MIN, config.LN_T_MAX, config.BVP_SOFT_CLAMP_WIDTH)
    return np.exp(lnP_safe), np.exp(lnT_safe)


def _safe_exp_state_derivatives(lnP, lnT):
    """dP/d(lnP), dT/d(lnT) for _safe_exp_state's ACTUAL (possibly saturating) output - needed
    everywhere implicit_rhs_jacobian/make_bc_jacobian_scaled differentiate through the clamp.

    d(exp(soft_clamp(x)))/dx = exp(soft_clamp(x)) * d(soft_clamp)/dx by the chain rule - NOT
    simply P or T themselves (that bare-P/T shortcut is only exact where the clamp's own
    derivative is identically 1, deep inside the safe range - see _safe_exp_state's own
    docstring for why silently assuming that everywhere broke the analytic Jacobian's
    consistency in the saturated region). Reduces to exactly (P, T) when config.
    BVP_CLAMP_EXTREME_TRIAL_VALUES is False (the unclamped identity's own derivative is 1).
    """
    P, T = _safe_exp_state(lnP, lnT)
    if not config.BVP_CLAMP_EXTREME_TRIAL_VALUES:
        return P, T
    dP_dlnP = P * _soft_clamp_derivative(lnP, -config.LN_P_CLAMP, config.LN_P_CLAMP, config.BVP_SOFT_CLAMP_WIDTH)
    dT_dlnT = T * _soft_clamp_derivative(lnT, config.LN_T_MIN, config.LN_T_MAX, config.BVP_SOFT_CLAMP_WIDTH)
    return dP_dlnP, dT_dlnT


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
    P, T = _safe_exp_state(lnP, lnT)
    # Diagnostic: report WHERE along the mesh a trial (P,T) needed clamping (_safe_exp_state)
    # to avoid crashing eos.density downstream - now survivable, still worth knowing about.
    bad = ~np.isfinite(P) | ~np.isfinite(T) | (P <= 0) | (T <= 0) | (np.abs(lnP) > 300) | (np.abs(lnT) > 300)
    if np.any(bad):
        idx = np.where(bad)[0]
        print(f"bvp_solver: [diag] extreme/non-finite (P,T) trial at {len(idx)} mesh point(s), "
              f"m/M_TOTAL in [{(m[idx]/config.M_TOTAL).min():.3e}, {(m[idx]/config.M_TOTAL).max():.3e}], "
              f"lnP range=[{np.nanmin(lnP[idx]):.3e},{np.nanmax(lnP[idx]):.3e}], "
              f"lnT range=[{np.nanmin(lnT[idx]):.3e},{np.nanmax(lnT[idx]):.3e}]", flush=True)
    T_prev, P_prev = _interp_state_prev(m, state_prev)
    dT_dt = (T - T_prev) / dt
    dP_dt = (P - P_prev) / dt
    y_full = np.array([r, P, L, T])   # shape (4, n) - odes.stellar_odes's native contract
    dr_dm, dP_dm, dL_dm, dT_dm_real = odes.stellar_odes(m, y_full, dT_dt, dP_dt)
    dlnP_dm = dP_dm / P

    # grad_ad=dlnT/dlnP by definition (eos.grad_adiabatic). Sub-task 8b: gamma_effective(T)
    # instead of the fixed config.GAMMA, so the adiabat itself softens through the H<->H2
    # transition (matches odes.stellar_odes's own gamma_T, used identically at alpha=1).
    dlnT_dm_ad = eos.grad_adiabatic(eos.gamma_effective(T)) * dlnP_dm

    # NaN-SAFE alpha blend (2026-08-12, Phase 1 pivot - PROGRESS.md has the full mechanism,
    # confirmed directly, not assumed): dT_dm_real needs grad_rad (hence opacity), computed
    # UNCONDITIONALLY above regardless of alpha - a genuinely extreme Newton TRIAL state
    # landing in a narrow/steep Bell & Lin regime crossing can make it non-finite. The
    # continuation ladder's whole premise is that alpha<1 makes the real-gradient term's
    # CONTRIBUTION vanish, but 0.0*nan=nan in IEEE float arithmetic - a zero weight does not
    # neutralize an already-corrupted value, so a pathological trial anywhere on the mesh could
    # corrupt the WHOLE blended result regardless of how small alpha was. The pure-adiabat
    # gradient (grad_ad(T) alone, no opacity dependence at all, hence never subject to this
    # failure mode) is used as the per-point fallback wherever the real gradient is not finite.
    dlnT_dm_real = dT_dm_real / T
    dlnT_dm_real_safe = np.where(np.isfinite(dlnT_dm_real), dlnT_dm_real, dlnT_dm_ad)
    if alpha == 1.0:
        dlnT_dm = dlnT_dm_real_safe
    else:
        dlnT_dm = (1.0 - alpha) * dlnT_dm_ad + alpha * dlnT_dm_real_safe
    # Chain rule: d/dx = m * d/dm, since x = ln(m)
    return np.array([dr_dm, dlnP_dm, dL_dm, dlnT_dm]) * m


def _h2_transition_derivatives(T):
    """Bundles every T-derivative of the H<->H2 recombination transition (Sub-task 8b) needed
    ANYWHERE in this file's analytic Jacobian - computed once per call site rather than
    re-deriving chi(T)'s derivatives piecemeal in each of the three places that need them.

    chi(T) = eos.molecular_fraction(T), a logistic - all higher derivatives below are the
    standard closed-form logistic-derivative identities (no numerical differentiation):
    dchi/dT = -chi*(1-chi)/W (eos.molecular_fraction_derivative), d2chi/dT2 =
    chi*(1-chi)*(1-2*chi)/W^2 (differentiate dchi/dT once more, product rule).
    """
    W = config.T_H2_TRANSITION_WIDTH
    chi = eos.molecular_fraction(T)
    dchi_dT = eos.molecular_fraction_derivative(T)
    d2chi_dT2 = chi * (1.0 - chi) * (1.0 - 2.0 * chi) / W**2

    mu = eos.mean_molecular_weight(T)
    d_inv_mu_dT = eos.mean_molecular_weight_inv_derivative(T)          # -(X/2)*dchi_dT
    d2_inv_mu_dT2 = -(config.X_HYDROGEN / 2.0) * d2chi_dT2              # chain rule, one more derivative

    gamma_eff = eos.gamma_effective(T)
    dgamma_eff_dT = (config.GAMMA_MOLECULAR - config.GAMMA) * dchi_dT

    return chi, dchi_dT, d2chi_dT2, mu, d_inv_mu_dT, d2_inv_mu_dT2, gamma_eff, dgamma_eff_dT


def _effective_heat_capacity_derivative(T, mu, d_inv_mu_dT, gamma_eff, dgamma_eff_dT, d2chi_dT2):
    """d(c_p_eff)/dT for c_p_eff = eos.specific_heat_cp(gamma_effective(T), mean_molecular_
    weight(T)) + eos.latent_heat_capacity(T) - the FULL, T-dependent effective heat capacity
    now used in the energy equation (odes.stellar_odes, Sub-task 8b). Needed by
    implicit_rhs_jacobian's row 2 (dL_dm depends on c_p_eff, which was a true T-independent
    constant before this sub-task, so this term did not exist previously).

    Two pieces, both from product/chain rule on c_p_frozen = K_B/M_H * [gamma/(gamma-1)] *
    (1/mu), gamma=gamma_eff(T), 1/mu=inv_mu(T):
      d(c_p_frozen)/dT = K_B/M_H * { -dgamma_dT/(gamma-1)^2 * (1/mu) + [gamma/(gamma-1)] * d_inv_mu_dT }
      d(latent_heat_capacity)/dT = -EPSILON_D_H2 * d2chi_dT2
    """
    inv_mu = 1.0 / mu
    d_cp_frozen_dT = config.K_B / config.M_H * (
        -dgamma_eff_dT / (gamma_eff - 1.0)**2 * inv_mu
        + (gamma_eff / (gamma_eff - 1.0)) * d_inv_mu_dT
    )
    d_latent_dT = -config.EPSILON_D_H2 * d2chi_dT2
    return d_cp_frozen_dT + d_latent_dT


def _eos_density_derivatives(P, T, rho, mu, d_inv_mu_dT=0.0):
    """d(rho)/dP, d(rho)/dT via IMPLICIT differentiation of the EOS's defining equation
    (NOT differentiating through eos.density's own Newton iteration): F(rho,P,T) =
    P_ideal(rho,T,mu(T)) + P_deg(rho) - P = 0.

    Standard implicit-function-theorem result: drho/dP = -(dF/dP)/(dF/drho),
    drho/dT = -(dF/dT)/(dF/drho). dF/drho is exactly the same quantity
    (dP_ideal_drho + dP_deg_drho) eos.density's own Newton loop already computes as its
    convergence-step denominator - reproduced here (eos.py exposes no derivative API by
    design, since gradients.py/odes.py never needed one before this solver).

    EXTENDED (Sub-task 8b): d_inv_mu_dT = d(1/mu)/dT (eos.mean_molecular_weight_inv_
    derivative) accounts for mu ITSELF varying with T. dF/dT now has an extra explicit-T
    channel through mu(T) on top of the plain T factor: dF/dT = -(P_ideal/T +
    rho*K_B*T*d_inv_mu_dT/M_H) - derived by product-rule differentiating
    P_ideal=rho*K_B*T/(mu(T)*M_H) w.r.t. T with mu=mu(T) (same M_H as dP_ideal_drho below).
    Reduces EXACTLY to the original formula when d_inv_mu_dT=0.
    """
    dP_ideal_drho = config.K_B * T / (mu * config.M_H)                     # dF/drho, ideal term
    P_deg = eos.degenerate_pressure(rho, config.MU_E)
    dP_deg_drho = (5.0 / 3.0) * P_deg / rho                                 # dF/drho, degenerate term
    D = dP_ideal_drho + dP_deg_drho                                        # dF/drho total
    drho_dP = 1.0 / D                                                      # dF/dP = -1
    P_ideal = rho * dP_ideal_drho
    drho_dT = -(P_ideal / T + rho * config.K_B * T * d_inv_mu_dT / config.M_H) / D   # dF/dT, extended
    return drho_dP, drho_dT


def _thermodynamic_delta_derivatives(rho, T, mu, drho_dP, drho_dT, delta, d_inv_mu_dT=0.0, d2_inv_mu_dT2=0.0):
    """d(delta)/dP, d(delta)/dT for eos.thermodynamic_delta's EXTENDED formula (Sub-task 8b:
    delta = N/(rho*D), N = P_ideal + rho*K_B*T^2*d_inv_mu_dT). Full re-derivation from the old
    formula's logarithmic-differentiation shortcut - the extra additive term in N breaks that
    shortcut's clean multiplicative structure, so this instead differentiates delta(rho,T)
    directly (partial derivatives, T-only channel first) then applies the SAME outer chain
    rule through rho(P,T) via drho_dP/drho_dT the old version used (quotient rule on N/(rho*D),
    matching the shape of the old formula's own construction).

    Reduces EXACTLY to the original formula's derivatives when d_inv_mu_dT=d2_inv_mu_dT2=0
    (both -> 0 as dP_deg_drho -> 0 too, pure ideal gas, delta=1 exactly - a true constant -
    matching eos.thermodynamic_delta's own limiting-case docstring, unchanged by this
    extension).
    """
    dP_ideal_drho = config.K_B * T / (mu * config.M_H)
    P_deg = eos.degenerate_pressure(rho, config.MU_E)
    dP_deg_drho = (5.0 / 3.0) * P_deg / rho
    D = dP_ideal_drho + dP_deg_drho
    P_ideal = rho * dP_ideal_drho
    N = P_ideal + rho * config.K_B * T**2 * d_inv_mu_dT / config.M_H   # delta's extended numerator (matches eos.thermodynamic_delta)

    # Partial derivatives at fixed T (rho only) - dP_ideal_drho has no rho-dependence, so N's
    # rho-derivative is just its two additive pieces' own rho-coefficients.
    dN_drho = dP_ideal_drho + config.K_B * T**2 * d_inv_mu_dT / config.M_H
    dD_drho = (2.0 / 3.0) * dP_deg_drho / rho   # P_deg ~ rho^(5/3): d(dP_deg_drho)/drho = (2/3)*dP_deg_drho/rho

    # Partial derivatives at fixed rho (T only) - P_deg/dP_deg_drho have no T-dependence.
    d_dPidrho_dT = config.K_B * mu**-1 / config.M_H + config.K_B * T * d_inv_mu_dT / config.M_H   # d(dP_ideal_drho)/dT
    dN_dT = rho * d_dPidrho_dT + rho * config.K_B * (2.0 * T * d_inv_mu_dT + T**2 * d2_inv_mu_dT2) / config.M_H
    dD_dT = d_dPidrho_dT   # dP_deg_drho has no T-dependence

    # delta = N/(rho*D): quotient rule, partial derivatives first...
    ddelta_drho_partial = dN_drho / (rho * D) - delta * (1.0 / rho + dD_drho / D)
    ddelta_dT_partial = dN_dT / (rho * D) - delta * dD_dT / D

    # ...then the outer chain rule through rho(P,T), matching the old formula's own convention.
    ddelta_dP = ddelta_drho_partial * drho_dP
    ddelta_dT = ddelta_drho_partial * drho_dT + ddelta_dT_partial
    return ddelta_dP, ddelta_dT


def _opacity_derivatives(rho, T, kappa):
    """d(kappa)/d(rho), d(kappa)/dT, dispatched to match whichever kappa(rho,T)
    opacity.bell_lin_opacity itself actually evaluated (config.OPACITY_SMOOTH_TRANSITIONS).

    FIX (2026-08-11, soft-clamp course correction - PROGRESS.md has the full report): this
    function unconditionally computed the HARD-switch regime's own power-law derivative
    (kappa=kappa_i*rho^a*T^b: d/d(rho)=a*kappa/rho, d/dT=b*kappa/T - exact almost everywhere,
    undefined only exactly AT a transition) even when the RESIDUAL (via bell_lin_opacity)
    used the SMOOTHED blend instead - the same class of Jacobian/residual mismatch as the P/T
    soft clamp, just narrower in reach (only matters within a few smoothing widths of a
    transition). opacity.bell_lin_opacity_smooth_derivatives properly differentiates the
    smoothed blend when the flag is on; verified against finite differences (validation.py
    Check 41) before being trusted here.
    """
    if config.OPACITY_SMOOTH_TRANSITIONS:
        return opacity.bell_lin_opacity_smooth_derivatives(rho, T)
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
    """(d(grad_eff)/d(grad_rad), d(grad_eff)/d(grad_ad)) for grad_eff=min_smooth(grad_rad,
    grad_ad) (gradients.effective_gradient), both differentiated from the simple form
    grad_eff=0.5*(a+b)-0.5*sqrt((a-b)^2+eps^2), a=grad_rad, b=grad_ad.

    EXTENDED (Sub-task 8b): grad_ad is no longer a true constant (eos.grad_adiabatic now
    takes gamma_effective(T)), so the d/d(grad_ad) channel - previously always exactly zero
    and not computed - is now needed too. Sanity identity: the two returned values sum to
    EXACTLY 1.0 everywhere (grad_eff is a smooth minimum of its two arguments, so its total
    differential is always a convex combination of them) - a useful invariant to check
    against if this is ever touched again.
    """
    eps_s = config.GRAD_EFF_SWITCH_EPSILON
    diff = grad_rad - grad_ad
    smooth = diff / np.sqrt(diff**2 + eps_s**2)
    dgeff_dgrad_rad = 0.5 * (1.0 - smooth)
    dgeff_dgrad_ad = 0.5 * (1.0 + smooth)
    return dgeff_dgrad_rad, dgeff_dgrad_ad


def implicit_rhs_jacobian(x, y, state_prev, dt, alpha):
    """d(dy/dx)/dy, shape (4,4,n) - solve_bvp's fun_jac contract, physical-space (y=[r,lnP,
    L,lnT]). Mirrors implicit_rhs_vectorized's physics exactly. Derived by hand; cross-checked
    against finite differences before use (validation.py's Jacobian-correctness check).

    dP_dlnP, dT_dlnT (_safe_exp_state_derivatives) are the ACTUAL d(P)/d(lnP), d(T)/d(lnT)
    through the soft clamp - every row below that converts a physical d/dP or d/dT into
    d/d(lnP) or d/d(lnT) must multiply by these, NOT the bare P/T value (that shortcut is
    only exact where the clamp's own derivative is identically 1, deep inside the safe range -
    config.BVP_SOFT_CLAMP_WIDTH's own comment has the full history of why silently assuming
    that everywhere broke the analytic Jacobian's consistency with the clamped residual).
    """
    m = np.exp(x)
    r, lnP, L, lnT = y
    P, T = _safe_exp_state(lnP, lnT)   # same clamp as implicit_rhs_vectorized - see _safe_exp_state's docstring
    dP_dlnP, dT_dlnT = _safe_exp_state_derivatives(lnP, lnT)
    n = len(np.atleast_1d(x))

    # H<->H2 recombination equilibrium (Sub-task 8b): mu, gamma both now vary with T, so their
    # OWN T-derivatives feed several rows below that had NO T-dependence through these channels
    # under the old constant-mu/gamma physics (flagged at each new term below).
    _chi, _dchi_dT, d2chi_dT2, mu, d_inv_mu_dT, d2_inv_mu_dT2, gamma_T, dgamma_eff_dT = \
        _h2_transition_derivatives(T)

    rho = eos.density(P, T, mu, config.MU_E)
    drho_dP, drho_dT = _eos_density_derivatives(P, T, rho, mu, d_inv_mu_dT)

    dP_dm = -config.G * m / (4.0 * np.pi * r**4)   # f1 numerator, before dividing by P
    f0 = 1.0 / (4.0 * np.pi * r**2 * rho)           # dr_dm
    f1 = dP_dm / P                                   # dlnP_dm

    kappa = opacity.bell_lin_opacity(rho, T)
    dkappa_drho, dkappa_dT = _opacity_derivatives(rho, T, kappa)
    grad_ad = eos.grad_adiabatic(gamma_T)
    dgrad_ad_dT = dgamma_eff_dT / gamma_T**2   # NEW (Sub-task 8b): grad_ad=(gamma-1)/gamma was a true constant before, d(grad_ad)/dgamma=1/gamma^2
    grad_rad = gradients.grad_radiative(L, m, P, T, kappa)
    dgrad_rad_dL, dgrad_rad_dP, dgrad_rad_dT = _grad_radiative_derivatives(
        L, m, P, T, kappa, rho, drho_dP, drho_dT, dkappa_drho, dkappa_dT, grad_rad)

    J = np.zeros((4, 4, n))

    # Row 0: f0 = dr_dm, depends on r (explicit) and rho(P,T)
    J[0, 0] = -2.0 * f0 / r
    J[0, 1] = -dP_dlnP * f0 / rho * drho_dP           # d/d(lnP) = dP_dlnP*d/dP
    J[0, 2] = 0.0
    J[0, 3] = -dT_dlnT * f0 / rho * drho_dT           # d/d(lnT) = dT_dlnT*d/dT

    # Row 1: f1 = dlnP_dm, depends on r (via dP_dm) and P (via the /P) only
    J[1, 0] = -4.0 * f1 / r
    J[1, 1] = -(f1 / P) * dP_dlnP                      # d(dP_dm/P)/d(lnP) = -(f1/P)*dP_dlnP (reduces to -f1 exactly when unclamped, dP_dlnP=P)
    J[1, 2] = 0.0
    J[1, 3] = 0.0

    # Row 2: f2 = dL_dm = -c_p_eff*dT_dt + delta*dP_dt/rho, depends on P, T only (not r, not L
    # itself). delta is the genuine EOS-dependent coefficient (eos.thermodynamic_delta),
    # PLAN_BVP.md Milestone 6. c_p_eff includes the H2 recombination latent-heat term
    # (Sub-task 8b, eos.latent_heat_capacity) on top of the frozen-composition specific heat.
    _T_prev, P_prev = _interp_state_prev(m, state_prev)
    dP_dt = (P - P_prev) / dt
    delta = eos.thermodynamic_delta(rho, T, mu, config.MU_E, d_inv_mu_dT)
    ddelta_dP, ddelta_dT = _thermodynamic_delta_derivatives(
        rho, T, mu, drho_dP, drho_dT, delta, d_inv_mu_dT, d2_inv_mu_dT2)
    df2_dP = delta / (dt * rho) + dP_dt * ddelta_dP / rho - delta * dP_dt / rho**2 * drho_dP
    c_p = eos.specific_heat_cp(gamma_T, mu) + eos.latent_heat_capacity(T)
    dT_dt = (T - _T_prev) / dt
    dcp_dT = _effective_heat_capacity_derivative(T, mu, d_inv_mu_dT, gamma_T, dgamma_eff_dT, d2chi_dT2)
    # NEW (Sub-task 8b): -dT_dt*dcp_dT - c_p was a true T-independent constant before, so this
    # term did not exist previously (the old formula's bare "-c_p/dt" was exact as written).
    df2_dT = -c_p / dt - dT_dt * dcp_dT + dP_dt * ddelta_dT / rho - delta * dP_dt / rho**2 * drho_dT
    J[2, 0] = 0.0
    J[2, 1] = dP_dlnP * df2_dP
    J[2, 2] = 0.0
    J[2, 3] = dT_dlnT * df2_dT

    # Row 3: f3 = dlnT_dm = G_blend*f1, G_blend=(1-alpha)*grad_ad + alpha*grad_eff -
    # dlnT/dm = grad_eff*dlnP/dm identically, so f3 factors through f1 exactly.
    #
    # NaN-SAFE (2026-08-12, Phase 1 pivot - PROGRESS.md has the full mechanism): must stay the
    # EXACT derivative of the residual's own NaN-safe blend (implicit_rhs_vectorized), not an
    # independently-designed guard - otherwise this reintroduces the same class of Jacobian/
    # residual mismatch bug fixed for the P/T soft clamp earlier this session. Wherever the
    # residual falls back to the pure-adiabat gradient (grad_rad/grad_eff non-finite - a
    # genuine, confirmed opacity-evaluation failure on an extreme Newton trial, not assumed),
    # its TRUE local derivative is the pure-adiabat Jacobian term, not the real-gradient one -
    # so this Jacobian must switch per-point on the SAME finiteness criterion, not blend them.
    G_blend_ad = grad_ad
    dGblend_dP_ad = np.zeros(n)
    dGblend_dL_ad = np.zeros(n)
    # NEW (Sub-task 8b): grad_ad=grad_ad(T) now, even in the pure-adiabat fallback - was
    # correctly absent (grad_ad a true constant) before this sub-task.
    dGblend_dT_ad = dgrad_ad_dT

    if alpha == 0.0:
        G_blend, dGblend_dP, dGblend_dL, dGblend_dT = G_blend_ad, dGblend_dP_ad, dGblend_dL_ad, dGblend_dT_ad
    else:
        grad_eff = gradients.effective_gradient(grad_rad, grad_ad)[0]
        dgeff_dgrad_rad, dgeff_dgrad_ad = _effective_gradient_derivative(grad_rad, grad_ad)
        dGblend_dL_real = alpha * dgeff_dgrad_rad * dgrad_rad_dL
        dGblend_dP_real = alpha * dgeff_dgrad_rad * dgrad_rad_dP
        # NEW (Sub-task 8b): grad_eff depends on grad_ad too (not just grad_rad), and grad_ad
        # is now T-dependent - both channels contribute to dGblend_dT.
        dGblend_dT_real = alpha * dgeff_dgrad_rad * dgrad_rad_dT + (1.0 - alpha + alpha * dgeff_dgrad_ad) * dgrad_ad_dT
        G_blend_real = (1.0 - alpha) * grad_ad + alpha * grad_eff

        finite = (np.isfinite(G_blend_real) & np.isfinite(dGblend_dP_real)
                  & np.isfinite(dGblend_dL_real) & np.isfinite(dGblend_dT_real))
        G_blend = np.where(finite, G_blend_real, G_blend_ad)
        dGblend_dP = np.where(finite, dGblend_dP_real, dGblend_dP_ad)
        dGblend_dL = np.where(finite, dGblend_dL_real, dGblend_dL_ad)
        dGblend_dT = np.where(finite, dGblend_dT_real, dGblend_dT_ad)
    f3 = G_blend * f1
    J[3, 0] = G_blend * J[1, 0]                                    # via f1's r-dependence only
    J[3, 1] = (dP_dlnP * dGblend_dP) * f1 + G_blend * J[1, 1]      # explicit G_blend(P) + f1(lnP) terms (J[1,1] already carries its own dP_dlnP factor)
    J[3, 2] = dGblend_dL * f1                                       # f1 has no L-dependence
    J[3, 3] = (dT_dlnT * dGblend_dT) * f1                           # f1 has no T-dependence

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
        # _safe_exp_state (not raw np.exp): the boundary residual is evaluated at trial
        # (r,P,L,T) too, subject to the SAME wild Newton-exploration risk as the interior RHS -
        # see _safe_exp_state's own docstring for the full mechanism (2026-08-10 debugging).
        r_a, L_a = y_a[0], y_a[2]
        r_b, L_b = y_b[0], y_b[2]
        P_a, T_a = _safe_exp_state(y_a[1], y_a[3])
        P_b, T_b = _safe_exp_state(y_b[1], y_b[3])

        res = np.zeros(4)
        # Sub-task 8b: mu evaluated LOCALLY at each boundary's own T, not the fixed config.MU -
        # the photospheric boundary in particular sits well inside the cool, partly-molecular
        # regime this sub-task targets.
        rho_c = eos.density(P_a, T_a, eos.mean_molecular_weight(T_a), config.MU_E)
        # Analytic constant-density-center relation (PLAN_BVP.md Milestone 2):
        # r(m_min) = (3*m_min/(4*pi*rho_c))^(1/3)   [cm]
        r_analytic = (3.0 * m_min / (4.0 * np.pi * rho_c)) ** (1.0 / 3.0)
        res[0] = za[0] - r_analytic / R_SCALE   # r_hat(m_min) = r_analytic/R_SCALE
        res[1] = za[2]                            # L_hat(m_min) = arcsinh(0/L_SCALE) = 0 (no nuclear source)

        # Mechanical (photospheric) residual, log space - consistent with P,T already being
        # log-transformed throughout (PLAN_BVP.md Milestone 3): a linear P_b-P_photo residual
        # sits a ~1e11 dyn/cm^2-scale term in the same vector as center residuals near
        # machine-zero, a scale mismatch the log form removes. zb[1] is already ln(P_b).
        P_photo = boundary_conditions.photospheric_pressure(r_b, P_b, T_b, eos.mean_molecular_weight(T_b), config.MU_E)
        res[2] = zb[1] - np.log(P_photo)

        # Thermal (net-flux radiative) residual, in the SAME arcsinh units as the state
        # vector's own L_hat.
        L_expected = 4.0 * np.pi * r_b**2 * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        res[3] = zb[2] - np.arcsinh(L_expected / L_SCALE)
        return res
    return bc


def make_bc_jacobian_scaled(m_min):
    """Scaled-state counterpart of make_bc_scaled - dbc/d(za), dbc/d(zb), each (4,4). Derived
    by direct differentiation of make_bc_scaled's residuals.

    dP_dlnP_*, dT_dlnT_* (_safe_exp_state_derivatives) are the ACTUAL d(P)/d(lnP), d(T)/d(lnT)
    through the soft clamp - see implicit_rhs_jacobian's own docstring for why the bare P_*/T_*
    value is only an exact stand-in for these deep inside the safe range, and was silently
    wrong in the saturated region under the old hard clamp.
    """
    def bc_jac(za, zb):
        y_a, y_b = _to_physical(za), _to_physical(zb)
        # _safe_exp_state (not raw np.exp) - matches make_bc_scaled's bc(), which this Jacobian
        # must differentiate consistently (2026-08-10 debugging, see _safe_exp_state's docstring).
        r_a, L_a = y_a[0], y_a[2]
        r_b, L_b = y_b[0], y_b[2]
        P_a, T_a = _safe_exp_state(y_a[1], y_a[3])
        P_b, T_b = _safe_exp_state(y_b[1], y_b[3])
        dP_dlnP_a, dT_dlnT_a = _safe_exp_state_derivatives(y_a[1], y_a[3])
        dP_dlnP_b, dT_dlnT_b = _safe_exp_state_derivatives(y_b[1], y_b[3])

        # Sub-task 8b: mu evaluated LOCALLY at each boundary's own T (matches make_bc_scaled's
        # own residual, which this Jacobian must differentiate consistently).
        _chi_a, _dchi_a, _d2chi_a, mu_a, d_inv_mu_dT_a, _d2im_a, _gT_a, _dgT_a = _h2_transition_derivatives(T_a)
        rho_a = eos.density(P_a, T_a, mu_a, config.MU_E)
        drho_dP_a, drho_dT_a = _eos_density_derivatives(P_a, T_a, rho_a, mu_a, d_inv_mu_dT_a)
        r_analytic = (3.0 * m_min / (4.0 * np.pi * rho_a)) ** (1.0 / 3.0)
        dr_analytic_drho = -r_analytic / (3.0 * rho_a)

        dbc_dza = np.zeros((4, 4))
        dbc_dza[0, 0] = 1.0
        dbc_dza[0, 1] = -(dr_analytic_drho * drho_dP_a * dP_dlnP_a) / R_SCALE
        dbc_dza[0, 3] = -(dr_analytic_drho * drho_dT_a * dT_dlnT_a) / R_SCALE
        dbc_dza[1, 2] = 1.0

        _chi_b, _dchi_b, _d2chi_b, mu_b, d_inv_mu_dT_b, _d2im_b, _gT_b, _dgT_b = _h2_transition_derivatives(T_b)
        rho_b = eos.density(P_b, T_b, mu_b, config.MU_E)
        drho_dP_b, drho_dT_b = _eos_density_derivatives(P_b, T_b, rho_b, mu_b, d_inv_mu_dT_b)
        kappa_b = opacity.bell_lin_opacity(rho_b, T_b)
        dkappa_drho_b, dkappa_dT_b = _opacity_derivatives(rho_b, T_b, kappa_b)
        P_photo = boundary_conditions.photospheric_pressure(r_b, P_b, T_b, mu_b, config.MU_E)

        dPphoto_dr = -2.0 * P_photo / r_b
        dPphoto_dP = -P_photo * (dkappa_drho_b / kappa_b) * drho_dP_b
        dPphoto_dT = -P_photo * ((dkappa_drho_b / kappa_b) * drho_dT_b + dkappa_dT_b / kappa_b)

        L_expected = 4.0 * np.pi * r_b**2 * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        dLexp_dr = 8.0 * np.pi * r_b * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        dLexp_dT = 16.0 * np.pi * r_b**2 * config.SIGMA_SB * T_b**3   # plain d(L_expected)/dT_b - see the chain-rule note below
        d_arcsinh = 1.0 / np.sqrt(L_expected**2 + L_SCALE**2)   # d(arcsinh(L_expected/L_SCALE))/d(L_expected)

        dbc_dzb = np.zeros((4, 4))
        dbc_dzb[2, 0] = (-dPphoto_dr / P_photo) * R_SCALE                # d/d(r_hat_b) = R_SCALE * d/d(r_b)
        dbc_dzb[2, 1] = 1.0 - (dPphoto_dP / P_photo) * dP_dlnP_b
        dbc_dzb[2, 3] = -(dPphoto_dT / P_photo) * dT_dlnT_b
        dbc_dzb[3, 0] = -d_arcsinh * dLexp_dr * R_SCALE                  # d/d(r_hat_b) = R_SCALE * d/d(r_b)
        dbc_dzb[3, 2] = 1.0
        dbc_dzb[3, 3] = -d_arcsinh * dLexp_dT * dT_dlnT_b                # d/d(lnT_b) = dT_dlnT_b * d/d(T_b), chain rule applied once here

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


def _attempt_direct_solve(state_prev, dt, bc, bc_jac, x, y_guess, alpha, use_analytic_jacobian=True):
    def fun(x_, y_):
        return implicit_rhs_scaled(x_, y_, state_prev, dt, alpha)
    fun_jac = None
    if use_analytic_jacobian:
        def fun_jac(x_, y_):
            return implicit_rhs_jacobian_scaled(x_, y_, state_prev, dt, alpha)
    t0 = time.time()
    sol = _safe_solve_bvp(fun, bc, x, y_guess, fun_jac, bc_jac)
    return sol, time.time() - t0


def _attempt_continuation_solve(state_prev, dt, bc, bc_jac, x, y_guess, alpha_steps, use_analytic_jacobian=True):
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
        fun_jac = None
        if use_analytic_jacobian:
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


def _solve_structure_bvp(state_prev, dt, warm_start_L, switch_epsilon, use_analytic_jacobian=True):
    """Shared solve_bvp orchestration for both relax_initial_state (dt=pseudo-relaxation
    timestep) and solve_timestep (dt=real elapsed time) - both warm-start the mesh/initial
    guess from state_prev's own profile (matching the old shooting code's warm-start
    convention: relax_initial_state seeded from state_0 itself, solve_timestep from the
    actual previous converged state). warm_start_L: see _build_mesh_and_guess's docstring -
    False for relax_initial_state (state_0.L is diagnostic-only), True for solve_timestep
    (state_prev.L is a genuine previously-converged solution).

    switch_epsilon: the Schwarzschild-switch smoothing width (gradients.effective_gradient)
    to use for THIS solve - config.GRAD_EFF_SWITCH_EPSILON's own comment has the full
    reasoning (2026-08-08). Applied by temporarily overriding config.GRAD_EFF_SWITCH_EPSILON
    for the duration of the solve (same try/finally pattern _build_mesh_and_guess already
    uses for N_GRID_POINTS/GRID_OUTER_REFINEMENT) rather than threading a new parameter
    through gradients.py/odes.py - keeps those modules' existing pure-function signatures
    untouched; the choice of WHICH epsilon to use is entirely bvp_solver.py's own
    orchestration concern.

    use_analytic_jacobian=False: use scipy's own finite-difference Jacobian instead of
    implicit_rhs_jacobian_scaled/make_bc_jacobian_scaled. Originally added (2026-08-11, Senior
    Numerical Analyst report) because the analytic Jacobian did not differentiate through
    _safe_exp_state's then-hard clamp; RESOLVED the same day by replacing that clamp with a
    smooth construction whose derivative the analytic Jacobian now correctly includes
    everywhere (_safe_exp_state_derivatives, verified in the saturation region by Check 37) -
    this escape hatch remains available for debugging/isolation, but production callers no
    longer need it purely because the clamp is active.

    Attempts a direct alpha=1 solve first (cheap, matches the old shooting solve_timestep's
    behavior of not re-relaxing every step); falls back to the config.BVP_ALPHA_CONTINUATION_
    STEPS ladder only if that fails - bvp_experiment.py's proven strategy (PLAN_BVP.md
    Milestone 6).

    Returns (sol, m_min) - the raw solve_bvp OptimizeResult (scaled-state solution).
    """
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    bc = make_bc_scaled(m_min)
    bc_jac = make_bc_jacobian_scaled(m_min) if use_analytic_jacobian else None
    x, y_guess = build_mesh_and_guess_scaled(state_prev, warm_start_L)
    _smoke_test_vectorization(state_prev, dt, bc, x, y_guess)

    switch_epsilon_orig = config.GRAD_EFF_SWITCH_EPSILON
    config.GRAD_EFF_SWITCH_EPSILON = switch_epsilon
    try:
        print(f"bvp_solver: attempting direct solve_bvp at alpha=1.0 (switch_epsilon="
              f"{switch_epsilon:.1e}, analytic_jacobian={use_analytic_jacobian}) ...", flush=True)
        sol, elapsed = _attempt_direct_solve(state_prev, dt, bc, bc_jac, x, y_guess, alpha=1.0,
                                              use_analytic_jacobian=use_analytic_jacobian)
        if sol.status != 0:
            print(f"bvp_solver: direct solve_bvp did not converge (status={sol.status}, "
                  f"{sol.message}) after {elapsed:.1f}s - falling back to alpha-continuation", flush=True)
            sol, elapsed = _attempt_continuation_solve(state_prev, dt, bc, bc_jac, x, y_guess,
                                                        config.BVP_ALPHA_CONTINUATION_STEPS,
                                                        use_analytic_jacobian=use_analytic_jacobian)
    finally:
        config.GRAD_EFF_SWITCH_EPSILON = switch_epsilon_orig

    if sol.status != 0:
        raise RuntimeError(
            f"bvp_solver: solve_bvp failed to converge even via alpha-continuation "
            f"(status={sol.status}, {sol.message}) after {elapsed:.1f}s - not a genuine solution"
        )
    return sol, m_min


def _bvp_solution_to_state(sol, m_min, state_prev, t) -> state.SimulationState:
    """Converts a converged scaled-state solve_bvp solution into a SimulationState. solve_bvp's
    domain [m_min, M_TOTAL] is FIXED and known exactly (unlike shooting's event-determined
    surface) - a structural simplification of this pivot, not an approximation.

    2026-08-08 (PROGRESS.md): a densified BVP_MESH_N_GRID_POINTS/BVP_MESH_OUTER_REFINEMENT
    output grid (matching _build_mesh_and_guess's own densification) was tried as a candidate
    fix for the step-2 solve_timestep convergence failure - state_prev's coarse default
    N_GRID_POINTS=200/GRID_OUTER_REFINEMENT=1e-4 grid measurably failed to represent the true
    dense solve_bvp solution (~1-2% error) near T_surface->T_NEB. An isolated test (with the
    log-interpolation fix also reverted) confirmed this was NOT the decisive fix either -
    reverted to the plain default grid pending the wide-epsilon Schwarzschild-switch
    investigation instead (the dominant cause: a genuine marginal-convection band, not a
    resolution artifact).
    """
    m = _build_output_grid(m_min, config.M_TOTAL)
    z = sol.sol(np.log(m))
    r, lnP, L, lnT = _to_physical(z)
    P, T = np.exp(lnP), np.exp(lnT)
    # Sub-task 8b: mu(T), not the fixed config.MU - the OUTPUT rho must match what the solver
    # actually assumed internally (implicit_rhs_vectorized/odes.stellar_odes), or this field
    # would silently disagree with the converged (P,T) profile it's supposed to describe.
    rho = eos.density(P, T, eos.mean_molecular_weight(T), config.MU_E)

    print(f"bvp_solver: t>0 solve_bvp converged, t={t:.4e} s, nodes={sol.x.size}, "
          f"P_center={P[0]:.6e} dyn/cm^2, T_center={T[0]:.6e} K, "
          f"r_surface={r[-1]/config.R_JUPITER_CM:.4f} R_Jup, L_surface={L[-1]:.4e} erg/s")

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=t, prev=state_prev)


# ==========================================
# SECTION: Public t>0 Solves (relax_initial_state, solve_timestep)
# ==========================================

def relax_initial_state(state_0, force_clamp_off_stage1=True) -> state.SimulationState:
    """Relax state_0 (solve_static_structure's output - built by forcing the pure ideal-gas
    adiabat, not a genuine solution of the real 4-ODE system's Schwarzschild-selected
    temperature gradient) into a state that IS self-consistent with the same implicit
    equations solve_timestep() uses, via solve_bvp at a small pseudo-timestep
    (config.RELAX_DT_FRACTION*T_KH_TIMESCALE_S - NOT real elapsed time, t is left unchanged).

    2026-08-08: re-platformed from shooting (bvp_solver_shooting_archive.py) onto solve_bvp
    collocation (PLAN_BVP.md Milestone 6) - same physical role, different numerical method.

    TWO-STAGE (2026-08-10, Sub-task 8b - PROGRESS.md has the full debugging trail):
    solve_static_structure()'s adiabatic seed assumes constant atomic mu/gamma throughout -
    consistent with the physics BEFORE Sub-task 8b, but a genuine, sharp inconsistency at the
    cool photospheric boundary now that mu(T)/gamma_eff(T) are real (mu jumps ~1.83x there,
    Check 38's own finding). Asking one monolithic BVP solve to absorb BOTH the adiabat-
    >Schwarzschild correction AND this fresh composition jump from the raw seed caused a
    genuine Newton-iteration failure (singular Jacobian - confirmed independent of
    GRAD_EFF_SWITCH_EPSILON width, so not the historical convective-switch fragility recurring,
    a new failure mode specific to the composition discontinuity). RE-CONFIRMED 2026-08-11
    (soft-clamp course correction, PROGRESS.md) even with the analytic Jacobian now properly
    differentiating through the clamp everywhere (Check 37's saturation-region extension) - a
    single-stage attempt with real physics, clamp on, analytic Jacobian still fails the same
    way from a fresh state_0. The composition jump is a genuinely separate difficulty from the
    clamp-Jacobian bug, not a symptom of it - the two-stage structure stays.

    Stage 1 relaxes under the OLD constant-mu/gamma physics (config.
    USE_H2_RECOMBINATION_PHYSICS=False) - proven convergent (this is exactly this function's
    pre-Sub-task-8b behavior, unchanged) - WITH config.BVP_CLAMP_EXTREME_TRIAL_VALUES also
    forced off. RE-TESTED 2026-08-11 with the fixed soft clamp (not assumed still true): stage 1
    STILL regresses to a singular Jacobian if the clamp is forced on, even though the clamp's
    own derivative is now correct everywhere (Check 37/40b) - a single bad Newton step can
    still outrun any FINITE-width clamp's restoring gradient (it only pushes the point of no
    return out to ~75 log-units past the boundary, not away entirely - config.
    BVP_SOFT_CLAMP_WIDTH's own comment), and stage 1's own trajectory apparently needs
    exactly that unconditional headroom. Genuinely independent of the Jacobian-consistency bug,
    not a regression to "fix" further - left off. Stage 1 uses the fast analytic Jacobian
    throughout (use_analytic_jacobian=True, the default) since it's provably unneeded there.

    Stage 2 is a single MICRO solve_bvp call (config.RELAX_RECOMBINATION_MICRO_DT_FRACTION*
    T_KH_TIMESCALE_S, deliberately smaller than stage 1's own dt_relax) with the real physics
    AND the clamp both turned back on, warm-started from stage 1's already self-consistent
    solution (not the raw adiabatic seed) - the implicit dt-damped energy equation walks the
    boundary layer's composition onto the new mu(T) curve gradually, instead of one undamped
    Newton correction attempting the whole jump at once. Uses the fast analytic Jacobian
    (use_analytic_jacobian=True) as of 2026-08-11 - previously forced to scipy's own numerical
    Jacobian specifically to sidestep the (now-fixed) clamp-derivative inconsistency; re-tested
    directly against the numerical-Jacobian baseline (same 9 iterations, same r_surface=4.9313
    R_Jup, 4442 vs 4445 nodes - a trivial mesh-refinement-path difference, not a different
    answer) before trusting the switch.

    t is left unchanged throughout, matching this function's existing contract (both stages
    pass t=state_0.t explicitly, not solve_timestep's own t=state_prev.t+dt advancement).

    force_clamp_off_stage1 (2026-08-12, Phase 1 / First Hydrostatic Core pivot - PROGRESS.md
    has the full report): defaults True, preserving the exact stage-1 behavior described above
    (proven for Phase 3's compact, atomic regime - its own trajectory genuinely needs the
    clamp's unconditional headroom, confirmed independent of the Jacobian fix). Phase 1's
    diffuse, degeneracy-negligible regime is different: stage 1 there crashes outright with the
    clamp off (a raw np.exp() overflow, not a mesh explosion) and converges cleanly - first
    attempt, no continuation fallback needed - with the clamp left ON at its global default.
    Callers with a fresh, non-Phase-3 regime should verify directly which setting their own
    trajectory needs (as done here), not assume either one transfers.
    """
    dt_relax = config.RELAX_DT_FRACTION * config.T_KH_TIMESCALE_S

    recomb_flag_orig = config.USE_H2_RECOMBINATION_PHYSICS
    clamp_flag_orig = config.BVP_CLAMP_EXTREME_TRIAL_VALUES
    config.USE_H2_RECOMBINATION_PHYSICS = False
    if force_clamp_off_stage1:
        config.BVP_CLAMP_EXTREME_TRIAL_VALUES = False
    try:
        print(f"bvp_solver: relax_initial_state stage 1/2 (old constant-mu/gamma physics, "
              f"clamp {'off' if force_clamp_off_stage1 else 'at global default'}) ...", flush=True)
        sol, m_min = _solve_structure_bvp(state_0, dt_relax, warm_start_L=False,
                                           switch_epsilon=config.GRAD_EFF_SWITCH_EPSILON,
                                           use_analytic_jacobian=True)
    finally:
        config.USE_H2_RECOMBINATION_PHYSICS = recomb_flag_orig
        config.BVP_CLAMP_EXTREME_TRIAL_VALUES = clamp_flag_orig
    state_mid = _bvp_solution_to_state(sol, m_min, state_0, t=state_0.t)

    if not config.USE_H2_RECOMBINATION_PHYSICS:
        return state_mid   # matches pre-Sub-task-8b behavior exactly if the flag is off globally

    print("bvp_solver: relax_initial_state stage 2/2 (recombination-physics micro-step, "
          "clamp on, analytic Jacobian) ...", flush=True)
    dt_micro = config.RELAX_RECOMBINATION_MICRO_DT_FRACTION * config.T_KH_TIMESCALE_S
    sol, m_min = _solve_structure_bvp(state_mid, dt_micro, warm_start_L=True,
                                       switch_epsilon=config.GRAD_EFF_SWITCH_EPSILON,
                                       use_analytic_jacobian=True)
    return _bvp_solution_to_state(sol, m_min, state_mid, t=state_0.t)


def solve_timestep(state_prev, dt) -> state.SimulationState:
    """Solve the envelope structure at t = state_prev.t + dt via solve_bvp collocation.

    2026-08-08: re-platformed from shooting (bvp_solver_shooting_archive.py) onto solve_bvp
    (PLAN_BVP.md Milestone 6) - same physical role (implicit Henyey-style time differencing,
    photospheric + net-flux-radiative surface conditions), different numerical method. Uses
    config.GRAD_EFF_SWITCH_EPSILON_TIMESTEP (wider than relax_initial_state's switch width -
    see that constant's own comment) - a real timestep can collapse the outer envelope's L
    enough to land the whole outer profile in a genuinely marginal-convection band that the
    narrow switch cannot resolve without the mesh growing without bound.
    """
    sol, m_min = _solve_structure_bvp(state_prev, dt, warm_start_L=True,
                                       switch_epsilon=config.GRAD_EFF_SWITCH_EPSILON_TIMESTEP)
    return _bvp_solution_to_state(sol, m_min, state_prev, t=state_prev.t + dt)
