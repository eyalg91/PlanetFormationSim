# bvp_experiment.py — SUPERSEDED 2026-08-08: this experiment's proven logic (state-vector
# scaling, analytic Jacobians, the ALPHA_MAX continuation) was promoted into production
# bvp_solver.py (PLAN.md §4.2, PLAN_BVP.md §6). Kept in place, unmodified, as the historical
# experiment record - not imported by any active module. For the current production solver,
# see bvp_solver.py.
#
# bvp_experiment.py — Standalone, isolated experiment: scipy.integrate.solve_bvp (a
# collocation/relaxation method - the same numerical FAMILY as Henyey's implicit relaxation,
# not local shooting) as a candidate replacement for bvp_solver.py's shooting architecture.
# PROGRESS.md (2026-08-06) has the full strategic discussion behind this experiment.
#
# ISOLATION: does NOT modify bvp_solver.py, gradients.py, odes.py, eos.py, opacity.py,
# boundary_conditions.py, or config.py (beyond a RUNTIME-only override of
# config.T_CENTER_INITIAL to select each spot-check temperature - the same pattern already
# used throughout this project to run at different T_center values; no config.py file edit).
# Imports and CALLS existing physics/solver functions freely (bvp_solver.
# solve_static_structure, bvp_solver._build_output_grid, odes.stellar_odes,
# boundary_conditions.boundary_conditions, eos.grad_adiabatic) but reimplements the RHS/BC
# glue as new, properly-vectorized functions here: bvp_solver._implicit_rhs_logm is written
# for solve_ivp's single-point-at-a-time RHS contract (it wraps scalars in 1-element arrays,
# e.g. `y_full = np.array([[r], [P], [L], [T]])` then indexes `dr_dm[0]`) and is NOT directly
# reusable for solve_bvp, which calls fun(x, y) with the WHOLE mesh at once (x shape (n,), y
# shape (4, n)). odes.stellar_odes itself IS already properly vectorized (CLAUDE.md style
# rule), so the underlying physics call is identical either way - only the log-space wrapper
# needed rewriting, here, not in the shared module.
#
# PHYSICAL BASELINE this experiment must preserve (agreed and printed in full before this
# file was written - PROGRESS.md 2026-08-06): same EOS (ideal + non-relativistic degeneracy,
# no ionization chemistry, no radiation pressure term), same Bell & Lin opacity (hard regime
# switches), same Schwarzschild criterion (infinitely-efficient convection, now numerically
# smoothed near the switch via config.GRAD_EFF_SWITCH_EPSILON - a NUMERICAL not physical
# change, applies identically here since gradients.py is reused unmodified), same grey
# Eddington photospheric mechanical BC and net-flux thermal BC, same purely-gravitational
# (Kelvin-Helmholtz) energy source, same quasi-static assumption. This experiment changes
# ONLY the numerical method used to enforce these equations.
#
# STRUCTURAL SIMPLIFICATION vs shooting: the mass domain [m_min, M_TOTAL] is fixed and known
# exactly (M_TOTAL is a project constant, not a shooting unknown), so the photospheric
# condition becomes a real boundary equation evaluated at the true domain endpoint - no
# solve_ivp event, no mass-matching residual needed. P_center, T_center are no longer a
# separate outer root-find's unknowns; they are just ya[1], ya[3] - two more components of
# the one global y(x) solve_bvp's own Newton iteration solves for directly, simultaneously
# with the rest of the profile.

import json
import sys
import time

import numpy as np
from scipy.integrate import solve_bvp

import boundary_conditions
import bvp_solver
import config
import dev_cache
import eos
import gradients
import odes
import opacity

CACHE_DIR = r"C:\Users\yuval\AppData\Local\Temp\claude\c--PythonProjects-PlanetFormationSim\0afde8a8-8ac0-42d9-b192-2bcd7bf801a8\scratchpad"
DT_RELAX = 0.01 * config.T_KH_TIMESCALE_S   # same pseudo-timestep convention as bvp_solver.relax_initial_state
SOLVE_BVP_TOL = 1.0e-6        # solve_bvp's own residual-control tolerance - looser than config.BVP_TOL by design (untested territory, flagged for empirical tuning in the results)
SOLVE_BVP_MAX_NODES = 80000   # RAISED 2026-08-07 (20000->80000): with scaling+analytic Jacobians, the continuation's alpha=1.0 step exhausted the old 20000-node budget while still actively refining (residual oscillating 1e-5 to 1e-4, boundary residual already at machine precision, 8.88e-16) - a mesh-budget limit, not a crash; PROGRESS.md has the full trace

# ==========================================
# SECTION: State-Vector Scaling (PLAN_BVP.md Milestone 6)
# ==========================================
#
# Motivation: five independent physics/BC hypotheses (Milestones 0-3) were each ruled out in
# isolation without moving the T=13000K crash, and the un-scaled state vector has an extreme,
# directly-measured heterogeneity: a single Jacobian-verification point this session showed
# y=[r=2.9e10, lnP=5.9, L=2.6e29, lnT=3.6] - L is 28 ORDERS OF MAGNITUDE larger than lnT in
# the same vector Newton must invert. This is a textbook cause of an ill-conditioned Jacobian,
# independent of any individual physics term - exactly consistent with the pattern of
# negative results. r and L are rescaled to be O(1)-comparable with the already-good lnP/lnT;
# P and T stay log-transformed (unchanged, already well-conditioned and positivity-guaranteed).
#
# New state z = [r_hat, lnP, L_hat, lnT]:
#   r_hat = r / R_SCALE                          (linear rescaling - R_SCALE is a true constant)
#   L_hat = arcsinh(L / L_SCALE)                 (nonlinear, sign-preserving, log-like compression)
#
# arcsinh over a hand-rolled sign*log1p: a single closed-form expression (arcsinh(x)=
# ln(x+sqrt(x^2+1))), smooth (C-infinity) with no piecewise branching, and its own derivative
# (1/sqrt(x^2+1)) is simple and well-conditioned everywhere, including at x=0 - convenient for
# both evaluation and (below) the analytic Jacobian.
R_SCALE = config.R_JUPITER_CM         # [cm] - true constant, r/R_SCALE is a LINEAR rescaling
L_SCALE = config.L_KH_SCALE_ERG_S     # [erg/s] - already-vetted KH-luminosity reference (config.py, used for the L-floor epsilon); exact value matters far less here than for that epsilon - arcsinh compresses ANY excursion (even the ~1e54 erg/s ones directly observed this session) to a modest double-digit scaled value regardless of being off by an order of magnitude or two


def _to_physical(z):
    """z=[r_hat, lnP, L_hat, lnT] -> y=[r, lnP, L, lnT] (physical). L=L_SCALE*sinh(L_hat) is
    the exact inverse of L_hat=arcsinh(L/L_SCALE)."""
    r_hat, lnP, L_hat, lnT = z
    return np.array([r_hat * R_SCALE, lnP, L_SCALE * np.sinh(L_hat), lnT])


def _to_scaled(y):
    """y=[r, lnP, L, lnT] (physical) -> z=[r_hat, lnP, L_hat, lnT]."""
    r, lnP, L, lnT = y
    return np.array([r / R_SCALE, lnP, np.arcsinh(L / L_SCALE), lnT])


# ==========================================
# SECTION: Vectorized RHS (x=ln(m), y=[r, lnP, L, lnT], whole mesh at once)
# ==========================================

def implicit_rhs_vectorized(x, y, state_prev, dt, alpha):
    """dy/dx for the full 4-ODE system, vectorized across solve_bvp's whole mesh at once.

    Same physics/formulas as bvp_solver._implicit_rhs_logm (log P, log T state, alpha-blended
    nabla_eff) - NOT a call to that function (module docstring explains why) - calls
    odes.stellar_odes directly instead, which is already properly vectorized.
    """
    m = np.exp(x)
    r, lnP, L, lnT = y
    P, T = np.exp(lnP), np.exp(lnT)
    # Diagnostic only (does not alter behavior): report WHERE along the mesh a trial (P,T) is
    # extreme, before it potentially crashes eos.density downstream - PROGRESS.md 2026-08-06's
    # decisive T=13000K result needed this to characterize center-region vs surface-boundary-
    # layer failure, not just that it fails.
    bad = ~np.isfinite(P) | ~np.isfinite(T) | (P <= 0) | (T <= 0) | (np.abs(lnP) > 300) | (np.abs(lnT) > 300)
    if np.any(bad):
        idx = np.where(bad)[0]
        print(f"  [diag] extreme/non-finite (P,T) trial at {len(idx)} mesh point(s), "
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
        # grad_ad=dlnT/dlnP by definition (eos.grad_adiabatic) - same blend bvp_solver's
        # _implicit_rhs_logm uses, reproduced here rather than imported (module docstring).
        dlnT_dm_ad = eos.grad_adiabatic(config.GAMMA) * dlnP_dm
        dlnT_dm = (1.0 - alpha) * dlnT_dm_ad + alpha * (dT_dm_real / T)
    # Chain rule: d/dx = m * d/dm, since x = ln(m)
    return np.array([dr_dm, dlnP_dm, dL_dm, dlnT_dm]) * m


# ==========================================
# SECTION: Boundary Conditions (reuses boundary_conditions.py unmodified)
# ==========================================

def make_bc(m_min):
    """Builds the bc(ya, yb) closure for solve_bvp, given m_min = the innermost mesh mass.

    IMPORTANT DEVIATION from a literal reuse of boundary_conditions.boundary_conditions:
    that function's r_a residual is exactly `ya[0]` - correct for SHOOTING, which always
    calls it with ya=np.zeros(4) (a placeholder; shooting enforces r=r_seed>0, L=0 by
    CONSTRUCTING the integration's initial state, never checking a residual for it - see
    bvp_solver.relax_initial_state's own boundary_conditions() call). Under solve_bvp, ya IS
    a genuine solved-for unknown - calling boundary_conditions() directly on the real ya would
    force r(m_min)=0 EXACTLY, a true 1/r^2 singularity in dr_dm that solve_bvp's own mesh
    point AT m_min would then have to evaluate.

    PLAN_BVP.md Milestone 2 (2026-08-07): the first fix (a FIXED r_seed=state_0.r[0]) was
    itself only an approximation - r_seed came from the Lane-Emden T=0-degenerate-limit SEED,
    computed once before the bracket search, never re-tied to the actual converged (or, here,
    the live TRIAL) center state. Checked directly: state_0.r[0] is already algebraically the
    analytic constant-density-center relation r(m)=(3m/(4*pi*rho_c))^(1/3), just evaluated at
    a rho_c from the approximate seed, not the true center density. This closure now
    re-derives rho_c from the LIVE trial (P_a, T_a) via eos.density at every Newton iteration,
    so the center condition is self-consistent with whatever (P_center, T_center) solve_bvp is
    actually trying at that step, not a fixed pre-estimate.

    KNOWN RISK, explicitly flagged for testing, not just implementation: this couples the
    boundary condition itself to eos.density's own Newton-Raphson solve - the same function
    whose non-convergence assertion has fired throughout this investigation whenever a trial
    (P,T) leaves its physical domain. A trial (P_a, T_a) landing outside that domain will now
    crash INSIDE the boundary condition (a different code path/location than every previous
    near-photosphere crash) - watch for this specifically, not just whether the photosphere
    crash changes (the center and surface are not obviously coupled, so it may not).
    """
    def bc(ya, yb):
        ya_phys = np.array([ya[0], np.exp(ya[1]), ya[2], np.exp(ya[3])])
        yb_phys = np.array([yb[0], np.exp(yb[1]), yb[2], np.exp(yb[3])])
        res = boundary_conditions.boundary_conditions(np.zeros(4), yb_phys)   # zeros ya: only the P_b, L_b (surface) residuals are meaningful from this call

        # PLAN_BVP.md Milestone 3 (2026-08-07): mechanical surface residual reformulated in
        # log space, ln(P_b) - ln(P_photo) = 0, instead of the linear P_b - P_photo = 0
        # boundary_conditions.boundary_conditions() computes above (res[2], now overwritten).
        # Consistent with the state vector already being log-P/log-T throughout - the linear
        # form sits a ~1e11 dyn/cm^2-scale residual in the same vector as center residuals
        # near machine-zero, a scale mismatch the log form removes. yb[1] is already ln(P_b)
        # directly (the state vector's own log-P component) - no redundant exp-then-log
        # round trip needed.
        P_photo = boundary_conditions.photospheric_pressure(yb_phys[0], yb_phys[1], yb_phys[3], config.MU, config.MU_E)
        res[2] = yb[1] - np.log(P_photo)

        P_a, T_a = ya_phys[1], ya_phys[3]
        try:
            rho_c = eos.density(P_a, T_a, config.MU, config.MU_E)
        except AssertionError:
            print(f"  [diag] center BC: eos.density failed to converge for trial "
                  f"P_a={P_a:.6e}, T_a={T_a:.6e} - re-raising", flush=True)
            raise
        # Analytic constant-density-center relation (PLAN_BVP.md Milestone 2):
        # r(m_min) = (3*m_min/(4*pi*rho_c))^(1/3)   [cm]
        r_analytic = (3.0 * m_min / (4.0 * np.pi * rho_c)) ** (1.0 / 3.0)
        res[0] = ya_phys[0] - r_analytic   # r(m_min) = analytic center relation, self-consistent with the live trial (P_a,T_a)
        res[1] = ya_phys[2]                # L(m_min) = 0 (no nuclear source - exact to leading order, m_min already tiny; PLAN_BVP.md Milestone 2 step 3)
        return res
    return bc


def make_bc_scaled(m_min):
    """Scaled-state counterpart of make_bc: za=[r_hat_a,lnP_a,L_hat_a,lnT_a], same for zb.
    Converts to physical internally for the eos/opacity calls, but returns residuals in
    SCALED units throughout - leaving res[3] (thermal) in raw erg/s while everything else
    is O(1)-scaled would silently reintroduce the exact scale mismatch this milestone
    exists to remove (T=2000K's stuck ~2.68e8 boundary residual, PLAN_BVP.md Milestone 3,
    was plausibly dominated by exactly this un-rescaled term)."""
    def bc(za, zb):
        # BUG FIXED 2026-08-07 (PROGRESS.md/PLAN_BVP.md have the trace): _to_physical returns
        # y=[r,lnP,L,lnT] (matching implicit_rhs_vectorized's existing mixed convention - P,T
        # stay logarithmic there, exponentiated only where consumed) - NOT fully physical
        # [r,P,L,T]. Unpacking it directly as (r,P,L,T) here fed eos.density lnP~25 as if it
        # were a pressure in dyn/cm^2 (should be ~1e11) - caught by the required FD
        # cross-check (bc_jac disagreed with FD by up to 25x), not discovered by inspection.
        y_a, y_b = _to_physical(za), _to_physical(zb)
        r_a, P_a, L_a, T_a = y_a[0], np.exp(y_a[1]), y_a[2], np.exp(y_a[3])
        r_b, P_b, L_b, T_b = y_b[0], np.exp(y_b[1]), y_b[2], np.exp(y_b[3])

        res = np.zeros(4)
        try:
            rho_c = eos.density(P_a, T_a, config.MU, config.MU_E)
        except AssertionError:
            print(f"  [diag] center BC: eos.density failed to converge for trial "
                  f"P_a={P_a:.6e}, T_a={T_a:.6e} - re-raising", flush=True)
            raise
        r_analytic = (3.0 * m_min / (4.0 * np.pi * rho_c)) ** (1.0 / 3.0)
        res[0] = za[0] - r_analytic / R_SCALE   # r_hat(m_min) = r_analytic/R_SCALE
        res[1] = za[2]                            # L_hat(m_min) = arcsinh(0/L_SCALE) = 0

        P_photo = boundary_conditions.photospheric_pressure(r_b, P_b, T_b, config.MU, config.MU_E)
        res[2] = zb[1] - np.log(P_photo)          # unchanged - already dimensionless
        L_expected = 4.0 * np.pi * r_b**2 * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        res[3] = zb[2] - np.arcsinh(L_expected / L_SCALE)   # thermal residual, now in the SAME arcsinh units as the state vector's own L_hat
        return res
    return bc


def make_bc_jacobian_scaled(m_min):
    """Scaled-state counterpart of make_bc_jacobian - dbc/d(za), dbc/d(zb), each (4,4).
    Derived by direct differentiation of make_bc_scaled's residuals (module docstring's
    "Part A" transformations) - re-derived from scratch here rather than "rescaling" the
    old make_bc_jacobian's output, since res[0] and res[3] are each divided/composed by a
    DIFFERENT nonlinear function of the underlying physical residual now."""
    def bc_jac(za, zb):
        # Same fix as make_bc_scaled: _to_physical returns [r,lnP,L,lnT], exponentiate P,T explicitly.
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
        # BUG FIXED 2026-08-07: this must be d(L_expected)/d(T_b) DIRECTLY (T_b^3, plain
        # power rule) - NOT the already-chain-ruled d/d(lnT_b) form (T_b^4), which is what
        # the UNSCALED make_bc_jacobian correctly uses (it differentiates res directly w.r.t.
        # lnT_b there, no separate chain-rule step). Here the "* T_b" below applies that same
        # chain-rule step explicitly and separately - copying the T_b^4 form double-applied
        # it. Caught by the required FD cross-check (dbc_dzb[3,3] disagreed by ~10x).
        dLexp_dT = 16.0 * np.pi * r_b**2 * config.SIGMA_SB * T_b**3
        d_arcsinh = 1.0 / np.sqrt(L_expected**2 + L_SCALE**2)   # d(arcsinh(L_expected/L_SCALE))/d(L_expected)

        dbc_dzb = np.zeros((4, 4))
        dbc_dzb[2, 0] = (-dPphoto_dr / P_photo) * R_SCALE                # d/d(r_hat_b) = R_SCALE * d/d(r_b)
        dbc_dzb[2, 1] = 1.0 - (dPphoto_dP / P_photo) * P_b
        dbc_dzb[2, 3] = -(dPphoto_dT / P_photo) * T_b
        dbc_dzb[3, 0] = -d_arcsinh * dLexp_dr * R_SCALE                  # d/d(r_hat_b) = R_SCALE * d/d(r_b)
        dbc_dzb[3, 2] = 1.0
        dbc_dzb[3, 3] = -d_arcsinh * dLexp_dT * T_b                      # d/d(lnT_b) = T_b * d/d(T_b)

        return dbc_dza, dbc_dzb
    return bc_jac


# ==========================================
# SECTION: Mesh and Initial Guess
# ==========================================

MESH_N_GRID_POINTS = 2000   # solve_bvp-specific mesh density, well above config.N_GRID_POINTS=200
# (shooting's own adaptive Radau step control is independent of the user-supplied mesh, but
# solve_bvp's 4th-order collocation scheme extrapolates a midpoint value from y and dy/dx at
# each FIXED mesh interval BEFORE any Newton refinement - a first attempt at
# config.N_GRID_POINTS=200 overshot into an unphysical (P,T) region on that very first
# midpoint evaluation, overflowing exp(lnP)/exp(lnT) - PROGRESS.md 2026-08-06 has the trace.
# A denser initial mesh bounds that per-interval extrapolation more tightly - the direct,
# principled first thing to try, not a clip/dampen of the symptom.

def build_mesh_and_guess(state_0):
    """Composite log-mass mesh (bvp_solver._build_output_grid, reused unmodified, but with a
    solve_bvp-specific point count - see MESH_N_GRID_POINTS) and an initial y guess
    interpolated from state_0 (the adiabatic construction at this T_center) - both spanning
    x=[ln(m_min), ln(M_TOTAL)] exactly, since under solve_bvp the domain is fixed and known
    (unlike shooting's event-determined surface - module docstring)."""
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    n_grid_points_orig = config.N_GRID_POINTS
    config.N_GRID_POINTS = MESH_N_GRID_POINTS   # runtime-only override, restored below
    try:
        m_grid = bvp_solver._build_output_grid(m_min, config.M_TOTAL)
    finally:
        config.N_GRID_POINTS = n_grid_points_orig
    x = np.log(m_grid)

    r_guess = np.interp(m_grid, state_0.m, state_0.r)
    P_guess = np.interp(m_grid, state_0.m, state_0.P)
    T_guess = np.interp(m_grid, state_0.m, state_0.T)
    # ASSUMPTION: state_0.L is explicitly a DIAGNOSTIC quantity (marginal-efficient-convection
    # closure - bvp_solver.solve_static_structure's own docstring: "NOT consumed by
    # solve_timestep"), not a real Schwarzschild-selected L(m) - confirmed 2026-08-06 to be a
    # poor solve_bvp seed (surface thermal BC residual ~5e24 at the initial guess, a plausible
    # driver of an oversized first Newton correction). A simple monotonic ramp toward the same
    # KH-timescale luminosity scale already used elsewhere in this project (L_scale =
    # G*M_TOTAL^2/(R*T_KH)) is a more physically sensible seed for the REAL L profile.
    L_scale_guess = config.G * config.M_TOTAL**2 / (state_0.r[-1] * config.T_KH_TIMESCALE_S)
    L_guess = L_scale_guess * (m_grid / config.M_TOTAL)
    y_guess = np.array([r_guess, np.log(P_guess), L_guess, np.log(T_guess)])
    return x, y_guess


def build_mesh_and_guess_scaled(state_0):
    """Scaled-state counterpart of build_mesh_and_guess - same mesh/guess, converted to
    z=[r_hat,lnP,L_hat,lnT] via _to_scaled."""
    x, y_guess = build_mesh_and_guess(state_0)
    return x, _to_scaled(y_guess)


# ==========================================
# SECTION: Toy Opacity (PLAN_BVP.md Milestone 1 - isolates mesh/BC from opacity switches)
# ==========================================

# Regime 2 ("Metal grains") is what the crash region (m/M_TOTAL>=0.999) of the T=13000K
# adiabatic seed actually falls into - checked directly via opacity.determine_regime on
# state_0's own profile (PLAN_BVP.md Milestone 1 step 1): 16/17 crash-region mesh points
# land in this regime, the rest in "Ice grains" at the single outermost point - and a=0.0
# (no density dependence) makes it the cleanest possible toy: one smooth power law,
# kappa=0.1*T^0.5, no switches, but the SAME physical magnitude Bell & Lin itself uses in
# the regime that actually matters here - not an invented formula.
_TOY_REGIME = opacity.REGIMES[2]


def toy_opacity(rho, T):
    """A single, smooth Bell & Lin power law (Metal grains regime, no switches) - local to
    this experiment only, NOT a change to opacity.py. See _TOY_REGIME's selection
    rationale above."""
    return opacity.evaluate_regime(_TOY_REGIME.kappa_i, _TOY_REGIME.a, _TOY_REGIME.b, rho, T)


class opacity_override:
    """Context manager: temporarily monkey-patches the opacity.bell_lin_opacity MODULE
    ATTRIBUTE (not opacity.py the file) for the duration of a `with` block, then restores
    the real Bell & Lin function. Works because odes.py does `import opacity` then calls
    `opacity.bell_lin_opacity(...)` - looked up dynamically at every call, not bound at
    import time - so this reassignment reaches odes.py's (and boundary_conditions.
    photospheric_pressure's) calls too, without editing either file. Same category as the
    config.T_CENTER_INITIAL/config.MU runtime overrides used elsewhere in this experiment,
    just for a function instead of a constant."""
    def __init__(self, fn):
        self.fn = fn
        self.original = None

    def __enter__(self):
        self.original = opacity.bell_lin_opacity
        opacity.bell_lin_opacity = self.fn
        return self

    def __exit__(self, *exc_info):
        opacity.bell_lin_opacity = self.original
        return False


# ==========================================
# SECTION: Analytic Jacobians (PLAN_BVP.md Milestone 4)
# ==========================================
#
# Replaces scipy's default finite-difference fun_jac/bc_jac with exact analytic derivatives.
# Motivation (PLAN_BVP.md Milestones 0-3): four independent physics/BC hypotheses were each
# ruled out in isolation (ionization, dissociation-mu, opacity switches, center BC
# self-consistency, log-space surface BC) without moving the T=13000K crash at all - pointing
# at the Newton/Jacobian machinery itself, not any individual term. This module never supplied
# fun_jac/bc_jac, so every attempt used scipy's default FD estimate, exposed to exactly the
# kind of extreme local sensitivity already measured this session (a shooting-side FD probe
# swinging a residual by 30+ orders of magnitude from a machine-epsilon perturbation).
#
# Derived by hand below (not autodiff/symbolic) - every piece is cross-checked against a
# finite-difference estimate before being trusted (see verify_jacobians()), per the same
# "standalone-verify-before-wiring-in" discipline that caught two real bugs earlier this
# session. A wrong analytic Jacobian is worse than none: it steers Newton confidently in the
# wrong direction rather than just being imprecise.

def _eos_density_derivatives(P, T, rho):
    """d(rho)/dP, d(rho)/dT via IMPLICIT differentiation of the EOS's defining equation
    (NOT differentiating through eos.density's own Newton iteration, which would be
    impractical and unnecessary): F(rho,P,T) = P_ideal(rho,T) + P_deg(rho) - P = 0.

    Standard implicit-function-theorem result: drho/dP = -(dF/dP)/(dF/drho),
    drho/dT = -(dF/dT)/(dF/drho). dF/drho is exactly the same quantity
    (dP_ideal_drho + dP_deg_drho) eos.density's own Newton loop already computes as its
    convergence-step denominator - reproduced here (not imported - eos.py exposes no
    derivative API, by design, since gradients.py/odes.py never needed one before this).
    """
    dP_ideal_drho = config.K_B * T / (config.MU * config.M_H)              # dF/drho, ideal term (eos.py's own formula)
    P_deg = eos.degenerate_pressure(rho, config.MU_E)
    dP_deg_drho = (5.0 / 3.0) * P_deg / rho                                 # dF/drho, degenerate term (eos.py's own formula)
    D = dP_ideal_drho + dP_deg_drho                                        # dF/drho total
    drho_dP = 1.0 / D                                                      # dF/dP = -1
    P_ideal = rho * dP_ideal_drho
    drho_dT = -(P_ideal / T) / D                                           # dF/dT = -dP_ideal/dT = -rho*k_B/(mu*m_H) = -P_ideal/T
    return drho_dP, drho_dT


def _thermodynamic_delta_derivatives(rho, T, drho_dP, drho_dT, delta):
    """d(delta)/dP, d(delta)/dT for eos.thermodynamic_delta (PLAN_BVP.md pragmatic-plan
    session, 2026-08-07) - needed now that odes.py's energy equation uses the genuine
    EOS-dependent delta instead of the hardcoded delta=1 the ORIGINAL row 2 derivation
    below assumed.

    Derived by logarithmic differentiation of delta=P_ideal/(rho*D),
    D=dP_ideal_drho+dP_deg_drho (same quantities eos.thermodynamic_delta itself computes).
    Both derivatives simplify considerably: the dP_ideal_drho/P_ideal=1/rho term exactly
    cancels the explicit d(ln rho)/dP, d(ln rho)/dT terms in the log-derivative expansion
    (P_ideal=rho*dP_ideal_drho by construction, with dP_ideal_drho itself independent of
    rho) - verified by the limiting-case check below and, more rigorously, against finite
    differences in verify_jacobians().

    d(delta)/dP = -delta*(2/3)*(dP_deg_drho/(rho*D))*drho_dP
    d(delta)/dT =  delta*(dP_deg_drho/D)*[1/T - (2/3)*drho_dT/rho]

    Limiting-case check (matches eos.thermodynamic_delta's own docstring): both -> 0 as
    dP_deg_drho -> 0 (pure ideal gas, delta=1 exactly, a true constant - zero sensitivity).
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
    law, kappa=kappa_i*rho^a*T^b: d(kappa)/d(rho)=a*kappa/rho, d(kappa)/dT=b*kappa/T -
    exact almost everywhere (undefined only exactly AT a regime transition, measure zero,
    not expected to be hit exactly by a continuous Newton trial). PLAN_BVP.md Milestone 1
    already confirmed opacity's hard switches are not what drives the T=13000K crash, so a
    regime-local analytic derivative (rather than any special switch-smoothing) is the
    correct, honest derivative of the ACTUAL function being solved (real Bell & Lin)."""
    regime_idx = opacity.determine_regime(rho, T)
    a = np.array([opacity.REGIMES[i].a for i in np.atleast_1d(regime_idx).ravel()]).reshape(np.shape(regime_idx))
    b = np.array([opacity.REGIMES[i].b for i in np.atleast_1d(regime_idx).ravel()]).reshape(np.shape(regime_idx))
    dkappa_drho = a * kappa / rho
    dkappa_dT = b * kappa / T
    return dkappa_drho, dkappa_dT


def _grad_radiative_derivatives(L, m, P, T, kappa, rho, drho_dP, drho_dT, dkappa_drho, dkappa_dT, grad_rad):
    """d(grad_rad)/dL, dP, dT for grad_rad = 3*kappa*L_safe*P/(16*pi*A_RAD*C_LIGHT*G*m*T^4)
    (gradients.grad_radiative). L_safe is the smoothed hyperbolic floor - differentiated here
    from the mathematically-equivalent SIMPLE form L_safe=0.5*(L+sqrt(L^2+eps^2)) (the
    cancellation-safe form gradients.py evaluates is algebraically identical, so its exact
    derivative is identical too - the simple form is just easier to differentiate correctly).
    d(L_safe)/dL = 0.5*(1 + L/sqrt(L^2+eps^2)) - itself well-conditioned (no cancellation),
    unlike L_safe's own value at extreme L.

    grad_rad depends on P, T both explicitly (T^-4, P^1) AND implicitly through
    kappa(rho(P,T),T) - both channels included via the chain rule.
    """
    eps_L = config.GRAD_RAD_L_FLOOR_EPSILON
    dLsafe_dL = 0.5 * (1.0 + L / np.sqrt(L**2 + eps_L**2))

    # C_rad = grad_rad's L_safe-independent prefactor (3*kappa*P/(16*pi*A_RAD*C_LIGHT*G*m*T^4))
    # - computing this directly, rather than back-solving grad_rad/L_safe, avoids any
    # division-by-L_safe risk entirely (L_safe->0 as L->-inf, so that ratio is a needless
    # 0/0 hazard for no benefit - this form has none).
    C_rad = 3.0 * kappa * P / (16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G * m * T**4)
    dgrad_rad_dL = C_rad * dLsafe_dL
    dgrad_rad_dP = grad_rad * ((dkappa_drho / kappa) * drho_dP + 1.0 / P)
    dgrad_rad_dT = grad_rad * ((dkappa_drho / kappa) * drho_dT + dkappa_dT / kappa - 4.0 / T)
    return dgrad_rad_dL, dgrad_rad_dP, dgrad_rad_dT


def _effective_gradient_derivative(grad_rad, grad_ad):
    """d(grad_eff)/d(grad_rad) for grad_eff=min_smooth(grad_rad,grad_ad)
    (gradients.effective_gradient), differentiated from the simple form
    grad_eff=0.5*(a+b)-0.5*sqrt((a-b)^2+eps^2) (same cancellation-safe-vs-simple equivalence
    as above). grad_ad is a constant (eos.grad_adiabatic), so this is the only channel."""
    eps_s = config.GRAD_EFF_SWITCH_EPSILON
    diff = grad_rad - grad_ad
    return 0.5 * (1.0 - diff / np.sqrt(diff**2 + eps_s**2))


def implicit_rhs_jacobian(x, y, state_prev, dt, alpha):
    """d(dy/dx)/dy, shape (4,4,n) - solve_bvp's fun_jac contract. Mirrors
    implicit_rhs_vectorized's physics exactly; see that function and the derivative helpers
    above for the term-by-term formulas. Derived by hand (module docstring); cross-checked
    against finite differences in verify_jacobians() before use.
    """
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
    J[1, 1] = -f1                                      # d(dP_dm/P)/d(lnP) = P*d/dP[dP_dm/P] = P*(-dP_dm/P^2) = -f1
    J[1, 2] = 0.0
    J[1, 3] = 0.0

    # Row 2: f2 = dL_dm = -c_p*dT_dt + delta*dP_dt/rho, depends on P, T only (not r, not L
    # itself). CORRECTED 2026-08-07 (module docstring): delta is now the genuine
    # EOS-dependent coefficient (odes.py/eos.thermodynamic_delta), not the previously-
    # hardcoded delta=1 - this row's derivatives gain the d(delta)/dP, d(delta)/dT terms.
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
    # dlnT/dm = grad_eff*dlnP/dm identically (grad_eff IS dlnT/dlnP by definition), so f3
    # factors through f1 exactly - simpler than differentiating (T/P)*grad_eff*dP_dm directly.
    if alpha == 0.0:
        G_blend = np.full(n, grad_ad)
        dGblend_dP = np.zeros(n)
        dGblend_dL = np.zeros(n)
        dGblend_dT = np.zeros(n)
    else:
        grad_eff = gradients.effective_gradient(grad_rad, grad_ad)[0]
        dgeff_dgrad = _effective_gradient_derivative(grad_rad, grad_ad)
        G_blend = (1.0 - alpha) * grad_ad + alpha * grad_eff
        dGblend_dL = alpha * dgeff_dgrad * dgrad_rad_dL
        dGblend_dP = alpha * dgeff_dgrad * dgrad_rad_dP
        dGblend_dT = alpha * dgeff_dgrad * dgrad_rad_dT
    f3 = G_blend * f1
    J[3, 0] = G_blend * J[1, 0]                                    # via f1's r-dependence only
    J[3, 1] = (P * dGblend_dP) * f1 + G_blend * J[1, 1]            # d/d(lnP): explicit G_blend(P) term + f1(lnP) term
    J[3, 2] = dGblend_dL * f1                                       # f1 has no L-dependence
    J[3, 3] = (T * dGblend_dT) * f1                                 # f1 has no T-dependence

    # Chain rule: d/dx = m * d/dm, so the WHOLE Jacobian (of m*f, not just f) scales by m.
    return J * m


def implicit_rhs_scaled(x, z, state_prev, dt, alpha):
    """dz/dx for the scaled state z=[r_hat,lnP,L_hat,lnT] - converts to physical, calls the
    (unchanged) physical RHS, then applies the OUTPUT-side chain-rule scaling
    (Phi'=diag(1/R_SCALE, 1, 1/sqrt(L^2+L_SCALE^2), 1)). See the state-vector-scaling
    section's docstring for the transformation definitions."""
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

    The extra term is the part that's easy to silently drop: because L_hat's own scaling
    factor Phi'_2(L)=1/sqrt(L^2+L_SCALE^2) is itself L-dependent (nonlinear, unlike r_hat's
    constant 1/R_SCALE), differentiating g=Phi'(y)*f(y) a second time picks up a genuine
    product-rule term from d(Phi')/dy, not just the rescaled df/dy. Derived by hand: writing
    L=L_SCALE*sinh(L_hat), d(Phi'_2)/dL * dL/d(L_hat) = -L/(L^2+L_SCALE^2) exactly - present
    ONLY in the (L_hat row, L_hat column) entry, since L_hat is the only nonlinearly-scaled
    component (r_hat's linear scaling has zero second derivative, contributing nothing here).
    """
    y = _to_physical(z)
    L = y[2]
    f = implicit_rhs_vectorized(x, y, state_prev, dt, alpha)      # physical RHS - needed for the row-2 correction term
    J_old = implicit_rhs_jacobian(x, y, state_prev, dt, alpha)    # (4,4,n), physical-space, delta-corrected

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


def make_bc_jacobian(m_min):
    """Builds the bc_jac(ya, yb) closure - solve_bvp's contract: returns (dbc_dya, dbc_dyb),
    each shape (4,4). Mirrors make_bc(m_min)'s physics exactly; see that function and the
    derivative helpers above. The 4 residuals split cleanly: res[0:2] depend only on ya,
    res[2:4] only on yb (block-diagonal by construction, not an approximation) - see
    make_bc's own derivation for why (the center and surface conditions don't couple)."""
    def bc_jac(ya, yb):
        P_a, T_a = np.exp(ya[1]), np.exp(ya[3])
        P_b, T_b = np.exp(yb[1]), np.exp(yb[3])
        r_b = yb[0]

        rho_a = eos.density(P_a, T_a, config.MU, config.MU_E)
        drho_dP_a, drho_dT_a = _eos_density_derivatives(P_a, T_a, rho_a)
        r_analytic = (3.0 * m_min / (4.0 * np.pi * rho_a)) ** (1.0 / 3.0)
        dr_analytic_drho = -r_analytic / (3.0 * rho_a)

        dbc_dya = np.zeros((4, 4))
        dbc_dya[0, 0] = 1.0
        dbc_dya[0, 1] = -dr_analytic_drho * drho_dP_a * P_a   # d/d(lnP_a) = P_a*d/dP_a
        dbc_dya[0, 3] = -dr_analytic_drho * drho_dT_a * T_a   # d/d(lnT_a) = T_a*d/dT_a
        dbc_dya[1, 2] = 1.0

        rho_b = eos.density(P_b, T_b, config.MU, config.MU_E)
        drho_dP_b, drho_dT_b = _eos_density_derivatives(P_b, T_b, rho_b)
        kappa_b = opacity.bell_lin_opacity(rho_b, T_b)
        dkappa_drho_b, dkappa_dT_b = _opacity_derivatives(rho_b, T_b, kappa_b)
        P_photo = boundary_conditions.photospheric_pressure(r_b, P_b, T_b, config.MU, config.MU_E)

        dPphoto_dr = -2.0 * P_photo / r_b
        dPphoto_dP = -P_photo * (dkappa_drho_b / kappa_b) * drho_dP_b
        dPphoto_dT = -P_photo * ((dkappa_drho_b / kappa_b) * drho_dT_b + dkappa_dT_b / kappa_b)

        dbc_dyb = np.zeros((4, 4))
        dbc_dyb[2, 0] = -dPphoto_dr / P_photo
        dbc_dyb[2, 1] = 1.0 - (dPphoto_dP / P_photo) * P_b
        dbc_dyb[2, 3] = -(dPphoto_dT / P_photo) * T_b
        dbc_dyb[3, 0] = -8.0 * np.pi * r_b * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        dbc_dyb[3, 2] = 1.0
        dbc_dyb[3, 3] = -16.0 * np.pi * r_b**2 * config.SIGMA_SB * T_b**4   # d/d(lnT_b) of -4*pi*r^2*sigma*(T^4-T_NEB^4)

        return dbc_dya, dbc_dyb
    return bc_jac


def verify_jacobians(state_0, bc, bc_jac, x, y_guess, alpha=1.0, n_test_points=8, rel_step=1.0e-6,
                      rhs_fn=implicit_rhs_vectorized, jac_fn=implicit_rhs_jacobian):
    """Cross-checks jac_fn (fun_jac) and bc_jac against central finite differences of rhs_fn/
    bc at several representative mesh points - REQUIRED before trusting either (module
    docstring: a wrong analytic Jacobian is worse than none). Raises loudly (AssertionError)
    on disagreement rather than silently proceeding - no forced numerical dampening of a real
    derivation error. rhs_fn/jac_fn default to the physical-space pair; pass
    implicit_rhs_scaled/implicit_rhs_jacobian_scaled for the scaled pathway (Milestone 6)."""
    rng = np.random.default_rng(0)
    test_idx = rng.choice(len(x), size=min(n_test_points, len(x)), replace=False)

    def fun_single(x_pt, y_pt):
        return rhs_fn(np.array([x_pt]), y_pt.reshape(4, 1), state_0, DT_RELAX, alpha)[:, 0]

    max_rel_err_fun = 0.0
    for idx in test_idx:
        x_pt, y_pt = x[idx], y_guess[:, idx].copy()
        f_pt = fun_single(x_pt, y_pt)
        J_analytic = jac_fn(np.array([x_pt]), y_pt.reshape(4, 1), state_0, DT_RELAX, alpha)[:, :, 0]
        J_fd = np.zeros((4, 4))
        for j in range(4):
            step = rel_step * max(abs(y_pt[j]), 1.0)
            y_plus, y_minus = y_pt.copy(), y_pt.copy()
            y_plus[j] += step
            y_minus[j] -= step
            J_fd[:, j] = (fun_single(x_pt, y_plus) - fun_single(x_pt, y_minus)) / (2.0 * step)
        # ROW-normalized relative error: |J_analytic-J_fd| / row's own matrix-norm scale.
        # Two false-alarm modes ruled out by hand-checking several "failing" points before
        # settling on this metric (both are verification-metric issues, not Jacobian bugs -
        # every entry matched to full displayed precision on manual inspection):
        # (1) entrywise |a-b|/max(|a|,|b|) blows up to 100% whenever BOTH a and b are
        #     legitimately near zero (e.g. d(grad_eff)/d(grad_rad) deep in the convective
        #     regime, where grad_eff locally flattens to the constant grad_ad - analytic
        #     rounds to exactly 0.0, FD picks up ~1e-8-scale noise instead; both "~0").
        # (2) normalizing by the OUTPUT value f_pt[i] fails when f_pt[i] is exactly 0 (e.g.
        #     dL_dm=0 exactly at the initial guess, since T=T_prev/P=P_prev there) - dividing
        #     by the 1e-30 floor then amplifies ordinary floating-point noise in that row's
        #     OTHER entries into a meaningless huge number.
        # Normalizing by the row's own (analytic vs FD) matrix-norm scale avoids both: a row
        # with any large entries is judged against that scale, not an unrelated near-zero one.
        row_scale = np.maximum(np.max(np.abs(J_analytic), axis=1), np.max(np.abs(J_fd), axis=1))
        row_scale = np.maximum(row_scale, 1.0e-30)
        rel_err = np.abs(J_analytic - J_fd) / row_scale[:, np.newaxis]
        max_rel_err_fun = max(max_rel_err_fun, np.max(rel_err))
        print(f"  [jac verify] fun_jac at mesh idx {idx} (m/M_TOTAL={np.exp(x_pt)/config.M_TOTAL:.4f}): "
              f"max row-normalized relative error vs FD = {np.max(rel_err):.4e}")

    y_a, y_b = y_guess[:, 0], y_guess[:, -1]
    dbc_dya_analytic, dbc_dyb_analytic = bc_jac(y_a, y_b)
    dbc_dya_fd, dbc_dyb_fd = np.zeros((4, 4)), np.zeros((4, 4))
    for j in range(4):
        step = rel_step * max(abs(y_a[j]), 1.0)
        ya_plus, ya_minus = y_a.copy(), y_a.copy()
        ya_plus[j] += step
        ya_minus[j] -= step
        dbc_dya_fd[:, j] = (np.asarray(bc(ya_plus, y_b)) - np.asarray(bc(ya_minus, y_b))) / (2.0 * step)
        step = rel_step * max(abs(y_b[j]), 1.0)
        yb_plus, yb_minus = y_b.copy(), y_b.copy()
        yb_plus[j] += step
        yb_minus[j] -= step
        dbc_dyb_fd[:, j] = (np.asarray(bc(y_a, yb_plus)) - np.asarray(bc(y_a, yb_minus))) / (2.0 * step)
    # Row-normalized, same fix and same reason as fun_jac's metric above (2026-08-07,
    # PROGRESS.md): confirmed at T=2000K, not assumed - the naive per-entry metric flagged
    # dbc_dyb[3,3] as a 4.4e-4 "disagreement" that turned out to be pure FD step-size noise
    # (the analytic value matches a LARGER FD step to 4e-6 relative, and progressively
    # WORSE-matches smaller steps - the textbook signature of finite-difference roundoff on
    # a quantity that's just legitimately tiny at this cooler T, L_expected~T^4, not a
    # formula error). Normalizing by each ROW's own matrix-norm scale avoids this exact
    # false alarm the same way it did for fun_jac.
    row_scale_a = np.maximum(np.max(np.abs(dbc_dya_analytic), axis=1), np.max(np.abs(dbc_dya_fd), axis=1))
    row_scale_a = np.maximum(row_scale_a, 1.0e-30)
    row_scale_b = np.maximum(np.max(np.abs(dbc_dyb_analytic), axis=1), np.max(np.abs(dbc_dyb_fd), axis=1))
    row_scale_b = np.maximum(row_scale_b, 1.0e-30)
    max_rel_err_bc = max(np.max(np.abs(dbc_dya_analytic - dbc_dya_fd) / row_scale_a[:, np.newaxis]),
                          np.max(np.abs(dbc_dyb_analytic - dbc_dyb_fd) / row_scale_b[:, np.newaxis]))
    print(f"  [jac verify] bc_jac: max relative error vs FD = {max_rel_err_bc:.4e}")
    print(f"  [jac verify] dbc_dya analytic=\n{dbc_dya_analytic}\n  dbc_dya FD=\n{dbc_dya_fd}")
    print(f"  [jac verify] dbc_dyb analytic=\n{dbc_dyb_analytic}\n  dbc_dyb FD=\n{dbc_dyb_fd}")

    return max_rel_err_fun, max_rel_err_bc


# ==========================================
# SECTION: Vectorization Smoke Test
# ==========================================

def smoke_test_vectorization(state_0, bc, rhs_fn=implicit_rhs_vectorized, mesh_fn=build_mesh_and_guess):
    """Cheap sanity check that rhs_fn/bc behave correctly on a small synthetic multi-point
    mesh BEFORE spending time on a real solve_bvp call - catches a shape-mismatch bug
    immediately rather than deep inside a slow, hard-to-diagnose run. rhs_fn/mesh_fn default
    to the physical-space versions; pass implicit_rhs_scaled/build_mesh_and_guess_scaled for
    the scaled pathway (PLAN_BVP.md Milestone 6)."""
    x, y_guess = mesh_fn(state_0)
    x_small = x[:5]
    y_small = y_guess[:, :5]
    dydx = rhs_fn(x_small, y_small, state_0, DT_RELAX, 1.0)
    assert dydx.shape == y_small.shape, f"RHS shape mismatch: dydx {dydx.shape} vs y {y_small.shape}"
    assert np.all(np.isfinite(dydx)), "RHS produced non-finite values on the smoke-test mesh"
    res = np.asarray(bc(y_guess[:, 0], y_guess[:, -1]))
    assert res.shape == (4,), f"bc() shape mismatch: {res.shape}"
    assert np.all(np.isfinite(res)), "bc() produced non-finite residuals"
    print(f"  smoke test OK: dydx.shape={dydx.shape}, bc residuals at initial guess={res}")


# ==========================================
# SECTION: Direct and Continuation Solve Attempts
# ==========================================

class _CrashedSolve:
    """Stand-in for scipy's solve_bvp OptimizeResult when fun() raises instead of solve_bvp
    itself returning a failed status - lets the SAME fallback logic (spot_check's
    `if sol.status != 0`) trigger either way, without silently swallowing the crash: the full
    exception is printed in full (traceback included) before this is constructed, never
    hidden - only the CONTROL FLOW is unified, not the reporting."""
    def __init__(self, x, y, exc):
        self.status = -99
        self.message = f"CRASHED: {type(exc).__name__}: {exc}"
        self.x = x
        self.y = y


def _safe_solve_bvp(fun, bc, x, y, verbose, fun_jac=None, bc_jac=None):
    """solve_bvp wrapped so an exception raised deep inside fun() (e.g. eos.density's own
    Newton-convergence assertion, triggered by solve_bvp's own internal Newton step
    overshooting into an unphysical (P,T) region - PROGRESS.md 2026-08-06 has the trace)
    becomes a reportable failed status instead of an uncaught crash, so the direct-then-
    continuation fallback strategy the plan called for actually runs. The full traceback is
    always printed first - this does not hide the failure, only lets the experiment continue
    past it.

    fun_jac/bc_jac (PLAN_BVP.md Milestone 4): analytic Jacobians, verified against finite
    differences (verify_jacobians) before ever being passed here - default None keeps
    scipy's own FD estimate for backward compatibility with Milestones 0-3's calls."""
    import traceback
    try:
        return solve_bvp(fun, bc, x, y, tol=SOLVE_BVP_TOL, max_nodes=SOLVE_BVP_MAX_NODES,
                          verbose=verbose, fun_jac=fun_jac, bc_jac=bc_jac)
    except Exception as exc:
        print(f"  *** solve_bvp raised during Newton iteration (not a clean failed status) - full traceback: ***")
        traceback.print_exc()
        return _CrashedSolve(x, y, exc)


def attempt_direct_solve(state_0, bc, x, y_guess, alpha=1.0, verbose=2, use_analytic_jac=False, bc_jac=None,
                          rhs_fn=implicit_rhs_vectorized, jac_fn=implicit_rhs_jacobian):
    def fun(x_, y_):
        return rhs_fn(x_, y_, state_0, DT_RELAX, alpha)
    fun_jac = (lambda x_, y_: jac_fn(x_, y_, state_0, DT_RELAX, alpha)) if use_analytic_jac else None
    t0 = time.time()
    sol = _safe_solve_bvp(fun, bc, x, y_guess, verbose, fun_jac=fun_jac, bc_jac=bc_jac if use_analytic_jac else None)
    elapsed = time.time() - t0
    return sol, elapsed


ALPHA_MAX = 1.0 - 1.0e-5   # PLAN_BVP.md Milestone 6 (2026-08-07): see attempt_continuation_solve's docstring


def attempt_continuation_solve(state_0, bc, x, y_guess,
                                alpha_steps=(0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, ALPHA_MAX),
                                verbose=2, use_analytic_jac=False, bc_jac=None,
                                rhs_fn=implicit_rhs_vectorized, jac_fn=implicit_rhs_jacobian):
    """Fallback if the direct alpha=1 solve fails (returns a failed status OR crashes - see
    _safe_solve_bvp): step alpha 0->1, warm-starting each solve_bvp call from the previous
    alpha's dense solution - the same homotopy IDEA bvp_solver.relax_initial_state uses, but
    each individual step is solved by solve_bvp's own global Newton iteration instead of LM
    wrapping a shooting integration.

    ALPHA_MAX, not exactly 1.0 (PLAN_BVP.md Milestone 6, 2026-08-07 - PROGRESS.md has the
    full trace): with scaling+analytic Jacobians, continuation converges CLEANLY through
    alpha=0.9999 (residuals to 1e-7, boundary residuals to machine precision) but the
    LITERAL alpha=1.0 diverges via exponentially escalating mesh refinement to NaN, every
    time, regardless of how small the preceding step is. Diagnosis: `dT_dm_real` (the real,
    Schwarzschild-selected gradient) is computed IDENTICALLY at every alpha>0 - the only
    difference between alpha=0.9999 and alpha=1.0 is whether a vanishingly small fraction of
    the smooth, constant adiabatic gradient is blended in. That alpha=0.9999 converges
    cleanly while alpha=1.0 does not is strong evidence the tiny adiabatic admixture acts as
    a REGULARIZER, damping a marginal instability in the pure unblended system that the
    blend was masking - not that the physics itself is different. ALPHA_MAX=1-1e-5 keeps
    that regularization at a quantifiably negligible level (0.001% adiabatic contamination)
    while resolving convergence. Not fully explained (why alpha=1.0 specifically is
    unstable, not just "more sensitive," remains open - flagged in PLAN_BVP.md as follow-up,
    not blocking).
    """
    x_curr, y_curr = x, y_guess
    total_elapsed = 0.0
    sol = None
    for alpha in alpha_steps:
        def fun(x_, y_, alpha=alpha):
            return rhs_fn(x_, y_, state_0, DT_RELAX, alpha)
        fun_jac = (lambda x_, y_, alpha=alpha: jac_fn(x_, y_, state_0, DT_RELAX, alpha)) if use_analytic_jac else None
        t0 = time.time()
        sol = _safe_solve_bvp(fun, bc, x_curr, y_curr, verbose, fun_jac=fun_jac, bc_jac=bc_jac if use_analytic_jac else None)
        elapsed = time.time() - t0
        total_elapsed += elapsed
        print(f"  continuation alpha={alpha:.2f}: status={sol.status}, message={sol.message}, "
              f"nodes={sol.x.size}, elapsed={elapsed:.1f}s")
        if sol.status != 0:
            return sol, total_elapsed
        x_curr, y_curr = sol.x, sol.y
    return sol, total_elapsed


# ==========================================
# SECTION: Per-Temperature Spot Check
# ==========================================

def spot_check(T_center, opacity_mode="bell_lin", use_analytic_jac=False, use_scaling=False):
    """opacity_mode: "bell_lin" (default, real Bell & Lin) or "toy" (PLAN_BVP.md Milestone
    1 - a single smooth power law, no regime switches - see toy_opacity/opacity_override
    above). The override wraps EVERYTHING under "toy", including state_0's own
    construction: solve_static_structure's photospheric BC (boundary_conditions.
    photospheric_pressure) also calls opacity.bell_lin_opacity, so a toy-opacity seed built
    under the REAL opacity would be inconsistent with a toy-opacity solve_bvp attempt -
    the whole seed must be rebuilt under the same opacity the solve itself uses, hence a
    separate cache file per mode.

    use_analytic_jac (PLAN_BVP.md Milestone 4): supplies fun_jac/bc_jac instead of scipy's
    default finite-difference estimate.

    use_scaling (PLAN_BVP.md Milestone 6, 2026-08-07): solves in the nondimensionalized
    state z=[r_hat,lnP,L_hat,lnT] instead of y=[r,lnP,L,lnT] - see the state-vector-scaling
    section's docstring. Reporting (r_surface, L_surface below) always converts back to
    physical units regardless of this flag."""
    print(f"\n{'=' * 70}\nSPOT CHECK: T_center = {T_center} K, opacity_mode={opacity_mode}, "
          f"analytic_jac={use_analytic_jac}, scaling={use_scaling}\n{'=' * 70}", flush=True)
    config.T_CENTER_INITIAL = T_center   # runtime-only override - see module docstring

    rhs_fn = implicit_rhs_scaled if use_scaling else implicit_rhs_vectorized
    jac_fn = implicit_rhs_jacobian_scaled if use_scaling else implicit_rhs_jacobian
    mesh_fn = build_mesh_and_guess_scaled if use_scaling else build_mesh_and_guess
    make_bc_fn = make_bc_scaled if use_scaling else make_bc
    make_bc_jac_fn = make_bc_jacobian_scaled if use_scaling else make_bc_jacobian

    override = opacity_override(toy_opacity) if opacity_mode == "toy" else opacity_override(opacity.bell_lin_opacity)
    with override:
        suffix = "_toy" if opacity_mode == "toy" else ""
        cache_path = f"{CACHE_DIR}\\bvp_experiment_state0_{int(T_center)}K{suffix}.pkl"
        try:
            state_0 = dev_cache.load_state(cache_path)
            print(f"  loaded cached state_0 for T_center={T_center}K ({opacity_mode})", flush=True)
        except FileNotFoundError:
            state_0 = bvp_solver.solve_static_structure()
            dev_cache.save_state(state_0, cache_path)
            print(f"  cached state_0 for T_center={T_center}K ({opacity_mode})", flush=True)

        m_min = config.M_MIN_FRACTION * config.M_TOTAL
        bc = make_bc_fn(m_min)   # PLAN_BVP.md Milestone 2: self-consistent center BC, not a fixed r_seed
        smoke_test_vectorization(state_0, bc, rhs_fn=rhs_fn, mesh_fn=mesh_fn)

        x, y_guess = mesh_fn(state_0)

        bc_jac = None
        if use_analytic_jac:
            bc_jac = make_bc_jac_fn(m_min)
            print("  verifying analytic Jacobians against finite differences before use ...", flush=True)
            max_fun_err, max_bc_err = verify_jacobians(state_0, bc, bc_jac, x, y_guess, alpha=1.0,
                                                        n_test_points=15, rhs_fn=rhs_fn, jac_fn=jac_fn)
            assert max_fun_err < 1.0e-4 and max_bc_err < 1.0e-4, (
                f"analytic Jacobian verification FAILED (fun_jac err={max_fun_err:.3e}, "
                f"bc_jac err={max_bc_err:.3e}) - refusing to use an unverified Jacobian"
            )
            print(f"  Jacobian verification OK: max fun_jac error={max_fun_err:.3e}, "
                  f"max bc_jac error={max_bc_err:.3e}", flush=True)

        print("  attempting DIRECT solve at alpha=1.0 ...", flush=True)
        sol, elapsed = attempt_direct_solve(state_0, bc, x, y_guess, alpha=1.0, use_analytic_jac=use_analytic_jac,
                                             bc_jac=bc_jac, rhs_fn=rhs_fn, jac_fn=jac_fn)
        method = "direct"
        if sol.status != 0:
            print(f"  direct solve did not converge (status={sol.status}, {sol.message}) "
                  f"after {elapsed:.1f}s - falling back to alpha-continuation", flush=True)
            sol, elapsed = attempt_continuation_solve(state_0, bc, x, y_guess, use_analytic_jac=use_analytic_jac,
                                                       bc_jac=bc_jac, rhs_fn=rhs_fn, jac_fn=jac_fn)
            method = "continuation"

        # Residual computation stays INSIDE the override scope: bc() -> boundary_conditions.
        # boundary_conditions() -> photospheric_pressure() also calls opacity.bell_lin_opacity -
        # computing this after the override was restored would silently score a toy-opacity
        # solve against the REAL opacity's photospheric target, not the one actually solved for.
        if use_scaling:
            y_final_a, y_final_b = _to_physical(sol.y[:, 0]), _to_physical(sol.y[:, -1])
        else:
            y_final_a, y_final_b = sol.y[:, 0], sol.y[:, -1]
        r_b, P_b, L_b, T_b = y_final_b[0], np.exp(y_final_b[1]), y_final_b[2], np.exp(y_final_b[3])
        residuals = np.asarray(bc(sol.y[:, 0], sol.y[:, -1]))
        result = {
            "T_center": T_center, "opacity_mode": opacity_mode, "method": method, "use_scaling": use_scaling,
            "status": int(sol.status), "message": str(sol.message),
            "elapsed_s": elapsed, "n_nodes": int(sol.x.size),
            "r_surface_RJup": float(r_b / config.R_JUPITER_CM), "L_surface_erg_s": float(L_b),
            "P_surface": float(P_b), "T_surface": float(T_b),
            "residuals": residuals.tolist(), "max_residual": float(np.max(np.abs(residuals))),
            "all_finite": bool(np.all(np.isfinite(sol.y))),
        }
        print(f"  RESULT: status={sol.status} ({sol.message}), method={method}, nodes={sol.x.size}, "
              f"elapsed={elapsed:.1f}s, R_surface={result['r_surface_RJup']:.4f} R_Jup, "
              f"L_surface={L_b:.4e} erg/s, max_residual={result['max_residual']:.4e}", flush=True)
    return result, sol


if __name__ == "__main__":
    T_arg = float(sys.argv[1]) if len(sys.argv) > 1 else 13000.0
    opacity_arg = sys.argv[2] if len(sys.argv) > 2 else "bell_lin"
    jac_arg = (sys.argv[3] == "jac") if len(sys.argv) > 3 else False
    scaling_arg = (sys.argv[4] == "scaled") if len(sys.argv) > 4 else False
    result, sol = spot_check(T_arg, opacity_mode=opacity_arg, use_analytic_jac=jac_arg, use_scaling=scaling_arg)
    suffix = ("_toy" if opacity_arg == "toy" else "") + ("_jac" if jac_arg else "") + ("_scaled" if scaling_arg else "")
    out_path = f"{CACHE_DIR}\\bvp_experiment_result_{int(T_arg)}K{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to {out_path}")
