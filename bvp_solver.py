# bvp_solver.py — Solves the t=0 static envelope structure (the boundary value problem
# PLAN.md assigns to this module) and returns a populated SimulationState.
#
# PHYSICAL NOTE: this module only ever solves the t=0 static structure (zero frozen
# time-derivative source terms dT_dt = dP_dt = 0, since there is no previous timestep to
# difference against yet). With those sources zero, the energy equation gives dL/dm = 0
# identically (odes.py); combined with the center BC L(0)=0, L(m)=0 everywhere. With L=0,
# nabla_rad=0 (gradients.py) is always < nabla_ad, so the Schwarzschild criterion never
# trips convective and dT/dm=0: the converged t=0 envelope is exactly isothermal at T_neb
# and carries no luminosity. This is a hard mathematical consequence of odes.py's equations,
# not a numerical artifact - and it is also the physically correct t=0 state for this
# scenario (gravitational-instability disk fragmentation, PLAN.md Sec 1): confirmed against
# config.P_NEB, config.T_NEB via the Bonnor-Ebert criterion, M_TOTAL/M_BE ~ 0.089 (deeply
# subcritical), a freshly-fragmented clump sits in stable, COLD, non-luminous hydrostatic
# equilibrium with its surrounding disk, exactly matching the literature picture for
# newly-fragmented GI clumps (Boss, Mayer, Helled & Bodenheimer et al.: extended, tens-of-AU,
# near-disk-temperature clumps that only contract to planetary size over the much longer
# Kelvin-Helmholtz timescale). A "hot start" (adiabatic interior, T_center ~600-800K) was
# tested four independent ways (single adiabat + assumed L(m); single adiabat + L solved for
# marginal convection; two-zone core+envelope with L reverse-derived from a chosen dT/dm;
# a photospheric L=4*pi*R^2*sigma*T_eff^4 construction) and ALL FOUR require L ~1e34-1e37
# erg/s (thousands to tens of thousands of solar luminosities) - physically impossible for a
# sub-Jupiter-mass clump. This is not a construction flaw: M_TOTAL being deeply sub-critical
# forces the pressure profile to be nearly uniform regardless of the assumed temperature
# distribution, and cramming a large T range into that narrow a P range demands an enormous
# |dT/dm|, which the radiative diffusion equation says demands an enormous L to sustain.
# Introducing nonzero L to actually start Kelvin-Helmholtz contraction is therefore not a t=0
# state-construction problem - it is a time_stepper.py (Sub-tasks 7-8) bootstrap problem: the
# first REAL evolutionary step needs a literature-motivated assumed initial cooling rate
# (not "return zero arrays", which would leave the envelope sitting at this exact fixed point
# forever, since two identical states difference to zero). That bootstrap is out of scope here.
#
# NUMERICAL NOTE: because L and T are therefore already known exactly, only the reduced,
# well-posed 2-ODE system for (r, P) needs to be solved numerically, holding L=0 and
# T=T_neb fixed. That reduction is not just a simplification of convenience: the full
# 4-ODE Jacobian is structurally rank-deficient here (confirmed by direct computation:
# rank 2 of 4, unchanged for any dT_dt/dP_dt, since dL/dm depends only on those externally
# prescribed source arrays, never on the current state). scipy.integrate.solve_bvp's Newton
# iteration hit a singular Jacobian from this every time. Solving even the reduced 2-ODE
# system via solve_bvp's collocation method turned out to be unreliable too, for a second,
# independent reason: near the surface, P approaches the much smaller P_neb with a modest
# absolute gradient but a very short pressure scale height (a genuine, short-lived boundary
# layer), which drives d(ln P)/d(ln m) to enormous values there. No combination tried -
# linear or log mass grid, log or linear independent variable, mesh resolution up to 8000
# points - kept solve_bvp's global collocation Jacobian well behaved through its first
# iteration. This module instead uses a shooting method: integrate the 2-ODE system outward
# from the center with scipy.integrate.solve_ivp (an adaptive stiff integrator, which handles
# the wide dynamic range and the surface boundary layer natively, with no global Jacobian)
# starting from a trial central pressure, and root-find on that pressure until the surface
# condition P(M_TOTAL) = P_neb is met. This is a genuine, deliberate deviation from PLAN.md's
# stated "call solve_bvp" deliverable for this module, made after solve_bvp was confirmed
# unreliable for this specific problem across many configurations.

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import config
import eos
import odes
import state

# ==========================================
# SECTION: Reduced (r, P) Right-Hand Side
# ==========================================

def _reduced_rhs_logm(x, y):
    """dr/dx, dP/dx at fixed L=0, T=T_neb, x = ln(m); via odes.stellar_odes (single source of truth).

    solve_ivp's adaptive step control needs a bounded-range independent variable: m itself spans
    ~6 decades (M_MIN_FRACTION*M_TOTAL to M_TOTAL), which drove step sizes below floating-point
    spacing near the small end. x = ln(m) instead ranges over ~13.8 units; dy/dx = dy/dm * dm/dx,
    dm/dx = m.
    """
    m = np.exp(x)
    r, P = y
    y_full = np.array([[r], [P], [0.0], [config.T_NEB]])
    zero_source = np.zeros(1)
    dr_dm, dP_dm, _, _ = odes.stellar_odes(np.array([m]), y_full, zero_source, zero_source)
    return [dr_dm[0] * m, dP_dm[0] * m]


# ==========================================
# SECTION: Shooting Integration
# ==========================================

def _integrate_outward(P_center, x_span, r_start):
    """Integrate (r, P) from x_span[0] (near center, r=r_start) to x_span[1] (surface) for a trial P_center.

    ASSUMPTION: atol must be tight relative to the actual target scales (P_neb here), not a
    fixed generic value - an earlier attempt used atol ~ 1e-2*P_neb, comparable to the target
    itself, which produced pure numerical noise masquerading as a non-monotonic P_center-to-
    surface-pressure relationship. r's atol is set far below any physically relevant radius, so
    rtol dominates there; P's atol is tied to config.BVP_TOL relative to config.P_NEB.
    """
    return solve_ivp(
        _reduced_rhs_logm, x_span, [r_start, P_center], method="Radau", dense_output=True,
        rtol=config.BVP_TOL, atol=[1.0, config.P_NEB * config.BVP_TOL],
    )


# ==========================================
# SECTION: Static (t=0) Structure via Shooting
# ==========================================

def solve_static_structure() -> state.SimulationState:
    """Solve the t=0 static envelope structure and return a populated SimulationState.

    Shoots on the central pressure P_center until the surface condition P(M_TOTAL) = P_neb is
    satisfied (bisection via brentq); L=0 and T=T_neb are assigned directly (module docstring).
    """
    m_min = config.M_MIN_FRACTION * config.M_TOTAL
    x_span = (np.log(m_min), np.log(config.M_TOTAL))

    R_guess = (3.0 * config.M_TOTAL / (4.0 * np.pi * config.RHO_GUESS_INITIAL)) ** (1.0 / 3.0)
    r_start = R_guess * (m_min / config.M_TOTAL) ** (1.0 / 3.0)
    P_center_guess = config.P_NEB + (2.0 / 3.0) * np.pi * config.G * config.RHO_GUESS_INITIAL**2 * R_guess**2

    def surface_pressure_error(P_center):
        sol = _integrate_outward(P_center, x_span, r_start)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed during shooting at P_center={P_center:.6e}: {sol.message}")
        return sol.y[1, -1] - config.P_NEB

    # P must decrease monotonically outward (dP/dm < 0 everywhere), so the surface pressure is a
    # monotonically increasing function of P_center: P_center=P_neb itself already undershoots
    # (verified: surface pressure ~0.83*P_neb there), and the constant-density hydrostatic guess
    # P_center_guess overshoots comfortably (verified: surface pressure ~55*P_neb there), while
    # staying well within the range solve_ivp integrates reliably (empirically stable up to
    # ~500x P_center_guess; failures/step-size collapse only appear far beyond that).
    P_low, P_high = config.P_NEB, P_center_guess
    P_center = brentq(surface_pressure_error, P_low, P_high, xtol=config.P_NEB * 1.0e-10, rtol=config.BVP_TOL)

    sol = _integrate_outward(P_center, x_span, r_start)
    residual_norm = abs(sol.y[1, -1] - config.P_NEB) / config.P_NEB
    print(f"bvp_solver: shooting converged, P_center={P_center:.6e} dyn/cm^2, "
          f"surface P relative residual={residual_norm:.3e}, solve_ivp steps={sol.t.size}")

    m = np.logspace(np.log10(m_min), np.log10(config.M_TOTAL), config.N_GRID_POINTS)
    r, P = sol.sol(np.log(m))
    L = np.zeros_like(m)
    T = np.full_like(m, config.T_NEB)
    rho = eos.density(P, T, config.MU)

    return state.SimulationState(m=m, r=r, P=P, L=L, T=T, rho=rho, t=0.0, prev=None)
