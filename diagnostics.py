# diagnostics.py — Post-solve physical diagnostics: global energy balance, opacity
# regime census, and an independent continuity-equation consistency check. Pure
# functions that compute and report on a SimulationState; unlike validation.py, this
# module is NOT a test suite (no asserts) - it is the runtime monitoring/reporting
# layer PLAN.md's architecture calls for, meant to be run after every future solve
# (time_stepper.py, Sub-tasks 7-8) so a physicist can see whether a solution still
# makes physical sense, not just whether the solver reported success.

import numpy as np

import config
import opacity

# ==========================================
# SECTION: Generalized Virial Balance (Pressure-Confined)
# ==========================================

def virial_balance(state):
    """(E_grav, E_therm, surface_term) [erg] for a pressure-confined ideal-gas envelope.

    The standard zero-surface-pressure virial theorem (2*E_therm + E_grav = 0 for a monatomic
    gas) does not apply here: P_neb is not negligible - it is the entire reason the envelope
    has its size and structure (Bonnor-Ebert confinement, PLAN.md Sec 4.6). Integrating
    hydrostatic equilibrium dP/dr = -G*m*rho/r^2 by parts over the envelope instead gives the
    pressure-confined form (derivation: multiply by 4*pi*r^3, integrate 0 to R, integrate the
    dP/dr term by parts):
    E_grav + 3*(gamma-1)*E_therm = 3*P_neb*V   [erg]
    which is exactly the Bonnor-Ebert virial balance already used to validate Sub-task 5's
    T_NEB/P_NEB (PROGRESS.md); this function reports its three terms rather than asserting
    the balance itself (see run_diagnostics), since the point is to see which physical term
    dominates, not to chase numerical precision for its own sake.
    """
    # Gravitational self-energy, built up shell by shell: E_grav = -integral G*m/r dm   [erg]
    E_grav = -np.trapezoid(config.G * state.m / state.r, state.m)

    # Thermal (internal) energy: P = (gamma-1)*rho*u (ideal gas) => E_therm = integral u dm
    # = 1/(gamma-1) * integral (P/rho) dm   [erg]
    E_therm = np.trapezoid(state.P / state.rho, state.m) / (config.GAMMA - 1.0)

    # Surface confinement term: 3*P_neb*V, V = (4/3)*pi*R_surface^3   [erg]
    R_surface = state.r[-1]
    V = (4.0 / 3.0) * np.pi * R_surface**3
    surface_term = 3.0 * config.P_NEB * V

    return E_grav, E_therm, surface_term


# ==========================================
# SECTION: Opacity Regime Census
# ==========================================

def opacity_regime_distribution(state):
    """Fraction of grid points in each of the 8 Bell & Lin (1994) opacity regimes [dimensionless]."""
    regime_index = opacity.determine_regime(state.rho, state.T)
    n_points = regime_index.size
    return np.array([np.sum(regime_index == idx) / n_points for idx in range(len(opacity.REGIMES))])


# ==========================================
# SECTION: Mass Reconstruction from the Continuity Equation
# ==========================================

def mass_reconstruction(state):
    """M(r) = m[0] + integral 4*pi*r^2*rho dr, reconstructed from the converged (r, rho)
    profile via cumulative trapezoidal quadrature [g].

    Independent check of dr/dm = 1/(4*pi*r^2*rho) (odes.py's continuity equation) and the
    shooting integration together: this is the inverse relation of the same ODE, computed by
    a completely different numerical method (quadrature over the converged profile, not the
    adaptive ODE integrator that produced it), so systematic bugs in either should show up as
    a mismatch against the Lagrangian grid state.m.
    """
    dM_dr = 4.0 * np.pi * state.r**2 * state.rho
    M_cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (dM_dr[1:] + dM_dr[:-1]) * np.diff(state.r))])
    return state.m[0] + M_cumulative


# ==========================================
# SECTION: Runtime Diagnostic Report
# ==========================================

def run_diagnostics(state) -> None:
    """Print a physical diagnostic report for a converged SimulationState."""
    E_grav, E_therm, surface_term = virial_balance(state)
    lhs = E_grav + 3.0 * (config.GAMMA - 1.0) * E_therm
    imbalance = abs(lhs - surface_term) / abs(surface_term)

    print(f"diagnostics: t = {state.t:.4e} s")
    print("  Virial balance: E_grav + 3*(gamma-1)*E_therm = 3*P_neb*V")
    print(f"    E_grav               = {E_grav:.4e} erg")
    print(f"    3*(gamma-1)*E_therm  = {3.0 * (config.GAMMA - 1.0) * E_therm:.4e} erg")
    print(f"    LHS total            = {lhs:.4e} erg")
    print(f"    3*P_neb*V (surface)  = {surface_term:.4e} erg")
    print(f"    relative imbalance   = {imbalance:.3e}")

    regime_fractions = opacity_regime_distribution(state)
    print("  Opacity regime distribution:")
    for regime, fraction in zip(opacity.REGIMES, regime_fractions):
        if fraction > 0.0:
            print(f"    {regime.name:<32s} {fraction:.1%}")

    M_recon = mass_reconstruction(state)
    rel_err = np.abs((M_recon - state.m) / state.m)
    print(f"  Mass reconstruction: max relative error = {rel_err.max():.3e} "
          f"(interior points away from center: {rel_err[30:].max():.3e})")
