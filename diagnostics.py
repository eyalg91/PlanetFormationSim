# diagnostics.py — Post-solve physical diagnostics: global energy balance, opacity
# regime census, and an independent continuity-equation consistency check. Pure
# functions that compute and report on a SimulationState; unlike validation.py, this
# module is NOT a test suite (no asserts) - it is the runtime monitoring/reporting
# layer PLAN.md's architecture calls for, meant to be run after every future solve
# (time_stepper.py, Sub-tasks 7-8) so a physicist can see whether a solution still
# makes physical sense, not just whether the solver reported success.

import os

import matplotlib.pyplot as plt
import numpy as np

import config
import eos
import gradients
import opacity

PLOT_DIR = "diagnostic_plots"   # all diagnostic PNGs save here by default, not the project root

# ==========================================
# SECTION: Generalized Virial Balance (Unconfined)
# ==========================================

def virial_balance(state):
    """(E_grav, E_therm) [erg] for a self-gravitating ideal-gas envelope with negligible
    surface pressure.

    Integrating hydrostatic equilibrium dP/dr = -G*m*rho/r^2 by parts over the envelope
    (multiply by 4*pi*r^3, integrate 0 to R, integrate the dP/dr term by parts) gives
    E_grav + 3*(gamma-1)*E_therm = 4*pi*R^3*P_surface. Under Sub-task 5's photospheric outer
    BC, P_surface (~1e4 dyn/cm^2) is ~15 orders of magnitude below the interior energy scale
    (E_grav, E_therm ~1e42-1e43 erg for this structure), so the standard zero-surface-pressure
    limit applies:
    E_grav + 3*(gamma-1)*E_therm = 0   [erg]
    (reduces to the familiar 2*E_therm + E_grav = 0 only for a monatomic gas, gamma=5/3; kept
    general in gamma here since config.GAMMA=1.4). This function reports the two terms rather
    than asserting the balance itself (see run_diagnostics), since the point is to see the
    terms are commensurate and nearly cancel, not to chase numerical precision for its own sake.
    """
    # Gravitational self-energy, built up shell by shell: E_grav = -integral G*m/r dm   [erg]
    E_grav = -np.trapezoid(config.G * state.m / state.r, state.m)

    # Thermal (internal) energy: P = (gamma-1)*rho*u (ideal gas) => E_therm = integral u dm
    # = 1/(gamma-1) * integral (P/rho) dm   [erg]
    E_therm = np.trapezoid(state.P / state.rho, state.m) / (config.GAMMA - 1.0)

    return E_grav, E_therm


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
    E_grav, E_therm = virial_balance(state)
    thermal_term = 3.0 * (config.GAMMA - 1.0) * E_therm
    lhs = E_grav + thermal_term
    # Normalized against the scale of the terms actually being balanced, not an external
    # reference - P_neb is now ~15 orders of magnitude below the interior energy scale
    # (diagnostics.virial_balance docstring), so normalizing against it would be meaningless.
    imbalance = abs(lhs) / max(abs(E_grav), abs(thermal_term))

    print(f"diagnostics: t = {state.t:.4e} s")
    print("  Virial balance (unconfined): E_grav + 3*(gamma-1)*E_therm = 0")
    print(f"    E_grav               = {E_grav:.4e} erg")
    print(f"    3*(gamma-1)*E_therm  = {thermal_term:.4e} erg")
    print(f"    LHS total            = {lhs:.4e} erg")
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


# ==========================================
# SECTION: Visual Diagnostics (converged-state profile plots)
# ==========================================
# Per CLAUDE.md's preference for a visible check over a print-only one: run_diagnostics
# above reports scalar summaries, but for a compact, differentiated structure the profiles
# themselves are more informative. Each function takes a SimulationState and an output_path,
# matching validation.py's existing plt.subplots/savefig house style.

def plot_structure_profile(state, output_path=f"{PLOT_DIR}/structure_profile.png") -> None:
    """Temperature, density, and pressure vs Lagrangian mass coordinate - the primary visual
    sanity check on a converged structure."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    x = state.m / config.M_TOTAL
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(x, state.T)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("T [K]")
    axes[0].set_title(f"Structure profile at t={state.t:.3e} s")

    axes[1].plot(x, state.rho)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rho [g cm^-3]")

    axes[2].plot(x, state.P)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("P [dyn cm^-2]")
    axes[2].set_xlabel("m / M_TOTAL")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved structure profile plot to {output_path}")


def plot_mass_radius(state, output_path=f"{PLOT_DIR}/mass_radius.png") -> None:
    """Enclosed mass vs radius - shows how mass concentrates toward the center for this
    degenerate-supported structure (most of M_TOTAL sits well inside the outer radius)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(state.r / config.R_JUPITER_CM, state.m / config.M_TOTAL)
    ax.set_xlabel("r [R_Jup]")
    ax.set_ylabel("m / M_TOTAL")
    ax.set_title(f"Mass-radius distribution at t={state.t:.3e} s")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved mass-radius plot to {output_path}")


def plot_convective_zones(state, output_path=f"{PLOT_DIR}/convective_zones.png") -> None:
    """nabla_rad(m) vs nabla_ad (Schwarzschild criterion, gradients.effective_gradient), with
    convective zones shaded - visually confirms which layers are convective vs radiative.

    Excludes the innermost grid point: grad_radiative diverges as m->0 (removable 0/0 at the
    exact center, module docstring), and callers must not evaluate it there.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m, L, P, T, rho = state.m[1:], state.L[1:], state.P[1:], state.T[1:], state.rho[1:]
    kappa = opacity.bell_lin_opacity(rho, T)
    grad_ad = eos.grad_adiabatic(config.GAMMA)
    grad_rad = gradients.grad_radiative(L, m, P, T, kappa)
    _grad_eff, is_convective = gradients.effective_gradient(grad_rad, grad_ad)

    x = m / config.M_TOTAL
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, grad_rad, label="nabla_rad", color="C0")
    ax.axhline(grad_ad, color="k", linestyle="--", label="nabla_ad")
    ax.fill_between(x, 0, 1, where=is_convective, transform=ax.get_xaxis_transform(),
                     color="C0", alpha=0.15, label="convective")
    ax.set_yscale("log")
    ax.set_xlabel("m / M_TOTAL")
    ax.set_ylabel("nabla (dlnT/dlnP)")
    ax.set_title(f"Convective vs radiative zones at t={state.t:.3e} s")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved convective/radiative zone plot to {output_path}")


def plot_diagnostics(state, output_dir=PLOT_DIR) -> None:
    """Generate all three visual diagnostic plots for a converged SimulationState."""
    plot_structure_profile(state, f"{output_dir}/structure_profile.png")
    plot_mass_radius(state, f"{output_dir}/mass_radius.png")
    plot_convective_zones(state, f"{output_dir}/convective_zones.png")
