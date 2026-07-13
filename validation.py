# validation.py — Sanity checks, unit-consistency verification, and
# convergence criteria for PlanetFormationSim. Per CLAUDE.md, all validation
# and testing logic lives here, never inside operational physics or solver
# modules (odes.py, bvp_solver.py, time_stepper.py, etc.).

import matplotlib.pyplot as plt
import numpy as np

import config
import eos
import opacity

# ==========================================
# SECTION: CGS Unit Consistency — Ideal Gas EOS
# ==========================================

def check_ideal_gas_eos() -> None:
    """Confirm P = rho*k_B*T/(mu*m_H) yields dyn/cm^2 when all inputs are CGS."""
    rho_test = 1.0e-6   # Representative envelope density [g cm^-3]
    T_test = 500.0      # Representative envelope temperature [K]

    # Ideal gas EOS: P = rho * k_B * T / (mu * m_H)   [dyn cm^-2]
    # Unit check: [g cm^-3] * [erg K^-1] * [K] / [g] = [erg cm^-3] = [dyn cm^-2]
    P_test = rho_test * config.K_B * T_test / (config.MU * config.M_H)

    print("Check 1 - Ideal gas EOS: P = rho*k_B*T/(mu*m_H)")
    print(f"  rho = {rho_test:.3e} g/cm^3, T = {T_test:.3e} K, mu = {config.MU}")
    print(f"  -> P = {P_test:.3e} dyn/cm^2 (erg/cm^3)")
    assert P_test > 0.0 and np.isfinite(P_test), "Ideal gas EOS produced a non-physical pressure"


# ==========================================
# SECTION: CGS Unit Consistency — Hydrostatic Equilibrium
# ==========================================

def check_hydrostatic_equilibrium() -> None:
    """Confirm dP/dm = -G*m/(4*pi*r^4) yields dyn cm^-2 g^-1 when all inputs are CGS."""
    m_test = 1.0e29   # Representative enclosed mass [g]
    r_test = 5.0e9    # Representative radius [cm]

    # Hydrostatic equilibrium: dP/dm = -G*m / (4*pi*r^4)   [dyn cm^-2 g^-1]
    # Unit check: [cm^3 g^-1 s^-2] * [g] / [cm^4] = [g cm^-2 s^-2] = [dyn cm^-2] per gram
    dPdm_test = -config.G * m_test / (4.0 * np.pi * r_test**4)

    print("Check 2 - Hydrostatic equilibrium: dP/dm = -G*m/(4*pi*r^4)")
    print(f"  m = {m_test:.3e} g, r = {r_test:.3e} cm")
    print(f"  -> dP/dm = {dPdm_test:.3e} dyn cm^-2 g^-1")
    assert dPdm_test < 0.0 and np.isfinite(dPdm_test), "Hydrostatic gradient has the wrong sign or is non-finite"


# ==========================================
# SECTION: CGS Unit Consistency — Continuity (Mass-Radius) Relation
# ==========================================

def check_continuity_equation() -> None:
    """Confirm dr/dm = 1/(4*pi*r^2*rho) yields cm/g when all inputs are CGS."""
    r_test = 5.0e9     # Representative radius [cm]
    rho_test = 1.0e-6  # Representative density [g cm^-3]

    # Continuity / mass-radius relation: dr/dm = 1/(4*pi*r^2*rho)   [cm g^-1]
    # Unit check: 1 / ([cm^2] * [g cm^-3]) = 1 / [g cm^-1] = [cm g^-1]
    drdm_test = 1.0 / (4.0 * np.pi * r_test**2 * rho_test)

    print("Check 3 - Continuity equation: dr/dm = 1/(4*pi*r^2*rho)")
    print(f"  r = {r_test:.3e} cm, rho = {rho_test:.3e} g/cm^3")
    print(f"  -> dr/dm = {drdm_test:.3e} cm/g")
    assert drdm_test > 0.0 and np.isfinite(drdm_test), "Continuity relation produced a non-physical radius gradient"


# ==========================================
# SECTION: EOS — Ideal Gas Density Inversion
# ==========================================

def check_ideal_gas_density_inverts_pressure() -> None:
    """Confirm eos.density(P, T, mu) reproduces a hand-solved rho from the ideal gas law."""
    P_test = 1.0e4    # Test pressure [dyn cm^-2]
    T_test = 150.0    # Test temperature [K]
    mu_test = 2.34    # Mean molecular weight, H2/He mix [dimensionless]

    # Hand-solved: rho = P*mu*m_H/(k_B*T)  [g cm^-3]
    rho_expected = P_test * mu_test * config.M_H / (config.K_B * T_test)
    rho_computed = eos.density(P_test, T_test, mu_test)

    print("Check 4 - eos.density() inverts the ideal gas law")
    print(f"  P = {P_test:.3e} dyn/cm^2, T = {T_test:.3e} K, mu = {mu_test}")
    print(f"  -> rho_computed = {rho_computed:.6e} g/cm^3, rho_expected = {rho_expected:.6e} g/cm^3")
    assert np.isclose(rho_computed, rho_expected, rtol=1e-12), "eos.density() does not match hand-solved ideal gas law"


# ==========================================
# SECTION: EOS — Adiabatic Gradient and Specific Heat Limits
# ==========================================

def check_adiabatic_gradient_and_cp_limits() -> None:
    """Confirm grad_adiabatic and specific_heat_cp at the two gamma values used in this project."""
    gamma_monatomic = 5.0 / 3.0   # Reference case: monatomic ideal gas [dimensionless]
    gamma_diatomic = config.GAMMA  # Actual project value: diatomic H2/He mix [dimensionless]
    mu_test = config.MU

    # nabla_ad = (gamma - 1)/gamma  [dimensionless]
    grad_ad_monatomic = eos.grad_adiabatic(gamma_monatomic)
    grad_ad_diatomic = eos.grad_adiabatic(gamma_diatomic)

    print("Check 5 - eos.grad_adiabatic() and eos.specific_heat_cp() at reference gamma values")
    print(f"  gamma = 5/3 (monatomic) -> nabla_ad = {grad_ad_monatomic:.6f} (expected 0.400000)")
    print(f"  gamma = {gamma_diatomic} (diatomic, config.GAMMA) -> nabla_ad = {grad_ad_diatomic:.6f} (expected {(gamma_diatomic - 1.0) / gamma_diatomic:.6f})")
    assert np.isclose(grad_ad_monatomic, 0.4, rtol=1e-12), "grad_adiabatic(5/3) should equal 0.4 for a monatomic gas"

    cp_monatomic = eos.specific_heat_cp(gamma_monatomic, mu_test)
    cp_diatomic = eos.specific_heat_cp(gamma_diatomic, mu_test)
    print(f"  c_p(gamma=5/3)  = {cp_monatomic:.6e} erg g^-1 K^-1")
    print(f"  c_p(gamma={gamma_diatomic}) = {cp_diatomic:.6e} erg g^-1 K^-1")
    assert cp_monatomic > 0.0 and np.isfinite(cp_monatomic), "specific_heat_cp is non-physical for monatomic gas"
    assert cp_diatomic > 0.0 and np.isfinite(cp_diatomic), "specific_heat_cp is non-physical for diatomic gas"


# ==========================================
# SECTION: Opacity — Regime Table Reference Points
# ==========================================

def check_regime_table_reference_points() -> None:
    """Confirm each Bell & Lin (1994) regime evaluates to its own kappa_i at rho = T = 1 (cgs)."""
    rho_ref = 1.0   # Reference density, chosen so rho^a = 1 for any a [g cm^-3]
    T_ref = 1.0     # Reference temperature, chosen so T^b = 1 for any b [K]

    print("Check 6 - opacity.REGIMES table reference-point evaluation")
    for regime in opacity.REGIMES:
        kappa = opacity.evaluate_regime(regime.kappa_i, regime.a, regime.b, rho_ref, T_ref)
        print(f"  {regime.name:<32s} kappa_i = {regime.kappa_i:.3e} -> kappa(1,1) = {kappa:.3e}")
        assert np.isclose(kappa, regime.kappa_i, rtol=1e-12), (
            f"Regime '{regime.name}' does not evaluate to its own kappa_i at (rho, T) = (1, 1)"
        )
    assert len(opacity.REGIMES) == 8, "Bell & Lin (1994) opacity law must have exactly 8 regimes"


# ==========================================
# SECTION: Opacity — Transition Temperature Log-Log Slopes
# ==========================================

def check_transition_temperature_loglog_slopes() -> None:
    """Confirm each transition_temperature(rho, n) has the analytic log-log slope in rho."""
    rho_lo = 1.0e-12   # Low-density probe point for slope estimation [g cm^-3]
    rho_hi = 1.0e-8    # High-density probe point for slope estimation [g cm^-3]

    print("Check 7 - opacity.transition_temperature() log-log slope vs analytic exponent")
    for n in range(len(opacity.REGIMES) - 1):
        lower, upper = opacity.REGIMES[n], opacity.REGIMES[n + 1]

        # Analytic slope: dlog(T)/dlog(rho) = (a_(n+1) - a_n) / (b_n - b_(n+1))  [dimensionless]
        analytic_slope = (upper.a - lower.a) / (lower.b - upper.b)

        T_lo = opacity.transition_temperature(rho_lo, n)
        T_hi = opacity.transition_temperature(rho_hi, n)
        empirical_slope = (np.log10(T_hi) - np.log10(T_lo)) / (np.log10(rho_hi) - np.log10(rho_lo))

        print(f"  {n}->{n + 1} {lower.name:<24s} -> {upper.name:<24s} "
              f"analytic = {analytic_slope:+.6f}, empirical = {empirical_slope:+.6f}")
        assert np.isclose(empirical_slope, analytic_slope, atol=1e-9), (
            f"Transition {n}->{n + 1} log-log slope does not match analytic exponent"
        )


# ==========================================
# SECTION: Opacity — Transition Temperature Diagnostic Plot
# ==========================================

def plot_transition_temperatures(output_path="opacity_transitions.png") -> None:
    """Save a log-log plot of T_(n->n+1)(rho) for all 7 Bell & Lin transitions, rho in [1e-15, 1e-5] g/cm^3."""
    rho = np.logspace(-15, -5, 200)   # Density sweep for visual inspection [g cm^-3]

    fig, ax = plt.subplots(figsize=(7, 5))
    for n in range(len(opacity.REGIMES) - 1):
        lower, upper = opacity.REGIMES[n], opacity.REGIMES[n + 1]
        T = opacity.transition_temperature(rho, n)
        ax.plot(rho, T, label=f"{n}->{n + 1}: {lower.name} -> {upper.name}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rho [g cm^-3]")
    ax.set_ylabel("Transition temperature T [K]")
    ax.set_title("Bell & Lin (1994) opacity regime transition temperatures")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved opacity transition-temperature plot to {output_path}")


# ==========================================
# SECTION: Opacity — Regime Continuity at Transitions
# ==========================================

def check_regime_continuity() -> None:
    """Confirm adjacent regimes agree at their own analytic transition temperature."""
    rho_test = 1.0e-10   # Representative envelope density [g cm^-3]
    n_transitions = len(opacity.REGIMES) - 1

    print("Check 8 - Regime continuity: kappa agrees across each of the 7 transitions")
    for n in range(n_transitions):
        lower, upper = opacity.REGIMES[n], opacity.REGIMES[n + 1]
        T_transition = opacity.transition_temperature(rho_test, n)

        kappa_lower = opacity.evaluate_regime(lower.kappa_i, lower.a, lower.b, rho_test, T_transition)
        kappa_upper = opacity.evaluate_regime(upper.kappa_i, upper.a, upper.b, rho_test, T_transition)
        rel_diff = abs(kappa_lower - kappa_upper) / abs(kappa_lower)

        print(f"  {n}->{n + 1} {lower.name:<24s} -> {upper.name:<24s} "
              f"T = {T_transition:.4e} K, rel_diff = {rel_diff:.3e}")
        assert rel_diff < 1e-10, f"Regime {n}->{n + 1} is discontinuous at its own transition temperature"


# ==========================================
# SECTION: Opacity — Regime Ordering with Temperature
# ==========================================

def check_regime_ordering_monotonic() -> None:
    """Confirm regime index is monotonically non-decreasing in T, at every density in the grid.

    A single-density check (e.g. rho = 1e-10 g/cm^3 alone) missed a real bug: the raw
    Kramers -> electron-scattering transition (n=6) falls to ~179 K at rho = 1e-15 g/cm^3,
    below several cooler regimes' own transitions, which corrupted the low-T regime
    assignment at low density. This check sweeps the full physically relevant (rho, T) grid
    so a regression at any single density is caught.
    """
    rho_sweep = np.logspace(-15, -5, 21)         # Density sweep [g cm^-3]
    T_sweep = np.linspace(100.0, 50000.0, 500)   # Temperature sweep at each density [K]

    print("Check 9 - Regime ordering: index is non-decreasing over T, for every rho in [1e-15, 1e-5] g/cm^3")
    worst_violation = 0
    for rho_test in rho_sweep:
        regime_indices = opacity.determine_regime(rho_test, T_sweep)
        step = np.diff(regime_indices)
        worst_violation = min(worst_violation, step.min())
        assert np.all(step >= 0), (
            f"Regime index decreases somewhere in the T sweep at rho = {rho_test:.3e} g/cm^3"
        )

    # Regression check for the specific reported failure: cold, very low-density gas must not
    # be misassigned to a hot regime (e.g. electron scattering) via a spurious transition value.
    regime_at_bug_point = opacity.determine_regime(1.0e-15, 190.0)
    print(f"  regression point (rho=1e-15 g/cm^3, T=190 K) -> regime index {regime_at_bug_point} "
          f"({opacity.REGIMES[int(regime_at_bug_point)].name})")
    assert regime_at_bug_point == 1, "Cold, low-density gas at (rho=1e-15, T=190K) should be Ice grain evaporation"

    print(f"  swept {len(rho_sweep)} densities x {len(T_sweep)} temperatures, worst step = {worst_violation} (>= 0 required)")


# ==========================================
# SECTION: Opacity — Vectorization Stress Test
# ==========================================

def check_bell_lin_vectorization_stress_test() -> None:
    """Confirm bell_lin_opacity handles a 2D mesh spanning all 8 regimes with no NaN/Inf, no shape loss.

    Note: Sub-task 2d also calls for a reference-point check against Bell & Lin (1994)'s own
    tabulated values. The paper's only such comparison is Figure 9a (kappa(T) curves at several
    fixed rho), a plotted figure with no accompanying numeric table, so it cannot be digitized
    reliably from the text; the 0->1 transition temperature (166.81 K), matched independently in
    Sub-task 2b, is the closest available hand-verifiable reference point from this paper.
    """
    rho_mesh, T_mesh = np.meshgrid(
        np.logspace(-14, -4, 60),   # Density sweep spanning all 8 regimes [g cm^-3]
        np.logspace(1.5, 6.0, 60),  # Temperature sweep spanning all 8 regimes [K]
    )

    kappa_mesh = opacity.bell_lin_opacity(rho_mesh, T_mesh)
    regime_mesh = opacity.determine_regime(rho_mesh, T_mesh)

    print("Check 10 - Vectorization stress test: 2D mesh across all 8 regimes")
    print(f"  input shape = {rho_mesh.shape}, output shape = {kappa_mesh.shape}")
    print(f"  regimes present in mesh: {sorted(np.unique(regime_mesh))}")
    assert kappa_mesh.shape == rho_mesh.shape, "bell_lin_opacity output shape does not match input shape"
    assert not np.any(np.isnan(kappa_mesh)), "bell_lin_opacity produced NaN over the physically relevant domain"
    assert not np.any(np.isinf(kappa_mesh)), "bell_lin_opacity produced Inf over the physically relevant domain"
    assert set(np.unique(regime_mesh)) == set(range(8)), "Mesh does not exercise all 8 opacity regimes"


# ==========================================
# SECTION: Opacity <-> Gradients Interface Preview
# ==========================================

def plot_opacity_along_synthetic_profile(output_path="opacity_profile_preview.png") -> None:
    """Preview kappa(m) along a synthetic centrally-condensed profile, ahead of gradients.py (Sub-task 3).

    rho(m) and T(m) are a rough polytropic-shaped placeholder (not a converged structure -
    that requires bvp_solver.py in Sub-task 5), used only to confirm opacity.py dispatches
    across regimes at physically plausible depths.

    NOTE: T_center here is chosen well above config.T_DISSOCIATION_LIMIT so the preview
    exercises the hot, Kramers/electron-scattering regimes. The real simulation halts at
    T_center = 2000 K (config.T_DISSOCIATION_LIMIT), which Check 9 shows sits below the
    Molecules -> H- scattering transition (~3340 K at rho = 1e-10 g/cm^3) - so the physical
    run itself will likely never leave the cool grain/molecular regimes (indices 0-4).
    """
    n_points = config.N_GRID_POINTS
    x = np.linspace(0.0, 1.0, n_points)   # Fractional mass coordinate, m/M_TOTAL [dimensionless]

    rho_center, rho_surface = 1.0, 1.0e-9   # Centrally-condensed density placeholder [g cm^-3]
    T_center, T_surface = 2.0e4, config.T_NEB  # Placeholder temperature profile [K]

    # Polytropic-shaped placeholder profile: monotonically decreasing from center (x=0) to surface (x=1)
    rho_profile = rho_surface + (rho_center - rho_surface) * (1.0 - x) ** 3
    T_profile = T_surface + (T_center - T_surface) * (1.0 - x) ** 2

    kappa_profile = opacity.bell_lin_opacity(rho_profile, T_profile)
    regime_profile = opacity.determine_regime(rho_profile, T_profile)

    print("Check 11 - Opacity <-> gradients interface preview: kappa(m) along a synthetic profile")
    print(f"  regime at center (m=0):        {opacity.REGIMES[regime_profile[0]].name}")
    print(f"  regime at surface (m=M_TOTAL): {opacity.REGIMES[regime_profile[-1]].name}")
    assert not np.any(np.isnan(kappa_profile)) and not np.any(np.isinf(kappa_profile)), (
        "bell_lin_opacity produced NaN/Inf along the synthetic profile"
    )
    assert regime_profile[-1] <= 2, "Surface should sit in a cool grain-opacity regime (ice grains or evaporation)"
    assert regime_profile[0] >= 4, "Center should sit in a hot regime (molecules or hotter) for this preview profile"

    fig, (ax_profile, ax_kappa) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax_profile.plot(x, T_profile, color="tab:red", label="T(m) [K]")
    ax_profile.set_yscale("log")
    ax_profile.set_ylabel("T [K]")
    ax_profile.legend(loc="upper right")
    ax_profile_twin = ax_profile.twinx()
    ax_profile_twin.plot(x, rho_profile, color="tab:blue", label="rho(m) [g/cm^3]")
    ax_profile_twin.set_yscale("log")
    ax_profile_twin.set_ylabel("rho [g cm^-3]")
    ax_profile_twin.legend(loc="upper left")
    ax_profile.set_title("Synthetic profile (placeholder for Sub-task 5's converged structure)")

    ax_kappa.plot(x, kappa_profile, color="black")
    ax_kappa.set_yscale("log")
    ax_kappa.set_xlabel("m / M_TOTAL")
    ax_kappa.set_ylabel("kappa(m) [cm^2 g^-1]")
    ax_kappa.set_title("Bell & Lin opacity along the synthetic profile")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved opacity profile preview plot to {output_path}")


# ==========================================
# SECTION: Constants Printout
# ==========================================

def print_all_constants() -> None:
    """Print every physical constant and simulation parameter defined in config.py."""
    print("=== Physical Constants (CGS) ===")
    print(f"G          = {config.G:.6e} cm^3 g^-1 s^-2")
    print(f"C_LIGHT    = {config.C_LIGHT:.6e} cm s^-1")
    print(f"A_RAD      = {config.A_RAD:.6e} erg cm^-3 K^-4")
    print(f"K_B        = {config.K_B:.6e} erg K^-1")
    print(f"M_H        = {config.M_H:.6e} g")
    print(f"SIGMA_SB   = {config.SIGMA_SB:.6e} erg cm^-2 s^-1 K^-4")

    print("=== Nebula Boundary Conditions ===")
    print(f"P_NEB      = {config.P_NEB:.6e} dyn cm^-2")
    print(f"T_NEB      = {config.T_NEB:.6e} K")

    print("=== Envelope Bulk Properties ===")
    print(f"M_TOTAL    = {config.M_TOTAL:.6e} g")
    print(f"MU         = {config.MU:.6e} (dimensionless)")
    print(f"GAMMA      = {config.GAMMA:.6e} (dimensionless)")

    print("=== Grid & Solver Parameters ===")
    print(f"N_GRID_POINTS = {config.N_GRID_POINTS}")
    print(f"OPACITY_SMOOTH_TRANSITIONS = {config.OPACITY_SMOOTH_TRANSITIONS}")

    print("=== Physical Validity Limits ===")
    print(f"T_DISSOCIATION_LIMIT = {config.T_DISSOCIATION_LIMIT:.6e} K")


# ==========================================
# SECTION: Entry Point
# ==========================================

if __name__ == "__main__":
    print_all_constants()
    print()
    check_ideal_gas_eos()
    check_hydrostatic_equilibrium()
    check_continuity_equation()
    check_ideal_gas_density_inverts_pressure()
    check_adiabatic_gradient_and_cp_limits()
    check_regime_table_reference_points()
    check_transition_temperature_loglog_slopes()
    plot_transition_temperatures()
    check_regime_continuity()
    check_regime_ordering_monotonic()
    check_bell_lin_vectorization_stress_test()
    plot_opacity_along_synthetic_profile()
    print("\nAll CGS unit-consistency and Sub-task 2a-2e checks passed.")
