# validation.py — Sanity checks, unit-consistency verification, and
# convergence criteria for PlanetFormationSim. Per CLAUDE.md, all validation
# and testing logic lives here, never inside operational physics or solver
# modules (odes.py, bvp_solver.py, time_stepper.py, etc.).

import matplotlib.pyplot as plt
import numpy as np

import boundary_conditions
import bvp_solver
import config
import diagnostics
import eos
import gradients
import odes
import opacity
import state
import time_stepper

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
    """Confirm eos.density(P, T, mu, mu_e) reduces to the hand-solved ideal-gas rho at a
    density far below the electron-degeneracy crossover (Sub-task 2f) - not exact equality
    anymore (the combined EOS has a small, quantifiable degenerate correction even here), but
    a deviation matching the analytically-predicted first-order correction
    P_degenerate(rho_ideal)/P, confirming eos.density's Newton solve behaves as expected in
    the regime where the original (pre-Sub-task-2f) ideal-gas-only formula should dominate."""
    P_test = 5.291281895336678e-3    # Test pressure [dyn cm^-2], chosen so rho_ideal ~ 1e-12 g/cm^3
    T_test = 150.0                    # Test temperature [K]
    mu_test = 2.34                    # Mean molecular weight, H2/He mix [dimensionless]

    # Hand-solved ideal-gas-only value: rho = P*mu*m_H/(k_B*T)  [g cm^-3]
    rho_ideal = P_test * mu_test * config.M_H / (config.K_B * T_test)
    rho_computed = eos.density(P_test, T_test, mu_test, config.MU_E)

    # First-order correction: at this rho, the combined EOS carries a small degenerate
    # pressure fraction P_degenerate(rho_ideal)/P_test that pulls rho_computed slightly below
    # rho_ideal (some of P_test is already supplied by degeneracy, so less ideal-gas density
    # is needed) - the SAME fraction, to leading order, as the relative rho deviation.
    predicted_fractional_deviation = eos.degenerate_pressure(rho_ideal, config.MU_E) / P_test
    actual_fractional_deviation = (rho_ideal - rho_computed) / rho_ideal

    print("Check 4 - eos.density() reduces to the ideal-gas limit far below the degenerate crossover")
    print(f"  P = {P_test:.6e} dyn/cm^2, T = {T_test:.3e} K, mu = {mu_test}")
    print(f"  -> rho_computed = {rho_computed:.6e} g/cm^3, rho_ideal (old formula) = {rho_ideal:.6e} g/cm^3")
    print(f"  predicted fractional deviation (P_degenerate/P) = {predicted_fractional_deviation:.6e}")
    print(f"  actual fractional deviation = {actual_fractional_deviation:.6e}")
    assert np.isclose(actual_fractional_deviation, predicted_fractional_deviation, rtol=0.05), (
        "eos.density()'s deviation from the ideal-gas limit does not match the predicted "
        "first-order degenerate-pressure correction"
    )
    assert actual_fractional_deviation < 1.0e-4, "test point is not far enough below the degenerate crossover to isolate the ideal-gas limit"


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
# SECTION: EOS — Electron Degeneracy Pressure (Sub-task 2f)
# ==========================================

def check_degenerate_pressure_reference_point() -> None:
    """Confirm eos.degenerate_pressure(rho, mu_e) matches a hand-computed value at a
    reference point, guarding against a sign/unit/exponent error in the formula."""
    rho_test = 1.0     # Reference density [g cm^-3]
    mu_e_test = 2.0    # Reference mean molecular weight per electron [dimensionless]

    # Hand-computed: P = (h^2/(20*m_e)) * (3/pi)^(2/3) * (rho/(mu_e*m_H))^(5/3)  [dyn cm^-2]
    n_e_expected = rho_test / (mu_e_test * config.M_H)
    P_expected = (config.PLANCK_H**2 / (20.0 * config.M_E)) * (3.0 / np.pi) ** (2.0 / 3.0) * n_e_expected ** (5.0 / 3.0)
    P_computed = eos.degenerate_pressure(rho_test, mu_e_test)

    print("Check 33 - eos.degenerate_pressure() matches the hand-computed Fermi-gas formula")
    print(f"  rho = {rho_test} g/cm^3, mu_e = {mu_e_test}")
    print(f"  -> P_computed = {P_computed:.6e} dyn/cm^2, P_expected = {P_expected:.6e} dyn/cm^2")
    assert np.isclose(P_computed, P_expected, rtol=1e-12), "degenerate_pressure() does not match the hand-computed reference value"
    assert P_computed > 0.0 and np.isfinite(P_computed), "degenerate_pressure() produced a non-physical pressure"


def check_combined_eos_asymptotic_limits() -> None:
    """Confirm the combined EOS correctly reduces to pure ideal gas at low density and to
    pure electron degeneracy at high density, at fixed T=T_CENTER_INITIAL - the two
    physical limits the additive combination (Sub-task 2f) is designed to interpolate
    between."""
    T_test = config.T_CENTER_INITIAL
    # Crossover density where P_ideal = P_degenerate at T_test (PROGRESS.md Sub-task 2f):
    # rho_cross = [k_B*T/(mu*m_H*K1)]^1.5, K1 from degenerate_pressure(1,mu_e)/1^(5/3).
    K1 = eos.degenerate_pressure(1.0, config.MU_E)
    rho_cross = (config.K_B * T_test / (config.MU * config.M_H * K1)) ** 1.5

    rho_low = rho_cross * 1.0e-8    # deep ideal-gas regime
    rho_high = rho_cross * 1.0e8    # deep degenerate regime

    P_ideal_low = rho_low * config.K_B * T_test / (config.MU * config.M_H)
    P_deg_low = eos.degenerate_pressure(rho_low, config.MU_E)
    P_ideal_high = rho_high * config.K_B * T_test / (config.MU * config.M_H)
    P_deg_high = eos.degenerate_pressure(rho_high, config.MU_E)

    print("Check 34 - combined EOS reduces to the correct limit far below/above the crossover density")
    print(f"  T = {T_test} K, rho_cross = {rho_cross:.6e} g/cm^3")
    print(f"  rho_low  = {rho_low:.3e}: P_deg/P_ideal = {P_deg_low/P_ideal_low:.3e} (expect << 1)")
    print(f"  rho_high = {rho_high:.3e}: P_ideal/P_deg = {P_ideal_high/P_deg_high:.3e} (expect << 1)")
    assert P_deg_low / P_ideal_low < 1.0e-4, "degenerate pressure not negligible far below the crossover density"
    assert P_ideal_high / P_deg_high < 1.0e-4, "ideal-gas pressure not negligible far above the crossover density"


def check_density_inverts_combined_eos() -> None:
    """Confirm eos.density()'s Newton-Raphson solve correctly inverts P=P_ideal(rho,T)+
    P_degenerate(rho) across a range of densities spanning the ideal/degenerate crossover
    (not just the ideal-gas limit, Check 4) - the critical correctness check for the new
    iterative solve, which has no closed-form inverse to compare against directly."""
    T_test = config.T_CENTER_INITIAL
    K1 = eos.degenerate_pressure(1.0, config.MU_E)
    rho_cross = (config.K_B * T_test / (config.MU * config.M_H * K1)) ** 1.5

    print("Check 35 - eos.density() round-trips the combined EOS across the ideal/degenerate crossover")
    for factor in [1.0e-4, 1.0e-1, 1.0, 1.0e1, 1.0e4]:
        rho_true = rho_cross * factor
        P_total = rho_true * config.K_B * T_test / (config.MU * config.M_H) + eos.degenerate_pressure(rho_true, config.MU_E)
        rho_recovered = eos.density(P_total, T_test, config.MU, config.MU_E)
        rel_err = abs(rho_recovered - rho_true) / rho_true
        print(f"  rho/rho_cross = {factor:.1e}: rho_true = {rho_true:.6e}, rho_recovered = {rho_recovered:.6e}, rel_err = {rel_err:.3e}")
        assert rel_err < 1.0e-6, f"eos.density() failed to invert the combined EOS at rho/rho_cross={factor:.1e}"


def plot_combined_eos_pressure_vs_density() -> None:
    """Visible check: P_ideal(rho), P_degenerate(rho), and their sum vs. rho at
    T=T_CENTER_INITIAL, log-log, with the crossover density marked - shows the new EOS's
    interpolation between the two physical regimes directly (CLAUDE.md preference for a
    visible check wherever one naturally fits)."""
    T_test = config.T_CENTER_INITIAL
    K1 = eos.degenerate_pressure(1.0, config.MU_E)
    rho_cross = (config.K_B * T_test / (config.MU * config.M_H * K1)) ** 1.5

    rho_grid = np.logspace(np.log10(rho_cross) - 6, np.log10(rho_cross) + 6, 200)
    P_ideal = rho_grid * config.K_B * T_test / (config.MU * config.M_H)
    P_deg = eos.degenerate_pressure(rho_grid, config.MU_E)
    P_total = P_ideal + P_deg

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(rho_grid, P_ideal, "--", label="P_ideal(rho)")
    ax.loglog(rho_grid, P_deg, "--", label="P_degenerate(rho)")
    ax.loglog(rho_grid, P_total, "-", color="black", label="P_total (combined EOS)")
    ax.axvline(rho_cross, color="gray", linestyle=":", label=f"rho_cross = {rho_cross:.2e} g/cm^3")
    ax.set_xlabel("rho [g/cm^3]")
    ax.set_ylabel("P [dyn/cm^2]")
    ax.set_title(f"Combined ideal-gas + electron-degeneracy EOS at T={T_test:.0f} K")
    ax.legend()
    fig.tight_layout()
    output_path = f"{diagnostics.PLOT_DIR}/combined_eos_pressure_vs_density.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Check 36 - saved {output_path} (visible check)")


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

def plot_transition_temperatures(output_path=f"{diagnostics.PLOT_DIR}/opacity_transitions.png") -> None:
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

def plot_opacity_along_synthetic_profile(output_path=f"{diagnostics.PLOT_DIR}/opacity_profile_preview.png") -> None:
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
# SECTION: Gradients — Convection Trigger vs. Critical Luminosity
# ==========================================

def check_convection_triggers_at_grad_rad_exceeds_grad_ad() -> None:
    """Confirm is_convective flips True exactly where grad_rad crosses grad_ad, sweeping L at fixed (rho, T)."""
    m_test = 1.0e29     # Representative enclosed mass shell [g]
    P_test = 1.0e6      # Representative pressure [dyn cm^-2]
    T_test = 1000.0     # Representative temperature [K]
    rho_test = 1.0e-6   # Representative density [g cm^-3]
    kappa_test = opacity.bell_lin_opacity(rho_test, T_test)   # [cm^2 g^-1]
    grad_ad_test = eos.grad_adiabatic(config.GAMMA)

    # nabla_rad(L) is linear in L, so the crossing nabla_rad = nabla_ad solves directly:
    # L_crit = nabla_ad * 16*pi*a_rad*c*G*m*T^4 / (3*kappa*P)   [erg s^-1]
    L_crit = (grad_ad_test * 16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G * m_test * T_test**4
              / (3.0 * kappa_test * P_test))

    grad_rad_below = gradients.grad_radiative(0.99 * L_crit, m_test, P_test, T_test, kappa_test)
    grad_rad_above = gradients.grad_radiative(1.01 * L_crit, m_test, P_test, T_test, kappa_test)
    _, conv_below = gradients.effective_gradient(grad_rad_below, grad_ad_test)
    _, conv_above = gradients.effective_gradient(grad_rad_above, grad_ad_test)

    print("Check 12 - gradients.effective_gradient() convection trigger vs. critical luminosity")
    print(f"  L_crit = {L_crit:.6e} erg/s (where nabla_rad = nabla_ad = {grad_ad_test:.6f})")
    print(f"  L = 0.99*L_crit -> nabla_rad = {grad_rad_below:.6f}, is_convective = {conv_below}")
    print(f"  L = 1.01*L_crit -> nabla_rad = {grad_rad_above:.6f}, is_convective = {conv_above}")
    assert not conv_below, "Below L_crit, envelope should be radiative (is_convective should be False)"
    assert conv_above, "Above L_crit, envelope should be convective (is_convective should be True)"


# ==========================================
# SECTION: Gradients — Radiative Limit Exactness
# ==========================================

def check_radiative_limit_grad_eff_equals_grad_rad() -> None:
    """Confirm grad_eff equals grad_rad exactly (not just approximately) when radiation dominates."""
    m_test = 1.0e29
    P_test = 1.0e6
    T_test = 1000.0
    rho_test = 1.0e-6
    kappa_test = opacity.bell_lin_opacity(rho_test, T_test)
    grad_ad_test = eos.grad_adiabatic(config.GAMMA)

    # Same critical-luminosity construction as Check 12; a luminosity far below it guarantees
    # nabla_rad << nabla_ad regardless of the absolute magnitude of kappa or T at this point.
    L_crit = (grad_ad_test * 16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G * m_test * T_test**4
              / (3.0 * kappa_test * P_test))
    L_test = 1.0e-3 * L_crit

    grad_rad_test = gradients.grad_radiative(L_test, m_test, P_test, T_test, kappa_test)
    grad_eff_test, is_convective_test = gradients.effective_gradient(grad_rad_test, grad_ad_test)

    print("Check 13 - Radiative limit: grad_eff == grad_rad when nabla_rad << nabla_ad")
    print(f"  L = 1e-3*L_crit -> nabla_rad = {grad_rad_test:.6e}, nabla_ad = {grad_ad_test:.6f}, "
          f"is_convective = {is_convective_test}")
    assert not is_convective_test, "Deep radiative regime should not be flagged convective"
    # ASSUMPTION: exact equality (used until 2026-08-06) no longer holds - effective_gradient's
    # min(grad_rad,grad_ad) switch is now a smooth hyperbolic approximation (config.
    # GRAD_EFF_SWITCH_EPSILON), which is never bit-exact anywhere by construction. Here
    # nabla_rad is ~1e-3*nabla_ad, i.e. |nabla_rad-nabla_ad| >> epsilon, so the smoothing
    # distortion is of order epsilon^2/(2*|nabla_rad-nabla_ad|) ~ 1e-7 - orders of magnitude
    # below this 1e-4 tolerance, which still tightly catches a genuinely broken formula.
    assert abs(grad_eff_test - grad_rad_test) < 1.0e-4, "grad_eff should be indistinguishable from grad_rad in the radiative limit"


# ==========================================
# SECTION: Gradients — Convective Limit Exactness
# ==========================================

def check_convective_limit_grad_eff_equals_grad_ad() -> None:
    """Confirm grad_eff equals grad_ad exactly (not just approximately) when convection dominates."""
    m_test = 1.0e29
    P_test = 1.0e6
    T_test = 1000.0
    rho_test = 1.0e-6
    kappa_test = opacity.bell_lin_opacity(rho_test, T_test)
    grad_ad_test = eos.grad_adiabatic(config.GAMMA)

    L_crit = (grad_ad_test * 16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G * m_test * T_test**4
              / (3.0 * kappa_test * P_test))
    L_test = 1.0e3 * L_crit

    grad_rad_test = gradients.grad_radiative(L_test, m_test, P_test, T_test, kappa_test)
    grad_eff_test, is_convective_test = gradients.effective_gradient(grad_rad_test, grad_ad_test)

    print("Check 14 - Convective limit: grad_eff == grad_ad when nabla_rad >> nabla_ad")
    print(f"  L = 1e3*L_crit -> nabla_rad = {grad_rad_test:.6e}, nabla_ad = {grad_ad_test:.6f}, "
          f"is_convective = {is_convective_test}")
    assert is_convective_test, "Steep radiative gradient should be flagged convective"
    # ASSUMPTION: see Check 13's comment - exact equality no longer holds under the smoothed
    # switch; nabla_rad here is ~1e3*nabla_ad, so the same tiny, tolerance-swamped distortion applies.
    assert abs(grad_eff_test - grad_ad_test) < 1.0e-4, "grad_eff should be indistinguishable from grad_ad in the convective limit"


# ==========================================
# SECTION: Gradients — Full Opacity-Regime Sweep
# ==========================================

def check_grad_radiative_over_full_opacity_regime_sweep() -> None:
    """Confirm grad_radiative/effective_gradient stay finite and physically bounded across the full T sweep.

    Reuses opacity Check 9's T in [100, 50000] K range so this check exercises every regime
    kappa can return, not just the one or two the limit checks above happen to land in.
    """
    m_test = 1.0e29                              # Representative enclosed mass shell [g]
    P_test = 1.0e6                                # Representative pressure [dyn cm^-2]
    rho_test = 1.0e-10                             # Representative density [g cm^-3]
    T_sweep = np.linspace(100.0, 50000.0, 500)    # [K], matches opacity Check 9's range

    # Kelvin-Helmholtz luminosity estimate, L_KH ~ G*M_TOTAL^2/(R*t_KH) (PLAN.md Sub-task 5
    # exit criterion), R ~ present Jupiter radius, t_KH ~ 1e6 yr - a representative surface
    # luminosity scale, not a converged solution (bvp_solver.py does not exist yet).
    R_test = 7.0e9                    # Representative envelope radius [cm]
    t_KH_test = 1.0e6 * 3.156e7       # Kelvin-Helmholtz timescale, 1e6 yr in seconds [s]
    L_test = config.G * config.M_TOTAL**2 / (R_test * t_KH_test)

    kappa_sweep = opacity.bell_lin_opacity(rho_test, T_sweep)
    grad_ad_test = eos.grad_adiabatic(config.GAMMA)

    grad_rad_sweep = gradients.grad_radiative(L_test, m_test, P_test, T_sweep, kappa_sweep)
    grad_eff_sweep, is_convective_sweep = gradients.effective_gradient(grad_rad_sweep, grad_ad_test)

    print("Check 15 - gradients over the full T in [100, 50000] K opacity-regime sweep")
    print(f"  L = {L_test:.3e} erg/s (Kelvin-Helmholtz estimate), rho = {rho_test:.3e} g/cm^3")
    print(f"  regimes exercised: {sorted(np.unique(opacity.determine_regime(rho_test, T_sweep)))}")
    print(f"  nabla_rad range: [{grad_rad_sweep.min():.3e}, {grad_rad_sweep.max():.3e}]")
    print(f"  convective points: {int(np.sum(is_convective_sweep))} / {len(T_sweep)}")
    assert np.all(np.isfinite(grad_rad_sweep)), "grad_radiative produced non-finite values over the T sweep"
    assert np.all(grad_rad_sweep > 0.0), "grad_radiative should be strictly positive for L, kappa > 0"
    assert np.all(grad_eff_sweep <= grad_ad_test), "grad_eff must never exceed the adiabatic ceiling nabla_ad"


# ==========================================
# SECTION: Gradients — Non-Positive Kappa Guard
# ==========================================

def check_grad_radiative_rejects_nonpositive_kappa() -> None:
    """Confirm grad_radiative raises (via its assert) when handed a non-positive kappa."""
    m_test = 1.0e29    # Representative enclosed mass shell [g]
    P_test = 1.0e6     # Representative pressure [dyn cm^-2]
    T_test = 1000.0    # Representative temperature [K]

    print("Check 16 - grad_radiative() rejects kappa <= 0")
    try:
        gradients.grad_radiative(1.0e25, m_test, P_test, T_test, kappa=0.0)
    except AssertionError:
        print("  kappa = 0.0 correctly raised AssertionError")
    else:
        raise AssertionError("grad_radiative should reject kappa = 0.0 but did not raise")

    try:
        gradients.grad_radiative(1.0e25, m_test, P_test, T_test, kappa=-1.0)
    except AssertionError:
        print("  kappa = -1.0 correctly raised AssertionError")
    else:
        raise AssertionError("grad_radiative should reject kappa < 0 but did not raise")


# ==========================================
# SECTION: ODEs — Constant-Density Analytic Profile Agreement
# ==========================================

def check_stellar_odes_matches_constant_density_analytic_profile() -> None:
    """Compare stellar_odes()'s dr/dm, dP/dm, dT/dm against a closed-form uniform-density
    self-gravitating sphere, restricted to interior mass shells.

    r(m) = (3m/4*pi*rho0)^(1/3) and P(m) = (2/3)*pi*G*rho0^2*(R^2 - r(m)^2) are the classical
    zero-surface-pressure hydrostatic solution for a constant-density sphere; neither depends on
    T, so dP/dm can be checked directly against this pair. dr/dm, however, is computed by
    stellar_odes() from an EOS-derived rho(P,T), not from rho0 directly - so it needs its own T
    array, T_rho(m) = P(m)*mu*m_H/(k_B*rho0), chosen specifically to invert the ideal gas law
    back to exactly rho0 at every point. A single T(m) cannot serve both purposes: the profile
    below also wants an adiabatic T(m) = T_center*(P/P_center)^nabla_ad (forcing full convection
    with a large L) to test dT/dm, but that relation does not, in general, reproduce rho0 via the
    EOS, which is exactly why an earlier version of this check failed on dr/dm alone despite
    dP/dm and dT/dm already agreeing - a real self-consistency gap in the test construction, not
    a bug in odes.py.

    m = 0 and m = M_TOTAL are excluded: they are genuine coordinate singularities (r=0 makes
    dr/dm formally divergent) that boundary_conditions.py, not stellar_odes, is responsible for.
    """
    rho0 = 1.33   # Representative constant density, ~Jupiter's mean density [g cm^-3]
    R = (3.0 * config.M_TOTAL / (4.0 * np.pi * rho0)) ** (1.0 / 3.0)   # Sphere radius [cm]
    m_check = np.linspace(0.01 * config.M_TOTAL, 0.99 * config.M_TOTAL, 1500)   # Interior shells [g]

    r_check = (3.0 * m_check / (4.0 * np.pi * rho0)) ** (1.0 / 3.0)
    P_center = (2.0 / 3.0) * np.pi * config.G * rho0**2 * R**2
    P_check = (2.0 / 3.0) * np.pi * config.G * rho0**2 * (R**2 - r_check**2)

    # EOS-inverted temperature: rho = P*mu*m_H/(k_B*T) => T = P*mu*m_H/(k_B*rho0) reproduces
    # rho0 exactly, by algebraic construction, regardless of P(m)'s shape.
    T_rho_check = P_check * config.MU * config.M_H / (config.K_B * rho0)
    dr_dm_analytic = 1.0 / (4.0 * np.pi * r_check**2 * rho0)   # Closed-form target [cm g^-1]

    grad_ad_test = eos.grad_adiabatic(config.GAMMA)
    T_center = 1500.0   # Representative center temperature, below T_DISSOCIATION_LIMIT [K]
    T_ad_check = T_center * (P_check / P_center) ** grad_ad_test
    kappa_check = opacity.bell_lin_opacity(rho0, T_ad_check)

    # Pick L large enough to force convection at every point: L_crit(m) is the luminosity where
    # nabla_rad = nabla_ad at that point (same construction as gradients Check 12); 1e3x its max
    # over the profile guarantees nabla_rad > nabla_ad everywhere, so T_ad_check is self-consistent.
    L_crit_check = (grad_ad_test * 16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G
                    * m_check * T_ad_check**4 / (3.0 * kappa_check * P_check))
    L_test = 1.0e3 * np.max(L_crit_check)
    L_check = np.full_like(m_check, L_test)
    zero_source = np.zeros_like(m_check)   # dT_dt = dP_dt = 0: static (t=0) solve

    y_rho_check = np.vstack([r_check, P_check, L_check, T_rho_check])
    dr_dm_computed, dP_dm_computed_rho, _, _ = odes.stellar_odes(
        m_check, y_rho_check, zero_source, zero_source
    )

    y_ad_check = np.vstack([r_check, P_check, L_check, T_ad_check])
    _, dP_dm_computed, dL_dm_computed, dT_dm_computed = odes.stellar_odes(
        m_check, y_ad_check, zero_source, zero_source
    )

    dP_dm_fd = np.gradient(P_check, m_check)
    dT_dm_fd = np.gradient(T_ad_check, m_check)

    # Exclude the first/last few points, where np.gradient falls back to a less accurate
    # one-sided difference.
    interior = slice(5, -5)
    rel_err_r = np.max(np.abs((dr_dm_computed[interior] - dr_dm_analytic[interior]) / dr_dm_analytic[interior]))
    rel_err_P = np.max(np.abs((dP_dm_computed[interior] - dP_dm_fd[interior]) / dP_dm_fd[interior]))
    rel_err_T = np.max(np.abs((dT_dm_computed[interior] - dT_dm_fd[interior]) / dT_dm_fd[interior]))

    grad_rad_check = gradients.grad_radiative(L_check, m_check, P_check, T_ad_check, kappa_check)
    is_fully_convective = np.all(grad_rad_check > grad_ad_test)

    print("Check 17 - stellar_odes() vs. constant-density sphere (dr/dm) and adiabatic profile (dP/dm, dT/dm)")
    print(f"  rho0 = {rho0} g/cm^3, R = {R:.4e} cm, P_center = {P_center:.4e} dyn/cm^2, "
          f"T_center = {T_center} K, L = {L_test:.4e} erg/s")
    print(f"  fully convective across profile: {is_fully_convective}")
    print(f"  max relative error: dr/dm (vs analytic) = {rel_err_r:.3e}, "
          f"dP/dm (vs finite diff) = {rel_err_P:.3e}, dT/dm (vs finite diff) = {rel_err_T:.3e}")
    assert is_fully_convective, "L was not large enough to force convection everywhere; T_ad_check reference is invalid"
    assert rel_err_r < 1.0e-9, "stellar_odes() dr/dm disagrees with the closed-form 1/(4*pi*r^2*rho0) target"
    assert rel_err_P < 1.0e-3, "stellar_odes() dP/dm disagrees with the analytic profile's finite-difference derivative"
    assert rel_err_T < 1.0e-2, "stellar_odes() dT/dm disagrees with the analytic profile's finite-difference derivative"
    assert np.all(dP_dm_computed_rho == dP_dm_computed), "dP/dm must not depend on T (it doesn't appear in the formula)"
    assert np.all(dL_dm_computed == 0.0), "dL/dm should be exactly zero when dT_dt = dP_dt = 0 (static solve)"


# ==========================================
# SECTION: ODEs — Output Shape, Finiteness, and Sign Sanity
# ==========================================

def check_stellar_odes_output_shape_finite_and_signs() -> None:
    """Confirm stellar_odes() returns the right shape, all-finite values, and physically correct signs."""
    n_points = 50
    m_test = np.linspace(1.0e28, 1.0e30, n_points)     # Representative interior mass grid [g]
    r_test = np.linspace(1.0e9, 7.0e9, n_points)         # Representative radius, increasing outward [cm]
    P_test = np.linspace(1.0e10, 1.0e4, n_points)        # Representative pressure, decreasing outward [dyn cm^-2]
    T_test = np.linspace(2000.0, 150.0, n_points)        # Representative temperature, decreasing outward [K]
    L_test = np.full(n_points, 1.0e28)                   # Representative luminosity [erg s^-1]

    y_test = np.vstack([r_test, P_test, L_test, T_test])
    zero_source = np.zeros(n_points)
    dydm = odes.stellar_odes(m_test, y_test, zero_source, zero_source)
    dr_dm, dP_dm, dL_dm, dT_dm = dydm

    print("Check 18 - stellar_odes() output shape, finiteness, and sign sanity")
    print(f"  input y shape = {y_test.shape}, output dy/dm shape = {dydm.shape}")
    assert dydm.shape == y_test.shape, "stellar_odes() output shape must match the input state vector shape"
    assert np.all(np.isfinite(dydm)), "stellar_odes() produced non-finite values for a physically reasonable profile"
    # nabla_eff >= 0 always (nabla_rad from positive L, kappa; nabla_ad = (gamma-1)/gamma > 0 for gamma > 1),
    # so dT/dm's sign is set entirely by dP/dm's sign regardless of radiative vs. convective regime.
    assert np.all(dr_dm > 0.0), "dr/dm should be positive: radius must increase with enclosed mass"
    assert np.all(dP_dm < 0.0), "dP/dm should be negative: pressure must decrease outward"
    assert np.all(dT_dm < 0.0), "dT/dm should be negative for this outward-cooling profile"


# ==========================================
# SECTION: ODEs — Visual Check: Analytic Profile and Residual
# ==========================================

def plot_constant_density_profile_ode_check(output_path=f"{diagnostics.PLOT_DIR}/odes_profile_check.png") -> None:
    """Save a diagnostic plot of the constant-density analytic profile and the stellar_odes() vs.
    analytic/finite-difference residual, as a visible sanity check of the whole ODE RHS ahead of
    bvp_solver.py (Sub-task 5), which is the first module that will produce a real converged
    profile to compare against. Mirrors the two-temperature-array construction in Check 17
    (see its docstring): T_rho_check for dr/dm, T_ad_check for dP/dm and dT/dm.
    """
    rho0 = 1.33
    R = (3.0 * config.M_TOTAL / (4.0 * np.pi * rho0)) ** (1.0 / 3.0)
    m_check = np.linspace(0.01 * config.M_TOTAL, 0.99 * config.M_TOTAL, 1500)

    r_check = (3.0 * m_check / (4.0 * np.pi * rho0)) ** (1.0 / 3.0)
    P_center = (2.0 / 3.0) * np.pi * config.G * rho0**2 * R**2
    P_check = (2.0 / 3.0) * np.pi * config.G * rho0**2 * (R**2 - r_check**2)

    T_rho_check = P_check * config.MU * config.M_H / (config.K_B * rho0)
    dr_dm_analytic = 1.0 / (4.0 * np.pi * r_check**2 * rho0)

    grad_ad_test = eos.grad_adiabatic(config.GAMMA)
    T_center = 1500.0
    T_ad_check = T_center * (P_check / P_center) ** grad_ad_test
    kappa_check = opacity.bell_lin_opacity(rho0, T_ad_check)

    L_crit_check = (grad_ad_test * 16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G
                    * m_check * T_ad_check**4 / (3.0 * kappa_check * P_check))
    L_test = 1.0e3 * np.max(L_crit_check)
    L_check = np.full_like(m_check, L_test)
    zero_source = np.zeros_like(m_check)

    y_rho_check = np.vstack([r_check, P_check, L_check, T_rho_check])
    dr_dm_computed, _, _, _ = odes.stellar_odes(m_check, y_rho_check, zero_source, zero_source)

    y_ad_check = np.vstack([r_check, P_check, L_check, T_ad_check])
    _, dP_dm_computed, _, dT_dm_computed = odes.stellar_odes(m_check, y_ad_check, zero_source, zero_source)

    dP_dm_fd = np.gradient(P_check, m_check)
    dT_dm_fd = np.gradient(T_ad_check, m_check)

    x = m_check / config.M_TOTAL
    fig, (ax_profile, ax_resid) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax_profile.plot(x, r_check / R, label="r(m) / R")
    ax_profile.plot(x, P_check / P_center, label="P(m) / P_center")
    ax_profile.plot(x, T_ad_check / T_center, label="T_ad(m) / T_center")
    ax_profile.set_ylabel("normalized profile")
    ax_profile.set_title("Constant-density analytic profile (Sub-task 4 ODE check)")
    ax_profile.legend(loc="best")

    ax_resid.plot(x, np.abs((dr_dm_computed - dr_dm_analytic) / dr_dm_analytic), label="dr/dm residual (vs analytic)")
    ax_resid.plot(x, np.abs((dP_dm_computed - dP_dm_fd) / dP_dm_fd), label="dP/dm residual (vs finite diff)")
    ax_resid.plot(x, np.abs((dT_dm_computed - dT_dm_fd) / dT_dm_fd), label="dT/dm residual (vs finite diff)")
    ax_resid.set_yscale("log")
    ax_resid.set_xlabel("m / M_TOTAL")
    ax_resid.set_ylabel("relative residual\n|stellar_odes - finite diff| / |finite diff|")
    ax_resid.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved ODE analytic-profile check plot to {output_path}")


# ==========================================
# SECTION: Boundary Conditions — Residual Indexing and Sign Check
# ==========================================

def check_boundary_conditions_residuals() -> None:
    """Confirm boundary_conditions() is exactly zero at the true BCs, that perturbing each of
    ya[0], ya[2], yb[1] shifts exactly its own (linear) residual component and no other, and
    that the net-flux radiative surface condition (nonlinear in T_b, r_b) matches its exact
    analytic formula under perturbation of yb[3] and yb[0].

    Revised for the boundary_conditions.py surface thermal condition change (Sub-task 8,
    PROGRESS.md): T_b - T_neb replaced with L_b - 4*pi*r_b^2*sigma_SB*(T_b^4-T_neb^4), so the
    surface residual is no longer linear in T_b or independent of r_b.
    """
    r_test = 7.0e9         # Representative surface radius [cm]
    P_center_test = 1.0e13   # Arbitrary center-side pressure, not constrained by the BCs [dyn cm^-2]
    T_center_test = 1500.0   # Arbitrary center-side temperature, not constrained by the BCs [K]

    ya0 = np.array([0.0, P_center_test, 0.0, T_center_test])
    # L_b = 0 matches the net-flux radiative condition exactly when T_b = T_neb (equilibrium).
    yb0 = np.array([r_test, config.P_NEB, 0.0, config.T_NEB])
    res0 = boundary_conditions.boundary_conditions(ya0, yb0)

    print("Check 19 - boundary_conditions() residual indexing and sign check")
    print(f"  residual at exact BCs: {res0}")
    assert res0.shape == (4,), "boundary_conditions() must return exactly 4 residuals for a 4-ODE system"
    assert np.allclose(res0, 0.0), "Residual should be exactly zero when ya, yb already satisfy all 4 boundary conditions"

    delta = 1.0e5
    # (label, which side, index in y, expected residual index that should shift) - linear terms only
    linear_cases = [
        ("ya[0] (r_center)", "ya", 0, 0),
        ("ya[2] (L_center)", "ya", 2, 1),
        ("yb[1] (P_surface)", "yb", 1, 2),
    ]
    for label, which, y_idx, res_idx in linear_cases:
        ya, yb = ya0.copy(), yb0.copy()
        (ya if which == "ya" else yb)[y_idx] += delta
        res = boundary_conditions.boundary_conditions(ya, yb)
        shift = res - res0
        expected_shift = np.zeros(4)
        expected_shift[res_idx] = delta
        print(f"  perturb {label} by {delta:.1e} -> residual shift = {shift}")
        assert np.allclose(shift, expected_shift), f"Perturbing {label} should shift only residual[{res_idx}] by {delta}"

    # yb[3] (T_surface) and yb[0] (r_surface) both enter the new radiative term nonlinearly -
    # verify against the exact analytic formula rather than a simple linear delta, and confirm
    # residuals 0-2 stay unaffected.
    for label, y_idx in [("yb[3] (T_surface)", 3), ("yb[0] (r_surface)", 0)]:
        yb = yb0.copy()
        yb[y_idx] += delta
        res = boundary_conditions.boundary_conditions(ya0, yb)
        r_b, _, L_b, T_b = yb
        L_expected = 4.0 * np.pi * r_b**2 * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
        expected_res3 = L_b - L_expected
        print(f"  perturb {label} by {delta:.1e} -> residual[3] = {res[3]:.6e} (analytic: {expected_res3:.6e})")
        assert np.isclose(res[3], expected_res3), f"Perturbing {label} should match the exact radiative-flux formula in residual[3]"
        assert np.allclose(res[:3], res0[:3]), f"Perturbing {label} should not affect residuals 0-2"


# ==========================================
# SECTION: Nebula Conditions vs. MMSN at 50 AU
# ==========================================

def check_nebula_conditions_match_mmsn_at_50au() -> None:
    """Confirm config.T_NEB, config.P_NEB agree with the Hayashi (1981) MMSN midplane at 50 AU."""
    AU = 1.496e13     # Astronomical unit [cm]
    M_SUN = 1.989e33  # Solar mass, for the local Keplerian orbital frequency [g]
    r_AU = 50.0        # Assumed GI-fragmentation disk radius [AU]
    r = r_AU * AU

    # Hayashi (1981) MMSN midplane temperature: T(r) = 280 K * (r/AU)^-0.5   [K]
    T_hayashi = 280.0 * r_AU**-0.5
    # Hayashi (1981) MMSN gas surface density: Sigma_gas(r) = 1700 g/cm^2 * (r/AU)^-1.5   [g cm^-2]
    Sigma_hayashi = 1700.0 * r_AU**-1.5

    # Vertical hydrostatic disk structure: H = c_s/Omega, rho_mid = Sigma/(sqrt(2*pi)*H),
    # P_mid = rho_mid*c_s^2   [dyn cm^-2]
    Omega = np.sqrt(config.G * M_SUN / r**3)
    c_s2 = config.K_B * T_hayashi / (config.MU * config.M_H)
    H = np.sqrt(c_s2) / Omega
    rho_mid = Sigma_hayashi / (np.sqrt(2.0 * np.pi) * H)
    P_hayashi = rho_mid * c_s2

    T_ratio = config.T_NEB / T_hayashi
    P_ratio = config.P_NEB / P_hayashi

    print("Check 21 - config.T_NEB, config.P_NEB vs. Hayashi (1981) MMSN midplane at 50 AU")
    print(f"  Hayashi (1981): T = {T_hayashi:.2f} K, P_mid = {P_hayashi:.3e} dyn/cm^2")
    print(f"  config.py:      T_NEB = {config.T_NEB} K, P_NEB = {config.P_NEB:.3e} dyn/cm^2")
    print(f"  ratios: T_NEB/T_hayashi = {T_ratio:.3f}, P_NEB/P_hayashi = {P_ratio:.3f}")
    assert 0.5 < T_ratio < 2.0, "config.T_NEB is not within a factor of 2 of the Hayashi (1981) MMSN value at 50 AU"
    assert 0.2 < P_ratio < 5.0, "config.P_NEB is not within a factor of 5 of the Hayashi (1981) MMSN value at 50 AU"


# ==========================================
# SECTION: Bonnor-Ebert Subcriticality
# ==========================================

def check_envelope_mass_is_bonnor_ebert_subcritical() -> None:
    """Confirm config.M_TOTAL is below the Bonnor-Ebert critical mass at config.T_NEB, config.P_NEB.

    A pressure-confined isothermal sphere above M_BE has no stable hydrostatic equilibrium at
    all (Bonnor 1956; Ebert 1955) - this is exactly the failure mode that broke the original
    T_NEB=150K, P_NEB=1e4 configuration (M_TOTAL/M_BE was ~99). Checking this here means a future
    change to T_NEB, P_NEB, or M_TOTAL that reopens that crisis fails loudly with a clear physical
    explanation, rather than surfacing later as an opaque bvp_solver.py convergence failure.
    """
    # Bonnor-Ebert critical mass for a pressure-confined isothermal sphere:
    # M_BE = 1.18 * c_s^4 / sqrt(G^3 * P_ext)   [g]
    c_s2 = config.K_B * config.T_NEB / (config.MU * config.M_H)
    M_BE = 1.18 * c_s2**2 / np.sqrt(config.G**3 * config.P_NEB)
    ratio = config.M_TOTAL / M_BE

    print("Check 22 - Bonnor-Ebert subcriticality: M_TOTAL vs. M_BE(T_neb, P_neb)")
    print(f"  M_BE = {M_BE:.4e} g,  M_TOTAL = {config.M_TOTAL:.4e} g,  M_TOTAL/M_BE = {ratio:.4f}")
    assert ratio < 1.0, (
        "M_TOTAL exceeds the Bonnor-Ebert critical mass: no stable isothermal hydrostatic "
        "equilibrium exists at these T_neb, P_neb - bvp_solver.py's shooting method would fail"
    )


# ==========================================
# SECTION: t=0 Static Structure — Hydrostatic Balance
# ==========================================

def check_static_structure_hydrostatic_balance() -> None:
    """Confirm bvp_solver.solve_static_structure()'s profile satisfies dP/dr = -G*m*rho/r^2
    (Eulerian form), the PLAN.md Sub-task 5 exit criterion, on the converged shooting solution."""
    s = bvp_solver.solve_static_structure()

    dP_dr_fd = np.gradient(s.P, s.r)
    dP_dr_analytic = -config.G * s.m * s.rho / s.r**2

    # Exclude points near the center (rapidly varying r ~ m^(1/3), finite-difference truncation
    # error dominates there) and near the surface (one-sided np.gradient difference).
    interior = slice(20, -15)
    rel_err = np.abs((dP_dr_fd[interior] - dP_dr_analytic[interior]) / dP_dr_analytic[interior])

    print("Check 23 - solve_static_structure() Eulerian hydrostatic balance: dP/dr vs -G*m*rho/r^2")
    print(f"  max relative error (interior points) = {rel_err.max():.3e}")
    assert rel_err.max() < 1.0e-3, "Converged structure fails hydrostatic balance in Eulerian form"


# ==========================================
# SECTION: t=0 Static Structure — Exact Isothermal/Zero-Luminosity and Monotonicity
# ==========================================

def check_static_structure_isothermal_and_monotonic() -> None:
    """Confirm solve_static_structure() gives T=T_neb, L=0 exactly, and a monotonic profile
    with the correct surface pressure."""
    s = bvp_solver.solve_static_structure()

    print("Check 24 - solve_static_structure() exact isothermal/zero-luminosity and monotonicity")
    print(f"  r range: {s.r[0]:.4e} to {s.r[-1]:.4e} cm ({s.r[-1] / 1.496e13:.2f} AU)")
    print(f"  P range: {s.P[0]:.6e} to {s.P[-1]:.6e} dyn/cm^2 (target P_neb = {config.P_NEB:.6e})")
    assert np.all(s.T == config.T_NEB), "T must equal T_neb exactly everywhere (isothermal, see bvp_solver.py docstring)"
    assert np.all(s.L == 0.0), "L must equal 0 exactly everywhere (see bvp_solver.py docstring)"
    assert np.all(np.diff(s.r) > 0.0), "r must be strictly increasing outward"
    assert np.all(np.diff(s.P) < 0.0), "P must be strictly decreasing outward"
    assert np.isclose(s.P[-1], config.P_NEB, rtol=config.BVP_TOL * 10.0), "Surface pressure does not match P_neb to the solver tolerance"


# ==========================================
# SECTION: t=0 Static Structure — Visual Check
# ==========================================

def plot_static_structure_profile(output_path=f"{diagnostics.PLOT_DIR}/static_structure_t0.png") -> None:
    """Save a diagnostic plot of the converged t=0 r(m), P(m) profile.

    T(m) and L(m) are trivially flat/zero by construction (bvp_solver.py docstring) so are noted
    rather than plotted as their own (degenerate) panels.
    """
    s = bvp_solver.solve_static_structure()
    AU = 1.496e13

    fig, (ax_r, ax_P) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    x = s.m / config.M_TOTAL

    ax_r.plot(x, s.r / AU)
    ax_r.set_ylabel("r(m) [AU]")
    ax_r.set_title(f"t=0 static structure: isothermal at T={config.T_NEB} K, L=0")

    ax_P.plot(x, s.P)
    ax_P.set_yscale("log")
    ax_P.set_xlabel("m / M_TOTAL")
    ax_P.set_ylabel("P(m) [dyn cm^-2]")
    ax_P.axhline(config.P_NEB, color="tab:red", linestyle="--", linewidth=0.8, label="P_neb")
    ax_P.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved t=0 static structure profile plot to {output_path}")


# ==========================================
# SECTION: Sub-task 6 — Pressure-Confined Virial Balance
# ==========================================

def check_virial_balance_unconfined() -> None:
    """Confirm the converged t=0 state satisfies the standard (zero-surface-pressure) virial
    theorem: E_grav + 3*(gamma-1)*E_therm = 0 (diagnostics.virial_balance).

    Derived independently (integrate hydrostatic equilibrium by parts). Valid once the
    photospheric outer BC (Sub-task 5a) makes the surface pressure negligible against the
    interior energy scale - diagnostics.virial_balance's docstring has the full derivation
    and the ~15-orders-of-magnitude check confirming that limit applies here.
    """
    s = bvp_solver.solve_static_structure()
    E_grav, E_therm = diagnostics.virial_balance(s)
    thermal_term = 3.0 * (config.GAMMA - 1.0) * E_therm
    lhs = E_grav + thermal_term
    # Normalized against the scale of the terms actually being balanced (diagnostics.
    # run_diagnostics uses the same normalization) - there is no longer an external reference
    # scale (P_neb) worth normalizing against.
    imbalance = abs(lhs) / max(abs(E_grav), abs(thermal_term))

    print("Check 26 - Unconfined virial balance: E_grav + 3*(gamma-1)*E_therm = 0")
    print(f"  E_grav = {E_grav:.4e} erg, 3*(gamma-1)*E_therm = {thermal_term:.4e} erg")
    print(f"  LHS = {lhs:.4e} erg, relative imbalance = {imbalance:.3e}")
    assert imbalance < 1.0e-2, "Virial balance violated beyond expected numerical precision"
    # The two terms must themselves be commensurate for the near-cancellation to be a genuine,
    # non-trivial balance - not e.g. one term ~0 trivially satisfying a small LHS.
    ratio = abs(E_grav) / abs(thermal_term)
    assert 0.1 < ratio < 10.0, "E_grav and the thermal term are not commensurate - near-cancellation would be trivial, not a genuine balance"


# ==========================================
# SECTION: Sub-task 6 — Opacity Regime Distribution
# ==========================================

def check_static_structure_opacity_regime_distribution() -> None:
    """Confirm the converged t=0 state spans more than one Bell & Lin opacity regime, from a
    hot convective interior to a cold photospheric surface.

    Sub-task 5's current t=0 state is a compact, differentiated hot-start protoplanet (not
    Premise 1's isothermal T_neb=50K cloud, which had no regime spread by construction) - the
    center (T_CENTER_INITIAL) should sit in a strictly hotter opacity regime than the
    photospheric surface. Regime *indices* aren't hardcoded here (they would need updating
    every time T_CENTER_INITIAL or the grid changes) - only the physically-required ordering.
    """
    s = bvp_solver.solve_static_structure()
    fractions = diagnostics.opacity_regime_distribution(s)
    regime_profile = opacity.determine_regime(s.rho, s.T)

    print("Check 27 - Opacity regime distribution at t=0 (compact, differentiated structure)")
    for regime, fraction in zip(opacity.REGIMES, fractions):
        if fraction > 0.0:
            print(f"  {regime.name:<32s} {fraction:.1%}")
    print(f"  regime at center: {opacity.REGIMES[regime_profile[0]].name}")
    print(f"  regime at surface: {opacity.REGIMES[regime_profile[-1]].name}")
    n_regimes_present = np.sum(fractions > 0.0)
    assert n_regimes_present > 1, "Compact, differentiated t=0 structure should span more than one opacity regime"
    assert regime_profile[0] > regime_profile[-1], "Center should sit in a strictly hotter opacity regime than the photospheric surface"


# ==========================================
# SECTION: Sub-task 6 — Mass Reconstruction
# ==========================================

def check_mass_reconstruction_matches_lagrangian_grid() -> None:
    """Confirm diagnostics.mass_reconstruction() (independent quadrature over the converged
    (r, rho) profile) matches the Lagrangian grid state.m away from the center.

    An independent check of the continuity equation (dr/dm = 1/(4*pi*r^2*rho)) and the
    shooting integration together: this quadrature is the inverse relation of that same ODE,
    computed by a completely different numerical method than the adaptive ODE integrator that
    produced the profile in the first place.
    """
    s = bvp_solver.solve_static_structure()
    M_recon = diagnostics.mass_reconstruction(s)
    rel_err = np.abs((M_recon - s.m) / s.m)

    print("Check 28 - Mass reconstruction: integral 4*pi*r^2*rho dr vs. Lagrangian grid m")
    print(f"  max relative error (full array) = {rel_err.max():.3e}")
    print(f"  max relative error (excluding first 30 points near center) = {rel_err[30:].max():.3e}")
    assert rel_err[30:].max() < 1.0e-2, "Mass reconstruction disagrees with the Lagrangian grid away from the center"


# ==========================================
# SECTION: Sub-task 6 — Visual Check: Mass Reconstruction Error
# ==========================================

def plot_mass_reconstruction_error(output_path=f"{diagnostics.PLOT_DIR}/mass_reconstruction_check.png") -> None:
    """Save a diagnostic plot of the mass-reconstruction relative error vs. radius."""
    s = bvp_solver.solve_static_structure()
    M_recon = diagnostics.mass_reconstruction(s)
    rel_err = np.abs((M_recon - s.m) / s.m)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s.r / 1.496e13, rel_err)
    ax.set_yscale("log")
    ax.set_xlabel("r [AU]")
    ax.set_ylabel("relative error: |M_reconstructed - m| / m")
    ax.set_title("Mass reconstruction check (continuity eq. consistency)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved mass reconstruction check plot to {output_path}")


# ==========================================
# SECTION: Sub-task 7 — Finite-Difference Time Derivatives
# ==========================================

def check_finite_difference_time_derivatives_and_interpolation() -> None:
    """Confirm compute_time_derivatives finite-differences correctly against a synthetic
    previous state, including interpolation when the previous state's grid differs."""
    n_curr = 50
    m_curr = np.linspace(1.0e27, 1.0e30, n_curr)   # Representative Lagrangian grid [g]
    r_curr = np.linspace(1.0e10, 1.0e14, n_curr)    # Representative radius [cm]
    P_curr = np.linspace(1.0e-3, 1.0e-4, n_curr)    # Representative pressure [dyn cm^-2]
    T_curr = np.linspace(100.0, 40.0, n_curr)       # Representative temperature [K]
    state_curr = state.SimulationState(
        m=m_curr, r=r_curr, P=P_curr, L=np.zeros(n_curr), T=T_curr, rho=np.ones(n_curr), t=100.0
    )

    # Previous state on a DIFFERENT (coarser) grid, to exercise the interpolation path.
    n_prev = 37
    m_prev = np.linspace(1.0e27, 1.0e30, n_prev)
    r_prev = np.linspace(1.0e10, 1.0e14, n_prev)
    P_prev = np.linspace(0.9e-3, 0.9e-4, n_prev)
    T_prev = np.linspace(90.0, 35.0, n_prev)
    state_prev = state.SimulationState(
        m=m_prev, r=r_prev, P=P_prev, L=np.zeros(n_prev), T=T_prev, rho=np.ones(n_prev), t=0.0
    )

    dt = 50.0   # Representative timestep [s]
    dT_dt, dP_dt = time_stepper.compute_time_derivatives(state_curr, state_prev, dt)

    T_prev_interp_expected = np.interp(m_curr, m_prev, T_prev)
    P_prev_interp_expected = np.interp(m_curr, m_prev, P_prev)
    dT_dt_expected = (T_curr - T_prev_interp_expected) / dt
    dP_dt_expected = (P_curr - P_prev_interp_expected) / dt

    print("Check 31 - Finite-difference time derivatives with grid interpolation")
    print(f"  dT_dt range: {dT_dt.min():.4e} to {dT_dt.max():.4e} K/s")
    print(f"  dP_dt range: {dP_dt.min():.4e} to {dP_dt.max():.4e} dyn/cm^2/s")
    assert np.allclose(dT_dt, dT_dt_expected), "dT_dt does not match hand-computed finite difference with interpolation"
    assert np.allclose(dP_dt, dP_dt_expected), "dP_dt does not match hand-computed finite difference with interpolation"


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

    print("=== Reference Length Scale ===")
    print(f"R_JUPITER_CM = {config.R_JUPITER_CM:.6e} cm")

    print("=== Simulation Halt Condition ===")
    print(f"R_HALT = {config.R_HALT:.6e} cm ({config.R_HALT / config.R_JUPITER_CM:.3f} R_Jup)")


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
    check_degenerate_pressure_reference_point()
    check_combined_eos_asymptotic_limits()
    check_density_inverts_combined_eos()
    plot_combined_eos_pressure_vs_density()
    check_regime_table_reference_points()
    check_transition_temperature_loglog_slopes()
    plot_transition_temperatures()
    check_regime_continuity()
    check_regime_ordering_monotonic()
    check_bell_lin_vectorization_stress_test()
    plot_opacity_along_synthetic_profile()
    check_convection_triggers_at_grad_rad_exceeds_grad_ad()
    check_radiative_limit_grad_eff_equals_grad_rad()
    check_convective_limit_grad_eff_equals_grad_ad()
    check_grad_radiative_over_full_opacity_regime_sweep()
    check_grad_radiative_rejects_nonpositive_kappa()
    check_stellar_odes_matches_constant_density_analytic_profile()
    check_stellar_odes_output_shape_finite_and_signs()
    plot_constant_density_profile_ode_check()
    check_boundary_conditions_residuals()
    check_nebula_conditions_match_mmsn_at_50au()
    check_envelope_mass_is_bonnor_ebert_subcritical()
    check_static_structure_hydrostatic_balance()
    check_static_structure_isothermal_and_monotonic()
    plot_static_structure_profile()
    check_virial_balance_unconfined()
    check_static_structure_opacity_regime_distribution()
    check_mass_reconstruction_matches_lagrangian_grid()
    plot_mass_reconstruction_error()
    check_finite_difference_time_derivatives_and_interpolation()
    print("\nAll CGS unit-consistency and Sub-task 1, 2a-2e, 3, 4, 5, 6, 7 checks passed.")
