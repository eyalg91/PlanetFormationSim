# eos.py — Ideal-gas + non-relativistic electron-degeneracy equation of state, and
# adiabatic constitutive relations. Pure, vectorized functions with no side effects
# (CLAUDE.md Architecture Rules).

import numpy as np
from scipy.special import expit

import config

# ==========================================
# SECTION: Electron Degeneracy Pressure
# ==========================================

def degenerate_pressure(rho, mu_e):
    """Non-relativistic electron-degeneracy pressure (Chandrasekhar 1939; Kippenhahn &
    Weigert Ch. 15) from Fermi statistics at T=0.

    P_degenerate = (h^2/(20*m_e)) * (3/pi)^(2/3) * (rho/(mu_e*m_H))^(5/3)   [dyn cm^-2]
    Units: [erg s]^2 [g]^-1 * [g cm^-3 g^-1]^(5/3) = erg^2 s^2 g^-1 g^-5/3 cm^-5 ... reduces
    to dyn cm^-2 (verified dimensionally against the standard reference form).

    ASSUMPTION: non-relativistic (electron Fermi momentum << m_e*c) - holds for Jupiter-mass/
    density objects (central density ~0.1-1 g/cm^3, PROGRESS.md Sub-task 2f estimate), would
    need the relativistic correction only for much denser degenerate stars (white dwarfs near
    the Chandrasekhar mass).
    """
    n_e = rho / (mu_e * config.M_H)   # electron number density [cm^-3]
    return (config.PLANCK_H**2 / (20.0 * config.M_E)) * (3.0 / np.pi) ** (2.0 / 3.0) * n_e ** (5.0 / 3.0)


# ==========================================
# SECTION: Combined Equation of State
# ==========================================

def density(P, T, mu, mu_e):
    """Mass density from the combined ideal-gas + non-relativistic electron-degeneracy EOS:
    P = P_ideal(rho,T) + P_degenerate(rho).

    ASSUMPTION: additive combination (P_total = P_ideal + P_degenerate) is a standard
    first-order approximation for this purpose (Kippenhahn & Weigert; not a full tabulated
    equation-of-state) - ideal gas dominates at low density/high T, degeneracy dominates at
    high density regardless of T (PROGRESS.md Sub-task 2f has the crossover-density estimate).
    Also inherits the ideal-gas term's own assumption: breaks down near H2 dissociation
    (config.T_DISSOCIATION_LIMIT).

    The degenerate term is nonlinear in rho, so unlike the pure ideal-gas law this has no
    closed-form inverse for rho given (P,T) - solved via vectorized Newton-Raphson, seeded
    from the ideal-gas-only inversion (exact when degeneracy is negligible, and a reasonable
    starting point otherwise since P_total(rho) is smooth and monotonically increasing in rho
    at fixed T, guaranteeing a unique root).
    """
    rho = P * mu * config.M_H / (config.K_B * T)   # ideal-gas-only inversion: the Newton seed

    dP_ideal_drho = config.K_B * T / (mu * config.M_H)   # constant in rho at fixed T
    for _ in range(50):
        P_ideal = rho * dP_ideal_drho
        P_deg = degenerate_pressure(rho, mu_e)
        dP_deg_drho = (5.0 / 3.0) * P_deg / rho   # d/drho[K1*rho^(5/3)] = (5/3)*K1*rho^(2/3)
        residual = (P_ideal + P_deg) - P
        rho = rho - residual / (dP_ideal_drho + dP_deg_drho)
        # ASSUMPTION: P_total(rho) is monotonically increasing for rho>0, so the true root is
        # always positive - but a caller probing near a solver's own convergence boundary
        # (e.g. fsolve's internal Jacobian-estimation finite-difference perturbations) can feed
        # in a trial (P,T) far enough outside the well-behaved region that a single Newton step
        # overshoots past zero. Clamping keeps the iteration inside the physical domain (a
        # domain guard, not hiding non-convergence - the final assertion below still catches a
        # genuinely failed solve) rather than evaluating rho**(5/3) at a negative rho, which
        # would silently produce NaN and propagate into the ODE integration.
        rho = np.maximum(rho, 1.0e-300)

    # Guard against a silently non-converged Newton iteration returning the wrong rho -
    # real failures should surface, not be papered over (CLAUDE.md / project convention).
    P_check = rho * dP_ideal_drho + degenerate_pressure(rho, mu_e)
    assert np.all(np.abs(P_check - P) < 1.0e-8 * np.abs(P)), (
        "eos.density: Newton-Raphson did not converge to the target pressure - "
        "check for an unphysical (P, T, mu, mu_e) input outside the solver's basin of convergence"
    )
    return rho


# ==========================================
# SECTION: Energy-Equation Thermodynamic Coefficient
# ==========================================

def thermodynamic_delta(rho, T, mu, mu_e, d_inv_mu_dT=0.0):
    """delta = -(d ln rho / d ln T)_P for the combined ideal+degenerate EOS (Kippenhahn &
    Weigert eq. 4.26's energy-equation coefficient: dL/dm = -c_p*dT/dt + (delta/rho)*dP/dt).

    ASSUMPTION (CORRECTED 2026-08-07, PROGRESS.md/PLAN_BVP.md have the full trail):
    odes.py previously hardcoded delta=1, correct ONLY for a pure ideal gas. This project's
    combined EOS (density()) is not pure ideal gas wherever degeneracy is non-negligible -
    dominant in the interior at this project's T_center range - so that hardcoding was a
    silent, systematic energy-equation error, independent of whether the solver converges.

    Derived by IMPLICIT differentiation of the EOS's defining equation
    P = P_ideal(rho,T) + P_deg(rho) = 0 (same method as the analytic Jacobian's eos density
    derivatives): delta = P_ideal/(rho*D), D = dP_ideal_drho + dP_deg_drho (the same
    denominator density()'s own Newton iteration uses for its convergence step).

    Limiting-case check: delta -> 1 exactly as P_deg -> 0 (pure ideal gas, dP_ideal_drho/D ->
    1); delta -> 0 as P_deg dominates (fully degenerate limit - degenerate pressure is
    ~T-independent, so density stops responding to T at fixed P, exactly as expected).

    EXTENDED (Sub-task 8b): d_inv_mu_dT = d(1/mu)/dT accounts for mu ITSELF varying with T
    (the H<->H2 recombination transition, mean_molecular_weight_inv_derivative below) via the
    same implicit-differentiation method - differentiating P=P_ideal(rho,T,mu(T))+P_deg(rho)
    with mu(T) now T-dependent (chain rule) adds a rho*K_B*T^2*d(1/mu)/dT/M_H term to the
    numerator (same M_H as dP_ideal_drho below - P_ideal=rho*K_B*T/(mu*M_H) throughout).
    Reduces EXACTLY to the original formula when d_inv_mu_dT=0 (mu held fixed), preserving
    every existing caller's behavior unchanged (default value).
    """
    dP_ideal_drho = config.K_B * T / (mu * config.M_H)
    P_ideal = rho * dP_ideal_drho
    P_deg = degenerate_pressure(rho, mu_e)
    dP_deg_drho = (5.0 / 3.0) * P_deg / rho
    D = dP_ideal_drho + dP_deg_drho
    return (P_ideal + rho * config.K_B * T**2 * d_inv_mu_dT / config.M_H) / (rho * D)


# ==========================================
# SECTION: Specific Heat and Adiabatic Gradient
# ==========================================

def specific_heat_cp(gamma, mu):
    """Specific heat at constant pressure for an ideal gas, given the adiabatic index and mean molecular weight."""
    # c_p = gamma * R_specific / (gamma - 1), with R_specific = k_B/(mu*m_H)  [erg g^-1 K^-1]
    return gamma * config.K_B / ((gamma - 1.0) * mu * config.M_H)


def grad_adiabatic(gamma):
    """Adiabatic temperature gradient (dlnT/dlnP at constant entropy) for an ideal gas."""
    # nabla_ad = (gamma - 1) / gamma  [dimensionless]
    # ASSUMPTION: constant gamma ideal gas within a single call - callers needing the
    # H2-dissociation-softened gamma_eff(T) (Sub-task 8b, RESOLVED 2026-08-10) pass
    # gamma_effective(T) here explicitly rather than the old fixed config.GAMMA.
    return (gamma - 1.0) / gamma


# ==========================================
# SECTION: H<->H2 Recombination Equilibrium (Sub-task 8b)
# ==========================================
# JUSTIFIED 2026-08-10 by validation.check_outer_envelope_recombination_sensitivity (Check 38:
# Delta r_surface=-3.1% from a static mu(T) proxy, ~5x its own control floor). Also the shared
# physical mechanism expected to end Phase 1 (First Core collapse) once T_center approaches
# ~2000K and Gamma_1 softens below 4/3 (PROGRESS.md/PLAN.md 2026-08-10 have the full
# derivation) - built once, generally in T, not as a Phase-3-only patch.
#
# ASSUMPTION: T-only (no rho dependence) - a real H2 dissociation equilibrium constant depends
# on density too; this proxy is validated only against Phase 3's outer-envelope density regime
# (Check 38) and must be re-checked before being trusted as Phase 1's core collapse trigger.

def molecular_fraction(T):
    """chi(T): molecular H2 fraction (1=fully molecular/cold, 0=fully atomic/hot), a smooth
    logistic proxy for the true density-dependent H/H2 mass-action equilibrium - see this
    section's own ASSUMPTION above. Deliberately WIDE (config.T_H2_TRANSITION_WIDTH=180K), not
    sharp, per this project's own hard-won lesson from GRAD_EFF_SWITCH_EPSILON (a narrow
    transition risks reintroducing exactly the solver stiffness that smoothing was meant to
    avoid there).

    Uses scipy.special.expit (a numerically stable sigmoid, avoiding np.exp overflow at large
    positive arguments - encountered directly during validation.py's Check 17/19 edge-case
    sweeps, which probe T far outside this project's physical range) rather than a bare
    1/(1+exp(x)): chi(T) = expit(-(T-T_MID)/WIDTH), mathematically identical, ->0 or ->1
    cleanly at either extreme instead of via an intermediate inf.
    """
    return expit(-(T - config.T_H2_TRANSITION_MID) / config.T_H2_TRANSITION_WIDTH)


def molecular_fraction_derivative(T):
    """d(chi)/dT [K^-1] - analytic logistic derivative (standard identity: d(sigmoid)/dx =
    -sigmoid*(1-sigmoid)/width for this sign convention). Feeds mean_molecular_weight_inv_
    derivative (thermodynamic_delta's mu(T) correction) and latent_heat_capacity (the energy
    equation's extra effective heat capacity) below."""
    chi = molecular_fraction(T)
    return -chi * (1.0 - chi) / config.T_H2_TRANSITION_WIDTH


def mean_molecular_weight(T):
    """mu(T): mean molecular weight, interpolated LINEARLY IN 1/mu (exact for a two-state
    H/H2 + atomic He mixture - 1/mu = X*(1+f_atomic)/2 + Y/4, PROGRESS.md 2026-08-10 has the
    derivation) between the atomic limit config.MU (T->high, chi->0) and the molecular limit
    (T->low, chi->1). Reduces EXACTLY to config.MU as T->high, matching every existing call
    site's previous hardcoded behavior in that hot limit.

    config.USE_H2_RECOMBINATION_PHYSICS=False falls back to the constant config.MU everywhere -
    an internal escape hatch for bvp_solver.relax_initial_state's two-stage warm-start, not a
    user-facing physics option (config.py's own comment has the full reasoning).
    """
    if not config.USE_H2_RECOMBINATION_PHYSICS:
        return np.full_like(np.asarray(T, dtype=float), config.MU)
    chi = molecular_fraction(T)
    inv_mu = 1.0 / config.MU - (config.X_HYDROGEN / 2.0) * chi
    return 1.0 / inv_mu


def mean_molecular_weight_inv_derivative(T):
    """d(1/mu)/dT [K^-1] - feeds thermodynamic_delta's implicit-differentiation correction
    when mu=mu(T) (Sub-task 8b). 1/mu is LINEAR in chi(T) (mean_molecular_weight's own
    derivation), so this is a simple chain-rule multiply, not a fresh derivation."""
    return -(config.X_HYDROGEN / 2.0) * molecular_fraction_derivative(T)


def gamma_effective(T):
    """gamma_eff(T): adiabatic index, interpolated with the SAME chi(T) as mean_molecular_
    weight between the atomic (monatomic, config.GAMMA=5/3) and molecular (diatomic,
    config.GAMMA_MOLECULAR=7/5) limits - one shared transition function for both, not two
    independently-tuned ones. Reduces EXACTLY to config.GAMMA as T->high.

    config.USE_H2_RECOMBINATION_PHYSICS=False falls back to the constant config.GAMMA
    everywhere - see mean_molecular_weight's own docstring for the full reasoning.
    """
    if not config.USE_H2_RECOMBINATION_PHYSICS:
        return np.full_like(np.asarray(T, dtype=float), config.GAMMA)
    chi = molecular_fraction(T)
    return config.GAMMA + (config.GAMMA_MOLECULAR - config.GAMMA) * chi


def latent_heat_capacity(T):
    """Extra effective specific heat [erg g^-1 K^-1] from H2 recombination/dissociation
    latent heat, to be ADDED to specific_heat_cp(gamma_effective(T), mean_molecular_weight(T))
    in the energy equation - NOT a standalone c_p replacement.

    u_chem(chi) = -chi*EPSILON_D_H2 [erg/g] (molecular bonds are the LOWER-energy state, atomic
    chi=0 is the zero reference): d(u_chem)/dT = -EPSILON_D_H2*d(chi)/dT. As T rises, chi falls
    (dissociating, endothermic), so this term is POSITIVE - heating drives dissociation, which
    absorbs extra energy beyond simple translational heating (quantified in config.py's
    EPSILON_D_H2 comment: 3-16x the local thermal energy content across T=1000-5000K - the
    DOMINANT term in c_p_eff, not a minor correction).
    """
    return -config.EPSILON_D_H2 * molecular_fraction_derivative(T)
