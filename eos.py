# eos.py — Ideal-gas + non-relativistic electron-degeneracy equation of state, and
# adiabatic constitutive relations. Pure, vectorized functions with no side effects
# (CLAUDE.md Architecture Rules).

import numpy as np

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
# SECTION: Specific Heat and Adiabatic Gradient
# ==========================================

def specific_heat_cp(gamma, mu):
    """Specific heat at constant pressure for an ideal gas, given the adiabatic index and mean molecular weight."""
    # c_p = gamma * R_specific / (gamma - 1), with R_specific = k_B/(mu*m_H)  [erg g^-1 K^-1]
    return gamma * config.K_B / ((gamma - 1.0) * mu * config.M_H)


def grad_adiabatic(gamma):
    """Adiabatic temperature gradient (dlnT/dlnP at constant entropy) for an ideal gas."""
    # nabla_ad = (gamma - 1) / gamma  [dimensionless]
    # ASSUMPTION: constant gamma ideal gas — invalid once H2 dissociation begins to lower gamma_eff
    return (gamma - 1.0) / gamma
