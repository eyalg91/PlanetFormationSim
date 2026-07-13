# eos.py — Ideal-gas equation of state and adiabatic constitutive relations.
# Pure, vectorized functions with no side effects (CLAUDE.md Architecture Rules).

import config

# ==========================================
# SECTION: Ideal Gas Equation of State
# ==========================================

def density(P, T, mu):
    """Mass density from the ideal gas law, given pressure, temperature, and mean molecular weight."""
    # Ideal gas EOS: P = rho*k_B*T/(mu*m_H)  =>  rho = P*mu*m_H/(k_B*T)  [g cm^-3]
    # ASSUMPTION: ideal gas — breaks down near H2 dissociation (config.T_DISSOCIATION_LIMIT)
    return P * mu * config.M_H / (config.K_B * T)


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
