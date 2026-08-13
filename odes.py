# odes.py — Right-hand side of the 4-equation envelope structure system
# (continuity, hydrostatic equilibrium, energy, temperature gradient) for
# scipy.integrate.solve_bvp. Pure, vectorized function with no side effects
# (CLAUDE.md Architecture Rules).

import numpy as np

import config
import eos
import gradients
import opacity

# ==========================================
# SECTION: 4-ODE System Right-Hand Side
# ==========================================

def stellar_odes(m, y, dT_dt, dP_dt):
    """dy/dm for state vector y = [r, P, L, T] on the Lagrangian mass grid m.

    dT_dt, dP_dt are frozen time-derivative source arrays, pre-interpolated onto m by the
    caller (time_stepper.py); both are zero at t = 0, giving a purely static first solve.
    """
    r, P, L, T = y

    # H<->H2 recombination equilibrium (Sub-task 8b, JUSTIFIED 2026-08-10 by validation.py
    # Check 38): mu and gamma are no longer fixed at their atomic (config.MU/config.GAMMA)
    # values everywhere - both now vary smoothly with T via a shared logistic chi(T)
    # (eos.molecular_fraction), reducing EXACTLY to the old constants in the hot, atomic limit.
    mu_T = eos.mean_molecular_weight(T)     # [dimensionless]
    gamma_T = eos.gamma_effective(T)        # [dimensionless]

    # Combined ideal-gas + electron-degeneracy EOS (eos.py): P = P_ideal(rho,T) + P_degenerate(rho)
    # ASSUMPTION: ideal-gas term breaks down near H2 dissociation (config.T_DISSOCIATION_LIMIT);
    # degenerate term is non-relativistic only (eos.degenerate_pressure)
    rho = eos.density(P, T, mu_T, config.MU_E)

    kappa = opacity.bell_lin_opacity(rho, T)     # Bell & Lin (1994) opacity [cm^2 g^-1]
    grad_ad = eos.grad_adiabatic(gamma_T)        # nabla_ad = (gamma-1)/gamma [dimensionless]
    grad_rad = gradients.grad_radiative(L, m, P, T, kappa)
    grad_eff, _ = gradients.effective_gradient(grad_rad, grad_ad)   # Schwarzschild criterion

    # Continuity / mass-radius relation: dr/dm = 1/(4*pi*r^2*rho)   [cm g^-1]
    dr_dm = 1.0 / (4.0 * np.pi * r**2 * rho)

    # Hydrostatic equilibrium: dP/dm = -G*m/(4*pi*r^4)   [dyn cm^-2 g^-1]
    dP_dm = -config.G * m / (4.0 * np.pi * r**4)

    # Energy equation (Kelvin-Helmholtz contraction source, Kippenhahn & Weigert eq. 4.26):
    # dL/dm = -c_p*(dT/dt) + (delta/rho)*(dP/dt)   [erg s^-1 g^-1]
    # CORRECTED 2026-08-07 (eos.thermodynamic_delta's own comment has the full derivation):
    # delta = -(d ln rho/d ln T)_P was previously hardcoded to 1 (exact only for a pure ideal
    # gas) - this project's combined ideal+degenerate EOS needs the genuine, EOS-dependent
    # value (->0 as degeneracy dominates), not the ideal-gas special case, wherever
    # degeneracy contributes non-negligibly to the mechanical structure.
    #
    # EXTENDED (Sub-task 8b): c_p gains the H2 recombination latent-heat term (eos.
    # latent_heat_capacity - the DOMINANT effect, config.py's EPSILON_D_H2 comment has the
    # magnitude comparison), and delta gains the mu(T) correction (eos.mean_molecular_weight_
    # inv_derivative) via thermodynamic_delta's extended signature.
    d_inv_mu_dT = eos.mean_molecular_weight_inv_derivative(T)   # [K^-1]
    c_p = eos.specific_heat_cp(gamma_T, mu_T) + eos.latent_heat_capacity(T)   # [erg g^-1 K^-1]
    delta = eos.thermodynamic_delta(rho, T, mu_T, config.MU_E, d_inv_mu_dT)   # [dimensionless]
    dL_dm = -c_p * dT_dt + delta * dP_dt / rho

    # Temperature structure (Schwarzschild criterion): dT/dm = (T/P)*nabla_eff*(dP/dm)   [K g^-1]
    dT_dm = (T / P) * grad_eff * dP_dm

    return np.vstack([dr_dm, dP_dm, dL_dm, dT_dm])
