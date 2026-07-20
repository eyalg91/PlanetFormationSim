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

    # Ideal gas EOS: rho = P*mu*m_H/(k_B*T)  [g cm^-3]  (eos.py)
    # ASSUMPTION: ideal gas — breaks down near H2 dissociation (config.T_DISSOCIATION_LIMIT)
    rho = eos.density(P, T, config.MU)

    kappa = opacity.bell_lin_opacity(rho, T)     # Bell & Lin (1994) opacity [cm^2 g^-1]
    grad_ad = eos.grad_adiabatic(config.GAMMA)   # nabla_ad = (gamma-1)/gamma [dimensionless]
    grad_rad = gradients.grad_radiative(L, m, P, T, kappa)
    grad_eff, _ = gradients.effective_gradient(grad_rad, grad_ad)   # Schwarzschild criterion

    # Continuity / mass-radius relation: dr/dm = 1/(4*pi*r^2*rho)   [cm g^-1]
    dr_dm = 1.0 / (4.0 * np.pi * r**2 * rho)

    # Hydrostatic equilibrium: dP/dm = -G*m/(4*pi*r^4)   [dyn cm^-2 g^-1]
    dP_dm = -config.G * m / (4.0 * np.pi * r**4)

    # Energy equation (Kelvin-Helmholtz contraction source):
    # dL/dm = -c_p*(dT/dt) + (1/rho)*(dP/dt)   [erg s^-1 g^-1]
    c_p = eos.specific_heat_cp(config.GAMMA, config.MU)   # [erg g^-1 K^-1]
    dL_dm = -c_p * dT_dt + dP_dt / rho

    # Temperature structure (Schwarzschild criterion): dT/dm = (T/P)*nabla_eff*(dP/dm)   [K g^-1]
    dT_dm = (T / P) * grad_eff * dP_dm

    return np.vstack([dr_dm, dP_dm, dL_dm, dT_dm])
