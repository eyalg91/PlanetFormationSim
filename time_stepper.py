# time_stepper.py — Computes the frozen time-derivative source terms (dT_dt, dP_dt) that
# drive odes.py's energy equation at each timestep. Pure functions; the outer time loop
# itself (calling a per-timestep solver, advancing the state, checking the dissociation
# halt) is Sub-task 8.

import numpy as np

import config

# ==========================================
# SECTION: Homologous Contraction Bootstrap
# ==========================================

def _bootstrap_time_derivatives(state_curr):
    """(dT_dt, dP_dt) for the first real timestep, derived from an assumed homologous
    contraction (config.T_KH_BOOTSTRAP_S) rather than finite-differencing - there is no
    prior state at t=0 to difference against (see config.py's ASSUMPTION comment for the
    derivation: r=r0*f(t), rho=rho0/f^3, P=P0/f^4, T=T0/f under mass conservation and
    hydrostatic self-similarity).

    dT_dt = +T/T_KH_BOOTSTRAP_S, dP_dt = +4*P/T_KH_BOOTSTRAP_S   [K s^-1, dyn cm^-2 s^-1]
    Both positive: contraction (dr/dt<0) raises interior T and P, the standard negative-heat-
    capacity behavior of a self-gravitating gas losing energy. Substituting into odes.py's
    energy equation gives dL/dm = [(3*gamma-4)/(gamma-1)] * k_B*T/(mu*m_H*T_KH_BOOTSTRAP_S) -
    positive for gamma > 4/3 (config.GAMMA=1.4 satisfies this, the same stability threshold
    behind config.T_DISSOCIATION_LIMIT) - a genuine, well-defined L(m) > 0 profile, not an
    arbitrary nonzero source.
    """
    dT_dt = state_curr.T / config.T_KH_BOOTSTRAP_S
    dP_dt = 4.0 * state_curr.P / config.T_KH_BOOTSTRAP_S
    return dT_dt, dP_dt


# ==========================================
# SECTION: Finite-Difference Time Derivatives
# ==========================================

def compute_time_derivatives(state_curr, state_prev, dt):
    """(dT_dt, dP_dt) on state_curr.m: finite-differenced from state_prev, or the homologous
    bootstrap (_bootstrap_time_derivatives) if state_prev is None - the first real step, with
    no earlier state to difference against.

    state_prev's T, P are interpolated onto state_curr.m before differencing, in case the
    Lagrangian grid shifts between steps.
    """
    if state_prev is None:
        return _bootstrap_time_derivatives(state_curr)

    T_prev_interp = np.interp(state_curr.m, state_prev.m, state_prev.T)
    P_prev_interp = np.interp(state_curr.m, state_prev.m, state_prev.P)

    dT_dt = (state_curr.T - T_prev_interp) / dt
    dP_dt = (state_curr.P - P_prev_interp) / dt
    return dT_dt, dP_dt
