# gradients.py — Schwarzschild criterion: compares the temperature gradient
# radiative diffusion alone would require against the adiabatic gradient, and
# selects the physically realized dlnT/dlnP at each grid point.
# Pure, vectorized functions with no side effects (CLAUDE.md Architecture Rules).

import numpy as np

import config

# ==========================================
# SECTION: Radiative Temperature Gradient
# ==========================================

def grad_radiative(L, m, P, T, kappa):
    """Temperature gradient (dlnT/dlnP) radiative diffusion alone needs to carry L past mass shell m."""
    kappa_b = np.asarray(kappa, dtype=float)
    # Guard against a corrupted opacity.py result (e.g. an indexing bug in determine_regime)
    # propagating silently into the temperature structure equation.
    assert np.all(kappa_b > 0.0), "grad_radiative: kappa must be strictly positive"

    # Radiative diffusion gradient (Kippenhahn & Weigert stellar-structure form):
    # nabla_rad = 3*kappa*L*P / (16*pi*a_rad*c*G*m*T^4)   [dimensionless]
    # Units: [cm^2 g^-1][erg s^-1][dyn cm^-2] / ([erg cm^-3 K^-4][cm s^-1][cm^3 g^-1 s^-2][g][K^4])
    #      = (cm^3 g s^-5) / (cm^3 g s^-5) = dimensionless
    # ASSUMPTION: diverges as m -> 0. At the center L(m=0) = 0 by the inner boundary condition,
    # so the ratio is a removable 0/0; this function does not special-case it, so callers
    # (bvp_solver.py, odes.py) must not evaluate it exactly at m = 0.
    return (3.0 * kappa_b * L * P) / (16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G * m * T**4)


# ==========================================
# SECTION: Schwarzschild Stability Criterion
# ==========================================

def effective_gradient(grad_rad, grad_ad):
    """Schwarzschild criterion: the realized gradient is the shallower of radiative and adiabatic."""
    grad_rad_b = np.asarray(grad_rad, dtype=float)
    grad_ad_b = np.asarray(grad_ad, dtype=float)

    # Schwarzschild criterion: nabla_rad > nabla_ad means radiative transport alone would need
    # a steeper-than-adiabatic gradient, which is unstable to convective overturn. Convection
    # then sets in and flattens the realized gradient to nabla_ad; otherwise radiation carries
    # all of L and nabla_eff = nabla_rad.
    is_convective = grad_rad_b > grad_ad_b
    grad_eff = np.where(is_convective, grad_ad_b, grad_rad_b)
    return grad_eff, is_convective
