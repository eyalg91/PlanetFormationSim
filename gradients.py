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

    # ASSUMPTION: this formula is derived for strictly outward radiative flux (L>=0). Dividing
    # by T^4 (which -> 0 near a photosphere) makes it pathologically sensitive to any negative
    # excursion in L there, which would otherwise drive nabla_rad - and hence dT/dm via the
    # Schwarzschild selection in effective_gradient - to force an unphysical temperature
    # inversion (see PROGRESS.md for the numerical trace). Floored here, at the point the
    # outward-flux assumption actually breaks down, rather than patched downstream.
    #
    # A hard floor (L_safe=max(L,0), used until 2026-08-01) is a genuine non-differentiable
    # kink at L=0. Confirmed 2026-08-06 (PROGRESS.md has the full trace) to be the root cause
    # of relax_initial_state's adaptive alpha-stepping wall at alpha~0.046: both LM's outer
    # finite-difference Jacobian probe and Radau's own inner implicit-stage Newton iteration
    # (which also assumes a smooth RHS) were tripping on the kink, not converging past it
    # regardless of step size. Smoothed instead with the standard "smoothed absolute value"/
    # pseudo-Huber form, which -> max(L,0) exactly as epsilon->0 while remaining C-infinity
    # everywhere, including at L=0: L_safe = 0.5*(L + sqrt(L^2 + epsilon^2))   [erg s^-1]
    #
    # Computed here in the algebraically-equivalent, cancellation-safe form
    # L_safe = max(L,0) + 0.5*eps^2/(sqrt(L^2+eps^2)+|L|) instead: the naive form above
    # subtracts two O(|L|) quantities to recover an O(eps) result whenever L is large and
    # negative (observed directly, L probes down to -8e54 erg/s during relax_initial_state's
    # homotopy - PROGRESS.md 2026-08-06), losing precision at the scale of |L|, not eps - the
    # same cancellation bug independently caught by validation.py Check 15 in
    # effective_gradient's smoothed switch (see config.GRAD_EFF_SWITCH_EPSILON), fixed there
    # the same way. max(L,0)=0.5*(L+|L|) is itself always exact (L and -L cancel exactly, or
    # add exactly), and the correction term is a well-conditioned sum, not a difference.
    # config.GRAD_RAD_L_FLOOR_EPSILON's own comment has the full derivation/verification of
    # epsilon's scale.
    eps = config.GRAD_RAD_L_FLOOR_EPSILON
    L_abs = np.abs(L)
    L_safe = np.maximum(L, 0.0) + 0.5 * eps**2 / (np.sqrt(L**2 + eps**2) + L_abs)

    # Radiative diffusion gradient (Kippenhahn & Weigert stellar-structure form):
    # nabla_rad = 3*kappa*L*P / (16*pi*a_rad*c*G*m*T^4)   [dimensionless]
    # ASSUMPTION: diverges as m -> 0. At the center L(m=0) = 0 by the inner boundary condition,
    # so the ratio is a removable 0/0; callers (bvp_solver.py, odes.py) must not evaluate this
    # exactly at m = 0.
    return (3.0 * kappa_b * L_safe * P) / (16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G * m * T**4)


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
    # all of L and nabla_eff = nabla_rad. is_convective is a plain hard comparison (fine - it's
    # informational only, not fed back into the ODE integration; see grad_eff below).
    is_convective = grad_rad_b > grad_ad_b

    # ASSUMPTION: grad_eff = min(nabla_rad, nabla_ad) exactly (the np.where above is just that
    # comparison spelled out) - this idealizes convection as infinitely efficient, snapping
    # instantly to nabla_ad the moment nabla_rad crosses it. Confirmed 2026-08-06 (PROGRESS.md)
    # to be a second hard kink, same failure mode as the L>=0 floor: relax_initial_state's
    # alpha-ramp stalled again at alpha~0.050946 with a trial trajectory running almost exactly
    # along the convective boundary (nabla_rad within 3e-5 relative of nabla_ad at several
    # points). Not an artificial safety clamp like the L-floor - a real physical idealization
    # (the standard smoother alternative is mixing-length theory's continuous interpolation by
    # superadiabaticity; NOT implemented here, see PLAN.md's new mandatory future sub-task).
    # Smoothed as an interim numerical fix, same hyperbolic family as the L-floor:
    # min(a,b) = 0.5*(a+b) - 0.5*|a-b|  ->  0.5*(a+b) - 0.5*sqrt((a-b)^2 + epsilon^2)
    #
    # Computed here in the algebraically-equivalent, cancellation-safe form
    # min(a,b) - 0.5*eps^2/(sqrt((a-b)^2+eps^2)+|a-b|) instead: the naive subtraction form
    # above loses precision at the scale of max(|a|,|b|), not epsilon, whenever nabla_rad and
    # nabla_ad differ by many orders of magnitude (observed directly: validation.py Check 15
    # sweeps nabla_rad up to ~8.6e8 against nabla_ad~0.286, and the naive form let grad_eff
    # exceed nabla_ad - a genuine floating-point violation of a mathematically-exact bound, not
    # a physics issue). min(a,b) computed via np.minimum is itself exact (no cancellation), and
    # the correction term is a well-conditioned sum, not a difference - same fix applied to the
    # L-floor in grad_radiative above. config.GRAD_EFF_SWITCH_EPSILON's own comment has the
    # full derivation/verification.
    eps = config.GRAD_EFF_SWITCH_EPSILON
    abs_diff = np.abs(grad_rad_b - grad_ad_b)
    grad_eff = np.minimum(grad_rad_b, grad_ad_b) - 0.5 * eps**2 / (np.sqrt((grad_rad_b - grad_ad_b)**2 + eps**2) + abs_diff)
    return grad_eff, is_convective


# ==========================================
# SECTION: Marginally Efficient Convection — Diagnostic Luminosity
# ==========================================

def marginal_convective_luminosity(m, P, T, kappa, grad_ad):
    """L such that radiative diffusion alone would exactly carry the adiabatic gradient
    (nabla_rad(L,...) = nabla_ad), i.e. the "marginally efficient convection" closure.

    Used only where a fully convective interior's structure (r, P, T) is constructed directly
    from the adiabat, independent of L (bvp_solver.py's t=0 compact-protoplanet construction) -
    this backs out a physically meaningful diagnostic L(m) afterward, by inverting
    grad_radiative's formula rather than assuming any particular actual radiative/convective
    flux split. Not used by any RHS/ODE integration - a real trial L would need to come from
    solving the full coupled system (odes.py), which this construction deliberately bypasses.
    """
    # Invert nabla_rad = 3*kappa*L*P / (16*pi*a_rad*c*G*m*T^4) for L at nabla_rad = nabla_ad:
    # L = nabla_ad * 16*pi*a_rad*c*G*m*T^4 / (3*kappa*P)   [erg s^-1]
    return grad_ad * (16.0 * np.pi * config.A_RAD * config.C_LIGHT * config.G * m * T**4) / (3.0 * kappa * P)
