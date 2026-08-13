# opacity.py — Bell & Lin (1994) piecewise power-law opacity.
# Layer 1: immutable regime data + raw power-law evaluator.
# Layer 2: transition_temperature.
# Layer 3 (this sub-task): determine_regime and bell_lin_opacity, the sole
# public API. Pure functions, no side effects.

import warnings
from collections import namedtuple

import numpy as np
from scipy.special import expit

import config

# ==========================================
# SECTION: Bell & Lin (1994) Regime Data — Table 3, Appendix
# ==========================================

RegimeParams = namedtuple("RegimeParams", ["name", "kappa_i", "a", "b"])

# Power law: kappa = kappa_i * rho^a * T^b  [cm^2 g^-1], rho [g cm^-3], T [K]
# Ordered by ascending temperature (index 0 = coolest / outermost regime).
REGIMES = (
    RegimeParams("Ice grains",                      2.0e-4,  0.0,      2.0),
    RegimeParams("Ice grain evaporation",            2.0e16,  0.0,     -7.0),
    RegimeParams("Metal grains",                     0.1,     0.0,      0.5),
    RegimeParams("Metal grain evaporation",          2.0e81,  1.0,    -24.0),
    RegimeParams("Molecules",                        1.0e-8,  2.0 / 3.0, 3.0),
    RegimeParams("H- scattering",                    1.0e-36, 1.0 / 3.0, 10.0),
    RegimeParams("Bound-free/free-free (Kramers)",   1.5e20,  1.0,     -2.5),
    RegimeParams("Electron scattering",              0.348,   0.0,      0.0),
)


# ==========================================
# SECTION: Raw Power-Law Evaluator
# ==========================================

def evaluate_regime(kappa_i, a, b, rho, T):
    """Bell & Lin power law for a single opacity regime: kappa = kappa_i * rho^a * T^b [cm^2 g^-1]."""
    return kappa_i * np.power(rho, a) * np.power(T, b)


# ==========================================
# SECTION: Regime Transition Temperature
# ==========================================

def transition_temperature(rho, n):
    """Temperature at which regime n and regime n+1 give equal opacity, as a function of density."""
    lower = REGIMES[n]
    upper = REGIMES[n + 1]
    b_diff = lower.b - upper.b

    if b_diff == 0.0:
        # ASSUMPTION: regimes n and n+1 have distinct T-exponents; b_n == b_{n+1} means
        # kappa_n = kappa_{n+1} cannot be solved for a unique transition T (data error in REGIMES).
        warnings.warn(
            f"Degenerate opacity transition {n}->{n + 1} "
            f"('{lower.name}' -> '{upper.name}'): b_n == b_(n+1) = {lower.b}; "
            "transition temperature is undefined. Check REGIMES for a data error.",
            RuntimeWarning,
        )
        return np.full_like(np.asarray(rho, dtype=float), np.nan)

    # Transition temperature: kappa_i^n * rho^a_n * T^b_n = kappa_i^(n+1) * rho^a_(n+1) * T^b_(n+1)
    # => T_{n->n+1}(rho) = [(kappa_i^(n+1)/kappa_i^n) * rho^(a_(n+1)-a_n)] ^ (1/(b_n - b_(n+1)))  [K]
    a_diff = upper.a - lower.a
    kappa_ratio = upper.kappa_i / lower.kappa_i
    return np.power(kappa_ratio * np.power(rho, a_diff), 1.0 / b_diff)


# ==========================================
# SECTION: Vectorized Regime Determination
# ==========================================

def monotonic_transition_temperatures(rho):
    """The 7 regime-pair transition temperatures at each rho, clamped to be non-decreasing in n.

    ASSUMPTION: regimes are visited in temperature order (n=0 coolest ... n=7 hottest). The raw
    per-pair formula T_{n->n+1}(rho) is just where two power laws happen to cross, and at low
    density the Kramers -> electron-scattering crossing (n=6) can fall *below* several cooler
    regimes' own transitions (e.g. ~179 K at rho=1e-15 g/cm^3, under the ice-grain transitions).
    Sorting by value would then splice that spurious low temperature into the cool end of the
    table under the wrong regime identity. Instead each boundary is clamped up to the running
    maximum of the boundaries below it (in n-order), which preserves the n -> regime-pair
    correspondence: it says that at that density, regime n+1 (electron scattering) never
    actually wins until the hottest boundary reached so far, so the intervening regime (Kramers)
    is simply squeezed out rather than misidentified as a cooler regime.
    """
    rho_b = np.asarray(rho, dtype=float)
    n_transitions = len(REGIMES) - 1
    raw_transitions = np.stack(
        [transition_temperature(rho_b, n) for n in range(n_transitions)], axis=-1
    )
    return np.maximum.accumulate(raw_transitions, axis=-1)


def determine_regime(rho, T):
    """Bell & Lin regime index (0..7) at each (rho, T) point, fully vectorized."""
    rho_b, T_b = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(T, dtype=float))
    transitions = monotonic_transition_temperatures(rho_b)

    # Regime index = count of (clamped) transition temperatures at or below T  [dimensionless, 0..7]
    return np.sum(T_b[..., np.newaxis] >= transitions, axis=-1)


# ==========================================
# SECTION: Smoothed Regime Blending (RESOLVED 2026-08-11)
# ==========================================
# kappa(rho,T) is continuous at regime boundaries by construction (transition_temperature is
# defined as where the two power laws are EQUAL), but d(kappa)/dT is not - adjacent regimes
# have different (a,b) exponents, so the derivative genuinely jumps there. Flagged as the
# leading unconfirmed suspect for a "third wall" in the shooting-method era (2026-08-06,
# PLAN_BVP.md) and never resolved; directly confirmed 2026-08-11 as the cause of a real
# solve_timestep mesh explosion (PROGRESS.md has the full trail). config.
# OPACITY_TRANSITION_SMOOTH_WIDTH_DEX has the width-choice derivation.

def _regime_weights(rho, T):
    """Smooth partition-of-unity regime weights, shape (...,8), summing to EXACTLY 1 at every
    point for any width>0 (telescoping construction, not an approximate normalization).

    s_n = expit((log10(T)-log10(T_trans[n]))/width) in [0,1] is a smooth "have we crossed
    transition n yet" indicator (0=cold side, 1=hot side), built from the SAME monotonic-
    clamped transition temperatures determine_regime's hard switch uses. The weight for
    regime n is s_{n-1} - s_n (s_{-1}:=1, s_7:=0 by convention) - as width->0, each s_n becomes
    a step function and this reduces EXACTLY to the hard switch's one-hot selection. Handles
    arbitrarily narrow regimes (e.g. Metal grain evaporation, ~4% fractional in T) gracefully:
    a point near two boundaries at once gets a well-defined multi-way blend instead of an
    ill-defined or overflowing calculation.
    """
    rho_b, T_b = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(T, dtype=float))
    log10T = np.log10(T_b)
    transitions = monotonic_transition_temperatures(rho_b)   # shape (...,7)
    width = config.OPACITY_TRANSITION_SMOOTH_WIDTH_DEX

    s = expit((log10T[..., np.newaxis] - np.log10(transitions)) / width)   # shape (...,7), s_0..s_6

    weights = np.empty(rho_b.shape + (len(REGIMES),), dtype=float)
    s_prev = np.ones_like(rho_b, dtype=float)   # s_{-1} = 1
    for idx in range(len(REGIMES)):
        s_curr = s[..., idx] if idx < len(REGIMES) - 1 else np.zeros_like(rho_b, dtype=float)   # s_7 = 0
        weights[..., idx] = s_prev - s_curr
        s_prev = s_curr
    return weights


def bell_lin_opacity_smooth(rho, T):
    """Smooth (logistic-blended) Bell & Lin opacity - see this section's module comment for
    why. kappa = sum_n weight_n(rho,T) * kappa_n(rho,T), a weighted average of every regime's
    own power law, using _regime_weights' partition-of-unity construction.

    FIX (2026-08-11, confirmed by direct reproduction before trusting - PROGRESS.md has the
    full trail, including the WRONG first hypothesis this replaced): evaluate_regime is
    evaluated for EVERY regime at EVERY point (vectorized), including regimes far outside
    their own domain, where the raw power law can genuinely overflow to inf (some exponents
    are steep - e.g. b=10 for H- scattering, b=-24 for Metal grain evaporation). A weight of
    EXACTLY 0.0 there is mathematically correct, but 0.0*inf = NaN in IEEE float arithmetic -
    unlike the hard switch (np.where SELECTS between precomputed branches, never multiplies,
    so it never had this failure mode). np.where masks the contribution to exactly 0 wherever
    the weight is 0, discarding whatever NaN evaluate_regime produced there rather than
    letting it corrupt the sum. Verified: eliminates all failures from a 200,000-point sweep
    across the full (rho,T) range _safe_exp_state's clamp can produce, with ZERO change in
    output anywhere on the normal physical domain (exact match, not just "close").
    """
    rho_b, T_b = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(T, dtype=float))
    weights = _regime_weights(rho_b, T_b)

    kappa = np.zeros_like(rho_b, dtype=float)
    for idx, regime in enumerate(REGIMES):
        w = weights[..., idx]
        kappa_n = evaluate_regime(regime.kappa_i, regime.a, regime.b, rho_b, T_b)
        kappa += np.where(w > 0.0, w * kappa_n, 0.0)
    return kappa


def _regime_weights_derivatives(rho, T):
    """d(weight_n)/d(rho), d(weight_n)/dT for _regime_weights' partition-of-unity blend, each
    shape (...,8). weight_n = s_{n-1} - s_n (s_{-1}:=1, s_7:=0 CONSTANTS, so their own
    derivatives are exactly 0), s_n = expit((log10(T)-log10(T_trans_n(rho)))/width):
      ds_n/dT   = s_n*(1-s_n) / (width*T*ln10)                          (T_trans_n has no T-dependence)
      ds_n/drho = -s_n*(1-s_n) * (a_diff_n/b_diff_n) / (width*rho*ln10)  (chain rule through
                  T_trans_n(rho)'s own log-log power-law form - transition_temperature's
                  defining equation, log10(T_trans_n)=(1/b_diff_n)*[log10(kappa_ratio)+
                  a_diff_n*log10(rho)])

    ASSUMPTION: differentiates the RAW per-pair transition_temperature(rho,n), not
    monotonic_transition_temperatures' np.maximum.accumulate CLAMP of it - that clamp is a
    genuine, SEPARATE hard switch (selecting which pair's raw formula actually sets T_trans at
    a given rho), relevant only at low density where regimes get numerically reordered
    (monotonic_transition_temperatures' own docstring, e.g. ~1e-15 g/cm^3). Not expected to be
    reached by this project's actual envelope densities - checked empirically via
    validation.py Check 37's saturation/transition-window sampling, not silently assumed away.
    """
    rho_b, T_b = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(T, dtype=float))
    log10T = np.log10(T_b)
    width = config.OPACITY_TRANSITION_SMOOTH_WIDTH_DEX
    ln10 = np.log(10.0)

    n_transitions = len(REGIMES) - 1
    a_diff = np.array([REGIMES[n + 1].a - REGIMES[n].a for n in range(n_transitions)])
    b_diff = np.array([REGIMES[n].b - REGIMES[n + 1].b for n in range(n_transitions)])

    transitions = monotonic_transition_temperatures(rho_b)   # shape (...,7)
    u = (log10T[..., np.newaxis] - np.log10(transitions)) / width
    s = expit(u)
    ds_dT = s * (1.0 - s) / (width * T_b[..., np.newaxis] * ln10)
    ds_drho = -s * (1.0 - s) * (a_diff / b_diff) / (width * rho_b[..., np.newaxis] * ln10)

    dweights_dT = np.empty(rho_b.shape + (len(REGIMES),), dtype=float)
    dweights_drho = np.empty_like(dweights_dT)
    ds_prev_dT = np.zeros_like(rho_b)     # d(s_{-1})/dT = 0 (s_{-1}=1, a constant)
    ds_prev_drho = np.zeros_like(rho_b)   # d(s_{-1})/drho = 0
    for idx in range(len(REGIMES)):
        ds_curr_dT = ds_dT[..., idx] if idx < len(REGIMES) - 1 else np.zeros_like(rho_b)     # d(s_7)/dT = 0
        ds_curr_drho = ds_drho[..., idx] if idx < len(REGIMES) - 1 else np.zeros_like(rho_b)  # d(s_7)/drho = 0
        dweights_dT[..., idx] = ds_prev_dT - ds_curr_dT
        dweights_drho[..., idx] = ds_prev_drho - ds_curr_drho
        ds_prev_dT, ds_prev_drho = ds_curr_dT, ds_curr_drho
    return dweights_drho, dweights_dT


def bell_lin_opacity_smooth_derivatives(rho, T):
    """d(kappa)/d(rho), d(kappa)/dT for bell_lin_opacity_smooth - product rule through
    kappa_smooth = sum_n weight_n(rho,T)*kappa_n(rho,T):
    d(kappa_smooth)/dX = sum_n [d(weight_n)/dX * kappa_n + weight_n * d(kappa_n)/dX]
    d(kappa_n)/drho = a_n*kappa_n/rho, d(kappa_n)/dT = b_n*kappa_n/T (each regime's own exact
    power-law derivative - the same form the hard-switch _opacity_derivatives, bvp_solver.py,
    already uses for whichever single regime is locally active).

    Needed because bvp_solver._opacity_derivatives previously computed the HARD-switch
    derivative unconditionally, even when config.OPACITY_SMOOTH_TRANSITIONS=True made the
    RESIDUAL use this smoothed kappa instead - a real Jacobian/residual mismatch near a
    transition, the same class of bug as the P/T soft-clamp fix elsewhere this session
    (PROGRESS.md has the full report), just smaller in practice since it only matters within a
    few smoothing widths of a regime boundary rather than across a whole saturated tail.

    Same 0*inf guard as bell_lin_opacity_smooth itself (this section's own comment has the
    full mechanism/verification): an irrelevant regime's kappa_n (or its derivative) can
    overflow at extreme (rho,T) even where its weight (or weight-derivative) is exactly 0 -
    unlike weights (always >= 0), weight DERIVATIVES can be negative, so the guard is
    "!= 0.0", not "> 0.0".
    """
    rho_b, T_b = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(T, dtype=float))
    weights = _regime_weights(rho_b, T_b)
    dweights_drho, dweights_dT = _regime_weights_derivatives(rho_b, T_b)

    dkappa_drho = np.zeros_like(rho_b, dtype=float)
    dkappa_dT = np.zeros_like(rho_b, dtype=float)
    for idx, regime in enumerate(REGIMES):
        w = weights[..., idx]
        dw_drho = dweights_drho[..., idx]
        dw_dT = dweights_dT[..., idx]
        kappa_n = evaluate_regime(regime.kappa_i, regime.a, regime.b, rho_b, T_b)
        dkappa_n_drho = regime.a * kappa_n / rho_b
        dkappa_n_dT = regime.b * kappa_n / T_b
        dkappa_drho += np.where(dw_drho != 0.0, dw_drho * kappa_n, 0.0) + np.where(w > 0.0, w * dkappa_n_drho, 0.0)
        dkappa_dT += np.where(dw_dT != 0.0, dw_dT * kappa_n, 0.0) + np.where(w > 0.0, w * dkappa_n_dT, 0.0)
    return dkappa_drho, dkappa_dT


# ==========================================
# SECTION: Public API — Bell & Lin Opacity
# ==========================================

def bell_lin_opacity(rho, T):
    """Bell & Lin (1994) opacity: kappa(rho, T) [cm^2 g^-1], dispatched across all 8 regimes.

    config.OPACITY_SMOOTH_TRANSITIONS=True dispatches to bell_lin_opacity_smooth instead (a
    logistic-blended kappa(T), eliminating the hard switch's d(kappa)/dT discontinuity at
    regime boundaries - see that section's comment). Default True as of 2026-08-11.
    """
    if config.OPACITY_SMOOTH_TRANSITIONS:
        return bell_lin_opacity_smooth(rho, T)

    rho_b, T_b = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(T, dtype=float))
    regime_index = determine_regime(rho_b, T_b)

    kappa = np.zeros_like(rho_b, dtype=float)
    for idx, regime in enumerate(REGIMES):
        in_regime = regime_index == idx
        kappa = np.where(
            in_regime,
            evaluate_regime(regime.kappa_i, regime.a, regime.b, rho_b, T_b),
            kappa,
        )
    return kappa
