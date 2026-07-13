# opacity.py — Bell & Lin (1994) piecewise power-law opacity.
# Layer 1: immutable regime data + raw power-law evaluator.
# Layer 2: transition_temperature.
# Layer 3 (this sub-task): determine_regime and bell_lin_opacity, the sole
# public API. Pure functions, no side effects.

import warnings
from collections import namedtuple

import numpy as np

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
# SECTION: Public API — Bell & Lin Opacity
# ==========================================

def bell_lin_opacity(rho, T):
    """Bell & Lin (1994) opacity: kappa(rho, T) [cm^2 g^-1], dispatched across all 8 regimes."""
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
