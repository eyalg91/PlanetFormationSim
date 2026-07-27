# boundary_conditions.py — Boundary residuals for the 4-ODE system in odes.py: the
# center (m=0) and surface (photosphere) conditions. Pure function, no side effects
# (CLAUDE.md Architecture Rules).
#
# PHYSICAL NOTE on the surface conditions (both revised from the original diffuse-cloud
# design; PROGRESS.md has the full numerical trail behind each):
#
# THERMAL: originally a rigid T(M_TOTAL)=T_neb clamp (literal thermal contact with the
# ambient nebula gas). Correct for the old t=0 static solve, but created an exact,
# unbreakable degeneracy for real time evolution: whenever a state already satisfies
# T=T_neb at the surface, "no further change at all" trivially satisfies the same
# condition again - proven analytically and confirmed numerically (Sub-task 8
# investigation). A real photosphere does not stay clamped to the ambient gas temperature;
# it develops its own radiating temperature as internal energy flows out. Replaced with a
# net radiative flux balance: the photosphere emits at its own T and absorbs from the
# ambient field at T_neb, so the net outward luminosity is the difference,
# L = 4*pi*r^2*sigma_SB*(T^4 - T_neb^4). Reduces to exactly T=T_neb, L=0 at equilibrium.
#
# MECHANICAL: originally P(M_TOTAL)=P_neb (literal pressure confinement by the ambient
# nebula gas, appropriate for a diffuse, disk-pressure-confined cloud). Once t=0 became a
# compact, degenerate-pressure-supported protoplanet (Sub-task 2f), this had NO SOLUTION AT
# ALL - not a hard-to-find root, a genuine gap in achievable surface pressure (confirmed:
# P_end jumps discontinuously from trapped below ~0.05-0.08 dyn/cm^2 to >=2.79e6, with
# P_neb=1e-4 squarely inside the gap; not a tolerance artifact - tightening solve_ivp's
# rtol by 10^4 changed nothing). Physically expected: a degenerate object's atmosphere does
# not connect to the ambient nebula via the same bulk radiative-diffusion equation of state
# all the way down to P_neb - it hands off to a photosphere, defined by optical depth
# tau=2/3, at a pressure set by the star's own surface gravity and opacity, not by the
# ambient pressure. Replaced with the standard Eddington grey-atmosphere result
# (Kippenhahn & Weigert Ch. 12): integrating hydrostatic equilibrium (dP/dr=-g*rho) against
# optical depth (dtau=kappa*rho*dr) from tau=0 (vacuum edge, P=0) to tau=2/3, treating
# kappa as locally constant over this thin layer, gives P_photosphere = (2/3)*g/kappa.
# P_neb drops out of the mechanical condition entirely; only T_neb continues to matter
# (via the thermal term above). This changes HOW the surface is LOCATED, not just the
# residual formula - bvp_solver.py integrates outward with tau=2/3 as a solve_ivp event and
# matches the ENCLOSED MASS at that event to M_TOTAL, rather than checking a residual at a
# fixed m=M_TOTAL grid endpoint (the fixed-endpoint approach was tested and found to have
# the same reachability gap the old P=P_neb condition did - PROGRESS.md has the trail).

import numpy as np

import config
import eos
import opacity

# ==========================================
# SECTION: Photospheric Pressure (Eddington tau=2/3)
# ==========================================

def photospheric_pressure(r, P, T, mu, mu_e):
    """The pressure the Eddington tau=2/3 photospheric condition predicts at radius r,
    given the LOCAL (P, T) (used only to get rho -> kappa via the EOS/opacity - not treated
    as the answer itself): P_photo = (2/3)*g/kappa, g = G*M_TOTAL/r^2 (virtually all mass is
    interior to the photosphere), kappa = opacity.bell_lin_opacity(rho, T).

    Used two ways: (1) as a solve_ivp EVENT during outward integration
    (bvp_solver._photosphere_event returns P - photospheric_pressure(r, P, T, ...), zero-
    crossing marks the photosphere), and (2) as the boundary_conditions() mechanical
    residual below, evaluated at the converged event point.

    ASSUMPTION: grey atmosphere (kappa treated as constant over the thin tau=0 to 2/3
    layer) and that the atmosphere's own mass/thickness is negligible against the star's
    total (so g is well-approximated as constant there) - both standard, and both far more
    appropriate for a compact, self-gravitating object than the ambient-pressure-
    confinement condition this replaces (see module docstring).
    """
    rho = eos.density(P, T, mu, mu_e)
    kappa = opacity.bell_lin_opacity(rho, T)
    g = config.G * config.M_TOTAL / r**2
    return (2.0 / 3.0) * g / kappa


# ==========================================
# SECTION: Center and Surface Boundary Residuals
# ==========================================

def boundary_conditions(ya, yb):
    """4 residuals: r=0, L=0 at the center; P=P_photosphere (mechanical, Eddington tau=2/3)
    and the net radiative flux balance L=4*pi*r^2*sigma_SB*(T^4-T_neb^4) (thermal) at the
    surface (photosphere)."""
    r_a, _, L_a, _ = ya
    r_b, P_b, L_b, T_b = yb

    # Center (m=0): r=0 (no cavity at the envelope's center), L=0 (no energy source interior
    # to the center, since m=0 encloses no mass to generate or carry luminosity)
    # Surface (photosphere): P = (2/3)*g/kappa (Eddington tau=2/3, module docstring); the
    # photosphere's net radiated luminosity balances emission at T against absorption from
    # the ambient field at T_neb   [erg s^-1]
    P_expected = photospheric_pressure(r_b, P_b, T_b, config.MU, config.MU_E)
    L_expected = 4.0 * np.pi * r_b**2 * config.SIGMA_SB * (T_b**4 - config.T_NEB**4)
    return np.array([r_a, L_a, P_b - P_expected, L_b - L_expected])
