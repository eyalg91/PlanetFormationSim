# boundary_conditions.py — Boundary residuals for scipy.integrate.solve_bvp: the
# center (m=0) and surface (m=M_TOTAL) conditions that close the 4-ODE system in
# odes.py. Pure function, no side effects (CLAUDE.md Architecture Rules).

import numpy as np

import config

# ==========================================
# SECTION: Center and Surface Boundary Residuals
# ==========================================

def boundary_conditions(ya, yb):
    """4 residuals for solve_bvp: r=0, L=0 at the center; P=P_neb, T=T_neb at the surface."""
    r_a, _, L_a, _ = ya
    _, P_b, _, T_b = yb

    # Center (m=0): r=0 (no cavity at the envelope's center), L=0 (no energy source interior
    # to the center, since m=0 encloses no mass to generate or carry luminosity)
    # Surface (m=M_total): P=P_neb, T=T_neb (imposed nebular gas boundary conditions)
    return np.array([r_a, L_a, P_b - config.P_NEB, T_b - config.T_NEB])
