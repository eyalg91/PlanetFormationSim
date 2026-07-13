# state.py — SimulationState: the single mutable data container passed between
# all modules. All physics and solver modules receive a SimulationState and
# return a new one; no module holds mutable state of its own (CLAUDE.md
# Architecture Rules).

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ==========================================
# SECTION: SimulationState Dataclass
# ==========================================

@dataclass
class SimulationState:
    """Structural state of the collapsing gas envelope on the Lagrangian mass grid."""

    m: np.ndarray     # Lagrangian mass coordinate, enclosed mass at each grid point [g]
    r: np.ndarray     # Radius enclosing mass m, from continuity dr/dm = 1/(4*pi*r^2*rho) [cm]
    P: np.ndarray     # Gas pressure, from hydrostatic equilibrium dP/dm = -G*m/(4*pi*r^4) [dyn cm^-2]
    L: np.ndarray     # Luminosity carried through the shell at m (radiative + convective) [erg s^-1]
    T: np.ndarray     # Temperature, from Schwarzschild temperature-gradient equation [K]
    rho: np.ndarray   # Mass density, from ideal-gas EOS: rho = P*mu*m_H/(k_B*T) [g cm^-3]

    t: float = 0.0    # Physical time elapsed since simulation start [s]

    prev: Optional["SimulationState"] = None   # Reference to the previous timestep's converged state, used to finite-difference dT/dt and dP/dt
