# config.py — Single source of truth for all physical constants, simulation
# parameters, and numerical flags used throughout PlanetFormationSim.
# No numerical literals may appear in any other module; import from here.

# ==========================================
# SECTION: Fundamental Physical Constants (CGS)
# ==========================================

G = 6.67430e-8             # Newtonian gravitational constant [cm^3 g^-1 s^-2]
C_LIGHT = 2.99792458e10     # Speed of light [cm s^-1]
A_RAD = 7.5657e-15          # Radiation constant, a_rad = 4*sigma_SB/c [erg cm^-3 K^-4]
K_B = 1.380649e-16          # Boltzmann constant [erg K^-1]
M_H = 1.67262192369e-24     # Hydrogen atom mass [g]
SIGMA_SB = 5.670374419e-5   # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]

# ==========================================
# SECTION: Nebula Boundary Conditions
# ==========================================

P_NEB = 1.0e4    # Nebular gas pressure imposed at envelope surface, m = M_TOTAL [dyn cm^-2]
T_NEB = 150.0    # Nebular gas temperature imposed at envelope surface, m = M_TOTAL [K]

# ==========================================
# SECTION: Envelope Bulk Properties
# ==========================================

M_TOTAL = 1.898e30   # Total envelope mass, ~1 Jupiter mass [g]
MU = 2.34            # Mean molecular weight of H2/He gas mixture [dimensionless]
GAMMA = 1.4           # Adiabatic index of diatomic-dominated ideal gas [dimensionless]

# ==========================================
# SECTION: Grid & Solver Parameters
# ==========================================

N_GRID_POINTS = 200   # Number of nodes on the Lagrangian mass grid m in [0, M_TOTAL] [dimensionless]

# ==========================================
# SECTION: Opacity Model Flags
# ==========================================

OPACITY_SMOOTH_TRANSITIONS = False  # Bell & Lin (1994) regime switch: False = physically correct hard switch, True = logistic-blended kappa(T) near transitions

# ==========================================
# SECTION: Physical Validity Limits
# ==========================================

# ASSUMPTION: quasi-static hydrostatic equilibrium holds only while hydrogen
# remains molecular. Near T ~ 2000 K, H2 dissociation (~4.5 eV/molecule) drives
# gamma_eff below 4/3, violating hydrostatic stability and triggering dynamical
# free-fall collapse on timescales far shorter than the Kelvin-Helmholtz
# timescale this quasi-static solver assumes.
T_DISSOCIATION_LIMIT = 2000.0   # Core temperature ceiling above which H2 dissociation invalidates the quasi-static assumption [K]
