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

# Gravitational-instability (disk-fragmentation) scenario: envelope forms and is confined by
# the OUTER protoplanetary disk, ~50 AU, not the inner disk. Values are the Hayashi (1981)
# minimum-mass solar nebula (MMSN) midplane conditions at r = 50 AU:
#   T(r) = 280 K * (r/AU)^-0.5                          -> T(50 AU) ~ 39.6 K
#   Sigma_gas(r) = 1700 g cm^-2 * (r/AU)^-1.5, vertical hydrostatic balance (H = c_s/Omega,
#   rho_mid = Sigma/(sqrt(2*pi)*H), P_mid = rho_mid*c_s^2) -> P_mid(50 AU) ~ 4.0e-5 dyn cm^-2
# T_NEB, P_NEB below are within factors of 1.3 and 2.5 of these reference values respectively,
# well within normal disk-model uncertainty (flaring, viscous heating, disk mass normalization).
P_NEB = 1.0e-4   # Nebular gas pressure imposed at envelope surface, m = M_TOTAL [dyn cm^-2]
T_NEB = 50.0     # Nebular gas temperature imposed at envelope surface, m = M_TOTAL [K]

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

# ASSUMPTION: dr/dm = 1/(4*pi*r^2*rho) formally diverges at the true center (m=0, r=0). The mass
# grid starts at a tiny but nonzero m_min = M_MIN_FRACTION*M_TOTAL instead of exactly 0, standard
# practice for Lagrangian stellar-structure BVPs (Kippenhahn & Weigert); the innermost shell's
# mass is negligible, so the center BCs (r=0, L=0) still hold to excellent approximation there.
M_MIN_FRACTION = 1.0e-6   # Fractional mass of the innermost grid point, m_min/M_TOTAL [dimensionless]

# Representative density for bvp_solver.py's shooting-method radius/pressure scale only, NOT
# used in the physics equations. A freshly-fragmented GI clump confined by the low P_NEB above
# is far more diffuse than a mature planet, so this is ~Jupiter's mean density scaled down many
# orders of magnitude, not a "Jupiter-like" guess.
RHO_GUESS_INITIAL = 1.0e-6   # Representative density for the shooting-method radius scale only [g cm^-3]
BVP_TOL = 1.0e-8             # Relative/absolute tolerance for the bvp_solver.py shooting integration and root-find [dimensionless]

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
