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
M_E = 9.1093837015e-28      # Electron mass [g]
PLANCK_H = 6.62607015e-27   # Planck constant [erg s]
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

# ASSUMPTION: mean molecular weight PER ELECTRON (distinct from MU, mean weight per
# particle) for eos.py's electron-degeneracy pressure term (Sub-task 2f). Standard estimate
# for a fully-ionized, solar-like H/He composition: mu_e = 2/(1+X), X~0.71 hydrogen mass
# fraction (Kippenhahn & Weigert) - assumes full ionization, appropriate deep in a
# degenerate-pressure-dominated interior where this term matters, not the cool molecular
# outer envelope MU describes (a separate, smaller inconsistency accepted for this
# first-order treatment - see PROGRESS.md Sub-task 2f entry).
MU_E = 1.17          # Mean molecular weight per electron, solar-like H/He composition [dimensionless]

# ==========================================
# SECTION: Initial Condition — Compact Post-Collapse Protoplanet (t=0)
# ==========================================

# ASSUMPTION: t=0 is NOT a diffuse, pre-collapse GI clump in equilibrium with the ambient disk
# (that construction was tried and found to be an exact, unbreakable fixed point under
# hydrostatic physics - deeply Bonnor-Ebert subcritical, isothermal, L=0; PROGRESS.md Sub-task 5
# pivot). Initial GI collapse is inertia-dominated hydrodynamic free-fall, structurally outside
# what a quasi-static/hydrostatic solver can represent - the same reason T_DISSOCIATION_LIMIT
# below halts the code at the far end of validity. Standard practice (PMS Henyey tracks;
# Bodenheimer & Pollack 1986; Marley et al. 2007 "hot start" gas-giant models) is to hand off
# from an assumed compact, high-entropy post-collapse state and evolve forward quasi-statically
# from there. T_CENTER_INITIAL is therefore a CHOSEN "hot start" parameter, not derived - real
# formation entropy is genuinely uncertain; GI/disk-instability formation (this project's
# premise) generally predicts higher initial entropy than core-accretion "warm starts",
# motivating the upper-middle of a plausible 1000-1500 K range rather than the low end. Kept
# safely below T_DISSOCIATION_LIMIT (2000 K): a just-collapsed "second core" that has not yet
# reached H2 dissociation.
T_CENTER_INITIAL = 1200.0   # Prescribed central temperature of the t=0 compact protoplanet [K]

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
# used in the physics equations. t=0 is a compact, post-dynamical-collapse protoplanet (a few
# R_Jup, PROGRESS.md Sub-task 5 pivot), not a diffuse pre-collapse clump - this is a mean-density
# estimate for M_TOTAL confined to R~3 R_Jup (M_TOTAL/((4/3)*pi*(3*R_Jup)^3)), the same order as
# a real young gas giant's bulk density, not a "diffuse cloud" guess.
RHO_GUESS_INITIAL = 0.05   # Representative density for the shooting-method radius scale only [g cm^-3]
BVP_TOL = 1.0e-8             # Relative/absolute tolerance for the bvp_solver.py shooting integration and root-find [dimensionless]

# ==========================================
# SECTION: Time-Stepping Parameters
# ==========================================

# ASSUMPTION: order-of-magnitude Kelvin-Helmholtz contraction timescale for this mass (1e5-1e6 yr
# per the disk-fragmentation/gas-giant-formation literature), NOT a source term in the energy
# equation (odes.py's dL/dm is the textbook implicit form, dt entering only via the actual
# (T_new-T_prev)/dt, (P_new-P_prev)/dt differences - an earlier attempt to add an explicit
# homologous forcing term on top double-counted compressional heating and was reverted;
# PROGRESS.md Sub-task 8 entry). Used only as (a) a characteristic luminosity-scale reference
# (L_scale ~ G*M_TOTAL^2/(R*T_KH_TIMESCALE_S)) to non-dimensionalize bvp_solver.solve_timestep's
# residuals, and (b) a rough starting-dt reference for time_stepper.run() until Sub-task 9's
# adaptive stepping exists.
T_KH_TIMESCALE_S = 1.0e6 * 3.156e7   # Characteristic Kelvin-Helmholtz contraction timescale, ~1 Myr in seconds [s]


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
