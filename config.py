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

# CORRECTED 2026-08-07 (PLAN_BVP.md Milestone 0/5, PROGRESS.md has the full trail): MU=2.34
# (molecular H2/He) and GAMMA=1.4 (diatomic) were carried over from an earlier, cooler-regime
# assumption. Stage 3 (this project's actual scope, PLAN.md "Formation Scenario and Scope")
# starts PAST H2 dissociation by construction (T_CENTER_INITIAL=13000K, well above the ~2000K
# dissociation threshold) - the envelope is atomic, not molecular, essentially everywhere in
# this project's T range. Milestone 0's Saha calculation confirmed ionization itself is
# negligible here (peak x~5.9e-4), but also surfaced a SEPARATE, real ~1.83x gap: MU should be
# the neutral ATOMIC value (~1.278, solar H/He: mu~4/(1+3X+Y), X~0.71,Y~0.27), not the
# molecular one. GAMMA follows the same atomic-vs-molecular logic: monatomic species have only
# 3 translational degrees of freedom (Cv=3/2*k_B, gamma=5/3), not diatomic's rotational modes
# (Cv=5/2*k_B, gamma=7/5=1.4) - a bare H/He atom has no molecular bonds to store rotational
# energy in. ASSUMPTION: still no ionization-dependent mu(rho,T) (that remains Sub-task 8a,
# Saha equation, PLAN.md - confirmed non-negligible only in the tail of the T range, not
# needed for the bulk of Stage 3) - this is the atomic-vs-molecular correction only, verified
# independent of and much cheaper than that.
MU = 1.278           # Mean molecular weight of neutral ATOMIC H/He mixture, post-dissociation [dimensionless]
GAMMA = 5.0 / 3.0     # Adiabatic index of monatomic (atomic, post-dissociation) ideal gas [dimensionless]

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

# ASSUMPTION: t=0 represents Stage 3 of the three-stage GI-formation picture (PLAN.md
# "Formation Scenario and Scope"): the compact, hot "second core" that forms once the
# dynamical second collapse (triggered by H2 dissociation at T~2000K, Stage 1->2) halts due
# to ionization and electron degeneracy pressure re-stiffening the EOS - NOT the diffuse
# pre-collapse cloud (an exact, unbreakable fixed point under hydrostatic physics,
# PROGRESS.md Sub-task 5 pivot), and NOT Stage 1's first core either. T_CENTER_INITIAL is a
# CHOSEN "hot start" parameter, not derived - real formation entropy is genuinely uncertain.
#
# DECIDED 2026-08-01 ("Geometric Target" approach, PROGRESS.md has the full numerical
# trail): literature (present-day Jupiter's own modeled central temperature, ~2.2-2.5e4 K)
# motivates a value in the 2e4-5e4 K range, but this codebase's simplified EOS (fixed-mu
# ideal gas, no ionization physics) does NOT reproduce a compact R~2-4 R_Jup structure
# there - direct marching showed R already exceeds 4 R_Jup by T~1.7e4 K and keeps climbing.
# Chose to prioritize the geometric target (R) over the literature-motivated temperature
# target (T) for now, isolating the time-stepper infrastructure work (Sub-task 8) from the
# EOS ionization gap - which is deliberately deferred to, and must be closed by, the new
# mandatory Sub-task 8a before this T value (or Stage 1 modeling) can be trusted
# quantitatively. This value is well above T_DISSOCIATION_LIMIT's old 2000 K threshold
# (consistent with genuinely being past H2 dissociation, Stage 3) but is NOT self-consistent
# with respect to hydrogen ionization (Sub-task 8a) - accepted as a second-order error since
# degeneracy pressure dominates the mechanical structure here.
#
# LOWERED 2026-08-07 (13000K -> 11500K; PROGRESS.md/PLAN_BVP.md have the full trail): the
# same session's MU/GAMMA correction (molecular->atomic, dissociation-consistent - see MU's
# own comment above) increased ideal-gas thermal pressure support enough that 13000K's
# adiabatic seed construction became genuinely INFEASIBLE, not just numerically hard to
# bracket: swept P_center across nearly 6 orders of magnitude at 13000K and mass_error never
# reached zero - its minimum still overshoots M_TOTAL by 5.36%, confirmed by direct
# calculation, not assumed. A scan across T found the feasibility boundary between 12000K
# (root exists, m_surface/M_TOTAL=0.9922 at best) and 13000K (infeasible) - 11500K sits
# comfortably below that boundary (0.9613 at best) rather than right at its edge, while
# staying well above the ~2000K H2-dissociation threshold that defines Stage 3's lower bound.
T_CENTER_INITIAL = 11500.0   # Prescribed central temperature of the t=0 compact protoplanet [K]

# ==========================================
# SECTION: Grid & Solver Parameters
# ==========================================

N_GRID_POINTS = 200   # Number of nodes on the Lagrangian mass grid m in [0, M_TOTAL] [dimensionless]

# ASSUMPTION: dr/dm = 1/(4*pi*r^2*rho) formally diverges at the true center (m=0, r=0). The mass
# grid starts at a tiny but nonzero m_min = M_MIN_FRACTION*M_TOTAL instead of exactly 0, standard
# practice for Lagrangian stellar-structure BVPs (Kippenhahn & Weigert); the innermost shell's
# mass is negligible, so the center BCs (r=0, L=0) still hold to excellent approximation there.
M_MIN_FRACTION = 1.0e-6   # Fractional mass of the innermost grid point, m_min/M_TOTAL [dimensionless]

# ASSUMPTION: a single np.logspace(m_min, m_surface) puts the outer 10% of mass (where T, rho,
# P actually change fastest, near the photosphere) into only ~0.05 of the grid's ~6 decades in
# log-space, under-resolving that region regardless of solve_ivp's own accurate dense
# interpolant (PROGRESS.md 2026-08-01 entry). bvp_solver._build_output_grid instead composites
# a log-spaced core with a log-spaced-in-distance-to-surface outer region.
GRID_OUTER_MASS_FRACTION = 0.1     # Fraction of M_TOTAL (nearest the surface) sampled by the denser outer grid [dimensionless]
GRID_OUTER_POINT_FRACTION = 0.3    # Fraction of N_GRID_POINTS allocated to that outer region [dimensionless]
GRID_OUTER_REFINEMENT = 1.0e-4     # Finest outer sampling, as a fraction of the outer region's own mass span [dimensionless]

# Representative density for bvp_solver.py's shooting-method radius/pressure scale only, NOT
# used in the physics equations. t=0 is a compact, post-dynamical-collapse protoplanet (a few
# R_Jup, PROGRESS.md Sub-task 5 pivot), not a diffuse pre-collapse clump - this is a mean-density
# estimate for M_TOTAL confined to R~3 R_Jup (M_TOTAL/((4/3)*pi*(3*R_Jup)^3)), the same order as
# a real young gas giant's bulk density, not a "diffuse cloud" guess.
RHO_GUESS_INITIAL = 0.05   # Representative density for the shooting-method radius scale only [g cm^-3]
BVP_TOL = 1.0e-8             # Relative/absolute tolerance for the bvp_solver.py shooting integration and root-find [dimensionless]

# ASSUMPTION: scipy.optimize.fsolve's ier==1 reflects its xtol criterion on the STEP SIZE
# between iterates, NOT the residual/function value - these are not the same thing, and
# ier==1 can be reported even when the residual is far from converged, if the Jacobian is
# badly scaled (confirmed 2026-08-01, PROGRESS.md has the full trace: at
# T_CENTER_INITIAL=13000K, relax_initial_state's fsolve step collapsed to exactly 0 while
# the thermal residual stayed frozen at ~1.3e-2). NOT derived from BVP_TOL (there is no
# guaranteed relationship between fsolve's step-size tolerance and its residual size - that
# gap is exactly the bug this constant exists to catch) - chosen empirically, comfortably
# above the residuals fsolve-based genuine convergence has always produced in this codebase
# (<=1e-7 in every validated case so far) and comfortably below the spurious-convergence
# case found (~1.3e-2), so it cleanly separates the two without being so tight it flags
# healthy solver noise. Used by relax_initial_state and solve_timestep (both fsolve-based) -
# NOT solve_static_structure, which uses brentq, a different method with its own,
# independently-established normal precision (STATIC_STRUCTURE_RESIDUAL_TOL below).
RESIDUAL_TOL = 1.0e-4   # Maximum acceptable |residual| after fsolve reports ier==1 - independently verified, not just trusted [dimensionless]

# ASSUMPTION: solve_static_structure's brentq root-find is bracket-based, not subject to the
# fsolve ier-vs-residual gap RESIDUAL_TOL exists to catch (the true root stays trapped
# between P_low, P_high throughout). Its mass residual is limited instead by the coarseness
# of _build_output_grid's photosphere-event localization, and has always been ~1e-3 in every
# validated case so far (1.554e-3 at T_CENTER_INITIAL=1200K, 8.780e-4 at 13000K) - looser
# than RESIDUAL_TOL, but that reflects this method's own normal, already-accepted precision,
# not a red flag. A dedicated, looser tolerance for the same "verify, don't just trust"
# check without false-alarming on business-as-usual brentq precision.
STATIC_STRUCTURE_RESIDUAL_TOL = 1.0e-2   # Maximum acceptable mass residual from solve_static_structure's brentq root-find [dimensionless]

# ==========================================
# SECTION: Time-Stepping Parameters
# ==========================================

SECONDS_PER_YEAR = 3.156e7   # Julian-year-ish conversion, used for time_stepper.run()'s human-readable logging [s yr^-1]

# ASSUMPTION: order-of-magnitude Kelvin-Helmholtz contraction timescale for this mass (1e5-1e6 yr
# per the disk-fragmentation/gas-giant-formation literature), NOT a source term in the energy
# equation (odes.py's dL/dm is the textbook implicit form, dt entering only via the actual
# (T_new-T_prev)/dt, (P_new-P_prev)/dt differences - an earlier attempt to add an explicit
# homologous forcing term on top double-counted compressional heating and was reverted;
# PROGRESS.md Sub-task 8 entry). Used only as (a) a characteristic luminosity-scale reference
# (L_scale ~ G*M_TOTAL^2/(R*T_KH_TIMESCALE_S)) to non-dimensionalize bvp_solver.solve_timestep's
# residuals, and (b) a rough starting-dt reference for time_stepper.run() until Sub-task 9's
# adaptive stepping exists.
T_KH_TIMESCALE_S = 1.0e6 * SECONDS_PER_YEAR   # Characteristic Kelvin-Helmholtz contraction timescale, ~1 Myr in seconds [s]


# ==========================================
# SECTION: Opacity Model Flags
# ==========================================

OPACITY_SMOOTH_TRANSITIONS = False  # Bell & Lin (1994) regime switch: False = physically correct hard switch, True = logistic-blended kappa(T) near transitions

# ==========================================
# SECTION: Reference Units (reporting only, never used in the physics equations)
# ==========================================

R_JUPITER_CM = 6.9911e9    # Jupiter's present-day equatorial radius, used as a reporting/reference unit [cm]
L_SUN_ERG_S = 3.828e33     # IAU nominal solar luminosity, used to report L_surface in human-readable units [erg s^-1]

# ==========================================
# SECTION: Radiative Gradient — Smoothed L>=0 Floor (gradients.py)
# ==========================================

# ASSUMPTION: gradients.grad_radiative's L>=0 floor (added 2026-08-01 to stop a temperature-
# inversion runaway near the photosphere) was a HARD floor, L_safe=max(L,0) - a genuine
# non-differentiable kink at L=0. Confirmed 2026-08-06 (PROGRESS.md has the full trace) to be
# the root cause of relax_initial_state's adaptive alpha-stepping wall at alpha~0.046: an
# instrumented run of the exact failing LM call showed grad_radiative being invoked >40,000
# times with pre-floor L<0 (spanning m/M_TOTAL=0.89-1.00, down to L~-8e54 erg/s - far beyond
# any physical scale) from a SINGLE LM finite-difference Jacobian probe (~1e-7 relative
# perturbation in P_center/T_center) - both LM's own outer Jacobian estimate and Radau's inner
# implicit-stage Newton iteration (which also assumes RHS smoothness) were tripping on the
# same kink. Replaced with a smooth hyperbolic floor (gradients.py:
# L_safe = 0.5*(L + sqrt(L^2 + GRAD_RAD_L_FLOOR_EPSILON^2)) -> max(L,0) exactly as epsilon->0,
# the standard smoothed-absolute-value/pseudo-Huber form).
#
# epsilon is anchored to the same fixed Kelvin-Helmholtz-timescale luminosity scale already
# used to non-dimensionalize bvp_solver's thermal residual (L_scale in relax_initial_state/
# solve_timestep), but evaluated at a fixed reference radius (R_JUPITER_CM) rather than a live
# state's r[-1], so it can be a true config constant (gradients.py must stay a pure function -
# CLAUDE.md architecture rules) instead of a guessed number.
#
# CORRECTED 2026-08-06 (first pass used a 1e-3 fraction and was WRONG - PROGRESS.md has the
# full numerical trail): this KH-virial estimate is a fine order-of-magnitude *residual
# normalizer* (bvp_solver's L_scale already relies on it, and it doesn't matter there that
# it's off by a large factor - only the RATIO to itself matters for judging "is this small").
# It is NOT a good *absolute smoothing-width anchor* for the same reason: it is already
# established (2026-08-01, bvp_solver.py's L_scale ASSUMPTION) that the r[-1]-based version of
# this estimate overestimates the genuine converged photospheric L at T_CENTER_INITIAL=13000K
# by ~78x (~2.6e29 vs ~3.4e27 erg/s); evaluating at the smaller fixed R_JUPITER_CM instead of
# the true r[-1] inflates that to ~320x. A standalone check of L_safe=0.5*(L+sqrt(L^2+eps^2))
# against L_KH_SCALE_ERG_S*1e-3 confirmed this empirically: it distorted L by 21-24% at
# L~1e27, right where the genuine solution lives - not the negligible-except-very-near-zero
# behavior intended.
#
# CORRECTED AGAIN 2026-08-06 (second pass, 1e-6 fraction, was ALSO too large - PROGRESS.md):
# validation.py's Check 12 (an arbitrary, unrelated synthetic test point, m=1e29 g, T=500 K)
# produces its own L_crit~2.3e24 erg/s - and epsilon at the 1e-6 fraction (~1.1e24 erg/s) was
# comparable to, not negligible against, THAT scale too, silently distorting grad_eff there
# and breaking Check 12/13/14's convective/radiative-limit assertions. The lesson generalizes:
# epsilon must stay negligible against the SMALLEST L this function is plausibly ever handed
# (across validation checks, the full KH-contraction time evolution, and any future T_center),
# not just the one largest scale (T_CENTER_INITIAL=13000K's ~3e27) used to derive it - a
# single incidental low-L test point exposed the gap immediately. 1e-9 (rather than 1e-6)
# lands epsilon (~1.1e21 erg/s) at <0.05% of Check 12's L_crit AND <1e-6 of the genuine
# T=13000K scale - re-verified against both the standalone check and Check 12/13/14, and
# against a full re-run of the alpha-ramp (confirming the original alpha~0.046 wall stays
# fixed even at this much narrower width - PROGRESS.md has the numbers).
L_KH_SCALE_ERG_S = G * M_TOTAL**2 / (R_JUPITER_CM * T_KH_TIMESCALE_S)   # Reference KH-timescale luminosity at a fixed radius, for use as a smoothing scale [erg s^-1]
GRAD_RAD_L_FLOOR_EPSILON = 1.0e-9 * L_KH_SCALE_ERG_S   # Smoothing width of grad_radiative's hyperbolic L>=0 floor [erg s^-1]

# ==========================================
# SECTION: Schwarzschild Criterion — Smoothed Convective/Radiative Switch (gradients.py)
# ==========================================

# ASSUMPTION: gradients.effective_gradient's Schwarzschild selection (grad_eff =
# min(grad_rad, grad_ad), implemented as a hard np.where switch) idealizes convection as
# infinitely efficient - once grad_rad > grad_ad, the gradient snaps instantly to grad_ad.
# Confirmed 2026-08-06 (PROGRESS.md) to be a SECOND kink, structurally identical to the L>=0
# floor's, blocking relax_initial_state's adaptive alpha-ramp at alpha~0.050946 (right after
# the L-floor fix cleared the earlier alpha~0.0465 wall): an instrumented trace of the exact
# failing LM call found grad_rad landing within 3e-5 relative of grad_ad (=2/7 exactly, for
# config.GAMMA=1.4) at several points along the trial profile - the trajectory runs almost
# exactly along the convective boundary, so the hard switch flips under an infinitesimal
# (P_center, T_center) perturbation, the same failure mode as before.
#
# Unlike the L-floor, this switch is not an artificial safety clamp - it is a real physical
# idealization of the Schwarzschild criterion. The standard, smoother alternative is mixing-
# length theory's continuous interpolation between radiative and adiabatic transport as a
# function of superadiabaticity (grad_rad-grad_ad); NOT implemented here - flagged as a new
# mandatory future sub-task in PLAN.md, the same treatment given the EOS ionization gap. This
# is a numerical-only interim smoothing to unblock the relaxation homotopy, reusing the same
# hyperbolic "smoothed absolute value" family already used for the L-floor:
# min(a,b) = 0.5*(a+b) - 0.5*|a-b|  ->  min_smooth(a,b) = 0.5*(a+b) - 0.5*sqrt((a-b)^2+eps^2)
#
# grad_rad and grad_ad are both O(0.1-1), dimensionless, and do NOT span the many-decade range
# L does (grad_ad is in fact a pure constant, (gamma-1)/gamma) - so unlike
# GRAD_RAD_L_FLOOR_EPSILON, a single fixed epsilon is appropriate here without a regime-
# dependent derivation. Verified via a standalone check (PROGRESS.md) to comfortably bridge
# the ~3e-5-7e-4 relative gaps observed at the failing point while staying far below the
# ~0.01-0.1-scale differences typical away from the convective boundary.
#
# NOTE: grad_rad itself CAN still swing over many decades (opacity/L-dependent) even though
# grad_ad does not - validation.py Check 15 exercises nabla_rad up to ~8.6e8. gradients.py's
# effective_gradient computes the smoothing in a cancellation-safe form specifically because
# of this (caught 2026-08-06: the naive subtraction form let grad_eff exceed grad_ad in float64
# at that scale, purely from catastrophic cancellation, not a math error) - see its comment.
GRAD_EFF_SWITCH_EPSILON = 1.0e-4   # Smoothing width of effective_gradient's min(grad_rad,grad_ad) switch [dimensionless]

# ASSUMPTION 2026-08-08 (PROGRESS.md - full diagnostic trail): a SECOND, wider smoothing
# width, used ONLY by bvp_solver.solve_timestep's real-dt solves (relax_initial_state keeps
# the narrow GRAD_EFF_SWITCH_EPSILON above, unchanged - proven robust at every T_center
# tested this session). Motivation: a real timestep can collapse the outer envelope's L by
# ~70x in one step (the Kelvin-Helmholtz source term legitimately shrinking as the surface
# approaches radiative equilibrium), dragging grad_rad down from deeply super-adiabatic
# (ratio to grad_ad of 100-1000x) to GENUINELY MARGINAL (grad_rad landing within a few
# percent of grad_ad, with MULTIPLE sign changes, across an extended mass range m/M_TOTAL in
# [0.993, 0.99998]) - confirmed directly via a full-profile superadiabaticity histogram, not
# assumed. At the narrow GRAD_EFF_SWITCH_EPSILON, solve_bvp's adaptive mesh refinement chases
# this band's curvature (~1/eps) without bound (node count grew super-linearly,
# 43569->79660->129932, rather than plateauing, confirming a genuine local difficulty, not a
# simple resolution shortfall). A single GLOBAL widened epsilon was tried first and found to
# be a genuine either/or: eps>=0.07 fixes solve_timestep but BREAKS relax_initial_state's own
# continuation (a real regression - NaN residual at alpha=0.999, not just more nodes needed),
# because state_0's own forced-adiabat construction has a different-shaped near-surface
# transition that the SAME widened switch distorts. The two failure modes are cleanly
# separated by WHICH FUNCTION is calling, not by anything varying within a call, so a
# context-dependent value (narrow for relax_initial_state, wide for solve_timestep) resolves
# both: verified over a full 5-real-step chain (dt=1e4 yr each) - T_center and r_surface
# decrease smoothly and monotonically (contraction, PLAN.md Sub-task 8's exit criterion),
# T_surface/L_surface settle toward a small, steady, slightly-positive value near T_NEB with
# shrinking step-to-step increments (consistent with relaxing toward quasi-steady radiative
# equilibrium, not diverging).
#
# HONESTLY, NOT A PHYSICAL MODEL: this is a purely numerical regularization of the
# infinitely-efficient-convection idealization, not an approximation to mixing-length theory
# (no convective velocity, no mixing length, no dependence on superadiabaticity beyond this
# switch's own smoothing) - it trades a small, LOCALIZED distortion of the thin near-surface
# transition layer's exact T(m) gradient (confirmed to affect T_surface/L_surface measurably,
# e.g. ~0.17K/14x respectively between eps=0.1 and eps=0.5 - PROGRESS.md) for numerical
# tractability of the outer time loop. Bulk quantities (T_center, r_surface, core structure)
# are affected far less (<0.1% between those same two eps values) since the affected zone is
# an extremely thin, low-mass surface layer. A genuine mixing-length treatment remains the
# mathematically complete fix and is formally scheduled as future work (PLAN.md, post-thesis
# - explicitly out of scope for the one-week deadline this was developed under).
#
# Value chosen with margin, not guessed - and revised once, honestly, when the first choice
# proved insufficient: swept eps in [0.01, 0.5] against the step-2 failure specifically -
# the boundary there is between 0.05 (fails) and 0.07 (succeeds), so eps=0.1 (~1.5-2x margin)
# was tried first and DID resolve step 2 and a validated 5-real-step chain cleanly. But a
# full 10-step dry run (PLAN.md Sub-task 8, PROGRESS.md 2026-08-08) failed at step 6 with
# eps=0.1 - the same super-linear mesh-growth signature, meaning the marginal-convection
# band's difficulty is not a fixed, one-time obstacle but continues to evolve step to step,
# and eps=0.1's margin was not enough beyond step 5. eps=0.5 (already validated once, in
# isolation, for step 2, with an even more comfortable direct convergence than 0.1) was
# retried for the full chain and got through all 10 steps cleanly. Since relax_initial_state
# uses the SEPARATE, unchanged GRAD_EFF_SWITCH_EPSILON above and never sees this constant,
# raising this value carries no risk of reopening that regression - confirmed directly
# (relax_initial_state's own regression check is unaffected either way).
#
# HONEST LIMITATION, not swept under the rug: only validated for 10 real steps. If a longer
# production run is attempted later, re-apply the same margin-finding discipline (sweep a
# candidate value against the actual failing step, don't assume 0.5 holds indefinitely) -
# there is no proof yet that the marginal band's demands plateau rather than keep growing.
GRAD_EFF_SWITCH_EPSILON_TIMESTEP = 5.0e-1   # Wider smoothing width, solve_timestep's real-dt solves only [dimensionless]

# ==========================================
# SECTION: Simulation Halt Condition
# ==========================================

# ASSUMPTION: t=0 (Stage 3 of PLAN.md's "Formation Scenario and Scope") already starts PAST
# H2 dissociation - the dynamical second collapse that crosses that threshold (Stage 1->2)
# is out of scope, not modeled here. The former T_DISSOCIATION_LIMIT halt (checked for
# T_center rising toward 2000 K, as in a non-degenerate pre-main-sequence contraction) does
# not apply to this project's forward evolution, which instead COOLS from a hot, compact
# start (a degenerate-pressure-supported contraction, not a virial-theorem-driven heating
# one - PROGRESS.md 2026-08-01 entry). Replaced with a radius halt: contraction toward
# today's Jupiter is the physically meaningful endpoint of the modeled Kelvin-Helmholtz
# track. It will need reinstating (with its original 2000 K value) if Stage 1 (the first
# core, PLAN.md "Phase 3 - Extensions") is ever modeled separately.
R_HALT = 1.0 * R_JUPITER_CM   # Surface radius at which time_stepper.run() halts (Sub-task 8) [cm]

# ==========================================
# SECTION: solve_bvp Collocation Solver (bvp_solver.py t>0, promoted from bvp_experiment.py 2026-08-08)
# ==========================================
#
# PLAN_BVP.md Milestone 6 (2026-08-07) replaced bvp_solver.py's shooting-based t>0 solve
# (relax_initial_state/solve_timestep) with scipy.integrate.solve_bvp collocation, after the
# shooting Jacobian was found to be structurally rank-deficient (100% convective saturation
# under the infinitely-efficient-convection idealization - PLAN_BVP.md Milestone 4). These
# constants were proven at T_CENTER_INITIAL=11500K and 12000K in bvp_experiment.py before
# being promoted here as production values (PLAN.md Sub-task 5 update, 2026-08-08).

# ASSUMPTION: same pseudo-relaxation-timestep convention as the old (archived) shooting
# relax_initial_state - NOT real elapsed physical time (bvp_solver.relax_initial_state
# leaves state.t unchanged). Order-of-magnitude choice (1% of the KH contraction timescale),
# not derived; used only to make state_0's assumed-adiabatic construction into a genuine
# solution of the real, time-differenced 4-ODE system before real time evolution begins.
RELAX_DT_FRACTION = 0.01   # Fraction of T_KH_TIMESCALE_S used as relax_initial_state's pseudo-timestep [dimensionless]

# ASSUMPTION: looser than BVP_TOL (which governs solve_static_structure's solve_ivp/brentq
# precision, unaffected by this pivot) by design - solve_bvp's collocation residual control
# is a different numerical quantity (global mesh residual, not a local ODE step error) and
# was never tuned tighter than this in bvp_experiment.py's proven runs (residuals achieved
# were still machine-precision-level in practice, 1e-7 to 1e-16 - PLAN_BVP.md Milestone 6/
# section 3.6.4).
BVP_COLLOCATION_TOL = 1.0e-6   # scipy.integrate.solve_bvp's own tol parameter (global mesh residual control) [dimensionless]

# RAISED 2026-08-07 (20000->80000) - PLAN_BVP.md Milestone 6: with scaling+analytic
# Jacobians, the continuation's near-alpha=1 step exhausted the old 20000-node budget while
# still actively refining (a mesh-budget limit, not a divergence) - PROGRESS.md has the trace.
BVP_MAX_NODES = 80000   # scipy.integrate.solve_bvp's max_nodes parameter [dimensionless]

# ASSUMPTION: denser than N_GRID_POINTS=200 (the OUTPUT sampling grid, unaffected by this
# pivot) - solve_bvp's 4th-order collocation scheme extrapolates a midpoint value from y and
# dy/dx at each FIXED mesh interval BEFORE any Newton refinement; a first attempt at
# N_GRID_POINTS=200 overshot into an unphysical (P,T) region on that very first midpoint
# evaluation (PROGRESS.md 2026-08-06). A denser initial mesh bounds that per-interval
# extrapolation more tightly.
BVP_MESH_N_GRID_POINTS = 2000   # solve_bvp-specific initial mesh density [dimensionless]

# ASSUMPTION: much deeper than GRID_OUTER_REFINEMENT=1e-4 (the OUTPUT/reporting grid's own
# refinement, unaffected by this pivot). Found 2026-08-08 (PROGRESS.md) diagnosing a
# solve_timestep-only crash: GRID_OUTER_REFINEMENT's log-spaced outer points stop at
# delta_min=GRID_OUTER_REFINEMENT*0.1*M_TOTAL from the surface, leaving a SINGLE final,
# UNRESOLVED gap (m_surface-delta_min to m_surface itself) with no intermediate points - at
# 1e-4 this final gap is ~1e-5 of M_TOTAL, tiny in mass terms but P/T genuinely drop by
# ~3 orders of magnitude across it in a converged, self-consistent solution (a real,
# extremely thin photospheric "skin", not a numerical artifact). solve_bvp's cubic-Hermite
# collocation midpoint estimate (built from y AND dy/dx at each mesh interval's endpoints)
# massively overshoots when dy/dx itself jumps ~860x between the last two guess points - a
# swept comparison (1e-4 through 1e-12) found the max CONSECUTIVE dy/dx ratio drops from
# ~860x to ~1.2x already at 1e-6, giving solve_bvp a smooth enough initial trial to Newton-
# correct from, without needing the output grid's own resolution changed at all (solve_bvp's
# own dense interpolant is smooth regardless of how coarsely it's later SAMPLED for
# reporting - only the INITIAL guess mesh fed into the Newton iteration needs this).
BVP_MESH_OUTER_REFINEMENT = 1.0e-6   # Guess-mesh-only outer refinement (deeper than GRID_OUTER_REFINEMENT) [dimensionless]

# PLAN_BVP.md Milestone 6 (2026-08-07): the literal alpha=1.0 continuation endpoint diverges
# via exponentially escalating mesh refinement to NaN, while alpha=0.99999 converges cleanly
# - diagnosed as the tiny adiabatic admixture acting as a regularizer for a marginal
# instability in the pure, unblended system (consistent with, not fully explained by, the
# rank-deficiency finding above). A quantifiably negligible (0.001%) adiabatic contamination,
# not a discretization artifact.
BVP_ALPHA_MAX = 1.0 - 1.0e-5   # Continuation endpoint for the nabla_eff alpha-blend, just short of the unstable literal 1.0 [dimensionless]

# Empirically-tuned continuation ladder (PLAN_BVP.md Milestone 6/3.6.4): finer near alpha=1
# where the real, Schwarzschild-selected gradient's rank-deficient regions dominate; coarser
# near alpha=0 where the pure adiabat (well-conditioned) still holds. Proven at both
# T_CENTER_INITIAL=11500K and 12000K.
BVP_ALPHA_CONTINUATION_STEPS = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, BVP_ALPHA_MAX)   # nabla_eff alpha-blend continuation schedule [dimensionless]

# ASSUMPTION: a wrong analytic Jacobian is worse than none (steers Newton confidently in the
# wrong direction) - validation.py's Jacobian-correctness check cross-verifies fun_jac/bc_jac
# against central finite differences at this many randomly-sampled mesh points before they
# are trusted, using a row-normalized relative-error metric (PROGRESS.md 2026-08-07: a naive
# per-entry/output-value metric produces false alarms when both analytic and FD values are
# legitimately near zero).
JACOBIAN_VERIFY_N_POINTS = 15    # Number of randomly-sampled mesh points checked against finite differences [dimensionless]
JACOBIAN_VERIFY_REL_STEP = 1.0e-6   # Relative finite-difference step size for the Jacobian cross-check [dimensionless]
JACOBIAN_VERIFY_TOL = 1.0e-4     # Maximum acceptable row-normalized relative error before refusing to trust the analytic Jacobian [dimensionless]
