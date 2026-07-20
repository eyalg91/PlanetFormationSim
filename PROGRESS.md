# PlanetFormationSim — Progress & Documentation Log

**Audience:** you, as the physicist directing this project. This file exists so you can
open it at any point and reconstruct *what has been built, why it was built that way,
and what physical claims are already backed by a passing check* — without re-reading
diffs or code.

For the target physics, the full 4-ODE formulation, and the sub-task roadmap, see
[PLAN.md](PLAN.md). This file tracks actual implementation progress against that plan.

---

## 1. Current Status

**Phase 1 (Static Skeleton & Physical Validation) — Sub-tasks 1, 2a–2e, 3, 4, 5, and 6 are done.**

| Sub-task | Scope | Status |
|---|---|---|
| 1 | `config.py` + `state.py` | ✅ Done |
| 2a | `eos.py` + opacity regime table + power-law evaluator | ✅ Done |
| 2b | `transition_temperature` (density-dependent regime boundaries) | ✅ Done |
| 2c | `determine_regime` + `bell_lin_opacity` (public API) | ✅ Done |
| 2d | Opacity validation suite | ✅ Done (expanded beyond the plan's 4 checks, see §4) |
| 2e | `opacity.py` ↔ `gradients.py` interface preview | ✅ Done (synthetic profile, since `gradients.py` didn't exist yet at the time) |
| 3 | `gradients.py` (Schwarzschild criterion) | ✅ Done |
| 4 | `odes.py` + `boundary_conditions.py` | ✅ Done |
| 5 | `bvp_solver.py` (static solve) | ✅ Done — **significantly revised from PLAN.md, see §5** |
| 6 | `diagnostics.py` | ✅ Done — **exit criteria revised to match Sub-task 5's cold/L=0 state, see §5** |
| 7–10 | Time evolution, adaptive dt, output | ⬜ Not started — **next up: Sub-task 7, `time_stepper.py`** |

**Stubs present but empty:** `main.py`, `ReadMe.txt` — created as placeholders, no content yet.

The first converged structure now exists: `bvp_solver.solve_static_structure()` produces a
real, physically validated $t=0$ `SimulationState` — a cold (T=50 K), non-luminous (L=0),
~13 AU gravitationally-bound gas clump in hydrostatic equilibrium with its disk. This is
*not* what PLAN.md originally specified (a `solve_bvp` call producing a luminous structure) —
Sub-task 5 required a full physics/numerics investigation that changed both the nebula
parameters and the solution method. Read §5's Sub-task 5 entry before touching
`bvp_solver.py` or `config.py`'s nebula constants — the reasoning there is load-bearing.

`diagnostics.py` now reports on that converged state: a generalized (pressure-confined)
virial balance, the opacity regime census, and an independent mass-reconstruction check.
Its exit criteria were revised from PLAN.md's original wording for the same reason as
Sub-task 5's — see §5.

---

## 2. Module Reference

This section is kept in sync with the current contents of each file — it describes what
the code *does now*, not the history of how it got there (that's §5).

### `config.py` — single source of truth for numbers

Every physical constant (CGS: `G`, `C_LIGHT`, `A_RAD`, `K_B`, `M_H`, `SIGMA_SB`), every
nebula/envelope parameter (`P_NEB`, `T_NEB`, `M_TOTAL` ≈ 1 Jupiter mass, `MU` = 2.34,
`GAMMA` = 1.4 for the H₂/He mix), the grid resolution (`N_GRID_POINTS` = 200), and the
physical validity ceiling (`T_DISSOCIATION_LIMIT` = 2000 K) live here and nowhere else.
No other module is allowed to hardcode a numerical literal.

`P_NEB = 1.0e-4 dyn/cm²`, `T_NEB = 50.0 K`: these are the outer-disk (gravitational-
instability/disk-fragmentation) nebula conditions at ~50 AU, per the Hayashi (1981)
minimum-mass solar nebula model — **not** arbitrary round numbers. See Sub-task 5 in §5 for
why the original values (`P_NEB=1e4`, `T_NEB=150K`, inner-disk-like) were physically wrong
for this scenario and had to be replaced. `RHO_GUESS_INITIAL` (1e-6 g/cm³) is a numerical
scale used only by `bvp_solver.py`'s shooting method, not a physical input.

One flag is defined ahead of the module that will use it: `OPACITY_SMOOTH_TRANSITIONS`
(currently `False`), which will later control whether opacity regime transitions in
`opacity.py` use a hard switch (physically correct, but a kink in dκ/dT) or a logistic
blend (smoother for the BVP collocation Jacobian, slightly less "correct").

`T_DISSOCIATION_LIMIT` encodes the physical boundary of this whole simulation approach:
above ~2000 K, H₂ dissociation (endothermic, ~4.5 eV/molecule) drops γ_eff below 4/3,
the envelope becomes dynamically (not just thermally) unstable, and the quasi-static
assumption this solver depends on no longer holds. See PLAN.md §4.6 for the full
argument — this constant is where that physics boundary gets enforced in code
(enforcement itself belongs in the not-yet-written `time_stepper.py`).

### `state.py` — the one mutable data object

`SimulationState` is a `@dataclass` holding the Lagrangian mass grid `m` and the four
solved field arrays `r`, `P`, `L`, `T`, plus a derived `rho`, the elapsed time `t`, and
`prev` — a reference to the previous timestep's converged state (needed later to
finite-difference ∂T/∂t and ∂P/∂t as frozen source terms for the energy equation).

By convention, every physics/solver module takes a `SimulationState` in and returns a
new one — nothing mutates a state in place. No solver logic exists yet that actually
produces one of these beyond hand-constructed test arrays in `validation.py`.

### `eos.py` — ideal gas constitutive relations

Three pure functions:
- `density(P, T, mu)` — inverts the ideal gas law, ρ = Pμm_H/(k_BT). This is how `rho`
  gets derived from the solved `P`, `T` at every grid point (never solved directly).
- `specific_heat_cp(gamma, mu)` — c_p = γR_specific/(γ−1), needed later for the energy
  equation's −c_p ∂T/∂t term.
- `grad_adiabatic(gamma)` — ∇_ad = (γ−1)/γ, the adiabatic temperature gradient used by
  the (not-yet-written) Schwarzschild convection criterion in `gradients.py`.

All three carry an `# ASSUMPTION:` comment noting they're only valid while hydrogen
stays molecular (i.e., below `T_DISSOCIATION_LIMIT`).

### `opacity.py` — Bell & Lin (1994) 8-regime piecewise opacity

This is the most structurally complex module so far, built in three layers:

1. **Data layer:** `REGIMES`, a tuple of 8 immutable `RegimeParams(name, kappa_i, a, b)`
   entries, ordered coolest → hottest (ice grains, ice grain evaporation, metal grains,
   metal grain evaporation, molecules, H⁻ scattering, Kramers bound-free/free-free,
   electron scattering). Each regime is a power law κ = κᵢ·ρᵃ·T^b.
2. **Transitions layer:** `transition_temperature(rho, n)` solves κₙ = κₙ₊₁ analytically
   for the crossing temperature between adjacent regimes as a function of density (the
   regime boundaries are *not* fixed temperatures — they shift with ρ). Because the raw
   per-pair crossing formula can, at low density, produce a crossing temperature that
   falls below several cooler regimes' own transitions (a real bug hit during
   development — the Kramers→electron-scattering crossing lands at ~179 K at
   ρ = 10⁻¹⁵ g/cm³), `monotonic_transition_temperatures` clamps each boundary to the
   running maximum of the boundaries below it. This preserves the n ↔ regime identity
   instead of letting a spurious low crossing get sorted into the wrong slot.
3. **Public API layer:** `determine_regime(rho, T)` returns the regime index at each
   grid point (fully vectorized, via the clamped transition array), and
   `bell_lin_opacity(rho, T)` is the sole function anything outside this file should
   call — it dispatches each point to its regime's power law and returns κ in cm²/g.

### `gradients.py` — Schwarzschild criterion

Two pure functions implementing the temperature-gradient switch between radiative and
convective energy transport (PLAN.md §1, ODE 4 and §4.4):

- `grad_radiative(L, m, P, T, kappa)` — the gradient ∇_rad = 3κLP/(16π·a_rad·c·G·m·T⁴)
  that radiative diffusion *alone* would need in order to carry luminosity `L` past mass
  shell `m` (Kippenhahn & Weigert form). Asserts κ > 0 at entry, as a guard against a
  corrupted `opacity.py` result reaching the solver silently. Diverges at m = 0 by
  construction (L = 0 there too, from the inner boundary condition, making it a
  removable 0/0) — flagged with an `# ASSUMPTION:` comment; callers must not evaluate it
  exactly at the center.
- `effective_gradient(grad_rad, grad_ad)` — the Schwarzschild criterion itself: returns
  `(grad_eff, is_convective)`, where `is_convective = grad_rad > grad_ad` and `grad_eff`
  picks the shallower of the two. A radiative gradient steeper than adiabatic is
  unstable to convective overturn, which flattens the realized gradient to ∇_ad.

**Deviation from PLAN.md:** the plan's deliverable listed the signature as
`grad_radiative(L, m, P, T, kappa, rho)`, but its own formula never uses `rho` — density
doesn't appear anywhere in ∇_rad. Carrying an unused parameter would violate the
project's "no unnecessary abstractions" style rule, so `rho` was dropped from the
implemented signature. Purely a signature cleanup; no physics changed.

### `odes.py` — the 4-ODE right-hand side

One function, `stellar_odes(m, y, dT_dt, dP_dt)`, that assembles everything upstream
(`eos`, `opacity`, `gradients`) into `dy/dm` for `y = [r, P, L, T]` — the RHS that
`scipy.integrate.solve_bvp` will call once time evolution (Sub-tasks 7-8) is running.
Internally, at each mass point it: derives `rho` from the ideal-gas EOS (`eos.density`),
gets `kappa` from `opacity.bell_lin_opacity`, gets `grad_ad` from `eos.grad_adiabatic`,
gets `grad_rad` from `gradients.grad_radiative`, and combines the last two via
`gradients.effective_gradient` to get the Schwarzschild-selected `grad_eff`. From there
the four physical equations (PLAN.md §1) are direct: continuity (`dr/dm`), hydrostatic
equilibrium (`dP/dm`), the energy equation (`dL/dm`, driven by the frozen `dT_dt`/`dP_dt`
source terms), and the temperature structure equation (`dT/dm`, using `grad_eff`).
`bvp_solver.py`'s t=0 shooting method also calls this directly (only 2 of its 4 outputs
used) rather than duplicating the `dr/dm`, `dP/dm` formulas — see below.

### `boundary_conditions.py` — the 4 BVP residuals

One function, `boundary_conditions(ya, yb)`, returning the 4 residuals `solve_bvp` needs
to close the system: center conditions `r=0` and `L=0` (no cavity, no interior energy
source at m=0), and surface conditions `P=P_neb`, `T=T_neb` (the imposed nebular
boundary state at m=M_TOTAL). Not currently called by anything (`bvp_solver.py`'s t=0
shooting method bypasses it — see below) but unchanged and ready for `solve_bvp` calls
once Sub-tasks 7-8 introduce real time evolution.

### `bvp_solver.py` — t=0 static structure via shooting

`solve_static_structure()` returns the converged $t=0$ `SimulationState`. This module's
approach is the single biggest deviation from PLAN.md so far — read the "Sub-task 5"
entry in §5 for the full reasoning; summary:

- **Physics:** with `dT_dt=dP_dt=0` (no previous timestep to difference against — the
  literal definition of a $t=0$ solve), `odes.py`'s energy equation forces `dL/dm≡0`,
  which with the center BC `L(0)=0` forces `L≡0` everywhere, which forces `∇_rad≡0<∇_ad`,
  which forces `dT/dm≡0`: **the envelope is exactly isothermal at `T_neb` and carries zero
  luminosity, mathematically, regardless of solver or initial guess.** Four independent
  attempts to construct a "hot start" instead (to get nonzero L immediately) each required
  physically absurd luminosities (~10³⁴-10³⁷ erg/s) — not a construction flaw, but a
  consequence of `M_TOTAL` being deeply Bonnor-Ebert-subcritical (`M_TOTAL/M_BE≈0.089`),
  which forces a nearly-uniform pressure profile no matter the assumed temperature
  structure. The isothermal, `L=0` result is also the physically *correct* $t=0$ state for
  this scenario (a freshly-fragmented GI clump in equilibrium with its disk) — confirmed
  against the literature, not just accepted by elimination.
- **Numerics:** `scipy.integrate.solve_bvp` proved unreliable for this problem for two
  independent reasons (a structurally rank-deficient ODE Jacobian, and a near-surface
  pressure-scale-height boundary layer). `_reduced_rhs_logm` + `_integrate_outward` +
  `solve_static_structure` instead **shoot**: integrate the reduced 2-ODE `(r,P)` system
  outward from the center with `scipy.integrate.solve_ivp` (holding `L=0`, `T=T_neb`
  fixed, calling `odes.stellar_odes` directly so the ODE formulas aren't duplicated), and
  root-find (`scipy.optimize.brentq`) on the central pressure until `P(M_TOTAL)=P_neb`.
  The independent variable is `x=ln(m)` (m spans ~6 decades; `solve_ivp`'s adaptive step
  control needs a bounded-range independent variable).
- Converges to `P_center≈1.227e-4 dyn/cm²`, `R_surface≈13.0 AU`, surface-pressure residual
  at machine precision (~2e-14 relative).

### `diagnostics.py` — post-solve physical diagnostics

Unlike `validation.py`, this module is *not* a test suite — no asserts. It's the runtime
monitoring/reporting layer PLAN.md's architecture calls for: pure functions that compute
physical quantities from a `SimulationState`, plus `run_diagnostics(state)` which prints a
formatted report. Meant to be called after every future solve once `time_stepper.py`
exists (Sub-tasks 7-8), not just once at $t=0$.

- `virial_balance(state)` → `(E_grav, E_therm, surface_term)`. Implements the
  **pressure-confined** virial theorem, `E_grav + 3(γ-1)E_therm = 3·P_neb·V` — derived by
  integrating hydrostatic equilibrium by parts (not the textbook zero-surface-pressure
  form, which doesn't apply here: `P_neb` is the whole reason this envelope has the size
  it does, same Bonnor-Ebert confinement as Sub-task 5). Verified against the converged
  state before implementation: relative imbalance ~8e-6.
- `opacity_regime_distribution(state)` → fraction of grid points in each of the 8 Bell &
  Lin regimes. At $t=0$ (uniform T=50K), this comes out 100% "Ice grains," the single
  coldest regime — expected, not a bug (see Sub-task 5's isothermal result).
- `mass_reconstruction(state)` → `M(r) = m[0] + ∫4πr²ρ dr`, via cumulative trapezoidal
  quadrature over the converged `(r,ρ)` profile. An independent check on the continuity
  equation and the shooting integration together, since this quadrature is the inverse of
  the same ODE (`dr/dm=1/(4πr²ρ)`), computed by a different numerical method than the
  adaptive integrator that produced the profile.

### `validation.py` — sanity checks, unit consistency, and diagnostic plots

See §4 below for the full walkthrough of what's checked and why.

### `main.py`, `ReadMe.txt`

Both exist as empty placeholders. `main.py` is intended (per PLAN.md §2) to become the
orchestrator that parses config, runs the time loop, and saves output — that requires
`time_stepper.py` and `output.py`, neither of which exist yet.

---

## 3. Diagnostic outputs already on disk

Two plots are generated by `validation.py` when run directly (`python validation.py`):

- **`opacity_transitions.png`** — log-log plot of all 7 regime transition temperatures
  T_{n→n+1}(ρ) over ρ ∈ [10⁻¹⁵, 10⁻⁵] g/cm³. Used to visually confirm the transition
  curves have the expected analytic slopes and don't cross in unphysical ways.
- **`opacity_profile_preview.png`** — κ(m) along a synthetic, hand-built
  centrally-condensed ρ(m)/T(m) profile (not a converged structure), just to preview how
  opacity regimes stack up with depth ahead of `gradients.py` existing.
- **`odes_profile_check.png`** — the constant-density analytic profile plus the relative
  residual between `stellar_odes` and its analytic/finite-difference reference (Sub-task
  4, Check 19).
- **`static_structure_t0.png`** — the first *real, converged* structure: r(m) and P(m) for
  the actual $t=0$ `SimulationState` (Sub-task 5, Check 25). T(m)≡50K and L(m)≡0 are noted
  in the title rather than plotted, since they're trivially flat/zero by construction.
- **`mass_reconstruction_check.png`** — relative error between `diagnostics.mass_
  reconstruction()` and the Lagrangian grid, vs. radius (Sub-task 6, Check 29). Shows the
  expected sharp spike right at the center (finite-resolution effect) decaying to ~1e-4 by
  ~2 AU.

---

## 4. Validation Suite — what each check confirms and why

`validation.py` (run via `python validation.py`) executes 29 checks in sequence. All
currently pass. Per CLAUDE.md, this is the *only* place checks/tests live — none of this
logic is duplicated in the operational modules.

| # | Check | Confirms |
|---|---|---|
| 1 | `check_ideal_gas_eos` | P = ρk_BT/(μm_H) is dimensionally CGS-consistent and gives a finite, positive pressure for representative envelope values. |
| 2 | `check_hydrostatic_equilibrium` | dP/dm = −Gm/(4πr⁴) is CGS-consistent and has the correct (negative) sign. |
| 3 | `check_continuity_equation` | dr/dm = 1/(4πr²ρ) is CGS-consistent and positive. |
| 4 | `check_ideal_gas_density_inverts_pressure` | `eos.density()` reproduces a hand-solved ρ from the ideal gas law to 1e-12 relative tolerance — catches sign/ordering errors in the implementation. |
| 5 | `check_adiabatic_gradient_and_cp_limits` | `eos.grad_adiabatic(5/3)` = 0.4 exactly (monatomic reference case), and both `grad_adiabatic`/`specific_heat_cp` are finite and positive at the project's actual γ = 1.4. |
| 6 | `check_regime_table_reference_points` | Every one of the 8 `REGIMES` rows evaluates to its own `kappa_i` at (ρ, T) = (1, 1) — a trivial but effective guard against transposed/mistyped table entries. |
| 7 | `check_transition_temperature_loglog_slopes` | Each of the 7 transition curves has the analytic log-log slope (aₙ₊₁−aₙ)/(bₙ−bₙ₊₁) in ρ, verified numerically between ρ = 10⁻¹² and 10⁻⁸ g/cm³. |
| 8 | `check_regime_continuity` | κ computed from regime n and regime n+1 agree to <1e-10 relative difference *at* their shared analytic transition temperature — i.e., κ(T) has no jump discontinuity. |
| 9 | `check_regime_ordering_monotonic` | Regime index is non-decreasing as T increases, swept across ρ ∈ [10⁻¹⁵, 10⁻⁵] g/cm³ and T ∈ [100, 50000] K. This is the regression test for the low-density clamping bug described in §2 (`opacity.py`) — it also checks the specific failure point (ρ=10⁻¹⁵, T=190 K) resolves to regime 1 ("Ice grain evaporation"), not a hot regime. |
| 10 | `check_bell_lin_vectorization_stress_test` | `bell_lin_opacity` on a 60×60 (ρ, T) mesh spanning all 8 regimes returns the right shape, no NaN/Inf, and actually exercises every regime (no dead code paths). |
| 11 | `plot_opacity_along_synthetic_profile` | κ(m) along a synthetic profile behaves qualitatively as expected (cool grain opacity at the surface, hot regime at the center) — a sanity preview ahead of `gradients.py`. Notably, this check also records that `T_DISSOCIATION_LIMIT` (2000 K) sits *below* the Molecules→H⁻ transition (~3340 K at ρ=10⁻¹⁰), meaning the real physical run will likely never leave the cool grain/molecular opacity regimes before the dissociation halt triggers — worth remembering when interpreting later diagnostics. |

`print_all_constants()` also runs first as a plain printout, not an assertion-backed
check — just a human-readable dump of everything in `config.py`.

**Sub-task 3 additions (`gradients.py`):**

| # | Check | Confirms |
|---|---|---|
| 12 | `check_convection_triggers_at_grad_rad_exceeds_grad_ad` | Solves the critical luminosity L_crit where ∇_rad = ∇_ad analytically, then confirms `is_convective` is `False` just below it and `True` just above — the switch flips at the right place, not just in the right direction. |
| 13 | `check_radiative_limit_grad_eff_equals_grad_rad` | At L = 10⁻³·L_crit (deep radiative regime), `grad_eff` equals `grad_rad` *exactly* (bit-for-bit, not just numerically close) and `is_convective` is `False`. |
| 14 | `check_convective_limit_grad_eff_equals_grad_ad` | At L = 10³·L_crit (deep convective regime), `grad_eff` equals `grad_ad` *exactly* and `is_convective` is `True`. |
| 15 | `check_grad_radiative_over_full_opacity_regime_sweep` | Sweeps T ∈ [100, 50000] K (same range as opacity Check 9) with κ from `opacity.bell_lin_opacity` and L set to a Kelvin-Helmholtz luminosity estimate (L_KH ~ GM²/(Rt_KH), same construction PLAN.md uses for Sub-task 5). Confirms `grad_radiative` stays finite and positive across all 8 opacity regimes, and `grad_eff` never exceeds the adiabatic ceiling ∇_ad — i.e. the Schwarzschild switch can't produce an unphysically steep realized gradient no matter which opacity regime it's evaluated in. |
| 16 | `check_grad_radiative_rejects_nonpositive_kappa` | Confirms the `assert kappa > 0` guard in `grad_radiative` actually fires for κ = 0 and κ < 0, rather than being dead code that never triggers. |

**Sub-task 4 additions (`odes.py`, `boundary_conditions.py`):**

| # | Check | Confirms |
|---|---|---|
| 17 | `check_stellar_odes_matches_constant_density_analytic_profile` | Builds a closed-form uniform-density self-gravitating sphere (r(m), P(m)) and an adiabatic T(m), then compares `stellar_odes`'s `dr/dm`, `dP/dm`, `dT/dm` against it. See the note below — this check caught a real self-consistency bug in its own first draft, not in the code. |
| 18 | `check_stellar_odes_output_shape_finite_and_signs` | `stellar_odes` output shape matches input, all finite, and signs are physical (dr/dm>0, dP/dm<0, dT/dm<0) — holds regardless of radiative/convective regime, since ∇_eff ≥ 0 always. |
| 19 | `plot_constant_density_profile_ode_check` (**visible check**) | Saves `odes_profile_check.png`: the analytic r(m)/P(m)/T(m) profile plus the relative residual between `stellar_odes` and its analytic/finite-difference reference, across m. A visual sanity check of the full ODE RHS ahead of `bvp_solver.py`. |
| 20 | `check_boundary_conditions_residuals` | Confirms `boundary_conditions` returns exactly 4 residuals, is exactly zero at a state satisfying all 4 BCs, and that perturbing each of `ya[0]`, `ya[2]`, `yb[1]`, `yb[3]` individually shifts exactly its own residual component — a direct index/sign correctness test. |

**A note on Check 17's first draft — a real bug, but in the test, not the code:** the
first attempt built *one* T(m) array (the adiabatic profile) and used it for all three
of `dr/dm`, `dP/dm`, `dT/dm`. `dP/dm` and `dT/dm` agreed with finite differences
immediately, but `dr/dm` was off by ~100%. The cause: `dr/dm` is the only one of the
three that depends on `rho`, and `rho` is derived *internally* inside `stellar_odes` from
the ideal-gas EOS applied to `(P, T)` — it is never told the ρ₀ that was used to build
r(m) in the first place. The adiabatic T(m) (T ∝ P^∇_ad, ∇_ad ≈ 0.286) does not, in
general, feed back through the EOS to reproduce that same ρ₀ — only a T(m) satisfying
T ∝ P^1 does. The fix uses two different T arrays: `T_rho_check` (EOS-inverted so
ρ(P,T) = ρ₀ exactly, used only for the `dr/dm` comparison) and `T_ad_check` (the
adiabatic profile, used only for `dP/dm`/`dT/dm`, which don't depend on ρ at all). This
is a good illustration of why an "obviously correct" analytic test profile can hide a
coupling assumption — the fix is in `validation.py`'s Check 17, `odes.py` itself was
never wrong.

**Sub-task 5 additions (`bvp_solver.py`):**

| # | Check | Confirms |
|---|---|---|
| 21 | `check_nebula_conditions_match_mmsn_at_50au` | Recomputes the Hayashi (1981) MMSN midplane T and P at 50 AU from first principles (surface density + vertical hydrostatic balance) and asserts `config.T_NEB`/`config.P_NEB` agree within factors of 2 and 5 — turns the literature justification into a living, checked assertion rather than just a comment. |
| 22 | `check_envelope_mass_is_bonnor_ebert_subcritical` | Recomputes the Bonnor-Ebert critical mass from `config.py`'s values and asserts `M_TOTAL/M_BE < 1` — guards against a future parameter change silently reopening the over-critical crisis that broke the original `T_NEB=150K`/`P_NEB=1e4` configuration (see below). |
| 23 | `check_static_structure_hydrostatic_balance` | On the converged `solve_static_structure()` output, verifies `dP/dr ≈ -G·m·ρ/r²` in Eulerian form (PLAN.md's original Sub-task 5 exit criterion) — max relative error 2.4e-4 at interior points, well under the 1e-3 threshold. |
| 24 | `check_static_structure_isothermal_and_monotonic` | Confirms `T≡T_neb` and `L≡0` *exactly* (not approximately — this is a hard mathematical identity per `bvp_solver.py`'s derivation), `r` strictly increasing, `P` strictly decreasing, and `P[-1]` matching `P_neb` to solver tolerance. |
| 25 | `plot_static_structure_profile` (**visible check**) | Saves `static_structure_t0.png` — r(m)/P(m) for the converged structure. |

**Sub-task 6 additions (`diagnostics.py`):**

| # | Check | Confirms |
|---|---|---|
| 26 | `check_virial_balance_pressure_confined` | The converged state satisfies the pressure-confined virial theorem to relative imbalance < 1e-3 (actual: ~8e-6), and that `E_grav`, the thermal term, and the surface term are all commensurate in magnitude — catches a sign/unit bug that a coincidental cancellation might otherwise hide. |
| 27 | `check_static_structure_opacity_regime_distribution` | 100% of grid points sit in regime 0 ("Ice grains") — the specific, strong prediction for a uniform T=50K envelope, not just "some sensible-looking distribution." |
| 28 | `check_mass_reconstruction_matches_lagrangian_grid` | `diagnostics.mass_reconstruction()` matches the Lagrangian grid `state.m` to <1% away from the center (first 30 points excluded — a known finite-resolution effect, not a bug, see the plot below). |
| 29 | `plot_mass_reconstruction_error` (**visible check**) | Saves `mass_reconstruction_check.png` — relative error vs. radius, showing the expected center-to-edge falloff. |

---

## 5. Change Log

Most recent first. Each entry: what was done, and the physical/architectural reasoning.

### 2026-07-20 — Sub-task 6: `diagnostics.py` — exit criteria revised for the cold t=0 state

PLAN.md's original Sub-task 6 (virial check against the textbook zero-surface-pressure
coefficient, regime distribution expected to span cold-outer to hot-inner regimes, an
energy-flux check against a nonzero L) assumed the generic converged structure envisioned
before Sub-task 5's investigation. Given what Sub-task 5 actually produces (cold, uniform
T=50K, L≡0 — see that entry below), those criteria needed revision before implementation,
not just at the code level:

- **Virial theorem → pressure-confined form.** The standard virial theorem assumes zero
  surface pressure; `P_neb` is not negligible here (PLAN.md already flagged this as a
  caveat when Sub-task 6 was first written, before Sub-task 5 confirmed how central `P_neb`
  actually is). Derived the generalized form independently, by integrating hydrostatic
  equilibrium by parts: `E_grav + 3(γ-1)E_therm = 3·P_neb·V`. Verified against the actual
  converged state *before* writing any check code: relative imbalance ~8e-6 — confirms both
  the derivation and, independently, that Sub-task 5's shooting solution is highly
  self-consistent. Per instruction, the check logs all three terms rather than hard-failing
  at a tight percentage, so the physical balance is visible even if a future change (e.g. a
  much coarser grid) loosens the numerical agreement.
- **Opacity regime distribution → single-regime prediction.** With T≡50K everywhere, there
  is no hot/cold differentiation to check for (that assumed a differentiated structure this
  project's t=0 doesn't have). Confirmed: 100% of grid points in "Ice grains," the coldest
  Bell & Lin regime — an exact, strong prediction rather than a vague "looks reasonable."
- **Energy flux check → deferred to Sub-task 7.** Trivial with `L≡0` identically; carries no
  diagnostic information until real, nonzero L exists.
- **New: mass reconstruction check.** Not in the original PLAN.md deliverables. Computes
  `M(r) = ∫4πr²ρ dr` from the converged profile via independent quadrature and compares to
  the Lagrangian grid `m` — validates the continuity equation and the shooting integration
  together, since this is the same ODE's inverse relation computed by a different numerical
  method. Matches to <0.5% away from the center; the ~3% error at the innermost few points
  is an expected, understood finite-resolution effect (r changes fastest there), not a bug —
  confirmed by checking that the error shrinks monotonically moving outward before writing
  the check's threshold.

`diagnostics.py` itself holds no asserts (unlike `validation.py`) — per CLAUDE.md's
architecture, it's the operational reporting/monitoring layer PLAN.md assigns it to be,
producing physical quantities for a physicist to read, not a test suite. The corresponding
validation checks (26-29, approved before implementation per the standing protocol) live in
`validation.py` and call `diagnostics.py`'s functions. `PLAN.md`'s Sub-task 6 section was
rewritten in place (not just here) since, as with Sub-task 5, the original text would
actively mislead a future reader about what the checks now verify and why.

### 2026-07-20 — Sub-task 5: `bvp_solver.py` — full reset after a physics investigation

This took several rounds of debugging that each revealed something more fundamental than
the last. Recording the full chain of reasoning here because the *conclusion* (isothermal,
non-luminous $t=0$; shooting instead of `solve_bvp`) looks strange in isolation without it,
and the next person to touch `bvp_solver.py` or `config.py`'s nebula constants needs this
context before "fixing" anything.

**1. `solve_bvp` itself doesn't work for this problem, independent of physics.** First
attempt: call `scipy.integrate.solve_bvp` on the full 4-ODE system with `dT_dt=dP_dt=0`,
per PLAN.md's literal Sub-task 5 deliverable. It failed with a singular Jacobian on
iteration 1, every time, regardless of initial guess, mesh spacing (linear/log), or
independent-variable transform. Diagnosis: `dL/dm = -c_p·dT_dt + dP_dt/ρ` depends *only*
on the externally-prescribed, frozen `dT_dt`/`dP_dt` arrays — never on the state being
solved for. That makes the ODE Jacobian's L-row exactly zero at every point (confirmed by
direct numerical computation: rank 2 of 4), for *any* `dT_dt`/`dP_dt`, not just zero ones.
Reducing to the 2-ODE `(r,P)` subsystem (holding L,T fixed, since they're determined
independently — see point 2) removed that specific degeneracy, but `solve_bvp` still
failed: near the surface, P approaches the much smaller `P_neb` with a short pressure
scale height, driving `d(ln P)/d(ln m)` to enormous values that broke every collocation
mesh/scaling strategy tried. Resolution: abandon `solve_bvp` for this module, use a
**shooting method** (`scipy.integrate.solve_ivp`, an adaptive integrator with no global
Jacobian) instead — a genuine, deliberate deviation from PLAN.md's stated approach.

**2. The `L≡0` mathematical trap.** Independent of the numerics above: with
`dT_dt=dP_dt=0` (no previous timestep — literally what "the first, static solve" means),
`dL/dm≡0` identically, and with the center BC `L(0)=0` this forces **`L≡0` everywhere,
for any state**. With `L=0`, `∇_rad=0` (linear in L) is always below `∇_ad`, so the
Schwarzschild criterion never trips convective, forcing `dT/dm≡0`: the converged $t=0$
envelope is *exactly* isothermal at `T_neb`, mathematically, regardless of solver or
initial guess. This isn't a bug to work around — it's what the equations say.

**3. The original `T_NEB=150K`, `P_NEB=1e4` were physically wrong.** Running the
isothermal shooting solve with the original values failed to converge to any reasonable
structure. Checked against the Bonnor-Ebert critical mass for a pressure-confined
isothermal sphere (`M_BE = 1.18·c_s⁴/√(G³·P_ext)`, Bonnor 1956/Ebert 1955):
`M_TOTAL/M_BE ≈ 99` — the envelope was **99× over the critical mass**, meaning no stable
isothermal hydrostatic equilibrium exists at all at those conditions (the cloud would be
in free-fall collapse, which this quasi-static code cannot follow). Root cause: `T_NEB=
150K`, `P_NEB=1e4 dyn/cm²` describe *inner*-disk-like conditions, inconsistent with
PLAN.md's actual scenario (gravitational instability / disk fragmentation, which occurs in
the cold outer disk, ~50 AU). Corrected using the Hayashi (1981) minimum-mass solar nebula
(MMSN) model at r=50 AU: `T(r)=280·(r/AU)^-0.5` → 39.6 K; midplane pressure via
`Σ_gas(r)=1700·(r/AU)^-1.5` + vertical hydrostatic balance → 4.0e-5 dyn/cm². Set
`T_NEB=50.0 K`, `P_NEB=1.0e-4 dyn/cm²` (within factors of 1.3 and 2.5 of the MMSN
reference — solid agreement for a disk-model order-of-magnitude estimate). This alone
dropped `M_TOTAL/M_BE` to **0.089** — comfortably subcritical.

**4. Four independent attempts at a "hot start" (nonzero L at t=0) all failed the same
way.** Point 2 mathematically rules out nonzero L unless `dT_dt`/`dP_dt` are nonzero, but
before accepting that, four different constructions were tried to get a hot, luminous
initial state anyway (single global adiabat + assumed `L(m)∝m`; single adiabat + L solved
for marginal convection, `∇_rad=∇_ad`; two-zone adiabatic-core + radiative-envelope with L
reverse-derived from a chosen `dT/dm`; a photospheric `L=4πR²σT_eff⁴` construction). All
four require **L ~ 1e34-1e37 erg/s** — thousands to tens of thousands of solar
luminosities, for a sub-Jupiter-mass gas clump. This is not a construction flaw common to
all four attempts by coincidence: `M_TOTAL` being deeply Bonnor-Ebert-subcritical forces
the pressure profile to be nearly *uniform* (center-to-surface within a factor of a few)
**regardless of the assumed internal temperature distribution**. Cramming a large
temperature range (e.g. 50K→700K) into that narrow a pressure range demands an enormous
`|dT/dm|`, and the radiative diffusion equation (`gradients.grad_radiative`) says that
demands an enormous L to sustain. A "hot start" is therefore not achievable at $t=0$ for
this mass/pressure combination, full stop — not a numerical difficulty to push through.

**Conclusion:** the physically correct $t=0$ state is the cold, isothermal, `L=0`
equilibrium-with-the-disk state (point 2), which also happens to match the literature
picture for freshly-fragmented GI clumps (Boss, Mayer, Helled & Bodenheimer et al.:
newly-fragmented clumps are typically tens of AU and near the local disk temperature,
contracting to planetary size only over the much longer Kelvin-Helmholtz timescale) —
confirmed numerically: the shooting solve converges cleanly to **R_surface ≈ 13.0 AU**.
Getting nonzero L to actually start Kelvin-Helmholtz contraction is therefore **not** a
$t=0$ state-construction problem — it's a `time_stepper.py` (Sub-task 7) bootstrap
problem: the first real evolutionary step will need a literature-motivated assumed
initial cooling rate (not "return zero arrays," which would leave the envelope at this
exact fixed point forever, since differencing two identical states always gives zero).
**Flagged here for whoever implements Sub-task 7** — PLAN.md's current Sub-task 7
description says the bootstrap should return zero arrays; that will need to change.

**Implementation:** rewrote `bvp_solver.py` around `solve_static_structure()`, which
shoots on the central pressure via `solve_ivp` (independent variable `x=ln(m)`, since m
spans ~6 decades and `solve_ivp`'s adaptive stepping needs a bounded range) +
`brentq`, calling `odes.stellar_odes` directly (not duplicating its formulas) and holding
`L=0`, `T=T_neb` fixed rather than solving for them. `odes.py` and `boundary_conditions.py`
are untouched — `boundary_conditions.py` is unused for now but stays correct and ready for
Sub-tasks 7-8's real `solve_bvp` calls. Also updated `RHO_GUESS_INITIAL` (1.33 → 1e-6
g/cm³ — a diffuse GI clump, not a mature Jupiter-density planet) since it sets the
shooting method's radius/pressure scale. Per the user's explicit instruction: no blind
`try`/`except`, no forced numerical dampening — `_integrate_outward` raises `RuntimeError`
with the actual `solve_ivp` failure message on a failed trial point rather than silently
trapping it.

Added 5 validation checks (21–25) after proposing them and getting sign-off: an MMSN
reference-value check, a Bonnor-Ebert subcriticality check (guards against this exact
crisis recurring silently if `config.py`'s nebula constants ever change again), a
hydrostatic-balance check (PLAN.md's original exit criterion, in Eulerian form), an
exact-isothermal/monotonicity check, and a profile plot. All 25 checks in `validation.py`
pass. `PLAN.md`'s Sub-task 5 section was updated in place (not just PROGRESS.md) since the
deviation is substantial enough that the original text would actively mislead a future
reader.

### 2026-07-20 — CLAUDE.md rule: prefer visible checks where they fit
Added a standing rule to the Testing & Validation Protocol: prefer a visible check (a
plot of a profile, a residual vs. a coordinate, a comparison curve) over a print-only
assert whenever a check naturally has something to look at — generalizing the pattern
`opacity_profile_preview.png` already established, rather than treating it as a one-off.
First applied immediately after, in Check 19 below (`odes_profile_check.png`).

### 2026-07-20 — Sub-task 4: `odes.py` + `boundary_conditions.py`
Implemented `stellar_odes` (the full 4-equation RHS: continuity, hydrostatic
equilibrium, energy, temperature structure — wiring together `eos`, `opacity`, and
`gradients` for the first time) and `boundary_conditions` (the 4 residuals closing the
system at the center and surface). This is the last module before `bvp_solver.py`
(Sub-task 5) can make an actual `scipy.integrate.solve_bvp` call.

Added 4 validation checks (17–20) after proposing them and getting sign-off: an analytic
constant-density-sphere profile test for `dr/dm`/`dP/dm`/`dT/dm`, a shape/finiteness/sign
sanity check, a visible profile+residual plot, and a boundary-residual indexing test. The
first draft of Check 17 surfaced a genuine self-consistency bug in the *test*, not in
`odes.py` — see §4 above for the full explanation. All 20 checks in `validation.py` pass.

### 2026-07-20 — Sub-task 3: `gradients.py` (Schwarzschild criterion)
Implemented `grad_radiative` and `effective_gradient`, giving the simulation its
radiative-vs-convective energy-transport switch (ODE 4 in PLAN.md §1). This is the last
purely local physics module before `odes.py` (Sub-task 4) has to actually assemble the
4-equation RHS `solve_bvp` will integrate — `gradients.py` is the piece that decides,
pointwise, whether heat moves by radiative diffusion or convective overturn.

Deviated from the PLAN.md deliverable signature by dropping the unused `rho` parameter
from `grad_radiative` (the ∇_rad formula never uses density — see §2 above for the
reasoning). Added 5 validation checks (12–16) after proposing them and getting sign-off,
per the CLAUDE.md protocol: a critical-luminosity crossing test, exact-equality checks at
both the deep-radiative and deep-convective limits, a full T ∈ [100, 50000] K sweep
across all 8 opacity regimes bounding `grad_eff <= grad_ad`, and a check that the
non-positive-κ guard actually raises. All 16 checks in `validation.py` pass.

### 2026-07-20 — Progress log established
Created this file retroactively to document Sub-tasks 1 and 2a–2e (config, state, EOS,
opacity, and their validation suite), which were already implemented and committed
(`e58af91`). Added a standing rule to `CLAUDE.md` requiring this file to be updated
after every subsequent task, so project state stays legible without reconstructing it
from git history.

### Prior work (commit `e58af91`)
Implemented `config.py`, `state.py`, `eos.py`, `opacity.py`, and `validation.py`
end-to-end, covering PLAN.md Sub-tasks 1 and 2a–2e. This included resolving a real bug
in opacity regime assignment at low density (spurious low-temperature crossing between
the Kramers and electron-scattering regimes corrupting cooler regime boundaries) via the
monotonic-clamping approach described in §2. 11 validation checks and 2 diagnostic plots
were added as a result.

### Commit `95f923a` — PLAN.md established
Wrote the architecture and development plan (physics formulation, module structure,
data-flow diagram, key design decisions, and the 10-sub-task implementation sequence)
that all subsequent work is tracked against.
