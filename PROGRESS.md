# PlanetFormationSim — Progress & Documentation Log

**Audience:** you, as the physicist directing this project. This file exists so you can
open it at any point and reconstruct *what has been built, why it was built that way, and
what is and isn't currently trustworthy* — without re-reading diffs or chat history.

For the target physics, the full 4-ODE formulation, and the sub-task roadmap, see
[PLAN.md](PLAN.md). This file tracks actual implementation progress against that plan.

---

## 0. Architecture Baseline — How the Simulation Works Right Now (written 2026-08-13)

**Purpose of this section:** a from-the-code, ground-truth description of the current
simulation architecture, written fresh (not from memory of past sessions) specifically as a
troubleshooting baseline for the active Phase 1 (First Hydrostatic Core) investigation. §§1-5
below are the historical narrative of how each piece came to be; this section is the
snapshot of *what runs today*, in one place, with every active numerical knob named. Where
Phase 1 and Phase 3 configurations differ, both are noted, but Phase 1's values (the
currently active troubleshooting context) are called out explicitly.

### 0.1 Physical picture

The code solves a 1D, spherically symmetric, quasi-static (hydrostatic) gas envelope on a
Lagrangian mass grid $m\in[m_\text{min}, M_\text{TOTAL}]$, using implicit (Henyey-style) time
differencing: at each timestep, the full spatial structure is re-solved as a two-point
boundary value problem (BVP), with the *previous* timestep's converged profile supplying the
frozen $dT/dt$, $dP/dt$ source terms in the energy equation. There is no explicit time
integrator in the ODE sense — time enters only through these finite-difference source terms,
re-evaluated against a fully re-relaxed spatial structure every step. This is the standard
technique used by production stellar-evolution codes (e.g. MESA) for exactly this reason:
each step is unconditionally stable in time, at the cost of solving a nonlinear BVP per step.

Two distinct physical regimes are modeled with the *same* code, selected entirely via
`config.py` runtime overrides in the run script (never a code branch):
- **Phase 3** (`run_phase3_validation.py`, `config.py`'s own persisted defaults): a hot,
  compact, post-second-collapse protoplanet ($T_\text{center}$ starting at 11500 K, a few
  $R_\text{Jup}$), atomic composition (`MU=1.278`, `GAMMA=5/3`), electron-degeneracy pressure
  significant, H↔H₂ recombination physics active (`USE_H2_RECOMBINATION_PHYSICS=True`),
  Gyr-scale timesteps. Currently PAUSED (§1).
- **Phase 1** (`run_phase1_first_core.py` / `run_phase1_baseline_rerun.py`): a diffuse,
  cool, fully molecular First Hydrostatic Core ($T_\text{center}$ starting at 645 K, ~500
  $R_\text{Jup}$), constant molecular composition (`MU=2.34`, `GAMMA=1.4`), degeneracy
  pressure negligible, H↔H₂ recombination physics deliberately OFF
  (`USE_H2_RECOMBINATION_PHYSICS=False` — composition held strictly constant), yr-to-kyr-scale
  timesteps. **This is the currently active investigation.**

### 0.2 State representation

`state.SimulationState` (`state.py`) is the one mutable container: arrays `m, r, P, L, T,
rho` on the Lagrangian mass grid, plus scalar `t` and a `prev` back-reference (used only for
`is_convective` bookkeeping in snapshots, not by the solver itself — the solver always takes
an explicit `state_prev` argument). Every module is a pure function: given a state (and
parameters), returns a new state or new arrays. No module holds its own mutable state.

Internally, wherever the solver represents pressure and temperature, it always uses $\ln P$,
$\ln T$ (never raw $P,T$) — this guarantees $P=e^{\ln P}>0$, $T=e^{\ln T}>0$ hold
automatically through arbitrary Newton excursions, the standard Henyey/MESA convention. For
$t>0$ solves, $r$ and $L$ are *additionally* nondimensionalized (§0.9) because their natural
scales are wildly different from $\ln P,\ln T$'s $O(1$–$100)$ range — this is a genuine,
measured Jacobian-conditioning fix (a raw state vector was found with $L$ 28 orders of
magnitude larger than $\ln T$ in the same Newton system), not cosmetic.

### 0.3 Building the $t=0$ initial state — `bvp_solver.solve_static_structure`

The very first state is built by a **separate, simpler machinery** from everything else in
the file: a 3-ODE ($r,\ln P,\ln T$) fully-convective adiabat (`_adiabatic_rhs_logm`), assumed
uniform-composition and non-self-consistent in $L$ (no coupled energy equation — $T(P)$
literally follows the adiabat $\nabla_\text{ad}=(\gamma-1)/\gamma$ by construction, and a
diagnostic $L(m)$ is only backed out *afterward* via the marginally-efficient-convection
closure, `gradients.marginal_convective_luminosity` — never fed back into the construction).
This is integrated outward from a prescribed $T_\text{center}$ (`config.T_CENTER_INITIAL`,
645 K for Phase 1) with `scipy.integrate.solve_ivp` (Radau), shooting on the one free
parameter $P_\text{center}$ via `brentq`, until a `solve_ivp` **event**
(`_photosphere_event_adiabatic`) locates the photosphere (Eddington $\tau=2/3$,
`boundary_conditions.photospheric_pressure`) and the *enclosed mass at that event* is matched
to `M_TOTAL` (not a residual at a fixed grid endpoint — a diffuse/compact structure's
photosphere sits at a genuinely different enclosed-mass fraction, so the domain itself isn't
known in advance).

$P_\text{center}$ is bracketed, not blindly searched: an analytic Lane-Emden polytrope seed
gives a first estimate, then the bracket is expanded geometrically (×1.03 per step, either
direction, up to 300 steps) until `mass_error(P_center)` changes sign, and `brentq` polishes
the root (`xtol=config.STATIC_STRUCTURE_BRENTQ_XTOL=2e-12`, `rtol=config.BVP_TOL=1e-8`).
**Two Lane-Emden seed variants exist, dispatched by `use_ideal_gas_seed`:**
- `_adiabatic_center_guess()` (default) — the pure $T=0$ electron-degenerate $n=1.5$
  polytrope, a function of fundamental constants alone. Correct/proven for Phase 3's compact,
  degeneracy-dominated regime. **Silently wrong for a diffuse molecular structure** — it
  anchors the bracket search onto the wrong physical root entirely (verified directly, not
  assumed: gives a plausible-looking but wrong $r_\text{surface}=3.27\,R_\text{Jup}$ if used
  for Phase 1's composition).
- `_adiabatic_center_guess_ideal_gas(T_center)` — thermally-set polytropic constant
  ($n=1/(\gamma-1)$, $K=C\,T_\text{center}\,\rho_c^{-1/n}$), the seed Phase 1 uses
  (`use_ideal_gas_seed=True`). Calibrated: $T_\text{center}=645$ K gives
  $r_\text{surface}=500.83\,R_\text{Jup}$.

The result is packaged into a `SimulationState` at `t=0`, sampled on `_build_output_grid`'s
composite mass grid (§0.8) — but this state is **not yet a genuine solution** of the real
4-equation system (§0.5); it only satisfies the pure-adiabat 3-ODE construction. That's what
§0.4 fixes.

### 0.4 Relaxation — `bvp_solver.relax_initial_state`

Turns the $t=0$ adiabatic construction into a genuine solution of the full, coupled 4-ODE
system (§0.5) via one or two `solve_bvp` collocation solves (§0.8) at a small **pseudo**-time
step (`dt_relax = config.RELAX_DT_FRACTION * config.T_KH_TIMESCALE_S` — NOT real elapsed
time; `state.t` is left unchanged throughout). Runs in up to two stages:

- **Stage 1** (always runs): relaxes under `USE_H2_RECOMBINATION_PHYSICS` forced `False`
  (constant $\mu,\gamma$) regardless of the run's real setting — this is the numerically
  proven-convergent path. For Phase 1, since the run's real setting is *already*
  `USE_H2_RECOMBINATION_PHYSICS=False`, stage 1 **is** the final answer and stage 2 never
  runs (`relax_initial_state` returns `state_mid` directly — see the `if not
  config.USE_H2_RECOMBINATION_PHYSICS: return state_mid` early exit). For Phase 1
  specifically: `force_clamp_off_stage1=False` (soft clamp, §0.6, stays ON — Phase 1's
  trajectory was found to crash outright, raw `np.exp()` overflow, with it off) and
  `RELAX_DT_FRACTION` is overridden to `1e-3` (Phase 1-local; default is `0.01`).
- **Stage 2** (Phase 3 only, since it needs `USE_H2_RECOMBINATION_PHYSICS=True`): a single
  MICRO solve (`RELAX_RECOMBINATION_MICRO_DT_FRACTION=0.001`, smaller than stage 1's own
  `dt_relax`) that turns the real, T-dependent $\mu(T),\gamma_\text{eff}(T)$ physics back on,
  warm-started from stage 1's solution — lets the composition transition at the cool
  photospheric boundary walk on gradually rather than in one Newton leap. **Not exercised
  by Phase 1 at all** (the flag is off for the whole run, not just stage 1).

Both stages use `switch_epsilon=config.GRAD_EFF_SWITCH_EPSILON` (the *narrow* Schwarzschild
switch width, $10^{-4}$ — see §0.10's table; distinct from the wider value real timesteps
use).

### 0.5 The 4-ODE system — `odes.stellar_odes`

State vector $y=[r,P,L,T]$ on mass coordinate $m$; every $t>0$ solve (relaxation and every
real timestep alike) solves this same system:

1. **Continuity**: $dr/dm = 1/(4\pi r^2\rho)$
2. **Hydrostatic equilibrium**: $dP/dm = -Gm/(4\pi r^4)$
3. **Energy** (Kelvin-Helmholtz contraction source, Kippenhahn & Weigert eq. 4.26):
   $dL/dm = -c_p\,(dT/dt) + (\delta/\rho)(dP/dt)$, where $dT/dt,dP/dt$ are the **frozen**,
   implicit time derivatives — $(T_\text{trial}-T_\text{prev})/\Delta t$,
   $(P_\text{trial}-P_\text{prev})/\Delta t$, with $T_\text{prev},P_\text{prev}$ linearly
   interpolated from the previous converged state onto the current trial's mass grid
   (`_interp_state_prev`). $\delta=-(\partial\ln\rho/\partial\ln T)_P$ is the genuine,
   EOS-dependent thermodynamic coefficient (`eos.thermodynamic_delta`; $\to 1$ for pure ideal
   gas, $\to 0$ as degeneracy dominates — always $\approx 1$ for Phase 1, degeneracy
   negligible there). $c_p$ is `eos.specific_heat_cp(gamma_eff, mu) +
   eos.latent_heat_capacity(T)` — the second term is the H₂-dissociation latent-heat
   correction, which returns identically zero when `USE_H2_RECOMBINATION_PHYSICS=False`
   (Phase 1's case, as of the 2026-08-13 gating fix — see §5's Change Log).
4. **Temperature structure** (Schwarzschild criterion): $dT/dm = (T/P)\,\nabla_\text{eff}\,
   dP/dm$, where $\nabla_\text{eff}=\min(\nabla_\text{rad},\nabla_\text{ad})$ (§0.6) selects
   radiative or (idealized, infinitely-efficient) convective transport at each point.

$\mu(T)$ and $\gamma_\text{eff}(T)$ (`eos.mean_molecular_weight`, `eos.gamma_effective`) are
evaluated fresh at every call — under `USE_H2_RECOMBINATION_PHYSICS=False` (Phase 1) both
flatten exactly to the constants `config.MU=2.34`, `config.GAMMA=1.4` everywhere, so the
composition is *exactly* uniform through the whole envelope for every Phase 1 solve, by
construction.

### 0.6 Closure relations feeding the ODEs

- **EOS** (`eos.density`): $P=P_\text{ideal}(\rho,T)+P_\text{degenerate}(\rho)$, inverted for
  $\rho$ via vectorized Newton-Raphson (ideal-gas-only seed). For Phase 1's density range
  ($\rho\sim10^{-8}$–$10^{-6}\,\text{g/cm}^3$ at the wall, per the 2026-08-13 diagnostic scan)
  the degenerate term is utterly negligible — this is functionally a pure ideal gas for Phase
  1, even though the combined EOS is always evaluated.
- **Opacity** (`opacity.bell_lin_opacity`): Bell & Lin (1994), 8 piecewise power-law regimes
  in $(\rho,T)$ (Ice grains → Ice grain evaporation → Metal grains → Metal grain evaporation →
  Molecules → H⁻ scattering → Kramers bound-free/free-free → electron scattering). With
  `config.OPACITY_SMOOTH_TRANSITIONS=True` (always, currently), a logistic partition-of-unity
  blend (`bell_lin_opacity_smooth`, width `OPACITY_TRANSITION_SMOOTH_WIDTH_DEX=0.005 dex`)
  replaces the hard regime switch, eliminating a genuine $d\kappa/dT$ discontinuity at
  boundaries (confirmed root cause of a real Phase 3 mesh explosion, 2026-08-11).
- **Radiative gradient** (`gradients.grad_radiative`): $\nabla_\text{rad}=3\kappa L_\text{safe}
  P/(16\pi a_\text{rad} c\, G\, m\, T^4)$, with $L_\text{safe}$ a smoothed $L\ge0$ floor
  (hyperbolic, width `GRAD_RAD_L_FLOOR_EPSILON = 1e-9 * L_KH_SCALE_ERG_S`) — prevents a
  temperature-inversion runaway from a transient negative-$L$ Newton trial near the
  photosphere.
- **Schwarzschild switch** (`gradients.effective_gradient`): $\nabla_\text{eff}=
  \min_\text{smooth}(\nabla_\text{rad},\nabla_\text{ad})$, a smoothed minimum (same hyperbolic
  family as the $L$-floor) replacing a hard `np.where`. **This is the single most consequential
  smoothing parameter in the whole solver** for the current Phase 1 investigation — see §0.10.
- **Adiabatic gradient**: $\nabla_\text{ad}=(\gamma_\text{eff}(T)-1)/\gamma_\text{eff}(T)$ —
  a true constant, $0.2857\ldots$, for Phase 1 (since $\gamma_\text{eff}\equiv 1.4$ there,
  flag off).

### 0.7 Boundary conditions (`boundary_conditions.py`, applied via `bvp_solver.make_bc_scaled`)

Two conditions at each end of the mass domain, four residuals total:
- **Center** ($m=m_\text{min}=$ `M_MIN_FRACTION`$\times M_\text{TOTAL}=10^{-6}\,M_\text{TOTAL}$,
  not exactly $m=0$, avoiding continuity's removable $1/r^2$ singularity there): $r(m_\text{min})$
  is tied to the *live* trial central density via the analytic constant-density relation
  $r=(3m_\text{min}/4\pi\rho_c)^{1/3}$ (re-evaluated every Newton iteration, not a fixed
  pre-estimate), and $L(m_\text{min})=0$ (no core energy source).
- **Surface** ($m=M_\text{TOTAL}$): mechanical condition is the Eddington $\tau=2/3$
  photospheric pressure, $P_\text{photo}=(2/3)g/\kappa$ (`boundary_conditions.
  photospheric_pressure`) — NOT a fixed ambient pressure; thermal condition is a net radiative
  flux balance, $L=4\pi r^2\sigma_\text{SB}(T^4-T_\text{NEB}^4)$ — NOT a fixed $T=T_\text{NEB}$
  clamp (both were revised from an earlier, simpler design that had genuine degeneracy/
  unreachability problems — §5's older entries have the trail).

### 0.8 Per-timestep solve — `bvp_solver.solve_timestep`, the collocation machinery

Every real timestep (and both relaxation stages) is solved by
`scipy.integrate.solve_bvp` — a global collocation/relaxation method (4th-order, Lobatto IIIa
error control), the same numerical family as Henyey's implicit relaxation used in production
stellar-evolution codes. This REPLACED an earlier shooting-method solver (root-finding via
`scipy.optimize.root`/`fsolve` on two unknowns) in 2026-08-08, after that approach was traced
to a structural Jacobian rank-deficiency: under the idealized, infinitely-efficient-convection
Schwarzschild switch, a fully-convective-saturated region makes
$d\nabla_\text{eff}/d\nabla_\text{rad}=0$ identically, decoupling $L$ from the $P$-$T$
relation — **this is the same mechanism the 2026-08-13 diagnostic instrumentation
(`run_scripts/diag_singular_jacobian.py`) found reappearing inside `solve_bvp`'s own
collocation Jacobian for Phase 1's deeply-convective envelope**, not a new failure mode.

Mesh and initial guess (`_build_mesh_and_guess`): a composite mass grid — log-spaced in the
core, log-spaced-in-distance-to-surface over the outer `GRID_OUTER_MASS_FRACTION=10%` of mass
— at `BVP_MESH_N_GRID_POINTS=2000` initial points (denser than the `N_GRID_POINTS=200`
output/reporting grid), warm-started from the previous converged state's own $(r,\ln P,\ln
T,L)$ profile (`warm_start_L=True` for real timesteps — the previous state's $L(m)$ is itself
a genuine converged solution, a far better guess than any synthetic ramp).

`solve_bvp` is called with `tol=config.BVP_COLLOCATION_TOL=1e-6`, `max_nodes=
config.BVP_MAX_NODES=80000`, and **analytic** `fun_jac`/`bc_jac` (not scipy's default
finite-difference Jacobian) — `implicit_rhs_jacobian_scaled`/`make_bc_jacobian_scaled`, hand-
derived and cross-checked against finite differences (`validation.py`'s Jacobian-correctness
check, `JACOBIAN_VERIFY_N_POINTS=15` random mesh points, tolerance `1e-4`) before being
trusted. This is what the 2026-08-13 diagnostic instrumentation intercepts directly (the exact
sparse matrix scipy factorizes each Newton iteration), not a hand-reconstruction of it.

### 0.9 State-vector nondimensionalization (t>0 only)

$y=[r,\ln P,L,\ln T]\to z=[\hat r,\ln P,\hat L,\ln T]$: $\hat r=r/R_\text{SCALE}$
($R_\text{SCALE}=R_\text{Jup}$, linear), $\hat L=\text{arcsinh}(L/L_\text{SCALE})$
($L_\text{SCALE}=$ a fixed Kelvin-Helmholtz-timescale reference luminosity, $GM_\text{TOTAL}^2/
(R_\text{Jup}\,T_\text{KH})$ — nonlinear, sign-preserving, log-like compression). `solve_bvp`
only ever sees $z$; `implicit_rhs_scaled`/`implicit_rhs_jacobian_scaled` wrap the physical-
space RHS/Jacobian with the appropriate chain-rule scaling (including a genuine second-
derivative correction term in the $(\hat L,\hat L)$ Jacobian entry, since $\hat L$'s own
scaling factor is itself $L$-dependent).

### 0.10 The soft clamp (`_safe_exp_state`) — surviving wild Newton trials

Before ANY physics function sees $(P,T)$, `_safe_exp_state` passes $(\ln P,\ln T)$ through a
smooth two-sided saturation (`_soft_clamp`, a composed softplus construction) toward bounds
$\ln P\in[-100,100]$, $\ln T\in[0,100]$ (`LN_P_CLAMP`, `LN_T_MIN`, `LN_T_MAX`), width
`BVP_SOFT_CLAMP_WIDTH=0.1` (natural-log units) — replacing an earlier **hard** `np.clip` that
had exactly zero derivative once saturated (both preventing any Newton correction from ever
pulling a wayward trial back, and making the analytic Jacobian actively *wrong*, not just
imprecise, in the saturated region — confirmed root cause of a 2026-08-11 mesh explosion).
The soft version is C-∞ with a strictly nonzero derivative for roughly 75 natural-log units
past either boundary (`_soft_clamp_derivative`) — every Jacobian row that converts a
$d/dP,d/dT$ into $d/d(\ln P),d/d(\ln T)$ must multiply by the ACTUAL clamped derivative
(`_safe_exp_state_derivatives`), never assume $dP/d(\ln P)=P$.

### 0.11 Solve orchestration — `_solve_structure_bvp`: direct attempt → continuation → retry

Three nested layers, from fastest/cheapest to most defensive:

1. **Direct attempt** at $\alpha=1.0$ (the real, fully Schwarzschild-selected gradient) —
   cheap, and what every step tries first (matches the old shooting solver's behavior of not
   re-relaxing from scratch every step).
2. **Alpha-continuation fallback**, only if (1) fails: steps $\alpha$ through the ladder
   `config.BVP_ALPHA_CONTINUATION_STEPS = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999,
   BVP_ALPHA_MAX=1-1e-5)`, warm-starting each rung from the previous rung's converged dense
   solution. $\alpha$ blends the temperature-gradient equation between the pure adiabat
   ($\alpha=0$, no $L$-dependence at all, well-conditioned) and the real Schwarzschild-
   selected gradient ($\alpha=1$): $d\ln T/dm = (1-\alpha)\nabla_\text{ad}\,d\ln P/dm +
   \alpha\,(dT/dm)_\text{real}/T$. The literal endpoint $\alpha=1.0$ is deliberately never
   used in the continuation ladder itself (only in the direct attempt) — a tiny adiabatic
   admixture at `BVP_ALPHA_MAX` acts as a regularizer for a marginal instability confirmed in
   the pure unblended system. **This is exactly the $\alpha=0\to0.5$ jump the 2026-08-13
   diagnostic found failing via a genuinely singular Jacobian for Phase 1's ~1620K wall.**
   The blend is NaN-safe (2026-08-12 fix): the real gradient's opacity-dependent computation
   happens unconditionally regardless of $\alpha$, so a non-finite value on one extreme trial
   point falls back per-point to the adiabatic gradient rather than corrupting the whole
   blended result via IEEE's $0\times\text{NaN}=\text{NaN}$.
3. **Step-retry**, one level up in `time_stepper.run` (not inside `bvp_solver` at all): if
   layer (2) still raises `RuntimeError`, the whole step is retried at
   `dt *= STEP_RETRY_SHRINK_FACTOR=0.5`, up to `STEP_RETRY_MAX_ATTEMPTS=6` times (so up to
   64× smaller than the originally-proposed $dt$) before giving up and letting the exception
   propagate — this is what actually crashes a run end-to-end when it fails.

### 0.12 Outer time loop — `time_stepper.run`

Fixed seed $dt$ for step 1 (`DT_SEED`, 10 yr for Phase 1); every step after, if
`config.USE_ADAPTIVE_DT=True` (always, currently), `select_adaptive_dt` proposes
$dt_\text{raw}=\text{ADAPTIVE\_DT\_SAFETY\_FACTOR}(0.15)\times\min(\min_i T_i/|dT_i/dt|,
\min_i P_i/|dP_i/dt|)$ (both $T$ and $P$ timescales, deliberately excluding $L$ — $L=0$
exactly at the center by construction, a structural $0/0$), capped to grow by at most
`ADAPTIVE_DT_GROWTH_FACTOR=1.3`× the just-used $dt$ (shrinking is never restricted), then
clamped to `[ADAPTIVE_DT_MIN, ADAPTIVE_DT_MAX]` (Phase 1: `[10 yr, 1e4 yr]`, both runtime
overrides — the persisted defaults are Phase 3-scale, `[100 yr, 1e8 yr]`).

Halts on whichever of three conditions triggers first: $r_\text{surface}\le$`R_HALT`
($1\,R_\text{Jup}$), $t\ge$`T_MAX_S` (Phase 1: $10^6$ yr, a diagnostic ceiling), or
$T_\text{center}\ge$`PHASE1_T_CENTER_HALT` ($1900$ K — a deliberate ~100 K margin below where
H₂ dissociation would soften $\Gamma_1<4/3$ and trigger the out-of-scope Stage 2 dynamical
collapse). Every snapshot (`snapshot_interval`-th step, plus the halting step always) is
saved to disk immediately (`output.save_snapshot`), so a long run's progress survives an
interruption or crash.

### 0.13 All active smoothing/regularization mechanisms, at a glance

| Mechanism | Function | Parameter | Current value | Purpose |
|---|---|---|---|---|
| P/T soft clamp | `_safe_exp_state` | `BVP_SOFT_CLAMP_WIDTH` | 0.1 (log-units); bounds $\ln P\in[-100,100]$, $\ln T\in[0,100]$ | Survive wild Newton trial excursions without a zero-derivative wall |
| Opacity regime blend | `opacity.bell_lin_opacity_smooth` | `OPACITY_TRANSITION_SMOOTH_WIDTH_DEX` | 0.005 dex | Remove $d\kappa/dT$ discontinuity at Bell & Lin regime boundaries |
| Radiative-$L$ floor | `gradients.grad_radiative` | `GRAD_RAD_L_FLOOR_EPSILON` | $10^{-9}\times L_\text{KH,scale}$ | Smooth $L\ge0$ floor, prevents T-inversion runaway near photosphere |
| **Schwarzschild switch (relax)** | `gradients.effective_gradient` | `GRAD_EFF_SWITCH_EPSILON` | $10^{-4}$ (dimensionless $\nabla$ units) | Smooths $\min(\nabla_\text{rad},\nabla_\text{ad})$ — used by `relax_initial_state` only |
| **Schwarzschild switch (real dt)** | same | `GRAD_EFF_SWITCH_EPSILON_TIMESTEP` | **2.0** | SAME switch, wider width — used by every `solve_timestep` call; tuned against Phase 3's near-unity superadiabaticity, **not yet re-validated for Phase 1's 150–355 range** (2026-08-13 finding) |
| $\alpha$-continuation | `implicit_rhs_vectorized`/`_jacobian` | `BVP_ALPHA_CONTINUATION_STEPS` | (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1-1e-5) | Homotopy from pure adiabat to real Schwarzschild gradient |
| H↔H₂ recombination (mu/gamma/latent heat) | `eos.mean_molecular_weight` etc. | `USE_H2_RECOMBINATION_PHYSICS` | **False for Phase 1**, True for Phase 3 | Molecular↔atomic composition transition — OFF means $\mu,\gamma$ are hard constants everywhere for Phase 1 |
| Step retry | `time_stepper.run` | `STEP_RETRY_MAX_ATTEMPTS` / `STEP_RETRY_SHRINK_FACTOR` | 6 / 0.5 | Automatic shrink-and-retry on a failed step |
| dt growth cap | `select_adaptive_dt` | `ADAPTIVE_DT_GROWTH_FACTOR` | 1.3 | Caps how fast the adaptive step can grow, asymmetric (no shrink limit) |

### 0.14 What is genuinely NOT active for the current Phase 1 investigation

Worth stating explicitly, since Phase 3's code paths remain in the same file and are easy to
mistake for live: electron-degeneracy pressure (present in `eos.density` but numerically
negligible at Phase 1's densities), H↔H₂ recombination/dissociation physics (`molecular_
fraction`, `latent_heat_capacity`, `mean_molecular_weight_inv_derivative` — all gated off,
returning flat constants/zeros), the degenerate Lane-Emden seed (`_adiabatic_center_guess`,
Phase 1 uses the ideal-gas variant), and stage 2 of `relax_initial_state` (never runs when
the flag is off). Every one of these is a `config.py` runtime override made inside the run
script, not a permanent file-level change (hermetic isolation, confirmed by direct `git diff`
— see §5's 2026-08-12 entry).

---

## 1. Current Status

**★★★★★★ 2026-08-12 — Phase 3 PAUSED (PI directive); pivoted to Phase 1 (First Hydrostatic
Core) - a real, physically clean contraction achieved, ~73% of the T_center target, honest
open wall remaining.** Phase 3: brief note only, per explicit instruction to keep it short -
the resumed run's step 2 (original step 5) failed one step past the 2026-08-11 fixes, raw
pre-clamp `lnP`/`lnT` escaping to `~3.3e15`/`~-7e14`, same "max mesh nodes exceeded" signature,
a NOT YET diagnosed mechanism (full detail in §5's 2026-08-11 entry's closing note). Not
chased further; Phase 3 stops here until revisited.

**Phase 1**: a genuine architectural gap was found and fixed (the Lane-Emden seed used to
bracket `solve_static_structure`'s root-find was hard-coded to the compact/degenerate branch,
independent of composition - silently gave the WRONG structure, not a crash), plus a real,
shared, dimensionally-wrong `brentq` tolerance bug (fixed for both phases). Calibrated
`T_CENTER_INITIAL=645K` for the requested R~500 R_Jup. The resulting run produced 33 real,
physically clean, monotonic snapshots: **r_surface 500.83 -> 238.75 R_Jup, T_center 654.99 ->
1560.89 K** (~73% of the way to the `PHASE1_T_CENTER_HALT=1900K` target) before hitting a
persistent wall that did NOT resolve via either of the two well-motivated levers tried (`dt`
shrinking down to 28 days; `GRAD_EFF_SWITCH_EPSILON_TIMESTEP` widened to 5-10) - genuinely
different from the several smaller, transient pockets recovered along the way (which motivated
a real, general addition: `time_stepper.run` now auto-retries a failed step with a shrinking
`dt`, standard step-rejection practice). Hermetic isolation from Phase 3 directly verified
(`config.py`'s own file-level defaults confirmed byte-for-byte unchanged), not just designed
for. Full account, all numbers, and the honest state of the remaining wall: §5's 2026-08-12
entry.

---

**★★★★★★★ 2026-08-11 — Solver architecture course correction: soft-clamp replaces the hard
P/T clamp, its (and the smoothed opacity's) analytic Jacobian fixed, and the ACTUAL Phase 3
step-4 failure that triggered this whole investigation now converges.** Full trail: §5's
2026-08-11 "Soft-clamp course correction" entry. Prompted by a PI review: the day's sequence of
reactive fixes (a hard `np.clip` P/T clamp, then a tiered clamp-on/off + analytic/numerical-
Jacobian workaround around it) was diagnosed as fighting the solver rather than doing physics.
Root cause, confirmed by direct code inspection: `_safe_exp_state`'s clamp had EXACTLY ZERO
derivative once saturated, so `implicit_rhs_jacobian`/`make_bc_jacobian_scaled` (which
multiplied by the bare `P`/`T` value as a stand-in for `d(exp(lnX))/d(lnX)`) went silently
WRONG in that region rather than just losing sensitivity - directly explaining a step-4
mesh explosion where the center's trial state collapsed to `T=P=0`, `r~4.2e40 cm`. Replaced
with a smooth softplus-based saturation (`bvp_solver._soft_clamp`, `config.
BVP_SOFT_CLAMP_WIDTH`) whose derivative is threaded through the Jacobian everywhere; the SAME
class of bug was independently found and fixed in the smoothed Bell & Lin opacity's own
derivative. Both verified via finite differences (new Checks 40/40b/41, extended Check 37) -
**full regression: 42/44 pass, the only 2 failures are the same pre-existing Checks 17/23
this project has tracked since 2026-08-08, unrelated to any of this work.** Re-attempting the
exact reconstructed step-4 (state, dt) now converges cleanly via the real production
`solve_timestep` API (2.8s, 9094 nodes). One further, non-obvious finding along the way:
`GRAD_EFF_SWITCH_EPSILON_TIMESTEP` needed WIDENING (0.5->2.0), not narrowing as hoped - direct
evidence the marginal-convection band is real physics, strengthening the case for PLAN.md's
already-scoped Sub-task 8c (mixing-length theory) rather than continuing to chase this value.

---

**★★★★★★★ 2026-08-10 (evening) — Sub-task 8b IMPLEMENTED: H<->H2 recombination physics live
in the production solver, Delta r_surface=-5.05% on a real solve_timestep.** Full trail: §5's
2026-08-10 "Sub-task 8b implementation" entry. Following the roadmap pivot toward modeling
Phase 1 (First Core collapse) - where the SAME chi(T)/mu(T)/gamma_eff(T) mechanism is expected
to formally trigger Phase 2 once Gamma_1 softens below 4/3 - the physics justified by Check 38
was implemented for real, not just in a diagnostic proxy. Planning surfaced a critical scope
correction: `odes.py`'s `stellar_odes` is NOT the live solver path - `bvp_solver.py` has its
own separate RHS and **analytic Jacobian** (`implicit_rhs_jacobian`, `make_bc_jacobian_scaled`)
duplicating this physics inline, and the Jacobian needed real new derivative terms (not just
`config.MU`->`mu(T)` substitution) since `T` now couples to the system through channels that
did not exist under constant-`gamma`/`mu` physics: `grad_ad(T)` softening (row 3) and
`c_p_eff(T)`'s own T-derivative (row 2). Staged per plan: RHS-only with scipy's numerical
Jacobian first (converged cleanly, `Delta r_surface=-5.09%`, confirming the physics before
trusting any hand-derived math), then the analytic Jacobian, which **Check 37 caught a real
bug in on the first attempt** (a dropped `/M_H` factor in three places, `eos.thermodynamic_
delta` and two `bvp_solver.py` Jacobian helpers) - fixed, Check 37 then passed to 6.5e-7,
including explicit new coverage of the 2000-3000K transition window itself. Full regression
suite run end-to-end: **zero new failures** - the only two failing checks are the exact same
two pre-existing, already-documented failures from 2026-08-08 (confirmed by matching error
magnitudes), unrelated to this work. Final production result, corrected analytic Jacobian,
9 iterations, 1.55s: **r_surface 4.5966 -> 4.3642 R_Jup (-5.05%) for one real 1e8 yr timestep**
from the 10 Gyr cached state - a substantially larger effect than Check 38's static,
energy-equation-decoupled proxy predicted (-3.1%), as expected once the full 4-ODE system
(T, L, and r all responding self-consistently, not just rho at fixed P,T) is actually solved.

**★★★★★★ 2026-08-10 (later same day) — Outer-envelope H/H2 recombination sensitivity check: SIGNIFICANT (-3.1% r_surface), justifies real implementation.** Full trail: §5's
2026-08-10 "recombination sensitivity" entry. Direct follow-up to the 10 Gyr extension below:
that run's own physics review corrected the user's original diagnosis (`config.MU` is
already the ATOMIC value, not molecular - fixed 2026-08-07) and identified the real gap in
the OPPOSITE region: the cool outer envelope (T<~3000 K, 8.5-26% of the mass by 10 Gyr)
should have H recombining back into H2 (mu rising toward ~2.34) as it cools, but `odes.py`
uses the constant atomic `config.MU=1.278` everywhere. A cheap, sterile sensitivity check
(`validation.check_outer_envelope_recombination_sensitivity`, new Check 38) - a logistic
`mu(T)` proxy applied to the cached 10 Gyr snapshot's ALREADY-CONVERGED $P(m)$, $T(m)$, with
$r(m)$ independently re-integrated via the same continuity equation and compared against a
control run (isolating the perturbation from the re-integration method's own ~0.6% error
floor) - found **$\Delta r_\text{surface}=-0.144\,R_\text{Jup}$ (-3.1%)**, roughly 5x the
control floor and 3x the pre-agreed 1% "worth implementing" threshold. Direction makes sense
locally: raising $\mu$ at fixed $(P,T)$ raises the implied $\rho$ ($\rho=P\mu m_H/k_BT$),
which locally compresses those outer mass shells ($dr/dm=1/(4\pi r^2\rho)$ shrinks) - a
different (and here, opposite-signed) effect from the earlier global virial argument, since
this test holds the full $P(m)$/$T(m)$ profile fixed rather than re-equilibrating the whole
star. Caveat: a meaningful share of the -3.1% comes from the single least-resolved grid
segment at the very photosphere, so the magnitude is an order-of-magnitude estimate, not a
precise prediction - but the verdict (real, not negligible) looks robust. **Next: plan the
actual implementation** - the properly-scoped physics (two-state $\chi(T)$ threaded into
`eos.py`'s four `mu`/`gamma`-dependent functions, plus the matching latent-heat term in the
energy equation) that this check was built to justify, not yet started.

**★★★★★ 2026-08-10 — 10 Gyr diagnostic extension: contraction is real and continuing, but
decelerating — not yet enough to explain the slow pace without Sub-tasks 8a-8c.** Full trail:
§5's 2026-08-10 entry. Motivated directly by the 4.5 Gyr result below (only
$r_\text{surface}\approx4.83\,R_\text{Jup}$ reached, from a ~5.1 $R_\text{Jup}$ start): is
that genuinely slow Kelvin-Helmholtz contraction, or an artificial floor from missing
physics? `config.AGE_SOLAR_SYSTEM_S` was renamed to `config.T_MAX_S` and raised to 10 Gyr
(explicitly a diagnostic time budget, not a claim about the real planet's age), and
`extended_run_10gyr.py` **resumed the run directly from the 4.5 Gyr run's last saved
snapshot** (`snapshot_00077.npz`, `t=4.5215e9` yr) rather than re-solving from scratch — the
first real production use of Sub-task 10's snapshot-resumability design, not just a recovery
after a crash. 55 further steps converged cleanly (no continuation fallback, `dt` pinned the
entire time at the `ADAPTIVE_DT_MAX=1e8` yr defensive ceiling) to `t=1.0022e10` yr, halting
correctly on the new `T_MAX_S` condition. **Result: $r_\text{surface}$ contracts smoothly
and monotonically the entire way, with no repeat of the earlier bump**, reaching
$4.597\,R_\text{Jup}$ — genuine continued contraction, not a plateau. But the *rate* is
clearly slowing: $|dr/dt|$ fell from $\approx0.060\,R_\text{Jup}$/Gyr just after 4.5 Gyr to
$\approx0.041\,R_\text{Jup}$/Gyr by 9.5 Gyr (~32% slower over 5 Gyr), tracking
$T_\text{center}$ and $L_\text{surface}$ both roughly halving over the same window. This
smooth, gradual deceleration (not a sharp asymptote) reads as a genuinely lengthening
$\tau_\text{KH}\sim GM^2/(RL)$ as $L$ drops, rather than a hard numerical/physics floor — but
at this decelerating pace, reaching `R_HALT` (3.6 $R_\text{Jup}$ still to go) would take
enormously longer than 10 Gyr, which is itself the strongest concrete argument yet for doing
Sub-tasks 8a-8c (Saha, molecular→atomic $\mu(T)$, and especially MLT convective efficiency)
before trusting the absolute timescale, even though the qualitative trend looks physically
sound on its own. See `diagnostic_plots_10gyr/evolution_curves_FULL_0_to_10gyr.png` for the
complete 0-10 Gyr trajectory (both runs' snapshots combined).

**★★★★ 2026-08-09 (overnight) — First full run to `config.AGE_SOLAR_SYSTEM_S` completed;
the earlier `r_surface` anomaly turns out to be a bounded, self-correcting bump, not a
runaway divergence.** Full trail: §5's 2026-08-09 (overnight) entry. An unattended extended
run (`overnight_run.py`, up to 100 steps, unchanged tuning from the sanity check) converged
cleanly for **all 77 steps it took to reach `t=4.5215e9` yr** (the `AGE_SOLAR_SYSTEM_S` halt
condition, essentially exactly on target) - no crash, no continuation fallback needed at any
step, final structure profile smooth and well-resolved end to end. Plot generation crashed
immediately afterward on a real but trivial bug (`output.py`/`diagnostics.py`'s plot
functions never created their own output directory when it didn't already exist) - **all 78
snapshots survived on disk regardless**; the bug is fixed (both modules now create their
output directory defensively) and every plot was regenerated from the saved data with no
re-solve needed. Reviewing the FULL trajectory (not just the first 15 steps seen before)
shows $r_\text{surface}$ rises smoothly to a peak of ~5.2 $R_\text{Jup}$ around
$t\sim3$-$5\times10^8$ yr (coincident with a smooth $L_\text{surface}$ peak,
$\sim4.2\times10^{-10}L_\odot$), then the contraction trend **resumes and continues past the
starting value**, ending at $r_\text{surface}\approx4.83\,R_\text{Jup}$,
$T_\text{center}\approx10735$K by 4.5 Gyr — genuine net contraction over the full run, not a
diverging or oscillating trajectory. Whether the bump itself is genuine thermal-relaxation
physics or a large-`dt` resolution artifact is still open (see §5's entry) — but "does the
overall run stay bounded and physically sensible" is now answered: yes. Also scientifically
notable, independent of that open question: after 4.5 Gyr, the model is still at
~4.8$\,R_\text{Jup}$, far from `R_HALT=1`$\,R_\text{Jup}$ — worth a closer look at whether
that's expected given the input physics (no MLT, no Saha, etc.) or itself informative.

**★★★ 2026-08-09 (later same day) — Sub-task 10 (`output.py`) mechanically DONE; one open
accuracy finding surfaced before the real full run.** Full trail: §5's 2026-08-09 "Sub-task
10" entry. `.npz` snapshot I/O, evolution-curve/profile/opacity-regime plots, live per-step
logging (flushed), a corrupted-state guard, and a dual `R_HALT`/`AGE_SOLAR_SYSTEM_S` (4.5
Gyr) stopping condition are all implemented and verified working end-to-end on a 15-step
sanity run. **`ADAPTIVE_DT_MAX` was also raised** (5e4 yr → 1e8 yr, now a purely defensive
backstop) after the user correctly identified that further fixed-`dt` margin sweeps at
larger absolute values would be testing the wrong thing (a `dt` appropriate for a late,
slow evolutionary phase, applied to the current early, fast one) - trusting the growth-cap
mechanism directly instead. **That same sanity run surfaced a genuine, precisely-localized
open question**: once `dt` grew past the previously-validated 5e4 yr (into
~$6\times10^4$-$4\times10^5$ yr), $r_\text{surface}$ stopped decreasing and turned around,
diverging from an otherwise-identical lower-`dt` run exactly at the first step past the old
ceiling - a small (~0.1%) but clearly `dt`-correlated effect, with the solver itself showing
no convergence distress at all (accuracy, not stability). Not chased further without
explicit direction, per the user's own request to review before the real run - the real
full run to `R_HALT`/4.5 Gyr is deliberately not yet launched.

**★★★ 2026-08-09 — Sub-task 9 (adaptive time-stepping) implemented and validated.** Full
trail: §5's 2026-08-09 entry. Short version: `time_stepper.select_adaptive_dt` (dual
$T$/$P$ thermal-timescale limiter, $L$ deliberately excluded, asymmetric growth cap) - three
design refinements added after explicit review against standard stellar-evolution-code
practice, all justified and adopted, not rubber-stamped. Sterile-tested (5/5 synthetic
cases), margin-swept against the `GRAD_EFF_SWITCH_EPSILON_TIMESTEP` fix at 2e4/5e4 yr
*before* trusting it live, then validated over a real 15-step run: the growth cap was the
binding constraint every step (not the raw formula) until `ADAPTIVE_DT_MAX=5e4` yr was
reached, confirming the design intuition directly. Reached **5.76e5 yr of simulated time in
15 steps vs. 1e5 yr in 10 fixed-`dt` steps - a ~5.8x efficiency gain**. `ADAPTIVE_DT_MAX` is
explicitly a *temporary* validation ceiling, not production-ready - reaching `R_HALT` needs
`dt` several orders of magnitude larger; a staged, re-validated escalation plan (not a blind
increase) is recorded in §5's entry for the next session.

**★★★ 2026-08-08 (later same day) — Sub-task 8's dry-run exit criterion MET: a genuine
10-step time evolution, monotonic contraction, negative-$L_\text{surface}$ question
resolved.** Supersedes the "open, not yet resolved" step-2 item in the ★★ entry directly
below. Full trail: §5's "Multi-step time evolution achieved" entry (2026-08-08, above the
promotion entry in the log). Short version: the step-2 mesh explosion was genuine marginal
convection (verified via a superadiabaticity histogram, not assumed) — a real timestep can
collapse the outer envelope's $L$ ~70x, moving $\nabla_\text{rad}$ from deeply
super-adiabatic to sitting within a few percent of $\nabla_\text{ad}$ across an extended
band. Two candidate interpolation fixes were tried and honestly reverted (neither was
decisive). Resolved via a **context-dependent** Schwarzschild-switch smoothing width — the
user's proposed "wide smoothing" compromise, explicitly not MLT — narrow
(`config.GRAD_EFF_SWITCH_EPSILON=1e-4`, unchanged) for `relax_initial_state`, wide
(`config.GRAD_EFF_SWITCH_EPSILON_TIMESTEP=0.5`, tuned with margin, revised once from an
initial 0.1 that failed at step 6) for `solve_timestep` only. A full 10-step dry run now
shows $T_\text{center}$/$r_\text{surface}$ decreasing smoothly and monotonically
(contraction) and $L_\text{surface}$ settling to a small, steady, positive value with
shrinking increments — resolving the negative-$L$ question as a relaxation-transient, not a
bug. Genuine MLT remains formally scheduled (new `PLAN.md` Sub-task 8c), explicitly deferred
past the one-week deadline. **Honest limit**: only validated for 10 steps; not proof the
chosen $\varepsilon$ holds for an arbitrarily long run.

**★★ 2026-08-08 — `bvp_experiment.py`'s `solve_bvp` machinery PROMOTED to production;
shooting retired for t>0; first genuine real-time step achieved.** `PLAN_BVP.md`'s roadmap
is complete and merged into `PLAN.md` (§4.2, Sub-task 5 update). Concretely:
- `bvp_solver.py` rewritten: `solve_static_structure` (t=0, `brentq` shooting) unchanged;
  `relax_initial_state`/`solve_timestep` (t>0) now solve via `scipy.integrate.solve_bvp`
  collocation (scaled state, analytic Jacobians, `ALPHA_MAX` continuation) instead of the old
  `fsolve`/`root`-based shooting - same public function names/signatures, so no other module
  needed to change its call sites. The retired shooting code is archived verbatim in the new
  `bvp_solver_shooting_archive.py`, not deleted.
- Regression-verified: promoted `relax_initial_state` reproduces `bvp_experiment.py`'s own
  proven 11500K and 12000K results to <1% (mostly ppm-level).
- **New finding while wiring `solve_timestep` for a REAL `dt` (never exercised before,
  `bvp_experiment.py`'s milestones only tested the pseudo-relaxation step): a real, secondary
  mesh-construction issue** - the initial guess linearly interpolated `P`, `T` (not `ln P`,
  `ln T`) from an already-converged state's own output grid, badly misrepresenting the
  ~3-decade pressure drop in the final ~0.001% of mass near the photosphere. ~~Root-caused
  via direct inspection (not assumed) and fixed (log-interpolation + a deeper, warm-start-
  only outer mesh refinement, `config.BVP_MESH_OUTER_REFINEMENT`)~~ **CORRECTED, same day,
  see the ★★★ entry above**: this fix was tried, found NOT to be the decisive lever for
  step 2 (and to make `relax_initial_state` itself measurably harder), and reverted - the
  dominant cause was genuine marginal convection, resolved differently. Kept here as an
  honest record of a plausible-but-wrong first hypothesis, not deleted.
- **First genuine real-`dt` step (not a pseudo-step) achieved and physically informative**:
  `solve_timestep` converged directly (`status=0`, residual 9.73e-7) for one real step at
  `dt=1e4` yr from the relaxed T=11500K state. Over that step, `T_center` and `r_surface`
  both **decreased** (contraction, matching Sub-task 8's exit criterion), `T_surface` moved
  from 49.36K to 49.99K (essentially onto `T_NEB=50K`), and `L_surface`'s magnitude **dropped
  ~70x** (-2.85e23 → -4.03e21 erg/s) while staying negative. This is the direct evidence the
  negative-`L_surface` question needed: consistent with it being a `DT_RELAX`-pseudo-timestep
  artifact that decays fast under real time evolution, not a persistent bug - though only one
  step deep so far, not yet a settled conclusion.
- **RESOLVED, same day - see the ★★★ entry above**: a second real step (step 1 → step 2)
  initially did NOT converge within a generously raised node budget (200,000) even with a
  much finer `alpha` continuation ladder - node count grew super-linearly
  (43569→79660→129932 across `alpha`=0.80→0.85→0.90) rather than plateauing, confirmed to be
  genuine marginal convection, not a resolution shortfall. Fixed via a context-dependent
  Schwarzschild-switch smoothing width; a full 10-step dry run now meets Sub-task 8's exit
  criterion cleanly.

**🔴 2026-08-06 — ARCHITECTURAL PIVOT: shooting is abandoned. `scipy.integrate.solve_bvp`
(global relaxation/collocation) is now the project's target solver.** Everything below this
note describing `relax_initial_state`/the shooting $\alpha$-homotopy is historical —
accurate as a record of what was tried and why it was eventually abandoned, not current
guidance. Reason: three independent non-differentiable "kinks" ($L\ge0$ floor, Schwarzschild
switch, and a suspected-but-unconfirmed Bell & Lin opacity switch) were found in sequence,
each appearing immediately after the previous one was fixed, all clustering in the same
narrow region of the homotopy path and the same physical region of the star
(near-photosphere) — the diagnostic signature of a method (shooting) that cannot contain
local non-smoothness across a single long integration. The active roadmap is `PLAN_BVP.md`;
the active solver code is `bvp_experiment.py`.

**★ 2026-08-07 — `solve_bvp` CONVERGES (`status=0`, machine-precision residuals) at
`T_CENTER_INITIAL=11500`K — the pivot's first genuine success.** State-vector
nondimensionalization ($\hat r$, $\operatorname{arcsinh}$-scaled $\hat L$) + analytic
Jacobians + corrected EOS thermodynamics ($\gamma\to5/3$, $\mu\to1.278$, a previously
hardcoded energy-equation $\delta$ coefficient fixed) + a continuation endpoint just short
of literal $\alpha=1.0$ (`ALPHA_MAX=1-10^{-5}$`, an empirically-required regularization, not
yet fully explained) together produced the first fully-converged solution this
architecture has ever reached. `T_CENTER_INITIAL` moved from 13000K to 11500K as a *direct,
measured* consequence of the EOS correction (13000K became genuinely infeasible under the
corrected physics - PLAN_BVP.md §3.6 has the full derivation). Two real bugs were caught by
the mandatory FD-Jacobian cross-check before this was trusted. Open items remain (a
slightly negative `L_surface`; the un-derived $\alpha=1.0$ instability mechanism; only one
temperature tested so far) - see §5's 2026-08-07 Milestone 6 entry and `PLAN_BVP.md` §3.6
for the complete, unabridged trail. The $T_\text{CENTER\_INITIAL}$ decision and the
physical/formation-scenario reasoning below are otherwise unaffected by this pivot — it
remains, in spirit, a numerical rather than physical change, with the EOS correction being
the one deliberate exception (a genuine physics fix, not a numerics experiment).

**★ 2026-08-07 — Second-temperature confirmation: T=12000K ALSO converges cleanly
(`status=0`, residual $8.88\times10^{-16}$ — tighter than 11500K), closing the "only one
temperature tested" open item above for the atomic/post-dissociation regime.** A T=2000K
attempt was tried first and rejected as an invalid test point, not a fix failure: that
temperature sits outside the composition regime `config.MU=1.278`/`GAMMA=5/3` (atomic,
post-dissociation) actually describes, since real H at 2000K is molecular and `MU`/`GAMMA`
are global, not temperature-dependent constants. The negative `L_surface` finding is now
**reproduced** at 12000K (same sign, same order of magnitude, $T_\text{surf}$ again
landing $<1.3\%$ below $T_\text{NEB}=50$K) — confirmed a real, reproducible feature of this
converged solution family rather than one-run noise, though still not explained; see
`PLAN_BVP.md` §3.6.4 for the full trail. Recommended next diagnostic for the negative-`L`
question: `time_stepper`'s first real-$dt$ step, not further isolated investigation.

**⚠ `config.T_CENTER_INITIAL` is now 11500K, not 13000K** (changed 2026-08-07 - see the
★ note above and §5's Milestone 6 entry) - 13000K became genuinely infeasible once the
EOS's $\gamma$/$\mu$ were corrected to their physically-appropriate atomic values. The
"Geometric Target" reasoning below explains why 13000K was chosen in the first place and
remains accurate history; only the specific numeric value has since moved.

**✅ `config.T_CENTER_INITIAL=13000`K decided ("Geometric Target" approach, 2026-08-01) —
gives $R=4.15\,R_\text{Jup}$, not $\sim3$ as expected; flagged, not silently accepted.** The
formation-scenario reframing (three stages: first hydrostatic core → dynamical second
collapse → post-second-collapse hot start) correctly resolved *why*
$T_\text{CENTER\_INITIAL}=1200$K was wrong; the literature check then surfaced a real
tension between the literature-motivated temperature ($2\times10^4$-$5\times10^4$K, anchored
to present-day Jupiter's own modeled central temperature) and this codebase's actual
$R(T_\text{center})$ behavior (no ionization physics — $R$ already exceeds $4\,R_\text{Jup}$
by $T\sim1.7\times10^4$K). **Decision**: prioritize the geometric target, isolate Sub-task
8's time-stepper work from the EOS gap, close that gap later in a new mandatory Sub-task 8a
(PLAN.md) before Stage 1 modeling. Picked $T_\text{CENTER\_INITIAL}=13000$K — but the
precise converged value is $R=4.1544\,R_\text{Jup}$ (`solve_static_structure()`, confirmed
directly), not the $\sim3\,R_\text{Jup}$ expected when this number was chosen (nearby
marching points don't support $\sim3$ either — see §5's dated entry). Outside the stated
2-4 range, not by a lot, but not silently rounded down to "close enough."

**⏸ Sub-task 8 dry run still BLOCKED. Safety net PROVEN WORKING. Two fix attempts tried and
ruled out (self-scaling normalization; switching `fsolve`→Levenberg-Marquardt) — the
evidence now points away from "solver/scaling choice" and toward "the true root is
genuinely far from the $\alpha=0$ solution at $\Delta\alpha=0.1$." Read the three dated
entries below (oldest first) before touching `relax_initial_state`/`solve_timestep` again.**
Switching the inner root-find from `scipy.optimize.fsolve` (MINPACK `hybrd`, step-size-only
convergence) to `scipy.optimize.root(method="lm")` (genuine Levenberg-Marquardt, with a real
`ftol` residual-progress criterion) was well-motivated but **did not fix it**: LM converges
to `residual=[-7.657e-05, 8.016e-03]` at $\alpha=0.1$ — nearly identical to `hybrd`'s
`[-7.208e-05, 8.204e-03]` from the same seed. Two independently-implemented algorithms
landing on almost the same non-zero residual is strong evidence this is a genuine local
minimum of the residual near $\alpha=0.1$, not an algorithm-specific convergence-reporting
artifact — meaning the $\Delta\alpha=0.1$ step is likely too large for *any* local
Newton/Gauss-Newton-type method to bridge from the $\alpha=0$ seed, not a solver-choice
problem. Points toward finer/adaptive $\alpha$-stepping as the more promising next avenue.

Root cause (unchanged from before): `fsolve`'s `ier=1` reflects its *step-size* tolerance
(`xtol`), not the residual/function value — at $T_\text{CENTER\_INITIAL}=13000$K, this lets
spurious "convergence" through. **Now caught immediately and correctly**: with the new
residual-magnitude safety-net check in place, the $\alpha$-ramp raises
`RuntimeError` at $\alpha=0.1$ (residual `[-7.2e-5, 8.2e-3]`, exceeding
`config.RESIDUAL_TOL`) — earlier than first estimated ($\alpha=0.0$ is the *only* genuinely
well-converged step here, not through $\alpha\approx0.2$ as initially thought. $\alpha=0.1$
was already spurious, just not yet visibly "frozen."

**The scaling-fix attempt (self-normalizing the thermal residual by the trial's own
photospheric $L$ instead of a fixed estimate) was tried, caused a regression, and has been
reverted.** It broke $\alpha=0.0$, which previously converged essentially perfectly.
Direct calculation explains why: the fixed KH-timescale $L$ estimate
($\approx2.6\times10^{29}$erg/s) is actually **~78x *larger*** than the genuine converged
$L$ at $T=13000$K ($\approx3.4\times10^{27}$erg/s) — so switching the denominator to the
trial's own $L$ *shrinks* it, growing the normalized residual and its sensitivity, backwards
from the intended fix. It also introduced a new risk: dividing by a trial's own $L$ during
`fsolve`'s internal probing can approach zero, which is far more dangerous than a fixed,
always-reasonable constant. **Deeper conclusion**: the raw (un-normalized) thermal-residual
sensitivity to $T_\text{center}$ is enormous in this regime
($\sim10^{32}$erg/s per unit $\ln T_\text{center}$) — no single-constant rescaling, fixed or
self-adapting, changes that; the conditioning problem is more fundamental than a
normalization choice. See the dated entry for what's proposed next (not yet
implemented — pending direction).

**Sub-task 8's own code (`time_stepper.run()`) is implemented and NOT in question** — the
warm-start loop mechanism works exactly as designed; the failure is two layers down, in
`relax_initial_state`'s convergence *verification*, not its physics. Composite outer-mass
grid, `T_DISSOCIATION_LIMIT`→`R_HALT` config swap, `R_JUPITER_CM`, and the hardcoded-literal
cleanup are all done (§5's "Step 3 execution" entry, 2026-08-01).

**✅ Sub-task 5 is DONE and verified end-to-end (2026-08-01 session).** Three blockers were
found and fixed in sequence this session, each traced to its root cause before being patched
(no ad hoc clamp-tuning):

1. **Clamp-cascade in `relax_initial_state`'s $\alpha$-ramp** (§5, 2026-07-27 entry) — fixed
   by switching `bvp_solver.py`'s inner `solve_ivp` state to logarithmic $P$, $T$ ("Path 1"),
   removing the `1e-300` floor clamps entirely so positivity holds by construction.
2. **A $\nabla_\text{rad}$ blow-up near the photosphere** once $\alpha>0$ blended in the real
   Schwarzschild-selected gradient — traced to `gradients.grad_radiative`'s
   $\nabla_\text{rad}\propto\kappa LP/(mT^4)$ becoming pathologically sensitive as $T^4\to0$
   near the photosphere, so a small $L$-sign deviation flips $\nabla_\text{rad}$'s sign and
   forces an unphysical temperature inversion. Fixed by flooring $L$ at its actual point of
   use in `grad_radiative` (not downstream in `effective_gradient`) — confirmed the floor
   engages only transiently (for $\alpha\le0.7$) and is never active at the converged
   $\alpha=1$ solution, i.e. it has zero footprint on the final physics.
3. **A catastrophic-cancellation collapse in `solve_timestep`** when seeded from an
   already-self-consistent `state_prev` (the relaxed state) — the trial and `state_prev`
   coincide to near machine precision right at the seed, so
   $dT/dt=(T-T_\text{prev})/dt$ amplifies floating-point noise into a spurious blow-up near
   $m=0$. Fixed with the same tiny (`1e-6` relative) seed nudge `relax_initial_state` already
   used for the identical reason.

**Verified end-to-end**: `solve_static_structure()` → `relax_initial_state()` (all 11
$\alpha$ pseudo-steps converge, residuals $\lesssim10^{-5}$) → `solve_timestep()` (converges
from the relaxed state with residuals $[2.4\times10^{-10}, 3.1\times10^{-8}]$, giving a
physically sensible first step: $T_\text{center}$ cools $1251.9\to1215.3\,$K,
$r_\text{surface}\approx3.17\,R_\text{Jup}$, mass matched to $10^{-8}$). See §5's 2026-08-01
entries for the full numerical trace of all three fixes.

**Also this session**: a `dev_cache.py` utility (pickle save/load for a `SimulationState`)
and a new CLAUDE.md "Development Workflow" section were added, so future debugging of
downstream logic doesn't require re-running the ~15-20 minute `solve_static_structure`+
`relax_initial_state` chain from scratch. `bvp_solver.py`/`gradients.py` comments were also
cleaned of session-narrative content (kept: physical reasoning and ASSUMPTION flags; PROGRESS.md
remains the place for numerical trails and debugging history).

**Per the original Sub-task 5 instruction: stop and check in before starting Sub-task 6.**

| Sub-task | Scope | Status |
|---|---|---|
| 1 | `config.py` + `state.py` | Done |
| 2a–2e | `eos.py` (ideal-gas part) + `opacity.py` + validation | Done |
| 2f | `eos.py` — non-ideal EOS, electron degeneracy pressure | Done, validated (2026-07-27) |
| 3 | `gradients.py` (Schwarzschild criterion) | Done — `L>=0` floor and Schwarzschild `min()` switch both smoothed 2026-08-06 (were hard, now differentiable; §5) |
| 4 | `odes.py` + `boundary_conditions.py` | Done (surface conditions revised, §5); reused unmodified by `bvp_experiment.py` |
| 5 | `bvp_solver.py` ($t=0$ structure + $t>0$ solve) | **★★ PROMOTED to `solve_bvp` 2026-08-08.** `solve_static_structure` unchanged ($t=0$ seed). `relax_initial_state`/`solve_timestep` (t>0) rewritten in place onto `solve_bvp` collocation — regression-verified against `bvp_experiment.py`'s 11500K/12000K results; old shooting code archived in `bvp_solver_shooting_archive.py`. See §1/§5's 2026-08-08 entry. |
| — | `bvp_experiment.py` (SUPERSEDED, historical record) | Its proven logic (state-vector scaling, analytic Jacobians, `ALPHA_MAX` continuation) was promoted into `bvp_solver.py` 2026-08-08 — kept in place unmodified, banner-marked, not imported by anything active. |
| — | `bvp_solver_shooting_archive.py` (new, archival) | The retired $t>0$ shooting implementation, moved here verbatim 2026-08-08 — not imported by anything active. |
| 6 | `diagnostics.py` | **Done (2026-08-01)** — visual plots + virial theorem/opacity regime checks rewritten for the compact structure, all pass |
| 7 | `time_stepper.py` time derivatives | Unchanged code; now runs against the promoted `solve_bvp` solver (2026-08-08) — see row 5 |
| 8 | Outer time loop (`time_stepper.run`, `main.py`) | **★★★ Dry-run exit criterion MET (2026-08-08)**: `main.py` implemented; a full 10-step dry run converges cleanly at every step, with $T_\text{center}$/$r_\text{surface}$ decreasing monotonically (contraction) and $L_\text{surface}$ settling to a small positive value — resolved a genuine marginal-convection mesh-explosion via a context-dependent Schwarzschild-switch smoothing width (`config.GRAD_EFF_SWITCH_EPSILON_TIMESTEP`). Not yet a full run to `config.R_HALT`. See §1/§5's 2026-08-08 entries and `PLAN.md` Sub-task 8's status note. |
| 8c | MLT convection treatment | Not started — formally scheduled, explicitly deferred past the one-week deadline (`PLAN.md` Sub-task 8c) |
| 9 | Adaptive time-stepping (`time_stepper.select_adaptive_dt`) | **★★★ Done, validated over 15 real steps (2026-08-09)** — dual $T$/$P$ limiter, $L$ excluded, asymmetric growth cap; ~5.8x simulated-time efficiency gain measured. `ADAPTIVE_DT_MAX` is a temporary ceiling, not yet raised to production scale — see §5's 2026-08-09 entry |
| 10 | `output.py` | **★★★ Mechanically done (2026-08-09)** — `.npz` I/O, evolution/profile/opacity plots, all regenerate from disk alone (exit criterion met). Real full run deliberately not yet launched — a genuine, small, `dt`-correlated $r_\text{surface}$ non-monotonicity was found at `dt`>5e4 yr during the sanity check (accuracy, not stability — solver shows no distress) and flagged for review, not chased without direction. See §5's 2026-08-09 "Sub-task 10" entry. |

**Stub present but empty:** `ReadMe.txt`. (`main.py` implemented 2026-08-08, no longer a stub.)

### What we actually know, and how confident we are

This project went through a substantial premise change this session (full reasoning in §5's
change log). To avoid the previous version of this file blurring together what's solid and
what's still speculative, here is that distinction explicitly.

**Confirmed — proven analytically and/or directly reproduced in this codebase:**

- The original $t=0$ design (a diffuse, isothermal cloud in equilibrium with the ambient
  nebula) is an *exact*, unbreakable fixed point of any per-timestep scheme — frozen-source
  or genuinely implicit, any $dt$. Proven analytically (zero time-derivatives force
  $L\equiv0$, forcing $\nabla_\text{rad}\equiv0$, forcing $dT/dm\equiv0$, self-consistent
  with staying put) and confirmed numerically across $dt$ spanning six orders of magnitude
  and six different shooting starting guesses, all converging to the same
  machine-precision-identical answer.
- The net-flux radiative surface condition
  ($L=4\pi R^2\sigma_\text{SB}(T^4-T_\text{neb}^4)$, replacing a rigid $T=T_\text{neb}$
  clamp) is the physically correct fix for that specific degeneracy — it reduces to the old
  condition exactly at equilibrium but does not force $T$ back once displaced.
- Adding an explicit homologous-contraction rate as an *extra* source term inside the
  per-step energy equation (on top of the genuine implicit state difference) double-counts
  compressional heating. Proven via energy-conservation violation: it produced a state that
  was exactly frozen step-to-step yet continued to radiate a constant, non-decaying $L$ —
  there is no reservoir to draw from if nothing is changing.
- A pure ideal-gas, fully-convective (adiabatic) structure at `T_CENTER_INITIAL`=1200 K and
  `M_TOTAL`=1 Jupiter mass self-consistently settles at $R\approx300\,R_\text{Jup}$, not a
  compact few-$R_\text{Jup}$ radius. Confirmed via an independent analytic Lane-Emden
  solution (cross-checked against tabulated $n=1.5$, $n=3.0$ results before trusting it for
  this project's $n=2.5$ case), then reproduced in the actual shooting code.
- That same pure-adiabat construction is **not** a self-consistent solution of the real
  4-ODE system `solve_timestep` uses: evaluating `solve_timestep`'s residual at the
  construction's own unperturbed center values gives $\sim10^8$ (should be $\approx0$ for a
  genuinely consistent state), and this residual is essentially insensitive to $dt$
  (confirmed: a 10x smaller $dt$ barely changed it) — ruling out "just needs a smaller
  timestep" as the explanation.
- A genuinely self-consistent alternative construction (routes through the real
  `odes.stellar_odes`, with $L$ built from an assumed homologous contraction rate rather
  than bypassed) does have a solution matching $P(M_\text{total})=P_\text{neb}$ at the same
  $T_\text{center}$ — but it is *more* extended ($R\approx27{,}000\,R_\text{Jup}$), not
  compact. Confirmed as a clean, monotonic root-find (not a bracketing artifact or spurious
  second root).
- **(Confirmed in the 2026-07-27 correctness review, previously listed as an open
  hypothesis below):** that extended result is driven by a genuine transition to a
  radiative (not convective) temperature gradient at $m/M\approx0.70$, after which the
  structure becomes nearly isothermal and expands $153\times$ over the last 5 decades of
  pressure drop to reach $P_\text{neb}$ — the same mechanism as the original Bonnor-Ebert
  problem, now confined to the outer ~30% of the mass. This is a **second, independent**
  architectural gap from the missing EOS physics: the $P(M_\text{total})=P_\text{neb}$
  outer boundary condition (inherited from the old diffuse-cloud design) likely forces the
  integration well past where radiative diffusion is even a valid description of a real
  photosphere, and electron degeneracy pressure (negligible at these low densities) is not
  expected to fix it.

**Strong physical inference — well-supported by established results in the field, not yet
directly re-derived or tested inside this codebase:**

- The missing ingredient behind all three ideal-gas findings above is very likely electron
  degeneracy pressure. Real gas giants and brown dwarfs are partially electron-degenerate
  essentially from formation onward (at Jupiter's characteristic density, the electron
  Fermi temperature is order $10^5$–$10^6$ K, far above any plausible formation
  temperature) — not just late in a cooling history, which is the common but incorrect
  intuition carried over from the white-dwarf picture. This is why published gas-giant
  thermal-evolution codes (Bodenheimer & Pollack 1986; Marley et al. 2007; and the
  subsequent literature generally) universally use a non-ideal EOS, even for their
  earliest, hottest models. Classic reference for the underlying mass-radius argument:
  Zapolsky & Salpeter (1969).
- Initial GI collapse is a fast, inertia-dominated hydrodynamic process that a
  quasi-static/hydrostatic solver cannot represent — this is symmetric to why
  `T_DISSOCIATION_LIMIT` already halts the code at the far end of validity. Standard
  practice (pre-main-sequence Henyey-track modeling; gas-giant "hot start" models) is to
  hand off from an assumed post-collapse state rather than simulate the collapse itself.
- The Hayashi MMSN (used for `T_NEB`/`P_NEB`) describes a smooth, linearly-*stable*
  (Toomre $Q\gg1$) disk — a disk that actually fragments via GI must locally be denser/more
  massive than that. Using MMSN conditions as the confinement for a clump that supposedly
  *formed* by fragmenting that same disk is not self-consistent, and is the likely reason
  the original diffuse-clump premise gave a deeply Bonnor-Ebert-subcritical (stable, inert)
  result rather than one poised to contract.

- **(Confirmed 2026-07-27, Sub-task 2f implemented):** adding the non-relativistic
  electron-degeneracy pressure term resolves the interior-compactness problem as predicted.
  The analytic Zapolsky-Salpeter-style estimate (pure T=0 degenerate limit,
  $R\approx3.11\,R_\text{Jup}$) was confirmed almost exactly by the actual shooting code
  ($R\approx3.17\,R_\text{Jup}$ once a direction-search bug in the bracket algorithm — see
  below — was fixed).
- **(Confirmed 2026-07-27, a new and more fundamental blocker):** the
  $P(M_\text{total})=P_\text{neb}$ outer boundary condition has **no solution at all** for
  the degenerate-supported structure — not a razor-thin-but-findable root as in the
  ideal-gas-only case, a genuine **gap** in achievable surface pressure. Scanning
  $P_\text{center}$ broadly shows $P_\text{end}$ jumps discontinuously from being trapped
  below $\sim0.05$–$0.08\,\text{dyn/cm}^2$ (integration fails before completing the mass) to
  $\ge2.79\times10^6\,\text{dyn/cm}^2$ (integration succeeds) with **nothing in between** —
  and $P_\text{neb}=10^{-4}$ falls squarely inside that gap. Confirmed not a tolerance
  artifact: tightening `solve_ivp`'s `rtol` by 4 orders of magnitude changes nothing. This
  directly confirms and sharpens the outer-envelope finding above: a real degenerate
  object's atmosphere must hand off to a photospheric condition at a far higher pressure
  than $P_\text{neb}$ long before the bulk equation of state could ever get there. **Sub-task
  5 is now blocked on redesigning this boundary condition, not on the EOS** — tracked as the
  new immediate next step (design under review before implementation).

**Open — hypotheses that still need direct verification, not yet confirmed:**

- The exact form of the replacement photospheric outer boundary condition (candidate:
  Eddington/grey-atmosphere $\tau=2/3$, $P_\text{photosphere}\approx\frac{2}{3}\frac{g}{\kappa}$)
  has not yet been designed in detail or implemented — this is the immediate next step.

### What works right now, concretely (updated 2026-08-01)

- `config.py`, `gradients.py` (including `marginal_convective_luminosity` and the new
  `L>=0` floor in `grad_radiative`), `eos.py`, `opacity.py`, `odes.py`, `boundary_conditions.py`
  — clean, no known issues.
- `bvp_solver.solve_static_structure()` — converges cleanly, compact
  ($R\approx3.17\,R_\text{Jup}$, matching the analytic degenerate-polytrope prediction).
- `bvp_solver.relax_initial_state()` — all 11 $\alpha$ pseudo-steps converge, producing a
  state genuinely self-consistent with `solve_timestep`'s real 4-ODE equations (not just
  `solve_static_structure`'s forced adiabat).
- `bvp_solver.solve_timestep()` — verified end-to-end from a relaxed state: converges with
  residuals $\lesssim10^{-8}$, producing a physically sensible first step.
- `dev_cache.py` — pickle save/load for a `SimulationState`, for cheap iterative debugging
  of downstream logic without re-running the full solve chain.
- `diagnostics.py` — visual plots (`plot_structure_profile`, `plot_mass_radius`,
  `plot_convective_zones`) and the print report (`run_diagnostics`, virial balance now in
  the correct unconfined form) all run cleanly against the real converged structure.

### What's blocked / not working

- `time_stepper.py` is **unchanged** from its original Sub-task 7 implementation — it still
  contains the now-obsolete bootstrap dispatch (`_bootstrap_time_derivatives`,
  `state_prev=None` branch). Not yet edited.
- `time_stepper.run()` (Sub-task 8) does not exist yet.
- `validation.py` has **not** been re-run successfully since Sub-task 5's premise change —
  several checks (see §4) are known to be stale relative to the current code (in particular
  Check 19, which still references the pre-photospheric-BC residual formula) and would fail
  if run today.

---

## 2. Module Reference

This section describes what the code *does now*. Modules unaffected by this session's
investigation (`state.py`, `opacity.py`, `odes.py`) are unchanged from before and not
repeated in full detail here beyond a pointer — see git history / the code itself.

### `config.py` — single source of truth for numbers

All CGS physical constants, nebula/envelope parameters, grid resolution, and validity
limits. This session added `T_CENTER_INITIAL` (1200 K, the prescribed $t=0$ central
temperature for the compact hot-start construction, §5) and repurposed
`RHO_GUESS_INITIAL` (now a compact-protoplanet density scale, ~0.05 g/cm³, used only to
seed the shooting method's radius/pressure guess) and `T_KH_BOOTSTRAP_S` (renamed
`T_KH_TIMESCALE_S`; no longer a source-term rate law anywhere — kept only as an
order-of-magnitude KH timescale for residual non-dimensionalization and future
step-size selection). `BOOTSTRAP_KICK_DT_FRACTION` was added and then removed within this
same session, along with the "kick" mechanism it supported (§5's change log has the full
arc). **Sub-task 2f (2026-07-27) added** `M_E` (electron mass), `PLANCK_H` (Planck's
constant), and `MU_E=1.17` (mean molecular weight per electron, standard solar-composition
$2/(1+X)$ estimate, $X\approx0.71$ — distinct from `MU`, the mean weight per particle used
by the ideal-gas term; an accepted first-order inconsistency, see the `eos.py` entry below).
**Sub-task 9 (2026-08-09) added** the Time-Stepping Parameters section:
`USE_ADAPTIVE_DT`, `ADAPTIVE_DT_SAFETY_FACTOR`, `ADAPTIVE_DT_GROWTH_FACTOR`,
`ADAPTIVE_DT_MIN`/`MAX` (the latter raised from a 5e4 yr validation rail to a purely
defensive `1e8` yr backstop once the growth cap was proven self-limiting in practice — §5's
2026-08-09 entry). `AGE_SOLAR_SYSTEM_S` was renamed to `T_MAX_S` and raised 4.5→10 Gyr
(2026-08-10, §5) once its role shifted from "the solar system's present-day age" to "a
diagnostic time budget for testing whether contraction asymptotes."

### `state.py` — the one mutable data object

Unchanged. `SimulationState` is a `@dataclass` holding `m`, `r`, `P`, `L`, `T`, `rho`, `t`,
and `prev`.

### `eos.py` — combined ideal-gas + electron-degeneracy equation of state (Sub-task 2f, done)

**Revised this session.** `specific_heat_cp`, `grad_adiabatic` unchanged (still pure
ideal-gas — a full degenerate-gas thermodynamic treatment, including entropy/$c_p$, is
out of scope for this minimal, additive-pressure-only fix). Two changes:

- **New:** `degenerate_pressure(rho, mu_e)` — the classical non-relativistic
  electron-degeneracy pressure (Chandrasekhar 1939), $P=\frac{h^2}{20m_e}(3/\pi)^{2/3}
  (\rho/(\mu_e m_H))^{5/3}$. Validated against a hand-computed reference point (Check 33)
  and its numerical coefficient cross-checked against the standard literature constant
  ($\approx1.0\times10^{13}$ in the $P=C(\rho/\mu_e)^{5/3}$ convention) to ~1% (expected,
  given slightly different fundamental-constant precision across sources).
- **Revised:** `density(P, T, mu, mu_e)` — now inverts the *combined*
  $P=P_\text{ideal}(\rho,T)+P_\text{degenerate}(\rho)$, which has no closed-form inverse
  (the degenerate term is nonlinear in $\rho$). Solved via vectorized Newton-Raphson (50
  fixed iterations as of the outer-BC/relaxation work later this session, up from an initial
  20 — raised while chasing the `relax_initial_state` numerical issues, see the
  `bvp_solver.py` entry below; did not by itself fix that issue, so the extra iterations are
  cheap insurance, not confirmed necessary), seeded from the ideal-gas-only inversion, with a
  positivity clamp each step (`rho=max(rho,1e-300)` — a domain guard added the same session
  after a caller probing an extreme trial point drove the iteration negative, producing NaN
  via `rho**(5/3)`) and a final assertion that the recovered $\rho$ actually reproduces the
  target $P$ to $10^{-8}$ relative precision — a real failure surfaces loudly (project
  convention: no silent numerical dampening) rather than returning a wrong density.
  Round-trip tested (Check 35) across 8 decades of $\rho/\rho_\text{cross}$ (the
  ideal/degenerate crossover density) with relative error $\lesssim10^{-6}$ everywhere,
  mostly at machine precision. Every call site (`odes.py`, `bvp_solver.py` x3+) updated to
  pass `config.MU_E`.
- **Known first-order inconsistency, accepted as within scope:** `mu_e=1.17` assumes full
  ionization (appropriate deep in the degenerate interior, where this term actually
  matters), while `MU=2.34` describes the cool, *molecular* (un-ionized) outer envelope the
  ideal-gas term is used for. Using both in the same additive formula is not fully
  self-consistent across the whole profile, but degeneracy is negligible in the outer
  molecular region anyway (§1's crossover-density estimate), so this doesn't affect the
  regime where the degenerate term actually contributes.

This resolved the interior-compactness problem as predicted (§1) but exposed a more
fundamental, and blocking, problem in `bvp_solver.py`'s outer boundary condition — see that
entry below.

### `opacity.py` — Bell & Lin (1994) 8-regime piecewise opacity

**★★★★★★ 2026-08-11: smoothed regime blending (`bell_lin_opacity_smooth`,
`config.OPACITY_SMOOTH_TRANSITIONS`, default True) replaces the hard `np.where` regime switch**
- kappa(rho,T) was already continuous at each transition by construction, but `d(kappa)/dT`
genuinely jumped (different power-law exponents either side), confirmed as the cause of a real
mesh explosion. Partition-of-unity logistic blend (`_regime_weights`,
`config.OPACITY_TRANSITION_SMOOTH_WIDTH_DEX`), verified against the hard switch away from
transitions and visually (Check 39, `opacity_hard_vs_smooth_metal_grain_evaporation.png`).
**Same day, its own analytic derivative added** (`bell_lin_opacity_smooth_derivatives`,
`_regime_weights_derivatives`) after `bvp_solver._opacity_derivatives` was found still
computing the HARD-switch derivative even when the residual used this smoothed value - the
same Jacobian/residual mismatch class as the P/T soft clamp (§5's 2026-08-11 entry has the
full derivation and the FD-verification-methodology detour it took to confirm). One
documented gap, not silently assumed away: the derivative differentiates the RAW per-pair
`transition_temperature`, not `monotonic_transition_temperatures`' own `np.maximum.accumulate`
clamp (relevant only where regime ordering gets reshuffled at unusual densities) -
`_regime_weights_derivatives`'s own docstring has the reasoning; Check 37's real-mesh coverage
is the safety net if this gap is ever actually reached in practice.

### `gradients.py` — Schwarzschild criterion + new diagnostic helper

`marginal_convective_luminosity(m, P, T, kappa, grad_ad)` — inverts
$\nabla_\text{rad}(L,\ldots)=\nabla_\text{ad}$ for $L$ in closed form (the "marginally
efficient convection" closure). Used by `bvp_solver.solve_static_structure()` to populate a
physically meaningful, non-trivial $L(m)$ for a $t=0$ structure whose $T(m)$ was built
directly from the adiabat rather than solved for — not consumed by `solve_timestep` (which
only ever interpolates `state_prev.T`, `.P`, never `.L`), so this is a diagnostic/plotting
convenience, not something load-bearing for the time evolution itself.

**`grad_radiative` and `effective_gradient` revised 2026-08-06** (§5 has the full
diagnostic trail): both previously had a *hard* switch — `L_safe=max(L,0)` in
`grad_radiative`, `nabla_eff=min(nabla_rad,nabla_ad)` (via `np.where`) in
`effective_gradient` — each a genuine non-differentiable kink, confirmed by direct
instrumentation to be blocking `relax_initial_state`'s adaptive $\alpha$-ramp at
$T_\text{CENTER\_INITIAL}=13000$K. Both replaced with a smooth hyperbolic
("smoothed absolute value"/pseudo-Huber) form, computed in the algebraically-equivalent,
cancellation-safe way (`config.GRAD_RAD_L_FLOOR_EPSILON`,
`config.GRAD_EFF_SWITCH_EPSILON` — see those constants' own comments for the full scale
derivation and the two real bugs the standalone verification step caught before either was
trusted). `is_convective`'s hard comparison is unchanged and still exact — it's
informational only, never fed back into the ODE integration, so it doesn't need smoothing.
**This fix is necessary but not sufficient**: a third, still-unresolved wall appeared in the
same region of $\alpha$-space immediately after both were fixed (opacity's Bell & Lin hard
regime switches are the leading unconfirmed suspect) — this, combined with the pattern of
three independent kinks clustering in the same narrow region, motivated the 2026-08-06
architectural decision (§5) to abandon shooting for `scipy.integrate.solve_bvp` entirely.
`gradients.py` itself is unaffected by that pivot — it's reused unmodified by
`bvp_experiment.py`.

### `odes.py` — the 4-ODE right-hand side

Unchanged: `stellar_odes(m, y, dT_dt, dP_dt)`. The energy equation's textbook implicit form
(§4.8 of PLAN.md) was never wrong — an earlier session mistake was adding an *extra* term
to the `dT_dt`, `dP_dt` **inputs** it receives from `bvp_solver.py`, not to this function
itself; that extra term has since been removed at the call site (see below).

### `boundary_conditions.py` — the 4 boundary residuals

**Revised this session.** The surface thermal residual changed from a rigid
`T_b - config.T_NEB` clamp to the net-flux radiative condition
`L_b - 4*pi*r_b**2*SIGMA_SB*(T_b**4 - T_NEB**4)` (PLAN.md §4.7 has the full physical
reasoning for why). The mechanical residual (`P_b - P_neb`) and both center residuals
(`r_a`, `L_a`) are unchanged. Covered by validation.py's Check 19, which was rewritten in
place this session to match the new (nonlinear in $T_b$) formula.

### `bvp_solver.py` — $t=0$ shooting seed + $t>0$ `solve_bvp` collocation solver

**★★★★★★★ 2026-08-11: `_safe_exp_state`'s hard `np.clip` replaced by a smooth softplus-based
`_soft_clamp` (`config.BVP_SOFT_CLAMP_WIDTH`); its derivative (`_safe_exp_state_derivatives`)
now correctly threaded through `implicit_rhs_jacobian`/`make_bc_jacobian_scaled` everywhere -
the old hard clamp's zero derivative in saturation was BOTH why Newton had no restoring force
there AND why those Jacobians silently went wrong (they multiplied by the bare `P`/`T` value,
correct only where the clamp's true derivative is 1). `relax_initial_state`'s stage 2 now uses
the fast analytic Jacobian throughout (previously forced to scipy's numerical Jacobian to
sidestep this bug); stage 1 still needs the clamp forced off, confirmed independent of this
fix (a separate, genuine numerical property of that stage's own trajectory). See §5's
2026-08-11 entry for the full diagnosis, fix, and verification (Checks 37/40/40b).**

---

**★★ 2026-08-08: PROMOTED to `solve_bvp`, replacing this file's own `relax_initial_state`/
`solve_timestep` in place.** The 2026-08-06 pivot below is no longer conditional — it's
done. `solve_static_structure()` (this file's $t=0$ adiabatic seed, `brentq` shooting) is
**unchanged** and still the active implementation, exactly as before. `relax_initial_state`/
`solve_timestep` (t>0) have been **rewritten in place** to solve via `scipy.integrate.
solve_bvp` (nondimensionalized state, analytic Jacobians, `ALPHA_MAX` continuation - the
machinery `bvp_experiment.py` proved) instead of the shooting/LM code described in the rest
of this subsection, which is now **archived verbatim in `bvp_solver_shooting_archive.py`**
(not deleted, not imported by anything active). Both new functions keep their old names and
`SimulationState`-in/`SimulationState`-out signatures, so `time_stepper.py` and every other
caller needed zero changes. Full detail: `PLAN.md` §4.2/Sub-task 5's 2026-08-08 update.

**What's new, beyond a straight port of `bvp_experiment.py`'s logic:**
- A shared internal `_solve_structure_bvp(state_prev, dt, warm_start_L)` helper used by both
  public functions (the underlying physics/RHS already generalizes over `dt`;
  `bvp_experiment.py`'s own orchestration had just hardcoded the pseudo-relaxation `dt`).
- **A genuine new bug, found and fixed while verifying `solve_timestep` with a REAL `dt` for
  the first time ever** (`bvp_experiment.py`'s milestones only tested the pseudo-relaxation
  step): the initial-guess mesh linearly interpolated `P`, `T` directly (not `ln P`, `ln T`)
  from an already-converged warm-start state's OWN output grid. Near the photosphere, a
  converged solution's `P` genuinely drops ~3 decades over the final ~$10^{-5}$ of the mass
  (a real, extremely thin "skin," not a numerical artifact) - linear interpolation of `P`
  itself (then logged) badly misrepresented that drop, feeding `solve_bvp`'s very first
  collocation midpoint an unphysical trial that crashed `eos.density` before any Newton
  correction happened. This was silently masked for `relax_initial_state` (warm-starting
  from `solve_static_structure`'s output, whose shooting-event surface never reaches
  exactly $m=M_\text{TOTAL}$, so `np.interp`'s boundary clamping happened to avoid the worst
  of it) - `solve_timestep`'s full-domain warm start is what finally exercised it. **Fixed**:
  interpolate `ln P`, `ln T` directly (the actual solved-for state variables, not their
  exponentials), plus a much deeper `GRID_OUTER_REFINEMENT` (new `config.
  BVP_MESH_OUTER_REFINEMENT`) applied ONLY for `warm_start_L=True` (`relax_initial_state`
  doesn't need it and was confirmed to get WORSE - more mesh nodes, eventually exceeding
  the budget - if it were applied there too).
- `verify_jacobians` (the FD cross-check `bvp_experiment.py` ran before every solve) is
  **not** carried into the production hot path - re-running an FD check on every real
  timestep of a long run is wasteful once the Jacobians are trusted. It's promoted instead
  into `validation.py` (`check_bvp_jacobian_matches_finite_differences`, new Check 37),
  run once/on-demand rather than every solve.

**Verified before trusting this promotion** (not just "it imports"): `relax_initial_state`
reproduces `bvp_experiment.py`'s own recorded 11500K/12000K results to <1% (mostly far
tighter); a genuine real-`dt` `solve_timestep` call converges directly (`status=0`, residual
9.73e-7) from the relaxed 11500K state - the first time this has ever been demonstrated
under `solve_bvp`. See §5's 2026-08-08 entry for the full numbers, the physical result (a
70x drop in $|L_\text{surface}|$ over one real step - informative for the standing
negative-$L_\text{surface}$ question), and the **open, unresolved** finding that a SECOND
real step does not yet converge within a generously raised node budget.

---

**⚠ 2026-08-06 (historical from here down): shooting was architecturally ABANDONED for the
$t>0$ problem** — see §5's "Architectural decision" entry and `PLAN_BVP.md`.
`solve_static_structure()` (the $t=0$ adiabatic seed construction) remains in active use,
reused unmodified — but `relax_initial_state()` and the shooting/LM machinery described
below are the RETIRED implementation (now in `bvp_solver_shooting_archive.py`, per the
2026-08-08 note above), kept in this subsection only as historical reference, not current
architecture guidance.

**Substantially rewritten this session; current content described in full since this is
the module under active investigation.**

**Independent correctness review completed (2026-07-27) — verdict: the code is
correct, but a forced-adiabat modeling shortcut (not a bug) makes its output an
unreliable proxy for the real coupled system.** Before touching Sub-task 2f, every
line of `solve_static_structure` and `solve_timestep` was checked against the physics
and the shooting-method logic, not just "does it run." Full findings:

- **Lane-Emden wiring, `_solve_lane_emden`/`_adiabatic_center_guess`:** re-derived the
  scaling relations from scratch and confirmed each line matches; re-verified the
  converged structure satisfies the exact adiabatic relation
  $T/T_\text{center}=(P/P_\text{center})^{\nabla_\text{ad}}$ to $5\times10^{-9}$
  relative error, with $r$, $P$, $T$, $\rho$ strictly monotonic and everywhere
  positive (no sign flips, no hidden numerical corruption near the surface). Correct.
- **Outward integration and `brentq` root-find:** RHS and $x=\ln m$ chain-rule
  handling match `odes.py`; the bracket and convergence are correct and independently
  cross-validated (numerical root agrees with the analytic Lane-Emden estimate to
  0.08%). No explicit stopping event exists — the code relies on the natural endpoint
  plus letting `solve_ivp` fail on genuine stiffness, and this was confirmed to be a
  real "ran out of pressure" signal, not silently-corrupted output. Correct.
- **`boundary_conditions.py` usage inside `solve_timestep`:** `yb=[r_b,P_b,L_b,T_b]`
  ordering matches the unpacking exactly; the two residuals used
  (`res_full[2]`=$P_b-P_\text{neb}$, `res_full[3]`=$L_b-L_\text{expected}$) are
  correctly indexed, signed, and dimensioned. Correct.
- **Is $R\approx300\,R_\text{Jup}$ (pure adiabat) or $R\approx27{,}000\,R_\text{Jup}$
  (the unwritten, self-consistent alternative explored in scratch testing) explained
  by a bug? No, for either — but the full picture is more nuanced than "missing
  degeneracy pressure" alone.** Tracking $\nabla_\text{rad}$ vs. $\nabla_\text{ad}$
  along the self-consistent construction (previously an unconfirmed guess — now
  checked directly) shows it is genuinely convective out to $m/M\approx0.49$, then
  **transitions to radiative at $m/M\approx0.70$** ($r\approx177\,R_\text{Jup}$,
  $P\approx35\,\text{dyn/cm}^2$, $T\approx192\,\text{K}$). Past that point $T$ barely
  moves (192 K → 157 K) while $r$ explodes $153\times$ (to $27{,}146\,R_\text{Jup}$) as
  $P$ drops 5 more decades to reach $P_\text{neb}$ — the same nearly-isothermal,
  extended-envelope signature as the *original* Bonnor-Ebert problem (Premise 1), now
  confined to the outer ~30% of the mass instead of the whole star. A rough
  surface-gravity/opacity estimate at the transition point ($g\approx0.06\,\text{cm/s}^2$)
  suggests the true photosphere ($\tau=2/3$) sits at a much higher pressure than
  $P_\text{neb}$, meaning the current $P(M_\text{total})=P_\text{neb}$ outer BC likely
  forces the integration well past where the radiative-diffusion equation is even
  physically valid, independent of any numerical issue.
- **On the `P(M_\text{total})=P_\text{neb}$ outer BC specifically (PROJECT_CONTEXT.md
  §3 raised this for the earlier hot-start attempt and it had not been revisited under
  the current line of work — now revisited):** this looks like a **second, independent
  gap**, not a restatement of the missing-EOS one. Electron degeneracy pressure
  ($\propto\rho^{5/3}$) is negligible in the tenuous outer envelope where this
  extended radiative tail develops, so the planned Sub-task 2f fix should **not** be
  expected to resolve it. The two gaps affect different regions: degeneracy pressure
  is about the dense interior's ability to be compact; the outer BC issue is about
  whether the tenuous outskirts should be modeled by radiative diffusion at all the way
  down to $P_\text{neb}$, versus cutting off at a physically-motivated photospheric
  condition. **Tracked as a separate, likely still-necessary follow-up — not addressed
  by Sub-task 2f, and not blocking it either** (Sub-task 2f targets interior
  compactness specifically, which this finding does not undermine).
- **Net conclusion:** `solve_static_structure` and `solve_timestep` are each correct
  implementations of what they are individually designed to compute. The problem is
  architectural, not a coding bug: forcing a pure adiabat all the way to
  $P_\text{neb}$ is a poor proxy for what the real coupled 4-ODE system wants to do
  once given a chance (grow a large radiative envelope) — which is exactly why
  `solve_timestep`'s residual is so large when evaluated at `solve_static_structure`'s
  output. Proceeding with Sub-task 2f is still correct and necessary; the outer-BC
  question should be revisited afterward, before Sub-task 5 is declared complete.
- **UPDATE (2026-07-27, after Sub-task 2f was implemented): the outer-BC issue turned out
  to be worse than "should be revisited" — it's a hard blocker, confirmed directly.** With
  the combined EOS in place, `solve_static_structure` needed two bracket-search fixes
  (below) before converging at all; even converged, its surface-pressure residual is
  enormous ($P_\text{end}\approx1428$ vs. target $P_\text{neb}=10^{-4}$). Scanning
  $P_\text{center}$ broadly shows why: $P_\text{end}$ jumps discontinuously from trapped
  below $\sim0.05$–$0.08$ (integration fails) to $\ge2.79\times10^6$ (succeeds), with
  **no $P_\text{center}$ reaching anywhere near $P_\text{neb}$** — confirmed not a tolerance
  artifact (`rtol` tightened by $10^4\times$, no change). The degenerate-supported structure
  genuinely cannot bridge the same bulk equation of state down to the tiny ambient nebula
  pressure. **Sub-task 5 is now blocked on redesigning this boundary condition** (a
  photospheric condition, design under review) rather than merely "should revisit it later."

**Two bracket-search robustness fixes needed once the combined EOS was in place (both now
in `solve_static_structure`):**
1. **Search direction is not fixed.** The original bracket-expansion always searched by
   *increasing* $P_\text{center}$ (correct for the old ideal-gas-only construction). For the
   degenerate-dominated structure this is backwards — degenerate objects have an *inverted*
   mass-radius relation ($R\propto M^{-1/3}$), so *decreasing* $P_\text{center}$ is what
   reduces the surface-pressure residual here. Not a bug in the original code — a genuine
   consequence of which EOS term dominates. Fixed by expanding in both directions
   simultaneously and taking whichever finds a sign change first, rather than assuming a
   fixed direction.
2. **`brentq`'s stopping tolerance can land on a point that doesn't actually integrate
   successfully.** The crash/success transition can be razor-thin — down to the same
   $\sim10^{-13}$ relative step-size floor `solve_ivp` itself hits — sharper than `brentq`'s
   bracket-width stopping criterion. Fixed by verifying the returned root actually
   integrates successfully and nudging it toward the known-good side (the original analytic
   guess, verified successful) in small, geometrically-increasing steps if not — a small,
   bounded, and printed deviation from the mathematically exact root, raising loudly (not
   silently) if nudging doesn't recover.

**⚠ EVERYTHING BELOW IN THIS ENTRY IS CURRENT AS OF PAUSING (2026-07-27, end of session).**
The bracket-search robustness fixes above (search-direction-agnostic expansion, nudge-toward-
known-good) are still in the code but their ROLE changed later the same session — see the
"outer BC redesign" and "initial-state relaxation" sub-entries below, which supersede the
`P=P_\text{neb}`-based numbers quoted just above (kept for historical trail, not current
behavior).

**Outer BC redesign: `P(M_\text{total})=P_\text{neb}` replaced with a photospheric condition
— DONE, validated.** PROGRESS.md's blocker finding above (no $P_\text{center}$ reaches
$P_\text{neb}$) was fixed, not worked around, by replacing the mechanical surface condition
with the standard Eddington grey-atmosphere result ($\tau=2/3$):
$$P_\text{photosphere} = \frac{2}{3}\frac{g}{\kappa},\quad g=\frac{GM_\text{total}}{r^2}$$
(full derivation and physical reasoning in `boundary_conditions.py`'s module docstring, now
`boundary_conditions.photospheric_pressure(r, P, T, mu, mu_e)`). This is not just a new
residual formula — it changes **how the surface is located**: both `solve_static_structure`
and `solve_timestep` now integrate outward with the photosphere as a `solve_ivp` **event**
(`_photosphere_event_adiabatic`, `_photosphere_event_implicit` — same pattern as
`_solve_lane_emden`'s own surface-crossing event) and match the **enclosed mass at that
event** to `M_TOTAL`, rather than checking a residual at a fixed `m=M_TOTAL` grid endpoint. A
fixed-endpoint version of the photospheric condition was tested first and found to have the
*same* reachability-gap problem `P=P_\text{neb}` did; the event-based reformulation was
verified (read-only scratch test, before committing to the approach) to be smooth and
gap-free across the same `P_center` range that produced the old gap.

**Result: `solve_static_structure()` now converges cleanly.** $P_\text{center}\approx
7.686\times10^{11}\,\text{dyn/cm}^2$, $T_\text{center}=1200\,$K (prescribed),
$R_\text{surface}\approx3.172\,R_\text{Jup}$ (matches both the Step 3 analytic prediction and
the pre-implementation event-based scratch test closely), $T_\text{surface}\approx7.5\,$K,
mass relative residual $\approx0.16\%$ (down from the $\sim10^7$ relative residual the old
condition gave when force-nudged to "succeed"). $r$, $P$, $T$ all strictly monotonic,
everywhere finite. `_adiabatic_center_guess()` (the Lane-Emden bracket seed) is unchanged by
this — still a reasonable order-of-magnitude seed regardless of how the surface is located.

**Bridging to `solve_timestep`: still blocked — this is where the session paused.**
Evaluating the real 4-ODE system (genuine `odes.stellar_odes` Schwarzschild selection, not
the forced-adiabat shortcut) at `solve_static_structure`'s own center values confirmed the
Step-1-correctness-review concern sharply: $T$ diverges to $\sim3.4$ million K within one
full-sized implicit step ($dt=0.01\,t_\text{KH}$) — `state_0` is not a genuine solution of
the same equations `solve_timestep` uses, and naively feeding it in doesn't just give a large
residual, it's numerically unstable.

*Considered and rejected: scaling `dL/dm` by a homotopy parameter $\alpha$* (start at
$\alpha=0$ to "turn off" the mismatch, ramp to $\alpha=1$). **This is a real mathematical
trap, not just a bad idea**: `dL/dm` has no source in this codebase other than the
`dT_dt`,`dP_dt` terms, so `alpha=0` forces `dL/dm\equiv0$ identically — reproducing, exactly,
the original isothermal degeneracy this entire investigation exists to escape (PROGRESS.md's
very first finding). Caught before implementation.

**Corrected approach, implemented: homotopy on $\nabla_\text{eff}$ directly, not on
`dL/dm`.** `_implicit_rhs_logm(x, y, state_prev, dt, alpha=1.0)`:
$$\nabla_\text{eff,used} = (1-\alpha)\,\nabla_\text{ad} + \alpha\,\nabla_\text{eff,real}(L)$$
computed as `(1-alpha)*dT_dm_adiabatic + alpha*dT_dm_real` (both share the same `dP_dm`,
independent of the temperature gradient, so only `dT_dm` needs recombining). At `alpha=0`
this reproduces `solve_static_structure`'s own construction *exactly* (a genuinely
self-consistent starting point) rather than the degenerate one; `dL/dm` itself is computed
the same, unscaled way throughout (from the real, unmodified implicit `dT_dt`, `dP_dt`).
`alpha=1.0` (the default) is bit-for-bit the original formula — every real `solve_timestep`
call is unaffected.

`relax_initial_state(state_0)` ramps `alpha` over `np.linspace(0.0, 1.0, 11)` (fixed,
auditable spacing — 0.0, 0.1, ..., 1.0), `state_prev` held fixed at `state_0` throughout, a
fixed pseudo-timestep `dt_relax=0.01*T_KH_TIMESCALE_S` (not real elapsed time — same
"mathematical device" convention the old, removed bootstrap kick used). **Convergence
criteria per pseudo-step, deliberately strict (no blind continuation):** (1) `fsolve` must
report `ier==1`; a failed step **raises immediately** rather than continuing with an
unreliable intermediate state (contrast with `solve_timestep`'s real per-timestep calls,
which only warn — here we control the step size, so failure means "make it finer," not
"push through"). (2) The achieved `[mass, thermal]` residual is printed at every step for a
visible audit trail, not just "ier==1" (which only reflects `fsolve`'s own internal
criteria). (3) A smoothness guard checks the $(P_\text{center},T_\text{center})$ jump between
consecutive steps against a 50%-relative threshold, warning on a suspiciously large jump
(fsolve reporting success doesn't rule out having converged to a *different* solution
branch).

**Status: the `alpha=0.000` step converges beautifully** (residuals
$[-4.3\times10^{-13}, -3.4\times10^{-8}]$) **— validating the corrected homotopy is
physically and mathematically sound.** But reaching subsequent steps hit a cascade of
numerical edge cases in `scipy`'s stiff-solver internals (Radau), each fixed individually but
each revealing a new one, four rounds deep by the time the session paused:
1. Evaluating exactly at the (unperturbed) seed `u0` at `alpha=0` overflowed — `T_trial` and
   `T_prev` coincide almost to machine precision there (both follow the same adiabat by
   construction), so `(T-T_prev)/dt` divides near-zero floating-point noise by a small `dt`
   (catastrophic cancellation). **Fixed**: nudge the initial `fsolve` guess by a tiny
   ($10^{-6}$ relative), physically-negligible amount off the exact match point.
2. `eos.density`'s Newton-Raphson then failed to converge on a trial point probed during
   Radau's own internal stiff-solver Jacobian estimation. Increasing iterations 20→50 did
   **not** help — not an iteration-count problem.
3. Diagnosed as Radau probing **negative $P$ or $T$** (no EOS solution exists there — both
   pressure terms are non-negative for $\rho>0$). **Fixed**: added a positivity clamp on
   `_implicit_rhs_logm`'s own inputs (`P=max(P,1e-300)`, `T=max(T,1e-300)`), letting Radau's
   adaptive step control back away naturally rather than crashing on an intermediate probe
   that was never going to be the accepted step (same spirit as `eos.density`'s own
   rho-positivity clamp, added earlier the same session for the same class of reason).
4. That clamp was too permissive: Radau then probed a trial with $T$ driven toward the
   $10^{-300}$ floor while $P$ stayed large, giving the ideal-gas-only Newton seed
   $\rho=P/T\cdot(\ldots)$ an overflow-scale value, breaking `eos.density` again, differently.

**Paused here — explicitly NOT continuing to guess at domain-clamp values.** Two candidate
principled fixes are on the table, neither implemented, both to be evaluated next session:
- **Path 1 — log-transformed state variables.** Integrate $x=\ln P$, $y=\ln T$ (not $P$, $T$
  directly) so they are mathematically guaranteed positive by construction, eliminating
  negative-domain probing at its source rather than clamping after the fact.
- **Path 2 — graceful degradation inside `eos.density`.** If the Newton solve receives an
  extreme/unconvergeable probe, return a bounded penalty value (signaling "very wrong,
  back off") instead of a hard assertion failure, letting the integrator's own step-size
  control respond rather than crashing the whole `solve_ivp` call.

No decision made yet on which (or whether some third option) is right — see PLAN.md's
Sub-task 5 entry for the same status, kept in sync.

**$t>0$ step, ordinary calls (`solve_timestep`):** unchanged in this entry from before —
`_implicit_rhs_logm` (now `alpha`-aware, defaulting to 1.0 so real calls are bit-identical to
before), `_integrate_timestep_outward` (now also event-based, same photosphere event as
`solve_static_structure`), `solve_timestep()` itself (mass + thermal residuals now evaluated
at the photosphere event point, not a fixed grid endpoint). **Still not validated end-to-end**
— blocked on `relax_initial_state` producing a usable `state_1` to test it against.

### `bvp_experiment.py` — new 2026-08-06, the active `solve_bvp` architecture

Standalone, isolated `scipy.integrate.solve_bvp` implementation — the project's target
architecture going forward (§5's "Architectural decision" entry, `PLAN_BVP.md`). Imports and
calls `bvp_solver.solve_static_structure`/`_build_output_grid`, `odes.stellar_odes`,
`boundary_conditions.boundary_conditions`, `eos.grad_adiabatic` unmodified; reimplements
only the RHS/BC glue as new, properly mesh-vectorized functions (`bvp_solver.
_implicit_rhs_logm` is written for `solve_ivp`'s single-point-at-a-time contract and isn't
directly reusable for `solve_bvp`, which evaluates the whole mesh at once). Same state
representation as shooting (`y=[r, lnP, L, lnT]`, log $P$/$T$ for guaranteed positivity).

Structurally simpler than shooting for the surface condition: since `M_TOTAL` is a known
project constant (not a shooting unknown), the mass domain `[m_min, M_TOTAL]` is fixed, so
the photospheric condition is a genuine boundary equation at the true endpoint — no
`solve_ivp` event, no mass-matching residual. `P_center`/`T_center` are just `ya[1]`,
`ya[3]`, part of the one global unknown `solve_bvp` solves for directly.

**★ Current status (2026-08-07): CONVERGES at the project's active target, confirmed at
TWO temperatures (T=11500K and T=12000K).** `status=0` at both; residuals to
$9.79\times10^{-7}$ (11500K) and $8.88\times10^{-16}$ (12000K, tighter). Required, beyond the RHS/BC glue described above: a nondimensionalized state
vector $z=[\hat r,\ln P,\hat L,\ln T]$ (`implicit_rhs_scaled`, `implicit_rhs_jacobian_scaled`,
`make_bc_scaled`, `make_bc_jacobian_scaled` - `_to_physical`/`_to_scaled` convert to/from the
original `y=[r,lnP,L,lnT]`), hand-derived analytic Jacobians (`fun_jac`/`bc_jac`, verified
against finite differences before use - `verify_jacobians()`), corrected EOS thermodynamics
in the shared `eos.py`/`odes.py`/`config.py` (γ, μ, δ - see §5's Milestone 6 entry), and a
continuation endpoint of `ALPHA_MAX=1-1e-5` rather than literal `alpha=1.0` (empirically
required - the exact endpoint diverges via mesh-explosion-to-NaN, `ALPHA_MAX` doesn't).
`T_CENTER_INITIAL` moved 13000K→11500K as a direct consequence of the EOS correction (see
`config.py`'s own comment). Toy-opacity substitution (Milestone 1) remains available but
unused in this configuration - opacity was already ruled out as a cause.

Getting here required six milestones (`PLAN_BVP.md` §3.0-3.6), each ruling out one
candidate cause by direct, isolated test (ionization, dissociation-μ, opacity switches,
center-BC self-consistency, log-space surface BC, FD-Jacobian imprecision) before Milestone
6 combined nondimensionalization with the EOS corrections and succeeded. Two real
implementation bugs were caught by the mandatory FD cross-check before the scaled Jacobian
was trusted (§5's Milestone 6 entry has both); a third, more subtle case (the `bc_jac`
verification metric itself producing a false-alarm on near-zero entries at T=2000K) was
caught and fixed 2026-08-07 while preparing the second-temperature confirmation, mirroring
the same fix already applied to `fun_jac`'s metric (row-normalized error, not per-entry).
**Open items before this is production-ready**: a slightly negative `L_surface` — now
confirmed reproducible at both tested temperatures (`PLAN_BVP.md` §3.6.4), not resolved but
no longer a one-run curiosity; the `alpha=1.0`-specific instability's mechanism is inferred
(regularization) not derived; a full validation-suite pass against a converged solution has
not yet been run. T=2000K was attempted as a third data point and found to be outside the
atomic/post-dissociation EOS regime (`config.MU`/`GAMMA` don't describe molecular H) — not
re-attempted, documented as a scope boundary rather than a gap. `PLAN_BVP.md` §3.6/§3.6.4
and §6 have the complete trail and the promotion plan - not duplicated further here.

### `diagnostics.py` — post-solve physical diagnostics

**New this session (2026-08-01):** three visual diagnostic plots — `plot_structure_profile`
($T$, $\rho$, $P$ vs $m$), `plot_mass_radius` ($m$ vs $r$), `plot_convective_zones`
($\nabla_\text{rad}$ vs $\nabla_\text{ad}$, convective regions shaded) — plus
`plot_diagnostics(state)`, a convenience wrapper generating all three. Generated against the
cached, fully-relaxed state (`dev_cache.py`) and visually inspected:
`structure_profile.png`, `mass_radius.png`, and `convective_zones.png` all show smooth,
monotonic, physically sensible profiles. Notably, `convective_zones.png` shows
$\nabla_\text{rad}\gg\nabla_\text{ad}$ (by ~7 orders of magnitude) across the *entire*
structure — an independent visual confirmation, from the real Schwarzschild criterion
evaluated on the genuinely-relaxed state, that the fully-convective assumption
`_adiabatic_rhs_logm` forces for the $t=0$ construction is physically justified for this
object, not just a simplifying assumption.

**Sub-task 6 now fully done (2026-08-01):** `virial_balance` rewritten to the standard
unconfined form ($E_\text{grav}+3(\gamma-1)E_\text{therm}\approx0$, dropping the now
~15-orders-of-magnitude-irrelevant `P_NEB` surface term entirely); `run_diagnostics`'s
imbalance normalization updated to match (normalized against the terms actually being
balanced, not the vanished surface term). Confirmed via the real `solve_static_structure()`
output: $E_\text{grav}=-9.244\times10^{42}$, $3(\gamma-1)E_\text{therm}=+9.241\times10^{42}$
erg, relative imbalance $3.6\times10^{-4}$. Opacity regime census re-checked against the real
compact structure (no code change needed — `opacity_regime_distribution` was already
regime-agnostic): center sits in "Metal grains" (T=1200K), surface in "Ice grains" (T=7.5K),
confirming the expected multi-regime spread.

### `time_stepper.py` — outer Kelvin-Helmholtz contraction loop + adaptive `dt` selection

**Current state (2026-08-10), substantially rewritten across Sub-tasks 8-10 since the
paragraph below was written:**
- `compute_time_derivatives(state_curr, state_prev, dt)` — finite-differenced $\dot T$,
  $\dot P$ on `state_curr.m` (interpolating `state_prev` onto that grid first, in case the
  Lagrangian grid shifted between steps). No longer only a diagnostic utility — also
  `select_adaptive_dt`'s own input.
- `select_adaptive_dt(state_curr, state_prev, dt_used)` (Sub-task 9, **DONE 2026-08-09**) —
  the dual $T$/$P$ thermal-timescale limiter: $\Delta t_\text{raw}=\alpha\cdot\min(\min_i(T_i/
  |\dot T_i|),\min_i(P_i/|\dot P_i|))$, then an asymmetric growth cap ($\Delta t_\text{new}
  \le$ `ADAPTIVE_DT_GROWTH_FACTOR`$\times dt_\text{used}$, growth only), then clamped to
  `[ADAPTIVE_DT_MIN, ADAPTIVE_DT_MAX]`. Deliberately excludes $L$ (structural $0/0$ at the
  center every step; benign zero-crossings near the photosphere would otherwise chronically
  dominate the `min`) — full design reasoning in PLAN.md §4.5.
- `run(state_prev, n_steps, dt, snapshot_interval=1, snapshot_dir=None)` — the production
  loop. Seeds step 1 at the given `dt`; every step after that uses `select_adaptive_dt` if
  `config.USE_ADAPTIVE_DT`, else stays fixed. Flushed live per-step logging (step, $t$ in
  years, `dt`, $r_\text{surface}$, $T_\text{center}$, $L_\text{surface}$) so a long run's
  progress is always visible, never silently stale. A **dual stopping condition**: halts on
  $r_\text{surface}\le$ `config.R_HALT` OR $t\ge$ `config.T_MAX_S`, whichever first
  (`T_MAX_S`, renamed from `AGE_SOLAR_SYSTEM_S` 2026-08-10, is a diagnostic time budget —
  see its own `config.py` comment). An explicit finite/positivity guard raises immediately
  on a corrupted state (NaN or non-positive $r_\text{surface}$/$T_\text{center}$) rather than
  letting it propagate. If `snapshot_dir` is given, every retained step is also saved to disk
  via `output.save_snapshot` as the run proceeds, independent of the in-memory return value.

**Obsolete code removed**: `_bootstrap_time_derivatives` and `compute_time_derivatives`'s
old `state_prev=None` dispatch (referenced the no-longer-existing `config.T_KH_BOOTSTRAP_S`,
confirmed broken 2026-08-01) — no longer needed once Sub-task 8's real per-step derivatives
made the bootstrap unnecessary (§1, PLAN.md's Sub-task 7 entry).

**Validated against the real solver twice at full production scale**: the 2026-08-09
overnight run (77 steps to 4.5 Gyr) and the 2026-08-10 extension (55 more steps to 10 Gyr,
resumed directly from a saved snapshot) — both converged every step directly, no
continuation fallback, `dt` growing smoothly under the growth cap early on and eventually
pinning at `ADAPTIVE_DT_MAX` for the majority of both runs' later steps.

### `validation.py` — sanity checks, unit consistency, and diagnostic plots

**★★★★★ 2026-08-11 update**: Check 37 extended to sample synthetic points inside the P/T
soft-clamp's saturation region (it never did before - exactly how the clamp-Jacobian bug
stayed invisible). New Check 39 (opacity smoothing vs. hard switch, plus its visible plot),
Check 40/40b (soft-clamp identity/boundedness/derivative vs. finite differences), Check 41
(smoothed-opacity derivative vs. finite differences) - 44 checks total. **Running the full
suite**: `python validation.py` still stops at the long-standing Check 17 crash (pre-existing,
unrelated - §4/§5 have the history); the per-check-isolated sweep this project has used since
2026-08-08 to see past it remains the way to get the complete picture (42/44 pass, only
Checks 17/23 fail, both pre-existing).

**★★ 2026-08-10 update**: new Check 38 (`check_outer_envelope_recombination_sensitivity`,
plus `plot_outer_envelope_recombination_sensitivity`) — a sterile sensitivity test (§5 has
the full derivation) that found the missing outer-envelope H/H2 recombination physics moves
`r_surface` by -3.1%, well above the pre-agreed 1% action threshold, justifying real
implementation. Introduced two small private helpers local to this file
(`_mu_proxy_atomic_molecular`, `_reintegrate_radius`) rather than duplicating the mu-proxy/
re-integration logic between the check and its plot function - the first private helpers in
`validation.py`; also the first use of `output.py` from `validation.py` (loading a cached
production snapshot rather than a live solve) and the first use of `scipy.interpolate.
PchipInterpolator`/`scipy.integrate.solve_ivp` here.

See §4 below. **★ 2026-08-08 update**: Check 19 (`check_boundary_conditions_residuals`) and
Check 24 (`check_static_structure_isothermal_and_monotonic`) — both flagged stale in the
paragraph below and left unfixed for a long time — were finally revised while promoting
`bvp_solver.py` to `solve_bvp` (unrelated to that migration's own code, just discovered
along the way; §1/§5's 2026-08-08 entry has the full detail): Check 19 now solves for the
genuinely self-consistent photospheric $P_b$ rather than assuming the old $P_\text{neb}$
target; Check 24 now checks monotonicity and the photospheric surface pressure instead of
asserting the old Premise-1 isothermal/$L$=0 conditions. Both pass. A new Check 37
(`check_bvp_jacobian_matches_finite_differences`) was also added, promoting
`bvp_experiment.py`'s Jacobian-correctness verification into a standing check — passes.
Running the full suite end-to-end surfaced **one separate, pre-existing failure**
unrelated to this work (Check 17, `check_stellar_odes_matches_constant_density_analytic_profile`
— `dr/dm` off by a factor of ~17 from its analytic target; confirmed via `git diff` that
`odes.py`/`eos.py`/`gradients.py` are untouched this session, so this predates 2026-08-08,
most likely dating to the `config.MU`/`GAMMA` atomic-composition correction on 2026-08-07
never having been re-validated against it) — flagged, not fixed, out of this pass's scope.

The paragraph below is the pre-2026-08-08 state, kept for its own historical detail (Checks
33-36, 26, 27's history) — **not fully passing as of that writing** — several checks written
for Premise 1's isothermal $t=0$ state were stale relative to `bvp_solver.py`'s rewritten
`solve_static_structure()`. Checks 33-36 (Sub-task 2f's EOS/degeneracy checks) were
implemented and pass cleanly (2026-07-27); Check 26 (renamed `check_virial_balance_unconfined`)
and Check 27 (opacity regime distribution) were rewritten for the compact structure and now
pass cleanly (2026-08-01, see the `diagnostics.py` entry above for the physical detail).

### `main.py`

**Capped SANITY-CHECK orchestrator, not the production run script** (see `overnight_run.py`/
`extended_run_10gyr.py` below for that). `solve_static_structure()` → `relax_initial_state()`
→ a capped (`N_STEPS_SANITY_CHECK=15`) `time_stepper.run()` call with
`config.USE_ADAPTIVE_DT=True` and snapshot saving wired in, → `output.generate_all_plots()`.
Its role since Sub-tasks 9-10 landed: a small, fast end-to-end smoke test of the adaptive
loop + snapshot/plot pipeline together, before trusting either at production scale — not
meant to be raised to a full run itself (kept deliberately small).

### `output.py` — Sub-task 10, `.npz` snapshot I/O and post-processing plots

**★★★★ DONE (2026-08-09), validated in production twice (2026-08-09 overnight, 2026-08-10
extension).** Pure I/O and matplotlib helpers, no physics of its own:
- `save_snapshot`/`load_snapshot`/`load_all_snapshots` — the full `SimulationState` (mass
  grid + every field + $t$) plus a freshly-computed `is_convective` Schwarzschild mask
  (center point recorded `True` by convention, since `grad_radiative` has a removable $0/0$
  there — matches `diagnostics.py`'s own convention).
- `plot_evolution_curves` — $r_\text{surface}(t)$, $T_\text{center}(t)$, $L_\text{surface}(t)$
  across a sequence of snapshots; the primary Kelvin-Helmholtz contraction-track plot.
- `plot_opacity_regime_map` — $\kappa(m)$ colored by the active Bell & Lin regime, per
  snapshot; the one genuinely new plot type (per-snapshot structure/convective-zone plots
  reuse `diagnostics.py`'s existing functions directly).
- `generate_all_plots(snapshot_dir, output_dir, profile_snapshot_indices=None)` — the module's
  own exit criterion: rebuilds every plot from `.npz` files on disk alone, no re-solve.
  Demonstrated for real (not just in principle) when the 2026-08-09 overnight run's plotting
  step crashed on a missing output directory *after* the physics had already finished — every
  snapshot survived, and every plot was regenerated from disk with zero re-solve.

**Real-world resumability test, 2026-08-10**: `extended_run_10gyr.py` loaded
`snapshot_00077.npz` from a completed run and passed it straight into a fresh
`time_stepper.run()` call to extend that same trajectory to a longer time budget — the first
time this module's snapshots were used to *resume* a live production run rather than only to
*recover* one after a crash or to *plot* after the fact.

**Known minor debt, not yet worth fixing**: `_snapshot_path`'s zero-padded filename numbering
restarts at 0 for each `run()` call — a resumed run's own snapshots (e.g.
`snapshots_10gyr/snapshot_00000.npz` onward) therefore do NOT carry absolute step numbers
consistent with the run it resumed from; only the `t` field inside each file (used by all the
plotting functions) is authoritative. Fine for now since nothing indexes snapshots by
filename number across a resume boundary, but worth a `step_offset` parameter if resuming
becomes routine rather than occasional.

### `ReadMe.txt`

Empty placeholder, unchanged.

---

## 3. Diagnostic outputs already on disk

The plots below were generated by earlier `python validation.py` runs, **before**
`bvp_solver.py`'s premise change this session. They reflect Premise 1 (the diffuse,
isothermal $t=0$ state) and are now stale — kept on disk for historical reference only,
pending regeneration once Sub-task 5 is unblocked and `validation.py` is repaired.

- `opacity_transitions.png`, `opacity_profile_preview.png`, `odes_profile_check.png` — all
  independent of `bvp_solver.py`'s premise (synthetic/hand-built profiles), still current.
- `static_structure_t0.png` — **stale.** Shows Premise 1's isothermal, $L\equiv0$, ~13 AU
  structure with a hard-coded title reflecting that. Needs regeneration once Sub-task 5 is
  unblocked (and the check's title string updated to match whatever the final structure
  looks like).
- `mass_reconstruction_check.png` — **stale** (same reason; the underlying check's *logic*
  is regime-independent and should transfer once re-run against a new structure).
- `bootstrap_time_derivatives.png` — **stale and obsolete**: shows the now-removed
  homologous bootstrap's `dT/dt`, `dP/dt`, `dL/dm`. Should be deleted once Check 30 and its
  plot function are removed from `validation.py`, not regenerated.

---

## 4. Validation Suite — what each check confirms, and current status

**★ 2026-08-08 update, read this first — the table below is a stale historical snapshot,
kept for its per-check descriptions, not current status.** `validation.py` now has 37
checks (not 32). Running the full suite end-to-end (`python validation.py`, plus an
isolated per-check sweep to see past the first crash) gives the actual current picture:
**every check passes except two**, both confirmed pre-existing and unrelated to any of this
project's recent work (checked via `git diff` / cross-reference against when each was last
touched, not assumed):
- **Check 17** (`check_stellar_odes_matches_constant_density_analytic_profile`): `dr/dm`
  disagrees with its analytic target by ~17x. `odes.py`/`eos.py`/`gradients.py` are
  untouched by the 2026-08-08 `bvp_solver.py` promotion (confirmed via `git diff`) — likely
  dates to the `config.MU`/`GAMMA` atomic-composition correction (2026-08-07) never having
  been re-validated against this specific check.
- **Check 23** (`check_static_structure_hydrostatic_balance`): relative error 1.78e-2 against
  its 1e-3 tolerance. This table's own row for it (below) already flagged, before
  2026-08-08, that it was "not re-run since the rewrite to confirm" — genuinely never
  verified passing since an earlier rewrite, not a regression from anything done today.

Both are flagged, not fixed — out of scope for whatever session found them (this table's
own vintage predates both). Checks 19, 24, 26, 27, 33-37 (state below as "broken"/"not yet
existing") are now fixed/added and passing — see §1/§5's 2026-08-01/2026-08-07/2026-08-08
entries for each.

`validation.py` contained 32 checks as of the ORIGINAL writing of this table below. **As of
that writing, running it was expected to fail** — the checks below are listed as they
existed in the file at that time (this table describes what each one *tests*), but several
no longer matched `bvp_solver.py`'s actual behavior after that session's premise change.
Status flags added inline at that time; unflagged checks were believed to still pass (they
test regime-independent physics-module building blocks — EOS, opacity, gradients, ODE-RHS
mechanics — never the specific $t=0$ solve).

| # | Check | Confirms | Status |
|---|---|---|---|
| 1–18 | EOS / opacity / gradients / ODE-RHS mechanics (see below for full list) | Pure unit/physics-module checks against hand-picked or synthetic inputs | Unaffected, believed passing |
| 19 | `check_boundary_conditions_residuals` | `boundary_conditions.boundary_conditions()` residual indexing/signs, including the new nonlinear (in $T_b$) radiative-flux formula | **Fixed this session** for the new formula; passing |
| 20 | *(not used — historical numbering gap, not a current check)* | — | — |
| 21 | `check_nebula_conditions_match_mmsn_at_50au` | `config.T_NEB`/`P_NEB` against an independently-derived Hayashi MMSN disk midplane at 50 AU | Still passes; framing (ambient conditions the compact object radiates against) is still valid under Premise 2 |
| 22 | `check_envelope_mass_is_bonnor_ebert_subcritical` | `M_TOTAL < M_BE(T_neb, P_neb)` | Assertion still true and still passes, but its **docstring is stale** — it no longer explains why $t=0$ is isothermal (Premise 1); it now explains why a diffuse, quasi-statically-evolved clump was never viable at all (motivating Premise 2). Needs a docstring rewrite, not a logic change. |
| 23 | `check_static_structure_hydrostatic_balance` | Eulerian hydrostatic balance on `solve_static_structure()`'s output | Logic is generic and should still pass against the new structure — **not re-run since the rewrite to confirm** |
| 24 | `check_static_structure_isothermal_and_monotonic` | Hard-asserts `T≡T_NEB`, `L≡0` | **Broken.** These assertions are the literal Premise-1 result; the new structure is neither isothermal nor $L=0$. Needs a full rewrite (PLAN.md Sub-task 5) once a final structure exists. |
| 25 | `plot_static_structure_profile` | Saves `static_structure_t0.png` | Runs, but its hard-coded title string still says "isothermal at T=50K, L=0" — **stale**, and the plot itself is now the Premise-2 structure (mismatched title vs. content) |
| 26 | `check_virial_balance_pressure_confined` | Pressure-confined virial form | **Broken/wrong formula for Premise 2.** A compact, self-gravitating structure with negligible surface pressure needs the *standard* unconfined virial form instead (PLAN.md Sub-task 6) |
| 27 | `check_static_structure_opacity_regime_distribution` | Asserts 100% of grid in the coldest opacity regime | **Broken.** A hot-center-to-cool-surface structure should span multiple regimes; this assertion is the literal opposite of the expected Premise-2 behavior |
| 28 | `check_mass_reconstruction_matches_lagrangian_grid` | Continuity-equation self-consistency | Logic is regime-independent, data-source only depends on `solve_static_structure()` — likely still passes, **not re-run to confirm** |
| 29 | `plot_mass_reconstruction_error` | Saves `mass_reconstruction_check.png` | Runs; plot is stale (§3) |
| 30 | `check_bootstrap_time_derivatives_are_physical` | The now-removed homologous bootstrap's positivity/analytic-formula/energy-cross-check | **Obsolete.** Tests machinery (`config.T_KH_BOOTSTRAP_S`, the bootstrap dispatch) that no longer exists under its old name/role — will raise an `AttributeError` or similar, not just fail an assertion. Needs removal, not a fix. |
| 31 | `check_finite_difference_time_derivatives_and_interpolation` | `compute_time_derivatives`'s **non-bootstrap** branch, against fully synthetic states | Unaffected by anything this session — this branch is retained and this check's logic is untouched |
| 32 | `plot_bootstrap_time_derivatives` | Saves `bootstrap_time_derivatives.png` | **Obsolete**, same reason as Check 30 — needs removal |

**Checks 1–18 in full** (unaffected, retained for reference): ideal gas EOS/density
inversion, hydrostatic equilibrium and continuity dimensional checks, adiabatic
gradient/$c_p$ limits, Bell & Lin regime table/transition/continuity/ordering/vectorization
checks, a synthetic opacity profile preview, Schwarzschild convection-trigger and
radiative/convective-limit checks, a full opacity-regime sweep for `grad_radiative`, a
non-positive-$\kappa$ guard check, and `stellar_odes`'s shape/finiteness/sign and
constant-density-analytic-profile checks. None of these depend on `bvp_solver.py`'s $t=0$
construction or the bootstrap machinery.

**Next validation work** (deferred until Sub-task 5 is unblocked, per PLAN.md): fix Checks
22 (docstring), 24 (full rewrite), 25 (title string), 26 (virial formula), 27 (multi-regime
assertion); remove Checks 30 and 32 (and their `plot_bootstrap_time_derivatives`
counterpart); confirm 23 and 28 still pass against the final structure; propose new checks
for `solve_timestep`'s organic (no-bootstrap) evolution once that's validated.

---

## 5. Change Log

### 2026-08-14 — ★★★★ Overnight high-resolution Phase 1 run: first attempt crashed (a real, useful finding), second succeeded cleanly

`run_scripts/run_phase1_high_res_overnight.py` (own output dir, same physical constraints as
the baseline run - halt at 1900K, constant composition). Also added `time_stepper.run`'s
`max_wall_clock_s` parameter (optional, default `None`, zero effect on existing callers) as a
compute-budget safety net for unattended runs, and made the run script generate plots from
whatever snapshots exist even if the run itself raises, rather than leaving nothing to review.

**First attempt** raised `BVP_MESH_N_GRID_POINTS` (2000->3000), `BVP_MAX_NODES` (80000->
120000), and `BVP_COLLOCATION_TOL` (1e-6->1e-7) alongside the free resolution wins. It crashed
in `relax_initial_state`'s stage 1 - a step that had not failed once, at any $T_\text{center}$,
anywhere in this project's session-long investigation. The iteration trace showed why: the
residual already reached ~6-9e-7 (under the OLD 1e-6 tolerance) by iteration 8-9, then kept
refining past that chasing the tighter 1e-7 target, and that extra refinement is what diverged
(node count past 55,000, residual back up to 5e-3) - direct evidence that asking for more
precision than this system can stably deliver, near the same convective-saturation degeneracy
under investigation (§5's 2026-08-13 entries), makes things worse, not better.

**Second attempt** reverted `BVP_MESH_N_GRID_POINTS`/`BVP_MAX_NODES`/`BVP_COLLOCATION_TOL` to
config.py's own proven defaults (untouched, not just reset), keeping only `N_GRID_POINTS`
200->1000 and `GRID_OUTER_REFINEMENT` 1e-4->1e-6 (both OUTPUT-grid-only, zero solver cost) and
`ADAPTIVE_DT_GROWTH_FACTOR` 1.3->1.15 (time-stepping only, doesn't touch solve_bvp's mesh
economy). Converged cleanly, 43 steps, reaching `PHASE1_T_CENTER_HALT` at
$T_\text{center}=1938.0$K, $r_\text{surface}=193.16\,R_\text{Jup}$, $t=2804.6$ yr - 44 snapshots
and full plots in `outputs/diagnostic_plots/run_Phase1_high_res_overnight_20260813/`.

Most recent first. Each entry: what was done, and the physical/architectural reasoning.
Entries below marked **[SUPERSEDED]** describe conclusions that later investigation
overturned — kept rather than deleted because the reasoning inside them (numerical
findings, derivations, literature checks) remains accurate and load-bearing for
understanding *why* later decisions were made; only their final conclusion no longer holds.

### 2026-08-13 (still later) — ★★★★★★ Deliberate stress test past PHASE1_T_CENTER_HALT: found the genuine crash at step 12 (T_center~2798K), AND a physically important finding along the way — r_surface reverses and RE-EXPANDS above ~1900K, not just a numerical curiosity

**Explicit request**: push past the 1900K target to see where the solver actually gives out.
`run_scripts/run_phase1_stress_test_past_halt.py` resumes from the baseline run's own final,
successfully-halted state (`Phase1_baseline_rerun_20260813/snapshot_00035.npz`, T_center=
1923.684K, t=1518.6 yr) with `PHASE1_T_CENTER_HALT` overridden to 10000K (effectively disabled,
runtime-only — `config.py`'s persisted 1900K default is untouched). **Composition stays
artificially frozen throughout** (`USE_H2_RECOMBINATION_PHYSICS=False`, unchanged) — nothing
below is real Phase 1 physics past ~1900-2000K; it is a numerics-only characterization of the
current solver, exactly as intended.

**Physical finding, not just a numerical one:** steps 1-11 (T_center 1959K -> 2798K, t=1519 ->
2043 yr) all converged (three needed one dt-halving retry each; failure signatures matched the
2026-08-13 entries below - "max mesh nodes exceeded" on the direct attempt, occasionally
"Singular Jacobian... iteration 1" on the alpha=0.5 continuation rung). But `r_surface`, which
had been monotonically contracting for the entire run up to this point, **bottoms out at
~179 R_Jup around t=1800 yr and then turns around and re-expands to ~188 R_Jup** while
T_center keeps climbing and accelerating (superlinear: 1959->2002->2057->2126->2176->2237->
2320->2418->2530->2654->2798K over 11 steps) — confirmed directly in `evolution_curves.png`
(`outputs/diagnostic_plots/run_Phase1_stress_test_20260813/`), not a plotting artifact. This is
the expected signature of forcing a fixed-composition ($\Gamma=1.4$ constant, no latent-heat
sink) ideal gas through the temperature range where real hydrogen would be dissociating and
absorbing the contraction's released gravitational energy: with that sink disabled, the energy
goes straight into thermal pressure support instead, and the star unphysically re-expands
rather than continuing to collapse. **This is a direct, visible confirmation of why
`PHASE1_T_CENTER_HALT=1900K` is a physically-motivated boundary, not an arbitrary numerical
safety margin** - past it, the model's own held-fixed assumptions produce qualitatively wrong
behavior well before the numerics themselves fail.

**The genuine crash**, step 12, attempting to advance from t=2043.4 yr, T_center=2797.600K,
r_surface=188.07 R_Jup (step 11's converged state): 7 total solve attempts (the original
proposed dt=27.401 yr, then `STEP_RETRY_SHRINK_FACTOR=0.5` applied 6 times per
`STEP_RETRY_MAX_ATTEMPTS=6` — 27.401, 13.700, 6.850, 3.425, 1.7125, 0.856, 0.428 yr), **all
seven failed**. Signatures varied across attempts: most via "maximum number of mesh nodes
exceeded" on the direct $\alpha=1$ attempt (residual reaching `nan` after 40-55 iterations,
node count running away past 77,000-79,000 against `BVP_MAX_NODES=80000`); one attempt
(dt=1.7125 yr) via "Singular Jacobian encountered when solving the collocation system on
iteration 1" on the $\alpha=0.5$ continuation rung (same signature as the entries below); and,
notably, in the smaller-dt attempts **even the $\alpha=0$ pure-adiabat continuation rung — which
had converged cleanly in every prior failure this project has seen, including every case in the
entries below — itself started failing** the same "max mesh nodes exceeded" way. The final,
fatal attempt (dt=0.428 yr, the smallest tried, 1/64th of the original step): direct $\alpha=1$
ran 55 iterations to `nan` residual (239.8s), alpha-continuation's $\alpha=0$ rung then ALSO
failed (4 iterations, residual non-monotonic — 2.16e+02 -> 1.60e-01 -> 2.35e-01 -> 5.86e+00 —
node count 53974 with (107946) more flagged as needed, decisively over the 80000 cap). With no
attempt left, `time_stepper.run`'s retry loop re-raised, crashing the script:

```
RuntimeError: bvp_solver: solve_bvp failed to converge even via alpha-continuation
(status=1, The maximum number of mesh nodes is exceeded.) after 9.9s - not a genuine solution
```
raised from `_solve_structure_bvp` (`bvp_solver.py:1337`), via `solve_timestep`
(`bvp_solver.py:1483`) and `time_stepper.run` (`time_stepper.py:134`) — full traceback captured
in the run log. Exit code 1 (a genuine, unhandled crash, not a graceful halt).

**Read together with the entries below:** the $\alpha=0\to0.5$ singular-Jacobian mechanism is
still present here, but by T_center~2800K it is no longer the ONLY or even the dominant failure
mode — the pure-adiabat $\alpha=0$ rung, previously bulletproof, is now failing too. This is
consistent with (not yet proof of) the same near-global convective-saturation mechanism simply
getting more severe as $T_\text{center}$ climbs further past the point that mechanism was first
characterized at (~1600-1900K) — plausible given the star's superadiabaticity
($\nabla_\text{rad}/\nabla_\text{ad}$) was already 2-3 orders of magnitude at the earlier,
cooler wall. Not chased further — no speculative fix applied, per explicit instruction; this
was a deliberate, bounded stress test to locate the crash point, which it did. 11 pre-crash
snapshots and plots (evolution curves, structure/convective/opacity-regime profiles at t=1519,
1802, 2043 yr) are saved in `outputs/snapshots/Phase1_stress_test_20260813/` and `outputs/
diagnostic_plots/run_Phase1_stress_test_20260813/` for reference.

### 2026-08-13 (later) — ★★★★★★★ FIRST clean full Phase 1 run reaches PHASE1_T_CENTER_HALT (1900K) — the singular-Jacobian wall recurs twice more but is fully absorbed by the existing step-retry mechanism, not fixed

**Directive**: drop the other machine's non-determinism data, run a completely fresh $t=0$
Phase 1 baseline (`run_scripts/run_phase1_baseline_rerun.py` — identical to `run_phase1_first_
core.py`, own output directory to avoid mixing with the earlier partial run's stale snapshots)
under the current codebase, and report the exact crash state. **It did not crash.** Full,
physically clean run from $t=0$ to `PHASE1_T_CENTER_HALT`: 35 steps, $t=0\to1518.6$ yr,
$r_\text{surface}=558.6\to186.4\,R_\text{Jup}$, $T_\text{center}=654.9\to1923.7$ K (crossing the
1900K target between steps 34 and 35). 36 snapshots + evolution/profile/convective-zone/
opacity-regime plots generated in `outputs/diagnostic_plots/run_Phase1_baseline_rerun_20260813/`.
This is the first time this project has crossed the Phase 1 finish line.

**The wall investigated in the entry below recurred, twice, exactly as characterized there —
and was automatically absorbed, not resolved:**
- **Step 33** (attempting $dt=81.49$ yr from $T_\text{center}=1823.6$K): direct $\alpha=1$
  attempt failed via "maximum mesh nodes exceeded" (235.8s, NaN residual runaway, matching the
  earlier iteration-21-onward signature). Alpha-continuation's $\alpha=0$ rung converged cleanly
  (10 iter, 5236 nodes). $\alpha=0.5$ failed on its very first Newton iteration: **"Singular
  Jacobian encountered when solving the collocation system on iteration 1"** (max relative
  residual reported as `nan`/large, elapsed 1.8s). `time_stepper.run`'s step-retry caught the
  resulting `RuntimeError`, halved $dt\to40.75$ yr, and the retry converged directly (13
  iterations, 3902 nodes, no continuation needed) — step 33 final: $T_\text{center}=1853.6$K.
- **Step 35** (attempting $dt=68.86$ yr from $T_\text{center}=1890.4$K): identical signature —
  direct $\alpha=1$ failed the same way (65 iterations, node count run away to 78,057, "Number
  of nodes is exceeded"), $\alpha=0$ converged cleanly (10 iter, 5425 nodes), $\alpha=0.5$ failed
  with the same exact message on iteration 1 (elapsed 1.9s). Step-retry halved
  $dt\to34.43$ yr; the retry converged directly (12 iterations, 3865 nodes) at
  $T_\text{center}=1923.684$K — which crossed `PHASE1_T_CENTER_HALT=1900.0` and ended the run.

Both failures show the same diagnostic footprint as the 2026-08-13 instrumentation below: before
the singular-Jacobian message, `implicit_rhs_vectorized`'s own diagnostic print shows the
$\alpha=0.5$ trial's extreme/non-finite $(P,T)$ values spreading from a few scattered mesh
points to **the entire mesh** ($m/M_\text{TOTAL}\in[10^{-6},1.0]$, all ~5400 points) across
successive internal trial evaluations, with $\ln P,\ln T$ reaching $\sim10^{19}$–$10^{20}$ —
smaller in magnitude than the earlier instrumentation's $\sim10^{134}$–$10^{136}$ figures, but
the same qualitative runaway-then-singular pattern, now confirmed at a different $T_\text{center}$.

**What this means, stated plainly:** the $\alpha=0\to0.5$ Jacobian singularity identified in the
entry below is real, reproducible, and now confirmed to recur at multiple, DIFFERENT
$T_\text{center}$ values (not one fixed threshold) as the star heats through this range — but in
this run, the existing `STEP_RETRY_MAX_ATTEMPTS`/`STEP_RETRY_SHRINK_FACTOR` safety net (built
2026-08-12 for a different, unrelated class of transient failure) happened to be sufficient to
route around it every time, succeeding on the very first retry (dt halved once) in both cases.
This is NOT the same as the underlying Jacobian issue being fixed — it is being silently
absorbed by a mechanism that was not designed for it. No speculative fix was applied here, per
explicit instruction; this is a clean data point for the ongoing discussion, not a resolution.

### 2026-08-13 — ★★★★★ The 1561K wall's real cause found (parallel session) and confirmed live; a NEW ~1620-1660K wall instrumented, root mechanism identified as near-global convective saturation, not a local defect

**Context**: this entry picks up directly from 2026-08-12's honest "wall not yet diagnosed" close.
Two things happened in parallel: a second Claude session on the user's other machine kept
working the same repo independently (own commits, reconciled here via `git log`/`git show`
rather than assumed), and this session ran three targeted fix attempts of its own against the
1561K wall. Both threads are recorded here so the reasoning isn't lost.

**This session's three fix attempts against the 1561K wall (all tested in isolation against the
exact failing (state, dt) before being judged, per the project's sterile-before-wet discipline)**:
1. NaN-safe alpha-blend (`implicit_rhs_vectorized`/`implicit_rhs_jacobian`): traced the failure
   to IEEE float pathology, not a physics bug - `dT_dm_real` (needing opacity/`grad_rad` via
   `odes.stellar_odes`) was computed *unconditionally* regardless of `alpha`, so a NaN/inf real-
   gradient value survived multiplication by `alpha=0` (`0.0*nan=nan` in IEEE754, not `0.0`),
   corrupting the pure-adiabat fallback that was supposed to be immune to it. Fixed with an
   explicit `np.where(np.isfinite(...), ...)` fallback in both the residual and its Jacobian
   (verified against finite differences to 1.16e-10, zero validation regression). **Necessary
   general-correctness fix, kept - but alone did not resolve the wall**: confirmed dt-independent
   (5.0/2.5/1.25 yr all failed identically), ruling out NaN-corruption as this wall's primary
   cause.
2. Opacity transition width widening (`OPACITY_TRANSITION_SMOOTH_WIDTH_DEX`): failed *worse* -
   singular Jacobian, residuals to 1e35-1e86, non-finite values spread across nearly the whole
   star. **Rejected.**
3. Outer mesh refinement: residual *grew* across iterations even on a 2x denser/wider mesh - the
   signature of genuine non-convergence, not under-resolution. **Rejected.**

**The parallel session's fix (commit `8531c5f`) - confirmed today as the actual root cause of the
1561K wall.** `eos.latent_heat_capacity` and `eos.mean_molecular_weight_inv_derivative` (plus
`bvp_solver._h2_transition_derivatives`) were injecting the full H2-dissociation latent-heat
logistic spike (up to ~16x nominal $c_p$ near T=2500K) into the energy equation *even with
`USE_H2_RECOMBINATION_PHYSICS=False`* - Phase 1's whole point is to stay strictly molecular, and
this flag was supposed to disable that physics track entirely but didn't. Fixed by gating all
three functions on the flag (return zero derivative contributions when off). Verified today, live,
in this environment (not just trusted from the other machine's report): resuming from the
T_center=1560.89K snapshot, the exact step that used to explode as "maximum mesh nodes exceeded"
(>80,000 nodes) at every alpha rung now converges cleanly in 13 iterations / 3664 nodes, and a
second step converges equally cleanly to T_center=1594.02K. **The 1561K wall is closed.**

Also merged from the same commit and confirmed present: `boundary_conditions.py`'s stale,
unsynced module-level `boundary_conditions()` wrapper (hardcoded `config.MU` instead of
`eos.mean_molecular_weight(T)`) removed; `validation.py`'s Check 19 updated to a live
`_live_boundary_residual` using the real solver physics; `_safe_solve_bvp`'s exception handling
narrowed from bare `except Exception` to `(AssertionError, FloatingPointError,
np.linalg.LinAlgError)`; repo reorganized (`outputs/` consolidated and gitignored, run scripts
moved to `run_scripts/`, dead files `PLAN_BVP.md`/`bvp_experiment.py`/
`bvp_solver_shooting_archive.py` deleted).

**A new wall appeared one step later, ~T_center=1620-1660K.** Wrote an instrumentation script,
`run_scripts/diag_singular_jacobian.py`, that monkey-patches `scipy.integrate._bvp.prepare_sys`
to intercept the *exact* sparse collocation Jacobian `solve_bvp`'s own Newton iteration
factorizes on every call (not a hand-reconstruction of it), run against the real failing
(state=`Phase1_deep_diag/snapshot_00002.npz`, T_center=1594.02K; dt=6.05yr, the real next
adaptive step). Findings:

- **Non-determinism: not reproduced here.** Two identical runs in this environment produced
  bit-for-bit identical first-iteration Jacobians (`max|J1-J2|=0.0`). The 4th/5th-decimal
  differences reported from the other machine are real but likely environment-specific (BLAS
  backend, thread count, or scipy version divergence between the two machines) rather than a
  property of the physics/solver itself - worth a `np.show_config()`/`scipy.__version__` diff
  between the two machines before treating non-determinism as a structural clue.
- **Signature confirmed, but with a precursor the report didn't mention.** The direct alpha=1
  attempt does *not* fail via singular Jacobian - it converges steadily for 20 iterations then
  goes to `nan` residual at iteration 21 and runs away in mesh refinement (up to 77,516 nodes)
  until the node cap aborts it. *Then*, the alpha-continuation fallback's alpha=0 (pure adiabat)
  rung converges cleanly (11 iterations, 2794 nodes - notably clean, unlike the old 1561K wall
  where alpha=0 also failed). The reported "Singular Jacobian encountered ... iteration 1" is the
  *next* rung, alpha=0.5, confirmed exactly: `splu` on the captured matrix itself throws "Factor
  is exactly singular" - a genuine, not just ill-conditioned, matrix.
- **Location: not localized to m/M≈0.75.** Direct measurement of $d(\nabla_\text{eff})/
  d(\nabla_\text{rad})$ (the smoothed-Schwarzschild-switch derivative) across four windows
  (m/M∈[0.17,0.33], [0.42,0.58], [0.67,0.83], [0.82,0.98]) on the alpha=0 solution shows the
  *same* signature everywhere: $\nabla_\text{rad}$ exceeds $\nabla_\text{ad}$ by 2-3 orders of
  magnitude (150-355 vs. ~0.286) at every point sampled, pinning $d(\nabla_\text{eff})/
  d(\nabla_\text{rad})$ to ~1e-5-1e-3 (deep convective saturation - $\nabla_\text{eff}\approx
  \nabla_\text{ad}$, essentially decoupled from $L$) across the *entire* envelope, not a local
  pocket. This is the same rank-deficiency mechanism this project's own shooting-method era
  documented ("100% convective saturation ... decoupling L from the P-T relation"), but
  generalized: Phase 1's diffuse, cool, low-opacity envelope is nearly fully convective almost
  everywhere (physically expected for a First Hydrostatic Core), not marginally convective at
  one radius.
- **Mechanism at the actual failure.** At alpha=0.5's first Newton attempt (Jacobian assembled
  at the alpha=0 solution), scipy's internal trial-step evaluations already show catastrophic
  blowups (trial $|\ln P|$ reaching ~$10^{134}$) at scattered mass coordinates (m/M≈0.30, 0.57,
  0.87, and the near-surface 0.999-1.0 band) before the factorization itself is declared
  singular. Read together with the location finding: introducing the real (alpha>0) $L$-$T$
  coupling for the first time, on top of a structure where that coupling's derivative is
  legitimately ~1e-5 almost everywhere, appears to leave the discrete system too weakly
  conditioned in the $L$ direction for a single 0.5-sized continuation jump to survive - not
  disproven, but not yet confirmed as *the* mechanism either.

**Not yet decided - three options identified for joint review, none implemented:** (a) a finer
alpha ladder through the 0→0.5 jump specifically, since the $L$-$T$ coupling only exists for
alpha>0; (b) `GRAD_EFF_SWITCH_EPSILON_TIMESTEP=2.0` was tuned for Phase 3's near-unity
superadiabaticity and may simply be the wrong scale for Phase 1's 150-355 magnitude range
(Phase-1-local runtime override only, not a `config.py` default change - Phase 3's tuned value
stays untouched); (c) a longer-term architectural option - treat the deeply convective interior
with $\nabla_\text{eff}=\nabla_\text{ad}$ directly rather than blending everywhere, matching how
real stellar-structure codes detect the radiative-convective boundary explicitly instead of
smoothing across the whole star. Deliberately not executed pending discussion, per explicit
instruction to gather data and decide together rather than fix unilaterally.

### 2026-08-12 — ★★★★★★ Phase 3 paused; pivot to Phase 1 (First Hydrostatic Core) - a real, physically clean contraction achieved to ~73% of the T_center target, honest account of the remaining wall

**PI directive, explicit and immediate**: stop debugging Phase 3's step-5 wall (see §1's brief
note above and §5's 2026-08-11 entry closing note - not repeated here), pivot to Phase 1 to
guarantee a working, physically accurate deliverable before the thesis deadline. PLAN.md
already scopes Phase 1 exactly (§"Formation Scenario and Scope": diffuse, fully molecular
first core, $R\sim10^2$-$10^3\,R_\text{Jup}$, ends at $T_\text{center}\sim2000$K where H2
dissociation triggers the out-of-scope Stage 2 dynamical collapse).

**Architectural gap found and fixed, not routed around**: `bvp_solver._adiabatic_center_guess`'s
Lane-Emden seed is hard-coded to the $T=0$ electron-degenerate $n=1.5$ polytrope - a pure
function of fundamental constants, independent of `T_CENTER_INITIAL`/`MU`/`GAMMA`. Verified
directly: running `solve_static_structure()` with molecular `MU=2.34`/`GAMMA=1.4` does not
fail - it SILENTLY converges to the wrong physical branch (`r_surface=3.27 R_Jup`, a
plausible-looking `3.6e-3` residual), because the geometric-expansion bracket search starts
essentially on top of the compact/degenerate root and finds it within 5 iterations, 5-6
decades from the true diffuse root also present in `mass_error(P_center)`. Fixed with a new
`_adiabatic_center_guess_ideal_gas(T_center)` (thermally-set polytropic constant
$K=C T_\text{center}\rho_c^{-1/n}$ instead of the fixed degenerate $K_1$ - re-derives the same
$\alpha^2\propto\rho_c^{-1}$ collapse for ANY $n$, not just $n=1.5$; self-consistency verified
analytically AND numerically exact: $P_\text{center}$ reduces identically to the plain ideal
gas law at the seed's own $(\rho_c,T_\text{center})$). `solve_static_structure(use_ideal_gas_
seed=True)` selects it; default `False` keeps Phase 3's proven path byte-for-byte unchanged.

**A second, genuine, dimensionally-wrong bug found during calibration**: `solve_static_
structure`'s `brentq` call used `xtol=config.M_TOTAL*1e-12` (~1.9e18) as an absolute tolerance
on `P_center` (a pressure) - astronomically larger than `P_center` in EITHER regime (~1e4
diffuse, ~1e11-1e12 compact), so `brentq`'s stopping criterion was always dominated by this
oversized `xtol`, causing premature convergence after ~1-2 bisections regardless of how tight
`rtol=BVP_TOL` was set. Fixed with `config.STATIC_STRUCTURE_BRENTQ_XTOL=2e-12` (a genuinely
negligible absolute floor, scipy's own default order of magnitude), letting `rtol` do the real
work. Residuals dropped from marginal ~1e-2 (right at `STATIC_STRUCTURE_RESIDUAL_TOL`'s edge)
to ~1e-10/1e-11 (essentially exact) at EVERY `T_center` tested, **including Phase 3's own
T_CENTER_INITIAL=11500K case** - a shared, strictly-beneficial fix, not something narrow to
Phase 1.

**Calibration**: swept `T_CENTER_INITIAL` directly (cheap - `solve_static_structure` alone, no
relax needed) rather than trust the one available historical anchor point (1200K -> ~300
R_Jup) by extrapolation alone. Found **645K -> r_surface=500.83 R_Jup** - the requested R~500
R_Jup target, essentially exact.

**Timescale** (PI directive, Larson 1969: First Core phase ~$10^4$-$10^5$ yr, not Phase 3's
Gyr scale): `config.T_KH_TIMESCALE_S=1$ Myr would give a seed `dt`~1e4 yr (comparable to the
WHOLE expected process) and `ADAPTIVE_DT_MAX=1e8 yr` is 1,000-10,000x it - both overridden
(runtime-only, hermetically isolated) to timescales appropriate to this phase.

**Hermetic isolation (PI directive), directly verified, not just designed-for**: every Phase-1
value (composition, `T_CENTER_INITIAL`, timescales) is a runtime override local to
`run_phase1_first_core.py`'s own process, exactly matching `run_phase3_validation.py`'s
existing `T_MAX_S` pattern. Confirmed after the fact by direct inspection: `config.py`'s own
file-level `MU`, `GAMMA`, `USE_H2_RECOMBINATION_PHYSICS`, `T_CENTER_INITIAL`, `T_MAX_S`,
`ADAPTIVE_DT_MAX/MIN/GROWTH_FACTOR`, `RELAX_DT_FRACTION` are ALL exactly their pre-existing
Phase 3 values - zero permanent trace of any Phase-1-specific run left in the file.

**Sterile validation (Step 6, before the long run) surfaced two more genuine, fixed issues,
not assumed away**:
- `relax_initial_state`'s stage 1, under its DEFAULT settings (clamp forced off, `dt_relax`=
  `RELAX_DT_FRACTION*T_KH_TIMESCALE_S`=1e4 yr - both tuned for Phase 3), crashed outright for
  this diffuse structure (a raw `np.exp()` overflow reaching `eos.density` uncaught, clamp
  off). Direct, isolated tests found BOTH a smaller relax pseudo-timestep (1000 yr, via a new
  `force_clamp_off_stage1=True` default parameter added to `relax_initial_state` - defaults
  True, preserving Phase 3's exact proven behavior; Phase 1 passes `False`) AND leaving the
  clamp ON (its global default, already fixed/verified 2026-08-11) independently necessary -
  together, first-attempt convergence, no continuation fallback.
- The very first real `solve_timestep` call ALSO needed a gentler seed than the naive 100 yr
  guess (mesh-exploded; 10 yr converged directly) - and `config.ADAPTIVE_DT_MIN=100yr`
  (inherited, unchanged) sat ABOVE that 10 yr seed, forcibly clamping step 2 straight back up
  to the value that just failed, defeating the fix on the very next step - overridden to 10yr.

A clean 10-step sterile run (relax + 10 real steps) then passed with zero failures - the gate
for "push straight through" (PI directive).

**The actual run - genuine progress, then an honest, not-yet-diagnosed wall.** The full run
(`run_phase1_first_core.py`) produced 18 clean, physically beautiful steps (R: 556.8 -> 310.4
R_Jup, T_center: 669 -> 1234K, monotonic, dt growing smoothly under the 1.3x growth cap) before
crashing at step 19 - the raw pre-clamp trial state reached literal `inf` (a linear-solve
overflow, not something the soft clamp's pointwise saturation alone can prevent). **Resumed
three times**, each recovering with a smaller `dt` at the exact failing (state, dt) - verified
directly each time before relaunching, not guessed: 1000 yr (relax) -> 10 yr (first real step)
-> 5-10 yr (recurring pocket near T~1300K, twice) -> 5 yr (pocket near T~1500K). This
recurring "growth-capped dt fails, several-times-smaller dt at the SAME state succeeds every
time" pattern (confirmed 3+ times, not a one-off) motivated a genuine, general robustness
addition rather than continued manual firefighting: **`time_stepper.run` now automatically
retries a failed step with a shrunken `dt`** (`config.STEP_RETRY_MAX_ATTEMPTS=6`,
`STEP_RETRY_SHRINK_FACTOR=0.5` - standard adaptive-integrator step-rejection practice, strictly
additive - a step that already succeeds on its first attempt is completely unaffected). This
DID recover one further step automatically (T_center 1530->1561K) before hitting a
**qualitatively different, persistent wall** at T_center~1561K, r~238 R_Jup: unlike every prior
pocket, this one did NOT resolve even after shrinking `dt` all the way down to 0.078 yr (~28
days, the full 6-retry budget exhausted) or after independently widening
`GRAD_EFF_SWITCH_EPSILON_TIMESTEP` to 5.0 and 10.0 (both still failed, notably even at
`alpha=0` - the pure adiabat blend, before the Schwarzschild-selected gradient is even
involved - pointing away from the convective-switch smoothing as the primary cause, toward
something else not yet identified, possibly the opacity-regime risk flagged before this run
even started: Phase 1's entire envelope, not just Phase 3's thin surface skin, traverses the
historically fragile Metal/Ice-grain-evaporation transitions).

**Final result, honestly reported**: 33 real, physically clean, monotonic snapshots merged
into one continuous trajectory (`diagnostic_plots/run_Phase1_first_core/evolution_curves_
FULL.png`) spanning t=0 to 536 yr, **r_surface: 500.83 -> 238.75 R_Jup, T_center: 654.99 ->
1560.89 K** - genuine Kelvin-Helmholtz contraction, ~73% of the way from start to the 1900K
target (in $T_\text{center}$ terms), R more than halved. **Not yet reached**: the
`PHASE1_T_CENTER_HALT=1900K` halt condition itself - the run stops short at the wall described
above. Given the deadline, this wall was NOT chased further today (two well-motivated levers
tried and directly ruled out - `dt` and `GRAD_EFF_SWITCH_EPSILON_TIMESTEP` - rather than one
more guess); flagged here, honestly, as the concrete next step, not silently left unmentioned.

**Files added**: `bvp_solver._adiabatic_center_guess_ideal_gas`, `solve_static_structure(use_
ideal_gas_seed=...)`, `relax_initial_state(..., force_clamp_off_stage1=...)`;
`config.STATIC_STRUCTURE_BRENTQ_XTOL`, `PHASE1_T_CENTER_HALT`, `STEP_RETRY_MAX_ATTEMPTS`,
`STEP_RETRY_SHRINK_FACTOR`; `time_stepper.run`'s automatic step-retry; `run_phase1_first_
core.py` and three `resume_phase1_from_step*.py` scripts (mirroring `resume_phase3_from_
step3.py`'s established pattern - each resume writes to its own new snapshot directory,
`snapshots_Phase1_first_core[_resumedN]`, so no run's surviving snapshots were ever
overwritten).

**Context.** Launching the full Phase 3 validation run (4.5 Gyr / R_HALT, Sub-task 8b physics
live throughout) crashed immediately: `relax_initial_state` had never been run from a fresh
t=0 state under the new mu(T)/gamma_eff(T) physics before. A same-day investigation (not all
individually logged here - the trail is in this session's history) diagnosed and fixed, in
sequence: (1) a `P=inf` overflow crash in `eos.density`, patched with a hard `np.clip` on the
solver's trial `(lnP, lnT)` before exponentiating (`bvp_solver._safe_exp_state`); (2) the clamp
then caused a NEW failure (singular collocation Jacobian), diagnosed (at the user's explicit
request, framed as an external Senior Numerical Analyst review) as a Jacobian/residual
mismatch - the clamp changes the residual near its boundary but the analytic Jacobian was
never updated to match - fixed with a tiered workaround (`relax_initial_state`'s stage 1 with
the clamp forced OFF, stage 2 with the clamp ON but `fun_jac=None`, forcing scipy's own
numerical Jacobian); (3) the real production run then hit a DIFFERENT failure at step 4 (mesh
explosion, not a crash); (4) cross-referencing the failure location against the Bell & Lin
opacity table's transition temperatures (the user's own hypothesis) found a genuine, unfixed
C1 discontinuity - `kappa(rho,T)` is continuous at each regime boundary by construction, but
`d(kappa)/dT` genuinely jumps (different power-law exponents either side) - fixed with a
smooth partition-of-unity blend (`opacity.bell_lin_opacity_smooth`,
`config.OPACITY_SMOOTH_TRANSITIONS`), which measurably reduced but did not eliminate step 4's
mesh explosion at any tested smoothing width.

**The PI's assessment, verbatim in spirit**: this sequence looked like stacking band-aids on
top of band-aids - fighting `solve_bvp` instead of doing physics - and directed a pause for an
architectural review before any more patches, with two specific priorities: (1) fix the clamp
itself so it never has zero derivative (a proper soft clamp), since a zero-derivative region
gives Newton no restoring force and lets a bad trial wander into nonsense; (2) re-examine
whether `GRAD_EFF_SWITCH_EPSILON_TIMESTEP=0.5` (a documented, deliberately wide numerical
expedient for the Schwarzschild-switch kink, PLAN.md Sub-task 8c) could now shrink, given the
opacity kink - a plausible contributor to the same photospheric region's difficulty - was
fixed.

**Review finding, confirmed by reading the code directly, not from memory**: the clamp-
Jacobian mismatch diagnosed in step (2) above was real, but the TIERED WORKAROUND papered over
it rather than fixing it - `implicit_rhs_jacobian`/`make_bc_jacobian_scaled` still multiplied
by the bare `P`/`T` value everywhere as a stand-in for `d(exp(lnX))/d(lnX)`, correct only
where the clamp's true derivative is exactly 1 (deep inside the safe range). Wherever a trial
point saturated the clamp - exactly the production configuration (`relax_initial_state` stage
2, every `solve_timestep` call) - the Jacobian reported a nonzero sensitivity the residual no
longer had: not just "no restoring force," an ACTIVELY WRONG gradient. This single mechanism
unifies two things that had looked like separate leads: the general step-4 singular-Jacobian
failures, and a specific diagnostic where the center's trial state collapsed to
`T_a=P_a=0, r_analytic~rho_c^(-1/3)` diverging to `~4.2e40 cm` - once a variable crosses the
clamp boundary, the false "still sensitive" signal lets Newton keep pushing it with nothing
pulling it back.

**Fix 1 - the soft clamp** (`bvp_solver.py`, `config.py`). Replaced `_safe_exp_state`'s hard
`np.clip` with a smooth, C-infinity, strictly-monotonic two-sided saturation
(`_soft_clamp`/`_soft_clamp_derivative`), built from the standard numerically-stable softplus
identity (`max(u,0)+log1p(exp(-|u|))`), composed as smooth_max then smooth_min. Same
saturation CENTERS as the old hard clamp (`config.LN_P_CLAMP`/`LN_T_MIN`/`LN_T_MAX`, moved
here from bare `bvp_solver.py` literals - a pre-existing CLAUDE.md gap, fixed in passing), plus
a new width (`config.BVP_SOFT_CLAMP_WIDTH`). **Sterile-verified BEFORE touching any Jacobian**
(new Checks 40/40b): an initial width guess (2.0) FAILED its own identity-match check
immediately - `T_NEB=50K` sits only ~2 widths from `LN_T_MIN=0`, nowhere near the "tens of
widths" margin assumed - corrected to 0.1 (`config.py`'s own comment has the full margin
derivation). Verified: exact match to raw `exp()` across the real operating range
(`rel_err~2e-14`), bounded across the full extreme range including the actual logged failure
values, and the derivative stays meaningfully nonzero for ~30,000 widths past the boundary
(vs. exactly zero at distance 0 for the old hard clamp).

**Fix 2 - propagate the derivative through the Jacobian** (`bvp_solver.py`). Added
`_safe_exp_state_derivatives` (returns `dP/d(lnP), dT/d(lnT)` through the actual clamp) and
threaded it through every chain-rule factor in `implicit_rhs_jacobian`/`make_bc_jacobian_scaled`
that previously multiplied by the bare `P`/`T` value - `J[0,1]`, `J[0,3]`, `J[2,1]`, `J[2,3]`,
`J[3,1]`, `J[3,3]`, `dbc_dza[0,1]`, `dbc_dza[0,3]`, `dbc_dzb[2,1]`, `dbc_dzb[2,3]`,
`dbc_dzb[3,3]` - plus one site the initial derivation missed and only caught by re-deriving
`J[1,1]` by hand (`f1=dP_dm/P`'s own `d/d(lnP)`, previously the bare `-f1`, correct only when
unclamped). **The exact same class of bug was independently found in the smoothed opacity's
own Jacobian contribution**: `_opacity_derivatives` still computed the HARD-switch regime's
derivative unconditionally even though the residual used the smoothed `kappa` when
`config.OPACITY_SMOOTH_TRANSITIONS=True` - derived and implemented the smoothed blend's true
derivative (`opacity.bell_lin_opacity_smooth_derivatives`, product rule through the
partition-of-unity weights' own derivatives, `opacity._regime_weights_derivatives`) and wired
`_opacity_derivatives` to dispatch on the flag. **Extended Check 37** (the standing
finite-difference Jacobian cross-check) to explicitly sample synthetic points inside the
clamp's saturation zone (it never did before - exactly how this bug stayed invisible); **new
Check 41** verifies the opacity derivative fix at a handful of representative points. Both
pass (Check 37's saturation-region entries: `1.7e-6`/`2.5e-9`, well under
`config.JACOBIAN_VERIFY_TOL=1e-4`).

*A genuine numerical-analysis detour, worth recording*: a first, blind 20,000-point random
`(rho,T)` sweep to verify the opacity derivative produced apparent errors up to `1e299` -
investigated directly rather than dismissed, and traced every single time to an FD-
VERIFICATION-methodology artifact, never a math bug: (a) comparing the FULL composed
derivative against finite differences of the total weighted sum can bury a correct but
sub-dominant regime's contribution under a dominant term's own float64 precision floor
(confirmed by isolating that one regime's weight against FD of ITS OWN value alone - matched
to `4e-9`); (b) at `rho` below this problem's real ~1e-8 g/cm^3 floor, `d(weight)/d(rho)`
spans such extreme dynamic range that no single fixed FD step size is well-conditioned
everywhere (confirmed via a direct step-size convergence sweep at one such point - FD
converged cleanly to the analytic value for `h_rel>=1e-5`, then diverged from round-off as `h`
shrank further). The permanent Check 41 instead uses a handful of well-chosen points (mirrors
Check 39's own methodology) rather than a blind sweep - the sweep was a development tool, not
something worth enshrining as a check that fails on its own methodology's edge cases.

**Fix 3 - `relax_initial_state`'s tiered clamp/Jacobian staging, re-tested not just
simplified.** With the Jacobian now consistent everywhere, tested whether the composition-jump
problem that originally motivated the two-stage structure was itself just the clamp bug in
disguise: a single-stage attempt (real physics, clamp on, analytic Jacobian, from a fresh
`solve_static_structure()` output) still fails the same way (singular Jacobian) - genuinely
separate difficulty, two-stage structure stays. But re-tested each stage's OWN special-casing
against the fixed solver rather than assuming: stage 1 (old constant-mu physics) with the
clamp forced ON now REGRESSES even with the correct Jacobian (a single bad Newton step can
still outrun any finite-width clamp's restoring gradient, ~75 log-units out for
`BVP_SOFT_CLAMP_WIDTH=0.1`, not infinitely) - confirmed genuinely independent of today's fix,
stage 1 keeps clamp OFF. Stage 2 (already clamp ON) now converges with the FAST analytic
Jacobian instead of the forced-numerical workaround - verified directly against the numerical-
Jacobian baseline (same 9 iterations, same `r_surface=4.9313 R_Jup`, 4442 vs 4445 nodes, a
trivial mesh-path difference) before trusting the switch. Full `relax_initial_state()` re-run
end-to-end successfully (137s total).

**The actual target: Phase 3 step 4.** Reconstructed the EXACT failing `(state, dt)` from the
saved snapshots (`snapshots_Phase3_recombination/snapshot_00003.npz`, chaining
`time_stepper.select_adaptive_dt` forward through the recorded `t` values to recover
`dt=3.6616e11 s` - verified exactly reproducing each intermediate snapshot's own `t`). Retried
sterile-first (numerical Jacobian): the failure mode CHANGED from "singular Jacobian" to
"maximum mesh nodes exceeded" - direct confirmation the clamp fix eliminated its own failure
class, and what remains is the separate, already-documented `GRAD_EFF_SWITCH_EPSILON_TIMESTEP`
marginal-convection-band issue (PLAN.md Sub-task 8c). Swept that epsilon against this exact
step: **0.5 (unchanged) still fails; 1.0 is actively pathological (43 MINUTES for the direct
attempt alone before also failing - a red flag of real fragility, not just insufficient
smoothing); 2.0 converges directly, fast, cleanly (2.9s, 9068 nodes); 5.0 also succeeds.**
Adopted `GRAD_EFF_SWITCH_EPSILON_TIMESTEP=2.0` (smallest cleanly-converging value found,
same margin-sweep discipline as everywhere else in this project). **Stated plainly: this is
the OPPOSITE of what fixing the opacity kink was hoped to enable** - the marginal band needed
WIDENING, not narrowing, once the clamp/opacity kinks were cleared, which is itself evidence
this band is real physics (a genuinely marginal `nabla_rad~nabla_ad` region), not a numerical
artifact those fixes were expected to shrink - strengthening, not weakening, the case for
PLAN.md's already-scoped Sub-task 8c (mixing-length theory) as the eventual complete fix,
rather than continuing to chase this value step by step. Step 4 now converges via the real
production `bvp_solver.solve_timestep` API: 2.8s, 9094 nodes, `r_surface=4.9337 R_Jup`
(status=0, residual `9.96e-7`).

**Full regression**: `validation.py`'s 44 checks run individually (the standard practice for
seeing past the long-standing Check 17 crash-on-first-failure - PROGRESS.md's own 2026-08-08
entry) - **42/44 pass; the only 2 failures are the exact same pre-existing Checks 17
(`stellar_odes` vs. constant-density analytic profile) and 23 (`solve_static_structure`
Eulerian hydrostatic balance) this project has tracked and re-confirmed unrelated at every
regression pass since 2026-08-08.** Zero new failures.

**Still open**: PLAN.md Sub-task 8c (mixing-length theory) - not started, now more clearly
justified than before this session. `GRAD_EFF_SWITCH_EPSILON_TIMESTEP=2.0` is validated
against step 4 specifically, not a longer chain - the original 0.5 value's own "only validated
for 10 real steps, no proof the marginal band's demands plateau" limitation applies here too,
unresolved.

**Immediately after this entry was written**: `resume_phase3_from_step3.py` launched the
actual continuation (from `snapshot_00003.npz`, the same reconstructed `dt`). Step 1 of the
resume (= the original step 4) converged exactly as verified above (`snapshot_00001.npz` in
the new `snapshots_Phase3_recombination_resumed/` dir). `select_adaptive_dt` then chose a
sharply smaller next `dt` (11,602 -> 945 yr, ~12x drop) and THAT step failed - same "max mesh
nodes exceeded" signature as step 4's original failure, but with a new, more extreme detail:
the diagnostic print showed the raw (pre-clamp) trial `lnP`/`lnT` reaching `~3.3e15`/`~-7e14` -
many orders of magnitude past even the soft clamp's own extended "still has a meaningful
restoring gradient" reach (~75-175 log-units, config.BVP_SOFT_CLAMP_WIDTH's own comment). This
is consistent with, and sharpens, the mechanism already suspected for `relax_initial_state`
stage 1's own clamp-on regression this same session: the soft clamp guarantees the PHYSICAL
output (P, T) stays bounded and the JACOBIAN stays honest, but does nothing to stop the RAW
log-state Newton variable itself from being driven arbitrarily far by a single large step -
once that raw variable is astronomically deep in the saturated region, the (now-honest)
gradient there is so close to zero that recovery isn't guaranteed even though nothing is lying
to the solver anymore. A genuinely new, not-yet-investigated failure, one step further than
where this session's fixes got the run to - not chased further today given the length of this
session; flagged here for the next session rather than patched blindly.

### 2026-08-10 (evening) — ★★★★★★★ Sub-task 8b implemented: H<->H2 recombination physics live in the production solver

**Context**: after Check 38 justified the physics, the user shared a roadmap pivot from the
PI: the immediate next goal is modeling Phase 1 (First Core collapse, 300-1000 R_Jup down to
T_center~2000K triggering the dynamical Phase 2 collapse), with Phase 2 bridged to a revised,
hotter Phase 3 start (possibly ~40,000K instead of the current 13,000K placeholder) via a
future energy-conservation calculation. Sub-task 8b was reframed as dual-purpose: the SAME
chi(T)/mu(T)/gamma_eff(T)/latent-heat mechanism fixing Phase 3's outer envelope is also the
physical trigger (Gamma_1 softening below 4/3) expected to formally end Phase 1. Pushback
given before implementing (agreed, with refinements, not blocking): (1) "formally end Phase 1"
needs an EXPLICIT Gamma_1-averaged halt condition once `gamma_eff(T)` exists, not reliance on
`solve_bvp` failing to converge as an implicit signal - deferred to when the Phase 1 driver is
actually built, not this sub-task. (2) The T-only, fixed-threshold (2000-3000K) chi(T) proxy
is validated ONLY against Phase 3's tenuous outer-envelope density regime (Check 38); Phase
1's core at the point of dissociation is much denser, and a real dissociation equilibrium
constant depends on rho too - flagged as an open risk to re-check once Phase 1 modeling
begins, not assumed to transfer. (3) MLT deferral endorsed, with a refinement: not because the
cloud is "transparent" per se, but because MLT only refines superadiabaticity in an ALREADY-
convective zone (the Schwarzschild criterion already handles convective-vs-radiative without
it) - the newly-relevant feature (a persistent radiative zone, found in the 10 Gyr run) is
genuinely radiative, where MLT doesn't directly apply. (4) Saha (8a) agreed irrelevant for
Phase 1 (ionization energy scale argument), but a future 40,000K Phase 3 needs a FRESH Saha
check against the actual bridged density, not an assumption from T alone - Milestone-0's own
check found ionization negligible at 13,000K specifically because of density suppression.

**Formal plan** (`EnterPlanMode`/`ExitPlanMode`, matching Sub-task 9's precedent) surfaced a
critical scope correction during exploration, before any code was written: **`odes.py`'s
`stellar_odes` is not the live t>0 solver path.** `bvp_solver.py` has its own separate,
hand-derived RHS (`implicit_rhs_vectorized`) and analytic Jacobian (`implicit_rhs_jacobian`,
`make_bc_jacobian_scaled`) that duplicate this physics inline with `config.MU`/`config.GAMMA`
hardcoded throughout - threading `mu(T)`/`gamma_eff(T)` through `odes.py` alone would have
changed nothing about real solver output. This reshaped the plan into a two-step, staged
implementation (physics first via scipy's numerical Jacobian, analytic Jacobian second, gated
by Check 37) specifically to decouple "does the new physics converge and behave sensibly" from
"is the hand-derived Jacobian correct" - the same de-risking discipline this project has used
successfully before (margin sweeps, sterile tests before wiring in).

**Physics implemented** (`config.py`, `eos.py`): a shared logistic `chi(T)` (molecular
fraction, `T_MID=2500K`, `WIDTH=180K`, wide by design per the `GRAD_EFF_SWITCH_EPSILON`
lesson) feeds `mu(T)` (linear interpolation of `1/mu` in `chi` - exact two-state H/H2+He
mixing, anchored to `config.MU` at the atomic limit and a molecular limit derived from
`config.MU_E`'s own `X`, not new independent literals) and `gamma_eff(T)` (same `chi`,
interpolating `config.GAMMA=5/3` to a new `config.GAMMA_MOLECULAR=7/5`). Latent heat
(`config.EPSILON_D_H2`, derived from `D0_H2=4.478 eV` and `config.M_H`) injected as
`latent_heat_capacity(T) = -EPSILON_D_H2*d(chi)/dT`, added to `specific_heat_cp(gamma_eff(T),
mu(T))` in the energy equation. `eos.thermodynamic_delta` extended with an optional
`d_inv_mu_dT` parameter (default 0.0, exactly reproducing prior behavior for any caller that
doesn't pass it) for the mu(T) correction to the EOS's implicit differentiation.

**Step 1 (RHS + boundary conditions, numerical Jacobian)**: `odes.stellar_odes` (4 call sites),
`bvp_solver.implicit_rhs_vectorized`'s own direct `grad_adiabatic` call, `make_bc_scaled`
(center + photospheric `mu`), and `_bvp_solution_to_state` (the OUTPUT `rho` field, which
would otherwise have silently disagreed with the physics actually used during the solve) all
updated. Validated with a real `solve_timestep` call from the cached 10 Gyr snapshot, `fun_jac=
None` (scipy's own finite-difference Jacobian) - converged in 10 iterations, 1.9s,
`Delta r_surface=-5.09%`, same direction and order of magnitude as Check 38's static proxy
(-3.1%), larger as expected since the full system (T, L, r all responding) captures more than
a fixed-(P,T) density perturbation.

**Step 2 (analytic Jacobian) - Check 37 caught a real bug on the first pass, exactly as this
staging was meant to allow.** `_eos_density_derivatives` and `_thermodynamic_delta_derivatives`
extended with the mu(T) correction terms (derived by hand: an extra `rho*K_B*T*d_inv_mu_dT/M_H`
channel in `dF/dT`, and `thermodynamic_delta`'s own numerator gaining a
`rho*K_B*T^2*d_inv_mu_dT/M_H` term, needing chi's SECOND derivative for `_thermodynamic_delta_
derivatives`'s own T-derivative). `implicit_rhs_jacobian` threaded `mu(T)`/`gamma_eff(T)`
through every remaining `config.MU`/`config.GAMMA` reference, plus two genuinely NEW coupling
terms found by tracing the derivation rather than assumed: `d(grad_ad)/dT` contributing to
`J[3,3]` (grad_ad was a true constant before, so this term was correctly absent) and
`-dT_dt*d(c_p_eff)/dT` contributing to `J[2,3]` (same reasoning - c_p was T-independent).
`_effective_gradient_derivative` also needed extending to return BOTH `d(grad_eff)/d(grad_rad)`
and the previously-nonexistent `d(grad_eff)/d(grad_ad)` channel (grad_ad is no longer a true
constant, so `grad_eff`'s OTHER argument now has a real derivative too - the two channels sum
to exactly 1.0, a useful sanity identity confirmed directly). First Check 37 run: **64% relative
error** - a real, decisive failure, not noise. Traced to a dropped `/config.M_H` factor: `P_
ideal = rho*K_B*T/(mu*M_H)`, so its T-derivative's mu(T) term needs `/M_H` too, consistently
missed in THREE places (`eos.thermodynamic_delta`'s own formula, plus its two mirrors in
`bvp_solver.py`'s Jacobian helpers) since all three were derived from the same (flawed) mental
math. Fixed in all three; Check 37 re-run passed at 6.5e-7 (both `fun_jac` and `bc_jac`),
including new test points explicitly forced inside the 2000-3000K transition window (added to
Check 37 itself - the existing random mesh-point sample had only a ~3.6%-per-point chance of
covering that region at all, not a reliable guarantee the new terms would ever be exercised).

**Regression check: zero new failures.** Full `validation.py` suite run end-to-end (skipping
only the already-known-broken Check 17, confirmed unrelated below) - every other check passes,
including one NEW failure surfaced (`check_static_structure_hydrostatic_balance`, Check 23,
1.78e-2 relative error) that turned out to be, on inspection, the EXACT SAME pre-existing,
already-documented failure from the 2026-08-08 `solve_bvp` promotion entry (matching error
magnitude to 3 significant figures) - `solve_static_structure()`'s own adiabatic seed
construction is deliberately untouched by this sub-task (a coarse Newton-iteration starting
guess only, immediately corrected by `relax_initial_state`'s real 4-ODE solve), so its output
is byte-identical to before; this check simply had not been re-run since 2026-08-08 and was
already broken then. Check 17 similarly confirmed unrelated (its own docstring already
describes the failure mode: a test-construction gap, using `config.MU` to invert a target
density that no longer matches `stellar_odes`'s new `mu(T)`-based EOS - compounding the SAME
pre-existing category of staleness the check already had, not a new bug class).

**Also fixed**: `eos.molecular_fraction` used a bare `1/(1+exp(x))`, which overflows (`Runtime
Warning`) at extreme T values reachable by validation.py's own edge-case stress tests -
switched to `scipy.special.expit` (a numerically stable sigmoid), mathematically identical,
silent at every T.

**Final production result** (corrected analytic Jacobian, real `solve_timestep` from the
cached 10 Gyr snapshot, 1e8 yr step): converged in 9 iterations, 1.55s (faster than the
numerical-Jacobian test, as expected). **$r_\text{surface}$: 4.5966 -> 4.3642 $R_\text{Jup}$
($\Delta=-5.05\%$)** for one real timestep - noticeably larger than Check 38's simplified
static proxy (-3.1%), confirming the full self-consistent solve captures a bigger effect than
the pressure/density-only sensitivity test could see. This is the first real evidence that the
outer-envelope recombination physics meaningfully changes the production trajectory, not just
a diagnostic prediction - the natural next step is a full multi-step re-run (comparable to the
existing 10 Gyr trajectory) to see the cumulative effect over the whole simulated history, not
yet done in this pass.

### 2026-08-10 (later same day) — ★★★★★★ Outer-envelope H/H2 recombination sensitivity check: SIGNIFICANT, real implementation now justified

**Context**: reviewing the 10 Gyr run's plots (entry below), the user's own read was that the
model's slow contraction pace implied "the EOS is providing incorrect support" via a
molecular-vs-atomic `mu` error in the hot interior - a direct, testable physical claim. Before
agreeing, `config.py` was checked directly: `MU=1.278` (atomic) and `GAMMA=5/3` (monatomic)
are ALREADY in use, corrected on 2026-08-07 specifically because Stage 3 starts well past H2
dissociation (documented in `config.py`'s own comment history), and a prior Milestone-0 Saha
calculation had already found ionization negligible (peak x~5.9e-4) - so neither the original
diagnosis (still molecular) nor the natural alternative (under-ionized) matched the code.

**The real gap, found by inspecting the actual T(m) profile at t=10 Gyr**: mass-weighted,
8.5% of the envelope is below 2000 K, 26% below 3000 K, 45% below 5000 K - a substantial and
*growing* fraction as the whole structure cools further. Hydrogen there should be recombining
back into H2 (mu rising toward the molecular value ~2.34), but `odes.py` calls
`eos.density`/`grad_adiabatic`/`specific_heat_cp`/`thermodynamic_delta` with the constant
`config.MU`/`config.GAMMA` everywhere - the opposite-direction, opposite-region correction
from what was originally proposed. This region also coincides almost exactly with the
persistent radiative zone found by directly checking the `is_convective` mask across the
overnight run's snapshots (m/M_tot in [0.9548, 1.0], present from shortly before the earlier
`r_surface` "bump" through nearly the entire 4.5 Gyr run) - a genuine, non-trivial temporal
correlation, though the radiative zone clearly outlives the bump itself (open through most of
the subsequent, unremarkable contraction phase too), so a clean "convection breaks, causes the
bump, reopens, contraction resumes" story is too tidy; what the data actually shows is a
longer-lived structural feature coincident with the bump's onset.

**Quantitative case for doing this properly, not as a bare `mu(T)` lookup**: H2's
dissociation/recombination energy (~1.5e12 erg/g of H, computed from D0=4.478 eV/molecule and
`config.M_H`) is **3-16x LARGER than the local thermal energy content** across T=1000-5000K
(computed directly, not estimated). A `mu(T)` fix touching only the pressure/density relation,
without a matching latent-heat term in the energy equation's effective heat capacity, would be
thermodynamically incomplete enough to plausibly bias the result in the wrong direction, not
just omit a minor correction.

**The user's counter-proposal**: rather than a full two-state mass-action equilibrium solver
(root-finding at every BVP node - stiff, and this project has been burned before by exactly
this kind of near-marginal switch, see `GRAD_EFF_SWITCH_EPSILON`'s saga), use an explicit,
smooth logistic `chi(T)` for the molecular fraction, with an analytic `d(chi)/dT` injected as
a latent-heat term in `c_p`. **Reviewed and endorsed, with three additions found necessary
for full self-consistency** by checking `eos.py`/`odes.py` directly (all four EOS calls in
`odes.py` currently use the global constant `config.MU`/`config.GAMMA`, not a per-point
value):
1. `specific_heat_cp` already takes `mu` as a parameter - call it with the local `mu(T)`.
   This captures a real but secondary effect (the "frozen-composition" cp shift from mu(T)
   itself), quantified at ~10-15% the size of the latent-heat term across 2000-3000K -
   smaller than the user's dominant term, but not free to skip once dchi/dT is already
   being computed for the latent-heat piece.
2. **`grad_adiabatic` needs to become gamma_eff(T)-dependent too** - its own docstring
   already flags this gap ("invalid once H2 dissociation begins to lower gamma_eff"). This is
   likely the single most important piece for the specific "floor" question, since softening
   Γ1 in the transition zone is the textbook mechanism for destabilizing a radiative zone
   toward convection - and that zone is exactly where the persistent radiative layer above
   sits. Recommended: share ONE logistic transition between `mu(T)` and `gamma_eff(T)` rather
   than two independently-tuned functions.
3. `thermodynamic_delta`'s implicit-differentiation formula (delta = -(d ln rho/d ln T)_P)
   currently assumes mu fixed when differentiating the EOS; derived the needed correction
   term (`+ k_B*T^2*d(1/mu)/dT / D`, same `D` the Newton solve already uses) for when it's
   actually implemented.

MLT (Sub-task 8c) was also discussed and deprioritized for this specific question: the newly-
identified persistent feature is a genuinely RADIATIVE zone (grad_rad<grad_ad already, per the
Schwarzschild criterion), where MLT's superadiabatic-excess correction doesn't directly apply
- MLT would matter more for the deep convective interior, which looks close to adiabatic
already.

**The sensitivity check, implemented as agreed**
(`validation.check_outer_envelope_recombination_sensitivity`, new `Check 38`, plus
`plot_outer_envelope_recombination_sensitivity`): entirely sterile - a provisional
`_mu_proxy_atomic_molecular(T)` logistic (`T_MID=2500 K`, `WIDTH=180 K`, deliberately wide
per the `GRAD_EFF_SWITCH_EPSILON` lesson) defined LOCALLY in `validation.py`, not touching
`eos.py`/`odes.py`. Both limits anchored to `config.py`'s existing values rather than new
literals: the atomic limit reduces exactly to `config.MU`, and the molecular limit is derived
from X backed out of `config.MU_E`'s own `mu_e=2/(1+X)` definition (a nice internal
consistency check in itself - this independently reproduces `mu_atomic~1.279`, matching
`config.MU=1.278` to 3 decimals). Loads the 10 Gyr run's final snapshot (already
self-consistently converged under constant atomic `mu`), re-evaluates `rho` at the SAME
`(P,T)` via `eos.density` with the proxy `mu(T)`, and re-integrates `r(m)` via the same
continuity equation `odes.py` uses (`dr/dm=1/(4*pi*r^2*rho)`, independently, via a PCHIP
interpolant of `rho(m)` fed to `scipy.integrate.solve_ivp`) - holding `P(m)`, `T(m)` fixed,
isolating the pressure-support effect without touching the energy equation or re-solving the
coupled BVP.

A **control run** (re-integrating with the ORIGINAL `rho`, identical method) is used as the
baseline rather than the raw cached `r_surface`, specifically to cancel the method's own
discretization error: the saved snapshot is a `config.N_GRID_POINTS=200` resample of
`solve_bvp`'s actual ~4900-node adaptive mesh, and PCHIP reproduces the cached profile to
~1e-5 relative accuracy everywhere except the single outermost boundary segment (the fixed
output grid's endpoint spacing is coarser than its geometrically-refined neighbors there) -
found empirically while prototyping (plain linear interpolation gave ~1% control error, plain
log-interpolation made it *worse*, ~15%, by badly extrapolating across that one segment's
~300x density drop; PCHIP got it down to ~0.6%, concentrated entirely in that one point).

**Result: SIGNIFICANT.** `Delta r_surface = -0.144 R_Jup (-3.1%)`, control floor `0.63%`
(effect is ~5x the floor), against the pre-agreed `>=1%` "worth implementing" threshold.
Direction makes physical sense at the local level: raising `mu` at fixed `(P,T)` raises the
implied `rho` (`rho=P*mu*m_H/(k_B*T)`), which compresses those outer shells
(`dr/dm=1/(4*pi*r^2*rho)` shrinks) - the opposite sign from the earlier GLOBAL virial argument
used when first evaluating the user's original claim, because that argument held the whole
star's average T fixed and let mass/energy re-equilibrate, while this test holds the exact
`P(m)`, `T(m)` profile fixed and only perturbs the local EOS closure - a different, and here
oppositely-signed, thought experiment; both are physically legitimate, they just answer
different questions. The visual check (3-panel plot, `mu(T)` proxy / `rho_new/rho_old` /
`r_perturbed - r_control`, all vs `m/M_TOTAL` over the outer 10%) shows a smooth, monotonic
effect building through the outer envelope, though a meaningful share of the total -3.1% is
concentrated in the single least-resolved final grid segment - the magnitude should be read as
an order-of-magnitude estimate, not a precise prediction, but the qualitative verdict (real,
not negligible) looks robust to that caveat.

**Not yet done**: the real implementation (threading a shared `chi(T)`-based `mu(T)`/
`gamma_eff(T)` through all four `eos.py` call sites in `odes.py`, plus the latent-heat term in
the energy equation and the `thermodynamic_delta` correction derived above) - this check's
purpose was specifically to justify that work before investing in it, not to replace it.
`validation.py` gained `import os`, `scipy.integrate.solve_ivp`, `scipy.interpolate.
PchipInterpolator`, and `import output` (previously unused there) to support this.

### 2026-08-10 — ★★★★★ 10 Gyr diagnostic extension: contraction continues but decelerates; resumed live from a saved snapshot for the first time

**Context**: the 4.5 Gyr overnight run (entry below) reached only $r_\text{surface}\approx
4.83\,R_\text{Jup}$ from a ~5.1 $R_\text{Jup}$ start — the user read this correctly as "barely
contracted" and asked directly: is this genuinely slow Kelvin-Helmholtz physics, or is
missing physics (MLT convective efficiency, Saha ionization, the still-atomic-only $\mu(T)$
correction, the interim wide `GRAD_EFF_SWITCH_EPSILON_TIMESTEP`) creating an artificial
equilibrium floor well above `R_HALT`? Chosen diagnostic: more than double the time budget
(4.5 → 10 Gyr) and watch whether the trajectory keeps contracting, asymptotes near the same
value, or reaches `R_HALT`.

**`config.AGE_SOLAR_SYSTEM_S` renamed to `config.T_MAX_S`, value raised to 10 Gyr.** The old
name specifically meant "the solar system's present-day age" — no longer accurate once the
value's purpose became "a diagnostic time budget to test asymptotic behavior," so the
constant was renamed along with its comment (which now also records the reasoning above) to
avoid a misleading name persisting in `config.py`, the single source of truth for these
constants. All live references (`time_stepper.py`'s halt-condition check and log messages,
`main.py`'s docstring) were updated to match; no other module referenced the old name.

**Resumed the run directly from the last saved snapshot, rather than re-solving from
scratch** (`extended_run_10gyr.py`, new file): loaded `snapshots_overnight/snapshot_00077.npz`
(`t=4.5215e9` yr, the exact final state of the 4.5 Gyr run) via `output.load_snapshot` and
called `time_stepper.run()` on it directly, seeded at `dt=1e8` yr (matching the `dt` the
original run was already pinned at for its final several steps, the physically continuous
choice). This is the first time Sub-task 10's snapshot-resumability design was used for its
own sake in a live production run, rather than only as a post-crash recovery mechanism (the
overnight run's plotting-bug recovery, entry below, proved the same capability but only
retroactively) — it also avoided redundantly re-solving the already-converged first 77 steps.
Snapshots/plots were written to fresh `snapshots_10gyr`/`diagnostic_plots_10gyr` directories,
preserving `snapshots_overnight`/`diagnostic_plots_overnight` untouched for direct
before/after comparison.

**Run result: 55 further steps, all converged directly, no continuation fallback, `dt`
pinned at the `ADAPTIVE_DT_MAX=1e8` yr defensive ceiling for every one of them** (i.e. the
raw thermal/pressure-timescale formula's own estimate stayed at or above the ceiling
throughout this entire stretch — a further sign of a smoothly, slowly evolving state, not one
straining against the growth cap). Halted correctly at `t=1.0022e10` yr on the `T_MAX_S`
condition (not `R_HALT` — nowhere close, see below).

**The contraction is real and continues, but is decelerating — not a hard floor, not
unchanged either:**

| $t$ (Gyr) | $r_\text{surface}$ ($R_\text{Jup}$) | $T_\text{center}$ (K) |
|---|---|---|
| 4.52 (resume point) | 4.866 | 10730 |
| 5.52 | 4.806 | 10608 |
| 6.52 | 4.751 | 10491 |
| 7.52 | 4.702 | 10378 |
| 8.52 | 4.657 | 10270 |
| 9.52 | 4.616 | 10165 |
| 10.02 (final) | 4.597 | 10114 |

Local $|dr/dt|$ over successive ~1 Gyr windows: $0.060\to0.054\to0.049\to0.045\to0.041\,
R_\text{Jup}$/Gyr — a steady, gradual slowdown (~32% over 5 Gyr), not a sudden stop. Over the
FULL 0-10 Gyr trajectory (both runs' 133 combined snapshots,
`diagnostic_plots_10gyr/evolution_curves_FULL_0_to_10gyr.png`): starts at 5.109
$R_\text{Jup}$, rises to the already-characterized bump peak of 5.223 $R_\text{Jup}$ at
$t\approx2.5\times10^8$ yr, then contracts smoothly and monotonically the entire rest of the
way with **no repeat of the bump** — net contraction since the peak is 0.626 $R_\text{Jup}$
over 9.77 Gyr.

**Physical read, not fully settled**: the smooth, gradual character of the deceleration
(tracking $T_\text{center}$ and $L_\text{surface}$, both roughly halving over the same 4.5-10
Gyr window) is consistent with a genuinely lengthening Kelvin-Helmholtz timescale
($\tau_\text{KH}\sim GM^2/(RL)$, which grows as $L$ drops) rather than an abrupt numerical or
physics-completeness floor — a hard artificial floor would more plausibly show as a sharp
asymptote, not a continuously-relaxing rate. That said, extrapolating the current decelerating
pace, reaching `R_HALT` (3.6 $R_\text{Jup}$ still to go from the final state) would take far
longer than this already-generous 10 Gyr budget — which is now the concrete, quantitative
argument for treating Sub-tasks 8a (Saha), 8b (molecular→atomic $\mu(T)$), and especially 8c
(MLT convective efficiency, currently the crudest approximation in the energy transport
chain) as the natural next physics work: not because the current qualitative trend looks
wrong, but because the model cannot yet distinguish "this pace is physically correct for a
planet with 2026-08-09/10-era physics" from "a more complete treatment of convective
efficiency or ionization would meaningfully speed up the energy transport and hence the
contraction." Neither hypothesis is confirmed by this run alone; both remain open.

**No code changes to the physics or the adaptive-`dt`/growth-cap tuning** — this was purely
a time-budget extension and a resume-from-snapshot exercise, deliberately keeping every
tuning parameter validated by Sub-task 9 unchanged so the comparison against the 4.5 Gyr run
stays apples-to-apples.

### 2026-08-09 (overnight) — ★★★★ First full run to 4.5 Gyr completed unattended; a real but trivial plotting bug found and fixed; the `r_surface` anomaly characterized as a bounded bump, not a divergence

**Context**: after stopping for the night, the user asked to run something unattended so
there would be data to work with in the morning. Rather than launching the literal, still-
unvalidated "real" production run (risking hours of compute built entirely on the unresolved
`dt`>5e4 yr accuracy question from the same day's earlier entry), a compromise was proposed
and agreed in spirit: an explicitly-labeled EXTENDED DIAGNOSTIC run (`overnight_run.py`, new
file, capped at 100 steps, same tuning as the sanity check, separate `snapshots_overnight`/
`diagnostic_plots_overnight` directories so it wouldn't mix with the earlier sanity-check
output) - framed as more data for the open investigation, not a trustworthy final result.

**Run result: complete success on the physics side.** All 77 steps taken converged directly
(no continuation fallback needed once), reaching `config.AGE_SOLAR_SYSTEM_S` (4.5 Gyr) almost
exactly on target (`t=4.5215e9` yr) - the dual stopping condition's time-based branch firing
correctly for the first time. No crash, no NaN/corruption triggering the safety guard.

**Plot generation crashed immediately after - a real bug, caught and fixed, not a physics
problem.** `output.generate_all_plots` (and the `diagnostics.py` functions it calls) never
created their own output directory - worked by accident every previous time because
`diagnostics.PLOT_DIR="diagnostic_plots"` already existed from earlier sessions' work; the
overnight run's deliberately separate `diagnostic_plots_overnight` directory did not exist,
and matplotlib's `savefig` does not create parent directories on its own. **All 78 `.npz`
snapshots (steps 0-77) survived on disk regardless** - `save_snapshot` already created its
own directory correctly, and the crash happened strictly after the physics loop finished.
Fixed by adding `os.makedirs(..., exist_ok=True)` to every plot-producing function in both
`output.py` and `diagnostics.py` (the latter had the identical latent bug, just never
triggered) - all plots regenerated successfully from the saved snapshots afterward, with NO
re-solve needed (confirms `output.py`'s own exit criterion works exactly as designed: real
recovery from disk after a mid-pipeline failure, not just a demo of the happy path).

**Reviewing the FULL 4.5 Gyr trajectory (not just the first 15 steps seen before) resolves
the most pressing part of the open question from earlier the same day.** The `r_surface`
non-monotonicity found at `dt`>5e4 yr is NOT a runaway divergence - it is a smooth, BOUNDED
bump: $r_\text{surface}$ rises from 5.10 to a peak of ~5.2 $R_\text{Jup}$ around
$t\sim3$-$5\times10^8$ yr, then the contraction trend resumes and continues past the
starting value, reaching $r_\text{surface}\approx4.83\,R_\text{Jup}$ by $t=4.5$ Gyr -
genuine net contraction across the full run. $L_\text{surface}$ shows a matching smooth
peak ($\sim4.2\times10^{-10}L_\odot$) at the same time as the $r_\text{surface}$ peak - the
two are plausibly linked directly via $L\propto r^2(T_\text{surf}^4-T_\text{NEB}^4)$, not
independent anomalies. $T_\text{center}$ decreases smoothly and monotonically throughout,
with visible acceleration in log-time, no irregularity at all. The final structure profile
($t=4.5$ Gyr snapshot) is smooth and well-resolved center to surface across $T$, $\rho$,
$P$ - no sign of numerical noise even at the end of the longest run yet attempted.

**Still genuinely open, not resolved by this run**: WHETHER the bump itself is real
thermal-relaxation physics (a smooth radius/luminosity bump during early cooling is not
implausible physically - real gas-giant "hot start" evolutionary tracks can show non-trivial
early-time behavior as the interior entropy profile relaxes) or a `dt`-resolution artifact
that a finer step through that specific window would resolve differently. The two hypotheses
are no longer equally likely in the way they were before this run, though: a genuine
numerical instability would be expected to grow or destabilize further, not smoothly peak
and reverse on its own while the solver shows zero convergence stress throughout - the
BOUNDED, SELF-CORRECTING character of the bump is itself evidence (not proof) leaning toward
"real behavior, coarsely resolved" over "runaway numerical error." Confirming this properly
would mean re-running that specific time window at a deliberately smaller `dt` and checking
whether the bump's shape/magnitude converges - not yet done, a natural next step.

**Also scientifically notable, independent of the bump question**: after the full 4.5 Gyr
(the model's own physically-motivated total-time budget), $r_\text{surface}$ has only
reached ~4.83 $R_\text{Jup}$, far from `config.R_HALT`$=1\,R_\text{Jup}$ (today's real
Jupiter). This is either expected given the deliberately-simplified input physics this
session has repeatedly flagged as approximate (no MLT convection treatment, no Saha
ionization, an atomic-composition EOS correction rather than a full temperature-dependent
$\mu(T)$, the interim wide-epsilon Schwarzschild-switch regularization) - all of which point
toward LESS efficient cooling/contraction than a fully physical treatment would give - or a
sign that one of these approximations is actually the dominant discrepancy, worth
investigating directly rather than assumed. Not chased further in this pass; recorded here
as a concrete, thesis-relevant number to revisit once the `r_surface` bump question is
settled.

### 2026-08-09 (later same day) — ★★★ Sub-task 10 (`output.py`) implemented; `ADAPTIVE_DT_MAX` raised to a defensive backstop; a genuine large-dt accuracy finding surfaced and deliberately not chased without direction

**Goal**: build `PLAN.md` Sub-task 10's snapshot I/O and plotting deliverables, and add two
safeguards the user explicitly requested before considering the pipeline ready for the real
full run - robust live logging, and a dual (`R_HALT` OR 4.5 Gyr) stopping condition.

**1. `ADAPTIVE_DT_MAX` reconsidered before Sub-task 10's own work began.** The previous
entry's own open item proposed a staged margin-sweep escalation (test progressively larger
fixed `dt` values against the current state before raising the ceiling). The user identified
a real flaw in that plan: a large fixed `dt` (e.g. 5e5 yr) tested against the CURRENT early
state (short thermal timescale) conflates two different questions - "does the numerics hold
at this `dt`" and "is this `dt` even appropriate for this evolutionary phase" - a failure
wouldn't distinguish between them. Since `ADAPTIVE_DT_GROWTH_FACTOR` already self-limits how
fast `dt` can reach any given scale (at most 1.3x/step - it cannot arrive at a large `dt`
before the star has had many steps of real evolution to get there too), the mechanism was
trusted directly instead: `ADAPTIVE_DT_MAX` raised from 5e4 yr to `1e8` yr, reframed in
`config.py` as a purely defensive backstop (protects against a genuine bug, e.g. an
unconsidered masking edge case, not a physically-motivated production ceiling) rather than a
validation-scale rail.

**2. `output.py` built**, reusing `diagnostics.py`'s existing single-state plotting
functions (`plot_structure_profile`, `plot_convective_zones`) for per-snapshot output rather
than duplicating them - the genuinely new pieces are `.npz` I/O
(`save_snapshot`/`load_snapshot`/`load_all_snapshots`), multi-snapshot evolution curves
(`plot_evolution_curves`), and an opacity-regime-colored $\kappa(m)$ map
(`plot_opacity_regime_map`, `PLAN.md`'s one plot type nothing existing already covered).
`generate_all_plots` regenerates everything from disk alone - verified directly (round-trip
tested against a cached state before touching the real pipeline, then exercised end-to-end
on the real sanity run's output).

**3. `time_stepper.run()` gained the two requested safeguards**: every print statement now
flushes explicitly (a long run's terminal output was previously subject to buffering delays
- not acceptable for live monitoring); an explicit finite/positivity check halts immediately
and loudly if `r_surface`/`T_center` ever go non-finite or non-positive, rather than letting
a corrupted state propagate silently into further steps; and a new
`config.AGE_SOLAR_SYSTEM_S=4.5\times10^9` yr halt condition applies alongside `R_HALT`,
whichever triggers first - a physically-motivated backstop (the age of the solar system,
matching Stage 3's own scope) against an indefinitely long run if `R_HALT` is never reached.
`run()` also gained an optional `snapshot_dir` parameter, saving each snapshot to disk as
the run proceeds (not just held in memory) - both for long-run resilience and to feed
`output.py` directly.

**4. Full sanity-check run (`main.py`, 15 steps, `USE_ADAPTIVE_DT=True`, the raised
ceiling) - mechanics all verified working, but surfaced a genuine open finding.** `dt` now
grows well past the old 5e4 yr ceiling (reaching $3.9\times10^5$ yr by step 15, still
climbing), reaching $t=1.67\times10^6$ yr total (vs. $5.76\times10^5$ yr for the same 15
steps under the old ceiling) - confirms the raised backstop works as intended. Every step
converged directly (no continuation fallback), node counts stable (~4300-5100). Live
logging, snapshot saving, and `output.generate_all_plots()` all worked correctly on the
first real end-to-end pass.

**But**: comparing this run point-by-point against the earlier 5e4-yr-capped 15-step run
(PROGRESS.md's previous entry) shows the two are **identical through step 7** (identical
`dt` there, since both are still below the old ceiling) and **diverge starting exactly at
step 8** - the first step where `dt` exceeds the old 5e4 yr ceiling for the first time. From
that point, $r_\text{surface}$ **stops decreasing and turns around**, reaching a minimum of
5.0935 $R_\text{Jup}$ at step 10 then climbing back to 5.0988 by step 15 (~0.10% of the
initial radius, still trending upward when the run ended). $T_\text{center}$ continues
decreasing smoothly and monotonically throughout in both runs, unaffected. The evolution
plot itself (`diagnostic_plots/evolution_curves.png`) shows this is a small wobble on the
scale relevant to reaching `R_HALT` (visually near-flat against the $r$-axis needed to reach
$1\,R_\text{Jup}$), but the correlation with `dt` is precise and reproducible, not noise -
and the solver's own residuals/node-counts show zero sign of numerical distress at any of
these steps, meaning this is an ACCURACY/truncation-error effect, not a
stability/convergence one - a different class of concern than anything else this project has
diagnosed. Working hypothesis, explicitly not confirmed: a purely-gravitational,
degenerate-EOS Kelvin-Helmholtz contraction has no obvious mechanism for genuine radius
re-expansion while the core keeps cooling monotonically - more likely the large implicit
step under-resolving the true continuous trajectory, plausibly connected to
$L_\text{surface}$'s own fractional rate of change (deliberately excluded from
`select_adaptive_dt`'s formula, Sub-task 9, for good and still-valid reasons) growing large
in this same `dt` range. **Deliberately not chased further without explicit direction** -
per the user's own explicit request to review before launching the real run, this is
reported as an open finding, not silently patched or dismissed. The real full run to
`R_HALT`/4.5 Gyr remains un-launched pending that review.

### 2026-08-09 — ★★★ Sub-task 9 (adaptive time-stepping) implemented and validated: growth cap confirmed as the binding safety mechanism, ~5.8x simulated-time efficiency gain demonstrated

**Goal**: replace the fixed `dt` in `time_stepper.run()` with a thermal/pressure-timescale
limiter, per `PLAN.md` §4.5/Sub-task 9 - motivated directly by the previous entry's own
numbers (a full run to `config.R_HALT` at the validated fixed `dt=1e4` yr is thousands of
steps, plausibly hours of compute). Deliberately sequenced ahead of Sub-tasks 8a (Saha)/8b
(molecular→atomic $\mu(T)$)/8c (MLT) - user's explicit decision, both EOS refinements being
orthogonal to this work.

**Design reviewed and revised before implementation, not accepted as first-drafted.** The
initial proposal (this session, following `PLAN.md`'s original $T$-only spec) was reviewed
by the user against standard stellar-evolution-code practice (MESA-style multi-variable
timestep controls) and revised on three points, all adopted after genuine technical
evaluation, not rubber-stamped:
1. **Dual $T$/$P$ constraint, not $T$ alone** - $P$ has been directly measured swinging by
   ~3 decades over a tiny mass range near the photosphere all session (the exact region
   every numerical fight this project has had originated in); a $T$-only limiter could stay
   blind to a fast-evolving $P$ profile there. Both already available from
   `compute_time_derivatives`, zero extra cost.
2. **$L$ deliberately excluded** - $L\equiv0$ exactly at the center by construction (a
   literal $0/0$ every single step, not an edge case), and near the photosphere $L$ has been
   observed crossing zero as *normal* behavior (not a danger signal) more than once this
   session. Including it would make the selector's `min` chronically dominated by benign
   near-zero points for no real signal gained.
3. **Asymmetric growth-factor cap** (`ADAPTIVE_DT_GROWTH_FACTOR=1.3`, growth only, never
   shrinkage) - protects against a sudden 2-3x jump producing a warm-start guess far from the
   true next solution, the same failure character behind every mesh-explosion this project
   has hit.

Final formula: $\Delta t_\text{raw}=\alpha\cdot\min(\min_i(T_i/|\dot T_i|),\min_i(P_i/|\dot
P_i|))$, then growth-capped relative to $dt_\text{used}$, then clamped to
`[ADAPTIVE_DT_MIN, ADAPTIVE_DT_MAX]` - in that order (raw formula → growth cap → absolute
rail).

**Implementation, sterile then wet (CLAUDE.md discipline followed exactly):**
1. New pure function `time_stepper.select_adaptive_dt(state_curr, state_prev, dt_used)`,
   using the already-existing `compute_time_derivatives` (previously diagnostic-only, now
   this function's first real consumer) for the realized $\dot T$, $\dot P$ from the
   just-completed step - a lagged estimate for the *next* step's `dt`, not a
   predictor-corrector.
2. **Sterile test**: 5 synthetic `SimulationState` cases (masking of exactly-zero rates,
   confirming $P$ can bind the min when $T$ is quiet and vice versa, growth-cap engagement,
   both absolute clamps) - all 5 passed. One test-design bug caught along the way (not a
   function bug): a "wild upward spike" case was found to never produce a small timescale
   mathematically ($T/|\dot T|$ is bounded below by $\sim dt_\text{used}$ for any growth
   ratio, since $|T_\text{curr}-T_\text{prev}|<T_\text{curr}$ always when $T_\text{prev}>0$)
   - corrected to a sharp *drop* instead, which does produce a small timescale as intended.
3. **De-risk the epsilon interaction before trusting the live selector** - a fixed-`dt`
   margin sweep at 2e4 and 5e4 yr (bracketing the range `ADAPTIVE_DT_MAX` would allow)
   confirmed `config.GRAD_EFF_SWITCH_EPSILON_TIMESTEP=0.5` (the previous entry's fix) still
   converges directly at both, with modest node counts (11,381 and 16,075) - no reopening of
   the marginal-convection wall.
4. **Wired into `time_stepper.run()`**, gated by `config.USE_ADAPTIVE_DT` (default `False`)
   - fixed-`dt` behavior confirmed byte-for-byte unaffected by the refactor via a direct
   2-step sanity check before touching the adaptive path.

**Real 15-step validation result (T=11500K relaxed seed, seed `dt=1e4` yr,
`USE_ADAPTIVE_DT=True`):**
```
step 1: dt=1.00e4 yr -> next dt selected: 1.30e4 yr   (growth cap binding, ratio exactly 1.3x)
step 2: dt=1.30e4 yr -> next dt selected: 1.69e4 yr   (growth cap binding)
...continues at exactly 1.3x every step...
step 7: dt=4.83e4 yr -> next dt selected: 5.00e4 yr   (ADAPTIVE_DT_MAX reached)
step 8-15: dt=5.00e4 yr (clamped at the ceiling)
```
**The growth cap - not the raw $T$/$P$ formula - was the binding constraint every step from
1 through 7**, confirming the design intuition behind adding it directly, not just in
principle: the raw formula wanted to jump further at every one of those steps, and the cap
throttled it back. All 15 steps converged directly (no continuation fallback ever needed),
node counts stable (~4300-4800, nowhere near the 80,000 budget) even as `dt` grew 5x.
$T_\text{center}$ (11519.92→11469.25K) and $r_\text{surface}$ (5.1035→5.0840 $R_\text{Jup}$)
decreased smoothly and monotonically throughout. $L_\text{surface}$ stayed positive but was
NOT perfectly monotonic - a mild dip (2.126→2.038 ×$10^{-11}L_\odot$) over steps 1-6, then
rising again through step 15 (→2.695×$10^{-11}$) - flagged honestly as an open, unexplained
detail (possibly a genuine local luminosity minimum during this contraction phase, possibly
an artifact of the changing `dt` sampling; not yet investigated, not alarming on its own
since it stays positive and smoothly varying).

**The concrete efficiency win**: 15 adaptive steps reached
$t=5.76\times10^5$ yr of simulated time, vs. the fixed-`dt` run's $1\times10^5$ yr in 10
steps - **~5.8x more simulated time for 1.5x more steps**. This is what makes a full run to
`config.R_HALT` plausible within the remaining timeframe, not just a numerical nicety.

**Full validation.py suite re-run as a final regression check** (`USE_ADAPTIVE_DT=False`
confirmed as the default) - all checks pass except the two already-documented pre-existing,
unrelated failures (Checks 17, 23).

**Open item, explicitly not resolved - the natural next question**: `ADAPTIVE_DT_MAX=5e4`
yr is a deliberately *temporary* validation ceiling (config.py's own comment says so), not a
production value. Reaching `R_HALT` requires simulated time in the billions of years, so
`dt` will eventually need to reach $10^5$-$10^6$+ yr per step - two to three orders of
magnitude beyond anything tested. **Recommendation for next session (not yet executed)**:
raise the ceiling in staged, re-validated increments (e.g. 10x at a time - margin-sweep at
the new ceiling with 2-3 fixed `dt` values, then a real multi-step adaptive run at that
ceiling, THEN raise again) rather than a single large jump - the session's own repeated
lesson (the epsilon requirement grew with EVOLVING STATE complexity, not just with `dt` in
isolation - Sub-task 8's step-6-not-step-2 failure is the direct precedent) argues against
assuming validated behavior at 5e4 yr extrapolates cleanly to $10^6$ yr. The growth cap
itself provides some protection during exploration (a raised ceiling doesn't cause an
immediate jump there, since `dt` still only grows 1.3x/step), but an unmonitored long run
against an unvalidated ceiling risks a late, expensive failure rather than an early, cheap
one. Also worth noting as a distinct, non-numerical risk: as `dt` approaches the KH
timescale itself ($T_\text{KH\_TIMESCALE\_S}\sim10^6$ yr), the quasi-static/implicit-
differencing assumption's own physical validity (not just solver convergence) deserves a
second look, independent of whether the solver happens to converge.

### 2026-08-08 — ★★★ Multi-step time evolution achieved: the step-2 mesh-explosion diagnosed as genuine marginal convection, resolved via a context-dependent Schwarzschild-switch smoothing; 10-step dry run meets Sub-task 8's exit criterion

**Goal**: the previous entry's promotion left one open item — `solve_timestep`'s SECOND real
step failed to converge (node count growing super-linearly, not just under-resolved). User
explicitly ruled out a full mixing-length-theory (MLT) implementation as too risky for the
one-week deadline; this entry is the full diagnose-then-fix trail that respects that
constraint.

**1. Structural diagnosis (not guessed) — three complementary diagnostics, run together:**
- **Superadiabaticity comparison, `state_relaxed` (input to step 1) vs `state_1` (output)**:
  `state_relaxed` is deeply, unambiguously convective almost everywhere
  ($\nabla_\text{rad}/\nabla_\text{ad}$ ratios of 100-1000x); `state_1` has THREE
  convective/radiative flips instead of one, with $\nabla_\text{rad}$ collapsed to sit
  within a few percent of $\nabla_\text{ad}=0.4$ across an extended band
  ($m/M_\text{TOTAL}\in[0.9991,0.9998]$, later confirmed via a full-profile histogram to
  extend to $[0.993,0.99998]$, 13% of profile points below $|\nabla_\text{rad}-
  \nabla_\text{ad}|<0.1$ vs 0.5% for `state_relaxed`). Mechanism: $L$ collapsed ~70x during
  step 1 (already known from the previous entry), and $\nabla_\text{rad}\propto L$, so the
  whole outer envelope moved from saturated-convective to genuinely marginal.
- **Mesh-concentration inspection**: re-ran the failing continuation with a node-density
  histogram at each `alpha` step — confirmed nodes piling up exactly in the two identified
  bands (near-surface $L$-zero-crossing region and the new marginal-convection band), not
  scattered randomly.
- **Interpolation-fidelity check**: re-solved step 1 keeping the dense `solve_bvp`
  interpolant, compared against what `state_1`'s 200-point output grid actually feeds
  `implicit_rhs_vectorized`'s `np.interp` calls for the NEXT solve — found a real but
  secondary ~1-2% error near $T_\text{surface}\to T_\text{NEB}$.

**2. Two candidate fixes tried, tested honestly, and REVERTED — neither was the decisive
lever.** (a) Log-space interpolation of `state_prev.T`/`.P` in `implicit_rhs_vectorized`/
`implicit_rhs_jacobian` (new `_interp_state_prev` helper) — technically more accurate, but
an isolated test found it made `relax_initial_state` itself measurably HARDER (52,949 vs
21,682 nodes, though still converging) by shifting which nearby equally-valid solution the
Newton path lands on, and did NOT fix step 2. (b) Densifying `_bvp_solution_to_state`'s
output grid to match the guess-mesh's own density — did not fix step 2 either, and when
combined with (a) made `relax_initial_state`'s own step 1 fail outright. Both reverted
cleanly (verified via `git diff`-clean isolation tests, one variable at a time) before
proceeding — a real example of the "propose, test, revert if wrong" discipline this project
has followed throughout, not just when it validates the first idea.

**3. Root cause re-framed, and the user's proposed mitigation.** The interpolation fixes
addressed a real but secondary issue; the dominant cause is finding (1) above — genuine
marginal convection, structurally the SAME idealization (`gradients.effective_gradient`'s
$\nabla_\text{eff}=\min(\nabla_\text{rad},\nabla_\text{ad})$, infinitely-efficient
convection) already flagged as this architecture's deepest unresolved liability back in
`PLAN_BVP.md` Milestone 4 — there it showed up as saturated-convective rank deficiency, here
as marginal-convective mesh explosion, both symptoms of the same missing physics. User
proposed widening `GRAD_EFF_SWITCH_EPSILON` substantially (a "wide smoothing" compromise,
explicitly NOT full MLT) rather than implementing MLT under deadline pressure.

**4. Wide-epsilon testing — mechanistically justified, then empirically confirmed, with a
genuine tension found and resolved.**
- Mechanism (verified, not hand-waved): the existing smoothed-min switch's curvature scales
  as $\sim1/\varepsilon$ in the transition band; at $\varepsilon=10^{-4}$ that curvature is
  enormous relative to how fast $\nabla_\text{rad}(m)$ moves through the new marginal band
  (measured: 0.399→1.2 between adjacent output points) — `solve_bvp`'s adaptive refinement
  chases that curvature down to `BVP_COLLOCATION_TOL` without bound. Widening
  $\varepsilon$ reduces the curvature by the same factor.
- A full-profile superadiabaticity histogram (both states) confirmed `state_relaxed` has
  essentially zero marginal region (1/200 points) — a wide $\varepsilon$ would touch nothing
  else in a normal state, only the pathological band.
- Empirical sweep against the actual step-2 failure: $\varepsilon=0.01$ fails;
  $\varepsilon=0.1,0.5$ both converge DIRECTLY (no continuation fallback), with modest node
  counts (10,024 and 5,679 respectively). $L_\text{surface}$ flips from negative to
  positive at step 2 under both — consistent with the negative-$L$ finding being a
  relaxation-pseudo-timestep transient, further evidence beyond the previous entry's single
  data point.
- **Genuine tension found, not glossed over**: a single GLOBAL $\varepsilon$ that fixes
  `solve_timestep` (boundary measured at 0.05-0.07) BREAKS `relax_initial_state`'s own
  continuation at the exact same magnitude ($\varepsilon=0.07$: NaN residual at
  `alpha`=0.999, not just more nodes — a real divergence). `state_0`'s forced-adiabat
  construction has a differently-shaped near-surface transition that the same widened switch
  distorts differently. The two failure modes are cleanly separated by WHICH FUNCTION is
  calling, not by anything varying within a call.
- **Resolution: context-dependent $\varepsilon$**, not one value — `relax_initial_state`
  keeps the original, proven `config.GRAD_EFF_SWITCH_EPSILON=10^{-4}$` unchanged;
  `solve_timestep` alone uses a new `config.GRAD_EFF_SWITCH_EPSILON_TIMESTEP`. Implemented
  by temporarily overriding `config.GRAD_EFF_SWITCH_EPSILON` for the duration of each
  `solve_bvp` call inside `bvp_solver._solve_structure_bvp` (same try/finally pattern
  already used for `N_GRID_POINTS`/`GRID_OUTER_REFINEMENT`) — `gradients.py` itself needed
  ZERO changes, and `validation.py` Checks 12-14 (which test `effective_gradient` at its
  default config value) needed zero changes either, confirmed by direct re-run.
- Verified over a 5-real-step chain (isolated test, before wiring into production): T_center/
  r_surface decrease smoothly and monotonically; T_surface/L_surface settle toward a small,
  steady, slightly-positive value near T_NEB with shrinking increments each step.

**5. Wired into production; the FIRST chosen value ($\varepsilon=0.1$) proved insufficient
for a longer run — caught and fixed honestly, not silently patched over.** After wiring the
context-dependent $\varepsilon$ into `bvp_solver.py` (new `switch_epsilon` parameter on
`_solve_structure_bvp`) and re-verifying the `relax_initial_state` regression (unaffected,
byte-for-byte the original behavior since it never sees the new constant), the full 10-step
dry run (`main.py`) was run for the first time. **Steps 1-5 matched the isolated 5-step test
exactly and converged cleanly, but step 6 failed with the same super-linear mesh-growth
signature** — the marginal band's difficulty evolves step to step, $\varepsilon=0.1$'s
margin (measured 1.5-2x above the 0.05-0.07 boundary for step 2 specifically) was not enough
for step 6. Since `relax_initial_state` is now completely decoupled from
`GRAD_EFF_SWITCH_EPSILON_TIMESTEP`, raising it further carried no risk of reopening the
global-epsilon regression — $\varepsilon=0.5$ (already validated once, in isolation, with
even more comfortable direct convergence than 0.1) was retried for the full chain and
**got all 10 steps through cleanly**.

**Final result — PLAN.md Sub-task 8's dry-run exit criterion genuinely met:**
```
step 1:  T_c=11519.92K  r_surf=5.1035 R_Jup  L_surf=+2.126e-11 L_sun
step 5:  T_c=11505.68K  r_surf=5.0967 R_Jup  L_surf=+2.532e-11 L_sun
step 10: T_c=11487.93K  r_surf=5.0863 R_Jup  L_surf=+2.683e-11 L_sun
```
$T_\text{center}$ and $r_\text{surface}$ decrease smoothly, monotonically, with very regular
step-to-step decrements across all 10 steps (contraction). $L_\text{surface}$ stays
positive and nonzero throughout, with visibly shrinking increments each step (0.19, 0.10,
0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02 $\times10^{-11}L_\odot$) — settling toward
quasi-steady radiative equilibrium, not decaying to zero or diverging. This resolves the
standing negative-$L_\text{surface}$ question about as cleanly as the data allows: a
relaxation-pseudo-timestep transient, decaying through zero and settling to a small,
physically sensible positive value under genuine real-time evolution.

**Honest limitations, explicitly not hidden:**
- Only validated for 10 real steps at $\varepsilon=0.5$. No proof this value holds
  indefinitely if the marginal band's demands keep growing with further evolution — the
  step-6 failure at $\varepsilon=0.1$ is direct evidence the required margin is NOT a fixed,
  one-time quantity. Re-apply the same margin-sweep discipline before any longer run.
- This is a purely numerical regularization, not MLT — no convective velocity, no mixing
  length, no genuine superadiabaticity-dependence beyond the switch's own smoothing. It
  measurably distorts $T_\text{surface}$/$L_\text{surface}$ (the exact quantities the
  negative-$L$ question concerns) by an amount that scales with $\varepsilon$ — bulk
  quantities are far less affected (<0.1% between $\varepsilon=0.1$ and $0.5$) since the
  affected zone is an extremely thin, low-mass surface layer.
- A genuine MLT convection treatment remains the mathematically complete fix — formally
  scheduled as new `PLAN.md` Sub-task 8c, explicitly deferred past the one-week deadline by
  the user's own explicit decision, not silently dropped.
- Not yet a full run to `config.R_HALT` (thousands of steps at this `dt` from
  $r_\text{surface}\approx5\,R_\text{Jup}$ down to $1\,R_\text{Jup}$) — the 10-step dry run
  is Sub-task 8's own stated exit criterion, not the end of the project's evolution work.

### 2026-08-08 — ★★ `solve_bvp` promoted to production; shooting retired for t>0; PLAN_BVP.md merged into PLAN.md; first genuine real-time step run

**Goal**: turn Milestone 6's proven experiment (`bvp_experiment.py`) into the project's
actual solver, per `PLAN_BVP.md` §6's own "Path to Production" plan (already written,
already agreed) — retire shooting for the t>0 problem, re-point nothing (the promoted code
keeps the same function names/signatures `time_stepper.py` already calls), and actually run
`time_stepper.run()` for the first time, which doubles as the direct diagnostic for the
standing negative-`L_surface` question.

**1. Documentation merge.** `PLAN_BVP.md`'s architectural decision and Milestone 6 result
folded into `PLAN.md` §4.2 (which now documents BOTH the original 2026-07 shooting-vs-
`solve_bvp` decision AND this second reversal back to `solve_bvp` for t>0 specifically) and
a new note on Sub-task 5's status entry. `PLAN_BVP.md` itself is kept in place, banner-marked
as merged/historical, for its detailed milestone-by-milestone numerical trail — not deleted,
matching this project's standing "archive, don't delete" convention.

**2. `config.py`**: `bvp_experiment.py`'s local numerical constants (tolerance, max nodes,
mesh density, the `ALPHA_MAX`/continuation ladder, Jacobian-verification parameters, the
relaxation `dt` fraction) promoted to named `config.py` constants (`BVP_COLLOCATION_TOL`,
`BVP_MAX_NODES`, `BVP_MESH_N_GRID_POINTS`, `BVP_ALPHA_MAX`, `BVP_ALPHA_CONTINUATION_STEPS`,
`JACOBIAN_VERIFY_*`, `RELAX_DT_FRACTION`) — CLAUDE.md's "no numerical literals outside
config.py" rule, previously not binding on `bvp_experiment.py` as an isolated experiment.

**3. `bvp_solver.py` rewritten; old shooting archived.** `solve_static_structure` (t=0,
`brentq` shooting on the 3-ODE adiabat) is untouched — it was never implicated in the crash
investigation and `bvp_experiment.py` itself always called it unmodified as a seed-builder.
`relax_initial_state`/`solve_timestep` (t>0) are rewritten in place with the `solve_bvp`
machinery, same names/signatures. The retired shooting implementation (`_implicit_rhs_logm`,
`_integrate_timestep_outward`, the old `relax_initial_state`/`solve_timestep`) is preserved
verbatim in new `bvp_solver_shooting_archive.py` — not deleted, not imported by anything
active, kept as the historical record of why this pivot happened (mirrors how
`bvp_experiment.py` itself was kept after being superseded, now banner-marked as such).

**4. Regression check (Phase 4 of the promotion plan)**: re-ran the promoted
`relax_initial_state` against the two cached seeds `bvp_experiment.py` already proved
(T=11500K, T=12000K). Both reproduced the recorded `status=0` results to <1% relative error
(mostly far tighter — e.g. 12000K's $R_\text{surface}$ agreed to $6.7\times10^{-6}$) —
confirms the promotion is a faithful port, not a new physics/numerics experiment in
disguise.

**5. New territory: verifying `solve_timestep` with a REAL `dt` — found and fixed a genuine
mesh-construction bug never before exercised.** `bvp_experiment.py`'s milestones only ever
tested the pseudo-relaxation step (`state_0`→itself at `DT_RELAX`); a real subsequent
timestep from an already-converged state had never been run under `solve_bvp`. First attempt
crashed immediately (`eos.density` Newton-Raphson failure at solve_bvp's very first
collocation evaluation, before any Newton correction). **Diagnosed directly, not assumed**:
printed the actual RHS at every guess-mesh point and found `dlnP/dx` jumping ~860x between
the last two mesh points (`m/M_TOTAL`=0.99999 to 1.0 exactly) — a real, extremely steep
$P(m)$ transition in the converged solution (P drops ~3 decades over the final $10^{-5}$ of
the mass, a genuinely thin photospheric "skin," not a bug), badly under-resolved by the
mesh-construction's linear (not log) interpolation of `P`, `T` from the warm-start state's
own 200-point output grid. **First fix attempt (log-interpolating `P`,`T`) didn't change the
jump at all** — traced further and found the two adjacent points were EXACT (not
interpolated) matches to the warm-start state's own coarse grid, whose own last interval is
fundamentally under-resolved by `_build_output_grid`'s fixed `GRID_OUTER_REFINEMENT=1e-4`.
**Second fix**: a much deeper `config.BVP_MESH_OUTER_REFINEMENT=1e-6`, applied only when
warm-starting from an already-converged state (a swept comparison, 1e-4 through 1e-12,
showed the max consecutive-point RHS ratio drop from 860x to 1.2x already at 1e-6). Applying
this deeper refinement unconditionally was tried first and found to actively HARM
`relax_initial_state` (mesh nodes balloon across the continuation ladder, exceeding the node
budget where the original, coarser mesh converged cleanly) — scoped to `solve_timestep`'s
`warm_start_L=True` case only, confirmed by re-running the `relax_initial_state` regression
check again afterward (unaffected, identical results).

**Result: `solve_timestep` converges directly** (`status=0`, residual 9.73e-7, no
continuation fallback needed) for one real step, `dt`=1e4 yr, from the relaxed T=11500K
state. Physical result, directly relevant to the standing negative-$L_\text{surface}$
question:
```
                    t=0 (relaxed)        t=1e4 yr (1 real step)
T_center            11523.59 K           11520.10 K       (decreasing - contraction)
r_surface            5.1089 R_Jup         5.1069 R_Jup     (decreasing - contraction)
T_surface            49.36 K              49.99 K          (-> T_NEB=50K)
L_surface           -2.847e23 erg/s      -4.033e21 erg/s   (magnitude dropped ~70x)
```
$T_\text{center}$ and $r_\text{surface}$ both decreasing over real time is exactly Sub-task
8's exit criterion, one step in. The $|L_\text{surface}|$ drop (~70x in one step, while
$T_\text{surface}$ moved to within 0.02% of $T_\text{NEB}$, from 1.3% before) is strong
supporting evidence — not yet proof — that the negative $L_\text{surface}$ finding was a
`DT_RELAX`-pseudo-timestep artifact decaying under genuine time evolution, as hypothesized
when it was first found (PLAN_BVP.md §3.6.4).

**6. Open, unresolved: a SECOND real step does not converge.** `solve_timestep` from step
1's own result (`warm_start_L=True`, same mesh-refinement fix applied) exceeds the node
budget (80000) even via continuation. Diagnosed, not just retried blindly: swept a much
finer `alpha` ladder (steps of 0.1 instead of jumping straight to 0.5) — this got further
(converged cleanly through `alpha`=0.7) but still failed exceeding the budget at 0.9-0.95;
raising the node budget to 200,000 got further still (converged through `alpha`=0.90 at
129,932 nodes) but the node count is growing **super-linearly** across steps
(43569→79660→129932 for `alpha`=0.80→0.85→0.90), the signature of approaching a genuine
local difficulty for this specific state, not a bounded resolution shortfall that a bigger
budget alone will fix. Not chased further today, per this project's own established
timeboxed-investigation discipline (PLAN_BVP.md §4). **Working hypothesis, not yet tested**:
step 1's state has $T_\text{surface}$ already extremely close to $T_\text{NEB}$
(49.99K, 0.02% below) — possibly related to (though not proven to be the same mechanism as)
the historical "frozen fixed point near a rigid $T=T_\text{neb}$ clamp" degeneracy discussed
in `PLAN.md` §4.7, even though the net-flux BC was specifically designed to avoid that exact
literal degeneracy.

**7. `validation.py`: two pre-existing stale checks fixed, one new check added.** Found
while confirming the "full validation-suite pass" checklist item, unrelated to this
migration's own code (just never updated before now): **Check 19**
(`check_boundary_conditions_residuals`) still asserted the ORIGINAL $P_b=P_\text{neb}$
mechanical condition, when `boundary_conditions.py` has used the Eddington $\tau=2/3$
photospheric pressure since Sub-task 5 (2026-07-27) — fixed by solving for the genuinely
self-consistent $P_b$ (implicit, via a small bracketed root-find) rather than assuming a
constant target, and moving the now-nonlinear mechanical-residual perturbation test into the
same nonlinear-formula-testing pattern already used for the thermal residual. **Check 24**
(`check_static_structure_isothermal_and_monotonic`) still asserted `T==T_neb`, `L==0`
everywhere and `P_surface==P_neb` — leftovers from the original diffuse-cloud (Premise 1)
design; rewritten to check what the CURRENT compact, differentiated construction actually
guarantees (monotonic r/P/T, center T matching `T_CENTER_INITIAL`, surface P matching the
photospheric target) — confirmed directly that `solve_static_structure`'s raw output does
NOT reach `T_neb` at the surface (10.6K at 11500K, well below it), since this construction
has no thermal boundary condition at all. `plot_static_structure_profile` updated to match
(now a real 3-panel r/P/T plot, not a 2-panel with T noted as "trivially flat"). **New Check
37** (`check_bvp_jacobian_matches_finite_differences`): promotes `bvp_experiment.py`'s
`verify_jacobians` into a standing validation check (run once, not on every production
solve) — confirmed `fun_jac`/`bc_jac` still match finite differences to $6.5\times10^{-7}$/
$5.0\times10^{-11}$ at the production `T_CENTER_INITIAL`.

**8. Full-suite run surfaced two PRE-EXISTING, unrelated failures — flagged, not fixed.**
Running `python validation.py` end-to-end (not just the checks touched above), plus an
isolated per-check sweep to see past the first crash, found every check passes except two:
`check_stellar_odes_matches_constant_density_analytic_profile` (Check 17: `dr/dm` disagrees
with its closed-form target by ~17x) and `check_static_structure_hydrostatic_balance` (Check
23: relative error 1.78e-2 against a 1e-3 tolerance). **Confirmed via `git diff` that
`odes.py`, `eos.py`, and `gradients.py` are completely untouched by this session's work**,
and §4's own validation-suite table already flagged Check 23 as "not re-run since [an
earlier] rewrite to confirm" — both pre-existing, most likely Check 17 dating to the
`config.MU`/`GAMMA` atomic-composition correction (2026-08-07) never having been
re-validated against it. Out of scope for this pass (this promotion's own explicit scope was
Checks 19/24 only); recorded here so neither is mistaken for a regression this session
caused, and so neither is lost before the next validation-suite pass addresses them.

**Deliberately not done in this pass** (matches the promotion plan's own explicit scope):
a full multi-step production run to `config.R_HALT` (blocked on item 6 above); Sub-task 9
adaptive `dt`; a full audit of `validation.py` beyond Checks 19/24/37.

### 2026-08-07 — ★ Milestone 6 (PLAN_BVP.md): first genuine, fully-converged `solve_bvp` solution — state-vector nondimensionalization + EOS corrections (γ, μ, δ), executed under a one-week thesis deadline

**This is the headline result of the `solve_bvp` pivot to date.** After Milestones 0–4 each
independently ruled out a candidate physics/BC cause of the T=13000K near-photosphere
crash without moving it at all, and after an independent scientific review (conducted in a
separate session, its conclusions carried into `PLAN_BVP.md` directly) found the analytic
Jacobian to be effectively singular at nearly every mesh point — traced to
`gradients.effective_gradient`'s $d(\nabla_\text{eff})/d(\nabla_\text{rad})=0$ identically
wherever convection has locally saturated (100% of the sampled profile, given the
"infinitely-efficient convection" idealization) — a full mixing-length-theory fix was
explicitly ruled out as too slow for the deadline. This entry documents the pragmatic
alternative that was executed instead: attack the Jacobian's *conditioning* directly via
nondimensionalization, and close two independent, silently-wrong physics terms at the same
time, rather than the deeper structural rank issue.

**1. Physics corrections — applied directly to `config.py`/`eos.py`/`odes.py` (shared,
permanent, not `bvp_experiment.py`-local, since these are genuine model corrections, not
solver-architecture experiments):**

- `config.GAMMA`: $1.4\to5/3$, `config.MU`: $2.34\to1.278$. Stage 3 (this project's scope)
  is, by definition, already past H$_2$ dissociation - the envelope is atomic, not
  molecular, essentially everywhere in the relevant T range, so the molecular
  (diatomic $\gamma=7/5$) values were wrong throughout. Consequence accepted in advance:
  $\nabla_\text{ad}=(\gamma-1)/\gamma$ rises from 0.2857 to 0.4, a genuine ~40% shift in the
  Schwarzschild threshold everywhere.
- New `eos.thermodynamic_delta(rho,T,mu,mu_e)`: the energy equation's
  $\delta=-(\partial\ln\rho/\partial\ln T)_P$ coefficient (Kippenhahn & Weigert eq. 4.26,
  $dL/dm=-c_p\,dT/dt+(\delta/\rho)\,dP/dt$) was hardcoded to 1 in `odes.py` - exact only for
  a pure ideal gas. Derived by implicit differentiation of the EOS's defining equation
  ($\delta=P_\text{ideal}/(\rho D)$); verified against both limiting cases ($\to1$ ideal,
  $\to0$ fully degenerate) and the actual T=11500K center density
  ($\delta\approx0.205$ there - the old hardcoded value was off by nearly 5x at exactly the
  point it mattered most, given degeneracy dominates this project's interior).
- `config.T_CENTER_INITIAL`: $13000\text{K}\to11500\text{K}$ - a **direct, measured
  consequence** of the $\gamma$/$\mu$ correction, not an independent choice. Swept
  `mass_error(P_\text{center})` across nearly 6 orders of magnitude at 13000K under the
  corrected EOS: it never reaches zero, its minimum overshoots $M_\text{TOTAL}$ by 5.36% -
  no compact adiabatic seed exists there anymore (more ideal-gas thermal support at fixed
  $\rho,T$ inflates the structure). Scanned T, found the feasibility boundary between
  12000K (root exists, marginally, 0.9922) and 13000K (does not); chose 11500K for margin
  (0.9613). `bvp_solver.solve_static_structure`'s bracket-search window was widened as a
  related companion fix (`1.01`$^{200}$→`1.03`$^{300}$, ~7.3x→~7100x) - the same
  T-independent-seed limitation already flagged for Sub-task 8a/Milestone 5, now also
  binding here at the *current* target, not just at high T.

**2. State-vector scaling — `bvp_experiment.py`-local.** New state
$z=[\hat r,\ln P,\hat L,\ln T]$: $\hat r=r/R_\text{Jup}$ (linear),
$\hat L=\operatorname{arcsinh}(L/L_\text{scale})$ (nonlinear, sign-preserving, log-like
compression; $L_\text{scale}=$ `config.L_KH_SCALE_ERG_S`, already vetted this session).
Motivated directly, not by intuition: a single Jacobian-verification point this session
showed $L$ **28 orders of magnitude** larger than $\ln T$ in the same state vector Newton
was inverting. The Jacobian transform required an extra, easy-to-drop nonlinear correction
term ($-L/(L^2+L_\text{scale}^2)\cdot f_2(y)$, present only in the $\hat L$ row/column,
since only $\hat L$'s scaling is nonlinear) beyond simple row/column rescaling - derived
and implemented explicitly (`implicit_rhs_jacobian_scaled`).

**Two real bugs caught by the mandatory FD cross-check, neither found by inspection:**
1. `_to_physical(z)` intentionally returns $[r,\ln P,L,\ln T]$ (mixed, matching the
   existing RHS convention), but the new scaled boundary-condition functions unpacked it as
   fully physical - feeding `eos.density` $\ln P\approx25$ as if it were a pressure in
   dyn/cm² (should be $\sim10^{11}$). Caught as a 25x `bc_jac`-vs-FD disagreement.
2. A thermal-BC Jacobian term copied the already-chain-ruled $d/d(\ln T_b)$ form ($T_b^4$,
   correct for the *old*, unscaled Jacobian) into a slot meant for the plain $d/dT_b$
   derivative ($T_b^3$), then chain-ruled it a second time. Caught as a ~10x disagreement.

After both fixes: `fun_jac` matches finite differences to $6.5\times10^{-7}$, `bc_jac` to
$1.5\times10^{-5}$ - both comfortably inside the $10^{-4}$ verification gate, every sampled
entry checked, not spot-checked.

**3. Result.** With scaling + corrected physics + analytic Jacobians combined, the
$\alpha$-continuation converged **cleanly through $\alpha=0.00,0.25,0.50,0.75$** -
residuals to $9\times10^{-7}$, boundary residuals to machine precision
($6.94\times10^{-18}$) - the first time in this entire investigation the ramp advanced past
the historical $\alpha\approx0.05$-$0.09$ wall at all. The literal $\alpha=1.0$ endpoint
initially still failed, via a *new* mode (exponentially escalating mesh refinement to NaN,
not the old `eos.density` crash). Finer stepping showed $\alpha=0.9,0.95,0.98,0.99,0.995,
0.999$ **all converge cleanly** - only the exact value 1.0 fails, regardless of step size
approaching it. Since the real gradient term is computed identically at every $\alpha>0$,
this points to the vanishing adiabatic admixture acting as a regularizer on a marginal
instability in the pure unblended system (consistent with, though not fully derived from,
the rank-deficiency finding above) - not a genuine physical discontinuity at $\alpha=1$.
**Fix: `ALPHA_MAX=1-10^{-5}$`, not exactly 1.0** (0.001% adiabatic contamination,
quantifiably negligible). With this, the full continuation **converges completely,
`status=0`, every step**:

```
center:  P_c=1.152e+11 dyn/cm^2, T_c=1.152e+04 K   (self-consistent with the 11500K target)
surface: R=5.109 R_Jup, T_surf=49.36 K, L_surf=-2.85e+23 erg/s (-7.4e-11 L_sun)
max residual: 9.79e-7 | boundary residuals: ~1e-18 to 1e-36 (machine precision)
```

**Honest flag, not swept under the rug**: `L_surface` is slightly *negative* - internally
consistent with the thermal BC (T_surf landed just below T_NEB=50K, and
$L=4\pi r^2\sigma(T^4-T_\text{NEB}^4)$ is negative whenever the photosphere is marginally
cooler than the ambient field), not a numerical artifact, but a genuine open physical
question (net inward energy flow for a contracting, cooling protoplanet) not yet resolved.

**What remains open, explicitly**: why $\alpha=1.0$ exactly is unstable while $0.9999$ is
not (regularization is the working hypothesis, not a proven mechanism); the negative
`L_surface` finding; this result is at T=11500K only, not yet re-tested at other
temperatures; the underlying rank-deficiency (100% convective saturation) was routed
around, not fixed - a real mixing-length treatment remains the mathematically complete fix
and stays explicitly out of scope for this deadline. Full derivation, code changes, and
this discussion are in `PLAN_BVP.md` §3.6 - not duplicated further here.

### 2026-08-07 — Milestone 0 (PLAN_BVP.md): ionization ruled out as the cause of the T=13000K crash; a real but *different* gap found instead (dissociation, not ionization)

**Context**: a joint architectural review proposed elevating `PLAN.md` Sub-task 8a
(Saha-equation EOS ionization upgrade) to Priority 1, ahead of the `bvp_experiment.py`
mesh/BC/opacity work, on the hypothesis that missing hydrogen ionization at
$T_\text{CENTER\_INITIAL}=13000$K was mechanically forcing the observed near-photosphere
`solve_bvp` crash. Rather than accept or reject that by argument, ran a cheap, decisive
empirical test first (`PLAN_BVP.md` §3.0).

**1. Saha ionization fraction, computed along the actual converged T=13000K profile**
(pure-H Saha quadratic, He assumed neutral across this range given its much higher first
ionization potential - a standard simplification, revisit only if warranted): negligible
everywhere. Peak $x\approx5.9\times10^{-4}$ at the hottest point (the center); it falls
monotonically outward and vanishes to $\lesssim10^{-70}$ well before the crash region
(`m/M_TOTAL\ge0.999`), reaching $x\le4.5\times10^{-72}$ there. Hydrogen is not
"significantly ionized" anywhere in this structure at this $T_\text{center}$ - the
premise of the original hypothesis does not hold, quantitatively.

**2. A real, separate gap was found in the same calculation**: `config.MU=2.34` is a
*molecular* value, but this profile (already past H$_2$ dissociation, per Stage 3's own
framing - PLAN.md's `T_CENTER_INITIAL` documentation) should use the neutral-*atomic*
value, $\mu\approx1.278$ - a genuine $\approx1.83\times$ discrepancy, present essentially
uniformly across the whole profile (both the hot interior and the crash region alike). The
original intuition that "the model has roughly 3x too little thermal pressure" was
directionally right in *magnitude* but wrong in *mechanism* - it's dissociation, not
ionization, and the two call for very different fixes (a simple $\mu(T)$ correction vs.
the full, much more numerically severe Saha-equation machinery of Sub-task 8a).

**3. Direct sensitivity test, not just a plausibility argument**: holding the starting
mesh/guess exactly fixed (the same well-matched `MU=2.34` seed every other T=13000K crash
test used) and changing only `config.MU\to1.278` for the `solve_bvp` attempt itself, **the
crash is unchanged** - same location (`m/M_TOTAL\gtrsim0.9995`), same `eos.density`
Newton-Raphson failure mechanism, same near-instant (<0.1s) timing, for both the direct
and continuation attempts. A first version of this test needed a widened bracket search
(the default $\approx7.3\times$ geometric-expansion window failed under the new `MU`, the
same class of failure already known at high T) and landed on a poorly-matched seed
(`m_surface/M_TOTAL=1.43`); that version showed a more benign failure mode, but the clean
rerun confirms this was the mismatched seed, not `MU`, doing the work - a useful caution
for Sub-task 8a's/Milestone 5's own future seed-generation design, not just a footnote.

**Decision**: `PLAN.md` Sub-task 8a / `PLAN_BVP.md` Milestone 5 (Saha) stays scheduled and
still mandatory for physical accuracy (especially at the 40000K target) but is **not**
elevated ahead of `bvp_experiment.py`'s mesh/BC/opacity work - the evidence points away
from it as the cause of the current crash. `PLAN_BVP.md` §3.0 has the full result recorded
against the milestone.

**Follow-up physics finding, flagged for future work, independent of Sub-task 8a**: the
verified $\approx1.83\times$ dissociation gap in `config.MU` is real and worth closing
eventually, but does not need Saha's ionization machinery (and its documented severe
stiffness) to fix - a much simpler, lower-risk dissociation-aware $\mu(T)$ correction
(e.g. a smooth interpolation between the molecular value 2.34 and the atomic value ~1.28
across the H$_2$ dissociation temperature range, no ionization physics needed) would close
it far more cheaply. Not implemented; recorded here and in `PLAN_BVP.md` as a small,
separately-schedulable future item, distinct from and much cheaper than Sub-task 8a.

### 2026-08-06 — `bvp_experiment.py` built and run: partial convergence at T=2000K, decisive near-photosphere crash at T=13000K, hot-end seed generation blocked; `PLAN_BVP.md` roadmap established

Following the architectural decision below, built `bvp_experiment.py` — a standalone,
isolated `scipy.integrate.solve_bvp` implementation (imports and calls `bvp_solver.py`,
`odes.py`, `boundary_conditions.py`, `eos.py` unmodified; reimplements only the RHS/BC glue
as properly mesh-vectorized functions, since `bvp_solver._implicit_rhs_logm` is written for
`solve_ivp`'s single-point contract and isn't directly reusable). Three timeboxed spot-checks
(2000K, 13000K, 40000K), per the approved plan.

**Structural note, confirmed by implementation, not just theory**: under `solve_bvp` the
mass domain `[m_min, M_TOTAL]` is fixed and known exactly (`M_TOTAL` is a project constant,
not a shooting unknown), so the photospheric condition becomes a genuine boundary equation
evaluated at the true endpoint — no `solve_ivp` event, no mass-matching residual needed.
`P_center`/`T_center` are no longer a separate outer root-find's unknowns; they're just
`ya[1]`, `ya[3]`, solved for as part of the one global `y(x)` `solve_bvp` finds directly.

**A real bug found and fixed along the way**: naively reusing
`boundary_conditions.boundary_conditions(ya, yb)` for `solve_bvp`'s `bc()` forces
`r(m_min)=0` *exactly* (its `r_a` residual is literally `ya[0]`) — correct for shooting,
which always calls it with `ya=np.zeros(4)` as a placeholder (shooting enforces `r=r_seed>0`
by constructing the integration's initial state, never checking a residual for it). Under
`solve_bvp`, `ya` is a genuine unknown, and `r=0` exactly is a true $1/r^2$ singularity in
`dr/dm` sitting on a real mesh point. Fixed by rebuilding the center residual to match
`r(m_min)=r_\text{seed}` (the same value shooting seeds with), not `r(m_min)=0`.

**T=13000K: decisive crash, both direct ($\alpha=1$) and continuation ($\alpha=0\to1$)
attempts.** `solve_bvp`'s own Newton iteration — not the initial guess, tested across two
mesh densities, two $L$-guess strategies, and the center-BC fix above — proposes $\ln P$ as
extreme as $-5.3\times10^9$, concentrated exclusively at `m/M_TOTAL` $\in[0.9992, 1.000]$:
the outermost ~0.08% of the mass, at the photosphere. Same physical region as the shooting
kinks in the entry below.

**T=2000K: genuinely more encouraging.** Direct attempt crashes the same way; the
$\alpha$-continuation fallback runs 14 real Newton iterations, with the ODE/collocation
residual converging to $9.99\times10^{-7}$ (near machine precision — the differential
equations themselves are satisfied) while the *boundary-condition* residual stalls,
oscillating around $2.68\times10^8$ (`status=3`, "unable to satisfy boundary conditions
tolerance"). Not a clean pass, but real, attributable progress — and the clearest existing
evidence that the boundary-condition system specifically, not the interior ODEs, is the
harder piece to close.

**T=40000K: blocked before `solve_bvp` runs at all.** `bvp_solver.solve_static_structure()`
(existing, unmodified, shared by both architectures) cannot bracket a root for its $t=0$
adiabatic seed. Also fails at 20000K and 25000K; 30000K did not resolve within a 10-minute
check. Traced (not yet fixed) to the seed's geometric-expansion bracket search
(`P *= 1.01`, 200 iterations $\Rightarrow$ only a fixed $\approx7.3\times$ window) starting
from a **T-independent** T=0-degenerate-limit estimate — as thermal pressure support becomes
non-negligible at higher T, the true root drifts further from that fixed seed. Whether a
compact-radius root exists at all up there, or the search window is simply too narrow, was
not distinguished this session (the diagnostic — a wide `mass_error(P)` sweep at 20000K —
was proposed but not run). Independent, already-documented evidence (`config.py`'s
`T_CENTER_INITIAL` comment: R already exceeds $4\,R_\text{Jup}$ by $T\sim1.7\times10^4$K and
keeps climbing) suggests the fixed-`MU` ideal-gas term is the deeper cause, not just the
search algorithm — tied directly to the already-mandatory Sub-task 8a (Saha ionization,
PLAN.md).

**Reviewed four hypotheses for why the T=13000K crash happens** (opacity-driven, boundary-
formulation-driven, seed-generation-driven, and a "toy opacity" bootstrapping strategy) and
consolidated the result into `PLAN_BVP.md` — the new forward-looking roadmap for the
`solve_bvp` architecture, with four prioritized milestones (opacity bootstrapping, log-space
boundary formulation, analytic Jacobians, Saha-equation high-T seeds). See that file for the
full plan; not duplicated here.

### 2026-08-06 — Architectural decision: shooting abandoned, `solve_bvp` (global relaxation) adopted as the sole path forward

**The kink-whack-a-mole pattern became decisive.** This session found and fixed two
independent hard non-smoothness sources in the shooting-based relaxation homotopy — see the
entry below — and, immediately after both fixes, a **third** wall appeared at
$\alpha\approx0.0466$: the residual plateaus at $\sim1.0\times10^{-4}$ regardless of step
size shrinking to the (now-tightened) $10^{-6}$ floor, the same textbook signature as the
first two. All three sit in essentially the same narrow neighborhood of the homotopy path and
the same physical region of the star (the near-photosphere transition, `m/M_TOTAL`
$\gtrsim0.84$). Opacity's Bell & Lin hard regime switches
(`config.OPACITY_SMOOTH_TRANSITIONS=False`, already anticipated but never engaged) are the
leading unconfirmed suspect for this third one.

**Diagnosis**: shooting integrates the whole domain in one continuous pass, so any local
non-smoothness anywhere along that path corrupts the signal the outer root-find sees at the
far end — patching kinks one at a time as they're discovered is not a bounded, one-time cost,
it scales with how much of the (T, $\rho$, opacity-regime, convective-boundary) space the
project needs to cover, and the required scope ($\approx2000$–$40000$K) covers a lot of it.

**Revisited the historical case against `scipy.integrate.solve_bvp`** (PLAN.md §4.2,
abandoned July 2026) and found both of its two documented blockers plausibly obsolete: the
rank-deficient-Jacobian failure was a structural artifact of the old `dT_dt=dP_dt=0` static
formulation, which no longer exists anywhere in the codebase; the mesh-breaking near-surface
boundary layer was diagnosed under the old `P(M_TOTAL)=P_neb` mechanical condition, since
replaced by the photospheric BC, plus the log-$P$/log-$T$ state transform adopted afterward
for an unrelated reason — both fixed later, for unrelated reasons, while working on shooting,
never retested against `solve_bvp`.

**Decision: shooting is abandoned. `solve_bvp` — a collocation/relaxation method, the same
numerical family as Henyey's implicit relaxation used by essentially every production
stellar-evolution code (MESA, STARS/TWIN, the classical Kippenhahn code) — is the sole path
forward.** Physical baseline (EOS, opacity law, Schwarzschild criterion, photospheric BC,
energy source, quasi-static assumption) explicitly reconfirmed and unchanged before any
`solve_bvp` code was written — this is a numerical, not physical, pivot. Full physical
baseline list and the empirical `solve_bvp` results are in the entry above and in
`PLAN_BVP.md`.

### 2026-08-06 — Two hard kinks in the relaxation homotopy fixed: `gradients.grad_radiative`'s $L\ge0$ floor and `gradients.effective_gradient`'s Schwarzschild switch; two real bugs caught by validation along the way

**Confirmed, by direct instrumentation, that the $\alpha\approx0.046$ wall (previous
entries) was the $L\ge0$ hard floor added 2026-08-01, not the earlier-diagnosed
`fsolve`/residual-verification bug** (that bug is separately and correctly fixed — see prior
entries; this is a second, different obstruction found immediately after). Monkey-patched
`gradients.grad_radiative` to log every pre-floor-negative-$L$ call during the *exact*
failing LM step: over 40,000 engagements in a single call, spanning `m/M_TOTAL=0.89-1.00`,
with probed $L$ down to $\approx-8\times10^{54}$erg/s — 27 orders of magnitude beyond the
physical scale — from finite-difference-probe-scale ($\sim10^{-7}$ relative) perturbations in
$(P_\text{center}, T_\text{center})$. Both LM's outer finite-difference Jacobian and Radau's
own inner implicit-stage Newton iteration (which also assumes RHS smoothness) were tripping
on the same non-differentiable point.

**Fixed with a smooth hyperbolic floor**, $L_\text{safe}=\max(L,0)+\tfrac12\epsilon^2/
(\sqrt{L^2+\epsilon^2}+|L|)$ (the cancellation-safe form — see below), replacing the hard
$L_\text{safe}=\max(L,0)$. **Two real bugs caught before this was trustworthy, both by the
"verify standalone before wiring in" discipline**:

1. **Epsilon chosen for the wrong regime, twice.** First pass ($\epsilon=10^{-3}\times$ a
   fixed KH-luminosity scale) distorted $L$ by 21-24% right where the genuine T=13000K
   solution lives — the KH-virial estimate is a fine *residual normalizer* (already used
   elsewhere) but a bad *absolute smoothing-width anchor*, since it's known
   ($\sim$78-320$\times$) larger than genuine converged $L$. Corrected to
   $\epsilon=10^{-6}\times$ that scale ($\approx1.1\times10^{24}$erg/s) — then
   `validation.py` Check 12 (an unrelated synthetic test point,
   $L_\text{crit}\approx2.3\times10^{24}$erg/s) immediately broke, because that epsilon was
   *also* comparable to, not negligible against, a completely different low-$L$ regime the
   check happened to exercise. Landed on $\epsilon=10^{-9}\times$ the KH scale
   ($\approx1.1\times10^{21}$erg/s), re-verified against both the standalone check and Check
   12/13/14. **Lesson, stated generally**: a smoothing width must stay negligible against the
   *smallest* value the function is ever plausibly handed, not just the one scale used to
   derive it — a single incidental validation test point exposed the gap immediately.
2. **Catastrophic cancellation in the naive formula.** `validation.py` Check 15 (a full
   $T\in[100,50000]$K opacity sweep) caught `effective_gradient`'s naive smoothed-min
   formula, $0.5(a+b)-0.5\sqrt{(a-b)^2+\epsilon^2}$, letting $\nabla_\text{eff}$ *exceed*
   $\nabla_\text{ad}$ in float64 — mathematically impossible, but real in finite precision
   when $|\nabla_\text{rad}|\sim10^8\gg\nabla_\text{ad}\sim0.29$: subtracting two
   $O(10^8)$-scale quantities to recover an $O(0.1)$-scale result loses precision at the
   scale of the large operand, not $\epsilon$. Fixed with the algebraically-equivalent,
   cancellation-safe identity
   $\sqrt{x^2+\epsilon^2}-|x|=\epsilon^2/(\sqrt{x^2+\epsilon^2}+|x|)$, applied to both the
   $L$-floor and the Schwarzschild switch.

**The Schwarzschild switch itself was a second, independent kink**, found the same way after
the $L$-floor fix let the ramp advance further before stalling again at
$\alpha\approx0.050946$: `gradients.effective_gradient`'s hard
$\nabla_\text{eff}=\min(\nabla_\text{rad},\nabla_\text{ad})$ idealizes convection as
infinitely efficient, snapping instantly to $\nabla_\text{ad}$ the moment
$\nabla_\text{rad}$ crosses it. An instrumented trace of the failing LM call found
$\nabla_\text{rad}$ within $3\times10^{-5}$ relative of $\nabla_\text{ad}$ ($=2/7$ exactly,
`config.GAMMA=1.4`) at several profile points — the trial trajectory runs almost exactly
along the convective boundary. Unlike the $L$-floor, this is not an artificial safety clamp
— it's a real physical idealization (the standard smoother alternative is mixing-length
theory's continuous interpolation by superadiabaticity, not implemented here — flagged as a
new mandatory future sub-task in PLAN.md alongside 8a). Fixed as an interim *numerical*
smoothing, same hyperbolic family, same cancellation-safe form,
`config.GRAD_EFF_SWITCH_EPSILON=10^{-4}` (dimensionless — $\nabla_\text{rad}$/
$\nabla_\text{ad}$ don't span the many-decade range $L$ does, so a single fixed epsilon is
appropriate here without the regime-dependent derivation the $L$-floor needed).

**Also**: `bvp_solver.relax_initial_state`'s `alpha_step_min` tightened
$10^{-4}\to10^{-6}$, based on direct evidence (not assumption) that a secondary stopping
point at $\alpha\approx0.0508$ was a genuine, smoothly-converging root that the old floor
simply cut off one halving too early (residual shrank cleanly to below tolerance at a step
just past the old floor) — contrasted explicitly against the real kink's signature (residual
frozen at a fixed value regardless of arbitrarily small step size), which is what the third
wall above showed instead, motivating the architectural decision rather than a fourth
numerical patch.

`validation.py` Checks 13/14 updated from exact-equality assertions to tolerance-based ones
($10^{-4}$), since the smoothed switch is never bit-exact anywhere by construction — not a
new check, maintenance to match the approved formula change, same pattern as the earlier
`solve_static_structure` residual-assertion fix.

### 2026-08-01 — Solver algorithm evaluated and switched (fsolve→LM); does not fix the stall, but is a genuine improvement kept regardless

**Before implementing, evaluated four candidate fixes** (user request): (1) switch the inner
solver to `scipy.optimize.root(method="lm")`; (2) normalize the thermal residual by a fixed
astrophysical constant ($L_\odot$) instead of the KH-timescale estimate; (3) adaptive
$\alpha$-step halving on failure; (4) a larger fixed $\alpha$-step count (e.g. 50-100).
Recommended (1) first, in isolation, before (3): checked scipy's actual parameter set
(not just general reputation) and found a concrete, structural reason to prefer LM -
`fsolve`/`hybrd` exposes only `xtol` (no `ftol` option exists for that method at all), while
`method="lm"` exposes a genuine `ftol` (relative reduction in the sum of squares) alongside
`xtol` - a real, tunable, residual-adjacent stopping criterion `hybrd` structurally lacks.
Recommended against (2): mathematically equivalent in kind to the already-reverted
self-scaling attempt (just another single fixed constant), and against (4): the user's own
instinct that brute-force step count doesn't address *why* the Jacobian is steep.

**Implemented (1)**: both `relax_initial_state` and `solve_timestep` switched from `fsolve`
to `scipy.optimize.root(method="lm")`, with `options={"xtol": config.BVP_TOL, "ftol":
config.RESIDUAL_TOL}`. `solve_timestep`'s soft `warnings.warn`-on-failure path is retained
(some tolerance for a marginally-non-ideal step there, per the original design intent), but
now backed by the same hard `RESIDUAL_TOL` check as `relax_initial_state`, applied
unconditionally regardless of `opt_result.success`. Also a minor efficiency/cleanliness
win: `scipy.optimize.root`'s result object exposes `.fun` (the residual at the returned
point) directly, removing the need to call `residual()` a second time the old `fsolve` path
required.

**Result: LM does NOT fix the stall.** Cleared the cache, re-ran the $\alpha$-ramp (11-step
grid unchanged, per instruction). $\alpha=0.0$ still converges genuinely (residual
`[-1.3e-14, 1.8e-9]`, consistent with every prior run). $\alpha=0.1$ still fails the
`RESIDUAL_TOL` check - and critically, **the residual LM converges to is nearly identical to
what `hybrd` converged to from the identical seed**: `[-7.657e-05, 8.016e-03]` (LM) vs.
`[-7.208e-05, 8.204e-03]` (`hybrd`, previous entry) - agreeing to within a few percent,
despite being two structurally different algorithms (trust-region hybrid vs. damped
Gauss-Newton) with different internal step-taking strategies.

**Interpretation**: two independent algorithms landing on almost the same non-zero residual
from the same starting point is meaningful evidence *against* "the convergence-reporting
was buggy" (which was the mechanism originally diagnosed for `hybrd` specifically) and
*for* "this point is a genuine local minimum of the residual's sum of squares, and the true
root is simply too far away for a local Newton/Gauss-Newton-type step to reach from here in
one jump" - i.e., $\Delta\alpha=0.1$ itself may be too large a step at this $T_\text{center}$,
regardless of which local solver drives it. This shifts weight toward Alternative 3
(adaptive $\alpha$-step halving) as the more likely fix, not primarily a solver-choice
problem.

**Keeping the LM switch regardless of what happens next** - it's a genuine improvement
(real `ftol` criterion `hybrd` lacks entirely) even though it didn't resolve this specific
issue alone, and costs nothing to retain.

**Not yet done**: testing adaptive $\alpha$-stepping (Alternative 3), now the leading
hypothesis given this result.

### 2026-08-01 — Safety net implemented and confirmed working; scaling fix attempted, reverted, needs a different approach

**Part 1 (safety net) — done, proven, both `relax_initial_state` and `solve_timestep`.**
Added `config.RESIDUAL_TOL=1e-4` and, after `fsolve` reports `ier==1`, an explicit check
`max(|residuals|) <= RESIDUAL_TOL`, raising a descriptive `RuntimeError` naming the exact
mechanism (fsolve's `ier` reflects step size, not residual) if it fails.
`solve_timestep`'s old soft `warnings.warn`-only path on `ier!=1` is now *also* backed by
this hard check, regardless of `ier` — a silently-accepted bad state there would corrupt
every subsequent timestep in the loop, so this one always raises. Also added an equivalent
check to `solve_static_structure` (`brentq`-based, not vulnerable to the same `fsolve`
mechanism, but "verify, don't just trust" applies generally) — **caught its own bug on
first use**: reused `RESIDUAL_TOL` there too, and it immediately failed on `brentq`'s
completely normal, always-accepted $\sim10^{-3}$ mass residual (confirmed: this exact
residual, $8.780\times10^{-4}$, was already the accepted, working result before any of
today's changes). Added a separate `config.STATIC_STRUCTURE_RESIDUAL_TOL=1e-2`, matched to
`brentq`'s own established precision, instead of reusing the `fsolve`-specific threshold —
a reminder that "verify the residual" still needs a *method-appropriate* threshold, not one
borrowed wholesale from a different solver's diagnosed failure mode.

**Verified the safety net catches the real problem immediately**: re-ran the $\alpha$-ramp
with it in place (still using the original, proven fixed `L_scale` at this point) —
$\alpha=0.0$ converges genuinely (residual `[-2.8e-14, 2.3e-10]`, unchanged from every prior
run), and $\alpha=0.1$ now correctly raises: `residual [-7.208e-05, 8.204e-03] exceeds
config.RESIDUAL_TOL=1.0e-04`. This is *earlier* than the previous entry's estimate
($\alpha\approx0.2$-$0.3$) — with hard verification in place rather than eyeballing printed
residuals, it's now clear $\alpha=0.1$ was already spurious, just not yet frozen solid.

**Part 2 (scaling fix) — first attempt WRONG, reverted; deeper analysis included.**
Tried self-normalizing the thermal residual by the trial's own photospheric $L$
(`max(abs(L_b), 1.0)`) instead of the fixed KH-timescale-based estimate, reasoning that a
fixed scale computed once (from the very different $T=1200$K-era assumptions) might be
badly mismatched once $T_\text{center}$ runs hot. **This made things worse, not better**:

- Broke $\alpha=0.0$, which had converged essentially perfectly in *every* prior run
  (including with this exact fixed `L_scale`) — `fsolve`'s own internal Jacobian-probing
  now wandered to $P_\text{center}=2.1\times10^{13}$, $T_\text{center}=1.66\times10^6$K, a
  wildly unphysical point, and raised `RuntimeError` immediately.
- **Root of the regression, confirmed by direct calculation, not assumed**: the fixed
  `L_scale` at this geometry is $2.62\times10^{29}$erg/s — **~78x *larger*** than the
  genuine converged $L$ at $T=13000$K ($\approx3.39\times10^{27}$erg/s, from the earlier
  successful relaxation before the safety net existed). Switching the denominator to the
  trial's own $L$ therefore *shrinks* it relative to the fixed estimate, which *grows* the
  normalized residual and its sensitivity — the opposite of the intended fix. It also
  introduces a new hazard the fixed constant never had: dividing by a quantity
  (`L_b`) that can approach zero during `fsolve`'s own exploratory probing, especially near
  the center's own $L=0$ boundary condition.
- **Reverted cleanly** to the original fixed `L_scale` in both functions, keeping only the
  proven safety-net check. Re-verified: $\alpha=0.0$ converges genuinely again, $\alpha=0.1$
  is caught by the safety net exactly as it should be (same numbers as above).

**Deeper conclusion, not yet acted on**: the raw (un-normalized) sensitivity of the thermal
residual to $\ln T_\text{center}$ is enormous in this regime — even divided by the
*larger* fixed scale, the normalized sensitivity was already $\sim-870$ to $-1200$
(previous entry), meaning the raw sensitivity is of order $10^{32}$erg/s per unit
$\ln T_\text{center}$. No single-constant rescaling of the *residual* (fixed or
self-adapting) changes that — it's a property of how steeply the whole structure's surface
flux responds to the deep central temperature in this compact, degenerate, hot regime, not
a normalization artifact. **Proposed next steps, not yet implemented — pending direction:**
1. **Finer $\alpha$-ramp granularity** (more, smaller pseudo-steps than the current fixed
   11) — lower-risk, directly tests whether the problem is "too large a jump into a steep
   region" rather than "wrong overall scale." Cheap to try.
2. **`fsolve`'s `diag` parameter** (`mode=2`) — explicit column (parameter) scaling, telling
   the solver to take smaller steps specifically in the $T_\text{center}$ direction. The
   diagnosed Jacobian imbalance is mostly row-dominant (thermal residual row ~100-1000x more
   sensitive across *both* columns), so this may be a secondary effect at best, but worth
   testing given the user's original suggestion.
3. A genuinely different shooting-variable reparametrization, if (1) and (2) don't resolve
   it - not yet designed.

### 2026-08-01 — Root cause found: `fsolve`'s `ier=1` is a step-size false positive, not genuine convergence

**Tested Hypothesis 1 (`dt` too large) first, per instruction — ruled out cleanly.** Reran
the 5-step dry run with `dt` reduced $1000\times$ ($10^4\to10$ yr). Failed identically:
`RuntimeError: solve_ivp did not reach the photosphere`, same seed point. Traced the full
trajectory at the reduced `dt` and compared directly against the original: **qualitatively
identical** — $T$ pins flat at exactly 13000K starting right after the center
($m/M_\text{total}\sim0.0002$), $L$ flips deeply negative at the same mass fraction, $r$
explodes the same way. Only $L$'s absolute magnitude changed, scaling $\propto1/dt$ exactly
as $dL/dm=-c_p\,dT/dt+dP/dt/\rho$ predicts. A 1000x change in `dt` producing only a linear
rescaling of one term, with the qualitative failure completely unchanged, is strong evidence
`dt` magnitude is not the driver.

**Pivoted to Hypothesis 2 (per instruction, "only if Hypothesis 1 fails") — confirmed, with
a precise mechanism.** Reproduced `relax_initial_state`'s $\alpha$-ramp with extra
instrumentation (`fsolve`'s `nfev`, the residual evaluated at the warm-started seed *before*
`fsolve` moves it, and a finite-difference Jacobian at each step from $\alpha=0.3$ onward):

| $\alpha$ | `nfev` | jump from seed | residual (mass, thermal) |
|---|---|---|---|
| 0.0 | 13 | $2.19\times10^{-3}$ | $[-2.8\times10^{-14},\ 2.3\times10^{-10}]$ — genuine, tight convergence |
| 0.1 | 30 | $5.40\times10^{-4}$ | $[-7.2\times10^{-5},\ 8.2\times10^{-3}]$ |
| 0.2 | 22 | $1.44\times10^{-5}$ | $[-8.6\times10^{-5},\ 9.5\times10^{-3}]$ |
| 0.3 | 22 | $2.27\times10^{-6}$ | $[-1.17\times10^{-4},\ 1.29\times10^{-2}]$ |
| 0.4 | 13 | **$0.0$ exactly** | $[-1.17\times10^{-4},\ 1.29\times10^{-2}]$ — **identical to $\alpha=0.3$**, `fsolve` made *no* change at all |

`ier=1` ("converged") is reported at every single step. But `scipy.optimize.fsolve`'s
`ier=1` reflects its `xtol` criterion — the *relative change between consecutive iterates*
— **not** the residual/function value. From $\alpha\approx0.2$ onward, the step size
collapses toward zero while the thermal residual stays frozen around
$1.3\times10^{-2}$ — three orders of magnitude looser than `config.BVP_TOL`$=10^{-8}$, the
tolerance actually being *requested*. This is a spurious, false-positive convergence report,
not a genuine root.

**Why the step collapses**: the finite-difference Jacobian at $\alpha=0.3$,
$\begin{pmatrix}0.19&7.99\\39.8&-871.2\end{pmatrix}$ (growing to
$\begin{pmatrix}0.19&10.96\\39.8&-1204.2\end{pmatrix}$ by $\alpha=0.4$), is **not singular**
(determinant $\approx-483$, well away from zero) but is severely **badly scaled**: the
thermal-residual-vs-$\ln T_\text{center}$ sensitivity is 2-3 orders of magnitude larger than
the other three entries. A Newton-type step computed from a matrix this poorly scaled maps
even a genuinely large residual to a tiny parameter correction — exactly the observed
"step $\to0$ while residual stays large" pattern. This is a different failure class from
Sub-task 5's near-singular-Jacobian concerns (checked and ruled out there too) — it's a
scaling/conditioning pathology specific to how sensitive the thermal residual has become to
$T_\text{center}$ at this hotter starting point, not a rank-deficiency.

**The gap that let this through silently**: `relax_initial_state`'s own convergence check
is `if ier != 1: raise RuntimeError(...)` — it trusts `fsolve`'s own success flag and never
independently verifies the residual magnitude is actually small. The pseudo-step's own
printed residuals (`[-1.171e-04, 1.295e-02]`) were visible in the original relaxation run's
output and *did* look distinguishably worse than $\alpha=0$'s, but nothing flagged them as
disqualifying — the existing smoothness-jump guard (checks $>50\%$ swings in
$(P_\text{center},T_\text{center})$ between steps) doesn't catch this either, since the
values stopped changing at all, which reads as *stability*, not failure.

**This fully explains the dry-run failure**: the cached `relaxed_state_13000K.pkl` is not a
genuine solution of the real ($\alpha=1$) 4-ODE system — it's effectively still carrying
much of $\alpha\approx0.2$-$0.3$'s character. `solve_timestep`, using the real, unblended
Schwarzschild-selected gradient and differencing against this not-actually-self-consistent
state, diverges from it almost immediately rather than staying close, exactly the pattern
traced in the previous entry.

**Proposed fixes, not yet implemented — pending approval:**
1. **Minimum safety net (should happen regardless of anything else)**: `relax_initial_state`
   should independently assert the residual magnitude is small (e.g. both components below
   some explicit tolerance) after `fsolve` reports `ier==1`, and raise if not — closing the
   exact gap that let this propagate silently. Cheap, mechanical, directly addresses "trust
   but verify."
2. **Fix the underlying scaling pathology**: the thermal residual is already
   non-dimensionalized by `L_scale`, but that scaling is evidently insufficient once the
   *sensitivity* to $T_\text{center}$ itself grows this large at higher starting
   temperatures. Options to evaluate: rescale the shooting parameters themselves (e.g. shoot
   in a variable better matched to the residual's actual sensitivity, not raw $\ln
   T_\text{center}$), or pass `fsolve` an explicit `diag`/scaling hint so its internal step
   calculation isn't misled by the raw Jacobian's disparate entry sizes.
3. Re-run the full $\alpha$-ramp once (1) and/or (2) land, and confirm genuine (not
   spurious) convergence all the way to $\alpha=1$ before trusting the cached state again.

### 2026-08-01 — Sub-task 8 dry run: genuine convergence failure at step 1, not the earlier cancellation bug

**⏸ Stopped here to report rather than keep guessing nudge/dt values — read in full before
resuming.**

Resumed the paused dry run (previous entry). `time_stepper.run()`'s very first
`solve_timestep(relaxed, dt)` call (`dt=0.01\times T_\text{KH\_TIMESCALE\_S}=10^4$ yr, the
same value already validated for the $T=1200$K case) raised, loudly and with a full
traceback (exactly per standing instruction — no blind `try`/`except` anywhere in the loop):
`RuntimeError: solve_ivp did not reach the photosphere during timestep shooting`.

**Traced, not guessed, and this is a genuinely different failure than the earlier one:**

1. **Unnudged seed**: reproduces the *exact same* catastrophic-cancellation collapse
   diagnosed and fixed earlier this session for the $T=1200$K case ($T$ underflowing to 0
   within the first fraction of a percent of the mass, $r$ exploding). Confirms the existing
   `1e-6` nudge mechanism is still doing its job for *that* specific failure mode.
2. **But the existing `1e-6` nudge, which fully fixed this at $T=1200$K, does not fix it
   here.** Traced the trajectory directly: with the `1e-6` nudge, the integration no longer
   collapses, but $T$ gets pinned exactly flat (not decreasing, not inverting - just frozen)
   from $m/M_\text{total}\sim0.0002$ all the way to $m/M_\text{total}\sim0.9$, while $L$
   flips deeply negative ($-9.5\times10^{30}$ by the end) and $r$ explodes to $\sim1.5\times
   10^{12}$cm - no photosphere ever reached. A `1e-4` nudge does better (genuine cooling for
   the first $\sim9\%$ of the mass) before hitting the same flat-$T$/negative-$L$ pattern
   further out. Only a `1e-2` nudge, evaluated as a single trial point, actually reaches a
   photosphere.
3. **But `1e-2` is not a genuine fix either - checked properly, not just "an event fired":**
   running the *full* `fsolve` search (not just evaluating the seed once) at nudges
   `1e-6`/`1e-4`/`1e-3`/`1e-2` shows **none of them converge**: `1e-6` fails immediately
   (same as the unnudged case, just slower to trigger); `1e-4` gives `fsolve` `ier=5` ("not
   making good progress"); `1e-3` and `1e-2` both raise `RuntimeError`s **mid-search, at
   points different from the seed** - `fsolve`'s own Jacobian-estimation/Newton-step probing
   wanders into (P_center, T_center) territory where no photosphere is reached at all, not
   just the exact seed point.
4. **Checked whether this is a near-singular Jacobian (it is not).** Finite-difference
   Jacobian of the 2D residual map at a `1e-2`-nudged point:
   $\begin{pmatrix}0.21&0.30\\33.3&-17.7\end{pmatrix}$, determinant $\approx-13.5$,
   condition number $\approx105$ - a perfectly reasonable, well-conditioned matrix, nowhere
   near singular. Rules out "near-degenerate root-finding problem" as the cause.
5. **But the residual AT that well-conditioned point is large**: mass residual $\approx8\%$,
   thermal residual $\approx8$ (dimensionless, should be $\ll1$ near a root) - meaning the
   true self-consistent $(P_\text{center},T_\text{center})$ is genuinely far from the
   relaxed state's own center values here, unlike at $T=1200$K, where the bare relaxed state
   itself already gave tiny residuals. A well-conditioned Jacobian pointed at a large
   residual is exactly what makes `fsolve` take a large corrective step - which is
   apparently landing in territory where the shooting integration itself fails outright.

**Two live hypotheses for *why* the true root is so far away, neither confirmed:**

- **`dt` may be disproportionately large for this hotter starting point.** Compared the
  *local* KH timescale each relaxed state's own luminosity implies
  ($t_\text{KH,local}=GM^2/(RL_\text{surface})$) against the fixed `dt`: at $T=13000$K,
  $L_\text{surface}=3.39\times10^{27}$erg/s gives $t_\text{KH,local}=7.73\times10^7$yr; at
  $T=1200$K, $L_\text{surface}=6.69\times10^{24}$erg/s gives $t_\text{KH,local}=5.11\times
  10^{10}$yr - a $\sim660\times$ shorter local timescale at the hotter start (physically
  sensible: far more luminous, so draining its thermal reservoir far faster). $dt/t_
  \text{KH,local}$ grew from $2.0\times10^{-7}$ to $1.3\times10^{-4}$ - still small in
  absolute terms, so this may not be the whole story, but it's a real, quantified shift in
  the right direction and worth testing directly (a substantially smaller `dt` for this
  starting point) before assuming it's not the cause.
- **`relax_initial_state`'s own $T=13000$K convergence may not have produced as genuinely
  self-consistent a state as the $T=1200$K case did.** Already flagged in the previous
  entry, not yet investigated: the converged $(P_\text{center},T_\text{center})$ and
  residuals were bit-for-bit identical across $\alpha=0.3$ through $\alpha=1.0$, with a
  thermal residual $\sim500\times$ looser than the $T=1200$K case's $\alpha=1$ step. If
  `relax_initial_state`'s `fsolve` calls were themselves struggling with the same
  root-far-from-seed geometry (just less severely, since they difference against `state_0`
  rather than the relaxed state's own values), the cached "relaxed" state might be a less
  reliable genuine solution than assumed, independent of `dt`.

**Not yet done**: testing either hypothesis directly (a smaller `dt`; re-scrutinizing
`relax_initial_state`'s $T=13000$K convergence path in detail, e.g. checking its own
residual/Jacobian behavior the same way). Deliberately stopped here to report rather than
try a third guessed nudge value or `dt` - matches the standing "trace to root cause before
patching" discipline that already paid off earlier this session (the $L\geq0$ floor, the
original cancellation nudge). No code changed in this entry - only diagnostic scripts run
(scratchpad, not committed).

### 2026-08-01 — Sub-task 8 dry-run PREPARED, not run: session paused before execution

**⏸ Read this in full before running anything in `time_stepper.py` — nothing below has
actually been executed yet.**

Following up on the previous entry (Sub-task 8 implementation): the user asked for a
strict, guardrailed first test rather than jumping straight to a long run — specifically:
(1) a hard `MAX_TEST_STEPS=5` cap, (2) confirmation that each step warm-starts from the
previous step's own converged solution, (3) verbose per-step output in human-readable units
(time and $dt$ in years, $r_\text{surface}$ in $R_\text{Jup}$, $T_\text{center}$ in K,
$L_\text{surface}$ in $L_\odot$) so the trend (contracting, cooling, stable) could be
visually confirmed before a long run.

**Done:**
- **Warm-start confirmed already correct, no code change needed**: `bvp_solver.
  solve_timestep`'s `u0` seed is built directly from `state_prev.P[0]`/`state_prev.T[0]`
  (line ~462, the `1e-6`-nudged seed from the catastrophic-cancellation fix), and `time_
  stepper.run`'s loop reassigns `state` each iteration, so every step's `state_prev`
  argument is genuinely the previous step's own converged output — this was already exactly
  the requested behavior, verified by reading the code, not assumed.
- **`config.py`**: added `SECONDS_PER_YEAR` (refactored `T_KH_TIMESCALE_S` to use it
  instead of an inline `3.156e7` literal) and `L_SUN_ERG_S` (IAU nominal solar luminosity).
- **`time_stepper.run`'s per-step print** upgraded to report $t$, $dt$ in years,
  $r_\text{surface}$ in $R_\text{Jup}$, $T_\text{center}$ in K, $L_\text{surface}$ in
  $L_\odot$ — a permanent improvement to the loop's own logging (not a separate,
  dry-run-only format), since human-readable units are what any future run (short or long)
  should report.
- **Relaxed $T_\text{center}=13000$K state cached**: `solve_static_structure()` →
  `relax_initial_state()` re-run at the new `T_CENTER_INITIAL` and cached
  (`state_0_13000K.pkl`, `relaxed_state_13000K.pkl`) — this is the expensive (~15-20 min)
  step that should never need re-running just to test `time_stepper.run()`. All 11 $\alpha$
  steps converged; final relaxed state: $T_\text{center}=1.301427\times10^4$K,
  $r_\text{surface}=4.153\,R_\text{Jup}$ (consistent with `solve_static_structure`'s own
  $4.154\,R_\text{Jup}$ — the relaxation barely moved the structure, as expected given how
  degeneracy-dominated it is). **Observation, not yet investigated**: the converged
  $(P_\text{center},T_\text{center})$ and residuals are bit-for-bit *identical* across
  $\alpha=0.3$ through $\alpha=1.0$ (thermal residual $\approx1.3\times10^{-2}$, versus
  $\approx2.6\times10^{-5}$ for the earlier $T=1200$K relaxation's $\alpha=1$ step — about
  500x looser, though still small in absolute terms and `ier==1` was achieved at every
  step, no smoothness-guard warning fired). Most likely just a flatter response surface at
  this hotter, still strongly degenerate-dominated starting point (the same weak
  $\alpha$-dependence, only more pronounced, was already seen at $T=1200$K). Not a blocker,
  but worth a closer look if `solve_timestep` behaves unexpectedly in the dry run.
- **Dry-run test script written** (`dry_run_5_steps.py`, scratchpad — not a permanent
  project file): loads the cached relaxed state, calls
  `time_stepper.run(relaxed, n_steps=MAX_TEST_STEPS=5, dt=0.01*config.T_KH_TIMESCALE_S)`.

**NOT done — paused here, explicitly, before execution**: the dry run itself was never
launched. **To resume**: re-run (or re-derive) the dry-run script above against the cached
`relaxed_state_13000K.pkl` and inspect the 5-step output for: $r_\text{surface}$ strictly
decreasing, $T_\text{center}$ decreasing (cooling — PLAN.md Sub-task 8's exit criterion, not
increasing, since this is a degenerate-pressure-supported track), and no solver failures.
Only after that passes should a long run (toward `config.R_HALT`=1.0$R_\text{Jup}$) be
attempted.

### 2026-08-01 — $T_\text{CENTER\_INITIAL}$ finalized (13000K, geometric target); Sub-task 8 (`time_stepper.run()`) implemented

**Decision**: "Geometric Target" approach — prioritize $R$ over the literature-motivated
$T$, isolating the time-stepper work from the EOS ionization gap (previous entry). Set
`config.T_CENTER_INITIAL=13000.0`, with the ASSUMPTION comment recording the full
reasoning and forward-referencing the new mandatory Sub-task 8a.

**Flagging a real discrepancy, not smoothing over it**: `solve_static_structure()` at
exactly $T_\text{center}=13000$K converges to $R=4.1544\,R_\text{Jup}$ (confirmed directly,
not interpolated) — noticeably above the $\sim3\,R_\text{Jup}$ expected when 13000K was
picked, and technically outside the stated $2$-$4\,R_\text{Jup}$ target (the nearest
marched points, 10,765K→3.88 and 13,405K→4.22, don't support a $\sim3\,R_\text{Jup}$
reading near 13000K either — the earlier marching table was already the right data to check
against). Reported clearly rather than rounded down to "close enough" or silently adjusted;
proceeding with $T_\text{CENTER\_INITIAL}=13000$K per explicit instruction, since it's
comfortably post-dissociation and within an order-of-magnitude-reasonable range, but this is
worth revisiting if $R\approx4.15$ (vs. $\sim3$) turns out to matter for later comparisons
(e.g. against literature hot-start tracks).

**PLAN.md**: new mandatory Sub-task 8a ("EOS Ionization Upgrade — Saha Equation") added,
positioned after Sub-task 8, before Stage 1 modeling — not deferred to the Extensions list.
Includes an explicit numerical warning (requested): Saha ionization will introduce sharp
$\mu(\rho,T)$ gradients coupled directly into the hydrostatic-equilibrium ODE (unlike the
existing Bell & Lin opacity transitions, which only affect the radiative term) — expect the
same class of `solve_ivp` stiffness failures already diagnosed this session, requiring
reduced `dt` near transitions (likely pulling Sub-task 9's adaptive stepping forward) and
localized `atol`/`rtol` retuning, not a global one. The former Extensions-table item 14
("full non-ideal EOS if Sub-task 2f's degeneracy term proves insufficient") is absorbed
into this new sub-task and removed from Extensions (renumbered: former item 15, the Stage-1
modeling + unified plot, is now item 14).

**Sub-task 8 implemented**: `time_stepper.py` rewritten — the obsolete homologous
bootstrap (`_bootstrap_time_derivatives`, `compute_time_derivatives`'s `state_prev=None`
branch) removed entirely (already flagged as scheduled for removal, not a fix, since the
last correctness pass; `config.T_KH_BOOTSTRAP_S` no longer exists under that name).
`compute_time_derivatives` retained as the finite-difference-only diagnostic utility it
already was (`bvp_solver._implicit_rhs_logm` does its own inline differencing; unaffected).
New `run(state_prev, n_steps, dt, snapshot_interval=1)`: takes an already-relaxed starting
state as a parameter (not constructed internally) — keeps `run()` a focused loop mechanism,
supports the sterile-pass/wet-pass development split (a mock or `dev_cache`-loaded state
can be fed in directly), and makes no bootstrap/kick special-case for the first call, exactly
per PLAN.md's Sub-task 8 deliverables. Halts when `state.r[-1] <= config.R_HALT`; logs every
step (`t`, $T_\text{center}$, $r_\text{surface}$, $L_\text{surface}$) — no blind
`try`/`except` anywhere in the loop, a genuine failure in `solve_timestep` propagates with
its full traceback, per standing instruction.

**`validation.py` cleanup** (already flagged as needed, not new scope): removed Check 30
(`check_bootstrap_time_derivatives_are_physical`) and Check 32
(`plot_bootstrap_time_derivatives`) — both tested the now-deleted bootstrap mechanism
specifically, not something to fix, only to remove. Check 31 (finite-difference derivatives,
the retained non-bootstrap branch) is untouched. Deleted the now-doubly-stale
`bootstrap_time_derivatives.png`.

### 2026-08-01 — Literature check on $T_\text{center}$ surfaces a real EOS-specific tension; Step 3 (grid, config) implemented, $T_\text{CENTER\_INITIAL}$ left open

**Literature check (web search, not memory alone)**: confirmed a $20{,}000$K number is a
*central*, not surface, temperature — a 20,000K blackbody surface for a 1$M_\text{Jup}$
object is unphysical (checked directly: matching a realistic hot-start luminosity,
$\sim10^{-4}\,L_\odot$, to $R\approx3\,R_\text{Jup}$ via $L=4\pi R^2\sigma T_\text{eff}^4$
gives $T_\text{eff}\approx1000$K, not 20,000K). Also found: present-day Jupiter's own
modeled central temperature is robustly $\sim2.2$-$2.5\times10^4$K (multiple independent
sources: EBSCO astronomy reference, Physics LibreTexts planetary astronomy text, general
references converge on this range). Since the object only cools from formation onward, this
means the *true* post-second-collapse $T_\text{center}$ should sit *above* $2.5\times10^4$K,
not at the low end of the originally-discussed $2\times10^4$-$5\times10^4$K range.

**But this doesn't hold under our own EOS.** Rather than trust either the literature number
or guess, `T_CENTER_INITIAL` was marched from 1200K to 50,000K directly against
`solve_static_structure()` (warm-started geometric bracket search, seeded from each
previous step's converged $P_\text{center}$ — the blind attempt at 20,000K, seeded from the
$T=0$ degenerate-limit guess alone, failed to bracket at all, exactly the kind of seed
mismatch flagged as a risk two sessions ago):

| $T_\text{center}$ (K) | $R$ ($R_\text{Jup}$) |
|---|---|
| 1,200 | 3.17 |
| 5,574 | 3.43 |
| 8,644 | 3.67 |
| 10,765 | 3.88 |
| 13,405 | 4.22 |
| 16,694 | 4.89 |
| ~20,789 | *(bracket search failed — R climbing too fast to track)* |

$R=2$-$4\,R_\text{Jup}$ is achieved between $\sim1200$K and $\sim1.3\times10^4$K under this
model's actual physics — nowhere near $2\times10^4$-$5\times10^4$K. Most likely explanation:
this codebase's EOS has no ionization physics (the ideal-gas term uses a fixed, neutral-gas
$\mu=2.34$ at every temperature); a real, substantially-ionized $2\times10^4$-$5\times10^4$K
plasma would have a much lower effective $\mu$ (more free particles from ionization), hence
*more* thermal pressure support at the same $(\rho,T)$ than this simplified model computes —
meaning the true, fully-physical object would be *even more* extended at that temperature,
not less. The literature-motivated $T$ describes the real object; this simplified 4-ODE,
non-ionized-EOS code does not reproduce that same $(T,R)$ pair self-consistently.

**Decision required, not made unilaterally**: prioritize the geometric target ($R\sim2$-$4$,
pick $T_\text{CENTER\_INITIAL}$ from the range this model actually produces it in) vs. the
literature-motivated temperature target (accept $R$ will exceed 4, likely substantially, and
find out how much by continuing past where the bracket search failed). Recommended:
geometric target — $R$ is what the outer-BC/grid/halt-condition work all depends on and is
directly checkable, whereas $T$ is a much softer target given the known missing-ionization
gap in this specific EOS. `config.py`'s `T_CENTER_INITIAL` ASSUMPTION comment documents this
in full; the value itself is left at the pre-reframing 1200K as an explicit placeholder
(flagged in-line: "PLACEHOLDER, see ASSUMPTION above") rather than guessed.

**Step 3 execution (everything not gated on the $T_\text{center}$ decision) — done:**
- **Composite outer-mass grid** (`bvp_solver._build_output_grid`): log-spaced core + a
  log-spaced-in-distance-to-surface outer `config.GRID_OUTER_MASS_FRACTION` (0.1) of the
  mass, taking `config.GRID_OUTER_POINT_FRACTION` (0.3) of the point budget. Fixes the
  diagnosed cause of the jagged outer-profile artifact: pure `np.logspace` across the full
  $\sim6$-decade mass range put the outer 10% of mass (where $T,\rho,P$ actually change
  fastest) into only $\sim0.05$ of the grid's decades — 1-2 points total. Now: 59 of 200.
  Verified bit-identical physics before/after (same $P_\text{center}$, $R$, residual at
  $T_\text{center}=1200$K — output-sampling-only change, confirmed via `structure_profile.png`
  regeneration: the drop is now smooth). Replaces the single duplicated `np.logspace` call in
  all three of `solve_static_structure`, `relax_initial_state`, `solve_timestep`.
- **`config.py`**: `T_DISSOCIATION_LIMIT` removed (Stage 3 starts already past H2
  dissociation — doesn't apply to this project's forward evolution, which cools rather than
  heats, §1); `R_HALT`=1.0$R_\text{Jup}$ added (Sub-task 8's new halt condition, not yet
  wired up since `time_stepper.run()` doesn't exist yet); new `R_JUPITER_CM` constant, and
  the three pre-existing hardcoded `6.9911e9` literals (`bvp_solver.py` ×2, `diagnostics.py`
  ×1 — a real, pre-existing CLAUDE.md violation) replaced with it.
- **`validation.py`**: `print_all_constants()`'s live reference to the now-removed
  `config.T_DISSOCIATION_LIMIT` fixed (would have raised `AttributeError`) — updated to
  print `R_JUPITER_CM`/`R_HALT` instead.
- **`try`/`except` audit**: confirmed clean two sessions ago (only two exist, both
  legitimate `AssertionError`-is-raised tests in `validation.py` Check 16) — reconfirmed no
  new ones introduced.
- **Verified**: full compile check clean; `solve_static_structure()` still converges
  cleanly against the fully-updated config (placeholder $T_\text{center}$), no exceptions,
  no try/except involved — real failures would propagate with a full traceback.

### 2026-08-01 — Formation scenario reframed into three stages; resolves the $T_\text{center}$/$R$ tension without a parameter sweep

**Context.** Sub-task 6's plots prompted a supervisor review, which flagged that
$T_\text{CENTER\_INITIAL}=1200$K ($R\approx3.17\,R_\text{Jup}$) is too "late" — the intended
$t=0$ hand-off point should be closer to $R\sim20$-$30\,R_\text{Jup}$. A parameter sweep to
find the $T_\text{center}$ giving that radius was planned (previous session) but paused: a
rough virial estimate suggested reaching $R\sim25\,R_\text{Jup}$ thermally would need
$T_\text{center}\sim2\times10^4$K, in direct conflict with `T_DISSOCIATION_LIMIT`=2000K.

**Resolution.** The tension was a category error, not a numerical one: it conflated two
distinct evolutionary stages from the standard (Larson 1969-style) two-step protostellar
collapse picture, now applied explicitly to this GI-formed-giant-planet context:

1. **First hydrostatic core** — the initially diffuse clump settles into a large
   ($10^2$-$10^3\,R_\text{Jup}$), ideal-gas-supported, quasi-static object. $R\sim20$-$30\,
   R_\text{Jup}$ is a point *within* this stage's own contraction, not a target radius for
   our hot-start hand-off. This stage ends at $T_\text{center}\sim2000$K, where H2
   dissociation (endothermic, drops $\Gamma_1$ below the stability threshold) triggers a
   second, dynamical collapse.
2. **Second (dynamical) collapse** — fast free-fall, out of scope (same reasoning as the
   original Sub-task 5 pivot).
3. **Post-second-collapse hot start** — the collapse halts once dissociation completes and
   ionization + electron degeneracy pressure re-stiffen the EOS at much higher density,
   producing a compact ($\sim2$-$4\,R_\text{Jup}$), very hot
   ($T_\text{center}\sim2\times10^4$-$5\times10^4$K) "second core." **This is what
   `bvp_solver.py`'s $t=0$ construction is actually meant to represent** — $t=0$ starts
   *already past* H2 dissociation (the collapse that crossed it isn't modeled), not
   approaching it from below. `T_CENTER_INITIAL=1200$K was too *cool* for this stage, not
   too compact — the earlier sweep-planning session's virial estimate wasn't wrong, it was
   answering a question about the wrong stage (stage 1's geometry at stage-1 temperatures).

**Numerical self-consistency check** (order-of-magnitude, not a full solve): using the
existing degenerate/ideal-gas crossover-density scaling ($\rho_\text{crossover}(T)\propto
T^{3/2}$; PROGRESS.md's Sub-task 2f entry already established $\rho_\text{center}$ at
$T=1200$K is $\sim600\times$ the crossover density there) — at the current $t=0$'s actual
$\rho_\text{center}\approx0.25\,\text{g/cm}^3$, degeneracy remains dominant
($\rho_\text{center}/\rho_\text{crossover}\sim9\times$) up to $T\sim2\times10^4$K, but the
margin shrinks substantially by $T\sim5\times10^4$K ($\sim2\times$) — so the *lower* end of
the requested range is more comfortably self-consistent with landing near
$R\sim2$-$4\,R_\text{Jup}$; the upper end may land measurably larger. **This needs a single
confirmatory `solve_static_structure()` run once `T_CENTER_INITIAL` is set — not a sweep,
but not assumed either.**

**A real, accepted approximation, not silently glossed over**: at $T_\text{center}\sim2
\times10^4$-$5\times10^4$K, hydrogen is substantially-to-fully ionized. `eos.py`'s
degenerate term already assumes full ionization ($\mu_e=1.17$, existing `config.py`
comment), but the *ideal-gas* term still uses a fixed, neutral-gas $\mu=2.34$ — not
self-consistent with an ionized thermal component. Since degeneracy pressure dominates the
mechanical structure in this regime (previous paragraph), this is judged a second-order
approximation, not a blocker — but it should be flagged with an `# ASSUMPTION:` comment
when `T_CENTER_INITIAL` is updated, not left implicit. Non-relativistic electron degeneracy
remains valid (Jupiter-mass densities are nowhere near where relativistic corrections
matter).

**Cooling direction, corrected**: Sub-task 8's exit criterion previously expected
$T_\text{center}$ *increasing* over the simulated evolution. For a degenerate-pressure-
supported contraction (this project's actual regime, unlike a non-degenerate pre-main-
sequence star, where the virial theorem's negative specific heat *does* drive $T_\text{center}$
up as the star contracts), the standard picture is white-dwarf-like cooling: $T_\text{center}$
*decreases* as the object contracts and radiates away its formation heat, consistent with
this project's own established narrative ("slowly radiates away its formation heat and
contracts," §1) and with `config.R_HALT`=1$R_\text{Jup}$ being the natural endpoint of a
cooling, not heating, track. Corrected in PLAN.md's Sub-task 8.

**Terminology note**: the three stages above are called "Stage 1/2/3" in PLAN.md, not
"Phase 1/2/3" — PLAN.md already uses "Phase 1/2/3" for its own top-level project-milestone
structure (Initial Setup / Dynamic Time Evolution / Extensions), and reusing the same
numerals for the astrophysical formation stages would collide with that.

**Decided, not yet implemented** (pending go-ahead): composite outer-mass grid in
`bvp_solver.py` (fixes the separately-diagnosed outer-5%-of-mass resolution artifact,
unrelated to this reframing but bundled into the same implementation pass); `config.py`:
remove `T_DISSOCIATION_LIMIT`, add `R_HALT` (1.0 $R_\text{Jup}$) and a proper
`R_JUPITER_CM` constant (currently `6.9911e9` is a hardcoded literal in three places —
`bvp_solver.py` x2, `diagnostics.py` x1 — a pre-existing CLAUDE.md violation, fixed while
`R_HALT` is added); set `T_CENTER_INITIAL` to a value in the $2\times10^4$-$5\times10^4$K
range (exact number pending one confirmatory run, not a sweep). Codebase audited for blind
`try`/`except`: only two exist (`validation.py` Check 16), both legitimate
"confirm this raises `AssertionError`" tests, not error-swallowing — no changes needed
there.

### 2026-08-01 — Sub-task 6 completed: virial theorem rewritten (unconfined), opacity regime check updated

Closes out the two remaining pieces of Sub-task 6 flagged when the visual plots were added
earlier today. Planned and confirmed with real numbers before implementing (per request).

**Virial theorem → standard unconfined form.** `diagnostics.virial_balance` dropped the
`3*P_neb*V` surface-confinement term entirely (not just simplified — confirmed physically
irrelevant: $3P_\text{neb}V\approx1.37\times10^{28}$ erg vs. $E_\text{grav}\approx
-9.24\times10^{42}$ erg, a ~15-order-of-magnitude gap) and now reports
$(E_\text{grav},E_\text{therm})$ for the standard zero-surface-pressure balance
$E_\text{grav}+3(\gamma-1)E_\text{therm}\approx0$. `run_diagnostics` and `validation.py`'s
Check 26 (renamed `check_virial_balance_unconfined`) both normalize the imbalance against
the terms actually being balanced ($\max(|E_\text{grav}|,|3(\gamma-1)E_\text{therm}|)$)
rather than the now-vanished surface term, which would have made the reported number
physically meaningless. Measured on the real converged structure:
$E_\text{grav}=-9.244\times10^{42}$, $3(\gamma-1)E_\text{therm}=+9.241\times10^{42}$ erg,
relative imbalance $3.6\times10^{-4}$ — asserted `<1e-2` (~28x margin). Also replaced the
old "commensurate with the surface term" sanity asserts with one checking $E_\text{grav}$
and the thermal term are commensurate *with each other* (ratio $\approx1.0004$, confirming
the near-cancellation is a genuine balance, not one term trivially dominating).

**Opacity regime check → multi-regime, index-robust.** `diagnostics.opacity_regime_distribution`
needed no code change (already regime-agnostic). `validation.py`'s Check 27 rewritten:
rather than hardcoding today's exact regime indices (brittle against future
`T_CENTER_INITIAL`/grid changes), asserts the physically-required *ordering* — center in a
strictly hotter regime than the surface — and that more than one regime is populated.
Measured: center (T=1200K) sits in "Metal grains," surface (T=7.5K) in "Ice grains," 99.5%/
0.5% split.

**Found and fixed in passing**: `validation.py`'s Check 30
(`check_bootstrap_time_derivatives_are_physical`, testing the separately-obsolete bootstrap
mechanism, Sub-task 7) also unpacked `virial_balance`'s old 3-tuple — updated the unpacking
arity only (minimal, mechanical fix) so it doesn't crash on the new signature. Running it
surfaced a genuinely separate, pre-existing bug: `time_stepper.py`'s bootstrap code still
references `config.T_KH_BOOTSTRAP_S`, which was renamed to `T_KH_TIMESCALE_S` earlier this
session — `time_stepper.py` was never updated. Confirmed via `grep`, not fixed (Sub-task 7's
job, not in scope here) — see `time_stepper.py`'s module reference entry above.

**Sub-task 6 is now done.** Both rewritten checks (26, 27) verified passing directly.

### 2026-08-01 — Sub-task 6 started: visual diagnostic plots for the relaxed $t=0$ structure

PLAN.md's existing Sub-task 6 scope covered only scalar/print diagnostics (virial balance,
opacity regime census, mass reconstruction) — no plots, despite CLAUDE.md's stated
preference for a visible check over a print-only one wherever a check naturally has
something to look at. Added to scope and implemented: `diagnostics.plot_structure_profile`
($T$, $\rho$, $P$ vs $m$), `plot_mass_radius` ($m$ vs $r$), `plot_convective_zones`
($\nabla_\text{rad}$ vs $\nabla_\text{ad}$, Schwarzschild criterion, convective regions
shaded), and `plot_diagnostics(state)` tying all three together — matching
`validation.py`'s existing `plt.subplots`/`savefig` house style.

Generated against the cached, fully-relaxed state from `dev_cache.py` (per the new
Development Workflow rule — no solver re-run, plots generated in well under a second) and
visually inspected. All three are smooth and physically sensible. `convective_zones.png` in
particular shows $\nabla_\text{rad}\gg\nabla_\text{ad}$ (~7 orders of magnitude) across the
*entire* structure, independently confirming — via the real Schwarzschild criterion
evaluated on the genuinely-relaxed state, not just assumed — that this object really is
fully convective throughout, exactly the assumption `_adiabatic_rhs_logm` forces for the
$t=0$ construction.

`virial_balance`'s pressure-confined form and the opacity-regime census (both flagged as
needing revision once Sub-task 5 landed a final structure) are **not yet revised** — out of
scope for this pass, which focused specifically on the requested visual plots.

### 2026-08-01 — $\nabla_\text{rad}$ floor fixes the relaxation blocker; solve_timestep's own seed needed the same cancellation nudge; Sub-task 5 DONE

**Sub-task 5 is complete and verified end-to-end** — this entry closes out the blocker
opened in the entry directly below.

**$\nabla_\text{rad}$ blow-up: diagnosed and fixed.** User's diagnosis (evaluated and agreed
before implementing): `gradients.grad_radiative`'s $\nabla_\text{rad}\propto\kappa LP/(mT^4)$
is derived assuming strictly outward flux ($L\ge0$); dividing by $T^4$, which $\to0$ near the
photosphere, makes it pathologically sensitive to any negative $L$ excursion there. Floored
$L$ at zero at its point of use inside `grad_radiative` itself (not downstream in
`effective_gradient`) — anchors the guard exactly where the outward-flux assumption breaks
down, rather than patching a symptom two steps removed. **Verified**: `relax_initial_state`
now completes all 11 $\alpha$ steps cleanly (residuals $\lesssim10^{-5}$,
$r_\text{surface}=3.184\,R_\text{Jup}$, $m_\text{surface}/M_\text{total}=1.0000026$).
Specifically checked (per request) whether the floor is a permanent physics change or a
bootstrapping-only artifact: it engages (min $L<0$ on the trial trajectory) for
$\alpha\le0.7$, shrinking in magnitude as $\alpha\to1$, and is **never active at the
converged $\alpha=1$ solution** (min $L=0$ exactly for $\alpha=0.8$–$1.0$) — confirms it is
a pure bootstrapping aid with zero footprint on the final, real physics.

**A third, different blocker then appeared**: `solve_timestep(relaxed, dt)` failed
immediately at its own seed point — not the photosphere-region instability just fixed, but a
violent collapse right at the center ($T$ underflowing to 0, $r$ exploding 13 orders of
magnitude within the first 0.002% of the mass, `solve_ivp` status $-1$, "step size less than
spacing between numbers"). Traced (not guessed) by sampling the trajectory: this is the same
catastrophic-cancellation mechanism already diagnosed and fixed once this session in
`relax_initial_state`'s $\alpha=0$ seed — `solve_timestep`'s seed
(`u0=[ln P_center, ln T_center]` from `state_prev` itself) has no nudge, so when
`state_prev` is already a genuine self-consistent solution (as `relaxed` now is), the trial
and `state_prev` coincide to near machine precision right at the seed, and
$dT/dt=(T-T_\text{prev})/dt$ amplifies the floating-point noise into a spurious blow-up.
**Fixed** with the identical `1e-6` relative seed nudge `relax_initial_state` already uses.
**Verified end-to-end** (via the new `dev_cache.py` workflow, see below): loading the cached
relaxed state and calling `solve_timestep()` converges with residuals
$[2.4\times10^{-10}, 3.1\times10^{-8}]$, giving a physically sensible first step
($T_\text{center}$ cools $1251.9\to1215.3\,$K over $dt=0.01\,t_\text{KH}$,
$r_\text{surface}\approx3.17\,R_\text{Jup}$, mass matched to $10^{-8}$).

**Development workflow overhaul** (user-directed, now in CLAUDE.md's Development Workflow
section): the `relax_initial_state` chain takes ~15-20 minutes per run, which had turned
each downstream debugging round into a long wait. Added `dev_cache.py` (pickle save/load
for a `SimulationState`) so a relaxed state is computed once and reused for all subsequent
`solve_timestep` debugging — the seed-nudge diagnosis above was confirmed against a cached
state in seconds, not minutes. Also adopted: a sterile-before-wet development approach for
upcoming wrapper code (PLAN.md's Sub-tasks 8-9 now specify mock/cached-state testing before
a full solver "wet test"), and a standing rule to keep long-running processes logging
periodically rather than silent.

**Code cleanup**: `bvp_solver.py` and `gradients.py`'s comments were pared down —
session-narrative content (numeric failure traces, "confirmed via X", references to specific
conversational exchanges) removed in favor of terse, present-tense physical/algorithmic
reasoning; ASSUMPTION flags and formula citations kept. PROGRESS.md (this file) remains the
place for the numerical trails and debugging history; code comments now explain the current
design, not how it was arrived at. Verified behavior-preserving (`solve_static_structure()`
reproduces bit-identical output before/after).

**Not yet done**: `validation.py` Check 19 still references the pre-photospheric-BC residual
formula and needs rewriting; no new checks have been proposed for the photospheric condition,
the relaxation homotopy, or the `L>=0` floor. Per the original Sub-task 5 instruction: stop
and check in before starting Sub-task 6.

### 2026-08-01 — Path 1 (logarithmic P, T state variables) implemented and verified; new $\nabla_\text{rad}$ blow-up blocker found, PAUSED

**⏸ Session paused here. Resume by reading this entry in full before touching
`gradients.py`/`odes.py`/`bvp_solver.py` again — do not re-guess.**

Picking up from the 2026-07-27 pause (below): reviewed the two candidate fixes for the
`relax_initial_state` clamp cascade. Diagnosis (proposed by the user, verified before
implementing): tracing the four-round failure cascade back to its origin shows the `1e-300`
floor clamps were never the root problem, only a reactive patch — the AssertionError from
`eos.density` (failure #2) happened *before* any clamp existed, meaning Radau's own internal
stiff-solver Jacobian probing was already generating non-positive trial $P$/$T$ on its own;
the clamps then created a *second*, independent failure (an over/underflow from flooring $P$
and $T$ independently and distorting their ratio). Agreed root cause: nothing in the linear
$(P,T)$ state representation stops Radau's internal probing from going non-positive.
**Verified no hidden traps** in the proposed fix beyond implementation bookkeeping (solver
`atol` needs re-deriving for the log components; scope correctly limited to $P$/$T$, not
$r$/$L$; `eos.density`'s own internal `rho` floor is a *different* clamp, protecting a
converging Newton iterate rather than external probing, and was deliberately left in place).

**Implemented** (plan reviewed and approved before coding): both `_adiabatic_rhs_logm` and
`_implicit_rhs_logm` now integrate $(\ln P,\ln T)$ instead of $(P,T)$ — $P=e^{\ln P}>0$,
$T=e^{\ln T}>0$ by construction, eliminating the entire non-positive-probe failure family
rather than patching each symptom. This is also the standard Henyey/MESA-style state
representation for stellar-structure codes, not a one-off workaround. Side benefit: the
$\alpha$-homotopy blend simplifies to $d(\ln T)/dm=\nabla_\text{ad}\cdot d(\ln P)/dm$
directly, the literal definition of $\nabla_\text{ad}\equiv d\ln T/d\ln P$. `eos.py`,
`odes.py`, `opacity.py`, `boundary_conditions.py` are all untouched — they still
receive/return linear, physical $(P,T)$, preserving their pure-function signatures; only the
`bvp_solver.py` wrappers convert at entry/exit (event functions, the two `residual()`
closures' `boundary_conditions()` calls, and the three `sol.sol(...)`-unpacking call sites).
The `1e-300` clamps and their justifying comment in `_implicit_rhs_logm` are removed outright.

**Verified**: `solve_static_structure()` reproduces the pre-existing validated result exactly
($P_\text{center}=7.686\times10^{11}$, $r_\text{surface}=3.172\,R_\text{Jup}$, mass residual
$1.554\times10^{-3}$) — confirms the log-transform is numerically transparent absent
pathological probing, as expected. `relax_initial_state`'s $\alpha=0.000$ step now converges
*cleanly*, with no crash of any kind (residuals $[-1.4\times10^{-12},-1.1\times10^{-9}]$,
tighter than the pre-fix run) — the clamp-cascade blocker is gone. **Path 1 is complete and
did exactly what it was diagnosed to do.**

**But ramping $\alpha$ to $0.05$–$0.10$ exposed a new, different, more fundamental problem**
— not a crash, a wrong answer. Traced directly (not guessed) by sampling $T(m)$, $L(m)$
along the trajectory at the exact (non-perturbed) $\alpha=0.05$ seed point: from the center
out through $m/M_\text{total}\approx0.60$, the profile is smooth and physical ($T$ dropping
$1252\to720\,$K, $L$ rising to $8.5\times10^{27}\,\text{erg/s}$, consistent with the
$\alpha=0$/pure-adiabat trajectory). Between $m/M_\text{total}\approx0.60$ and
$\approx1.09$ — right around where the pure-adiabat structure's own photosphere sits
($T\approx7.7\,$K there at $\alpha=0$) — the profile catastrophically diverges: $T$ jumps to
$\approx1.26\times10^5\,$K and $L$ flips to $\approx-6.4\times10^{30}\,\text{erg/s}$, after
which the structure never cools back down and balloons outward ($r\to10^{12}\,$cm,
$\approx14\,R_\text{Jup}$) without ever crossing the photospheric-pressure threshold, even
out to $50\times M_\text{total}$.

**Likely mechanism** (grounded in `gradients.grad_radiative`'s formula, not yet confirmed
further): $\nabla_\text{rad}=3\kappa LP/(16\pi a_\text{rad}cGmT^4)$. Near the photosphere,
$T$ is dropping toward a few K on the adiabat, making $T^4$ tiny — this makes
$\nabla_\text{rad}$ extremely sensitive to $L$ there. Once $\alpha>0$ introduces even a
slightly different $dT/dt$, $dP/dt$ trajectory than the pure adiabat, $L$ (built up via
$dL/dm=-c_p\,dT/dt+dP/dt/\rho$, integrated from the center) can swing sign near that same
sensitive region. A sign-flipped $\nabla_\text{rad}$ then gets selected directly by
`gradients.effective_gradient`'s Schwarzschild criterion (`is_convective = grad_rad >
grad_ad`; a negative $\nabla_\text{rad}$ fails that test, so $\nabla_\text{eff}=\nabla_\text{rad}$
is used as-is) and fed into $dT/dm=(T/P)\nabla_\text{eff}(dP/dm)$ with the wrong sign,
flipping $T$'s trend from decreasing to increasing outward.

**Not yet diagnosed further or fixed.** This is architecturally distinct from the clamp
cascade Path 1 targeted — it lives in `gradients.py`/`odes.py`'s core physics formula near a
$T\to0$ sensitive limit, not in `bvp_solver.py`'s state representation, and touching it
needs the same before-writing-code review this session's other structural changes have had.
`bvp_solver.py`'s Path 1 changes are done, correct, and should not be reverted or touched
further while diagnosing this — the new failure is confirmed present with `solve_ivp`
reporting clean success (no crash, no assertion), so it is not related to the log-transform.

### 2026-07-27 — Photospheric outer BC implemented and validated; initial-state relaxation designed, partially working, PAUSED

**⏸ Session paused here. Resume by reading this entry in full before touching
`bvp_solver.py` again — do not re-guess at domain-clamp values.**

Following the plan approved earlier the same session (photospheric $\tau=2/3$ boundary
condition replacing $P=P_\text{neb}$), implemented and verified:

1. **`boundary_conditions.photospheric_pressure(r, P, T, mu, mu_e)`** — the Eddington
   grey-atmosphere formula $P=\frac{2}{3}g/\kappa$. `boundary_conditions()`'s mechanical
   residual now uses this; thermal residual formula unchanged.
2. **Event-based surface location**, not a fixed-endpoint residual — both
   `solve_static_structure` and `solve_timestep` integrate outward with the photosphere as a
   `solve_ivp` event, matching enclosed mass at that event to `M_TOTAL`. Verified by
   read-only scratch test *before* implementing that this resolves the reachability gap
   (smooth, monotonic, no crash across the same `P_center` range that produced the old gap)
   — a fixed-endpoint version was tried first and found to have the identical problem.
3. **`solve_static_structure()` now converges cleanly**: $R\approx3.172\,R_\text{Jup}$,
   mass relative residual $\approx0.16\%$ (down from $\sim10^7$). Full detail in the
   `bvp_solver.py` module reference above.

Then attempted to bridge this into a state usable by `solve_timestep`. Confirmed the Step-1
correctness-review concern sharply: evaluating the real 4-ODE system at
`solve_static_structure`'s own values diverges ($T\to3.4$ million K within one full step) —
`state_0` is not a genuine solution of `solve_timestep`'s equations. **User pushed back
correctly on my first instinct** (rebuild $t=0$ via the full 4-ODE system) by pointing out
this contradicts the project's own premise: $T_\text{CENTER\_INITIAL}$ is a *chosen*
hand-off snapshot, not something with a principled derivation from a "previous state" that
doesn't exist at $t=0$ — demanding `state_0` satisfy the same time-differenced equations
`solve_timestep` uses is a category error, not a bug to fix by force. This reframed the
problem correctly: **`solve_timestep` needs to be robust to an approximate starting state**
(standard "initial model relaxation," MESA/Henyey-style), not the other way around.

**User's homotopy proposal caught a genuine trap before implementation.** Scaling the whole
`dL/dm` source term by $\alpha\in[0,1]$ (starting at $\alpha=0$ to "turn off" the mismatch)
would have forced $dL/dm\equiv0$ identically at $\alpha=0$ (no other source exists) —
reproducing the *exact* isothermal degeneracy this whole investigation exists to escape.
Caught and explained before writing any code. **Corrected version, implemented**: homotopy
on $\nabla_\text{eff}$ directly (blend between $\nabla_\text{ad}$, matching `state_0`'s own
construction exactly at $\alpha=0$, and the real Schwarzschild-selected value at $\alpha=1$)
— `_implicit_rhs_logm` gained an `alpha` parameter (default 1.0, bit-identical to before for
every real `solve_timestep` call); new `relax_initial_state(state_0)` ramps `alpha` over 11
fixed steps with explicit, strict convergence checks (raise on `fsolve` non-convergence,
print every step's actual residual, flag suspiciously large jumps between steps as possible
solution-branch changes). Full design and convergence-criteria detail in the `bvp_solver.py`
module reference above.

**Result: the `alpha=0.000` step converges beautifully** (residuals
$[-4\times10^{-13}, -3\times10^{-8}]$) — the corrected homotopy is validated as physically
and mathematically sound. **But subsequent steps hit a four-round cascade of `scipy`
stiff-solver numerical edge cases** (floating-point catastrophic cancellation at the exact
seed point; `eos.density` Newton non-convergence on Radau's internal Jacobian probes;
negative-$P$/$T$ probing; an over-permissive clamp causing a new overflow) — each diagnosed
and fixed individually, each revealing a new one. **Decision: stop patching reactively.**
Two principled fixes are proposed, neither implemented: (1) integrate in log-transformed
state variables ($\ln P$, $\ln T$) so positivity is structural, not enforced by clamps: (2)
graceful degradation inside `eos.density` (return a bounded penalty rather than asserting,
letting the integrator's own step control respond). **No decision made — evaluate next
session.** `eos.density` did gain two lasting improvements from this chase regardless of
which path is chosen next: the positivity clamp during Newton iteration, and 50 (not 20)
iterations.

**Not yet done, flagged for next session:** `validation.py` Check 19 needs revision for the
new photospheric residual formula (currently still references `P_b-P_\text{neb}$); no new
checks proposed yet for the photospheric condition or `relax_initial_state`; `PLAN.md`'s
Sub-task 5 entry needs the same status sync as this entry.

### 2026-07-27 — Sub-task 2f implemented: degeneracy pressure fixes compactness, but exposes the outer BC as a hard blocker

Implemented the plan from the correctness review below: added `eos.degenerate_pressure`
and rewrote `eos.density` to invert the combined ideal+degenerate EOS via vectorized
Newton-Raphson (full detail in the `eos.py` module reference above). Added `config.M_E`,
`config.PLANCK_H`, `config.MU_E`. Proposed and got approval for 4 new `validation.py`
checks (a reference-point check, an asymptotic-limit check, a round-trip inversion check
across 8 decades of density, and a visible $P(\rho)$ plot) — all pass cleanly.

Before touching any solver code, derived an analytic order-of-magnitude prediction: the
pure T=0 degenerate (Zapolsky-Salpeter-style) limit for $M_\text{TOTAL}=1$ Jupiter mass
gives $R\approx3.11\,R_\text{Jup}$ — reassuringly close to Jupiter's actual radius (the
remaining ~3x is the well-known Coulomb-lattice correction this minimal treatment omits).

Re-running `solve_static_structure` with the new EOS **confirmed the prediction almost
exactly** ($R\approx3.17\,R_\text{Jup}$) once two bracket-search robustness issues were
fixed (search direction is not fixed - degenerate objects have an inverted mass-radius
relation; `brentq` can converge to a $P_\text{center}$ that doesn't itself integrate
successfully when the crash/success transition is razor-thin - both documented in the
`bvp_solver.py` module reference above). **But this exposed a more fundamental problem**:
the converged structure's surface-pressure residual is enormous ($\sim1.4\times10^7$
relative) because, as a broad scan of $P_\text{center}$ confirmed, **no central pressure
reaches anywhere near $P_\text{neb}$** — $P_\text{end}$ jumps discontinuously from trapped
below $\sim0.05$-$0.08$ to $\ge2.79\times10^6$, a genuine gap, not a numerical-precision
issue (confirmed: tightening `rtol` by $10^4\times$ changes nothing).

This directly confirms and sharpens the outer-BC concern flagged in the correctness review
below (itself reviving a concern from PROJECT_CONTEXT.md §3 that predates this session): a
degenerate-supported interior is far stiffer than the old ideal-gas-only structure and
cannot bridge the same bulk radiative-diffusion equation of state down to the tiny ambient
nebula pressure — a real photosphere would hand off to a different physical description
long before reaching that point. **Decision: Sub-task 2f is done and its physics is
validated; Sub-task 5 is now blocked on redesigning the outer boundary condition (a
photospheric condition, e.g. $\tau=2/3$) rather than on the EOS.** Design work is underway,
under review before implementation, per the user's explicit request to see the exact
residuals/equations first.

### 2026-07-27 — Independent correctness review of `bvp_solver.py`, and a second architectural gap found

Before starting Sub-task 2f, did a line-by-line review of `solve_static_structure` and
`solve_timestep` against the physics and shooting-method logic (not just "does it run") —
prompted by treating "already implemented" as "not yet trusted" rather than assuming the
prior session's rewrite was correct by default. No code changed in this entry.

**Result: no coding bugs found.** The Lane-Emden scaling relations were re-derived
independently and matched line-for-line; the converged structure satisfies the exact
adiabatic relation to $5\times10^{-9}$; the root-find's bracket and convergence are
cross-validated against the analytic Lane-Emden answer (0.08% agreement); the
`boundary_conditions.py` residual indexing/signs used inside `solve_timestep` are correct.

**But a second, independent gap was found and confirmed, not just guessed at.** A
previously-flagged "leading guess, unverified" (why the self-consistent full-4-ODE
construction lands on $R\approx27{,}000\,R_\text{Jup}$) was directly checked by tracking
$\nabla_\text{rad}$ vs. $\nabla_\text{ad}$ along that trajectory: the structure is
convective out to $m/M\approx0.49$, then genuinely transitions to radiative at
$m/M\approx0.70$, after which $T$ stays nearly flat (192 K → 157 K) while $r$ explodes
$153\times$ over the last 5 decades of pressure drop to reach $P_\text{neb}$ — a
nearly-isothermal extended envelope, mechanistically the same phenomenon as the *original*
Bonnor-Ebert problem (Premise 1), now confined to the outer ~30% of the mass. This directly
revisits a concern PROJECT_CONTEXT.md §3 raised for an earlier hot-start attempt (forcing
$P(M_\text{total})=P_\text{neb}$ is likely a structural mismatch once self-gravity should
dominate the bulk structure) that had not been re-examined under the current compact
hot-start line of work — it turns out to still apply, and independently of the missing-EOS
question: electron degeneracy pressure is negligible in the tenuous outer envelope where
this extended tail forms, so Sub-task 2f is not expected to fix it.

**Decision:** proceed with Sub-task 2f as planned (still correct and necessary for interior
compactness), but track the outer boundary condition as a separate, likely-necessary
follow-up before Sub-task 5 can be considered complete — not a blocker for 2f, and not
something 2f's exit criterion should be expected to resolve on its own. See the updated
`bvp_solver.py` entry above for full detail.

### 2026-07-27 — Documentation reorganization

This file and PLAN.md were substantially reorganized (not just appended to) to reflect
everything below — both now distinguish confirmed results, strong physical inference, and
open hypotheses explicitly, and the sub-task roadmap now reflects the electron-degeneracy
hypothesis as the next milestone rather than continuing to debug the ideal-gas-only
solver. No code changed in this entry.

### 2026-07-27 — Compact hot-start implementation: two numerical dead ends, and the electron-degeneracy-pressure hypothesis

Implemented Premise 2's compact hot-start construction in `bvp_solver.py` (Lane-Emden-seeded
adiabatic shooting for $t=0$; reverted `solve_timestep` to the pure implicit energy
equation). Two things went wrong, in sequence, each more fundamental than the last:

**1. Ideal gas alone is not compact.** The pure-adiabat construction, at
`T_CENTER_INITIAL`=1200 K, converges cleanly but to $R\approx300\,R_\text{Jup}$, not a
genuinely compact few-$R_\text{Jup}$ structure as originally envisioned. Verified via an
independent Lane-Emden analytic solution (cross-checked against tabulated $n=1.5$/$n=3.0$
results first). Root cause: an ideal gas at any physically reasonable (sub-dissociation)
temperature simply cannot generate enough pressure to compress a Jupiter mass to a compact
radius — $T_\text{center}\sim120{,}000$ K would be needed, absurd. **Decision (user
confirmed):** accept the $\sim300\,R_\text{Jup}$ result as the mathematically honest
consequence of the current (ideal-gas-only) physics rather than force compactness some
other way — still a genuine, non-isothermal, ~10x-more-compact-than-Premise-1 structure,
useful for testing the time-stepping machinery even if not literally "a few $R_\text{Jup}$."

**2. That construction is not self-consistent with the real governing equations.**
Discovered while smoke-testing `solve_timestep(state_0, dt)`: it failed to converge, and
diagnosing why revealed that `solve_timestep`'s residual is huge ($\sim10^8$) even
evaluated at `state_0`'s own unperturbed center values — confirmed not a $dt$-sensitivity
issue (a 10x smaller $dt$ barely changed the residual). Root cause: the pure-adiabat
construction bypasses `odes.stellar_odes`'s real Schwarzschild-criterion gradient selection
(it forces $dT/dm=\nabla_\text{ad}\cdot(T/P)\cdot dP/dm$ directly, since no self-consistent
$L$ exists for a purely-assumed-convective structural shortcut) — so it is not actually a
solution of the same equations `solve_timestep` uses to evolve forward.

Attempted a fix: reconstruct $t=0$ using the *real* 4-ODE system (`odes.stellar_odes`
unchanged), with $L$ genuinely sourced via the homologous-contraction rate
($\partial T/\partial t=T/t_\text{KH}$, $\partial P/\partial t=4P/t_\text{KH}$) — physically
honest for an object caught mid-collapse-relaxation (not a degeneracy-breaking trick, since
this is now motivating $L$ physically, not just algebraically). This needed a numerical
continuation approach to even locate (the viable $P_\text{center}$ window is extremely
narrow, and a naive/blind search never finds it — later understood as a Lane-Emden-type
"natural surface" sensitivity, and the analytic Lane-Emden estimate was reused successfully
to seed the bracket). Once found, this construction **does** converge to a solution
matching $P(M_\text{total})=P_\text{neb}$ — but at $R\approx27{,}000\,R_\text{Jup}$, *more*
extended than Premise 1's original diffuse-clump result, not compact. Confirmed clean and
monotonic (checked ~15 points between the two brackets, no sign of a hidden nearer root) —
not a numerical artifact. **This result was explored only in scratch testing and was never
written into `bvp_solver.py`** — the file currently still contains only the pure-adiabat
(~300 R_Jup) construction from point 1.

**Why this stopped implementation, rather than continuing to iterate on the shooting
method:** at this point, both a "compact but not self-consistent" and a "self-consistent
but not compact" construction existed, and no numerical trick was closing that gap. The
user asked the right question directly: does the literature actually build these models
with pure ideal gas? **No** — essentially all published gas-giant thermal-evolution codes
(Bodenheimer & Pollack 1986; Marley et al. 2007; and the subsequent literature generally)
use a non-ideal EOS including electron degeneracy, even for their hottest, youngest models,
because real planetary-mass objects are partially degenerate from formation onward (the
electron Fermi temperature at Jupiter's characteristic density is order $10^5$–$10^6$ K,
far above any plausible formation temperature — not a "late in cooling" effect, the common
but incorrect intuition). This directly explains finding 1 (ideal gas can't be compact at
this mass) and is a strong, literature-supported candidate for finding 2 as well (though
*why* the self-consistent construction specifically lands at ~27,000 R_Jup rather than
just failing to converge has not been directly diagnosed — see §1's "Open" list).

**Decision:** treat the missing non-ideal EOS as the leading hypothesis. PLAN.md's roadmap
was reorganized around it (new Sub-task 2f, inserted before Sub-task 5 resumes) rather than
continuing to debug the shooting method against ideal-gas-only physics. No EOS code has
been written yet — this entry documents the investigation and decision, not an
implementation.

### 2026-07-27 — Physical reassessment: the GI contradiction, the quasi-static limitation, and the compact hot-start pivot

A discussion/analysis pass (no code) that changed Sub-task 5's entire premise, triggered by
finding that the corrected radiative BC (previous entry) fixed the exact-degeneracy problem
but a bootstrapped/kicked version of the old Premise-1 diffuse clump still would not sustain
evolution past one step (see the two rejected fixes below). Three questions were worked
through:

1. **Is the Hayashi MMSN a self-consistent ambient condition for a GI-formed clump?** No.
   The MMSN is a smooth, quiescent-disk reconstruction with Toomre $Q\gg1$ (linearly
   stable) — a disk that actually fragments via GI at 50 AU must be locally much
   denser/more massive than that. Using MMSN conditions as confinement for a clump that
   supposedly formed by fragmenting that same disk is not self-consistent.
2. **Can a quasi-static code ever simulate the collapse itself?** No, structurally. Initial
   GI collapse is inertia-dominated free-fall; a hydrostatic-equilibrium solver assumes the
   opposite (force balance at every instant). This is exactly symmetric to why
   `T_DISSOCIATION_LIMIT` already halts the code at the far end of validity. Standard
   practice (PMS Henyey tracks; Bodenheimer & Pollack 1986; Marley et al. 2007) is to hand
   off from an assumed post-collapse state rather than simulate the collapse.
3. **What did the original "stall" actually mean?** Not that GI planet formation fails —
   that a smoothly pressure-confined, thermally-relaxed, non-self-gravitating-dominated
   parcel correctly has no reason to run away under fixed boundary conditions with no
   accretion. The code was being asked to quasi-statically evolve the *wrong phase*
   (pre-collapse), which is outside its domain of validity by construction, not a bug.

**Decision (user-approved, via Plan Mode): abandon the diffuse pre-collapse clump premise
entirely.** $t=0$ becomes a compact, hot, fully convective post-collapse protoplanet
(`config.T_CENTER_INITIAL`, prescribed, not derived — a standard "hot start" parameter,
Marley et al. 2007-style), eliminating the need for any bootstrap/kick mechanism (a
genuinely hot $t=0$ already has $T\neq T_\text{neb}$, so there's no isothermal degeneracy to
escape). This also potentially resolves an earlier finding (below, marked
**[SUPERSEDED]**) that a "hot start" needs impossible luminosities — that finding was
specific to forcing a hot $T_\text{center}$ into Premise 1's ~13 AU, $P_\text{neb}$-confined
geometry; a compact geometry supplies a vastly steeper natural pressure gradient from
self-gravity alone. (This turned out to be only partly right — see the next entry up.)

### 2026-07-27 — Sub-task 8 investigation: the isothermal fixed point, the radiative BC fix, and two rejected per-timestep mechanisms

Investigating why Sub-task 7's homologous bootstrap didn't produce visible evolution led to
proving the isothermal fixed point described in §1/§4.7 of PLAN.md — not a solver bug, an
exact property of the equations under a rigid $T=T_\text{neb}$ surface condition. Fixed by
replacing that condition with the net-flux radiative balance
(`boundary_conditions.py`, §5 module reference above; PLAN.md §4.7).

That fix alone was necessary but not sufficient: two further mechanisms were tried and
rejected before the physical reassessment (previous entry) reframed the problem correctly.

**Rejected: a one-time "kick" to displace $t=0$ off the isothermal fixed point.** Built
`solve_bootstrap_step`, which constructed a fresh, self-consistent isothermal equilibrium at
a slightly boosted reference temperature (via the homologous-contraction ansatz evaluated at
a small formal `dt_kick`). This did produce a measurably different, correctly-signed first
real step. But it was entirely specific to Premise 1's isothermal starting point and became
unnecessary once Premise 2 (this document, previous entry) made $t=0$ genuinely
non-isothermal from the start — removed along with the rest of the Premise-1-specific
machinery.

**Rejected: an explicit forcing term added to the per-step energy equation.** After the
kick fixed the first step, subsequent steps immediately stalled again (state relaxing back
toward $T\approx T_\text{neb}$, $L\to0$) — the corrected BC fixed "evolution is impossible
at all" but supplied no *ongoing* driver. Added the homologous-contraction rate as an
extra term inside `_implicit_rhs_logm`'s `dT_dt`, `dP_dt` (on top of the genuine implicit
state difference), which did produce a stable, non-decaying, correctly-signed $L$ — but
also an exactly-frozen $T$, $P$, $r$ step-to-step alongside it. That combination is a
direct energy-conservation violation (constant nonzero radiated power from a structure
that is not changing at all has no energy source) — proof the extra term was
double-counting compressional heating, not supplying genuine new physics. Reverted; see
PLAN.md §4.8.

### 2026-07-20 — Sub-task 7: `time_stepper.py` — homologous-contraction bootstrap implemented **[SUPERSEDED — see the 2026-07-27 pivot entries above]**

*The bootstrap mechanism described in this entry no longer exists in the intended
architecture — it was necessary only for Premise 1's isothermal $t=0$, which has since
been abandoned (see above). The derivation and numerical cross-checks below remain
correct as far as they go and are kept for reference, but `compute_time_derivatives`'s
bootstrap branch is now scheduled for removal (PLAN.md Sub-task 7), not further use.*

Implemented `compute_time_derivatives`, following the direction set in the physics-review
entry immediately below (that entry has the full reasoning for *why* a homologous ansatz,
not just *what*). Two things worth recording that emerged only once this was actually
built and tested, not just planned:

**The sign was worth double-checking carefully, and it's the opposite of the casual
"cooling" language used earlier.** Deriving the ansatz rigorously (every shell contracts as
r=r0·f(t); mass conservation forces ρ=ρ0/f³; hydrostatic equilibrium `dP/dm∝1/r⁴∝f⁻⁴` is
only satisfied at every instant if P=P0·f⁻⁴; the ideal gas law then forces T=T0·f⁻¹) gives
dT_dt=+T/t_KH and dP_dt=+4P/t_KH at t=0 - both *positive*. Contraction *raises* the
interior temperature - the standard negative-heat-capacity behavior of a self-gravitating
gas (losing energy via L>0 makes it hotter, not cooler), textbook Kelvin-Helmholtz physics,
but easy to get backwards without doing the derivation explicitly. Substituting both into
`odes.py`'s energy equation gives a clean closed form, dL/dm =
[(3γ-4)/(γ-1)]·k_BT/(μm_H·t_KH), confirmed to match the numerical result from
`odes.stellar_odes` to machine precision. It's positive only for γ>4/3 - which is exactly
the classical hydrostatic-stability threshold already underlying
`config.T_DISSOCIATION_LIMIT`'s justification (H2 dissociation drops γ_eff below 4/3,
triggering collapse) - a satisfying independent confirmation that this whole framework
hangs together, not something engineered to match. Cross-checked the integrated L(M_TOTAL)
against the completely independent `|E_grav|/t_KH` estimate (using `diagnostics.
virial_balance`'s E_grav): agree to within a factor of 2.24, real corroboration rather than
just internal self-consistency.

**Ran the "does solve_bvp work now" test the PLAN.md entry called for, rather than leaving
it for Sub-task 8 to discover the hard way.** Set up a full 4-ODE `solve_bvp` call (state_0
as initial guess, the bootstrap `dT_dt`/`dP_dt` as frozen source terms, the same
`boundary_conditions.py`) and ran it. Result is nuanced: the singular-Jacobian *crash* from
Sub-task 5's t=0 attempt is genuinely gone - 5 Newton iterations completed without
crashing, confirming the rank-deficiency prediction was correct. But it still does not
*converge* in any practical sense: residuals grow after iteration 2 and the mesh explodes
toward the node limit (status 1, boundary residuals ~1e7-1e9) - the same unnormalized
absolute-tolerance-across-wildly-different-scales problem that forced Sub-task 5 toward a
shooting method in the first place, now recurring for the full 4-ODE system. This finding
(solve_bvp unreliable regardless of source-term structure) remains valid and is why every
solve in this codebase now uses shooting (PLAN.md §4.2).

Added 3 validation checks (30-32): the bootstrap derivatives' positivity/analytic-formula/
energy-cross-check, a synthetic finite-difference check exercising the `np.interp`
grid-mismatch path, and a profile plot. **Checks 30 and 32 are now obsolete** (§4 above);
Check 31 (the synthetic finite-difference check, unrelated to the bootstrap) remains valid.

### 2026-07-20 — Physics review: hot-start t=0 reconsidered and rejected; Sub-task 7 bootstrap direction set **[PARTIALLY SUPERSEDED — see the 2026-07-27 pivot entries above]**

*This entry's numerical findings are still accurate and were directly useful this session
(the ~1900-350 billion L_sun luminosity requirements below are the same class of result
later reproduced for the compact-geometry case, just worse, before being traced to the
same root cause suspected now: missing degeneracy pressure). Its final conclusion — "t=0
stays cold/isothermal" — is superseded: Premise 2 (2026-07-27) instead moves t=0 itself to
a hot, compact start, precisely the option this entry considered and rejected, but at a
*different* (compact, not diffuse) radius, which changes the energetics substantially
(though not, it turned out, enough on its own without also addressing the EOS).*

No code changed in this entry — a discussion/analysis pass, prompted by revisiting whether
Sub-task 5's cold, isothermal, L=0 result (a mathematical fixed point that never
spontaneously evolves) should instead be replaced with a "hot start": an adiabatic,
convective interior at T_center ~ 600-1500K, framed as the post-free-fall-collapse "first
core" state, matched hydrostatically against P_neb for the full M_TOTAL via the same
shooting-method infrastructure as Sub-task 5.

**Rejected, with fresh numerical evidence.** Recomputed the required luminosity to sustain a
genuinely convective structure (∇_rad=∇_ad, "efficient convection" closure - same method
validated in the Sub-task 5 investigation) at T_center = 700, 1000, 1500K, using the
project's actual γ=1.4. Results: ~1,940 L☉ at 700K, ~61 million L☉ at 1000K, ~352 billion L☉
at 1500K - a steep runaway, not an improvement over the single data point already ruled out
in Sub-task 5. Root cause identified at the time: M_TOTAL/M_BE≈0.089 forces the pressure
profile to stay nearly uniform regardless of the assumed interior temperature — **this
diagnosis was correct for Premise 1's diffuse (~13 AU) geometry specifically**; it does not
directly apply to Premise 2's compact geometry (self-gravity supplies a much steeper natural
pressure gradient there), which is why the pivot seemed promising — though the compact
geometry brought its own, different problems (previous entries), now suspected to trace to
the same underlying missing physics (no degeneracy pressure) from a different angle.

**Conclusion at the time: t=0 stays the cold, isothermal, L=0 state (Sub-task 5,
unchanged).** Nonzero L was introduced via Sub-task 7's bootstrap instead (entry above).
**This conclusion no longer holds** — see the 2026-07-27 entries.

### 2026-07-20 — Sub-task 6: `diagnostics.py` — exit criteria revised for the cold t=0 state **[SUPERSEDED — see PLAN.md's Sub-task 6 entry]**

*Written for Premise 1's isothermal, pressure-confined t=0 state. The pressure-confined
virial form and single-regime opacity prediction described here no longer apply under
Premise 2 (PLAN.md Sub-task 6) — kept for the derivation technique (integrating hydrostatic
equilibrium by parts), which will be reused in standard (unconfined) form once Sub-task 5
is unblocked.*

PLAN.md's original Sub-task 6 (virial check against the textbook zero-surface-pressure
coefficient, regime distribution expected to span cold-outer to hot-inner regimes, an
energy-flux check against a nonzero L) assumed the generic converged structure envisioned
before Sub-task 5's investigation. Given what Sub-task 5 actually produced at the time
(cold, uniform T=50K, L≡0), those criteria were revised: a pressure-confined virial form
`E_grav + 3(γ-1)E_therm = 3·P_neb·V` (derived by integrating hydrostatic equilibrium by
parts, verified against the converged state: relative imbalance ~8e-6), a single-regime
("Ice grains") opacity prediction, an energy-flux check deferred to Sub-task 7, and a new
mass-reconstruction check (`M(r)=∫4πr²ρ dr` vs. the Lagrangian grid, independent of the
above, still valid and expected to transfer to Premise 2's structure directly).

`diagnostics.py` itself holds no asserts (unlike `validation.py`) — per CLAUDE.md's
architecture, it's the operational reporting/monitoring layer, producing physical
quantities for a physicist to read, not a test suite.

### 2026-07-20 — Sub-task 5: `bvp_solver.py` — full reset after a physics investigation **[SUPERSEDED — see the 2026-07-27 pivot entries above]**

*This entry documents Premise 1 in full: why `solve_bvp` was abandoned for shooting (§4.2
of PLAN.md — this finding is still valid and unchanged), why `T_NEB`/`P_NEB` were corrected
to Hayashi MMSN values (still valid — these are still the project's nebula constants), the
Bonnor-Ebert subcriticality calculation (still true as a fact about these constants, though
its physical role has changed — see PLAN.md's Sub-task 5 entry), and the original
isothermal/L=0 result and four rejected hot-start attempts (superseded as the intended t=0
state, but the specific numerical findings — required luminosities, the mechanism by which
Bonnor-Ebert subcriticality forces a near-uniform pressure profile — remain accurate and
were directly relevant background for this session's investigation).*

**1. `solve_bvp` itself doesn't work for this problem, independent of physics.** First
attempt: call `scipy.integrate.solve_bvp` on the full 4-ODE system with `dT_dt=dP_dt=0`,
per PLAN.md's literal original Sub-task 5 deliverable. It failed with a singular Jacobian on
iteration 1, every time, regardless of initial guess, mesh spacing, or independent-variable
transform. Diagnosis: `dL/dm = -c_p·dT_dt + dP_dt/ρ` depends *only* on the externally-
prescribed, frozen `dT_dt`/`dP_dt` arrays — never on the state being solved for. That makes
the ODE Jacobian's L-row exactly zero at every point (confirmed by direct numerical
computation: rank 2 of 4), for *any* `dT_dt`/`dP_dt`. Reducing to the 2-ODE `(r,P)`
subsystem removed that specific degeneracy, but `solve_bvp` still failed: near the surface,
P approaches the much smaller `P_neb` with a short pressure scale height, driving
`d(ln P)/d(ln m)` to enormous values that broke every collocation mesh/scaling strategy
tried. Resolution: abandon `solve_bvp` for this module entirely, use a shooting method
instead.

**2. The `L≡0` mathematical trap (Premise 1 specific).** With `dT_dt=dP_dt=0` (no previous
timestep — literally what "the first, static solve" meant under Premise 1), `dL/dm≡0`
identically, and with the center BC `L(0)=0` this forced `L≡0` everywhere, forcing
`∇_rad≡0<∇_ad`, forcing `dT/dm≡0`: the converged $t=0$ envelope was exactly isothermal at
`T_neb`. This specific mechanism is why Premise 1's $t=0$ was isothermal; it doesn't
directly apply to Premise 2 (which prescribes $T_\text{center}\neq T_\text{neb}$ and never
sets `dT_dt=dP_dt=0` in the first place).

**3. The original `T_NEB=150K`, `P_NEB=1e4` were physically wrong — still true, unchanged.**
Checked against the Bonnor-Ebert critical mass: `M_TOTAL/M_BE ≈ 99` at those values — 99×
over the critical mass, no stable isothermal equilibrium exists there at all. Root cause:
those values described inner-disk-like conditions, inconsistent with the GI/disk-
fragmentation scenario (outer disk, ~50 AU). Corrected using the Hayashi (1981) MMSN model
at 50 AU: `T_NEB=50.0 K`, `P_NEB=1.0e-4 dyn/cm²` (within factors of 1.3 and 2.5 of the MMSN
reference values). This dropped `M_TOTAL/M_BE` to 0.089. **These are still the project's
values** — nothing in this session's investigation changed them.

**4. Four independent "hot start" attempts, at Premise 1's diffuse geometry, all needed
$L\sim10^{34}$–$10^{37}$ erg/s.** (single global adiabat + assumed `L(m)∝m`; single adiabat
+ L solved for marginal convection; two-zone adiabatic-core + radiative-envelope; a
photospheric `L=4πR²σT_eff⁴` construction). Root cause: `M_TOTAL` being deeply
Bonnor-Ebert-subcritical forces the pressure profile to be nearly uniform regardless of the
assumed interior temperature distribution, at the ~13 AU geometry these four attempts all
used. **This is the same class of finding reproduced this session for the compact geometry**
(albeit for a different underlying reason — see the 2026-07-27 entries) — the two are
related but not identical results.

**Implementation at the time:** shooting on central pressure via `solve_ivp` + `brentq`,
holding `L=0`, `T=T_neb` fixed. Converged to `R_surface≈13.0 AU`. Added 5 validation checks
(21–25). **All of this — the isothermal construction itself, and Checks 24/25/26/27's
specific assertions — has since been superseded**; see §4 above for current check status.

### 2026-07-20 — CLAUDE.md rule: prefer visible checks where they fit
Added a standing rule to the Testing & Validation Protocol: prefer a visible check (a
plot of a profile, a residual vs. a coordinate, a comparison curve) over a print-only
assert whenever a check naturally has something to look at. Still in effect, unaffected by
anything above.

### 2026-07-20 — Sub-task 4: `odes.py` + `boundary_conditions.py`
Implemented `stellar_odes` (the full 4-equation RHS) and `boundary_conditions` (the 4
residuals closing the system). Added 4 validation checks (17–20) after proposing them and
getting sign-off. Check 17's first draft surfaced a genuine self-consistency bug in the
*test* (using one T(m) array for all three of dr/dm, dP/dm, dT/dm, when dr/dm alone depends
on ρ derived internally from the EOS) — not in `odes.py` itself; fixed by using two
different T arrays for the different comparisons. `odes.py` is unchanged since; only
`boundary_conditions.py`'s surface thermal residual has since been revised (§4.7 of
PLAN.md, this session).

### 2026-07-20 — Sub-task 3: `gradients.py` (Schwarzschild criterion)
Implemented `grad_radiative` and `effective_gradient`. Deviated from the PLAN.md deliverable
signature by dropping the unused `rho` parameter from `grad_radiative` (the formula never
uses density). Added 5 validation checks (12–16). Unchanged since, apart from gaining
`marginal_convective_luminosity` this session (§2 above).

### 2026-07-20 — Progress log established
Created this file retroactively to document Sub-tasks 1 and 2a–2e, which were already
implemented and committed (`e58af91`). Added a standing rule to `CLAUDE.md` requiring this
file to be updated after every subsequent task.

### Prior work (commit `e58af91`)
Implemented `config.py`, `state.py`, `eos.py`, `opacity.py`, and `validation.py`
end-to-end, covering PLAN.md Sub-tasks 1 and 2a–2e. Resolved a real bug in opacity regime
assignment at low density (spurious low-temperature crossing between the Kramers and
electron-scattering regimes) via monotonic clamping. 11 validation checks and 2 diagnostic
plots were added.

### Commit `95f923a` — PLAN.md established
Wrote the architecture and development plan that all subsequent work is tracked against.
