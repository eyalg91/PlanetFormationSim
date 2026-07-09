# PlanetFormationSim — Architecture & Development Plan

**Project:** 1D Quasi-Static Planetary Gas Envelope Collapse (Kelvin-Helmholtz Contraction)
**Course:** Computational Physics
**Last Updated:** 2026-07-06

---

## Table of Contents

1. [Physics Overview](#1-physics-overview)
2. [Module Structure](#2-module-structure)
3. [Architecture Data-Flow Diagram](#3-architecture-data-flow-diagram)
4. [Key Design Decisions](#4-key-design-decisions)
5. [Sequential Sub-Tasks Breakdown](#5-sequential-sub-tasks-breakdown)

---

## 1. Physics Overview

### Goal
Simulate the 1D quasi-static spherical collapse of a protoplanetary gas envelope via Kelvin-Helmholtz contraction. No hydrodynamics — only time-evolution of structural (stellar-interior-type) equations on a Lagrangian mass grid.

### Grid
- **Coordinate:** Lagrangian mass coordinate $m$ (mass enclosed within radius $r$)
- **Domain:** $m = 0$ (center) to $m = M_\text{total}$ (surface)

### System of 4 ODEs

| # | Equation | Physical meaning |
|---|---|---|
| 1 | $\dfrac{dr}{dm} = \dfrac{1}{4\pi r^2 \rho}$ | Continuity / mass-radius relation |
| 2 | $\dfrac{dP}{dm} = -\dfrac{Gm}{4\pi r^4}$ | Hydrostatic equilibrium |
| 3 | $\dfrac{dL}{dm} = -c_p \dfrac{\partial T}{\partial t} + \dfrac{1}{\rho}\dfrac{\partial P}{\partial t}$ | Energy equation (KH contraction source) |
| 4 | $\dfrac{dT}{dm} = \dfrac{T}{P} \nabla_\text{eff} \dfrac{dP}{dm}$ | Temperature structure (Schwarzschild criterion) |

### Constitutive Relations
- **EOS:** Ideal gas — $P = \dfrac{\rho k_B T}{\mu m_H}$
- **Opacity:** Bell & Lin (1994) piecewise power-law — $\kappa = \kappa_i \rho^a T^b$ across 8 regimes (see Section 4)
- **Temperature gradient:** Schwarzschild criterion — $\nabla_\text{eff} = \nabla_\text{rad}$ (radiative) if $\nabla_\text{rad} < \nabla_\text{ad}$; otherwise $\nabla_\text{eff} = \nabla_\text{ad}$ (convective)
  - $\nabla_\text{rad} = \dfrac{3\kappa L P}{64\pi a c G m T^4}$
  - $\nabla_\text{ad} = \dfrac{\gamma - 1}{\gamma}$ for ideal gas

### Boundary Conditions

| Location | Condition |
|---|---|
| Center ($m = 0$) | $r = 0$, $L = 0$ |
| Surface ($m = M_\text{total}$) | $P = P_\text{nebula}$, $T = T_\text{nebula}$ |

### Numerical Approach
- **Inner spatial solver:** Boundary Value Problem (BVP) solved at each time step using `scipy.integrate.solve_bvp` (collocation method). State vector: $\mathbf{y} = [r, P, L, T]$, independent variable $m$.
- **Outer time loop:** Advances time by $\Delta t$; finite-differences $\partial T/\partial t$ and $\partial P/\partial t$ from the previous step's state; injects them as frozen source terms into ODE 3; calls the BVP solver.

---

## 2. Module Structure

```
PlanetFormationSim/
├── main.py                   # Orchestrator: parse config, run time loop, save output
├── config.py                 # Physical constants, nebula params, grid settings, flags
├── state.py                  # SimulationState dataclass: all field arrays + time
├── eos.py                    # Constitutive relations (ideal gas: ρ, c_p, ∇_ad)
├── opacity.py                # Bell & Lin (1994) 8-regime piecewise opacity
├── gradients.py              # ∇_rad, ∇_ad, Schwarzschild criterion → ∇_eff
├── odes.py                   # RHS of the 4-ODE system for solve_bvp
├── boundary_conditions.py    # BVP boundary residuals (center + surface)
├── bvp_solver.py             # Wrapper: initial guess, call solve_bvp, handle failures
├── time_stepper.py           # Outer loop: compute ∂T/∂t, ∂P/∂t, advance state
├── diagnostics.py            # Virial theorem, energy conservation, regime logging
└── output.py                 # NPZ snapshots, matplotlib plotting helpers
```

### Module Responsibilities (one-line each)

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for all numerical values and simulation flags; no computation |
| `state.py` | `SimulationState` dataclass holding the current and previous grid + field arrays |
| `eos.py` | Vectorized ideal-gas constitutive relations; no side effects |
| `opacity.py` | Bell & Lin piecewise opacity with density-dependent regime boundaries |
| `gradients.py` | Schwarzschild switch; returns $\nabla_\text{eff}$ and a boolean `is_convective` mask |
| `odes.py` | Pure function: takes $(m, \mathbf{y}, \dot{T}, \dot{P})$ and returns $d\mathbf{y}/dm$ |
| `boundary_conditions.py` | Pure function: returns 4 residuals for `solve_bvp` BC interface |
| `bvp_solver.py` | Manages initial guesses, calls `solve_bvp`, logs convergence, returns new state |
| `time_stepper.py` | Outer time loop; finite-differences time derivatives; dispatches to BVP solver |
| `diagnostics.py` | Post-solve checks; virial theorem; opacity regime distribution per timestep |
| `output.py` | Snapshot I/O and matplotlib helpers; no physics |

---

## 3. Architecture Data-Flow Diagram

```mermaid
graph TD
%% Files and Modules
config["config.py

(Constants, Parameters)"]
state["SimulationState

(m, r, P, T, L, ρ, t)"]
eos["eos.py

(Ideal Gas: ρ, c_p, ∇_ad)"]
opacity["opacity.py

(Bell & Lin 1994 Piecewise)"]
gradients["gradients.py

(Schwarzschild switch → ∇_eff)"]
odes["odes.py

(4 ODEs: dr/dm, dP/dm, dL/dm, dT/dm)"]
bcs["boundary_conditions.py

(Center & Surface BCs)"]
bvp["bvp_solver.py

(solve_bvp Wrapper)"]
timeloop["time_stepper.py

(Calculates ∂T/∂t, ∂P/∂t)"]

%% Data Flow
config -->|Initializes| state
state -->|Provides previous step| timeloop
timeloop -->|Passes Time Derivatives| bvp

%% The BVP Solver inner workings
bvp -->|Evaluates| odes
bvp -->|Checks| bcs

%% ODE dependencies
eos -->|Density, Heat Capacity| odes
gradients -->|Temperature Gradient| odes
opacity -->|kappa (κ)| gradients
eos -->|Adiabatic Gradient| gradients

%% Output
bvp -->|Updates & Saves| state
```

---

## 4. Key Design Decisions

### 4.1 `SimulationState` as the Central Data Object

A `@dataclass` holding `m` (mass grid), `r`, `P`, `T`, `L`, `rho` (numpy arrays), current time `t`, and a reference to the previous state. All modules receive state and return new state — none hold mutable state themselves. This enforces functional purity and simplifies testing.

### 4.2 `solve_bvp` Formulation

- State vector: $\mathbf{y} = [r, P, L, T]$, independent variable $m$
- `fun(m, y, dT_dt, dP_dt)` → $d\mathbf{y}/dm$ (time derivatives are frozen source arrays interpolated from the previous timestep)
- `bc(ya, yb)` → 4 residuals: `[ya[0], ya[2], yb[1] - P_neb, yb[3] - T_neb]`
- At $t = 0$: `dT_dt` and `dP_dt` are zero arrays → first solve is purely static

### 4.3 Bell & Lin (1994) Opacity — Density-Dependent Regime Boundaries

The transition between regime $n$ and $n+1$ is defined by $\kappa_n = \kappa_{n+1}$, giving a transition temperature that depends on $\rho$:

$$T_{n \to n+1}(\rho) = \left[\frac{\kappa_i^{(n+1)}}{\kappa_i^{(n)}} \cdot \rho^{(a_{n+1} - a_n)}\right]^{1/(b_n - b_{n+1})}$$

The 8 regimes (CGS units, $\kappa$ in cm² g⁻¹):

| # | Physical process | $\kappa_i$ | $a$ | $b$ |
|---|---|---|---|---|
| 1 | Ice grains | $2 \times 10^{-4}$ | 0 | 2 |
| 2 | Ice grain evaporation | $2 \times 10^{16}$ | 0 | −7 |
| 3 | Metal grains | $0.1$ | 0 | 1/2 |
| 4 | Metal grain evaporation | $2 \times 10^{1}$ | 1 | −24 |
| 5 | Molecules | $10^{-8}$ | 2/3 | 3 |
| 6 | H⁻ scattering | $10^{-36}$ | 1/3 | 10 |
| 7 | Bound-free/free-free (Kramers) | $1.5 \times 10^{20}$ | 1 | −5/2 |
| 8 | Electron scattering | $0.348$ | 0 | 0 |

**Internal structure of `opacity.py`:**
- **Layer 1 — Data:** Immutable `RegimeParams` namedtuples (one per row above)
- **Layer 2 — Transitions:** `transition_temperature(rho, n)` → $T_{n \to n+1}(\rho)$; `determine_regime(rho, T)` → integer index, fully vectorized
- **Layer 3 — Public API:** `bell_lin_opacity(rho, T)` → $\kappa$, vectorized

**Smooth-transition flag:** $\kappa(T)$ is continuous at transitions by construction, but $d\kappa/dT$ has a kink, which can perturb the collocation Jacobian. `config.py` exposes `OPACITY_SMOOTH_TRANSITIONS: bool = False`. When `True`, a logistic blend of width $\delta\!\log T \approx 0.05$ dex softens the kink. Default is `False` (physically correct hard switch).

### 4.4 `gradients.py` Interface Contract

Returns a tuple `(grad_eff, is_convective)` where `is_convective` is a boolean numpy array. This mask is passed to `diagnostics.py` for per-timestep regime reporting, and can be used by `output.py` to shade convective zones on temperature-profile plots.

### 4.5 Adaptive Time-Stepping

$\Delta t \leq \alpha \cdot \min_i\!\left(T_i / |\dot{T}_i|\right)$ — a local thermal timescale limiter. The safety factor $\alpha$ is set in `config.py`. The fixed-dt path is retained for reproducibility and comparison.

---

## 5. Sequential Sub-Tasks Breakdown

### Phase 1 — Static Skeleton & Physical Validation

---

#### Sub-task 1 — `config.py` + `state.py`

**Goal:** Establish the single source of truth for all constants and the central data container.

**Deliverables:**
- All CGS physical constants: $G$, $c$, $a_\text{rad}$, $k_B$, $m_H$, $\sigma_\text{SB}$
- Simulation parameters: $M_\text{total}$, $P_\text{neb}$, $T_\text{neb}$, $\mu$, $\gamma$, `n_grid_points`, `OPACITY_SMOOTH_TRANSITIONS`
- `SimulationState` dataclass with typed numpy array fields and a `prev` reference slot

**Exit criterion:** Print all constants; confirm CGS unit self-consistency by hand for 3 key relations (e.g., $P = \rho k_B T / \mu m_H$ gives Pa when inputs are CGS).

---

#### Sub-task 2 — `eos.py` + `opacity.py`

This sub-task is split into five sequential steps due to the complexity of the Bell & Lin opacity model.

**Sub-task 2a — `eos.py` and raw power-law evaluator**

- Implement `density(P, T, mu)`, `specific_heat_cp(gamma, mu)`, `grad_adiabatic(gamma)` — all vectorized
- Define the `RegimeParams` namedtuple and the immutable `REGIMES` table (all 8 rows)
- Implement `evaluate_regime(kappa_i, a, b, rho, T)` — a single vectorized power-law call

*Exit criterion:* Each of the 8 rows evaluates correctly at a hand-verified (ρ, T) reference point.

**Sub-task 2b — Transition temperature function**

- Implement `transition_temperature(rho, n)` using the analytic formula in Section 4.3
- Handle the degenerate case $b_n = b_{n+1}$ (log a warning; this would indicate a data error)
- Plot $T_{n \to n+1}(\rho)$ over $\rho \in [10^{-15},\, 10^{-5}]$ g cm⁻³ for all 7 transitions

*Exit criterion:* Transition curve slopes in log-log space match the analytic exponent $[(a_{n+1}-a_n)/(b_n - b_{n+1})]$ for each pair.

**Sub-task 2c — `determine_regime` and `bell_lin_opacity`**

- Implement `determine_regime(rho, T)` vectorized; compute all 7 transition temperatures per point and find the bin (e.g., via `np.searchsorted` on the sorted transition array)
- Implement `bell_lin_opacity(rho, T)` as the sole public API function; dispatches to the correct regime using array indexing / `np.where`

*Exit criterion:* Function accepts and returns numpy arrays of arbitrary shape with no Python-level loops; no NaN or Inf over the physically relevant domain.

**Sub-task 2d — Validation suite (4 specific checks)**

| Check | Method | Pass Condition |
|---|---|---|
| **Regime continuity** | At each of the 7 transitions, evaluate $\kappa$ from both adjacent regimes at $(ρ,\; T_{n \to n+1})$ | Relative difference $< 10^{-10}$ |
| **Regime ordering in T** | At fixed $\rho = 10^{-10}$ g cm⁻³, sweep T from 100 K to 50 000 K; log regime index | Index is monotonically non-decreasing |
| **Reference point check** | Evaluate $\kappa$ at 3–4 (ρ, T) pairs traceable to Bell & Lin (1994) | Agree within the paper's own rounding |
| **Vectorization stress test** | Pass a 2D mesh covering all 8 regimes simultaneously | Output shape matches input; no NaN/Inf |

**Sub-task 2e — `opacity.py` ↔ `gradients.py` interface preview**

- Call `bell_lin_opacity` along a representative 1D profile (varying T with depth, ρ from a polytropic estimate)
- Plot $\kappa(m)$ and visually confirm that regime transitions occur at physically plausible depths

*Exit criterion:* Profile plot shows expected qualitative behavior (high $\kappa$ from grain/molecular opacity in cool outer layers, Kramers-like behavior in hot inner regions).

---

#### Sub-task 3 — `gradients.py`

**Goal:** Implement and validate the Schwarzschild criterion over the full temperature range.

**Deliverables:**
- `grad_radiative(L, m, P, T, kappa, rho)` using $\nabla_\text{rad} = \dfrac{3\kappa L P}{64\pi a c G m T^4}$
- `effective_gradient(grad_rad, grad_ad)` → `(grad_eff, is_convective)`
- Assert $\kappa > 0$ at entry (guards against upstream errors in `opacity.py`)

**Exit criterion:**
- Verify $\nabla_\text{rad} > \nabla_\text{ad}$ triggers convection at high $L$ or high $\kappa$
- Run validation over $T \in [100\,\text{K},\; 50\,000\,\text{K}]$ to exercise all opacity regimes
- Confirm `is_convective` mask is correct for both limiting cases

---

#### Sub-task 4 — `odes.py` + `boundary_conditions.py`

**Goal:** Formulate the 4-ODE RHS and BVP boundary residuals.

**Deliverables:**
- `stellar_odes(m, y, dT_dt, dP_dt, params)` → `[dr/dm, dP/dm, dL/dm, dT/dm]`
  - `y = [r, P, L, T]`; `rho` derived internally from EOS
  - `dT_dt`, `dP_dt` are pre-interpolated frozen arrays (zero at $t=0$)
- `boundary_conditions(ya, yb, params)` → 4 residuals:
  `[ya[0], ya[2], yb[1] - P_neb, yb[3] - T_neb]`

**Exit criterion:**
- Feed in a known analytic profile (constant-density sphere); verify each ODE term's numerical magnitude is dimensionally consistent
- Confirm 4 ODEs + 4 BCs → well-posed system

---

#### Sub-task 5 — `bvp_solver.py` (static solve, $t = 0$)

**Goal:** Obtain the first converged equilibrium structure.

**Deliverables:**
- Construct an initial guess (linear / polytropic profiles for $r$, $P$, $T$; zero $L$)
- Call `scipy.integrate.solve_bvp` with zero time derivatives
- Log solver status and residual norm; raise `RuntimeError` on hard failure, warning on soft failure
- Return a populated `SimulationState`

**Exit criterion:**
- Plot $r(m)$, $P(m)$, $T(m)$, $L(m)$ for the converged solution
- Verify hydrostatic balance pointwise: $|dP/dr + G M(r) \rho / r^2| / |dP/dr| < 10^{-3}$ at interior points
- $L(\text{surface})$ is physically reasonable (compare to Kelvin-Helmholtz luminosity estimate $L_{KH} \sim GM^2 / (Rt_{KH})$)

---

#### Sub-task 6 — `diagnostics.py` (static checks)

**Goal:** Verify the $t=0$ solution satisfies global energy constraints.

**Deliverables:**
- Virial theorem check: $|E_\text{grav}| \approx 2|E_\text{therm}|$ for ideal gas
- Opacity regime distribution: log the fraction of grid points in each Bell & Lin regime
- Energy flux check: $L(\text{surface}) = \int \epsilon \, dm$ (if an $\epsilon$ source term is defined)

**Exit criterion:** Virial theorem deviation $< 1\%$; regime distribution printed and physically sensible (outer grid in grain/molecular regimes, inner grid in Kramers/electron-scattering regime).

---

### Phase 2 — Dynamic Time Evolution

---

#### Sub-task 7 — Time derivative computation in `time_stepper.py`

**Goal:** Implement the finite-difference bridge between timesteps.

**Deliverables:**
- `compute_time_derivatives(state_curr, state_prev, dt)` → `(dT_dt, dP_dt)` as numpy arrays on the current grid
- Bootstrap: for $t = 0 \to t = 1$, return zero arrays
- Interpolation: if grids shift between steps, interpolate `state_prev` fields onto `state_curr.m` before differencing

**Exit criterion:** Derivative arrays have correct shape and magnitudes consistent with KH timescale estimates ($t_{KH} \sim GM^2/RL \sim 10^6$ yr for a Jupiter-mass envelope).

---

#### Sub-task 8 — Outer time loop in `time_stepper.py`

**Goal:** Wire the full time-evolution loop.

**Deliverables:**
- `run(n_steps, dt)`: at each step, compute time derivatives → call BVP solver → update state → call diagnostics → save snapshot
- Log BVP convergence status at each step; warn (do not raise) on soft convergence failures
- Store snapshots at configurable intervals

**Exit criterion:** Run 10 steps with a large $\Delta t$; confirm $r_\text{surface}$ decreases (envelope contracts) and $L_\text{surface}$ increases monotonically.

---

#### Sub-task 9 — Adaptive time-stepping

**Goal:** Ensure numerical stability over long runs.

**Deliverables:**
- Thermal timescale limiter: $\Delta t_\text{new} = \alpha \cdot \min_i(T_i / |\dot{T}_i|)$ with safety factor $\alpha$ from `config.py`
- Fixed-dt path retained via a `USE_ADAPTIVE_DT: bool` flag in `config.py`

**Exit criterion:** Compare fixed vs. adaptive $\Delta t$ runs; adaptive run conserves total energy measurably better over 100 steps.

---

#### Sub-task 10 — `output.py` (snapshots and plots)

**Goal:** Reproducible data output and diagnostic visualizations.

**Deliverables:**
- Save each snapshot as `.npz` (mass grid + all fields + time + `is_convective` mask)
- Post-processing script: $r(t)$, $L_\text{surface}(t)$, $T_\text{center}(t)$ evolution curves
- 2-panel profile plots ($P(m)$ and $T(m)$) at selected snapshots; convective zones shaded using `is_convective` mask
- Opacity regime plot: $\kappa(m)$ colored by regime index

**Exit criterion:** All plots generated from saved `.npz` files without re-running the simulation.

---

### Phase 3 — Extensions (Post-Course)

| # | Description | Key dependency |
|---|---|---|
| 11 | Replace Bell & Lin with OPAL tabulated opacity (bilinear interpolation) | `opacity.py` Layer 3 API unchanged |
| 12 | Solid core inner BC: $m = M_\text{core} > 0$, $r = R_\text{core}$ fixed | `boundary_conditions.py` |
| 13 | Accretion luminosity surface term | `boundary_conditions.py` + `time_stepper.py` |

---

## Implementation Order Summary

| # | File(s) | Depends on | Exit Criterion |
|---|---|---|---|
| 1 | `config.py`, `state.py` | — | CGS unit consistency |
| 2a | `eos.py`, regime table | 1 | 8 reference-point checks |
| 2b | `transition_temperature` | 2a | Log-log slope matches analytic |
| 2c | `determine_regime`, `bell_lin_opacity` | 2b | Vectorized, no NaN/Inf |
| 2d | Validation suite | 2c | 4 checks pass |
| 2e | Interface preview | 2d, 3 scaffold | $\kappa(m)$ profile sensible |
| 3 | `gradients.py` | 2c | Schwarzschild switch correct over full T range |
| 4 | `odes.py`, `boundary_conditions.py` | 1–3 | Dimensional consistency check |
| 5 | `bvp_solver.py` (static) | 4 | Converged $t=0$ structure |
| 6 | `diagnostics.py` | 5 | Virial theorem $<1\%$ error |
| 7–8 | `time_stepper.py` | 5–6 | Envelope contracts over time |
| 9 | Adaptive $\Delta t$ | 7–8 | Better energy conservation |
| 10 | `output.py` | all | Reproducible plots from `.npz` |
