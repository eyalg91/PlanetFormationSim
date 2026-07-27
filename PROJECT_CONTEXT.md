# PlanetFormationSim — Project Context

**Purpose of this file:** a fast-orientation summary of everything established so far —
the physical scenario, the key numbers, the architecture decisions (and why they deviate
from the original plan), and what's open. Read this first in a new session; go to
[PLAN.md](PLAN.md) for the forward-looking module-by-module architecture and
[PROGRESS.md](PROGRESS.md) for the full chronological reasoning behind each decision
(including the debugging dead ends — useful if you're re-deriving *why*, not just *what*).

---

## 1. What this simulates

A 1D quasi-static Kelvin-Helmholtz contraction of a gas giant forming via **gravitational
instability (disk fragmentation)** — a clump of gas in the outer protoplanetary disk becomes
self-gravitating, decouples from the disk, and slowly contracts and heats up over time as it
radiates away energy, until (outside this project's scope) hydrogen dissociation ends the
quasi-static regime.

- **Not** core accretion, **not** star formation.
- Pure gas envelope: no solid core, no accretion of additional mass — `M_TOTAL` is fixed for
  the entire simulation at ~1 Jupiter mass (`1.898e30 g`).
- Lagrangian mass coordinate `m ∈ [0, M_TOTAL]`; 4 coupled ODEs (continuity, hydrostatic
  equilibrium, energy, temperature structure) solved on that grid at each timestep.

## 2. The nebula environment — MMSN at 50 AU

`config.T_NEB = 50.0 K`, `config.P_NEB = 1.0e-4 dyn/cm²` — the outer-disk midplane
conditions where GI actually occurs, validated against the Hayashi (1981) minimum-mass
solar nebula model at r=50 AU (`T(r)=280·(r/AU)^-0.5` → 39.6 K; midplane pressure via
`Σ_gas(r)=1700·(r/AU)^-1.5` + vertical hydrostatic balance → 4.0e-5 dyn/cm²). Our values are
within factors of 1.3 and 2.5 of that reference — solid agreement.

**These were not the original values.** `config.py` originally had `T_NEB=150K`,
`P_NEB=1e4` (inner-disk-like), which is physically wrong for this scenario and made
`M_TOTAL` **99× the Bonnor-Ebert critical mass** — no stable hydrostatic equilibrium exists
at all under those conditions. The MMSN correction dropped this to `M_TOTAL/M_BE ≈ 0.089` —
deeply *subcritical*, meaning a stable equilibrium genuinely exists (no forced collapse).

## 3. The t=0 state: cold, isothermal, non-luminous — and why

**Mathematically forced, not a choice:** with `dT_dt = dP_dt = 0` (the literal definition of
a static solve — no previous timestep to difference against), `odes.py`'s energy equation
gives `dL/dm ≡ 0` identically. Combined with the center BC `L(0)=0`, this forces **`L≡0`
everywhere**, which forces `∇_rad≡0 < ∇_ad` always, which forces **`dT/dm≡0`**: the
converged envelope is exactly isothermal at `T_neb`, for any solver or initial guess. This
is a hard consequence of the equations as written, confirmed independently in two separate
investigations (Sub-task 5's initial implementation, and the later "should we hot-start
instead" review).

**Also the physically correct picture:** a freshly-fragmented GI clump in equilibrium with
its disk, not yet contracting, matches the literature (Boss, Mayer, Helled & Bodenheimer:
newly-fragmented clumps are typically tens of AU and near the local disk temperature).
Confirmed numerically: converges to **R_surface ≈ 13.0 AU**, `P_center ≈ 1.227e-4 dyn/cm²`,
surface-pressure residual at machine precision.

### The "hot start" alternative was investigated thoroughly and rejected

Multiple attempts to construct a hot (`T_center` ~600–1500K), luminous, convective t=0 state
instead (motivated by the "first hydrostatic core" concept from star-formation theory) were
tested — single global adiabat with an assumed or solved-for `L(m)`, a two-zone
core+envelope construction, a photospheric boundary condition, and a fixed-`T_center` sweep.
**All require physically absurd luminosities:**

| T_center | Required L | In solar luminosities |
|---|---|---|
| 700 K | 7.4×10³⁶ erg/s | ~1,940 L☉ |
| 1000 K | 2.4×10⁴¹ erg/s | ~61 million L☉ |
| 1500 K | 1.3×10⁴⁵ erg/s | ~352 billion L☉ |

**Root cause:** `M_TOTAL/M_BE ≈ 0.089` forces the pressure profile to stay nearly uniform
(center-to-surface within a factor of a few) *regardless of the assumed internal
temperature*. Cramming a large temperature range into that narrow a pressure range demands
an enormous `dT/dm`, and radiative diffusion says that demands an enormous L to sustain.
This isn't fixable by tuning `T_center` — higher `T_center` makes it dramatically worse
(~L∝T_center¹¹ in this regime), not better.

There's also a logical inconsistency in the "post-collapse first core" framing: it argues
from "if it crosses the critical mass, it collapses," but `M_TOTAL/M_BE≈0.089` is precisely
the statement that it does *not* cross the critical mass. Real GI-formation and
pre-main-sequence codes don't re-derive a hot initial model this way — they hand off an
assumed initial radius/entropy for an already-*compact* object (few-R_Jup scale) from a
separate collapse calculation, at which point `P_neb` is irrelevant to the bulk structure
(self-gravity dominates by orders of magnitude) and only sets a thin outer atmosphere. Our
current architecture enforces `P(M_TOTAL)=P_neb` across the entire mass at all times —
correct for Sub-task 5's diffuse, disk-pressure-confined clump, but a structural mismatch
for a compact hot-start object that a different `T_center` can't paper over.

**Conclusion:** t=0 stays cold/isothermal/L=0. Nonzero L comes from Sub-task 7's bootstrap
(now implemented, §6), not from reconstructing t=0 itself.

## 4. Numerical method: shooting, not `solve_bvp`

`scipy.integrate.solve_bvp` (PLAN.md's original choice) proved structurally unreliable for
this problem, for two independent reasons:

1. **Rank-deficient Jacobian.** `dL/dm` depends only on the externally-prescribed
   `dT_dt`/`dP_dt` source arrays, never on the state being solved for — confirmed by direct
   computation (rank 2 of 4), for *any* `dT_dt`/`dP_dt`, not just zero ones. This is a
   structural property of the frozen-source-term time-stepping scheme, not a t=0 quirk —
   see §6.
2. **Near-surface boundary layer.** P approaches the much smaller `P_neb` with a short
   pressure scale height, driving `d(ln P)/d(ln m)` to enormous values that broke every
   collocation mesh/scaling strategy tried.

`bvp_solver.py` instead **shoots**: integrates the reduced 2-ODE `(r,P)` system outward from
the center with `scipy.integrate.solve_ivp` (adaptive, no global Jacobian; holding `L=0`,
`T=T_neb` fixed since they're known exactly), and root-finds (`scipy.optimize.brentq`) on
the central pressure to match `P(M_TOTAL)=P_neb`. Independent variable is `x=ln(m)` (m spans
~6 decades; `solve_ivp`'s adaptive stepping needs a bounded-range independent variable). This
is a deliberate, documented deviation from PLAN.md.

## 5. Current implementation status

| Sub-task | Module | Status |
|---|---|---|
| 1 | `config.py`, `state.py` | ✅ |
| 2a–2e | `eos.py`, `opacity.py` | ✅ |
| 3 | `gradients.py` | ✅ |
| 4 | `odes.py`, `boundary_conditions.py` | ✅ |
| 5 | `bvp_solver.py` | ✅ (shooting method, see §4) |
| 6 | `diagnostics.py` | ✅ (exit criteria revised for the cold t=0 state, see §7) |
| 7 | `time_stepper.py` — `compute_time_derivatives` | ✅ (homologous bootstrap, see §6) |
| 8–10 | Outer time loop, adaptive dt, output | ⬜ **next up: Sub-task 8** |

32 validation checks in `validation.py`, all passing. Diagnostic plots on disk:
`opacity_transitions.png`, `opacity_profile_preview.png`, `odes_profile_check.png`,
`static_structure_t0.png`, `mass_reconstruction_check.png`,
`bootstrap_time_derivatives.png`.

## 6. Sub-task 7 — homologous-contraction bootstrap (done)

The bootstrap **does not** return zero arrays (PLAN.md's original text) — that would leave
the envelope at t=0's fixed point forever, since two identical states always difference to
zero. Implemented instead: a **homologous (self-similar) contraction ansatz**. Derivation:
every Lagrangian shell contracts as `r=r0·f(t)`, `df/dt|0=-1/t_KH`; mass conservation forces
`ρ=ρ0/f³`; hydrostatic equilibrium (`dP/dm∝1/r⁴∝f⁻⁴`) is only satisfied at every instant if
`P=P0·f⁻⁴`; the ideal gas law then forces `T=T0·f⁻¹`. At t=0 this gives `dT_dt=+T/t_KH`,
`dP_dt=+4P/t_KH` — **both positive** (contraction *heats* the envelope — the standard
negative-heat-capacity behavior of a self-gravitating gas losing energy; note this is the
opposite sign from casually saying "cooling," which is wrong here). `t_KH` =
`config.T_KH_BOOTSTRAP_S` (1 Myr, an assumed GI-clump contraction timescale).

Substituting both into `odes.py`'s energy equation gives a closed form, `dL/dm =
[(3γ-4)/(γ-1)]·k_BT/(μm_H·t_KH)` — confirmed to match `odes.stellar_odes`'s numerical
output to machine precision, positive for γ>4/3 (`config.GAMMA=1.4` — the same threshold
behind `T_DISSOCIATION_LIMIT`, a satisfying independent consistency check), and its
integral over the envelope agrees with the completely independent `\|E_grav\|/t_KH`
estimate (from `diagnostics.virial_balance`) to within a factor of 2.24.

**The Jacobian-rank prediction was tested directly and is confirmed, with a caveat.** Ran a
full 4-ODE `solve_bvp` call (state_0 as initial guess, bootstrap `dT_dt`/`dP_dt` as frozen
source terms): the singular-Jacobian *crash* from Sub-task 5's t=0 attempt is genuinely
gone — 5 Newton iterations completed without crashing. **But `solve_bvp` still doesn't
practically converge** for a real timestep: residuals grow after iteration 2 and the mesh
explodes toward the node limit (status 1, boundary residuals ~1e7-1e9) — the same
unnormalized-absolute-tolerance-across-vastly-different-scales problem that forced Sub-task
5 toward shooting in the first place. **Implication for Sub-task 8:** expect to need the
same non-dimensionalization treatment (or a shooting-based per-timestep solve extending
Sub-task 5's approach), not a bare `solve_bvp` call.

## 7. Sub-task 6 — diagnostics established

- **Pressure-confined virial theorem** (derived independently by integrating hydrostatic
  equilibrium by parts — the standard zero-surface-pressure form doesn't apply since `P_neb`
  is not negligible): `E_grav + 3(γ-1)E_therm = 3·P_neb·V`. Verified against the converged
  state: relative imbalance ~8×10⁻⁶.
- **Opacity regime distribution:** 100% "Ice grains" (the coldest Bell & Lin regime) — exact
  consequence of uniform T=50K, not a spread across regimes as PLAN.md originally assumed.
- **Mass reconstruction check** (new, not in the original PLAN.md): independently validates
  the continuity equation and the shooting integration by reconstructing `M(r)=∫4πr²ρdr`
  from the converged profile and comparing to the Lagrangian grid.
- Energy flux check deferred to Sub-task 8+ (trivial at t=0 while `L≡0`; meaningful once
  real timesteps with nonzero L exist).

## 8. Project conventions established this session

- `config.py` is the single source of truth for constants; `validation.py`'s test values are
  an accepted, established exception (representative numbers local to each check).
- Physics modules (`eos.py`, `opacity.py`, `gradients.py`, `odes.py`) are pure functions.
- `diagnostics.py` is a **reporting** module (pure functions + a print-report entry point,
  no asserts) — distinct from `validation.py`, which is the **only** place assertions/tests
  live, and which requires proposing new checks for approval before adding them.
- Prefer a visible check (plot) over print-only asserts wherever a check has something to
  show.
- No blind `try`/`except`, no forced numerical dampening to make a solver converge — real
  failures should surface with the actual underlying error message.
- `PLAN.md` gets revised in place when the actual implementation deviates substantially from
  what it originally said (not just noted in `PROGRESS.md`) — a stale plan actively misleads.
