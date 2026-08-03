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
self-gravitating and contracts and heats up over time as it radiates away energy, until
(outside this project's scope) hydrogen dissociation ends the quasi-static regime.

This splits into two phases, and the code only ever models the second:

1. **Initial gravitational collapse (out of scope).** Fast, inertia-dominated free-fall —
   structurally impossible for a hydrostatic-equilibrium solver to represent (force balance
   is assumed at every instant, the opposite of free-fall). Standard practice in the
   literature is to never simulate this directly.
2. **Kelvin-Helmholtz contraction (this project).** The already-collapsed, compact,
   high-entropy object slowly radiates its formation heat and contracts over ~Myr, staying
   close to hydrostatic equilibrium throughout.

- **Not** core accretion, **not** star formation.
- Pure gas envelope: no solid core, no accretion of additional mass — `M_TOTAL` is fixed for
  the entire simulation at ~1 Jupiter mass (`1.898e30 g`).
- Lagrangian mass coordinate `m ∈ [0, M_TOTAL]`; 4 coupled ODEs (continuity, hydrostatic
  equilibrium, energy, temperature structure) solved on that grid at $t=0$ and every
  timestep.

## 2. The nebula environment — MMSN at 50 AU

`config.T_NEB = 50.0 K`, `config.P_NEB = 1.0e-4 dyn/cm²` — the outer-disk midplane
conditions, validated against the Hayashi (1981) minimum-mass solar nebula model at r=50 AU
(within factors of 1.3 and 2.5 of the MMSN reference — solid agreement). `T_NEB` still sets
the ambient radiative field the photosphere balances against (§3); `P_NEB` no longer plays a
mechanical role at all (§3) — it's kept as a validated MMSN reference value, not because the
structure is confined by it.

**These were not the original values.** `config.py` originally had `T_NEB=150K`,
`P_NEB=1e4` (inner-disk-like), which made `M_TOTAL` **99× the Bonnor-Ebert critical mass** —
no stable hydrostatic equilibrium exists at all under those conditions. The MMSN correction
dropped this to `M_TOTAL/M_BE ≈ 0.089` — deeply subcritical, part of what proved the original
diffuse-cloud $t=0$ premise (§3) was the wrong physical picture entirely.

## 3. The $t=0$ state and Sub-task 5 — DONE, verified end-to-end (2026-08-01)

**This section describes the current, validated premise, which replaced an earlier one.**
The original design had $t=0$ as a diffuse, cold ($T_\text{neb}$=50K), non-luminous cloud in
equilibrium with the ambient nebula — mathematically forced by $\partial T/\partial t=
\partial P/\partial t=0$ at a literal first solve. It converged cleanly ($R\approx13$ AU) but
was a genuine, *proven* mathematical dead end under time-stepping: a diffuse cloud already in
stable equilibrium with fixed boundary conditions has no reason to evolve, for any timestep
scheme (frozen or implicit), for any `dt` — not a numerical artifact, an exact property of
the equations. The deeper reason: initial GI collapse is a fast, dynamical process a
quasi-static solver structurally cannot represent (§1) — the diffuse pre-collapse state was
simply the wrong phase to hand this code.

**Current premise:** $t=0$ is a compact, hot, fully convective post-collapse protoplanet,
with a *prescribed* central temperature (`config.T_CENTER_INITIAL` = 1200 K — a chosen "hot
start" parameter, not derived, standard practice in gas-giant thermal-evolution modeling —
e.g. Marley et al. 2007). This eliminates the need for any bootstrap/kick mechanism: a
genuinely hot $t=0$ already has $T\neq T_\text{neb}$, so there is no isothermal degeneracy
to escape.

Two physics gaps stood between this premise and a working, self-consistent structure — both
are now closed:

**1. Electron degeneracy pressure (Sub-task 2f) — done, validated.** `eos.py`'s equation of
state is now the additive combination $P=P_\text{ideal}(\rho,T)+P_\text{degenerate}(\rho)$
(non-relativistic electron degeneracy pressure; Chandrasekhar 1939, Kippenhahn & Weigert Ch.
15). A pure ideal-gas adiabat at `T_CENTER_INITIAL` settles at $R\approx300\,R_\text{Jup}$,
not a genuinely compact structure (confirmed via an independent Lane-Emden analytic
solution). Real gas giants/brown dwarfs are partially electron-degenerate essentially from
formation (electron Fermi temperature at Jupiter's characteristic density is order
$10^5$–$10^6$ K, far above any plausible formation temperature). Adding the degenerate term
reproduces the analytic Zapolsky & Salpeter (1969)-style prediction ($R\approx3.11\,
R_\text{Jup}$) almost exactly in the real shooting code ($R\approx3.17\,R_\text{Jup}$).

**2. Photospheric outer boundary condition (Sub-task 5a) — done, validated.** The inherited
$P(M_\text{total})=P_\text{neb}$ condition, appropriate for a diffuse ambient-pressure-
confined cloud, has **no solution at all** once the interior is degenerate-supported — not a
hard-to-find root, a genuine gap in achievable surface pressure (surface pressure jumps
discontinuously across many orders of magnitude with nothing landing near $P_\text{neb}$,
confirmed not a tolerance artifact). Replaced with the standard Eddington grey-atmosphere
condition, $P_\text{photosphere}=\frac{2}{3}\frac{g}{\kappa}$ ($\tau=2/3$;
`boundary_conditions.photospheric_pressure`), located via a `solve_ivp` **event** during
outward integration and matched by *enclosed mass* rather than a residual at a fixed mass
endpoint (a fixed-endpoint version of the same formula was tested first and found to have
the identical reachability gap — the event-based location, not just the formula, was
necessary). `solve_static_structure()` now converges cleanly: $R\approx3.172\,R_\text{Jup}$,
mass relative residual $\approx0.16\%$. $P_\text{neb}$ drops out of the mechanical condition
entirely; only $T_\text{neb}$ continues to matter, via the thermal (net radiative flux)
surface condition.

**Bridging to real time evolution (`relax_initial_state`) — also done, verified.**
`solve_static_structure()`'s output is built by *forcing* the pure adiabat everywhere, and is
not itself a genuine solution of `solve_timestep`'s real, Schwarzschild-selected equations
(evaluating them at its own values diverges, $T\to3.4$ million K in one step).
`bvp_solver.relax_initial_state()` bridges the two via homotopy continuation — standard
"initial model relaxation" practice (MESA-style pre-main-sequence relaxation): blend the
temperature gradient between the pure adiabat ($\alpha=0$, matching `solve_static_structure`
exactly) and the real Schwarzschild-selected value ($\alpha=1$) over 11 fixed steps, warm-
starting each from the last. Getting this to converge cleanly required three separate
numerical fixes, each traced to root cause before being patched (no blind clamp-tuning):
- **Logarithmic state variables** ($\ln P$, $\ln T$ instead of $P$, $T$) in both of
  `bvp_solver.py`'s `solve_ivp` integrations, so positivity holds by construction — the
  standard Henyey/MESA-style representation. Replaced `1e-300` floor clamps that were
  themselves causing a cascade of Radau-internal solver crashes (the clamps were a reactive
  patch on a symptom, not the root cause: Radau's own internal Jacobian probing had no
  structural reason to avoid non-positive trial $P$/$T$ when stored linearly).
- **An $L\geq0$ floor in `gradients.grad_radiative`**, applied exactly where its own
  outward-flux derivation requires it (not patched downstream in `effective_gradient`).
  Without it, a small negative $L$ excursion near the photosphere — where $T^4\to0$ makes
  $\nabla_\text{rad}\propto\kappa LP/(mT^4)$ pathologically sensitive — flips the temperature
  gradient's sign and drives a runaway. Confirmed to be a pure bootstrapping aid: it engages
  only for $\alpha\leq0.7$ and is never active at the converged $\alpha=1$ solution.
- **A tiny (`1e-6` relative) seed nudge** in `solve_timestep`'s `fsolve` seed, identical to
  one already used in `relax_initial_state`: seeding from an already-self-consistent
  `state_prev` makes the trial and `state_prev` coincide to near machine precision right at
  the seed, and $dT/dt=(T-T_\text{prev})/dt$ amplifies that floating-point noise into a
  spurious collapse without it.

**Verified end-to-end**: `solve_static_structure()` → `relax_initial_state()` (all 11
$\alpha$ pseudo-steps converge, residuals $\lesssim10^{-5}$) → `solve_timestep()` (converges
from the relaxed state, residuals $[2.4\times10^{-10}, 3.1\times10^{-8}]$), giving a
physically sensible first step ($T_\text{center}$ cools $1251.9\to1215.3\,$K over
$dt=0.01\,t_\text{KH}$). Full numerical trail: PROGRESS.md's 2026-07-27 and 2026-08-01
entries.

## 4. Numerical method: shooting, not `solve_bvp`

`scipy.integrate.solve_bvp` (PLAN.md's original choice) proved structurally unreliable for
this problem, for two independent reasons: a rank-deficient Jacobian (the energy equation's
`dL/dm` depends only on externally-prescribed source terms, never the state being solved
for) and a near-surface pressure-scale-height boundary layer that breaks its collocation
mesh regardless of scaling strategy. `bvp_solver.py` instead **shoots** for both cases,
outward from the center via `scipy.integrate.solve_ivp` (Radau), terminating at the
photosphere (§3) located as a `solve_ivp` *event*:

- **$t=0$:** `solve_static_structure` integrates $(r,\ln P,\ln T)$ outward, root-finding
  (`brentq`) on $P_\text{center}$ alone to match enclosed mass at the photosphere to
  $M_\text{total}$ ($T_\text{center}$ is fixed at `config.T_CENTER_INITIAL`, not solved for).
  The bracket is seeded from an analytic Lane-Emden estimate (now using the pure T=0
  electron-degeneracy limit, not the old ideal-gas polytrope) rather than a blind search.
- **$t>0$ (`solve_timestep`, and `relax_initial_state`'s pseudo-steps):** shoots on
  $(\ln P_\text{center},\ln T_\text{center})$ via `fsolve` to match both enclosed mass at the
  photosphere and the net-flux radiative surface condition,
  $L=4\pi R^2\sigma_\text{SB}(T^4-T_\text{neb}^4)$ (replacing an earlier rigid
  $T=T_\text{neb}$ clamp, which made "no change" an exact fixed point of any per-timestep
  scheme — proven, not a scheme artifact). $\partial T/\partial t$, $\partial P/\partial t$ in
  the energy equation are computed directly from the implicit state difference, with **no**
  additional forcing term (an earlier attempt to add one double-counted compressional
  heating and was reverted).

Both routines integrate in logarithmic state variables ($\ln P$, $\ln T$; §3) — a change
made this session to fix a Radau-internal solver-crash cascade, not part of the original
design.

## 5. Current implementation status (2026-08-01)

| Sub-task | Module | Status |
|---|---|---|
| 1 | `config.py`, `state.py` | Done |
| 2a–2e | `eos.py` (ideal-gas part), `opacity.py` | Done |
| 2f | `eos.py` — non-ideal EOS, electron degeneracy pressure | **Done, validated** |
| 3 | `gradients.py` | Done (includes the $L\geq0$ floor in `grad_radiative`, §3) |
| 4 | `odes.py`, `boundary_conditions.py` | Done (photospheric surface BC, §3) |
| 5a | `bvp_solver.py` outer BC redesign (photospheric) | **Done** |
| 5 | `bvp_solver.py` ($t=0$ + relaxation to self-consistency) | **Done, verified end-to-end** |
| 6 | `diagnostics.py` | **Done** — visual plots + virial theorem (standard unconfined form) + multi-regime opacity check, all pass |
| 7 | `time_stepper.py` time derivatives | Homologous bootstrap now **obsolete**; code not yet updated (confirmed broken: references a renamed config constant) |
| 8–10 | Outer time loop, adaptive dt, output | Not started — next up |

**`python validation.py` does not currently pass cleanly end-to-end** — Check 19
(boundary-condition residuals, still tests the pre-photospheric-BC formula) and Check 30
(bootstrap time derivatives, tests the now-confirmed-broken obsolete mechanism) are known
stale/broken. All EOS, opacity, gradient, virial, and mass-reconstruction checks pass. See
PROGRESS.md §4 for the per-check status.

## 6. Sub-task 7 — homologous-contraction bootstrap: obsolete, and now confirmed broken

**This mechanism is scheduled for removal, not further use.** It existed solely to break the
old isothermal $t=0$'s exact degeneracy — once $t=0$ is genuinely hot and non-isothermal
(§3), `solve_timestep` runs directly from `state_0` with no special first-step handling. The
underlying derivation (`compute_time_derivatives`'s bootstrap branch: homologous contraction
`r=r0·f(t)` ⟹ `dT_dt=+T/t_KH`, `dP_dt=+4P/t_KH`) was correct and thoroughly cross-checked at
the time — the derivation isn't wrong, it's just no longer needed.

**Confirmed broken as of 2026-08-01**, not just obsolete: `time_stepper.py`'s bootstrap code
still references `config.T_KH_BOOTSTRAP_S`, which was renamed to `config.T_KH_TIMESCALE_S`
earlier this session — `time_stepper.py` was never updated to match. The only caller
(`validation.py` Check 30) now raises `AttributeError` if run. Removing the bootstrap
dispatch (rather than fixing the stale reference) is tracked together with implementing the
outer time loop (`run()`, Sub-task 8).

Also confirmed at the time and still true: `scipy.integrate.solve_bvp` does not practically
converge for a real timestep even once the singular-Jacobian crash is fixed (mesh explosion,
unnormalized boundary residuals ~$10^7$–$10^9$) — the reason every solve in this codebase
uses shooting (§4), not just the $t=0$ case.

## 7. Sub-task 6 — diagnostics: done

`diagnostics.py` now has three visual diagnostic plots (`plot_structure_profile`,
`plot_mass_radius`, `plot_convective_zones` — all saved to `diagnostic_plots/`, §8) alongside
the print-based `run_diagnostics`. The virial-balance check was rewritten from the old
pressure-confined form ($E_\text{grav}+3(\gamma-1)E_\text{therm}=3P_\text{neb}V$, appropriate
for the superseded diffuse-cloud premise) to the standard unconfined form
($E_\text{grav}+3(\gamma-1)E_\text{therm}\approx0$) — confirmed $P_\text{neb}$'s
confinement term is now ~15 orders of magnitude below the interior energy scale, so the
unconfined limit genuinely applies. Measured on the real structure: relative imbalance
$3.6\times10^{-4}$. The opacity regime check now asserts the physically-required *ordering*
(center strictly hotter regime than the surface) rather than hardcoded regime indices;
measured: center in "Metal grains" (1200K), surface in "Ice grains" (7.5K) — genuinely
multi-regime, as expected for a differentiated hot-center-to-cold-surface structure. The
mass-reconstruction check (continuity-equation self-consistency) needed no changes.

## 8. Project conventions established

- `config.py` is the single source of truth for constants; `validation.py`'s test values are
  an accepted, established exception (representative numbers local to each check).
- Physics modules (`eos.py`, `opacity.py`, `gradients.py`, `odes.py`) are pure functions.
- `diagnostics.py` is a **reporting** module (pure/plotting functions + a print-report entry
  point, no asserts) — distinct from `validation.py`, which is the **only** place
  assertions/tests live, and which requires proposing new checks for approval before adding
  them.
- Prefer a visible check (plot) over print-only asserts wherever a check has something to
  show.
- All diagnostic/validation PNGs save to `diagnostic_plots/` (`diagnostics.PLOT_DIR`), not
  the project root — established 2026-08-01 to keep the root directory uncluttered.
- **Development workflow (established 2026-08-01, see CLAUDE.md's Development Workflow
  section):** never re-run a heavy solve (`relax_initial_state`, ~15-20 min) just to test
  unrelated downstream logic — cache the intermediate `SimulationState` via `dev_cache.py`
  and develop against the cached state. Build outer-loop wrappers (the upcoming time loop,
  Sub-task 8) sterile-first against a mock/cached state sequence, wet-test against the real
  solver only once the outer logic is validated on its own. Long-running processes must log
  progress periodically.
- No blind `try`/`except`, no forced numerical dampening to make a solver converge — real
  failures should surface with the actual underlying error message, traced to root cause
  before being patched (this discipline directly avoided a whack-a-mole clamp-tuning trap
  during Sub-task 5's relaxation work — PROGRESS.md's 2026-08-01 entries).
- `PLAN.md` gets revised in place when the actual implementation deviates substantially from
  what it originally said (not just noted in `PROGRESS.md`) — a stale plan actively misleads.
  Applies to summary/status tables too, not just prose sections — a stale table caused real
  confusion this session even after the prose sections were kept current.
- **Treat "already implemented" as "not yet trusted."** A substantial rewrite (e.g. the
  compact hot-start pivot) should get an independent, line-by-line correctness review before
  further work builds on top of it — this practice directly paid off (2026-07-27): it found
  a second, independent architectural gap (the photospheric BC need) that would otherwise
  have been silently misattributed entirely to the missing-EOS hypothesis.
