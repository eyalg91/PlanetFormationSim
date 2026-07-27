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
(within factors of 1.3 and 2.5 of the MMSN reference — solid agreement). These represent the
ambient conditions the compact protoplanet (§3) radiates against and is pressure-bounded
by — **not** conditions the whole envelope sits in quasi-static equilibrium with, which was
the old (superseded) picture.

**These were not the original values.** `config.py` originally had `T_NEB=150K`,
`P_NEB=1e4` (inner-disk-like), which made `M_TOTAL` **99× the Bonnor-Ebert critical mass**
— no stable hydrostatic equilibrium exists at all under those conditions. The MMSN
correction dropped this to `M_TOTAL/M_BE ≈ 0.089` — deeply subcritical. This fact is still
true and still relevant, but its *role* changed (§3): it no longer explains why $t=0$ looks
a particular way — it explains why the envelope could never be evolved quasi-statically
starting from a diffuse, disk-confined state at all, which is why $t=0$ is now something
else entirely.

## 3. The $t=0$ state — compact hot start, currently blocked on missing EOS physics

**This section describes the current premise, which replaced an earlier one.** The
original design had $t=0$ as a diffuse, cold (`T_neb`=50K), non-luminous cloud in
equilibrium with the ambient nebula — mathematically forced by `dT_dt=dP_dt=0` at a literal
first solve, and also matching real Bonnor-Ebert-subcritical GI-clump equilibria in the
literature. It converged cleanly (`R≈13 AU`) but turned out to be a genuine, *proven*
mathematical dead end under time-stepping: a diffuse cloud already in stable equilibrium
with fixed boundary conditions has no reason to evolve, for any timestep scheme (frozen or
implicit), for any `dt` — not a numerical artifact, an exact property of the equations. The
deeper reason: initial GI collapse is a fast, dynamical process a quasi-static solver
structurally cannot represent (§1) — the diffuse pre-collapse state was simply the wrong
phase to hand this code.

**Current premise:** $t=0$ is a compact, hot, fully convective post-collapse protoplanet,
with a *prescribed* central temperature (`config.T_CENTER_INITIAL` = 1200 K — a chosen "hot
start" parameter, not derived, standard practice in gas-giant thermal-evolution modeling —
e.g. Marley et al. 2007). This eliminates the need for any bootstrap/kick mechanism: a
genuinely hot $t=0$ already has $T\neq T_\text{neb}$, so there is no isothermal degeneracy
to escape.

**Status: implemented (`bvp_solver.solve_static_structure`), but not yet a validated
deliverable — two independent physics gaps found, one confirmed via an independent
correctness review (2026-07-27):**

1. **Missing electron degeneracy pressure (interior).** `eos.py` is ideal-gas only. An
   independent analytic Lane-Emden solution (cross-checked against tabulated references, then
   reproduced in the actual shooting code) shows a pure ideal-gas adiabat at
   `T_CENTER_INITIAL` settles at $R\approx300\,R_\text{Jup}$, not a genuinely compact few-
   $R_\text{Jup}$ radius. Real gas giants/brown dwarfs are partially electron-degenerate
   essentially from formation (the electron Fermi temperature at Jupiter's characteristic
   density is order $10^5$–$10^6$ K, far above any plausible formation temperature) — this is
   why every published gas-giant thermal-evolution code uses a non-ideal EOS, even for its
   hottest, youngest models. **This is the leading hypothesis, being tested next
   (Sub-task 2f).**
2. **The $P(M_\text{total})=P_\text{neb}$ outer boundary condition (outer envelope) —
   confirmed as a second, independent gap, not fixed by (1).** Directly checked (previously
   only a guess): the fully self-consistent version of the construction (real
   `odes.stellar_odes`, not a shortcut) is genuinely convective out to $m/M\approx0.49$, then
   transitions to *radiative* at $m/M\approx0.70$, after which $T$ goes nearly flat while $r$
   explodes $153\times$ over the last 5 decades of pressure drop to reach $P_\text{neb}$ — the
   same nearly-isothermal-extended-envelope mechanism as the original diffuse-cloud problem,
   now confined to the outer ~30% of the mass. Electron degeneracy pressure is negligible at
   those low densities, so fixing (1) will not fix this. The likely real fix is a
   physically-motivated photospheric outer BC (e.g. $\tau=2/3$) instead of forcing radiative
   diffusion all the way down to the ambient nebula pressure. **Tracked as a required
   follow-up after Sub-task 2f, not yet attempted.**

Full derivations, numbers, and the correctness-review methodology are in PROGRESS.md
(§1's Confirmed/Inference/Open breakdown, and the `bvp_solver.py` module reference).

## 4. Numerical method: shooting, not `solve_bvp` — now used for both $t=0$ and $t>0$

`scipy.integrate.solve_bvp` (PLAN.md's original choice) proved structurally unreliable for
this problem, for two independent reasons: a rank-deficient Jacobian (the energy equation's
`dL/dm` depends only on externally-prescribed source terms, never the state being solved
for) and a near-surface pressure-scale-height boundary layer that breaks its collocation
mesh regardless of scaling strategy. `bvp_solver.py` instead **shoots** for both cases:

- **$t=0$:** integrate $(r,P,T)$ outward from the center with `scipy.integrate.solve_ivp`,
  root-finding (`brentq`) on $P_\text{center}$ alone to match $P(M_\text{total})=P_\text{neb}$
  ($T_\text{center}$ is fixed at `config.T_CENTER_INITIAL`, not solved for). The bracket is
  seeded from an analytic Lane-Emden estimate rather than a blind search, which does not find
  the true root (it sits in an extremely narrow, ~0.1%-relative window).
- **$t>0$ (`solve_timestep`):** shoots on $(\ln P_\text{center}, \ln T_\text{center})$ via
  `fsolve` to match both $P(M_\text{total})=P_\text{neb}$ and a net-flux radiative surface
  condition, $L=4\pi R^2\sigma_\text{SB}(T^4-T_\text{neb}^4)$ (replacing an earlier rigid
  $T=T_\text{neb}$ clamp, which made "no change" an exact fixed point of any per-timestep
  scheme — proven, not a scheme artifact). $\partial T/\partial t$, $\partial P/\partial t$ in
  the energy equation are computed directly from the implicit state difference, with **no**
  additional forcing term (an earlier attempt to add one double-counted compressional
  heating and was reverted — see PROGRESS.md/PLAN.md §4.8).

Both routines were independently reviewed line-by-line (2026-07-27, PROGRESS.md) and found
correctly implemented — the open problems (§3) are architectural/physical, not coding bugs.

## 5. Current implementation status

| Sub-task | Module | Status |
|---|---|---|
| 1 | `config.py`, `state.py` | Done |
| 2a–2e | `eos.py` (ideal-gas part), `opacity.py` | Done |
| **2f** | `eos.py` — non-ideal EOS, electron degeneracy pressure | **Next milestone, not started** |
| 3 | `gradients.py` | Done |
| 4 | `odes.py`, `boundary_conditions.py` | Done (surface thermal BC revised, §4) |
| 5 | `bvp_solver.py` ($t=0$ + $t>0$ shooting) | Implemented, correctness-reviewed, **not validated — blocked on 2f and the outer-BC follow-up** |
| 6 | `diagnostics.py` | Blocked on 5 (existing checks assume the old cold/isothermal state) |
| 7 | `time_stepper.py` time derivatives | Homologous bootstrap now **obsolete** (no longer needed once $t=0$ is genuinely hot); code not yet updated |
| 8–10 | Outer time loop, adaptive dt, output | Not started — blocked on 5–7 |

**`python validation.py` does not currently pass cleanly** — several checks (isothermal
$t=0$ assertions, single-opacity-regime prediction, the pressure-confined virial form, the
now-removed bootstrap check) were written for the superseded cold-isothermal premise. See
PROGRESS.md §4 for the per-check status.

## 6. Sub-task 7 — homologous-contraction bootstrap: implemented, now obsolete

**This mechanism is scheduled for removal, not further use.** It existed solely to break
the old isothermal $t=0$'s exact degeneracy — once $t=0$ is genuinely hot and non-isothermal
(§3), `solve_timestep` runs directly from `state_0` with no special first-step handling. The
underlying derivation (`compute_time_derivatives`'s bootstrap branch: homologous contraction
`r=r0·f(t)` ⟹ `dT_dt=+T/t_KH`, `dP_dt=+4P/t_KH`) was correct and thoroughly cross-checked at
the time (matches `odes.stellar_odes` to machine precision, agrees with an independent
`|E_grav|/t_KH` virial estimate to within 2.24×) — the derivation isn't wrong, it's just no
longer needed. `time_stepper.py`'s code has **not** been edited yet; removing the bootstrap
dispatch is tracked together with implementing the outer time loop (`run()`, Sub-task 8).

Also confirmed at the time and still true: `scipy.integrate.solve_bvp` does not practically
converge for a real timestep even once the singular-Jacobian crash is fixed (mesh explosion,
unnormalized boundary residuals ~$10^7$–$10^9$) — the reason every solve in this codebase
uses shooting (§4), not just the $t=0$ case.

## 7. Sub-task 6 — diagnostics: checks now stale, pending revision

The existing `diagnostics.py` checks (pressure-confined virial theorem
$E_\text{grav}+3(\gamma-1)E_\text{therm}=3P_\text{neb}V$; a 100%-single-regime opacity
prediction) were derived for the old isothermal, pressure-confined $t=0$ state and no
longer apply once $t=0$ is compact and self-gravitating: a structure with negligible surface
pressure needs the *standard* (unconfined) virial form instead, and a hot-center-to-cold-
surface structure should span multiple Bell & Lin opacity regimes, not sit entirely in "Ice
grains." Revision is deferred until Sub-task 5 lands a final, validated structure to check
(PLAN.md Sub-task 6). The mass-reconstruction check (continuity-equation self-consistency)
is regime-independent and expected to transfer directly once that happens.

## 8. Project conventions established

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
- **Treat "already implemented" as "not yet trusted."** A substantial rewrite (e.g. the
  compact hot-start pivot) should get an independent, line-by-line correctness review before
  further work builds on top of it — this practice directly paid off (2026-07-27): it found
  a second, independent architectural gap (§3) that would otherwise have been silently
  misattributed entirely to the missing-EOS hypothesis.
