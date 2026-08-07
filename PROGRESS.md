# PlanetFormationSim — Progress & Documentation Log

**Audience:** you, as the physicist directing this project. This file exists so you can
open it at any point and reconstruct *what has been built, why it was built that way, and
what is and isn't currently trustworthy* — without re-reading diffs or chat history.

For the target physics, the full 4-ODE formulation, and the sub-task roadmap, see
[PLAN.md](PLAN.md). This file tracks actual implementation progress against that plan.

---

## 1. Current Status

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
| 5 | `bvp_solver.py` ($t=0$ structure + relaxation to self-consistency) | `solve_static_structure` still in active use ($t=0$ seed, reused by `bvp_experiment.py`). **`relax_initial_state`/shooting machinery ARCHITECTURALLY ABANDONED 2026-08-06 — see §1's pivot note and `PLAN_BVP.md`.** |
| — | `bvp_experiment.py` (new, `solve_bvp` — the active target architecture) | **★ CONVERGES (status=0, machine-precision residuals) at T=11500K AND T=12000K as of 2026-08-07** — state-vector scaling + analytic Jacobians + corrected EOS thermodynamics; see §5's Milestone 6 entry and `PLAN_BVP.md` §3.6/§3.6.4. Negative `L_surface` confirmed reproducible across both temperatures (not resolved, but no longer single-point); T=2000K confirmed out of scope for the atomic EOS, not re-attempted. Remaining item before promotion to production: full validation-suite pass. |
| 6 | `diagnostics.py` | **Done (2026-08-01)** — visual plots + virial theorem/opacity regime checks rewritten for the compact structure, all pass |
| 7 | `time_stepper.py` time derivatives | Original bootstrap now obsolete; code not yet updated; will need to target `bvp_experiment.py`'s solver once stabilized, not `bvp_solver.relax_initial_state` |
| 8–10 | Outer time loop, adaptive dt, output | Not started — blocked on `PLAN_BVP.md` Milestones 1–3 (T=13000K convergence) |

**Stubs present but empty:** `main.py`, `ReadMe.txt`.

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

Unchanged.

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

### `bvp_solver.py` — shooting-method solver, $t=0$ and every $t>0$ step

**⚠ 2026-08-06: shooting is architecturally ABANDONED — see §5's "Architectural decision"
entry and `PLAN_BVP.md`.** `solve_static_structure()` (the $t=0$ adiabatic seed
construction) remains in active use, reused unmodified by `bvp_experiment.py` to build
initial guesses — but `relax_initial_state()` and the shooting/LM machinery described below
are no longer the project's target solver, kept only as historical reference and a fallback
until the `solve_bvp` pivot (`PLAN_BVP.md`) is proven. Everything below this note describes
the shooting-era implementation and is retained for that historical record, not as current
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

### `time_stepper.py` — time-derivative bridge between timesteps

**Unchanged in code this session** — still contains `_bootstrap_time_derivatives` and
`compute_time_derivatives`'s `state_prev=None` dispatch exactly as originally implemented.
This is now understood to be **obsolete** (the bootstrap it computes is no longer needed —
§1, PLAN.md's Sub-task 7 entry) but has not yet been edited; flagged here so the gap
between "what the code does" and "what we now believe is correct" is explicit rather than
silently inconsistent. **Confirmed broken as of 2026-08-01** (not just obsolete):
`_bootstrap_time_derivatives` references `config.T_KH_BOOTSTRAP_S`, which no longer exists
(renamed to `config.T_KH_TIMESCALE_S` earlier this session) — `validation.py`'s Check 30
(the only caller) now raises `AttributeError` if run. Confirms this is genuinely Sub-task 7's
job, not a documentation nicety.

### `validation.py` — sanity checks, unit consistency, and diagnostic plots

See §4 below. **Not fully passing as of this writing** — several checks written for
Premise 1's isothermal $t=0$ state are now stale relative to `bvp_solver.py`'s rewritten
`solve_static_structure()`. **New this session:** Checks 33-36 (Sub-task 2f's EOS/degeneracy
checks) were implemented and pass cleanly (2026-07-27); Check 26 (renamed
`check_virial_balance_unconfined`) and Check 27 (opacity regime distribution) were rewritten
for the compact structure and now pass cleanly (2026-08-01, see the `diagnostics.py` entry
above for the physical detail). **Still pending, not yet done:** Check 19
(`check_boundary_conditions_residuals`) still tests the *old* `P_b-P_\text{neb}$ mechanical
residual formula — needs revision for the new photospheric one before it will even run
without erroring against the current `boundary_conditions.py`; Check 30
(`check_bootstrap_time_derivatives_are_physical`) is confirmed broken (see `time_stepper.py`
entry above) pending Sub-task 7's bootstrap removal; no new checks have been proposed for
the photospheric condition or the relaxation homotopy.

### `main.py`, `ReadMe.txt`

Empty placeholders, unchanged.

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

`validation.py` (`python validation.py`) contains 32 checks. **As of this writing, running
it is expected to fail** — the checks below are listed as they currently exist in the file
(this table describes what each one *tests*, unchanged from before), but several no longer
match `bvp_solver.py`'s actual behavior after this session's premise change. Status flags
added inline; unflagged checks are believed to still pass (they test regime-independent
physics-module building blocks — EOS, opacity, gradients, ODE-RHS mechanics — never the
specific $t=0$ solve).

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

Most recent first. Each entry: what was done, and the physical/architectural reasoning.
Entries below marked **[SUPERSEDED]** describe conclusions that later investigation
overturned — kept rather than deleted because the reasoning inside them (numerical
findings, derivations, literature checks) remains accurate and load-bearing for
understanding *why* later decisions were made; only their final conclusion no longer holds.

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
