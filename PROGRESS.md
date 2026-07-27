# PlanetFormationSim — Progress & Documentation Log

**Audience:** you, as the physicist directing this project. This file exists so you can
open it at any point and reconstruct *what has been built, why it was built that way, and
what is and isn't currently trustworthy* — without re-reading diffs or chat history.

For the target physics, the full 4-ODE formulation, and the sub-task roadmap, see
[PLAN.md](PLAN.md). This file tracks actual implementation progress against that plan.

---

## 1. Current Status

**⏸ IMPLEMENTATION PAUSED (end of 2026-07-27 session) — pick up here.** Sub-task 2f (EOS) and
the photospheric outer BC redesign for `solve_static_structure` are DONE and validated
($R\approx3.17\,R_\text{Jup}$, matching the analytic prediction). The remaining piece —
bridging `solve_static_structure`'s output into a state that's genuinely self-consistent
with `solve_timestep`'s real 4-ODE equations, via a homotopy/relaxation scheme
(`bvp_solver.relax_initial_state`) — is **implemented but not working**: its first pseudo-step
converges beautifully (validating the physical approach), but later steps hit a cascading
series of numerical edge cases in `scipy`'s stiff-solver internals, not yet resolved with a
principled fix. **Two candidate fixes are on the table, not yet evaluated or implemented** —
see §5's "Sub-task 5: initial-state relaxation" entry below for the full detail, exact
failure trace, and both proposals. Do not resume by re-guessing at domain-clamp values —
read that entry first.

| Sub-task | Scope | Status |
|---|---|---|
| 1 | `config.py` + `state.py` | Done |
| 2a–2e | `eos.py` (ideal-gas part) + `opacity.py` + validation | Done |
| 2f | `eos.py` — non-ideal EOS, electron degeneracy pressure | Done, validated (2026-07-27) |
| 3 | `gradients.py` (Schwarzschild criterion) | Done |
| 4 | `odes.py` + `boundary_conditions.py` | Done (surface conditions revised, §5) |
| 5 | `bvp_solver.py` ($t=0$ structure + relaxation to self-consistency) | `solve_static_structure` done & validated (photospheric BC); **`relax_initial_state` implemented but blocked — paused, see above** |
| 6 | `diagnostics.py` | Blocked on 5 |
| 7 | `time_stepper.py` time derivatives | Original bootstrap now obsolete; code not yet updated |
| 8–10 | Outer time loop, adaptive dt, output | Not started — blocked on 5–7 |

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

### What works right now, concretely

- `config.py`, `gradients.py` (including the new `marginal_convective_luminosity` helper) —
  clean, no known issues.
- `boundary_conditions.py`'s net-flux radiative surface condition — implemented, and its
  indexing/formula correctness is covered by validation.py's Check 19 (revised this
  session for the new formula).
- `bvp_solver.solve_static_structure()` — runs without crashing, produces a finite,
  monotonic, genuinely non-isothermal hot structure. **Not compact and not yet shown
  self-consistent with `solve_timestep`** (see above) — treat its output as a
  work-in-progress, not a validated deliverable.
- `bvp_solver.solve_timestep()` — the shooting/root-find machinery itself runs without
  crashing given a reasonable starting `state_prev`, but has not been validated as
  producing a *correct* result, since the only `state_prev` available to test it with
  (the current `solve_static_structure()` output) is itself not self-consistent.

### What's blocked / not working

- `bvp_solver.py`'s $t=0$ construction needs the Sub-task 2f EOS work before it can be
  finalized (§5, Sub-task 5 in PLAN.md).
- `time_stepper.py` is **unchanged** from its original Sub-task 7 implementation — it still
  contains the now-obsolete bootstrap dispatch (`_bootstrap_time_derivatives`,
  `state_prev=None` branch). Not yet edited.
- `time_stepper.run()` (Sub-task 8) does not exist yet.
- `validation.py` has **not** been re-run successfully since `bvp_solver.py`'s premise
  change — several checks (see §4) are known to be stale relative to the current code and
  would fail if run today.

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

`grad_radiative` and `effective_gradient` are unchanged. **New this session:**
`marginal_convective_luminosity(m, P, T, kappa, grad_ad)` — inverts
$\nabla_\text{rad}(L,\ldots)=\nabla_\text{ad}$ for $L$ in closed form (the "marginally
efficient convection" closure). Used by `bvp_solver.solve_static_structure()` to populate a
physically meaningful, non-trivial $L(m)$ for a $t=0$ structure whose $T(m)$ was built
directly from the adiabat rather than solved for — not consumed by `solve_timestep` (which
only ever interpolates `state_prev.T`, `.P`, never `.L`), so this is a diagnostic/plotting
convenience, not something load-bearing for the time evolution itself.

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

### `diagnostics.py` — post-solve physical diagnostics

Unchanged in code. Its existing checks (pressure-confined virial form, single-regime
opacity expectation) were written for Premise 1's isothermal state and are expected to need
revision once Sub-task 5 lands a final structure — see PLAN.md's Sub-task 6 entry.

### `time_stepper.py` — time-derivative bridge between timesteps

**Unchanged in code this session** — still contains `_bootstrap_time_derivatives` and
`compute_time_derivatives`'s `state_prev=None` dispatch exactly as originally implemented.
This is now understood to be **obsolete** (the bootstrap it computes is no longer needed —
§1, PLAN.md's Sub-task 7 entry) but has not yet been edited; flagged here so the gap
between "what the code does" and "what we now believe is correct" is explicit rather than
silently inconsistent.

### `validation.py` — sanity checks, unit consistency, and diagnostic plots

See §4 below. **Not fully passing as of this writing** — several checks written for
Premise 1's isothermal $t=0$ state are now stale relative to `bvp_solver.py`'s rewritten
`solve_static_structure()`. **New this session:** Checks 33-36 (Sub-task 2f's EOS/degeneracy
checks — reference point, asymptotic limits, round-trip inversion, visible $P(\rho)$ plot)
were proposed, approved, implemented, and pass cleanly. **Still pending, not yet done:**
Check 19 (`check_boundary_conditions_residuals`) still tests the *old*
`P_b-P_\text{neb}$ mechanical residual formula — needs revision for the new photospheric one
before it will even run without erroring against the current `boundary_conditions.py`; no
new checks have been proposed yet for the photospheric condition or the (still-blocked)
relaxation homotopy.

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
