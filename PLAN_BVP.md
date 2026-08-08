# PLAN_BVP.md — Roadmap: Pivoting to a Global Relaxation (`solve_bvp`) Architecture

> **★ MERGED INTO PLAN.md, 2026-08-08.** This roadmap's conclusion (Milestone 6, §3.6/§3.6.4)
> has been promoted into production `bvp_solver.py`, and the architectural decision is now
> recorded in `PLAN.md` §4.2 and its Sub-task 5 status note. `PLAN.md` is once again the
> single active forward-looking reference; this document is kept in place, unmaintained
> going forward, as the detailed milestone-by-milestone numerical trail (Milestones 0-6)
> behind that decision — link to it for the "why," not the current "what."

**Status (2026-08-07): Active. This supersedes shooting (`bvp_solver.py`) as the project's
target numerical architecture, per an explicit architectural decision recorded below.**
This document is the forward-looking roadmap for that pivot, in the same spirit as
`PLAN.md` §4 (Key Design Decisions) but scoped specifically to the BVP transition — it does
not duplicate `PLAN.md`/`PROGRESS.md`'s content and should be read alongside them,
particularly `PLAN.md` §4.2 (the original shooting-vs-`solve_bvp` decision, now revisited)
and `PROGRESS.md`'s 2026-08-06/2026-08-07 entries (the full numerical trail behind everything
summarized here).

**Revision 2026-08-07** — §3 restructured after a joint architectural review of the crash
evidence. Net changes: (1) a new Milestone 0 (ionization-sensitivity diagnostic) inserted
ahead of everything else, to empirically settle — cheaply, before committing to the much
larger Saha/EOS undertaking — whether missing ionization physics is actually implicated in
the T=13000K crash, given the crash is spatially localized at the photosphere while the
known EOS gap is more naturally a bulk/interior effect; (2) a new Milestone 2 (center
boundary-condition self-consistency), correcting `bvp_experiment.py`'s center seed to track
the live trial center state rather than a fixed pre-estimate; (3) the old Milestone 2
(log-space surface BC) merged with an explicit mesh-density verification step, since both
are "confirm before blaming the solver" checks; (4) the Saha/high-T-seed milestone
(previously deferred as future work) is now conditionally reprioritized based on Milestone
0's result rather than unconditionally last. The empirical findings in §2 and the
architectural decision in §1 are unchanged.

---

## 1. Architectural Decision & Rationale

### 1.1 Why shooting was tried in the first place

`scipy.integrate.solve_bvp` was the *original* design for this project (`PLAN.md` §4.2). It
was abandoned in July 2026 after two specific, diagnosed failures: a rank-deficient Jacobian
from the old static (`dT_dt=dP_dt=0`) t=0 formulation, and a near-surface pressure-scale-height
boundary layer that broke its collocation mesh under the old `P(M_TOTAL)=P_neb` mechanical
surface condition. Shooting (`scipy.integrate.solve_ivp` outward integration + root-finding
on the central conditions) was adopted as the fallback that worked well enough to make
progress — not because it is theoretically better suited to this problem.

### 1.2 Why shooting is being abandoned now

Both of the historical `solve_bvp` blockers were superseded by later, unrelated fixes (the
photospheric Eddington `τ=2/3` boundary condition replacing `P=P_neb`; the log-`P`/log-`T`
state representation). That raised the question of whether `solve_bvp` deserved a retest —
but before that retest happened, continuing to stabilize shooting at
`T_CENTER_INITIAL=13000K` surfaced a decisive pattern:

| # | Kink found | Location | Status |
|---|---|---|---|
| 1 | `gradients.grad_radiative`'s hard `L≥0` floor | near-photosphere, α≈0.046 | Fixed (smoothed) |
| 2 | `gradients.effective_gradient`'s hard Schwarzschild `min()` switch | near-photosphere, α≈0.051 | Fixed (smoothed) |
| 3 | A third wall at α≈0.0466, reappearing *after* both fixes above | same narrow region again | Unresolved; opacity hard-switches are the leading suspect |

Three independent non-smoothness sources, discovered one at a time by patching the previous
one and hitting the next, all clustering in the *same* narrow neighborhood of the homotopy
path and the *same* physical region of the star (the near-photosphere transition,
`m/M_TOTAL ≳ 0.84`). This is the diagnostic signature of a single-long-integration method
(shooting) that has no way to contain a local non-smoothness — any kink anywhere along the
outward integration corrupts the signal the outer root-finder sees at the surface, regardless
of which specific term produces it. Patching each one as found is not a bounded, one-time
cost; it scales with how much of the (T, ρ, opacity-regime, convective-boundary) space the
project needs to cover — and the required scope (≈2000K–40000K) covers a lot of it.

**Decision: shooting is abandoned. `scipy.integrate.solve_bvp` — a collocation/relaxation
method, the same numerical family as Henyey's implicit relaxation used by essentially every
production stellar-evolution code (MESA, STARS/TWIN, the classical Kippenhahn code) — is the
sole path forward.** Its global, simultaneous-solve structure does not compound local
sensitivity across a long integration the way shooting does; a kink stays a local residual
issue rather than corrupting the entire shot.

### 1.3 What does not change

This is a **numerical**, not a **physical**, pivot. `eos.py`, `opacity.py`, `gradients.py`,
`odes.py`, and `boundary_conditions.py` are reused unmodified. The physical baseline agreed
before the `solve_bvp` experiment began (quasi-static assumption, ideal+degenerate EOS with
no ionization chemistry, Bell & Lin opacity, Schwarzschild criterion with instantaneous
efficient convection, grey Eddington photosphere, purely gravitational energy source — full
list in `PROGRESS.md` 2026-08-06) remains the strict physical contract for everything below.

---

## 2. Empirical Findings To Date (`bvp_experiment.py`)

A standalone, isolated experiment (`bvp_experiment.py` — imports and calls existing physics
modules, modifies none of them) ran three timeboxed spot-checks. Full logs in the dev
scratchpad; summarized here because they directly motivate the milestones in §3.

| T_center | Result | Key diagnostic |
|---|---|---|
| 13000K | **Crash**, both direct (α=1) and continuation (α=0→1) attempts | `solve_bvp`'s own Newton iteration (not the initial guess) proposes `lnP` as extreme as −5.3×10⁹, localized to `m/M_TOTAL ∈ [0.9992, 1.000]` — the outermost ~0.08% of the mass, at the photosphere |
| 2000K | **Partial** — direct attempt crashes the same way; continuation runs 14 real Newton iterations, ODE/collocation residual converges to 9.99×10⁻⁷ (near machine precision), but the boundary-condition residual stalls and oscillates around 2.68×10⁸ | `status=3`: unable to satisfy boundary-condition tolerance |
| 40000K | **Blocked before the experiment starts** — `bvp_solver.solve_static_structure()` (existing, shared, unmodified) cannot bracket a root to build a valid t=0 seed. Also fails at 20000K, 25000K; 30000K did not resolve within a 10-minute check | Not a `solve_bvp` finding — a pre-existing gap in shared seed-construction code, independent of shooting vs. `solve_bvp` (see Milestones 0 and 5) |

**Reading these together:** the 13000K crash is concentrated exactly where the shooting
kinks lived, reinforcing that the near-photosphere transition is the real numerical
battleground for *both* architectures, not a shooting-specific artifact. The 2000K result is
the most encouraging data point so far — `solve_bvp` can make the ODE system internally
self-consistent to high precision; it is specifically the boundary-condition system that
won't close, which is exactly the symptom Milestone 2 targets.

---

## 3. Strategy & Milestones

**STATUS 2026-08-07: Milestone 6 achieved a genuine, fully-converged `solve_bvp` solution
— the primary goal of this entire roadmap.** `T_CENTER_INITIAL` is now 11500K (lowered
from 13000K, itself a consequence of Milestone 6's physics corrections — see §3.6). The
40000K target remains out of scope until Milestone 5 is complete (§3.5); 2000K is tracked
as a secondary data point, not a current blocker.

Execution discipline for every milestone below (carried over from the fixes made this
session, where it repeatedly caught real bugs): change **one variable at a time**, verify
each change in isolation with a standalone check before wiring it into a real
`bvp_experiment.py` run, and keep `bvp_experiment.py` isolated from `bvp_solver.py` /
`gradients.py` / `eos.py` / `opacity.py` (call, don't modify) until a milestone is proven —
see §4. (Milestone 6 deliberately breaks this isolation for `eos.py`/`odes.py`/`bvp_solver.py`
where the fix is a genuine, permanent physics/shared-infrastructure correction rather than a
BVP-architecture experiment — see §3.6's own note.)

| # | Milestone | Depends on | Exit criterion |
|---|---|---|---|
| 0 | Ionization-sensitivity diagnostic | — | Quantified $x(\rho,T)$ profile at T=13000K, identifying whether the near-photosphere crash region is ionization-sensitive; a crude `MU`-lowering test showing whether the crash signature changes |
| 1 | Opacity bootstrapping (toy model → smoothed Bell & Lin directly) | — | `solve_bvp` converges at T=13000K with toy opacity; converges again with real, smoothed Bell & Lin (raw Bell & Lin is not retested — see §3.2) |
| 2 | Center boundary-condition self-consistency | — (parallel to 0, 1) | `r(m_min)` tied to the live trial center density, not a fixed pre-estimate; center-BC residual well-behaved under Newton probing |
| 3 | Log-space surface BC + mesh-density verification | — (parallel to 0–2) | Boundary-condition residual converges (no oscillation) at T=2000K; T=13000K crash location/signature changes or resolves; mesh density near the photosphere confirmed (plotted, not assumed) |
| 4 | Analytic Jacobians (`fun_jac`, `bc_jac`) | 1, 2, 3 | `solve_bvp` converges without relying on scipy's default finite-difference Jacobian; matches FD-Jacobian solution where both converge |
| 5 | Saha-equation EOS + high-T seed generation | Priority set by Milestone 0's result | `solve_static_structure` produces a valid seed at T≥20000K; `PLAN.md` Sub-task 8a's own exit criterion met |
| **6** | **State-vector nondimensionalization + EOS thermodynamic corrections (γ, μ, δ)** | **4's rank-deficiency finding** | **`solve_bvp` converges (`status=0`) at the project's active `T_CENTER_INITIAL` with residuals to machine precision — ACHIEVED 2026-08-07, §3.6** |

### 3.0 Milestone 0 — Ionization-Sensitivity Diagnostic

**Rationale.** `PLAN.md` Sub-task 8a (Saha-equation EOS ionization upgrade) is real,
already mandatory, and already known to matter for the high-T seed problem (§3.5). What is
*not* yet established is whether it is the cause of the *current* T=13000K near-photosphere
crash specifically. The evidence available before this milestone points in a mixed
direction: the crash is localized at `m/M_TOTAL∈[0.9992,1.000]` (outermost 0.08% of mass,
at the photosphere), not spread through the bulk/interior where a missing-ionization
pressure deficit would most directly apply; `solve_static_structure()` — same fixed
`MU=2.34`, same T=13000K — already converges cleanly to a well-defined R=4.1544 R_Jup
structure, not an inflating or crashing one (the documented radius inflation is at
T≳17000K, not 13000K). Given `PLAN.md` Sub-task 8a's own warning that Saha implementation
introduces *severe* new stiffness of its own, this needs a cheap empirical answer before
committing to it as a gating blocker, not an assumption in either direction.

**Steps:**
1. **Standalone diagnostic, no solver changes.** Using the existing converged T=13000K
   adiabatic profile (`solve_static_structure()`'s output, `r(m), P(m), T(m), rho(m)`),
   compute the Saha ionization fraction $x(\rho,T)$ for hydrogen at every profile point
   (pure-H Saha quadratic, first order — helium's much higher first ionization potential
   means it should stay neutral across this T range, a standard simplifying assumption to
   revisit only if this pass suggests it matters). Report $x$ as a function of `m/M_TOTAL`,
   with explicit attention to the crash region (`m/M_TOTAL∈[0.9992,1.000]`) versus the bulk
   interior.
2. Report the implied corrected mean molecular weight $\mu(x)$ at each point and its ratio
   to the current fixed `config.MU=2.34`, and note in passing (not a full calculation) that
   `MU=2.34` is itself a *molecular* value — Stage 3's own framing (already past H₂
   dissociation) implies the neutral-but-dissociated value (~1.2) is already more
   appropriate than the molecular one, before ionization is even considered; both effects
   should be visible in the same plot for comparison.
3. **Crude sensitivity test**: temporarily lower `config.MU` (informed by step 2's result in
   the crash region specifically) and re-run `bvp_experiment.py`'s T=13000K spot-check.
   Report whether the crash location, magnitude, or signature changes at all.
4. **Branch on the result:**
   - Crash region shows negligible ionization fraction, or the `MU`-lowering test doesn't
     change the crash → Milestone 5 stays scheduled but non-gating; proceed with Milestones
     1–4 as the primary path to T=13000K convergence.
   - Crash region shows significant ionization, or lowering `MU` measurably changes the
     crash → elevate Milestone 5 to immediate priority, ahead of 1–4, with the expectation
     (per `PLAN.md` Sub-task 8a) that this will introduce its own substantial numerical work.

**This is a diagnostic step only** — no changes to `eos.py`, `config.MU`, or any shared
module persist past this milestone; the `MU` override in step 3 is runtime-only, in a
throwaway test script, exactly like every `T_CENTER_INITIAL` override elsewhere in this
project.

**RESULT (2026-08-07), branch taken: proceed with Milestones 1–4; Milestone 5 stays
scheduled but non-gating.**

- **Saha ionization fraction is negligible everywhere in the profile**, and vanishes
  identically to ~0 in the crash region specifically. Peak $x\approx5.9\times10^{-4}$ at
  the hottest point (the center, T=13000K); $x\le4.5\times10^{-72}$ throughout
  `m/M_TOTAL≥0.999`. The implied $\mu(x)$ is essentially indistinguishable from the
  neutral-atomic value (1.2773–1.2780) across the *entire* profile — ionization is not a
  meaningful correction anywhere in this structure at this $T_\text{center}$.
- **The real, verified gap is dissociation, not ionization**: `config.MU=2.34` is a
  *molecular* value, while this profile — consistent with Stage 3's own framing as
  post-H₂-dissociation — should use the neutral-*atomic* value ($\mu\approx1.278$), a
  factor of $\approx1.83\times$, present uniformly across the whole profile including the
  crash region. This is a real, quantified inconsistency, but a different physical
  mechanism than the one originally proposed.
- **Clean, single-variable sensitivity test**: holding the starting geometry/mesh fixed
  (the same well-matched `MU=2.34` seed every other crash test used) and changing only
  `MU→1.278` for the `solve_bvp` attempt itself, the crash is **unchanged** — same location
  (`m/M_TOTAL≳0.9995`), same `eos.density` Newton-Raphson failure mechanism, same
  near-instant (<0.1s) timing, for both the direct and continuation attempts. (A first,
  confounded version of this test — which needed a widened bracket search that landed on a
  poorly-matched seed, `m_surface/M_TOTAL=1.43` — showed a more benign failure mode; the
  clean rerun shows that was an artifact of the mismatched seed, not of `MU` itself, and
  is a useful lesson for Milestone 5's own seed-generation work.)

**Conclusion: neither ionization nor the (real, verified) dissociation correction is
implicated in the current T=13000K near-photosphere crash.** Milestone 5 remains real,
scheduled, and eventually necessary (particularly for the 40000K target and for physical
accuracy generally) — but a full Saha-equation implementation, with its own documented
severe stiffness, is not warranted as a gate on the current crash. A much cheaper,
lower-risk fix (a simple dissociation-aware $\mu(T)$ correction, no ionization physics
needed) would close the *verified* gap if desired, independent of Milestones 1–4's
progress on the crash itself.

### 3.1 Milestone 1 — Opacity Bootstrapping (The Toy Model)

**Rationale.** We are currently fighting the BVP engine and the full complexity of Bell &
Lin's 8-regime opacity simultaneously. Bell & Lin's opacity is not confined to the boundary
condition — `odes.stellar_odes` calls it at every mesh point — so a hard regime switch is a
*global*, not boundary-local, exposure to Newton-iteration non-smoothness. Isolating it
first gives a clean signal about whether the BVP mesh/BC/log-transform machinery itself is
sound, before also needing opacity's specific texture to cooperate.

**Steps:**
1. Check which Bell & Lin regime(s) the actual T=13000K near-photosphere profile falls into
   (`opacity.determine_regime`, already exists) — the toy model should match the physically
   relevant regime, not an arbitrary guess.
2. Implement a **local, `bvp_experiment.py`-only** toy opacity: reuse that regime's own
   `(κ_i, a, b)` power-law coefficients from `opacity.py`'s `REGIMES` table directly (e.g.
   the Kramers bound-free/free-free form), not an invented formula — this preserves the
   real physical magnitude and `(ρ,T)`-scaling while removing the hard switches. Do **not**
   modify `opacity.py`.
3. Re-run the T=13000K spot-check with only this substitution changed.
4. **Branch on the result:**
   - Crashes the same way → opacity is ruled out as the (sole) cause; deprioritize
     re-enabling smoothing and focus effort on Milestones 2–3.
   - Converges → opacity kinks are confirmed central. Read and standalone-verify
     `opacity.py`'s existing `OPACITY_SMOOTH_TRANSITIONS=True` logistic-blend
     implementation (not yet checked by us), then re-run with **real, smoothed** Bell & Lin
     to empirically confirm the fix transfers back — do not assume it does.

**Caveat to carry forward, explicitly:** a toy-opacity convergence proves the mesh/BC/
log-transform/degenerate-EOS/Newton machinery *can* handle a steep near-photosphere
transition when it is smooth. It does not by itself guarantee real (even smoothed) Bell &
Lin reproduces an equally tractable boundary layer — its different magnitude and
`(ρ,T)`-coupling can still matter. Treat this milestone as a necessary stepping stone, not a
final validation; the real validation is step 4 above.

**RESULT (2026-08-07), branch taken: opacity ruled out; proceed to Milestone 2, do not
pursue smoothed Bell & Lin.**

Regime check on the actual T=13000K profile (`opacity.determine_regime`): the crash region
(`m/M_TOTAL≥0.999`) falls almost entirely in regime 2 ("Metal grains", $\kappa=0.1\,T^{0.5}$,
no density dependence — the cleanest possible toy, chosen for exactly this reason). Toy
opacity implemented as a `bvp_experiment.py`-local monkeypatch of the
`opacity.bell_lin_opacity` module attribute (`opacity_override` context manager) — reaches
`odes.py`'s and `boundary_conditions.photospheric_pressure`'s calls without editing either
file, since both do `import opacity` and look the function up dynamically at call time.

**The T=13000K crash reproduced identically under toy opacity**: same location
(`m/M_TOTAL∈[0.9995,1.000]`), same `eos.density` Newton-Raphson failure mechanism, same
near-instant (<0.1s) timing, for both the direct and continuation attempts — indistinguishable
from every real-Bell&Lin run. Removing *every* opacity hard switch, down to a single smooth
power law spanning the whole domain (strictly smoother than smoothed Bell & Lin would be),
changed nothing.

**Combined with Milestone 0's results, three independent candidate causes have now each
been directly, empirically ruled out** (ionization, the dissociation-$\mu$ gap, and opacity
switches) — each removed or corrected in isolation, each leaving the crash's location,
mechanism, and timing completely unchanged. The evidence increasingly points at the mesh/BC
formulation or the underlying Newton/Jacobian conditioning itself (Milestones 2–4), not at
any single physics term. Per this milestone's own branch logic, the smoothed-Bell&Lin
follow-up step is skipped — not warranted given the toy result — and effort moves to
Milestone 2 (center boundary-condition self-consistency).

### 3.2 Milestone 2 — Center Boundary-Condition Self-Consistency

**Rationale.** `bvp_experiment.py`'s center residual currently pins `r(m_min)` to
`r_seed=state_0.r[0]`, a value computed once, before the bracket search, from the Lane-Emden
*T=0-degenerate-limit* seed — not re-derived from the actual converged (or, under
`solve_bvp`, the actual live *trial*) center state. Checked directly: this is algebraically
the same leading-order form as the analytic constant-density-center relation,
$r(m)\approx\left(\frac{3m}{4\pi\rho_c}\right)^{1/3}$ — `bvp_solver._adiabatic_center_guess`
already builds `r_start = R_guess*(m_min/M_TOTAL)^{1/3}`, which reduces to exactly that
formula once $\rho_c$ is expressed via $M_\text{TOTAL}=\frac{4}{3}\pi\rho_c R_\text{guess}^3$
— so this is not a missing formula, it's a *fixed, pre-converged* evaluation of the right
formula, never re-tied to the actual center state being solved for.

**Steps:**
1. Replace the fixed `r_seed` residual with `r_a - (3*m_min/(4*pi*rho_c))**(1/3)`, where
   `rho_c = eos.density(P_a, T_a, config.MU, config.MU_E)` is evaluated on the *live* trial
   center state (`ya[1]`, `ya[3]`) at every Newton iteration, not precomputed. This is a
   `bvp_experiment.py`-local change to the `bc()` closure; `boundary_conditions.py` is
   unaffected.
2. **Known risk to watch, not a reason to skip this**: this couples the boundary condition
   itself to `eos.density`'s own Newton-Raphson solve — the same function whose
   non-convergence assertion has fired throughout this investigation whenever a trial
   `(P,T)` leaves its physical domain. Test whether this introduces new failure modes near
   the center specifically (a different location from every crash seen so far), not just
   whether it changes the near-photosphere crash (it may not — the center and surface are
   not obviously coupled).
3. $L(m_\text{min})\approx0$ is left as-is — there is no nuclear source term in this
   project's physics (purely gravitational/KH), so this is already the exact leading-order
   condition, not an approximation needing a higher-order correction; deprioritized as
   polish, not a fix.

**RESULT (2026-08-07), branch taken: coupling risk did not materialize, but this is
another confirmed-negative result; proceed to Milestone 3.**

Standalone check first: at the actual converged T=13000K center state, the new
self-consistent $r_\text{analytic}$ differs from the old fixed `r_seed` by **30%** — a
real, non-trivial correction, not cosmetic, confirming the Lane-Emden T=0-degenerate-limit
seed was genuinely off even at the well-matched starting point, not just under extreme
trials.

Wired in and re-run at T=13000K (real Bell & Lin — opacity already ruled out, no need to
combine with the toy model): **the flagged coupling risk did not materialize** — no
`eos.density` failure occurred inside `bc()` itself anywhere in the run (an explicit
diagnostic print was added specifically to catch this; it never fired). But **the crash is
otherwise unchanged**: identical location (`m/M_TOTAL∈[0.9992,1.000]`), identical
`eos.density` Newton-Raphson failure mechanism inside the RHS (`implicit_rhs_vectorized`,
not the BC), identical near-instant timing, for both direct and continuation attempts.

**This is the fourth independent hypothesis ruled out by direct test** (ionization,
dissociation-$\mu$, opacity switches, and now the center boundary condition), each varied
in isolation, each leaving the crash completely unchanged. The pattern is starting to say
something on its own: every fix tried so far touches a specific *physics* term; none has
moved the crash at all. This raises the prior on Milestone 4 (analytic Jacobians) — if the
issue is in how scipy's default finite-difference Jacobian estimation navigates this
specific structural feature (the photospheric transition), rather than in any one physics
term, no amount of smoothing individual terms would be expected to help, which is
consistent with everything observed so far. Milestone 3 (log-space surface BC + mesh
verification) is still next per the agreed sequence, but going in with that expectation
explicitly named rather than assumed away.

### 3.3 Milestone 3 — Log-Space Surface Boundary Formulation + Mesh Verification

**Rationale.** The photospheric mechanical condition is currently evaluated as
`P_b − (2/3)(g/κ) = 0` in linear pressure (dyn/cm²), sitting in the same residual vector as
center conditions near machine-zero, while every *state* variable is already log-transformed
specifically to avoid this kind of scale mismatch. Reformulating the residual as

$$\ln P_b - \ln\!\left(\frac{2}{3}\frac{g}{\kappa}\right) = 0$$

is consistent with that existing design principle, not a new pattern. It is a plausible,
though not confirmed, contributor to the observed Newton-step blowup (`lnP → -5.3×10⁹`).

Separately: before attributing anything further to the solver, mesh resolution near the
photosphere should be confirmed rather than assumed. Checked directly: `_build_output_grid`
already allocates 30% of points to the outer 10% of mass, log-spaced in distance-to-surface
down to `GRID_OUTER_REFINEMENT=1e-4` of that region's span — and the crash diagnostic itself
found 290 of 2000 mesh points already concentrated in exactly the crashing region
(`m/M_TOTAL∈[0.9992,1.000]`), roughly half the outer-region point budget in the innermost 1%
of that zone. This is evidence *against* insufficient mesh resolution being the cause here,
not for it — but it has not yet been shown as an explicit, visible check, only inferred from
the crash diagnostic's own printout.

**Steps:**
1. Reformulate the mechanical surface residual in `bvp_experiment.py`'s `bc()` closure (log
   space); leave `boundary_conditions.py` itself untouched — the reformulation is BVP-side
   residual bookkeeping, not a change to the physical condition.
2. Test in isolation at T=13000K (real opacity, no Milestone 1 substitution) to attribute
   any improvement specifically to this change.
3. Also test at T=2000K, where the existing continuation attempt already converges the ODE
   system but stalls on the BC residual specifically — this is the most direct existing
   evidence of a boundary-condition-side conditioning problem, and the most likely case to
   show a clean before/after signal.
4. Produce a visible check (a plot of mesh point density vs. `m/M_TOTAL`, log-scaled near
   the surface) confirming the refinement claim above directly, rather than relying on the
   inferred point-count from the crash diagnostic.

**RESULT (2026-08-07): mesh confirmed as claimed; fifth negative result on the crash
itself. Proceeding to Milestone 4 per direction — not chased further.**

Mesh check: confirmed directly (not just inferred) — 286/2000 points (14.3% of the total
mesh) sit in the outermost 0.08% of mass alone; local point density rises by ~4 orders of
magnitude approaching the photosphere. The refinement claim holds exactly as designed.

Log-space mechanical BC: implemented as a `bvp_experiment.py`-local reformulation
(`res[2] = yb[1] - ln(P_photo)`, reusing `yb[1]` as the already-available `ln(P_b)` state
component directly, no redundant exp/log round trip); `boundary_conditions.py` untouched.
Standalone check passed (finite, sensibly small residual at the initial guess).

**T=13000K: crashes identically** — same location, same `eos.density` failure mechanism,
same near-instant timing, same reported `R_surface`/`L_surface` to 4+ significant figures.

**T=2000K: also essentially unchanged** — `status=3`, ODE residual converges to
$9.94\times10^{-7}$ (vs. the pre-Milestone-3 $9.99\times10^{-7}$), boundary residual still
stuck oscillating around $2.68\times10^8$, same erratic alternating pattern between
near-zero and large values iteration-to-iteration. Since only the *mechanical* (pressure)
residual was reformulated in log space and this milestone's target — this is consistent
with the *thermal* (L) residual being the actual dominant/binding term at T=2000K, not the
mechanical one; not confirmed directly (would need decomposing the 4-component residual
vector per-iteration to attribute cleanly), left as an open question rather than chased
further per explicit direction to move to Milestone 4.

**Fifth independent hypothesis (mesh resolution, boundary-condition scaling) ruled out or
inconclusive on the crash itself, joining ionization, dissociation-$\mu$, opacity, and the
center BC.** All evidence continues to point toward Milestone 4 (analytic Jacobians) as the
next real test of the underlying hypothesis - that the failure is in how scipy's default
finite-difference Jacobian estimation navigates this problem's local stiffness, not in any
individual physics or boundary term.

### 3.4 Milestone 4 — Analytic Jacobians

**Rationale.** Every `solve_bvp` attempt so far has used scipy's default finite-difference
Jacobian estimation (`fun_jac`/`bc_jac` were never supplied). This session independently
established that this problem has regions of extreme local sensitivity (a finite-difference
probe swinging a shooting-side residual by 30+ orders of magnitude from a machine-epsilon
perturbation, before the L-floor fix). scipy's default FD-Jacobian estimation is exposed to
exactly the same risk — even genuinely smooth-but-steep physics can produce a badly-estimated
FD Jacobian if the probe step isn't matched to the local curvature scale. Supplying analytic
derivatives removes this failure channel rather than hoping the FD step behaves.

**Steps:**
1. Derive `fun_jac(x, y)` (∂(RHS)/∂y for the 4-ODE system — tractable: mostly power-law/EOS
   terms plus one opacity table lookup, whose derivative is piecewise-analytic within a
   regime) and `bc_jac(ya, yb)` (∂(residuals)/∂(ya,yb)) by hand or via a symbolic/autodiff
   tool.
2. Cross-check every analytic derivative against a finite-difference estimate at several
   representative points before trusting it (the same standalone-verify-before-wiring-in
   discipline used for every fix this session) — a wrong analytic Jacobian is worse than
   none, since it can steer Newton confidently in the wrong direction.
3. Re-run T=13000K (and 2000K) with analytic Jacobians supplied; compare against the
   FD-Jacobian result wherever both converge, as a correctness check on the Jacobian itself.

**Sequencing note:** attempted after Milestones 1–3, since a correct analytic Jacobian for a
still-kinked or badly-scaled residual formulation just computes an exact derivative of the
wrong thing faster — it does not substitute for removing the non-smoothness/scaling issues
those milestones address.

**RESULT (2026-08-07): Jacobians derived and independently verified correct (1e-10
relative agreement with finite differences at every tested point, from deep interior to
the crash boundary) — and the crash reproduces IDENTICALLY anyway. This rules out
"inaccurate FD Jacobian estimation" as the cause entirely, and reveals what the actual
structural problem is.**

`fun_jac`/`bc_jac` derived by hand (implicit differentiation through `eos.density`'s
Newton-Raphson EOS inversion; regime-local analytic opacity derivatives; exact derivatives
of both smoothed hyperbolic terms). `verify_jacobians()` cross-checks both against central
finite differences before any real use — this caught a real bug in the *verification
metric itself* (not the Jacobian: dividing by a row's raw output value fails when that
output is legitimately zero, e.g. `dL_dm=0` exactly at the initial guess since `T=T_prev`
there; fixed by normalizing against each row's own matrix-norm scale instead). Once fixed,
every one of 15 randomly-sampled mesh points — spanning the full profile — matched to
$\sim10^{-10}$–$10^{-11}$ relative precision, and `bc_jac` matched to $\sim10^{-7}$
(FD's own truncation floor). Both are correct.

**Wired into `solve_bvp` via `fun_jac`/`bc_jac`, re-run at T=13000K: identical crash** —
same location, same `eos.density` failure, same near-instant timing, same reported
`R_surface`/`L_surface`.

**Follow-up structural diagnosis** (not originally scoped, pursued because the result was
unexpected enough to demand an explanation): checked the *condition number* of the local
4×4 Jacobian at points spanning the whole profile. **Effectively singular
($\text{cond}=\infty$ or $\sim10^{18}$) at nearly every point tested, not just near the
photosphere.** Traced to its exact cause: `d(grad_eff)/d(grad_rad) = 0.0` identically at
every sampled point, because `grad_rad` exceeds `grad_ad` by factors of $10^3$–$10^9$
throughout — **100% of the mesh is classified convective** at this trial state (the
adiabatic seed, evaluated against the real, non-adiabatic $\nabla_\text{rad}$ formula), so
the smoothed Schwarzschild switch (`config.GRAD_EFF_SWITCH_EPSILON=10^{-4}`) has completely
saturated. Algebraically, this makes row 3 (`dlnT/dm`) an exact scalar multiple of row 1
(`dlnP/dm`) at every such point — $L$ structurally decouples from the $P$-$T$ relation
whenever convection is locally "infinitely efficient," so the true 4-variable Jacobian is
rank-deficient there, independent of how precisely it is computed. An exact copy of a
near-singular matrix is still near-singular.

**This directly implicates the "infinitely-efficient convection" idealization** already
named (but treated as future physics polish, not a live blocker) when the Schwarzschild
kink was fixed: the standard smoother alternative, mixing-length theory, gives convective
zones a genuinely smooth, *never-exactly-zero* gradient dependence on the local state
(finite convective velocity as a function of superadiabaticity), rather than snapping to a
locally-constant $\nabla_\text{ad}$ with zero sensitivity. That would restore rank to
exactly the rows/points currently degenerate. Not implemented — this is a substantive
physics addition, not a numerical patch — but it is now the best-supported hypothesis for
*why* every one of Milestones 0–4 left the crash completely unchanged: none of them
addressed a structural rank deficiency in the state representation itself.

### 3.5 Milestone 5 — The Saha Equation & High-T Seeds

**Priority set by Milestone 0's result (§3.0), not unconditionally deferred.** If the
ionization-sensitivity diagnostic shows the T=13000K near-photosphere crash is genuinely
ionization-sensitive, this milestone moves ahead of 1–4. If not, it stays scheduled here,
gating only the 40000K target — current focus remains T=13000K (Milestones 1–4) either way.

**Known physical cause.** `bvp_solver.solve_static_structure()`'s adiabatic seed construction
fails to bracket a root at T≥20000K. `eos.py`'s ideal-gas term uses a fixed, neutral mean
molecular weight (`config.MU=2.34`) at every temperature — with no hydrogen/helium ionization
physics, thermal pressure support is systematically overestimated at high T, inflating the
equilibrium radius rather than producing the compact structure the seed search is looking
for. This is not a new discovery: `PLAN.md`'s own `T_CENTER_INITIAL` documentation already
records that direct marching showed R exceeding 4 R_Jup by T~17000K and continuing to climb,
and `PLAN.md` **Sub-task 8a — EOS Ionization Upgrade (Saha Equation)** is already the
formally-scoped, mandatory future sub-task for exactly this gap (full deliverables,
numerical warnings, and exit criterion already documented there — not repeated here).

**This milestone = Sub-task 8a's completion, viewed from the BVP-seed-generation angle.**
Implementing a Saha-equation ionization fraction $x(\rho,T)$ and replacing the fixed `MU`
with `μ(ρ,T)` in `eos.py`'s ideal-gas term is the physical fix required before a compact,
physically meaningful adiabatic seed can be constructed at 20000K–40000K at all.

**One nuance not to lose:** the bracket-search failure has two logically separate possible
causes — (a) the geometric-expansion search window is numerically too narrow (it explores
only a fixed ~7.3× range around a T-*independent* seed, derived from the pure T=0 degenerate
limit), or (b) no compact-radius root exists at all under the current EOS. A cheap diagnostic
(a wide, many-orders-of-magnitude `mass_error(P)` sweep at T=20000K) was proposed but not
yet run, and should be — it is possible the bracket search *also* needs widening or a
better (e.g. warm-started, T-marching) seed strategy even after Saha closes the physics gap,
since the true root may simply have moved further from a fixed T=0-limit seed than a search
window can find, independent of whether the root itself is now "compact" again.

**Exit criterion:** matches `PLAN.md` Sub-task 8a's own — `solve_timestep` (or its BVP
equivalent) converges through a full ionization transition with honestly-tuned tolerances —
plus, specifically for this roadmap, `solve_static_structure` producing a valid, compact
seed at T=40000K for `bvp_experiment.py` to consume.

---

### 3.6 Milestone 6 — State-Vector Nondimensionalization + EOS Thermodynamic Corrections

**★ ACHIEVED 2026-08-07 — the first genuine, fully-converged `solve_bvp` solution this
roadmap has produced.** This milestone was executed under an explicit one-week thesis
deadline, as a deliberately pragmatic response to Milestone 4's rank-deficiency finding: a
full mixing-length-theory fix was ruled out of scope (too slow for the deadline); this
milestone is the fast, defensible alternative — attack the Jacobian's *conditioning*
directly via nondimensionalization, and close two independent, silently-wrong physics terms
(§3.6.1) at the same time, rather than the deeper structural rank issue itself.

#### 3.6.1 Physics corrections (`config.py`, `eos.py`, `odes.py`) — permanent, not BVP-local

Unlike every other milestone in this roadmap, these are genuine physics fixes applied
directly to shared modules, not `bvp_experiment.py`-local experiments — deliberately, since
they are corrections to the physical model itself, load-bearing for shooting too, not
solver-architecture choices.

- **$\gamma:1.4\to5/3$, $\mu:2.34\to1.278$.** Stage 3 (this project's scope) is, by its own
  definition, already past H$_2$ dissociation — the envelope is atomic, not molecular,
  essentially everywhere in the relevant T range. Molecular values (diatomic $\gamma=7/5$,
  $\mu=2.34$) were being used regardless. Corrected to the atomic pair (monatomic
  $\gamma=5/3$; $\mu\approx1.278$, solar H/He). Consequence flagged and accepted in advance:
  $\nabla_\text{ad}=(\gamma-1)/\gamma$ rises from 0.2857 to 0.4, a genuine ~40% shift in the
  Schwarzschild threshold everywhere in the star.
- **$\delta$ coefficient in the energy equation.** `odes.py` hardcoded
  $\delta=1$ in $dL/dm=-c_p\,dT/dt+(\delta/\rho)\,dP/dt$ (Kippenhahn & Weigert eq. 4.26) — exact
  only for a pure ideal gas. This project's EOS is ideal+degenerate, and degeneracy
  dominates the interior at this project's T range. New `eos.thermodynamic_delta(rho,T,mu,
  mu_e)`, derived by implicit differentiation of the EOS's defining equation
  ($\delta=P_\text{ideal}/(\rho D)$, $D$ = the same denominator `eos.density`'s own
  Newton iteration already uses) — verified against both limiting cases ($\delta\to1$ pure
  ideal, $\delta\to0$ fully degenerate) and the actual T=11500K center density
  ($\delta\approx0.205$ there — confirming the old hardcoded value of 1 was off by nearly
  5× at exactly the point it mattered most).
- **`config.T_CENTER_INITIAL`: $13000\text{K}\to11500\text{K}$ — a direct, measured
  consequence of the $\gamma$/$\mu$ correction, not a separate choice.** The corrected,
  more atomic composition provides genuinely more ideal-gas thermal pressure support at
  fixed $(\rho,T)$. Swept `mass_error(P_\text{center})` across nearly 6 orders of magnitude
  at 13000K under the new EOS: it never reaches zero — its *minimum* still overshoots
  $M_\text{TOTAL}$ by 5.36%. Not a bracket-search-window problem (already ruled out by the
  sweep's range); genuinely no compact-radius adiabatic seed exists at 13000K anymore.
  Scanned T and found the feasibility boundary between 12000K (root exists, marginally) and
  13000K (does not); chose 11500K for comfortable margin below that edge rather than sitting
  on it. `bvp_solver.solve_static_structure`'s bracket-search window was also widened
  (`1.01`$^{200}$→`1.03`$^{300}$, ~7.3×→~7100×) as a related, necessary companion fix — the
  same T-independent-seed limitation flagged for Milestone 5, now also binding here.

#### 3.6.2 State-vector scaling (`bvp_experiment.py`, local — the numerics fix)

New state $z=[\hat r,\ln P,\hat L,\ln T]$, replacing $y=[r,\ln P,L,\ln T]$:

$$\hat r = \frac{r}{R_\text{Jup}} \qquad \hat L = \operatorname{arcsinh}\!\left(\frac{L}{L_\text{scale}}\right),\quad L_\text{scale}=\text{config.L\_KH\_SCALE\_ERG\_S}$$

Motivation, verified directly rather than assumed: a single Jacobian-verification point
this session showed $y=[r{=}2.9\times10^{10},\ \ln P{=}5.9,\ L{=}2.6\times10^{29},\ \ln
T{=}3.6]$ — $L$ is **28 orders of magnitude** larger than $\ln T$ in the same vector
Newton must invert. `arcsinh` was chosen over a hand-rolled sign$\cdot$log1p for being a
single smooth closed form with a simple, well-conditioned derivative
($1/\sqrt{L^2+L_\text{scale}^2}$) everywhere, including at $L=0$.

The Jacobian transform is **not** a trivial rescaling: because $\hat L$'s scaling factor
$\Phi'_2(L)=1/\sqrt{L^2+L_\text{scale}^2}$ is itself $L$-dependent (nonlinear, unlike
$\hat r$'s constant $1/R_\text{Jup}$), differentiating the scaled RHS a second time picks
up a genuine product-rule correction term,
$-L/(L^2+L_\text{scale}^2)\cdot f_2(y)$, present only in the $(\hat L,\hat L)$
entry — easy to silently drop, producing a confidently-wrong rather than absent Jacobian.
Implemented explicitly (`implicit_rhs_jacobian_scaled`).

**Two real bugs caught by the required FD cross-check before this was trusted** (not found
by inspection):
1. `_to_physical(z)` returns $[r,\ln P,L,\ln T]$ (matching `implicit_rhs_vectorized`'s
   existing mixed convention — $P,T$ stay logarithmic, exponentiated only where consumed),
   but the new scaled boundary-condition functions unpacked its output as if fully physical
   — feeding `eos.density` $\ln P\approx25$ as if it were a pressure in dyn/cm² (should be
   $\sim10^{11}$). Caught as a 25× `bc_jac` disagreement against FD.
2. A thermal-BC Jacobian term (`dLexp_dT`) copied the *already chain-ruled* $d/d(\ln T_b)$
   form ($T_b^4$, correct for the old, unscaled Jacobian which differentiates directly
   w.r.t. $\ln T_b$) into a slot meant to hold the *plain* $d/dT_b$ derivative ($T_b^3$),
   then applied the chain-rule multiplication a second time. Caught as a ~10× disagreement.

After both fixes: `fun_jac` matches FD to $6.5\times10^{-7}$, `bc_jac` to
$1.5\times10^{-5}$ — both well inside the $10^{-4}$ verification gate, every entry checked.

#### 3.6.3 Result

With scaling + corrected physics + analytic Jacobians combined, the $\alpha$-continuation
converged **cleanly through $\alpha=0.00,0.25,0.50,0.75$** — residuals to
$9\times10^{-7}$, boundary residuals to machine precision ($6.94\times10^{-18}$) — the
**first time in this entire investigation** the ramp advanced past the historical
$\alpha\approx0.05$–$0.09$ wall region at all, let alone this cleanly.

The exact literal $\alpha=1.0$ endpoint initially still failed — not via the old
`eos.density` crash, but via a *new* failure mode: exponentially escalating mesh refinement
(tens of thousands of nodes added per iteration) diverging to NaN, regardless of how small
the preceding step was. Diagnosed precisely, not by assumption: $\alpha=0.9,0.95,0.98,
0.99,0.995,0.999$ **all converge cleanly** (confirmed by finer stepping); only the literal
value $1.0$ fails. Since `dT_dm_real` (the real, Schwarzschild-selected gradient) is
computed *identically* at every $\alpha>0$ — the only difference between $\alpha=0.999$ and
$\alpha=1.0$ is whether a vanishing fraction of the smooth, constant adiabatic gradient is
blended in — this is strong evidence that the tiny adiabatic admixture acts as a
**regularizer**, damping a marginal instability in the pure, unblended system (consistent
with, though not fully explained by, Milestone 4's rank-deficiency finding) that the blend
was masking.

**Fix: `ALPHA_MAX = 1.0 - 1e-5`, not exactly 1.0** — a quantifiably negligible (0.001%)
adiabatic contamination, not a discretization artifact. With this, the full continuation
**converges completely, `status=0`, at every step**:

```
center:  P_c=1.152e+11 dyn/cm^2, T_c=1.152e+04 K   (target was 11500K - self-consistent)
surface: R=5.109 R_Jup, T_surf=49.36 K, L_surf=-2.85e+23 erg/s (-7.4e-11 L_sun)
max residual: 9.79e-7 | boundary residuals: ~1e-18 to 1e-36 (machine precision)
```

**Honest flag, not swept under the rug**: `L_surface` came out slightly *negative*.
Internally consistent with the thermal BC itself ($T_\text{surf}\approx49.36$K landed just
below $T_\text{NEB}=50$K, and $L=4\pi r^2\sigma(T^4-T_\text{NEB}^4)$ is negative whenever
the photosphere is marginally cooler than the ambient field) rather than a numerical
artifact — residuals are excellent throughout — but it is a genuine, open physical
question (a contracting, cooling protoplanet radiating net energy *inward*) that deserves
scrutiny before being trusted quantitatively, not something to bury under the headline
convergence result.

**What remains open, explicitly not resolved by this milestone:**
- *Why* $\alpha=1.0$ exactly is unstable while $\alpha=0.9999$ is not — regularization is
  the working hypothesis, not a proven mechanism. Milestone 4's rank-deficiency finding
  (100% convective saturation, $d(\nabla_\text{eff})/d(\nabla_\text{rad})=0$ almost
  everywhere) is the best-supported *structural* explanation on file, but the connection to
  this specific $\alpha$-threshold behavior is inference, not derivation.
- The negative `L_surface` finding above — **update 2026-08-07, see §3.6.4: now confirmed
  reproducible at a second temperature, not a one-off.**
- ~~This result is at $T_\text{CENTER\_INITIAL}=11500$K only — not yet re-tested at 2000K or
  (blocked regardless, Milestone 5) 40000K.~~ **Resolved by §3.6.4 for the atomic/
  post-dissociation regime (12000K); 2000K is explicitly out of scope for this EOS, see
  below; 40000K remains blocked on Milestone 5 as before.**

#### 3.6.4 Second-temperature confirmation (2026-08-07)

Per §6's "before thesis-ready" checklist item 2, the pipeline (scaling + corrected physics +
analytic Jacobians + `ALPHA_MAX`) was re-run at a second $T_\text{CENTER\_INITIAL}$, unchanged
code, to check whether the 11500K convergence was a single-point success.

**First attempt, T=2000K — rejected as an invalid test point, not a fix failure.** The
Jacobian-verification metric fix (bc_jac now uses the same row-normalized comparison as
fun_jac, closing a false-alarm gap found while preparing this run — `bc_jac` err dropped
from a spurious $4.37\times10^{-4}$ to $8.90\times10^{-11}$, confirming that earlier failure
was FD roundoff noise on near-zero entries, not a formula bug) passed cleanly. But the solve
itself did not converge: continuation advanced through $\alpha=0.00$ cleanly, then diverged
(mesh nodes exceeded, residual $5\times10^2$) at $\alpha=0.50$ — far earlier than 11500K's
smooth run to $\alpha_\text{max}$. Diagnosis: **this is expected, not a regression.**
`config.py`'s halt-condition section documents that this project's $t=0$ state is defined to
start *already past* H$_2$ dissociation (the old $T_\text{DISSOCIATION\_LIMIT}=2000$K halt
was deliberately removed — the model now cools *from* a hot compact start rather than heating
*toward* 2000K). `config.MU=1.278`/`GAMMA=5/3` (atomic) is a single global constant, not
temperature-dependent, and at $T=2000$K real hydrogen is molecular. Testing there exercises
the corrected EOS well outside the composition regime it was derived for — an EOS/scope
mismatch, not evidence against the Milestone 6 numerics. **2000K is retired as a Milestone 6
test point for this reason** (kept here as a documented negative result so it isn't
re-attempted and re-diagnosed from scratch later).

**Second attempt, T=12000K — the real second data point, confirmed.** Inside the atomic/
post-dissociation regime (previously flagged "marginal-feasible" in the T-scan that set
`T_CENTER_INITIAL=11500K`). Result:

```
status=0, message="The algorithm converged to the desired accuracy."
center:  P_c=7.518e+10 dyn/cm^2, T_c=1.200e+04 K
surface: R=5.6785 R_Jup, P_surf=109.10 dyn/cm^2, T_surf=49.556 K, L_surf=-2.460e+23 erg/s
max residual: 8.88e-16 (machine precision - tighter even than 11500K's 9.79e-7)
```

Converged cleanly through the *entire* continuation ladder ($\alpha=0.00,0.50,0.90,0.99$,
$\alpha_\text{max}$) with no mesh blow-ups. (The literal $\alpha=1.0$ direct attempt still
crashed first, via a **new** failure mode this time — `eos.density`'s Newton-Raphson failed
to converge on an extreme trial $(P,T)$ the solver probed mid-iteration, not the mesh-explosion
seen at 11500K — but the `ALPHA_MAX` continuation fallback absorbed it exactly as designed,
consistent with "the exact $\alpha=1.0$ endpoint is fragile in more than one way" rather than
one specific bug.)

**On the negative `L_surface`**: reproduced at 12000K, same sign, same order of magnitude
($-2.46\times10^{23}$ erg/s vs. 11500K's $-2.85\times10^{23}$), with $T_\text{surf}$ again
landing just below $T_\text{NEB}=50$K (49.556K here vs. 49.36K before, both $<1.3\%$ below).
This is no longer a single-run curiosity — it is a **reproducible feature of this converged
solution family**, not run-to-run noise. That doesn't resolve it, but it reframes it: it is
consistent enough to be a real property of the current photospheric BC + `DT_RELAX`
pseudo-timestep combination, not a fluke — worth explaining, but not disqualifying, and the
recommended next diagnostic remains `time_stepper`'s first real-$dt$ step (real time
evolution should push $T_\text{surf}$ decisively away from $T_\text{NEB}$ in one direction or
the other, unlike the pseudo-relaxation step used here).

---

## 4. Isolation & Development Conventions (standing rules for this roadmap)

- `bvp_experiment.py` remains isolated from `bvp_solver.py`, `gradients.py`, `eos.py`,
  `opacity.py`, `boundary_conditions.py` — call, never modify — until a milestone is proven
  end-to-end. This includes the Milestone 1 toy opacity (local to `bvp_experiment.py`) and
  the Milestone 2 log-space BC reformulation (local wrapper, not a `boundary_conditions.py`
  edit).
- `config.T_CENTER_INITIAL` is overridden at runtime per spot-check, never persisted to
  `config.py`, matching the pattern already established this session.
- One variable changed per test run. Do not combine Milestones 1 and 2 in a single test
  until each has an isolated, attributable result — the whole point of separating them is to
  know which one actually mattered.
- Every smoothing/reformulation gets a standalone numerical check (a plot or printed
  before/after table, per CLAUDE.md's visible-check preference) before being wired into a
  real `bvp_experiment.py` run — this caught two real bugs earlier this session (a
  cross-regime epsilon-scale mismatch, a catastrophic-cancellation formula error) that would
  otherwise have shown up as confusing, hard-to-diagnose downstream failures.
- Timeboxed spot-checks (30–45 min per configuration), same discipline as the original
  `solve_bvp` experiment — a milestone that isn't showing a clear signal within its box gets
  reported honestly, not chased indefinitely.

---

## 5. Open Questions & Risks

- **Toy-opacity transferability is partial, not proven** (§3.1) — must be explicitly
  re-tested with real, smoothed Bell & Lin, not assumed.
- **T=2000K's oscillating BC residual is not yet explained** (Milestone 2/3-era finding,
  predates the γ/μ atomic correction). Superseded in practical terms by §3.6.4: 2000K is now
  understood to be outside the atomic-EOS regime `config.MU=1.278`/`GAMMA=5/3` describes, so
  it is retired as a Milestone 6 test point rather than further chased — but the original
  conditioning question for a hypothetical molecular-regime BVP treatment remains formally
  unanswered if this project ever needs to model the pre-dissociation stage explicitly.
- **Analytic Jacobian derivation is itself error-prone** for a system this coupled (EOS →
  opacity table lookup → radiative gradient → Schwarzschild switch) — the FD cross-check
  step in §3.3 is not optional.
- **Milestone 5 may not be sufficient on its own** for the 40000K seed (see the
  bracket-search-window nuance in §3.5) — budget for a companion numerics fix even after
  Saha is implemented.
- **Milestone 0's diagnostic is a first-order (pure-H, no He) estimate** — if it turns out
  ambiguous (e.g. modest but non-negligible ionization in the crash region), that ambiguity
  itself is a result worth reporting honestly rather than forcing a binary priority call.
- **If Milestones 1–4 stabilize T=13000K but the near-photosphere region remains fragile
  under further scope expansion** (e.g. approaching 2000K or 40000K in production, not just
  as isolated spot-checks), the next-level response is not a fifth patch — it is treating
  the thin optically-thin-transition layer as its own coupled sub-problem (atmosphere/interior
  hand-off), the way mature stellar-atmosphere codes often do, rather than one numerical
  domain spanning six decades of mass uniformly. Not scoped for implementation now; recorded
  here so it isn't rediscovered from scratch later.

---

## 6. Path to Production

**Milestone 6 (§3.6) reached the primary goal — a genuine, fully-converged `solve_bvp`
solution — at `T_CENTER_INITIAL=11500K`.** This section is no longer purely hypothetical;
given the one-week deadline, it is the realistic near-term sequence, not a someday-item.

**Immediate, before calling this thesis-ready:**
1. ~~Resolve or explicitly bound the two open items flagged in §3.6.3~~ **Partially done
   (§3.6.4, 2026-08-07): `L_surface`'s negative sign is now confirmed a generic feature of
   the converged family (reproduced at both 11500K and 12000K, same sign/order of
   magnitude), not a one-run fluke — characterized, not yet explained. The `α=1.0`
   instability mechanism is still un-derived (regularization remains a working hypothesis).**
2. ~~Re-run the same pipeline at a second temperature... to confirm this isn't a
   single-point success.~~ **Done (§3.6.4): T=12000K converges cleanly, `status=0`,
   residual $8.88\times10^{-16}$ — tighter than the original 11500K run. (T=2000K was
   attempted first and rejected as physically out-of-regime for the atomic EOS, not as a
   fix failure — see §3.6.4.)**
3. Full validation-suite pass against the converged state (physical sanity: monotonicity,
   positivity, mass conservation to the residual's own precision) — not yet done for this
   specific solution.

**Then, promotion**: `bvp_experiment.py`'s logic (scaled state, analytic Jacobians, the
`ALPHA_MAX` continuation) is promoted into the main pipeline, replacing `bvp_solver.py`'s
shooting-specific machinery (`relax_initial_state`, the LM/fsolve root-finding), and
`time_stepper.py` is re-pointed at it. `bvp_solver.py`'s shooting code is retired/archived
rather than deleted, both as a historical record of why this pivot happened and as a
fallback reference.

**Explicitly deferred past the one-week deadline, not forgotten**: Milestone 4's
rank-deficiency finding (100% convective saturation from the infinitely-efficient-convection
idealization) was deliberately *routed around* by this milestone, not fixed. `ALPHA_MAX`
and state-vector scaling make the current target converge; they are not proof the same
approach scales cleanly to 40000K or to the full time-evolution run, where the star may
spend much more of its profile in marginal-convection territory. A real mixing-length
treatment remains the mathematically complete fix, and stays on the roadmap (alongside
Sub-task 8a/Milestone 5) as explicitly out of scope for now, not resolved.
