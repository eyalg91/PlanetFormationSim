# generate_paper_plots.py — 2026-08-14 (revised again). SINGLE SOURCE OF TRUTH for every
# figure in the Phase 1 paper: the "standard" per-run plots (evolution curves, structure
# profiles, luminosity profile, opacity regime maps, convective zones, photosphere zoom,
# pseudo-HR diagram - previously only produced piecemeal by output.py, in non-vector PNG,
# without academic formatting) PLUS the advanced/analysis figures (2D cross-sections, seed vs.
# relaxed, solver convergence, resolution convergence, contraction scaling, opacity smoothing).
# 13 figures total, all from this one script.
#
# FORMATTING: vector output only (FORMAT below, "svg" for direct Word embedding), large fonts
# throughout (rcParams block), low-alpha grids, CGS/astrophysical reporting units (R_Jup, K,
# L_sun) matching config.py's own R_JUPITER_CM/L_SUN_ERG_S convention. No text.usetex (a real
# LaTeX install is a fragile, easy-to-break dependency) - all math (subscripts, nabla, odot)
# via matplotlib's built-in mathtext, which needs nothing beyond matplotlib itself.

import contextlib
import io
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import bvp_solver
import config
import eos
import gradients
import opacity
import output

# ==========================================
# SECTION: Global Formatting
# ==========================================

FORMAT = "svg"   # vector output for direct, scalable Word embedding
OUTPUT_DIR = "outputs/paper_plots"
HIRES_RUN_DIR = "outputs/snapshots/Phase1_high_res_overnight_20260813"
BASELINE_RUN_DIR = "outputs/snapshots/Phase1_baseline_rerun_20260813"

# Shared across every "N representative snapshots" figure (structure_profiles, luminosity_
# profile, opacity_regime_maps, convective_zones, photosphere_zoom) so "the same 3 snapshots"
# holds exactly, not just approximately, across all five.
STORY_SNAPSHOT_INDICES = (0, 20, 43)   # start (t=0), mid-run (T_c~1035K), 1900K halt

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})


def _savefig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/{name}.{FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    print(f"generate_paper_plots: saved {path}")


def _set_phase1_config():
    """Runtime overrides matching run_phase1_high_res_overnight.py exactly (hermetic - never
    touches config.py's own persisted file). Needed by any plot that recomputes physics live
    from raw snapshot arrays (grad_ad, grad_rad, opacity), so the recomputation matches the
    physics that actually produced the snapshot, and by plot_seed_vs_relaxed's fresh
    solve_static_structure call."""
    config.MU = 2.34
    config.GAMMA = 1.4
    config.USE_H2_RECOMBINATION_PHYSICS = False
    config.T_CENTER_INITIAL = 645.0
    config.N_GRID_POINTS = 1000
    config.GRID_OUTER_REFINEMENT = 1.0e-6


# ==========================================
# SECTION: Plot 1 — Evolution Curves (the main story)
# ==========================================

def plot_evolution_curves():
    """R_surface, T_center, L_surface vs. t across the ENTIRE Phase 1 run - the single most
    important plot: the whole Kelvin-Helmholtz contraction track in one figure. Academically
    formatted re-implementation of output.plot_evolution_curves's content (that function stays
    in output.py for its own non-paper purpose - generating quick-look PNGs straight from a
    live run - this is the publication version).
    """
    snaps = output.load_all_snapshots(HIRES_RUN_DIR)
    t_yr = np.array([s.t for s, _ in snaps]) / config.SECONDS_PER_YEAR
    r_surf = np.array([s.r[-1] / config.R_JUPITER_CM for s, _ in snaps])
    T_center = np.array([s.T[0] for s, _ in snaps])
    L_surf = np.array([s.L[-1] / config.L_SUN_ERG_S for s, _ in snaps])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(t_yr, r_surf, color="tab:blue", marker=".", markersize=5)
    axes[0].axhline(config.R_HALT / config.R_JUPITER_CM, color="tab:red", linestyle="--",
                     linewidth=1.0, label=r"$R_{\rm HALT}$")
    axes[0].set_ylabel(r"$R_{\rm surface}$ [$R_{\rm Jup}$]")
    axes[0].set_title("Phase 1 Evolutionary Track")
    axes[0].legend()

    axes[1].plot(t_yr, T_center, color="tab:orange", marker=".", markersize=5)
    axes[1].axhline(config.PHASE1_T_CENTER_HALT, color="tab:red", linestyle="--",
                     linewidth=1.0, label="1900 K halt")
    axes[1].set_ylabel(r"$T_{\rm center}$ [K]")
    axes[1].legend()

    axes[2].plot(t_yr, L_surf, color="tab:green", marker=".", markersize=5)
    axes[2].axhline(0.0, color="k", linewidth=0.5)
    axes[2].set_ylabel(r"$L_{\rm surface}$ [$L_\odot$]")
    axes[2].set_xlabel(r"$t$ [yr]")

    fig.tight_layout()
    _savefig(fig, "01_evolution_curves")


# ==========================================
# SECTION: Plot 2 — Pseudo-HR Diagram
# ==========================================

def plot_pseudo_hr_diagram():
    """Evolutionary track: L_surface vs. T_center (left, inverted x-axis - conventional
    'hotter to the left' HR sense) and vs. R_surface (right, not inverted - no equivalent
    convention exists for radius). Track colored by elapsed time; start and the
    PHASE1_T_CENTER_HALT (1900K) endpoint both marked explicitly. Companion to plot_evolution_
    curves - same underlying data, cross-plotted to show the L-T_center/R relationship
    directly instead of both against time separately.
    """
    snaps = output.load_all_snapshots(HIRES_RUN_DIR)
    t_yr = np.array([s.t for s, _ in snaps]) / config.SECONDS_PER_YEAR
    T_center = np.array([s.T[0] for s, _ in snaps])
    r_surf = np.array([s.r[-1] / config.R_JUPITER_CM for s, _ in snaps])
    L_surf = np.array([s.L[-1] / config.L_SUN_ERG_S for s, _ in snaps])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, x, xlabel in ((axes[0], T_center, r"$T_{\rm center}$ [K]"),
                          (axes[1], r_surf, r"$R_{\rm surface}$ [$R_{\rm Jup}$]")):
        sc = ax.scatter(x, L_surf, c=t_yr, cmap="viridis", s=28, zorder=3)
        ax.plot(x, L_surf, color="gray", alpha=0.4, linewidth=1.0, zorder=2)
        ax.scatter(x[0], L_surf[0], marker="*", s=260, color="tab:orange",
                   edgecolor="k", zorder=4, label="Start ($t=0$)")
        ax.scatter(x[-1], L_surf[-1], marker="X", s=180, color="tab:red",
                   edgecolor="k", zorder=4, label="1900 K halt")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$L_{\rm surface}$ [$L_\odot$]")
        ax.legend(loc="best")

    axes[0].invert_xaxis()
    fig.colorbar(sc, ax=axes, label="$t$ [yr]", pad=0.02, fraction=0.046)
    fig.suptitle("Pseudo-HR Diagram: Phase 1 Kelvin-Helmholtz Contraction Track")

    _savefig(fig, "02_pseudo_hr_diagram")


# ==========================================
# SECTION: Plot 3 — Structure Profiles
# ==========================================

def plot_structure_profiles(snapshot_indices=STORY_SNAPSHOT_INDICES):
    """T, rho, P vs. m/M_total at 3 representative snapshots (start, mid-run, 1900K halt)."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshot_indices)))

    for idx, color in zip(snapshot_indices, colors):
        s, _ = output.load_snapshot(f"{HIRES_RUN_DIR}/snapshot_{idx:05d}.npz")
        t_yr = s.t / config.SECONDS_PER_YEAR
        x = s.m / config.M_TOTAL
        label = fr"$t={t_yr:.0f}$ yr, $T_c={s.T[0]:.0f}$ K"
        axes[0].plot(x, s.T, color=color, label=label)
        axes[1].plot(x, s.rho, color=color)
        axes[2].plot(x, s.P, color=color)

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$T$ [K]")
    axes[0].set_title("Structure Profiles")
    axes[0].legend()

    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\rho$ [g cm$^{-3}$]")

    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$P$ [dyn cm$^{-2}$]")
    axes[2].set_xlabel(r"$m / M_{\rm total}$")

    fig.tight_layout()
    _savefig(fig, "03_structure_profiles")


# ==========================================
# SECTION: Plot 4 — Luminosity Profile
# ==========================================

def plot_luminosity_profile(snapshot_indices=STORY_SNAPSHOT_INDICES):
    """L(m) at the SAME 3 representative snapshots as plot_structure_profiles - companion
    plot, L=0 at the center (inner boundary condition) and the rise toward the surface.
    Symlog y-axis: L crosses (very slightly) negative near the center at machine precision
    (the innermost grid point sits at m_min=1e-6*M_total, not exactly m=0 - a known, benign
    feature, not a bug), while also spanning ~10 decades up to the surface value - neither a
    pure linear nor pure log axis shows both regimes at once.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshot_indices)))

    for idx, color in zip(snapshot_indices, colors):
        s, _ = output.load_snapshot(f"{HIRES_RUN_DIR}/snapshot_{idx:05d}.npz")
        t_yr = s.t / config.SECONDS_PER_YEAR
        ax.plot(s.m / config.M_TOTAL, s.L, color=color,
                label=fr"$t={t_yr:.0f}$ yr, $T_c={s.T[0]:.0f}$ K")

    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=1.0e4)
    ax.set_xlabel(r"$m / M_{\rm total}$")
    ax.set_ylabel(r"$L$ [erg s$^{-1}$]")
    ax.set_title("Luminosity Profile Across the Contraction")
    ax.legend()

    fig.tight_layout()
    _savefig(fig, "04_luminosity_profile")


# ==========================================
# SECTION: Plot 5 — Opacity Regime Maps
# ==========================================

def plot_opacity_regime_maps(snapshot_indices=STORY_SNAPSHOT_INDICES):
    """kappa(m) colored by the locally active Bell & Lin (1994) regime, for the SAME 3
    snapshots as plot_structure_profiles/plot_convective_zones. One shared colorbar (the
    regime index -> name mapping is identical across all three panels, unlike a continuous
    field's own min/max)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    sc = None
    for ax, idx in zip(axes, snapshot_indices):
        s, _ = output.load_snapshot(f"{HIRES_RUN_DIR}/snapshot_{idx:05d}.npz")
        kappa = opacity.bell_lin_opacity(s.rho, s.T)
        regime_idx = opacity.determine_regime(s.rho, s.T)
        x = s.m / config.M_TOTAL
        sc = ax.scatter(x, kappa, c=regime_idx, cmap="tab10", vmin=-0.5, vmax=7.5, s=10)
        ax.set_yscale("log")
        ax.set_xlabel(r"$m / M_{\rm total}$")
        t_yr = s.t / config.SECONDS_PER_YEAR
        ax.set_title(fr"$t={t_yr:.0f}$ yr, $T_c={s.T[0]:.0f}$ K")

    axes[0].set_ylabel(r"$\kappa$ [cm$^2$ g$^{-1}$]")
    cbar = fig.colorbar(sc, ax=list(axes), ticks=range(len(opacity.REGIMES)), pad=0.015, fraction=0.025)
    cbar.ax.set_yticklabels([r.name for r in opacity.REGIMES])
    fig.suptitle("Opacity Regime Maps")
    _savefig(fig, "05_opacity_regime_maps")


# ==========================================
# SECTION: Plot 6 — Convective vs. Radiative Zones
# ==========================================

def plot_convective_zones(snapshot_indices=STORY_SNAPSHOT_INDICES):
    """nabla_rad vs. nabla_ad with shaded convective zones, for the SAME 3 snapshots as
    plot_structure_profiles/plot_opacity_regime_maps. Extended from a single-snapshot (halt-
    only) version: seeing the SAME near-global convective saturation already present at t=0
    (not just at the halt) is itself part of the numerical-methods story - the alpha=0->0.5
    Jacobian degeneracy this motivates was never a late-time-only phenomenon.
    """
    _set_phase1_config()
    grad_ad = eos.grad_adiabatic(config.GAMMA)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    for ax, idx in zip(axes, snapshot_indices):
        s, _ = output.load_snapshot(f"{HIRES_RUN_DIR}/snapshot_{idx:05d}.npz")
        kappa = opacity.bell_lin_opacity(s.rho[1:], s.T[1:])
        grad_rad = gradients.grad_radiative(s.L[1:], s.m[1:], s.P[1:], s.T[1:], kappa)
        _grad_eff, is_convective = gradients.effective_gradient(grad_rad, grad_ad)
        x = s.m[1:] / config.M_TOTAL

        ax.plot(x, grad_rad, color="tab:blue", label=r"$\nabla_{\rm rad}$")
        ax.axhline(grad_ad, color="k", linestyle="--", linewidth=1.2, label=r"$\nabla_{\rm ad}$")
        ax.fill_between(x, ax.get_ylim()[0], ax.get_ylim()[1], where=is_convective,
                         color="tab:blue", alpha=0.12, label="Convective")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\nabla$")
        t_yr = s.t / config.SECONDS_PER_YEAR
        ax.set_title(fr"$t={t_yr:.0f}$ yr, $T_c={s.T[0]:.0f}$ K")
        ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel(r"$m / M_{\rm total}$")
    fig.suptitle("Convective/Radiative Structure")
    fig.tight_layout()
    _savefig(fig, "06_convective_zones")


# ==========================================
# SECTION: Plot 7 — Photosphere Zoom-In
# ==========================================

def plot_photosphere_zoom(snapshot_indices=STORY_SNAPSHOT_INDICES, mass_fraction_max=1.0e-2):
    """T and nabla_rad vs log10(1-m/M_total), zoomed to the outer 1% of mass - where the
    steepest structure AND the historically fragile numerics (opacity regime transitions, the
    Schwarzschild-switch smoothing) both live. nabla_rad is recomputed live from each
    snapshot's own (rho,T,P,L,m). nabla_ad is overlaid as a reference line, showing the
    Schwarzschild criterion directly in the zoomed region. The exact surface point
    (1-m/M=0 exactly) is excluded - log10(0) is undefined.
    """
    _set_phase1_config()
    grad_ad = eos.grad_adiabatic(config.GAMMA)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshot_indices)))

    for idx, color in zip(snapshot_indices, colors):
        s, _ = output.load_snapshot(f"{HIRES_RUN_DIR}/snapshot_{idx:05d}.npz")
        frac = 1.0 - s.m / config.M_TOTAL
        mask = (frac > 0.0) & (frac <= mass_fraction_max)

        kappa = opacity.bell_lin_opacity(s.rho[mask], s.T[mask])
        grad_rad = gradients.grad_radiative(s.L[mask], s.m[mask], s.P[mask], s.T[mask], kappa)

        t_yr = s.t / config.SECONDS_PER_YEAR
        log_frac = np.log10(frac[mask])
        axes[0].plot(log_frac, s.T[mask], color=color, label=fr"$t={t_yr:.0f}$ yr, $T_c={s.T[0]:.0f}$ K")
        axes[1].plot(log_frac, grad_rad, color=color)

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$T$ [K]")
    axes[0].set_title(r"Photosphere Zoom: Outer 1% of Mass")
    axes[0].legend()

    axes[1].axhline(grad_ad, color="k", linestyle="--", linewidth=1.2, label=r"$\nabla_{\rm ad}$ (constant)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\nabla_{\rm rad}$")
    axes[1].set_xlabel(r"$\log_{10}(1 - m/M_{\rm total})$")
    axes[1].legend()

    fig.tight_layout()
    _savefig(fig, "07_photosphere_zoom")


# ==========================================
# SECTION: Plot 8 — 2D Radial Cross-Sections (flagship visual)
# ==========================================

def _radial_field_2d(r_profile, field_profile, R_max, n_pixels=320):
    """Projects a 1D radial profile field(r) into a 2D circular cross-section - azimuthally
    symmetric by construction, EXACT for this spherically symmetric 1D model. Pixel value =
    field interpolated at that pixel's own distance from center; masked to NaN (rendered fully
    transparent) outside the star's own r_profile[-1] (=R_surface). R_max is shared across
    every panel in the figure - see the caller for why that sharing is the whole point.
    """
    axis = np.linspace(-R_max, R_max, n_pixels)
    X, Y = np.meshgrid(axis, axis)
    R = np.sqrt(X**2 + Y**2)
    field_2d = np.interp(R, r_profile, field_profile, left=field_profile[0], right=np.nan)
    field_2d[R > r_profile[-1]] = np.nan
    return field_2d


def plot_radial_cross_sections(time_fractions=(0.0, 0.25, 0.75, 1.0)):
    """2D circular cross-section 'cutaway' visualization at 4 evolutionary snapshots (nearest
    available snapshot to each fraction of the TOTAL ELAPSED SIMULATED TIME - the adaptive dt
    means snapshots are not evenly time-spaced, so a step-index fraction would not correspond
    to the requested time fractions).

    Columns = time, rows = field (T, rho, kappa - all log-normed). Every panel in a row shares
    the SAME colorbar limits (global min/max across the 4 selected snapshots, within the
    stellar body only). Every panel in the FIGURE shares the SAME spatial extent (R_max = the
    largest R_surface among the 4, i.e. t=0's) so a panel's on-page circle size directly
    encodes its actual physical R_surface(t) - the contraction is visible by eye.
    """
    _set_phase1_config()
    snaps = output.load_all_snapshots(HIRES_RUN_DIR)
    t_arr = np.array([s.t for s, _ in snaps])
    t_final = t_arr[-1]
    indices = [int(np.argmin(np.abs(t_arr - f * t_final))) for f in time_fractions]
    states = [snaps[i][0] for i in indices]
    R_max = max(s.r[-1] for s in states) / config.R_JUPITER_CM

    per_state_fields = []
    for s in states:
        kappa = opacity.bell_lin_opacity(s.rho, s.T)
        per_state_fields.append({"T": s.T, "rho": s.rho, "kappa": kappa})

    row_specs = [("T", r"$T$ [K]", "inferno"),
                 ("rho", r"$\rho$ [g cm$^{-3}$]", "viridis"),
                 ("kappa", r"$\kappa$ [cm$^2$ g$^{-1}$]", "magma")]

    fig, axes = plt.subplots(3, 4, figsize=(16, 13))

    for row_idx, (key, cbar_label, cmap_name) in enumerate(row_specs):
        values_all = np.concatenate([f[key] for f in per_state_fields])
        norm = LogNorm(vmin=values_all.min(), vmax=values_all.max())
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(alpha=0.0)   # outside the star: fully transparent, not a solid color

        im = None
        for col_idx, (s, f) in enumerate(zip(states, per_state_fields)):
            ax = axes[row_idx, col_idx]
            r_profile = s.r / config.R_JUPITER_CM
            field_2d = _radial_field_2d(r_profile, f[key], R_max)
            im = ax.imshow(field_2d, extent=[-R_max, R_max, -R_max, R_max], origin="lower",
                            cmap=cmap, norm=norm)
            ax.add_patch(plt.Circle((0, 0), r_profile[-1], fill=False, color="k", linewidth=0.8))
            ax.grid(False)   # a numeric grid isn't meaningful over a radial cutaway image

            if row_idx == 0:
                t_yr = s.t / config.SECONDS_PER_YEAR
                ax.set_title(fr"$t={t_yr:.0f}$ yr" + "\n" + fr"$R={r_profile[-1]:.0f}\,R_{{\rm Jup}}$",
                             fontsize=13)
            ax.set_xlim(-R_max, R_max)
            ax.set_ylim(-R_max, R_max)
            ax.set_aspect("equal")
            if row_idx < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$R_{\rm Jup}$")
            if col_idx > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r"$R_{\rm Jup}$")

        fig.colorbar(im, ax=list(axes[row_idx, :]), label=cbar_label, fraction=0.025, pad=0.015)

    fig.suptitle("2D Radial Cross-Sections: Phase 1 Kelvin-Helmholtz Contraction", fontsize=18)
    _savefig(fig, "08_radial_cross_sections")


# ==========================================
# SECTION: Plot 9 — Static (Pure-Adiabat) Seed vs. Relaxed t=0 State
# ==========================================

def plot_seed_vs_relaxed():
    """Compares solve_static_structure's pure-adiabat construction against the same state
    after relax_initial_state (a genuine solution of the full 4-ODE system). The relaxed side
    is the high-res run's own snapshot_00000.npz (literally state_relaxed, not a re-derivation);
    the static seed was never saved to disk, so it is regenerated live under the identical
    Phase 1 configuration (cheap - a 3-ODE shooting solve, not the full BVP).

    Panel 4 (relative deviation): the raw T/P/rho profiles overlap almost exactly through the
    interior (expected - both constructions are near-adiabatic there), so this panel is what
    actually shows WHERE relaxation did its work - concentrated in the outer ~10% of mass.
    """
    _set_phase1_config()
    state_static = bvp_solver.solve_static_structure(use_ideal_gas_seed=True)
    state_relaxed, _ = output.load_snapshot(f"{HIRES_RUN_DIR}/snapshot_00000.npz")

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    x_static = state_static.m / config.M_TOTAL
    x_relaxed = state_relaxed.m / config.M_TOTAL

    axes[0].plot(x_static, state_static.T, color="tab:red", linestyle="--", label="Static adiabatic seed")
    axes[0].plot(x_relaxed, state_relaxed.T, color="tab:blue", label="Relaxed (self-consistent)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$T$ [K]")
    axes[0].legend()
    axes[0].set_title(r"$t=0$ Initial Condition: Static Seed vs. Relaxed State")

    axes[1].plot(x_static, state_static.P, color="tab:red", linestyle="--")
    axes[1].plot(x_relaxed, state_relaxed.P, color="tab:blue")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$P$ [dyn cm$^{-2}$]")

    axes[2].plot(x_static, state_static.rho, color="tab:red", linestyle="--")
    axes[2].plot(x_relaxed, state_relaxed.rho, color="tab:blue")
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$\rho$ [g cm$^{-3}$]")

    T_static_interp = np.interp(x_relaxed, x_static, state_static.T)
    P_static_interp = np.interp(x_relaxed, x_static, state_static.P)
    rho_static_interp = np.interp(x_relaxed, x_static, state_static.rho)
    axes[3].plot(x_relaxed, (state_relaxed.T - T_static_interp) / T_static_interp * 100.0,
                 label=r"$T$", color="tab:orange")
    axes[3].plot(x_relaxed, (state_relaxed.P - P_static_interp) / P_static_interp * 100.0,
                 label=r"$P$", color="tab:green")
    axes[3].plot(x_relaxed, (state_relaxed.rho - rho_static_interp) / rho_static_interp * 100.0,
                 label=r"$\rho$", color="tab:purple")
    axes[3].axhline(0.0, color="k", linewidth=0.8)
    axes[3].set_ylabel("Relaxed / Seed $-$ 1 [%]")
    axes[3].set_xlabel(r"$m / M_{\rm total}$")
    axes[3].legend()

    fig.tight_layout()
    _savefig(fig, "09_seed_vs_relaxed")


# ==========================================
# SECTION: Plot 10 — Solver (Newton/Collocation) Convergence
# ==========================================

def plot_solver_convergence(snapshot_index=5):
    """Residual and mesh-node growth vs. Newton/collocation iteration for a REAL,
    representative converged step - not a synthetic demo. Re-runs the exact (state, dt) pair
    that already succeeded at snapshot_index -> snapshot_index+1 in the high-res run (an
    early, numerically easy step, chosen so the direct alpha=1 attempt converges in a single
    table with no alpha-continuation fallback - avoids the ambiguity of multiple tables).

    scipy.integrate.solve_bvp's own verbose=2 per-iteration table (Iteration/Max residual/Max
    BC residual/Total nodes) is captured by redirecting stdout during the call and parsed with
    a regex - this is scipy's OWN internal accounting, not a re-derivation of it.
    """
    _set_phase1_config()
    s_prev, _ = output.load_snapshot(f"{HIRES_RUN_DIR}/snapshot_{snapshot_index:05d}.npz")
    s_next, _ = output.load_snapshot(f"{HIRES_RUN_DIR}/snapshot_{snapshot_index + 1:05d}.npz")
    dt = s_next.t - s_prev.t

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bvp_solver.solve_timestep(s_prev, dt)
    captured = buf.getvalue()

    row_pattern = re.compile(r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\d+)\s+", re.MULTILINE)
    matches = row_pattern.findall(captured)
    if not matches:
        raise RuntimeError(
            "plot_solver_convergence: could not parse any iteration rows from solve_bvp's "
            "captured verbose output - check this regex against the installed scipy version's "
            "print format before trusting this plot"
        )

    iterations, residuals, bc_residuals, nodes = [], [], [], []
    for it, res, bc_res, n in matches:
        it = int(it)
        if iterations and it <= iterations[-1]:
            break   # a second table (alpha-continuation) started - keep only the first
        iterations.append(it)
        residuals.append(float(res))
        bc_residuals.append(float(bc_res))
        nodes.append(int(n))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(iterations, residuals, marker="o", label="Max collocation residual")
    axes[0].plot(iterations, bc_residuals, marker="s", label="Max BC residual")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(fr"BVP Collocation Convergence (real step, $T_c: {s_prev.T[0]:.0f}\to{s_next.T[0]:.0f}$ K)")
    axes[0].legend()

    axes[1].plot(iterations, nodes, marker="o", color="tab:green")
    axes[1].set_ylabel("Total mesh nodes")
    axes[1].set_xlabel("Newton / collocation iteration")
    axes[1].set_xticks(iterations)

    fig.tight_layout()
    _savefig(fig, "10_solver_convergence")


# ==========================================
# SECTION: Plot 11 — Grid-Resolution Convergence
# ==========================================

def plot_resolution_convergence(target_T_center=1000.0):
    """Overlays the baseline run (N_GRID_POINTS=200, default solver mesh/tolerance) and the
    high-resolution run (N_GRID_POINTS=1000 output grid; solver-internal knobs deliberately
    left at the SAME proven defaults - PROGRESS.md 2026-08-14 has the full account of why) at
    the closest available snapshot to the SAME T_center in each run.

    Matched by T_center, NOT by elapsed simulated time t: checked directly (not assumed) that
    matching by t confounds resolution with trajectory-timing - the two runs use different
    ADAPTIVE_DT_GROWTH_FACTOR (1.3 vs. 1.15), and at a fixed t their T_center differs enough
    that "matching by t" was actually comparing two genuinely different evolutionary stages
    (~22% offset across the ENTIRE interior profile, not concentrated near the photosphere the
    way a real resolution effect should be). At matched T_center~1000K, r_surface agrees to
    ~2.8% (377 vs. 388 R_Jup) instead - a much fairer isolation of the resolution effect.
    """
    def _closest_snapshot(directory, target_T):
        snaps = output.load_all_snapshots(directory)
        T_arr = np.array([s.T[0] for s, _ in snaps])
        idx = int(np.argmin(np.abs(T_arr - target_T)))
        return snaps[idx][0]

    s_base = _closest_snapshot(BASELINE_RUN_DIR, target_T_center)
    s_hires = _closest_snapshot(HIRES_RUN_DIR, target_T_center)

    x_base = s_base.m / config.M_TOTAL
    x_hires = s_hires.m / config.M_TOTAL

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(x_base, s_base.T, color="tab:red", linestyle="--", linewidth=2.2,
                 label=fr"Baseline (200 pts), $T_c={s_base.T[0]:.0f}$ K, $t={s_base.t / config.SECONDS_PER_YEAR:.0f}$ yr")
    axes[0].plot(x_hires, s_hires.T, color="tab:blue", linewidth=1.4,
                 label=fr"High-res (1000 pts), $T_c={s_hires.T[0]:.0f}$ K, $t={s_hires.t / config.SECONDS_PER_YEAR:.0f}$ yr")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$T$ [K]")
    axes[0].set_title(r"Grid-Resolution Convergence (matched by $T_{\rm center}$)")
    axes[0].legend()

    T_base_interp = np.interp(x_hires, x_base, s_base.T)
    axes[1].plot(x_hires, (s_hires.T - T_base_interp) / T_base_interp * 100.0, color="tab:purple")
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    axes[1].set_ylabel("High-res / Baseline $-$ 1 [%]")
    axes[1].set_xlabel(r"$m / M_{\rm total}$")

    fig.tight_layout()
    _savefig(fig, "11_resolution_convergence")


# ==========================================
# SECTION: Plot 12 — Contraction Scaling / Power Law
# ==========================================

def plot_contraction_power_law():
    """R_surface(t) on log-log axes across the full run, with a power-law fit (R ~ t^alpha,
    linear regression in log-log space) overlaid. t=0 is excluded (log undefined). Reports the
    fit honestly (residual scatter annotated) rather than assuming a clean single power law
    holds over the whole run.
    """
    snaps = output.load_all_snapshots(HIRES_RUN_DIR)
    t_yr = np.array([s.t for s, _ in snaps]) / config.SECONDS_PER_YEAR
    r_surf = np.array([s.r[-1] / config.R_JUPITER_CM for s, _ in snaps])

    mask = t_yr > 0.0
    log_t = np.log10(t_yr[mask])
    log_r = np.log10(r_surf[mask])
    slope, intercept = np.polyfit(log_t, log_r, 1)
    fit_r = 10.0 ** intercept * t_yr[mask] ** slope
    residual_scatter = np.std(log_r - (slope * log_t + intercept))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_yr[mask], r_surf[mask], "o", color="tab:blue", markersize=5, label="Simulation")
    ax.plot(t_yr[mask], fit_r, color="tab:red", linestyle="--",
            label=fr"Power-law fit: $R \propto t^{{{slope:.3f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ [yr]")
    ax.set_ylabel(r"$R_{\rm surface}$ [$R_{\rm Jup}$]")
    ax.set_title("Contraction Scaling")
    ax.legend()
    ax.text(0.03, 0.05, fr"log-log fit scatter (std): {residual_scatter:.3f} dex",
            transform=ax.transAxes, fontsize=11)

    fig.tight_layout()
    _savefig(fig, "12_contraction_power_law")


# ==========================================
# SECTION: Plot 13 — Opacity Smoothing (Check 39)
# ==========================================

def plot_opacity_hard_vs_smooth():
    """Reproduces validation.py's Check 39 exactly (same density, same T sweep, same Metal
    grain evaporation window) with publication formatting - fully self-contained, no cached
    state needed, composition-independent. kappa(T) looks smooth in BOTH cases (continuous by
    construction either way); the derivative panel is where the fix is actually visible: a
    real jump for the hard switch, a smooth curve for the blend.
    """
    rho_fixed = 4.0e-3
    T = np.linspace(1600.0, 2100.0, 2000)
    rho = np.full_like(T, rho_fixed)

    smooth_orig = config.OPACITY_SMOOTH_TRANSITIONS
    config.OPACITY_SMOOTH_TRANSITIONS = True
    kappa_smooth = opacity.bell_lin_opacity(rho, T)
    config.OPACITY_SMOOTH_TRANSITIONS = False
    kappa_hard = opacity.bell_lin_opacity(rho, T)
    config.OPACITY_SMOOTH_TRANSITIONS = smooth_orig

    dkappa_dT_hard = np.gradient(kappa_hard, T)
    dkappa_dT_smooth = np.gradient(kappa_smooth, T)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(T, kappa_hard, color="tab:red", alpha=0.75, label="Hard switch")
    axes[0].plot(T, kappa_smooth, color="tab:blue", linestyle="--", label="Smoothed (production)")
    axes[0].axvspan(1820.6, 1896.6, color="gray", alpha=0.15, label="Metal grain evaporation")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\kappa$ [cm$^2$ g$^{-1}$]")
    axes[0].set_title(fr"Bell & Lin (1994) Opacity, Hard vs. Smoothed ($\rho={rho_fixed:.1e}$ g cm$^{{-3}}$)")
    axes[0].legend()

    axes[1].plot(T, dkappa_dT_hard, color="tab:red", alpha=0.75, label="Hard switch")
    axes[1].plot(T, dkappa_dT_smooth, color="tab:blue", linestyle="--", label="Smoothed (production)")
    axes[1].axvspan(1820.6, 1896.6, color="gray", alpha=0.15)
    axes[1].set_ylabel(r"$d\kappa/dT$ [cm$^2$ g$^{-1}$ K$^{-1}$]")
    axes[1].set_xlabel(r"$T$ [K]")
    axes[1].legend()

    fig.tight_layout()
    _savefig(fig, "13_opacity_hard_vs_smooth")


# ==========================================
# SECTION: Main
# ==========================================

def main():
    print("generate_paper_plots: 1/13 evolution curves ...", flush=True)
    plot_evolution_curves()
    print("generate_paper_plots: 2/13 pseudo-HR diagram ...", flush=True)
    plot_pseudo_hr_diagram()
    print("generate_paper_plots: 3/13 structure profiles ...", flush=True)
    plot_structure_profiles()
    print("generate_paper_plots: 4/13 luminosity profile ...", flush=True)
    plot_luminosity_profile()
    print("generate_paper_plots: 5/13 opacity regime maps ...", flush=True)
    plot_opacity_regime_maps()
    print("generate_paper_plots: 6/13 convective/radiative zones ...", flush=True)
    plot_convective_zones()
    print("generate_paper_plots: 7/13 photosphere zoom-in ...", flush=True)
    plot_photosphere_zoom()
    print("generate_paper_plots: 8/13 2D radial cross-sections ...", flush=True)
    plot_radial_cross_sections()
    print("generate_paper_plots: 9/13 static seed vs. relaxed state ...", flush=True)
    plot_seed_vs_relaxed()
    print("generate_paper_plots: 10/13 solver convergence ...", flush=True)
    plot_solver_convergence()
    print("generate_paper_plots: 11/13 resolution convergence ...", flush=True)
    plot_resolution_convergence()
    print("generate_paper_plots: 12/13 contraction power law ...", flush=True)
    plot_contraction_power_law()
    print("generate_paper_plots: 13/13 opacity hard vs. smooth (Check 39) ...", flush=True)
    plot_opacity_hard_vs_smooth()
    print(f"generate_paper_plots: done - all figures in {OUTPUT_DIR}/*.{FORMAT}", flush=True)


if __name__ == "__main__":
    main()
