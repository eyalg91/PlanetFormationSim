# output.py — Sub-task 10: snapshot I/O (.npz) and post-processing plots built ENTIRELY from
# saved snapshots, not a live run - the module's own exit criterion (PLAN.md) is that every
# plot below regenerates from disk without re-running the simulation. No physics of its own;
# pure I/O and matplotlib helpers (CLAUDE.md). Reuses diagnostics.py's existing single-state
# profile/convective-zone plotting functions for per-snapshot output rather than duplicating
# them - the genuinely new pieces here are .npz I/O and multi-snapshot evolution curves,
# neither of which existed before this sub-task.

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import config
import diagnostics
import eos
import gradients
import opacity
import state

SNAPSHOT_DIR = "snapshots"   # relative to wherever the run is launched from, matching diagnostics.PLOT_DIR's own convention

# ==========================================
# SECTION: Per-Run Output Directory Convention
# ==========================================

def run_output_dirs(run_name):
    """(snapshot_dir, plot_dir) for a named production run.

    HOUSEKEEPING 2026-08-10: named runs' plots now NEST under diagnostics.PLOT_DIR as
    diagnostic_plots/run_<run_name>/, instead of sitting as a sibling diagnostic_plots_
    <run_name>/ directory in the project root - the earlier overnight/10gyr runs did the
    latter and cluttered the root. Snapshot directories keep their existing flat
    snapshots_<run_name>/ convention unchanged (not part of this cleanup - a much smaller
    number of directories, one per run, not one PNG per snapshot/profile/check).
    """
    return f"snapshots_{run_name}", f"{diagnostics.PLOT_DIR}/run_{run_name}"


# ==========================================
# SECTION: Snapshot I/O (.npz)
# ==========================================

def _snapshot_path(output_dir, step):
    return f"{output_dir}/snapshot_{step:05d}.npz"


def save_snapshot(s, step, output_dir=SNAPSHOT_DIR) -> str:
    """Save one SimulationState as an .npz snapshot: the mass grid, every state field, the
    elapsed time, and the Schwarzschild is_convective mask (computed fresh here - not stored
    on SimulationState itself, matching diagnostics.plot_convective_zones' own convention).

    ASSUMPTION: is_convective is undefined at the exact center (grad_radiative's removable
    0/0 as m->0 - diagnostics.py's own docstring), so it is computed on m[1:] and the center
    point is recorded as True by convention rather than left undefined - every converged
    structure examined this project has the deep interior convective (PROGRESS.md's
    superadiabaticity histograms confirm this directly across every tested state).
    """
    os.makedirs(output_dir, exist_ok=True)
    kappa = opacity.bell_lin_opacity(s.rho[1:], s.T[1:])
    grad_ad = eos.grad_adiabatic(config.GAMMA)
    grad_rad = gradients.grad_radiative(s.L[1:], s.m[1:], s.P[1:], s.T[1:], kappa)
    _grad_eff, is_convective_outer = gradients.effective_gradient(grad_rad, grad_ad)
    is_convective = np.concatenate([[True], is_convective_outer])

    path = _snapshot_path(output_dir, step)
    np.savez(path, m=s.m, r=s.r, P=s.P, L=s.L, T=s.T, rho=s.rho, t=s.t,
             is_convective=is_convective)
    return path


def load_snapshot(path):
    """Load one .npz snapshot back into (SimulationState, is_convective)."""
    data = np.load(path)
    s = state.SimulationState(m=data["m"], r=data["r"], P=data["P"], L=data["L"],
                               T=data["T"], rho=data["rho"], t=float(data["t"]))
    return s, data["is_convective"]


def load_all_snapshots(output_dir=SNAPSHOT_DIR):
    """Every snapshot_*.npz in output_dir, sorted by step index (the zero-padded filename
    sorts correctly as a plain string, no need to parse the step number out)."""
    paths = sorted(glob.glob(f"{output_dir}/snapshot_*.npz"))
    return [load_snapshot(p) for p in paths]


# ==========================================
# SECTION: Evolution Curves (across snapshots, over simulated time)
# ==========================================

def plot_evolution_curves(snapshots, output_path=f"{diagnostics.PLOT_DIR}/evolution_curves.png") -> None:
    """r_surface(t), T_center(t), L_surface(t) across a sequence of (state, is_convective)
    snapshots - the actual Kelvin-Helmholtz contraction track, PLAN.md Sub-task 10's core
    deliverable, regenerable from disk alone."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    t_yr = np.array([s.t for s, _ in snapshots]) / config.SECONDS_PER_YEAR
    r_surf = np.array([s.r[-1] / config.R_JUPITER_CM for s, _ in snapshots])
    T_center = np.array([s.T[0] for s, _ in snapshots])
    L_surf = np.array([s.L[-1] / config.L_SUN_ERG_S for s, _ in snapshots])

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(t_yr, r_surf, marker=".")
    axes[0].axhline(config.R_HALT / config.R_JUPITER_CM, color="tab:red", linestyle="--",
                     linewidth=0.8, label="R_HALT")
    axes[0].set_ylabel("r_surface [R_Jup]")
    axes[0].set_title("Kelvin-Helmholtz contraction track")
    axes[0].legend(loc="best")

    axes[1].plot(t_yr, T_center, marker=".", color="tab:orange")
    axes[1].set_ylabel("T_center [K]")

    axes[2].plot(t_yr, L_surf, marker=".", color="tab:green")
    axes[2].axhline(0.0, color="k", linewidth=0.5)
    axes[2].set_ylabel("L_surface [L_sun]")
    axes[2].set_xlabel("t [yr]")
    # Log-scale the shared time axis only when there's a positive range to show on it - the
    # very first snapshot (t=0, the relaxed seed) would make log(0) undefined otherwise.
    if len(t_yr) > 1 and t_yr[1] > 0.0:
        axes[2].set_xscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"output.py: saved evolution curves ({len(snapshots)} snapshots) to {output_path}")


# ==========================================
# SECTION: Opacity Regime Map (single snapshot, colored by regime index)
# ==========================================

def plot_opacity_regime_map(s, output_path) -> None:
    """kappa(m) colored by the locally active Bell & Lin regime index, for one snapshot -
    PLAN.md Sub-task 10's explicit deliverable, distinct from diagnostics.py's existing
    convective-zone plot (Schwarzschild criterion) or opacity_profile_preview (a single
    synthetic profile, not a real converged snapshot)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    kappa = opacity.bell_lin_opacity(s.rho, s.T)
    regime_idx = opacity.determine_regime(s.rho, s.T)
    x = s.m / config.M_TOTAL

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(x, kappa, c=regime_idx, cmap="tab10", vmin=-0.5, vmax=7.5, s=10)
    ax.set_yscale("log")
    ax.set_xlabel("m / M_TOTAL")
    ax.set_ylabel("kappa [cm^2 g^-1]")
    ax.set_title(f"Opacity regime map at t={s.t / config.SECONDS_PER_YEAR:.3e} yr")
    cbar = fig.colorbar(sc, ax=ax, ticks=range(len(opacity.REGIMES)))
    cbar.ax.set_yticklabels([r.name for r in opacity.REGIMES])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"output.py: saved opacity regime map to {output_path}")


# ==========================================
# SECTION: Full Post-Processing Pipeline (from disk, no re-solve)
# ==========================================

def generate_all_plots(snapshot_dir=SNAPSHOT_DIR, output_dir=diagnostics.PLOT_DIR,
                        profile_snapshot_indices=None) -> None:
    """Regenerate every Sub-task 10 plot from saved .npz snapshots alone - the module's own
    exit criterion. profile_snapshot_indices selects which snapshots (by index into the
    time-sorted list) get a full structure/convective-zone/opacity-regime plot - defaults to
    first, middle, and last."""
    snapshots = load_all_snapshots(snapshot_dir)
    if not snapshots:
        raise FileNotFoundError(f"output.py: no snapshot_*.npz files found in {snapshot_dir}")

    # ASSUMPTION: created here, once, up front - not just relying on each individual plot
    # function's own defensive os.makedirs (present, but diagnostics.py's plot_structure_
    # profile/plot_convective_zones, called below, predate this module and don't create
    # their own output directory - found 2026-08-09 night when a non-default output_dir
    # crashed the very first savefig call, mid-run, after a full 77-step solve had already
    # completed and cost real compute - not repeating that mistake here.
    os.makedirs(output_dir, exist_ok=True)

    plot_evolution_curves(snapshots, f"{output_dir}/evolution_curves.png")

    if profile_snapshot_indices is None:
        n = len(snapshots)
        profile_snapshot_indices = sorted(set([0, n // 2, n - 1]))

    for i in profile_snapshot_indices:
        s, _is_convective = snapshots[i]
        t_yr = s.t / config.SECONDS_PER_YEAR
        diagnostics.plot_structure_profile(s, f"{output_dir}/profile_t{t_yr:.3e}yr.png")
        diagnostics.plot_convective_zones(s, f"{output_dir}/convective_t{t_yr:.3e}yr.png")
        plot_opacity_regime_map(s, f"{output_dir}/opacity_regime_t{t_yr:.3e}yr.png")

    print(f"output.py: generated all plots from {len(snapshots)} snapshots in {snapshot_dir}")
