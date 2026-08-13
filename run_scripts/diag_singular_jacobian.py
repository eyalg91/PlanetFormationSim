# diag_singular_jacobian.py — Instrumentation for the ~1620K "Singular Jacobian encountered
# on iteration 1" wall (PROGRESS.md has the full report). Does NOT reimplement or approximate
# scipy's collocation Jacobian assembly - monkey-patches scipy.integrate._bvp.prepare_sys to
# INTERCEPT the exact sparse global Jacobian matrix (csc_matrix, shape (4*n_mesh, 4*n_mesh))
# solve_bvp's own Newton iteration builds and factorizes on every iteration, via
# construct_global_jac (scipy's own internal collocation assembly - Kierzenka & Shampine's
# scheme). This is the ACTUAL matrix scipy inverts, not a hand-reconstruction of it - avoids
# any risk of a subtly-different formula giving a misleading answer.
#
# For each captured iteration, attempts the SAME sparse LU factorization scipy's own
# solve_newton uses internally (scipy.sparse.linalg.splu) to detect singularity exactly the
# way scipy does; on success, inspects the LU's U-factor diagonal for anomalously small pivots
# (the classic near-singularity signature) and maps the smallest ones back to
# (mesh point, m/M_TOTAL, equation) via the known block-row indexing (row = mesh_point_index*4
# + equation_index, equations 0-3 = [r_hat, lnP, L_hat, lnT]).
#
# Also runs a direct, controlled non-determinism check: the identical (state, dt) solved
# twice in the same process, diffing the first captured Jacobian bit-for-bit.
#
# Usage: python run_scripts/diag_singular_jacobian.py
# (edit SNAPSHOT_PATH/DT_YR below to point at whatever state+dt is currently failing)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.integrate._bvp as _bvp_internal
import scipy.sparse.linalg as spla

import bvp_solver
import config
import gradients
import output

SNAPSHOT_DIR, _ = output.run_output_dirs("Phase1_deep_diag")
SNAPSHOT_PATH = f"{SNAPSHOT_DIR}/snapshot_00002.npz"   # T_center=1594.02K, the last good state before the wall
DT_YR = 6.05                                            # the exact dt the next real step was attempting

EQUATION_NAMES = ["r_hat (continuity)", "lnP (hydrostatic eq.)", "L_hat (energy eq.)", "lnT (temperature grad.)"]


# ==========================================
# SECTION: Monkey-patch to capture every real Newton-iteration Jacobian
# ==========================================

_captured = []
_orig_prepare_sys = _bvp_internal.prepare_sys


def _patched_prepare_sys(n, m, k, fun, bc, fun_jac, bc_jac, x, h):
    col_fun, sys_jac = _orig_prepare_sys(n, m, k, fun, bc, fun_jac, bc_jac, x, h)

    def wrapped_sys_jac(y, p, y_middle, f, f_middle, bc0):
        J = sys_jac(y, p, y_middle, f, f_middle, bc0)
        _captured.append({"x": x.copy(), "h": h.copy(), "y": y.copy(), "J": J.copy()})
        return J
    return col_fun, wrapped_sys_jac


def _run_once(state_prev, dt):
    """One full _solve_structure_bvp attempt, with the monkey-patch active, capturing every
    Newton iteration's exact collocation Jacobian into the module-level _captured list."""
    _bvp_internal.prepare_sys = _patched_prepare_sys
    try:
        try:
            sol, m_min = bvp_solver._solve_structure_bvp(
                state_prev, dt, warm_start_L=True,
                switch_epsilon=config.GRAD_EFF_SWITCH_EPSILON_TIMESTEP, use_analytic_jacobian=True)
            return sol, m_min
        except (RuntimeError, AssertionError) as exc:
            print(f"  (attempt ended: {exc})")
            return None, None
    finally:
        _bvp_internal.prepare_sys = _orig_prepare_sys


# ==========================================
# SECTION: Near-singularity analysis of one captured Jacobian
# ==========================================

def _analyze_jacobian(entry, m_min, label):
    """Sparse LU factorization (exactly scipy's own solve_newton mechanism) + smallest-pivot
    inspection, mapped back to physical (mesh point, m/M_TOTAL, equation)."""
    J, x, h = entry["J"], entry["x"], entry["h"]
    n_mesh = len(x)
    m_grid = np.exp(x)
    print(f"\n--- {label}: matrix shape {J.shape}, mesh points={n_mesh} ---")

    try:
        lu = spla.splu(J.tocsc())
    except RuntimeError as exc:
        print(f"  splu FAILED (scipy's own singularity signal): {exc}")
        return

    U_diag = np.abs(lu.U.diagonal())
    # lu.perm_r/perm_c map factorization-order back to the ORIGINAL row/col indices.
    orig_row_of_pivot = np.argsort(lu.perm_r)   # row index in the ORIGINAL matrix for each pivot position
    order = np.argsort(U_diag)   # smallest pivots first

    print(f"  smallest 10 |U diagonal| pivots (of {len(U_diag)}) - near-zero = near-singular direction:")
    print(f"  {'pivot |U|':>12s}  {'orig row':>9s}  {'mesh pt':>8s}  {'m/M_TOTAL':>10s}  equation")
    for k in order[:10]:
        orig_row = orig_row_of_pivot[k]
        if orig_row < 4 * n_mesh:
            mesh_pt = orig_row // 4
            eq = orig_row % 4
            print(f"  {U_diag[k]:12.4e}  {orig_row:9d}  {mesh_pt:8d}  {m_grid[mesh_pt]/config.M_TOTAL:10.6f}  {EQUATION_NAMES[eq]}")
        else:
            print(f"  {U_diag[k]:12.4e}  {orig_row:9d}  (boundary-condition row, not a mesh point)")

    print(f"  condition proxy (max|U diag| / min|U diag|) = {U_diag.max()/max(U_diag.min(),1e-300):.4e}")


def _physical_profile_near(state_prev, dt, m_frac_center, half_width_frac=0.08):
    """Direct physical diagnostic (independent of the Jacobian capture): grad_rad, grad_ad,
    and the smoothed-Schwarzschild-switch derivative d(grad_eff)/d(grad_rad) across the warm-
    start guess mesh, focused on the m/M_TOTAL window the singular-row analysis points to -
    tests directly whether this is a deep-convective-saturation zone (d(grad_eff)/d(grad_rad)
    ~0, the SAME rank-deficiency mechanism that caused the original shooting-method
    abandonment, bvp_solver.py's own module docstring / PLAN_BVP.md history) rather than
    assumed."""
    x, z_guess = bvp_solver.build_mesh_and_guess_scaled(state_prev, warm_start_L=True)
    y = bvp_solver._to_physical(z_guess)
    r, lnP, L, lnT = y
    P, T = bvp_solver._safe_exp_state(lnP, lnT)
    m = np.exp(x)
    frac = m / config.M_TOTAL

    mask = (frac >= m_frac_center - half_width_frac) & (frac <= m_frac_center + half_width_frac)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"  (no guess-mesh points in m/M_TOTAL window [{m_frac_center-half_width_frac:.3f},{m_frac_center+half_width_frac:.3f}])")
        return

    import eos
    import opacity
    mu_T = eos.mean_molecular_weight(T[idx])
    rho = eos.density(P[idx], T[idx], mu_T, config.MU_E)
    kappa = opacity.bell_lin_opacity(rho, T[idx])
    grad_ad = eos.grad_adiabatic(eos.gamma_effective(T[idx]))
    grad_rad = gradients.grad_radiative(L[idx], m[idx], P[idx], T[idx], kappa)
    eps_s = config.GRAD_EFF_SWITCH_EPSILON_TIMESTEP
    diff = grad_rad - grad_ad
    smooth = diff / np.sqrt(diff**2 + eps_s**2)
    dgeff_dgrad_rad = 0.5 * (1.0 - smooth)   # from bvp_solver._effective_gradient_derivative

    print(f"\n--- Physical profile, guess mesh, m/M_TOTAL in [{m_frac_center-half_width_frac:.3f},{m_frac_center+half_width_frac:.3f}] ({len(idx)} pts) ---")
    print(f"  T range: [{T[idx].min():.2f}, {T[idx].max():.2f}] K")
    print(f"  rho range: [{rho.min():.3e}, {rho.max():.3e}] g/cm^3")
    print(f"  grad_ad range: [{grad_ad.min():.4f}, {grad_ad.max():.4f}]")
    print(f"  grad_rad range: [{grad_rad.min():.4e}, {grad_rad.max():.4e}]")
    print(f"  superadiabaticity (grad_rad-grad_ad) range: [{diff.min():.4e}, {diff.max():.4e}]")
    print(f"  d(grad_eff)/d(grad_rad) range: [{dgeff_dgrad_rad.min():.4e}, {dgeff_dgrad_rad.max():.4e}]  <- near 0 = convective-saturated (grad_eff pinned at grad_ad, decoupled from L), near 1 = radiative-saturated (grad_eff tracks grad_rad), near 0.5 = actively transitioning")
    n_near_0 = np.sum(dgeff_dgrad_rad < 0.01)
    n_near_1 = np.sum(dgeff_dgrad_rad > 0.99)
    print(f"  points with d(grad_eff)/d(grad_rad)<0.01 (convective-saturated, grad_rad>>grad_ad): {n_near_0}/{len(idx)}")
    print(f"  points with d(grad_eff)/d(grad_rad)>0.99 (radiative-saturated, grad_rad<<grad_ad): {n_near_1}/{len(idx)}")


def main():
    state_prev, _ = output.load_snapshot(SNAPSHOT_PATH)
    dt = DT_YR * config.SECONDS_PER_YEAR
    print(f"Loaded {SNAPSHOT_PATH}: T_center={state_prev.T[0]:.2f}K, r_surface={state_prev.r[-1]/config.R_JUPITER_CM:.2f} R_Jup")
    print(f"Attempting dt={DT_YR} yr (the real next step)\n")

    print("=" * 70)
    print("RUN 1")
    print("=" * 70)
    _captured.clear()
    _run_once(state_prev, dt)
    run1_captures = list(_captured)
    print(f"\nCaptured {len(run1_captures)} Newton-iteration Jacobians in run 1")

    if run1_captures:
        _analyze_jacobian(run1_captures[0], None, "RUN 1, iteration 1 (first Newton step)")
        if len(run1_captures) > 1:
            _analyze_jacobian(run1_captures[-1], None, f"RUN 1, iteration {len(run1_captures)} (last before stopping)")

    print("\n" + "=" * 70)
    print("RUN 2 (identical inputs - non-determinism check)")
    print("=" * 70)
    _captured.clear()
    _run_once(state_prev, dt)
    run2_captures = list(_captured)
    print(f"\nCaptured {len(run2_captures)} Newton-iteration Jacobians in run 2")

    if run1_captures and run2_captures:
        J1, J2 = run1_captures[0]["J"], run2_captures[0]["J"]
        same_shape = J1.shape == J2.shape
        print(f"\n--- Non-determinism check (iteration 1, run 1 vs run 2) ---")
        print(f"  same shape: {same_shape}")
        if same_shape:
            diff = (J1 - J2)
            max_abs_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0.0
            print(f"  max |J1 - J2| entrywise: {max_abs_diff:.6e} (0.0 = bit-for-bit identical)")

    # Physical-mechanism check at the location the user's own diagnostics flagged (m/M~0.75),
    # PLUS a broader sweep to see if other windows look similarly saturated.
    for center in [0.75, 0.50, 0.90, 0.25]:
        _physical_profile_near(state_prev, dt, center)


if __name__ == "__main__":
    main()
