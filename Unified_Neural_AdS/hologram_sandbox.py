"""
hologram_sandbox.py — Phase 3: Holographic Sandbox with Ground Truth
======================================================================

Design your own quantum boundary and watch the Neural-AdS engine
reconstruct the bulk — validated against the **exact LU solver**.

**Single-source mode** (default):
    Φ_boundary(x, t) = A · sin(k·x + ω·t + φ)

**Collision mode** (``--collision``):
    Φ_total(x, t) = A₁·sin(k₁·x + ω₁·t) + A₂·sin(k₂·x + ω₂·t + φ₂)

Both modes show: Ground Truth (LU Solver) | Neural-AdS | Error

Usage
-----
    python hologram_sandbox.py
    python hologram_sandbox.py --k 3.0 --omega 2.0 --amplitude 0.5
    python hologram_sandbox.py --collision --k 3 --omega 2 --k2 5 --omega2 2.5
    python hologram_sandbox.py --collision --no_display
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fno_architectures import FNO3d


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE   = 64
TIME_STEPS  = 20
MODEL_PATH_SINGLE    = os.path.join("models", "unified_time_final.pth")
MODEL_PATH_COLLISION = os.path.join("models", "collision_rigorous.pth")
DATA_DIR_SINGLE      = "data_holography_time"
DATA_DIR_COLLISION   = "data_collision_master"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 — Holographic Sandbox with Ground Truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode
    p.add_argument("--collision", action="store_true",
                    help="Dual-source collision mode")

    # Source A
    p.add_argument("--k", type=float, default=None,
                    help="Wavenumber (1.0–8.0)")
    p.add_argument("--omega", type=float, default=None,
                    help="Frequency (0.5–4.0)")
    p.add_argument("--amplitude", type=float, default=0.5,
                    help="Amplitude (training range: 0.1–0.75)")
    p.add_argument("--phase", type=float, default=0.0,
                    help="Phase offset (radians)")
    p.add_argument("--harmonics", type=int, default=1,
                    help="Harmonic overtones (single-source only)")

    # Source B (collision)
    p.add_argument("--k2", type=float, default=None)
    p.add_argument("--omega2", type=float, default=None)
    p.add_argument("--amp2", type=float, default=0.4)
    p.add_argument("--phi2", type=float, default=0.0)

    # Model / output
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no_display", action="store_true")
    return p.parse_args()


def interactive_input(args: argparse.Namespace) -> argparse.Namespace:
    """Prompt for parameters if not supplied via CLI."""
    if args.collision:
        if args.k is None:
            print("╔══════════════════════════════════════════════════════════╗")
            print("║   🌌  COLLISION SANDBOX — Dual-Source Interference      ║")
            print("╚══════════════════════════════════════════════════════════╝\n")
            try:
                print("   === Source A ===")
                args.k = float(input("   Wavenumber  k₁  (1.0–8.0)  → "))
                args.omega = float(input("   Frequency   ω₁  (0.5–4.0) → "))
                amp_str = input("   Amplitude   A₁  (0.1–0.75, default 0.5) → ").strip()
                if amp_str: args.amplitude = float(amp_str)

                print("\n   === Source B ===")
                args.k2 = float(input("   Wavenumber  k₂  (1.0–8.0)  → "))
                args.omega2 = float(input("   Frequency   ω₂  (0.5–4.0) → "))
                amp2_str = input("   Amplitude   A₂  (0.1–0.75, default 0.4) → ").strip()
                if amp2_str: args.amp2 = float(amp2_str)
                phi2_str = input("   Phase       φ₂  (default 0.0)  → ").strip()
                if phi2_str: args.phi2 = float(phi2_str)
            except (ValueError, EOFError):
                print("⚠  Using defaults.")
                args.k, args.omega = 3.0, 2.0
                args.k2, args.omega2 = 5.0, 2.5
        if args.k2 is None: args.k2 = 5.0
        if args.omega2 is None: args.omega2 = 2.5
        if args.omega is None: args.omega = 2.0
    else:
        if args.k is None:
            print("╔══════════════════════════════════════════════════════╗")
            print("║      🌌  HOLOGRAPHIC SANDBOX — Design Your Wave     ║")
            print("╚══════════════════════════════════════════════════════╝\n")
            try:
                args.k = float(input("   Wavenumber  k  (1.0–8.0)  → "))
                args.omega = float(input("   Frequency   ω  (0.5–4.0) → "))
                amp_str = input("   Amplitude   A  (0.1–0.75, default 0.5) → ").strip()
                if amp_str: args.amplitude = float(amp_str)
            except (ValueError, EOFError):
                args.k, args.omega = 3.0, 2.0
        elif args.omega is None:
            args.omega = 2.0
    print()
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Exact Laplace Solver (LU)
# ─────────────────────────────────────────────────────────────────────────────

def build_laplace_system(n: int):
    """LU-factorise the 2-D Laplacian with Dirichlet BCs."""
    N = n * n
    main_diag    = -4.0 * np.ones(N)
    side_diag    = np.ones(N - 1)
    side_diag[np.arange(1, N) % n == 0] = 0
    up_down_diag = np.ones(N - n)

    A = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -n, n], shape=(N, N), format="lil",
    )
    grid_idx = np.arange(N).reshape(n, n)
    all_bdy = np.unique(np.concatenate((
        grid_idx[0, :], grid_idx[-1, :],
        grid_idx[:, 0], grid_idx[:, -1],
    )))
    for idx in all_bdy:
        A[idx, :] = 0
        A[idx, idx] = 1
    return spla.factorized(A.tocsc()), grid_idx[0, :]


def solve_ground_truth(boundary: np.ndarray) -> np.ndarray:
    """Solve ∇²Φ = 0 for every time-slice. Returns (T, X, Z)."""
    T, X = boundary.shape
    solve, top_row = build_laplace_system(X)

    bulk = np.empty((T, X, X), dtype=np.float32)
    b = np.zeros(X * X, dtype=np.float64)
    for t_idx in range(T):
        b[:] = 0.0
        b[top_row] = boundary[t_idx]
        bulk[t_idx] = solve(b).reshape(X, X).astype(np.float32)
    return bulk


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation stats
# ─────────────────────────────────────────────────────────────────────────────

def load_norm_stats(data_dir: str, collision: bool = False) -> dict:
    bdy_file  = "bdy_collision.npy" if collision else "boundary_time.npy"
    bulk_file = "bulk_collision.npy" if collision else "bulk_time.npy"
    bdy  = np.load(os.path.join(data_dir, bdy_file),  mmap_mode="r")
    bulk = np.load(os.path.join(data_dir, bulk_file), mmap_mode="r")
    stats = dict(x_mean=float(bdy.mean()), x_std=float(bdy.std()),
                 y_mean=float(bulk.mean()), y_std=float(bulk.std()))
    del bdy, bulk
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Boundary generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_boundary(k, omega, amplitude=0.5, phase=0.0,
                              harmonics=1) -> np.ndarray:
    x = np.linspace(0.0, 2 * np.pi, GRID_SIZE, dtype=np.float64)
    t = np.linspace(0.0, 2 * np.pi, TIME_STEPS, dtype=np.float64)
    t_col, x_row = t[:, None], x[None, :]
    bdy = np.zeros((TIME_STEPS, GRID_SIZE), dtype=np.float32)
    for h in range(harmonics):
        n = h + 1
        bdy += (amplitude / n) * np.sin(n * k * x_row + omega * t_col + phase)
    return bdy


def generate_collision_boundary(k1, omega1, amp1, k2, omega2, amp2,
                                 phi2) -> np.ndarray:
    x = np.linspace(0.0, 2 * np.pi, GRID_SIZE, dtype=np.float64)
    t = np.linspace(0.0, 2 * np.pi, TIME_STEPS, dtype=np.float64)
    t_col, x_row = t[:, None], x[None, :]
    wave = (amp1 * np.sin(k1 * x_row + omega1 * t_col)
          + amp2 * np.sin(k2 * x_row + omega2 * t_col + phi2))
    return wave.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Input volume + model + inference
# ─────────────────────────────────────────────────────────────────────────────

def construct_input_volume(boundary: np.ndarray, norm: dict) -> np.ndarray:
    T, X = boundary.shape
    Z = GRID_SIZE
    bdy_norm = (boundary - norm["x_mean"]) / (norm["x_std"] + 1e-8)
    wave_3d  = np.tile(bdy_norm[:, :, None], (1, 1, Z))
    t_coord  = np.linspace(0.0, 1.0, T, dtype=np.float32)
    time_3d  = np.broadcast_to(t_coord[:, None, None], (T, X, Z)).copy()
    z_coord  = np.linspace(0.0, 1.0, Z, dtype=np.float32)
    depth_3d = np.broadcast_to(z_coord[None, None, :], (T, X, Z)).copy()
    return np.stack([wave_3d, time_3d, depth_3d], axis=0)[None].astype(np.float32)


def load_model(path, device, modes=8, width=20, n_layers=4):
    model = FNO3d(modes1=modes, modes2=modes, modes3=modes,
                  width=width, n_layers=n_layers, in_channels=3).to(device)
    if not os.path.isfile(path):
        print(f"❌ Checkpoint not found: {path}"); sys.exit(1)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"🧠 FNO3d loaded — {sum(p.numel() for p in model.parameters()):,} params")
    return model


@torch.no_grad()
def infer(model, volume, device, y_mean, y_std):
    x = torch.from_numpy(volume).to(device)
    pred = model(x).squeeze().cpu().numpy()
    return pred * (y_std + 1e-8) + y_mean


# ─────────────────────────────────────────────────────────────────────────────
# 1×3 Animation: Ground Truth | Prediction | Error
# ─────────────────────────────────────────────────────────────────────────────

def animate_comparison(
    boundary: np.ndarray,
    bulk_truth: np.ndarray,
    bulk_pred: np.ndarray,
    metrics: dict,
    title: str,
    save_path: str,
    fps: int = 4,
    dpi: int = 150,
    show: bool = True,
) -> None:
    """Create a 1×3 animated GIF: Truth | Prediction | Error."""
    T, X, Z = bulk_truth.shape
    error = np.abs(bulk_truth - bulk_pred)

    bulk_vmin = min(bulk_truth.min(), bulk_pred.min())
    bulk_vmax = max(bulk_truth.max(), bulk_pred.max())
    err_vmax  = error.max()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    # Metrics subtitle
    sub = (f"MAE: {metrics['mae']:.4f} ({metrics['mae_pct']:.2f}%)   │   "
           f"Max: {metrics['max_err']:.4f} ({metrics['max_pct']:.2f}%)   │   "
           f"Rel L₂: {metrics['rel_l2']:.4f}   │   "
           f"Speedup: {metrics['speedup']:.0f}×")
    fig.text(0.5, 0.925, sub, ha="center", fontsize=10, fontstyle="italic",
             color="#263238",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                       edgecolor="#81C784", alpha=0.95))

    # Panel 0: Ground Truth
    im0 = axes[0].imshow(bulk_truth[0], cmap="inferno",
                          vmin=bulk_vmin, vmax=bulk_vmax,
                          origin="upper", aspect="equal")
    axes[0].set_title("Ground Truth (LU Solver)", fontsize=13,
                       fontweight="bold", pad=10, color="#1B5E20")
    axes[0].set_xlabel("Boundary (x)"); axes[0].set_ylabel("Depth (z)")
    div0 = make_axes_locatable(axes[0])
    plt.colorbar(im0, cax=div0.append_axes("right", size="5%", pad=0.1))

    # Panel 1: Prediction
    im1 = axes[1].imshow(bulk_pred[0], cmap="inferno",
                          vmin=bulk_vmin, vmax=bulk_vmax,
                          origin="upper", aspect="equal")
    axes[1].set_title("Neural-AdS Prediction", fontsize=13,
                       fontweight="bold", pad=10, color="#1565C0")
    axes[1].set_xlabel("Boundary (x)"); axes[1].set_ylabel("Depth (z)")
    div1 = make_axes_locatable(axes[1])
    plt.colorbar(im1, cax=div1.append_axes("right", size="5%", pad=0.1))

    # Panel 2: Error
    im2 = axes[2].imshow(error[0], cmap="hot", vmin=0, vmax=err_vmax,
                          origin="upper", aspect="equal")
    axes[2].set_title("Absolute Error", fontsize=13,
                       fontweight="bold", pad=10, color="#B71C1C")
    axes[2].set_xlabel("Boundary (x)"); axes[2].set_ylabel("Depth (z)")
    div2 = make_axes_locatable(axes[2])
    plt.colorbar(im2, cax=div2.append_axes("right", size="5%", pad=0.1))

    time_badge = fig.text(0.02, 0.02, "", fontsize=11, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor="#E3F2FD", alpha=0.95))

    plt.tight_layout(rect=[0, 0.04, 1, 0.90], w_pad=2.5)

    def update(t):
        im0.set_data(bulk_truth[t])
        im1.set_data(bulk_pred[t])
        im2.set_data(error[t])
        mae_t = np.mean(error[t])
        axes[2].set_title(f"Absolute Error  (MAE: {mae_t:.4f})",
                           fontsize=13, fontweight="bold", pad=10, color="#B71C1C")
        time_badge.set_text(f"  t = {t + 1}/{T}  ")
        return im0, im1, im2, time_badge

    anim = animation.FuncAnimation(fig, update, frames=T,
                                    interval=1000 // fps, blit=False)
    anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
    print(f"\n🎬 Animation saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.no_display:
        matplotlib.use("Agg")
    args = interactive_input(args)

    import time as _time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}\n")

    # ── Resolve model & data ──────────────────────────────────────────
    if args.collision:
        model_path = args.model_path or MODEL_PATH_COLLISION
        data_dir   = args.data_dir or DATA_DIR_COLLISION
        mode_label = "Collision"
    else:
        model_path = args.model_path or MODEL_PATH_SINGLE
        data_dir   = args.data_dir or DATA_DIR_SINGLE
        mode_label = "Single-Source"

    print(f"🔧 Mode: {mode_label}  |  Model: {model_path}")

    # ── Norm stats ────────────────────────────────────────────────────
    print("📦 Loading normalisation statistics …")
    norm = load_norm_stats(data_dir, collision=args.collision)
    print(f"   x̄={norm['x_mean']:.4f}  σx={norm['x_std']:.4f}  "
          f"ȳ={norm['y_mean']:.4f}  σy={norm['y_std']:.4f}")

    # ── Generate boundary ─────────────────────────────────────────────
    if args.collision:
        print(f"\n🌊 Φ(x,t) = {args.amplitude:.2f}·sin({args.k:.1f}x + "
              f"{args.omega:.1f}t) + {args.amp2:.2f}·sin({args.k2:.1f}x + "
              f"{args.omega2:.1f}t + {args.phi2:.2f})")
        boundary = generate_collision_boundary(
            args.k, args.omega, args.amplitude,
            args.k2, args.omega2, args.amp2, args.phi2,
        )
        anim_title = (f"Collision  ·  k₁={args.k:.1f}  ω₁={args.omega:.1f}"
                      f"   ⊕   k₂={args.k2:.1f}  ω₂={args.omega2:.1f}")
        default_name = "collision_sandbox.gif"
    else:
        print(f"\n🌊 Φ(x,t) = {args.amplitude:.2f}·sin({args.k:.1f}x + "
              f"{args.omega:.1f}t)")
        boundary = generate_single_boundary(
            args.k, args.omega, args.amplitude, args.phase, args.harmonics,
        )
        anim_title = (f"Single-Source  ·  k = {args.k:.1f}   "
                      f"ω = {args.omega:.1f}")
        default_name = "custom_universe.gif"

    print(f"   Shape: {boundary.shape}  Range: [{boundary.min():.3f}, "
          f"{boundary.max():.3f}]")

    # ── Ground truth (LU solver) ──────────────────────────────────────
    print("\n🔬 Computing exact ground truth (LU solver) …")
    t0 = _time.perf_counter()
    bulk_truth = solve_ground_truth(boundary)
    solver_time = _time.perf_counter() - t0
    print(f"   ✔ Done in {solver_time:.3f}s")

    # ── Neural-AdS inference ──────────────────────────────────────────
    model = load_model(model_path, device,
                       args.modes, args.width, args.n_layers)
    volume = construct_input_volume(boundary, norm)

    print("\n⚡ Running Neural-AdS inference …")
    t1 = _time.perf_counter()
    bulk_pred = infer(model, volume, device, norm["y_mean"], norm["y_std"])
    nn_time = _time.perf_counter() - t1
    print(f"   ✔ Done in {nn_time:.3f}s")

    speedup = solver_time / (nn_time + 1e-9)

    # ── Metrics ───────────────────────────────────────────────────────
    error     = np.abs(bulk_truth - bulk_pred)
    mae       = float(np.mean(error))
    max_err   = float(np.max(error))
    data_range = float(bulk_truth.max() - bulk_truth.min()) + 1e-8
    mae_pct   = (mae / data_range) * 100
    max_pct   = (max_err / data_range) * 100
    rel_l2    = float(np.linalg.norm(bulk_pred.ravel() - bulk_truth.ravel())
                      / (np.linalg.norm(bulk_truth.ravel()) + 1e-12))

    metrics = dict(mae=mae, max_err=max_err, mae_pct=mae_pct,
                   max_pct=max_pct, rel_l2=rel_l2, speedup=speedup)

    print(f"\n{'═' * 60}")
    print(f"  📊 ACCURACY REPORT — Neural-AdS vs Exact Solver")
    print(f"{'═' * 60}")
    print(f"  MAE              : {mae:.6f}  ({mae_pct:.2f}% of range)")
    print(f"  Max Error        : {max_err:.6f}  ({max_pct:.2f}% of range)")
    print(f"  Relative L₂     : {rel_l2:.6f}")
    print(f"  LU Solver Time   : {solver_time:.3f}s")
    print(f"  FNO Time         : {nn_time:.3f}s")
    print(f"  Speedup          : {speedup:.0f}×")
    print(f"{'═' * 60}")

    # ── Animation ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name or default_name
    save_path = os.path.join(args.output_dir, output_name)

    print(f"\n🎨 Rendering {TIME_STEPS}-frame comparison animation …")
    animate_comparison(
        boundary, bulk_truth, bulk_pred, metrics,
        anim_title, save_path,
        fps=args.fps, dpi=args.dpi, show=not args.no_display,
    )

    print("\n✅ Done. 🌌")


if __name__ == "__main__":
    main()
