"""
evaluate_physics.py — Physical Consistency Analysis
=====================================================

Validates that the Neural-AdS model respects fundamental physics:

  1. **Boundary Preservation**  — Does Φ(x, z=0) match the input boundary?
  2. **Bulk Decay Profile**     — Does the field decay correctly with depth z?
  3. **PDE Residual**           — Is ∇²Φ ≈ 0 satisfied in the interior?
  4. **Error vs Amplitude**     — Does accuracy depend on input amplitude?

Generates a 2×2 publication figure: ``results/physics_consistency.png``

Usage
-----
    python evaluate_physics.py
    python evaluate_physics.py --n_eval 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fno_architectures import FNO3d


def parse_args():
    p = argparse.ArgumentParser(
        description="Physical consistency analysis for the collision model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", type=str,
                    default=os.path.join("models", "collision_rigorous.pth"))
    p.add_argument("--data_dir", type=str, default="data_collision_master")
    p.add_argument("--n_eval", type=int, default=None,
                    help="Number of samples (default: all)")
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_data(data_dir):
    bdy  = np.load(os.path.join(data_dir, "bdy_collision.npy")).astype(np.float32)
    bulk = np.load(os.path.join(data_dir, "bulk_collision.npy")).astype(np.float32)
    stats = dict(x_mean=float(bdy.mean()), x_std=float(bdy.std()),
                 y_mean=float(bulk.mean()), y_std=float(bulk.std()))
    return bdy, bulk, stats


def build_model(path, device, modes=8, width=20, n_layers=4):
    model = FNO3d(modes1=modes, modes2=modes, modes3=modes,
                  width=width, n_layers=n_layers, in_channels=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def construct_input(bdy_s, stats, Z):
    T, X = bdy_s.shape
    bdy_n = (bdy_s - stats["x_mean"]) / (stats["x_std"] + 1e-8)
    wave  = np.tile(bdy_n[:, :, None], (1, 1, Z))
    tc    = np.linspace(0, 1, T, dtype=np.float32)
    time  = np.broadcast_to(tc[:, None, None], (T, X, Z)).copy()
    zc    = np.linspace(0, 1, Z, dtype=np.float32)
    depth = np.broadcast_to(zc[None, None, :], (T, X, Z)).copy()
    return np.stack([wave, time, depth], axis=0)[None].astype(np.float32)


@torch.no_grad()
def predict(model, x_in, device, y_mean, y_std):
    x = torch.from_numpy(x_in).to(device)
    p = model(x).squeeze().cpu().numpy()
    return p * (y_std + 1e-8) + y_mean


def compute_pde_residual(field):
    """5-point Laplacian stencil. field: (T, X, Z)."""
    lap = (field[:, 2:, 1:-1] + field[:, :-2, 1:-1]
         + field[:, 1:-1, 2:] + field[:, 1:-1, :-2]
         - 4 * field[:, 1:-1, 1:-1])
    return np.abs(lap)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    bdy_all, bulk_all, stats = load_data(args.data_dir)
    N, T, X = bdy_all.shape
    Z = bulk_all.shape[3]
    n_eval = min(args.n_eval or N, N)
    print(f"Dataset: {N} samples, evaluating {n_eval}")

    model = build_model(args.model_path, device,
                        args.modes, args.width, args.n_layers)

    # ── Per-sample metrics ────────────────────────────────────────────
    bdy_errors = []         # boundary preservation (z=0 MAE)
    decay_truth = []        # mean |field| vs depth (truth)
    decay_pred  = []        # mean |field| vs depth (pred)
    pde_residuals = []      # mean PDE residual per sample
    amplitudes = []         # input amplitude (for error vs amplitude)
    rel_l2s = []

    for i in range(n_eval):
        bdy_s, bulk_s = bdy_all[i], bulk_all[i]
        x_in = construct_input(bdy_s, stats, Z)
        pred = predict(model, x_in, device, stats["y_mean"], stats["y_std"])

        # 1. Boundary preservation: predicted z=0 vs input boundary
        pred_bdy = pred[:, :, 0]  # (T, X) at z=0
        bdy_err = np.mean(np.abs(pred_bdy - bdy_s))
        bdy_errors.append(bdy_err)

        # 2. Decay profile: mean |field| at each depth z
        truth_decay = np.mean(np.abs(bulk_s), axis=(0, 1))   # (Z,)
        pred_decay  = np.mean(np.abs(pred),   axis=(0, 1))   # (Z,)
        decay_truth.append(truth_decay)
        decay_pred.append(pred_decay)

        # 3. PDE residual
        res_truth = compute_pde_residual(bulk_s)
        res_pred  = compute_pde_residual(pred)
        pde_residuals.append(np.mean(res_pred))

        # 4. Amplitude
        amplitudes.append(np.std(bdy_s))

        # 5. Rel L2
        rl2 = np.linalg.norm(pred.ravel() - bulk_s.ravel()) / (
            np.linalg.norm(bulk_s.ravel()) + 1e-12)
        rel_l2s.append(rl2)

        if (i + 1) % max(1, n_eval // 5) == 0:
            print(f"   [{i+1}/{n_eval}]  bdy_err={bdy_err:.5f}  "
                  f"pde={pde_residuals[-1]:.5f}  rl2={rl2:.5f}")

    bdy_errors = np.array(bdy_errors)
    pde_residuals = np.array(pde_residuals)
    amplitudes = np.array(amplitudes)
    rel_l2s = np.array(rel_l2s)
    decay_truth = np.array(decay_truth)
    decay_pred  = np.array(decay_pred)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  PHYSICAL CONSISTENCY REPORT")
    print(f"{'=' * 60}")
    print(f"  Boundary Preservation (z=0 MAE)")
    print(f"    Mean +/- Std : {bdy_errors.mean():.6f} +/- {bdy_errors.std():.6f}")
    print(f"  PDE Residual (mean |nabla^2 phi|)")
    print(f"    Mean +/- Std : {pde_residuals.mean():.6f} +/- {pde_residuals.std():.6f}")
    print(f"  Correlation: amplitude vs Rel L2 = "
          f"{np.corrcoef(amplitudes, rel_l2s)[0,1]:.4f}")
    print(f"{'=' * 60}")

    # ── 2x2 Publication Figure ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle("Physical Consistency Analysis — Neural-AdS Collision Model",
                 fontsize=16, fontweight="bold", y=0.98)

    # Panel A: Boundary Preservation Histogram
    ax = axes[0, 0]
    ax.hist(bdy_errors, bins=30, color="#1565C0", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(bdy_errors.mean(), color="#B71C1C", lw=2, ls="--",
                label=f"Mean = {bdy_errors.mean():.4f}")
    ax.set_xlabel("Boundary MAE  (z=0 prediction vs input)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("(A) Boundary Condition Preservation", fontsize=13,
                  fontweight="bold")
    ax.legend(fontsize=10)

    # Panel B: Bulk Decay Profile
    ax = axes[0, 1]
    z = np.linspace(0, 1, Z)
    mean_truth = decay_truth.mean(axis=0)
    mean_pred  = decay_pred.mean(axis=0)
    std_truth  = decay_truth.std(axis=0)
    std_pred   = decay_pred.std(axis=0)

    ax.plot(z, mean_truth, "k-", lw=2.5, label="Ground Truth (LU)")
    ax.fill_between(z, mean_truth - std_truth, mean_truth + std_truth,
                     alpha=0.15, color="black")
    ax.plot(z, mean_pred, "--", color="#D32F2F", lw=2.5, label="Neural-AdS")
    ax.fill_between(z, mean_pred - std_pred, mean_pred + std_pred,
                     alpha=0.15, color="#D32F2F")
    ax.set_xlabel("Normalised depth  z / L", fontsize=11)
    ax.set_ylabel("Mean |field|  (averaged over x, t)", fontsize=11)
    ax.set_title("(B) Bulk Decay Profile", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Panel C: PDE Residual Histogram
    ax = axes[1, 0]
    ax.hist(pde_residuals, bins=30, color="#7C4DFF", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(pde_residuals.mean(), color="#B71C1C", lw=2, ls="--",
                label=f"Mean = {pde_residuals.mean():.4f}")
    ax.set_xlabel("Mean PDE Residual  |nabla^2 phi|", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("(C) PDE Satisfaction (Laplace Equation)", fontsize=13,
                  fontweight="bold")
    ax.legend(fontsize=10)

    # Panel D: Error vs Amplitude scatter
    ax = axes[1, 1]
    sc = ax.scatter(amplitudes, rel_l2s, c=pde_residuals, cmap="viridis",
                     alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel("Input Amplitude (boundary std dev)", fontsize=11)
    ax.set_ylabel("Relative L2 Error", fontsize=11)
    ax.set_title("(D) Error vs Input Complexity", fontsize=13,
                  fontweight="bold")
    r = np.corrcoef(amplitudes, rel_l2s)[0, 1]
    ax.text(0.95, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("PDE Residual", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "physics_consistency.png")
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {save_path}")


if __name__ == "__main__":
    main()
