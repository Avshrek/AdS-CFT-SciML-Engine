"""
animate_collisions.py — Collision-Regime Spatiotemporal Animation
===================================================================

Loads the fine-tuned ``collision_rigorous.pth`` FNO3d checkpoint, picks a
random sample from the dual-source collision dataset, runs inference, and
produces a side-by-side animated GIF:

    [Dual-Source Boundary(t)]  |  [Predicted Bulk(t)]  |  [Absolute Error(t)]

with a metrics banner showing MAE, Relative L₂, and PDE Residual.

Saved to ``results/collision_evolution.gif``.

Usage
-----
    python animate_collisions.py
    python animate_collisions.py --sample_idx 42
    python animate_collisions.py --no_display
    python animate_collisions.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fno_architectures import FNO3d


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collision-regime spatiotemporal animation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", type=str,
                    default=os.path.join("models", "collision_rigorous.pth"),
                    help="Path to the fine-tuned FNO3d state dict")
    p.add_argument("--data_dir", type=str, default="data_collision_master",
                    help="Directory containing bdy_collision.npy & bulk_collision.npy")
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--sample_idx", type=int, default=None,
                    help="Specific sample index (default: random)")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--output_name", type=str, default="collision_evolution.gif")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no_display", action="store_true",
                    help="Skip plt.show() — save only")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PDE Residual
# ─────────────────────────────────────────────────────────────────────────────

_LAP_KERNEL = torch.tensor(
    [[0.0,  1.0,  0.0],
     [1.0, -4.0,  1.0],
     [0.0,  1.0,  0.0]],
    dtype=torch.float32,
).reshape(1, 1, 3, 3)


@torch.no_grad()
def compute_pde_residual(pred_tensor: torch.Tensor) -> float:
    """Mean absolute Laplacian residual ⟨|∇²Φ|⟩ over all time-slices."""
    device = pred_tensor.device
    kernel = _LAP_KERNEL.to(device)
    B, C, T, X, Z = pred_tensor.shape
    slices = pred_tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, X, Z)
    lap = F.conv2d(slices, kernel, padding=1)
    interior = lap[:, :, 1:-1, 1:-1]
    return float(torch.mean(torch.abs(interior)).cpu())


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_random_sample(
    data_dir: str,
    sample_idx: int | None = None,
):
    """Load one random sample and construct the 3-channel model input.

    Returns
    -------
    x_input    : ``(1, 3, T, X, Z)`` — normalised input volume.
    bdy_raw    : ``(T, X)``          — raw boundary (original scale).
    bulk_true  : ``(T, X, Z)``       — ground-truth bulk (original scale).
    y_mean, y_std : float            — bulk normalisation stats.
    idx        : int
    """
    bdy_all  = np.load(os.path.join(data_dir, "bdy_collision.npy")).astype(np.float32)
    bulk_all = np.load(os.path.join(data_dir, "bulk_collision.npy")).astype(np.float32)

    N, T, X = bdy_all.shape
    Z = bulk_all.shape[3]

    idx = sample_idx if sample_idx is not None else np.random.randint(0, N)
    idx = min(max(idx, 0), N - 1)
    print(f"📌 Selected sample index: {idx}  (out of {N})")

    bdy_raw   = bdy_all[idx]
    bulk_true = bulk_all[idx]

    x_mean, x_std = float(bdy_all.mean()),  float(bdy_all.std())
    y_mean, y_std = float(bulk_all.mean()), float(bulk_all.std())
    del bdy_all, bulk_all

    bdy_norm = (bdy_raw - x_mean) / (x_std + 1e-8)

    wave_3d  = np.tile(bdy_norm[:, :, np.newaxis], (1, 1, Z))
    t_coord  = np.linspace(0.0, 1.0, T, dtype=np.float32)
    time_3d  = np.broadcast_to(
        t_coord[:, np.newaxis, np.newaxis], (T, X, Z)
    ).copy()
    z_coord  = np.linspace(0.0, 1.0, Z, dtype=np.float32)
    depth_3d = np.broadcast_to(
        z_coord[np.newaxis, np.newaxis, :], (T, X, Z)
    ).copy()

    x_input = np.stack([wave_3d, time_3d, depth_3d], axis=0)[np.newaxis, ...]

    return x_input, bdy_raw, bulk_true, y_mean, y_std, idx


# ─────────────────────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────────────────────

def build_and_load_model(
    model_path: str,
    device: torch.device,
    *,
    modes: int = 8,
    width: int = 20,
    n_layers: int = 4,
) -> FNO3d:
    """Instantiate FNO3d and load fine-tuned collision weights."""
    model = FNO3d(
        modes1=modes, modes2=modes, modes3=modes,
        width=width, n_layers=n_layers, in_channels=3,
    ).to(device)

    if not os.path.isfile(model_path):
        print(f"❌ Checkpoint not found: {model_path}")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 FNO3d loaded — {total_params:,} parameters  ({model_path})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: FNO3d,
    x_input: np.ndarray,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> tuple[np.ndarray, torch.Tensor]:
    """Forward pass → (denormalised bulk, raw tensor for PDE validation)."""
    x_tensor = torch.from_numpy(x_input).to(device)
    pred_tensor = model(x_tensor)
    pred_np = pred_tensor.squeeze().cpu().numpy()
    bulk_pred = pred_np * (y_std + 1e-8) + y_mean
    return bulk_pred, pred_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────

def create_animation(
    bdy_raw: np.ndarray,
    bulk_true: np.ndarray,
    bulk_pred: np.ndarray,
    metrics_text: str,
    save_path: str,
    fps: int = 4,
    dpi: int = 150,
) -> None:
    """Build and save a 1×3 animated GIF with a metrics banner.

    Panels
    ------
    0 — Evolving dual-source interference boundary (line plot)
    1 — Predicted 2-D Bulk (heatmap)
    2 — Absolute Error (heatmap)
    """
    T, X, Z = bulk_true.shape
    error = np.abs(bulk_true - bulk_pred)

    # ── Global colour limits (fixed across all frames) ────────────────────
    bulk_vmin = min(bulk_true.min(), bulk_pred.min())
    bulk_vmax = max(bulk_true.max(), bulk_pred.max())
    err_vmax  = error.max()
    bdy_margin = 0.1 * (bdy_raw.max() - bdy_raw.min() + 1e-8)
    bdy_ymin  = bdy_raw.min() - bdy_margin
    bdy_ymax  = bdy_raw.max() + bdy_margin

    # ── Figure setup ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5),
                              gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Dual-Source Collision Regime — Spatiotemporal Evolution",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # Metrics subtitle
    fig.text(0.5, 0.925, metrics_text, ha="center", fontsize=9.5,
             fontstyle="italic", color="#37474F",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1",
                       edgecolor="#B0BEC5", alpha=0.9))

    x_axis = np.arange(X)

    # --- Panel 0: Dual-source boundary wave ──────────────────────────────
    axes[0].set_xlim(0, X - 1)
    axes[0].set_ylim(bdy_ymin, bdy_ymax)
    axes[0].set_xlabel("Boundary Coordinate (x)", fontsize=10)
    axes[0].set_ylabel("Φ_total (interference)", fontsize=10)
    axes[0].set_title("Dual-Source Interference", fontsize=13,
                       fontweight="bold", pad=10)
    axes[0].grid(True, alpha=0.3)
    line_bdy, = axes[0].plot([], [], lw=2.2, color="#7C4DFF")
    time_text = axes[0].text(
        0.02, 0.95, "", transform=axes[0].transAxes,
        fontsize=11, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EDE7F6", alpha=0.9),
    )

    # --- Panel 1: Predicted bulk ─────────────────────────────────────────
    im_pred = axes[1].imshow(
        bulk_pred[0], cmap="inferno", vmin=bulk_vmin, vmax=bulk_vmax,
        origin="upper", aspect="equal",
    )
    axes[1].set_title("Neural-AdS Predicted Bulk", fontsize=13,
                       fontweight="bold", pad=10)
    axes[1].set_xlabel("Boundary (x)", fontsize=10)
    axes[1].set_ylabel("Radial Depth (z)", fontsize=10)
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im_pred, cax=cax1)

    # --- Panel 2: Absolute error ─────────────────────────────────────────
    im_err = axes[2].imshow(
        error[0], cmap="hot", vmin=0, vmax=err_vmax,
        origin="upper", aspect="equal",
    )
    axes[2].set_title("Absolute Error", fontsize=13,
                       fontweight="bold", pad=10)
    axes[2].set_xlabel("Boundary (x)", fontsize=10)
    axes[2].set_ylabel("Radial Depth (z)", fontsize=10)
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im_err, cax=cax2)

    plt.tight_layout(rect=[0, 0, 1, 0.90], w_pad=3.0)

    # ── Update function ──────────────────────────────────────────────────
    def update(t: int):
        line_bdy.set_data(x_axis, bdy_raw[t])
        time_text.set_text(f"t = {t + 1}/{T}")
        im_pred.set_data(bulk_pred[t])
        im_err.set_data(error[t])

        mae_t = np.mean(error[t])
        axes[2].set_title(f"Absolute Error  (MAE: {mae_t:.4f})",
                           fontsize=13, fontweight="bold", pad=10)

        return line_bdy, im_pred, im_err, time_text

    # ── Build & save ─────────────────────────────────────────────────────
    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 // fps, blit=False,
    )
    anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    x_input, bdy_raw, bulk_true, y_mean, y_std, idx = load_random_sample(
        args.data_dir, args.sample_idx,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_and_load_model(
        args.model_path, device,
        modes=args.modes, width=args.width, n_layers=args.n_layers,
    )

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n⚡ Running collision inference …")
    bulk_pred, pred_tensor = predict(model, x_input, device, y_mean, y_std)
    print(f"   Prediction shape: {bulk_pred.shape}")

    # ── Metrics ───────────────────────────────────────────────────────────
    error = np.abs(bulk_true - bulk_pred)
    mae = float(np.mean(error))
    max_err = float(np.max(error))
    data_range = float(bulk_true.max() - bulk_true.min()) + 1e-8
    mae_pct = (mae / data_range) * 100
    max_pct = (max_err / data_range) * 100

    pred_flat = bulk_pred.ravel()
    true_flat = bulk_true.ravel()
    rel_l2 = float(np.linalg.norm(pred_flat - true_flat)
                    / (np.linalg.norm(true_flat) + 1e-12))
    pde_res = compute_pde_residual(pred_tensor)

    print(f"\n📊 Collision Spatiotemporal Metrics:")
    print(f"   MAE              : {mae:.6f}  ({mae_pct:.2f}% of range)")
    print(f"   Max Error        : {max_err:.6f}  ({max_pct:.2f}% of range)")
    print(f"   Relative L₂     : {rel_l2:.6f}")
    print(f"   PDE Residual     : {pde_res:.2e}")

    metrics_text = (
        f"MAE: {mae:.4f} ({mae_pct:.1f}%)   │   "
        f"Rel L₂: {rel_l2:.4f}   │   "
        f"PDE ⟨|∇²Φ|⟩: {pde_res:.2e}   │   "
        f"Sample: #{idx}"
    )

    # ── Animation ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)

    print(f"\n🎨 Rendering {bulk_true.shape[0]}-frame animation …")
    create_animation(
        bdy_raw, bulk_true, bulk_pred, metrics_text,
        save_path, fps=args.fps, dpi=args.dpi,
    )
    print(f"\n🎬 Animation saved → {save_path}")
    print("\n✅ Collision animation complete.")


if __name__ == "__main__":
    main()
