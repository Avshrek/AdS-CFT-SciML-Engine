"""
animate_unified.py — Phase 2 Spatiotemporal Evaluation & Animation
====================================================================

Loads the trained FNO3d model, picks a random sample from the
time-dependent holographic dataset, runs inference, and produces
a side-by-side animated GIF:

    [Boundary Wave(t)]  |  [Predicted Bulk(t)]  |  [Absolute Error(t)]

Saved to ``results/spacetime_evolution.gif``.

Usage
-----
    python animate_unified.py
    python animate_unified.py --model_path models/unified_time_final.pth
    python animate_unified.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fno_architectures import FNO3d


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 — Spatiotemporal evaluation & GIF animation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", type=str,
                    default=os.path.join("models", "unified_time_final.pth"),
                    help="Path to the trained FNO3d state dict")
    p.add_argument("--data_dir", type=str, default="data_holography_time",
                    help="Directory containing boundary_time.npy & bulk_time.npy")
    p.add_argument("--modes", type=int, default=8,
                    help="Fourier modes per axis")
    p.add_argument("--width", type=int, default=20,
                    help="Hidden channel width of FNO3d")
    p.add_argument("--n_layers", type=int, default=4,
                    help="Number of 3-D Fourier layers")
    p.add_argument("--output_dir", type=str, default="results",
                    help="Directory to save the animation")
    p.add_argument("--output_name", type=str, default="spacetime_evolution.gif",
                    help="Filename for the output GIF")
    p.add_argument("--fps", type=int, default=4,
                    help="Frames per second for the GIF")
    p.add_argument("--dpi", type=int, default=150,
                    help="DPI for the animation frames")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_random_sample(data_dir: str):
    """Load one random sample and construct the 3-channel model input.

    Returns
    -------
    x_input   : ``(1, 3, T, X, Z)`` — normalised, model-ready input.
    bdy_raw   : ``(T, X)``          — raw boundary wave (original scale).
    bulk_true : ``(T, X, Z)``       — ground-truth bulk (original scale).
    y_mean, y_std : float           — bulk normalisation stats for de-norm.
    idx       : int                 — selected sample index.
    """
    bdy_all  = np.load(os.path.join(data_dir, "boundary_time.npy")).astype(np.float32)
    bulk_all = np.load(os.path.join(data_dir, "bulk_time.npy")).astype(np.float32)

    N, T, X = bdy_all.shape
    Z = bulk_all.shape[3]

    idx = np.random.randint(0, N)
    print(f"📌 Selected sample index: {idx}  (out of {N})")

    bdy_sample  = bdy_all[idx]       # (T, X)
    bulk_sample = bulk_all[idx]      # (T, X, Z)

    # ── Full-dataset normalisation stats ─────────────────────────────────
    x_mean, x_std = bdy_all.mean(), bdy_all.std()
    y_mean, y_std = bulk_all.mean(), bulk_all.std()
    del bdy_all, bulk_all            # free memory

    bdy_norm = (bdy_sample - x_mean) / (x_std + 1e-8)

    # ── Channel 0: boundary wave tiled over Z ────────────────────────────
    wave_3d = np.tile(bdy_norm[:, :, np.newaxis], (1, 1, Z))        # (T, X, Z)

    # ── Channel 1: normalised time coordinate ────────────────────────────
    t_coord = np.linspace(0.0, 1.0, T, dtype=np.float32)
    time_3d = np.broadcast_to(
        t_coord[:, np.newaxis, np.newaxis], (T, X, Z)
    ).copy()

    # ── Channel 2: normalised Z-depth coordinate ────────────────────────
    z_coord = np.linspace(0.0, 1.0, Z, dtype=np.float32)
    depth_3d = np.broadcast_to(
        z_coord[np.newaxis, np.newaxis, :], (T, X, Z)
    ).copy()

    # ── Stack → (1, 3, T, X, Z) ─────────────────────────────────────────
    x_input = np.stack([wave_3d, time_3d, depth_3d], axis=0)[np.newaxis, ...]

    return x_input, bdy_sample, bulk_sample, y_mean, y_std, idx


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def build_and_load_model(
    model_path: str,
    device: torch.device,
    *,
    modes: int = 8,
    width: int = 20,
    n_layers: int = 4,
) -> FNO3d:
    """Instantiate FNO3d and load saved weights."""
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
) -> np.ndarray:
    """Run forward pass and denormalise to physical scale.

    Returns
    -------
    prediction : ``(T, X, Z)``
    """
    x_tensor = torch.from_numpy(x_input).to(device)
    pred_norm = model(x_tensor)                             # (1, 1, T, X, Z)
    pred_norm = pred_norm.squeeze().cpu().numpy()            # (T, X, Z)
    return pred_norm * (y_std + 1e-8) + y_mean


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────

def create_animation(
    bdy_raw: np.ndarray,
    bulk_true: np.ndarray,
    bulk_pred: np.ndarray,
    save_path: str,
    fps: int = 4,
    dpi: int = 150,
) -> None:
    """Build and save a 1×3 animated GIF.

    Panels
    ------
    0 — Evolving 1-D Boundary Wave (line plot)
    1 — Predicted 2-D Bulk (heatmap)
    2 — Absolute Error (heatmap)
    """
    T, X, Z = bulk_true.shape
    error = np.abs(bulk_true - bulk_pred)

    # ── Global colour limits (fixed across all frames) ───────────────────
    bulk_vmin = min(bulk_true.min(), bulk_pred.min())
    bulk_vmax = max(bulk_true.max(), bulk_pred.max())
    err_vmax  = error.max()
    bdy_ymin  = bdy_raw.min() - 0.1 * (bdy_raw.max() - bdy_raw.min() + 1e-8)
    bdy_ymax  = bdy_raw.max() + 0.1 * (bdy_raw.max() - bdy_raw.min() + 1e-8)

    # ── Figure setup ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5),
                              gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Unified Neural-AdS · Phase 2 — Spatiotemporal Evolution",
                 fontsize=15, fontweight="bold", y=0.98)

    x_axis = np.arange(X)

    # --- Panel 0: Boundary wave (line plot) ──────────────────────────────
    axes[0].set_xlim(0, X - 1)
    axes[0].set_ylim(bdy_ymin, bdy_ymax)
    axes[0].set_xlabel("Boundary Coordinate (x)", fontsize=10)
    axes[0].set_ylabel("Φ_boundary", fontsize=10)
    axes[0].set_title("Boundary Wave", fontsize=13, fontweight="bold", pad=10)
    axes[0].grid(True, alpha=0.3)
    line_bdy, = axes[0].plot([], [], lw=2.2, color="#2196F3")
    time_text = axes[0].text(0.02, 0.95, "", transform=axes[0].transAxes,
                              fontsize=11, fontweight="bold", va="top",
                              bbox=dict(boxstyle="round,pad=0.3",
                                        facecolor="#E3F2FD", alpha=0.9))

    # --- Panel 1: Predicted bulk (heatmap) ───────────────────────────────
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

    # --- Panel 2: Absolute error (heatmap) ───────────────────────────────
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

    plt.tight_layout(rect=[0, 0, 1, 0.94], w_pad=3.0)

    # ── Update function ──────────────────────────────────────────────────
    def update(t: int):
        # Boundary wave
        line_bdy.set_data(x_axis, bdy_raw[t])
        time_text.set_text(f"t = {t + 1}/{T}")

        # Predicted bulk heatmap
        im_pred.set_data(bulk_pred[t])

        # Error heatmap
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

    # ── Summary metrics ──────────────────────────────────────────────────
    overall_mae = np.mean(error)
    overall_max = np.max(error)
    data_range  = bulk_vmax - bulk_vmin
    mae_pct = (overall_mae / (data_range + 1e-8)) * 100
    max_pct = (overall_max / (data_range + 1e-8)) * 100

    print(f"\n📊 Spatiotemporal Evaluation Metrics")
    print(f"   Overall MAE       : {overall_mae:.6f}  ({mae_pct:.2f}% of range)")
    print(f"   Overall Max Error : {overall_max:.6f}  ({max_pct:.2f}% of range)")
    print(f"   Time steps        : {T}")
    print(f"   Spatial grid      : {X}×{Z}")
    print(f"\n🎬 Animation saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    x_input, bdy_raw, bulk_true, y_mean, y_std, idx = load_random_sample(
        args.data_dir
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_and_load_model(
        args.model_path, device,
        modes=args.modes, width=args.width, n_layers=args.n_layers,
    )

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n⚡ Running spatiotemporal inference …")
    bulk_pred = predict(model, x_input, device, y_mean, y_std)
    print(f"   Prediction shape: {bulk_pred.shape}")

    # ── Animation ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)

    print(f"\n🎨 Rendering {bulk_true.shape[0]}-frame animation …")
    create_animation(
        bdy_raw, bulk_true, bulk_pred,
        save_path, fps=args.fps, dpi=args.dpi,
    )

    print("\n✅ Phase 2 evaluation complete.")


if __name__ == "__main__":
    main()
