"""
evaluate_unified.py — Evaluation & Visualisation for Unified Neural-AdS (Phase 1)
===================================================================================

Loads the trained FNO2d model from ``models/unified_neural_ads_final.pth``,
picks a single random test sample from the holographic dataset, runs
inference in eval mode, and generates a publication-quality 1×3 comparison
figure:

    [Ground Truth]  |  [Neural-AdS Prediction]  |  [Absolute Error Heatmap]

The figure is saved to ``results/hologram_comparison.png``.

Usage
-----
    python evaluate_unified.py
    python evaluate_unified.py --model_path models/unified_neural_ads_final.pth
    python evaluate_unified.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Local imports -----------------------------------------------------------
# The FNO2d architecture from our unified codebase (channel-first convention).
from fno_architectures import FNO2d


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified Neural-AdS — evaluation & visualisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", type=str,
                    default=os.path.join("models", "unified_neural_ads_final.pth"),
                    help="Path to the trained FNO2d state dict")
    p.add_argument("--data_dir", type=str, default="data_holography",
                    help="Directory containing boundary_train.npy & bulk_train.npy")
    p.add_argument("--grid_size", type=int, default=64,
                    help="Spatial resolution of the 2-D grid")
    p.add_argument("--modes", type=int, default=12,
                    help="Number of Fourier modes retained per axis")
    p.add_argument("--width", type=int, default=32,
                    help="Hidden channel width of the FNO")
    p.add_argument("--n_layers", type=int, default=4,
                    help="Number of Fourier layers")
    p.add_argument("--output_dir", type=str, default="results",
                    help="Directory to save the output figure")
    p.add_argument("--output_name", type=str, default="hologram_comparison.png",
                    help="Filename for the output figure")
    p.add_argument("--dpi", type=int, default=300,
                    help="DPI for the saved figure")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_random_sample(
    data_dir: str,
    grid_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a single random sample and construct the 2-channel input tensor.

    Returns
    -------
    x_input : ``(1, 2, H, W)`` — model-ready input (normalised, float32).
    y_truth : ``(H, W)``       — ground-truth bulk field (original scale).
    idx     : int              — index of the selected sample.
    """
    boundary_raw = np.load(
        os.path.join(data_dir, "boundary_train.npy")
    ).astype(np.float32)                                     # (N, grid_size)
    bulk_raw = np.load(
        os.path.join(data_dir, "bulk_train.npy")
    ).astype(np.float32)                                     # (N, H, W)

    N = boundary_raw.shape[0]
    idx = np.random.randint(0, N)
    print(f"📌 Selected sample index: {idx}  (out of {N})")

    wave_1d = boundary_raw[idx]                              # (grid_size,)
    y_truth = bulk_raw[idx]                                  # (H, W)

    # ── Normalisation (same stats used during training) ──────────────────
    x_mean, x_std = boundary_raw.mean(), boundary_raw.std()
    wave_norm = (wave_1d - x_mean) / (x_std + 1e-8)

    # ── Channel 0: boundary wave tiled to (H, W) ────────────────────────
    wave_2d = np.tile(wave_norm[np.newaxis, :], (grid_size, 1))  # (H, W)

    # ── Channel 1: normalised Y-depth coordinate grid ────────────────────
    depth_col = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    depth_2d = np.tile(depth_col[:, np.newaxis], (1, grid_size))  # (H, W)

    # ── Stack → (1, 2, H, W) ────────────────────────────────────────────
    x_input = np.stack([wave_2d, depth_2d], axis=0)[np.newaxis, ...]

    return x_input, y_truth, idx


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def build_and_load_model(
    model_path: str,
    device: torch.device,
    *,
    modes: int = 12,
    width: int = 32,
    n_layers: int = 4,
) -> FNO2d:
    """Instantiate FNO2d and load the saved state dictionary."""
    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        n_layers=n_layers,
        in_channels=2,                  # wave + depth
    ).to(device)

    if not os.path.isfile(model_path):
        print(f"❌ Model checkpoint not found at: {model_path}")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 FNO2d loaded — {total_params:,} parameters  ({model_path})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: FNO2d,
    x_input: np.ndarray,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    """Run forward pass and denormalise the output back to physical scale.

    Parameters
    ----------
    x_input : ``(1, 2, H, W)``  normalised input.
    y_mean, y_std : bulk normalisation statistics from the full dataset.

    Returns
    -------
    prediction : ``(H, W)`` — predicted bulk field in original scale.
    """
    x_tensor = torch.from_numpy(x_input).to(device)
    pred_norm = model(x_tensor)                             # (1, 1, H, W)
    pred_norm = pred_norm.squeeze().cpu().numpy()            # (H, W)

    # Denormalise
    prediction = pred_norm * (y_std + 1e-8) + y_mean
    return prediction


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _add_colorbar(im, ax: plt.Axes) -> None:
    """Attach a neat, size-matched colorbar to *ax*."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.12)
    plt.colorbar(im, cax=cax)


def visualise(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    save_path: str,
    dpi: int = 300,
) -> None:
    """Create and save the 1×3 comparison figure.

    Panels
    ------
    0 — Ground Truth Bulk
    1 — Neural-AdS Predicted Bulk
    2 — Absolute Error Heatmap
    """
    error = np.abs(ground_truth - prediction)

    mae = np.mean(error)
    max_err = np.max(error)
    data_range = np.max(ground_truth) - np.min(ground_truth)
    mae_pct = (mae / (data_range + 1e-8)) * 100
    max_pct = (max_err / (data_range + 1e-8)) * 100

    # ── Shared colour limits for the first two panels ────────────────────
    vmin = min(ground_truth.min(), prediction.min())
    vmax = max(ground_truth.max(), prediction.max())

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # --- Panel 0: Ground Truth -----------------------------------------------
    im0 = axes[0].imshow(ground_truth, cmap="inferno", vmin=vmin, vmax=vmax,
                          origin="upper", aspect="equal")
    axes[0].set_title("Ground Truth Bulk", fontsize=14, pad=14, fontweight="bold")
    axes[0].set_xlabel("Boundary Coordinate (x)")
    axes[0].set_ylabel("Radial Depth (z)")
    _add_colorbar(im0, axes[0])

    # --- Panel 1: Neural-AdS Prediction --------------------------------------
    im1 = axes[1].imshow(prediction, cmap="inferno", vmin=vmin, vmax=vmax,
                          origin="upper", aspect="equal")
    axes[1].set_title(f"Neural-AdS Prediction  (MAE: {mae_pct:.2f}%)",
                      fontsize=14, pad=14, fontweight="bold")
    axes[1].set_xlabel("Boundary Coordinate (x)")
    axes[1].set_ylabel("Radial Depth (z)")
    _add_colorbar(im1, axes[1])

    # --- Panel 2: Absolute Error Heatmap -------------------------------------
    im2 = axes[2].imshow(error, cmap="hot", vmin=0, vmax=max_err,
                          origin="upper", aspect="equal")
    axes[2].set_title(f"Absolute Error  (Peak: {max_pct:.2f}%)",
                      fontsize=14, pad=14, fontweight="bold")
    axes[2].set_xlabel("Boundary Coordinate (x)")
    axes[2].set_ylabel("Radial Depth (z)")
    _add_colorbar(im2, axes[2])

    plt.tight_layout(w_pad=4.0)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n📊 Evaluation Metrics")
    print(f"   MAE             : {mae:.6f}  ({mae_pct:.2f}% of data range)")
    print(f"   Max Abs Error   : {max_err:.6f}  ({max_pct:.2f}% of data range)")
    print(f"   Data Range      : {data_range:.6f}")
    print(f"\n🖼️  Figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    x_input, y_truth, idx = load_random_sample(args.data_dir, args.grid_size)

    # Compute bulk normalisation stats from the FULL training set (required
    # to invert the normalisation applied during training).
    bulk_all = np.load(
        os.path.join(args.data_dir, "bulk_train.npy")
    ).astype(np.float32)
    y_mean, y_std = bulk_all.mean(), bulk_all.std()
    del bulk_all                                             # free memory

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_and_load_model(
        args.model_path,
        device,
        modes=args.modes,
        width=args.width,
        n_layers=args.n_layers,
    )

    # ── Inference ─────────────────────────────────────────────────────────
    prediction = predict(model, x_input, device, y_mean, y_std)

    # ── Visualisation ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)
    visualise(y_truth, prediction, save_path, dpi=args.dpi)

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
