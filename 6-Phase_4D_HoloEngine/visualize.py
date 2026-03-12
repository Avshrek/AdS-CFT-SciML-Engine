"""
visualize.py  –  Post-training visualisation & diagnostic toolkit
=================================================================

Generates:
    1. Loss-curve plots  (Phase A -> Phase B transition visible).
    2. Boundary phi_renorm reconstruction vs ground-truth collision data.
    3. Bulk radial slices  phi_renorm(u)  at several (t, x, y).
    4. Time evolution animation of the boundary scalar field.
    5. Bulk cross-section heatmaps  phi(x, y) at fixed (t, u) slices.
    6. Peak amplitude chirp plot  |phi|_max(t).

Usage:
    python visualize.py --checkpoint checkpoints/best_model.pt
    python visualize.py --checkpoint checkpoints/best_model.pt --device cuda
"""

import argparse
import os
import math
import torch
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[visualize] matplotlib not found – skipping plots.")

from config import Config
from data import load_data
from model import ConvEncoder3D, FiLMSiren


def load_models(ckpt_path: str, config: type = Config, device: str = "cpu"):
    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = ConvEncoder3D(config).to(device)
    siren   = FiLMSiren(config).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    siren.load_state_dict(ckpt["siren"])
    encoder.eval()
    siren.eval()
    return encoder, siren, ckpt


# ---------------------------------------------------------------------- #
#  1. Loss curves                                                          #
# ---------------------------------------------------------------------- #
def plot_loss_curves(history: list, save_dir: str = "plots"):
    if not HAS_MPL:
        return
    os.makedirs(save_dir, exist_ok=True)

    keys = [k for k in history[0].keys() if k != "warmup"]
    epochs = range(1, len(history) + 1)

    n_keys = len(keys)
    cols = min(n_keys, 4)
    rows = math.ceil(n_keys / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 3.5 * rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, k in enumerate(keys):
        ax = axes_flat[i]
        vals = [h.get(k, 0.0) for h in history]
        ax.semilogy(list(epochs), vals, linewidth=0.7)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.axvline(Config.CURRICULUM_PHASE_A_EPOCHS, color="red",
                    linestyle="--", linewidth=0.5, label="Phase B start")
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(n_keys, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[vis] Saved {path}")


# ---------------------------------------------------------------------- #
#  2. Boundary reconstruction                                              #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def plot_boundary_reconstruction(encoder, siren, config=Config,
                                  device="cpu", save_dir="plots"):
    if not HAS_MPL:
        return
    os.makedirs(save_dir, exist_ok=True)

    cnn_vol, _, _, _, _, source = load_data(config)
    if source == "master":
        bnd_input = cnn_vol
        while bnd_input.dim() < 5:
            bnd_input = bnd_input.unsqueeze(0)
        bnd_input = bnd_input.to(device)
    else:
        bnd_input = cnn_vol.unsqueeze(0).unsqueeze(0).to(device)

    z_lat = encoder(bnd_input)

    # Reconstruct on a grid (use synthetic data rendering for GT frames)
    from data import generate_synthetic_data
    bnd_data, _, _, _, _ = generate_synthetic_data(config)
    T, H, W = bnd_data.shape

    t_vals = torch.linspace(config.T_RANGE[0], config.T_RANGE[1], T)
    x_vals = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], W)
    y_vals = torch.linspace(config.Y_RANGE[0], config.Y_RANGE[1], H)

    frames_to_show = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    fig, axes = plt.subplots(2, len(frames_to_show),
                             figsize=(3.5 * len(frames_to_show), 7))

    for j, fi in enumerate(frames_to_show):
        t_val = t_vals[fi].item()
        yy, xx = torch.meshgrid(y_vals, x_vals, indexing="ij")
        coords = torch.stack([
            torch.full_like(xx.flatten(), t_val),
            xx.flatten(),
            yy.flatten(),
            torch.full((H * W,), config.U_BOUNDARY),
        ], dim=-1).to(device)

        z_exp = z_lat.expand(coords.shape[0], -1)
        phi_pred = siren(coords, z_exp).cpu().reshape(H, W).numpy()
        gt = bnd_data[fi].numpy()

        axes[0, j].imshow(gt, cmap="inferno", origin="lower",
                          extent=[-1, 1, -1, 1])
        axes[0, j].set_title(f"GT  t={t_val:.2f}", fontsize=9)
        axes[1, j].imshow(phi_pred, cmap="inferno", origin="lower",
                          extent=[-1, 1, -1, 1])
        axes[1, j].set_title(f"Pred  t={t_val:.2f}", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "boundary_reconstruction.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[vis] Saved {path}")


# ---------------------------------------------------------------------- #
#  3. Radial bulk slices  phi_renorm(u)                                    #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def plot_radial_slices(encoder, siren, bnd_input, config=Config,
                       device="cpu", save_dir="plots"):
    if not HAS_MPL:
        return
    os.makedirs(save_dir, exist_ok=True)

    z_lat = encoder(bnd_input)

    u_vals = torch.linspace(config.U_MIN, config.U_MAX, 200)
    test_points = [
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for (t_v, x_v, y_v) in test_points:
        coords = torch.stack([
            torch.full((200,), t_v),
            torch.full((200,), x_v),
            torch.full((200,), y_v),
            u_vals,
        ], dim=-1).to(device)

        z_exp = z_lat.expand(200, -1)
        phi = siren(coords, z_exp).cpu().squeeze().numpy()
        ax.plot(u_vals.numpy(), phi, label=f"t={t_v}")

    ax.set_xlabel("u = ln(z)")
    ax.set_ylabel("phi_renorm")
    ax.set_title("Radial bulk profile  (x=0, y=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "radial_slices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[vis] Saved {path}")


# ---------------------------------------------------------------------- #
#  4. Time-evolution animation                                             #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def animate_boundary(encoder, siren, bnd_input, config=Config,
                     device="cpu", save_dir="plots"):
    if not HAS_MPL:
        return
    os.makedirs(save_dir, exist_ok=True)

    z_lat = encoder(bnd_input)

    T = min(config.ENCODER_TEMPORAL_FRAMES, 50)
    H = W = min(config.ENCODER_SPATIAL_RES, 64)
    t_vals = torch.linspace(config.T_RANGE[0], config.T_RANGE[1], T)
    x_vals = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], W)
    y_vals = torch.linspace(config.Y_RANGE[0], config.Y_RANGE[1], H)
    yy, xx = torch.meshgrid(y_vals, x_vals, indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 5))

    frames_data = []
    for fi in range(T):
        t_val = t_vals[fi].item()
        coords = torch.stack([
            torch.full_like(xx.flatten(), t_val),
            xx.flatten(), yy.flatten(),
            torch.full((H * W,), config.U_BOUNDARY),
        ], dim=-1).to(device)
        z_exp = z_lat.expand(coords.shape[0], -1)
        pred = siren(coords, z_exp).cpu().reshape(H, W).numpy()
        frames_data.append((pred, t_val))

    vmin = min(f[0].min() for f in frames_data)
    vmax = max(f[0].max() for f in frames_data)

    im = ax.imshow(frames_data[0][0], cmap="inferno", origin="lower",
                    extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="phi_renorm")
    ax.set_title(f"Boundary t={frames_data[0][1]:.2f}")

    def _update(i):
        pred, t_v = frames_data[i]
        im.set_data(pred)
        ax.set_title(f"Boundary  t={t_v:.2f}")
        return [im]

    anim = animation.FuncAnimation(fig, _update, frames=T, interval=200, blit=True)
    path = os.path.join(save_dir, "boundary_evolution.gif")
    anim.save(path, writer="pillow", fps=5)
    plt.close()
    print(f"[vis] Saved {path}")


# ---------------------------------------------------------------------- #
#  5. Bulk cross-section heatmaps                                          #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def plot_bulk_cross_sections(encoder, siren, bnd_input, config=Config,
                              device="cpu", save_dir="plots"):
    """phi(x, y) heatmap at fixed t and several u depths."""
    if not HAS_MPL:
        return
    os.makedirs(save_dir, exist_ok=True)

    z_lat = encoder(bnd_input)

    N = 64  # grid resolution
    x_vals = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], N)
    y_vals = torch.linspace(config.Y_RANGE[0], config.Y_RANGE[1], N)
    yy, xx = torch.meshgrid(y_vals, x_vals, indexing="ij")
    xy_flat = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (N^2, 2)

    t_slices = [0.0, 0.3, 0.5, 0.7, 1.0]
    u_slices = [config.U_BOUNDARY, config.U_MIN + 3.0, -2.0, 0.0]
    u_labels = ["boundary", "near-bnd", "mid-bulk", "deep-bulk"]

    fig, axes = plt.subplots(len(u_slices), len(t_slices),
                             figsize=(3 * len(t_slices), 3 * len(u_slices)))

    for row, (u_val, u_label) in enumerate(zip(u_slices, u_labels)):
        for col, t_val in enumerate(t_slices):
            coords = torch.stack([
                torch.full((N * N,), t_val),
                xy_flat[:, 0],
                xy_flat[:, 1],
                torch.full((N * N,), u_val),
            ], dim=-1).to(device)

            z_exp = z_lat.expand(coords.shape[0], -1)
            phi = siren(coords, z_exp).cpu().reshape(N, N).numpy()

            ax = axes[row, col]
            im = ax.imshow(phi, cmap="RdBu_r", origin="lower",
                          extent=[-1, 1, -1, 1])
            ax.set_title(f"t={t_val} {u_label}", fontsize=7)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(save_dir, "bulk_cross_sections.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[vis] Saved {path}")


# ---------------------------------------------------------------------- #
#  6. Peak amplitude chirp                                                 #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def plot_chirp(encoder, siren, bnd_input, config=Config,
               device="cpu", save_dir="plots"):
    """Peak |phi| at the boundary vs time — the gravitational wave chirp."""
    if not HAS_MPL:
        return
    os.makedirs(save_dir, exist_ok=True)

    z_lat = encoder(bnd_input)

    n_t = 50
    n_spatial = 500
    t_vals = torch.linspace(config.T_RANGE[0], config.T_RANGE[1], n_t)
    peaks = []

    for t_val in t_vals:
        x = torch.rand(n_spatial) * 2.0 - 1.0
        y = torch.rand(n_spatial) * 2.0 - 1.0
        coords = torch.stack([
            torch.full((n_spatial,), t_val.item()),
            x, y,
            torch.full((n_spatial,), config.U_BOUNDARY),
        ], dim=-1).to(device)

        z_exp = z_lat.expand(n_spatial, -1)
        phi = siren(coords, z_exp).squeeze(-1)
        peaks.append(phi.abs().max().item())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_vals.numpy(), peaks, 'b-', linewidth=1.5)
    ax.set_xlabel("time t")
    ax.set_ylabel("max |phi_renorm|")
    ax.set_title("Gravitational Wave Chirp  (boundary peak amplitude)")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='merger')
    ax.legend()

    path = os.path.join(save_dir, "chirp_peak_amplitude.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[vis] Saved {path}")


# ---------------------------------------------------------------------- #
#  CLI entry point                                                         #
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Visualise holographic model")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    encoder, siren, ckpt = load_models(args.checkpoint, Config, args.device)
    history = ckpt.get("history", [])

    # Build encoder input
    cnn_vol, _, _, _, _, source = load_data(Config)
    if source == "master":
        bnd_input = cnn_vol
        while bnd_input.dim() < 5:
            bnd_input = bnd_input.unsqueeze(0)
        bnd_input = bnd_input.to(args.device)
    else:
        bnd_input = cnn_vol.unsqueeze(0).unsqueeze(0).to(args.device)

    if history:
        plot_loss_curves(history)
    plot_boundary_reconstruction(encoder, siren, Config, args.device)
    plot_radial_slices(encoder, siren, bnd_input, Config, args.device)
    plot_bulk_cross_sections(encoder, siren, bnd_input, Config, args.device)
    plot_chirp(encoder, siren, bnd_input, Config, args.device)
    animate_boundary(encoder, siren, bnd_input, Config, args.device)
    print("\n[vis] All plots saved to plots/")


if __name__ == "__main__":
    main()
