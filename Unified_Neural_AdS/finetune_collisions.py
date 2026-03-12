"""
finetune_collisions.py — Fine-Tune FNO3d on Dual-Source Collision Data
=======================================================================

Loads the pre-trained Phase 2 weights (``unified_time_final.pth``) and
fine-tunes on the dual-source collision dataset with a low learning rate
to preserve existing holographic knowledge while adapting to richer
interference patterns.

Loss
~~~~
    L_total = L_MSE  +  0.1 · L_PDE

where L_PDE enforces ∇²_{xz}Φ = 0 per time-slice via the 5-point stencil.

Usage
-----
    python finetune_collisions.py
    python finetune_collisions.py --epochs 100 --lr 5e-5
    python finetune_collisions.py --help
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from fno_architectures import FNO3d


# ─────────────────────────────────────────────────────────────────────────────
# Quasi-static Laplacian PDE loss (same as train_time_unified.py)
# ─────────────────────────────────────────────────────────────────────────────

class QuasiStaticLaplacianLoss(nn.Module):
    """Enforce ∇²_{xz} Φ = 0 at every time slice via conv2d stencil."""

    def __init__(self, dx: float = 1.0) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0,  1.0,  0.0],
             [1.0, -4.0,  1.0],
             [0.0,  1.0,  0.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / (dx ** 2)
        self.register_buffer("kernel", kernel)

    def forward(self, u_pred: torch.Tensor) -> torch.Tensor:
        B, C, T, X, Z = u_pred.shape
        slices = u_pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, X, Z)
        lap = F.conv2d(slices, self.kernel, padding=1)
        interior = lap[:, :, 1:-1, 1:-1]
        return torch.mean(interior ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune FNO3d on dual-source collision data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir", type=str, default="data_collision_master",
                    help="Directory with bdy_collision.npy & bulk_collision.npy")

    # Model
    p.add_argument("--pretrained", type=str,
                    default=os.path.join("models", "unified_time_final.pth"),
                    help="Path to Phase 2 pre-trained weights")
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--n_layers", type=int, default=4)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Low LR to preserve pre-trained knowledge")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_pde", type=float, default=0.1,
                    help="Fixed PDE loss weight (no ramp — model is warm)")

    # I/O
    p.add_argument("--save_dir", type=str, default="models")
    p.add_argument("--save_name", type=str, default="collision_master.pth")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load, normalise, and construct the 3-channel collision input.

    Returns
    -------
    x_train : ``(N, 3, T, X, Z)``
    y_train : ``(N, 1, T, X, Z)``
    stats   : dict with normalisation statistics
    """
    bdy_raw  = np.load(os.path.join(data_dir, "bdy_collision.npy")).astype(np.float32)
    bulk_raw = np.load(os.path.join(data_dir, "bulk_collision.npy")).astype(np.float32)

    N, T, X = bdy_raw.shape
    Z = bulk_raw.shape[3]

    # ── Dataset-specific normalisation ────────────────────────────────────
    x_mean, x_std = float(bdy_raw.mean()), float(bdy_raw.std())
    y_mean, y_std = float(bulk_raw.mean()), float(bulk_raw.std())

    print(f"   Collision dataset statistics:")
    print(f"   Boundary  →  μ = {x_mean:.6f}   σ = {x_std:.6f}")
    print(f"   Bulk      →  μ = {y_mean:.6f}   σ = {y_std:.6f}")

    bdy_norm  = (bdy_raw  - x_mean) / (x_std + 1e-8)
    bulk_norm = (bulk_raw - y_mean) / (y_std + 1e-8)

    # ── Channel 0: boundary wave tiled over Z ─────────────────────────────
    wave_3d = np.tile(bdy_norm[:, :, :, np.newaxis], (1, 1, 1, Z))

    # ── Channel 1: normalised time coordinate ─────────────────────────────
    t_coord = np.linspace(0.0, 1.0, T, dtype=np.float32)
    time_3d = np.broadcast_to(
        t_coord[np.newaxis, :, np.newaxis, np.newaxis], (N, T, X, Z)
    ).copy()

    # ── Channel 2: normalised Z-depth coordinate ─────────────────────────
    z_coord = np.linspace(0.0, 1.0, Z, dtype=np.float32)
    depth_3d = np.broadcast_to(
        z_coord[np.newaxis, np.newaxis, np.newaxis, :], (N, T, X, Z)
    ).copy()

    # ── Stack → (N, 3, T, X, Z) ──────────────────────────────────────────
    x_input  = np.stack([wave_3d, time_3d, depth_3d], axis=1)
    y_target = bulk_norm[:, np.newaxis, :, :, :]

    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    return torch.from_numpy(x_input), torch.from_numpy(y_target), stats


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def finetune(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("📦 Loading collision dataset …")
    x_train, y_train, stats = load_data(args.data_dir)
    print(f"   x_train : {tuple(x_train.shape)}")
    print(f"   y_train : {tuple(y_train.shape)}\n")

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    # ── Model (load pre-trained Phase 2 weights) ─────────────────────────
    model = FNO3d(
        modes1=args.modes, modes2=args.modes, modes3=args.modes,
        width=args.width, n_layers=args.n_layers, in_channels=3,
    ).to(device)

    print(f"🧠 Loading pre-trained weights from {args.pretrained} …")
    state = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(state)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✔ {total_params:,} parameters loaded — ready for fine-tuning.\n")

    # ── Losses ────────────────────────────────────────────────────────────
    mse_loss = nn.MSELoss()
    pde_loss = QuasiStaticLaplacianLoss(dx=1.0).to(device)
    lam = args.lambda_pde
    print(f"   L_total = L_MSE + {lam} · L_PDE  (fixed weight, no ramp)")
    print(f"   LR = {args.lr}  (low rate for knowledge-preserving fine-tune)\n")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── Checkpoint directory ──────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"{'Epoch':>6} | {'L_MSE':>10} | {'L_PDE':>10} | "
          f"{'L_total':>10} | {'Time':>7}")
    print("─" * 58)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_mse  = 0.0
        epoch_pde  = 0.0
        n_batches  = 0
        t0 = time.perf_counter()

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)

            l_mse = mse_loss(pred, yb)
            l_pde = pde_loss(pred)
            l_tot = l_mse + lam * l_pde

            l_tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            epoch_mse += l_mse.item()
            epoch_pde += l_pde.item()
            n_batches += 1

        elapsed  = time.perf_counter() - t0
        avg_mse  = epoch_mse / n_batches
        avg_pde  = epoch_pde / n_batches
        avg_tot  = avg_mse + lam * avg_pde

        print(f"{epoch + 1:>6d} | {avg_mse:>10.6f} | {avg_pde:>10.6f} | "
              f"{avg_tot:>10.6f} | {elapsed:>6.1f}s")

        if avg_tot < best_loss:
            best_loss = avg_tot

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, args.save_name)
    torch.save(model.state_dict(), final_path)

    print(f"\n✅ Fine-tuning complete.")
    print(f"   Best L_total     : {best_loss:.6f}")
    print(f"   Final model      → {final_path}")
    print(f"   Parameters       : {total_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    finetune(parse_args())
