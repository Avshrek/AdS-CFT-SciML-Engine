"""
train_time_unified.py — Phase 2 PINO Training: Spatiotemporal Bulk Evolution
=============================================================================

AdS/CFT Physics Context
------------------------
Phase 1 learned the *static* holographic map  Φ_bdy(x) → Φ_bulk(x, z).
Phase 2 promotes this to a *time-dependent* (2+1 D) map:

    Φ_bdy(x, t)  →  Φ_bulk(x, z, t)         via FNO3d

The model ingests a 3-channel volume ``(B, 3, T, X, Z)``:

    ch-0 : time-evolving boundary wave (tiled over Z-depth)
    ch-1 : normalised temporal coordinate grid
    ch-2 : normalised spatial Z-depth coordinate grid

and predicts the full spatiotemporal bulk ``(B, 1, T, X, Z)``.

Training Losses
~~~~~~~~~~~~~~~
    L_total  =  L_data  +  λ(epoch) · L_PDE

* **L_data** (MSE) — drives predictions toward the ground-truth bulk.
* **L_PDE** (quasi-static Laplacian) — for every time slice t_k, enforces
  ∇²_{xz} Φ(·, t_k) = 0 using the 2-D finite-difference stencil.  The time
  axis is *not* differentiated (adiabatic approximation).
* **λ(epoch)** — ramped 0 → λ_target via ``PINOScheduler``.

Usage
-----
    python train_time_unified.py
    python train_time_unified.py --epochs 200 --lr 3e-3 --batch_size 4
    python train_time_unified.py --help
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
from physics_engine import PINOScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Quasi-static Laplacian loss for (B, 1, T, X, Z) predictions
# ─────────────────────────────────────────────────────────────────────────────

class QuasiStaticLaplacianLoss(nn.Module):
    """Enforce ∇²_{xz} Φ = 0 independently at every time slice.

    For a prediction of shape ``(B, 1, T, X, Z)`` we reshape to
    ``(B·T, 1, X, Z)`` and apply the standard 2-D 5-point stencil, then
    compute MSE over interior grid points.  The time axis is never
    differentiated — this is the adiabatic (quasi-static) regime.

    Parameters
    ----------
    dx : float
        Grid spacing (assumed uniform in both spatial directions).
    """

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
        """
        Parameters
        ----------
        u_pred : ``(B, 1, T, X, Z)``

        Returns
        -------
        Scalar tensor — mean squared Laplacian residual over all time slices.
        """
        B, C, T, X, Z = u_pred.shape

        # Collapse batch and time → treat each (X, Z) slice independently
        u_flat = u_pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, X, Z)

        laplacian = F.conv2d(u_flat, self.kernel, padding=1)

        # Crop 1-pixel boundary ring (Dirichlet BCs, not part of residual)
        interior = laplacian[:, :, 1:-1, 1:-1]

        return torch.mean(interior ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 — Spatiotemporal PINO training (FNO3d).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir", type=str, default="data_holography_time",
                    help="Directory containing boundary_time.npy & bulk_time.npy")

    # Model
    p.add_argument("--modes", type=int, default=8,
                    help="Number of Fourier modes per axis (T, X, Z)")
    p.add_argument("--width", type=int, default=20,
                    help="Hidden channel width of FNO3d")
    p.add_argument("--n_layers", type=int, default=4,
                    help="Number of 3-D Fourier layers")

    # Training
    p.add_argument("--epochs", type=int, default=150,
                    help="Total training epochs")
    p.add_argument("--batch_size", type=int, default=4,
                    help="Mini-batch size (keep small for 3-D tensors)")
    p.add_argument("--lr", type=float, default=3e-3,
                    help="Peak learning rate for OneCycleLR")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                    help="AdamW weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0,
                    help="Max gradient norm for clipping")

    # PINO
    p.add_argument("--lambda_target", type=float, default=0.05,
                    help="Asymptotic PDE-loss weight λ")
    p.add_argument("--ramp_epochs", type=int, default=40,
                    help="Epochs over which to ramp λ from 0 → target")
    p.add_argument("--ramp_strategy", type=str, default="cosine",
                    choices=["linear", "cosine"],
                    help="λ ramp-up curve shape")

    # I/O
    p.add_argument("--save_dir", type=str, default="models",
                    help="Directory for checkpoints")
    p.add_argument("--save_every", type=int, default=50,
                    help="Save a checkpoint every N epochs")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    data_dir: str,
    grid_size: int = 64,
    time_steps: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load, normalise, and construct the 3-channel spatiotemporal input.

    Raw shapes from disk
    --------------------
    boundary_time.npy  →  (N, T, X)
    bulk_time.npy      →  (N, T, X, Z)

    Constructed input  ``(N, 3, T, X, Z)``
    ----------------------------------------
    ch-0 : boundary wave tiled across Z-depth
    ch-1 : normalised temporal coordinate  (0 → 1 over T)
    ch-2 : normalised spatial Z-depth coordinate (0 → 1 over Z)

    Target  ``(N, 1, T, X, Z)``
    """
    bdy_raw = np.load(
        os.path.join(data_dir, "boundary_time.npy")
    ).astype(np.float32)                                     # (N, T, X)
    bulk_raw = np.load(
        os.path.join(data_dir, "bulk_time.npy")
    ).astype(np.float32)                                     # (N, T, X, Z)

    N = bdy_raw.shape[0]
    T = bdy_raw.shape[1]
    X = bdy_raw.shape[2]
    Z = bulk_raw.shape[3]

    # ── Normalisation (full-dataset statistics) ──────────────────────────
    x_mean, x_std = bdy_raw.mean(), bdy_raw.std()
    y_mean, y_std = bulk_raw.mean(), bulk_raw.std()

    bdy_norm  = (bdy_raw  - x_mean) / (x_std + 1e-8)
    bulk_norm = (bulk_raw - y_mean) / (y_std + 1e-8)

    # ── Channel 0: boundary wave tiled over Z-depth ──────────────────────
    #    (N, T, X) → (N, T, X, 1) → tile → (N, T, X, Z)
    wave_3d = np.tile(bdy_norm[:, :, :, np.newaxis], (1, 1, 1, Z))

    # ── Channel 1: normalised time coordinate grid ───────────────────────
    #    shape (T,) → broadcast to (N, T, X, Z)
    t_coord = np.linspace(0.0, 1.0, T, dtype=np.float32)
    time_3d = np.broadcast_to(
        t_coord[np.newaxis, :, np.newaxis, np.newaxis],
        (N, T, X, Z),
    ).copy()

    # ── Channel 2: normalised Z-depth coordinate grid ────────────────────
    #    shape (Z,) → broadcast to (N, T, X, Z)
    z_coord = np.linspace(0.0, 1.0, Z, dtype=np.float32)
    depth_3d = np.broadcast_to(
        z_coord[np.newaxis, np.newaxis, np.newaxis, :],
        (N, T, X, Z),
    ).copy()

    # ── Stack → (N, 3, T, X, Z) ─────────────────────────────────────────
    x_input = np.stack([wave_3d, time_3d, depth_3d], axis=1)
    y_target = bulk_norm[:, np.newaxis, :, :, :]              # (N, 1, T, X, Z)

    x_train = torch.from_numpy(x_input)
    y_train = torch.from_numpy(y_target)

    norm_stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    return x_train, y_train, norm_stats


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}")
    print(f"   λ ramp: 0 → {args.lambda_target} over {args.ramp_epochs} epochs "
          f"({args.ramp_strategy})\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("📦 Loading spatiotemporal dataset …")
    x_train, y_train, norm_stats = load_data(args.data_dir)
    N, C, T, X, Z = x_train.shape
    print(f"   x_train : {tuple(x_train.shape)}   (3-ch: wave | time | depth)")
    print(f"   y_train : {tuple(y_train.shape)}")
    print(f"   Norm    : x̄={norm_stats['x_mean']:.4f}  σx={norm_stats['x_std']:.4f}  "
          f"ȳ={norm_stats['y_mean']:.4f}  σy={norm_stats['y_std']:.4f}\n")

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = FNO3d(
        modes1=args.modes,
        modes2=args.modes,
        modes3=args.modes,
        width=args.width,
        n_layers=args.n_layers,
        in_channels=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 FNO3d — {total_params:,} parameters\n")

    # ── Losses & schedulers ───────────────────────────────────────────────
    data_criterion = nn.MSELoss()
    pde_criterion  = QuasiStaticLaplacianLoss(dx=1.0).to(device)
    pino_scheduler = PINOScheduler(
        lambda_target=args.lambda_target,
        ramp_epochs=args.ramp_epochs,
        strategy=args.ramp_strategy,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(loader),
        epochs=args.epochs,
    )

    # ── Checkpoint directory ──────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"{'Epoch':>6} | {'L_data':>10} | {'L_PDE':>10} | {'λ':>8} | "
          f"{'L_total':>10} | {'LR':>10} | {'Time':>7}")
    print("─" * 82)

    for epoch in range(args.epochs):
        model.train()
        lam = pino_scheduler.get_lambda(epoch)

        epoch_data_loss = 0.0
        epoch_pde_loss  = 0.0
        n_batches = 0
        t0 = time.perf_counter()

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(x_batch)                          # (B, 1, T, X, Z)

            # ── Combined loss ─────────────────────────────────────────────
            loss_data = data_criterion(pred, y_batch)
            loss_pde  = pde_criterion(pred)
            loss_total = loss_data + lam * loss_pde

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            lr_scheduler.step()

            epoch_data_loss += loss_data.item()
            epoch_pde_loss  += loss_pde.item()
            n_batches += 1

        # ── Epoch summary ────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        avg_data  = epoch_data_loss / n_batches
        avg_pde   = epoch_pde_loss  / n_batches
        avg_total = avg_data + lam * avg_pde
        current_lr = lr_scheduler.get_last_lr()[0]

        print(f"{epoch + 1:>6d} | {avg_data:>10.6f} | {avg_pde:>10.6f} | "
              f"{lam:>8.4f} | {avg_total:>10.6f} | {current_lr:>10.2e} | "
              f"{elapsed:>6.1f}s")

        # ── Checkpointing ────────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"unified_time_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"   💾 Checkpoint → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "unified_time_final.pth")
    torch.save(model.state_dict(), final_path)

    print(f"\n✅ Phase 2 training complete.  Final model → {final_path}")
    print(f"   Total parameters : {total_params:,}")
    print(f"   Final L_data     : {avg_data:.6f}")
    print(f"   Final L_PDE      : {avg_pde:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
