"""
finetune_publication.py  —  Publication-Grade Collision Fine-Tuning
====================================================================

Fine-tunes FNO3d on the 5 000-sample collision dataset with:
  * 80/20 train / validation split  (4 000 + 1 000)
  * Combined loss:  L_data  +  lambda * L_pde   (Laplacian residual)
  * Cosine-annealing learning-rate schedule
  * Per-epoch logging to CSV  (for convergence plots)
  * Best-model checkpoint saved by validation Relative-L2

Upload to Colab
----------------
1. Upload this script + ``fno_architectures.py`` + ``data_collision_5k/``
   + ``models/unified_time_final.pth``
2. ``!pip install torch numpy scipy tqdm``  (already installed on Colab)
3. ``!python finetune_publication.py``

Outputs
-------
    models/collision_publication.pth     — best checkpoint (lowest val Rel L2)
    results/training_log.csv            — per-epoch metrics
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from fno_architectures import FNO3d


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publication-grade collision fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir", type=str, default="data_collision_5k")
    p.add_argument("--pretrained", type=str,
                    default=os.path.join("models", "unified_time_final.pth"),
                    help="Phase 2 checkpoint to initialise from")

    # Architecture (must match pretrained)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--n_layers", type=int, default=4)

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6,
                    help="Minimum LR for cosine schedule")
    p.add_argument("--pde_lambda", type=float, default=0.1,
                    help="Weight of PDE residual loss")
    p.add_argument("--val_split", type=float, default=0.20)
    p.add_argument("--val_every", type=int, default=10,
                    help="Compute full validation metrics every N epochs")

    # Output
    p.add_argument("--save_path", type=str,
                    default=os.path.join("models", "collision_publication.pth"))
    p.add_argument("--log_path", type=str,
                    default=os.path.join("results", "training_log.csv"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data  (LAZY — only raw arrays in RAM, 3-channel input built per-batch)
# ─────────────────────────────────────────────────────────────────────────────

def load_collision_data(data_dir: str, val_split: float, seed: int):
    """Load raw arrays and compute norm stats. No 3D expansion yet."""
    bdy  = np.load(os.path.join(data_dir, "bdy_collision.npy")).astype(np.float32)
    bulk = np.load(os.path.join(data_dir, "bulk_collision.npy")).astype(np.float32)

    N, T, X = bdy.shape
    Z = bulk.shape[3]
    print(f"   Dataset: {N} samples, {T} time-steps, {X}x{Z} grid")

    stats = dict(
        x_mean=float(bdy.mean()),  x_std=float(bdy.std()),
        y_mean=float(bulk.mean()), y_std=float(bulk.std()),
    )
    print(f"   x_mean={stats['x_mean']:.4f}  x_std={stats['x_std']:.4f}")
    print(f"   y_mean={stats['y_mean']:.4f}  y_std={stats['y_std']:.4f}")

    # Normalise in-place to save memory
    bdy  = (bdy  - stats["x_mean"]) / (stats["x_std"]  + 1e-8)
    bulk = (bulk - stats["y_mean"]) / (stats["y_std"] + 1e-8)

    # Pre-compute coordinate grids (tiny: T and Z floats)
    t_coord = np.linspace(0, 1, T, dtype=np.float32)
    z_coord = np.linspace(0, 1, Z, dtype=np.float32)

    # Shuffle and split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_val = max(1, int(N * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    bdy_mb  = bdy.nbytes  / 1024**2
    bulk_mb = bulk.nbytes / 1024**2
    print(f"   RAM: bdy={bdy_mb:.0f} MB  bulk={bulk_mb:.0f} MB  "
          f"(no 3D expansion)")
    print(f"   Train: {len(train_idx)}  Val: {len(val_idx)}")

    return bdy, bulk, t_coord, z_coord, train_idx, val_idx, stats


def make_batch(bdy_n, bulk_n, t_coord, z_coord, indices):
    """Construct (B, 3, T, X, Z) input and (B, 1, T, X, Z) target on-the-fly."""
    B = len(indices)
    T, X = bdy_n.shape[1], bdy_n.shape[2]
    Z = bulk_n.shape[3]

    # Wave channel: (B, T, X) -> (B, T, X, Z)
    wave = np.tile(bdy_n[indices][:, :, :, None], (1, 1, 1, Z))

    # Time channel: broadcast (T,) -> (B, T, X, Z)
    time_ch = np.broadcast_to(t_coord[None, :, None, None],
                               (B, T, X, Z)).copy()

    # Depth channel: broadcast (Z,) -> (B, T, X, Z)
    depth_ch = np.broadcast_to(z_coord[None, None, None, :],
                                (B, T, X, Z)).copy()

    x = np.stack([wave, time_ch, depth_ch], axis=1).astype(np.float32)
    y = bulk_n[indices][:, None].astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


# ─────────────────────────────────────────────────────────────────────────────
# PDE Loss (Laplacian residual in normalised space)
# ─────────────────────────────────────────────────────────────────────────────

class LaplacianLoss(nn.Module):
    """Mean |nabla^2 phi| via 5-point stencil on the interior."""
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        p = pred[:, 0]  # (B, T, X, Z)
        lap = (p[:, :, 2:, 1:-1] + p[:, :, :-2, 1:-1]
             + p[:, :, 1:-1, 2:] + p[:, :, 1:-1, :-2]
             - 4 * p[:, :, 1:-1, 1:-1])
        return lap.abs().mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n   Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────
    print("\n1. Loading data ...")
    bdy_n, bulk_n, t_coord, z_coord, train_idx, val_idx, stats = \
        load_collision_data(args.data_dir, args.val_split, args.seed)

    # ── Model ─────────────────────────────────────────────────────────
    print("\n2. Building FNO3d ...")
    model = FNO3d(modes1=args.modes, modes2=args.modes, modes3=args.modes,
                  width=args.width, n_layers=args.n_layers,
                  in_channels=3).to(device)

    if os.path.isfile(args.pretrained):
        print(f"   Loading pretrained weights: {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    else:
        print(f"   WARNING: pretrained not found, training from scratch!")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # ── Optimiser + Scheduler ─────────────────────────────────────────
    optimiser = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs,
                                   eta_min=args.lr_min)
    mse_fn = nn.MSELoss()
    pde_fn = LaplacianLoss()

    # ── Logging ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    log_file = open(args.log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "train_loss", "train_mse", "train_pde", "lr",
        "val_loss", "val_mse", "val_pde", "val_rel_l2",
    ])

    best_val_rl2 = float("inf")
    B = args.batch_size
    n_train = len(train_idx)

    print(f"\n3. Training — {args.epochs} epochs, batch={B}, "
          f"lr={args.lr} -> {args.lr_min} (cosine)")
    print(f"   PDE lambda = {args.pde_lambda}")
    print(f"   {'─' * 74}")
    print(f"   {'Epoch':>5} | {'Train Loss':>10} | {'MSE':>10} | "
          f"{'PDE':>10} | {'LR':>10} | {'Val RL2':>10}")
    print(f"   {'─' * 74}")

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_mse = 0.0
        epoch_pde = 0.0
        n_batches = 0

        for start in range(0, n_train, B):
            batch_idx = train_idx[perm[start : start + B]]
            xb, yb = make_batch(bdy_n, bulk_n, t_coord, z_coord, batch_idx)
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss_mse = mse_fn(pred, yb)
            loss_pde = pde_fn(pred)
            loss = loss_mse + args.pde_lambda * loss_pde

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_mse += loss_mse.item()
            epoch_pde += loss_pde.item()
            n_batches += 1

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        avg_mse = epoch_mse / n_batches
        avg_pde = epoch_pde / n_batches
        avg_loss = avg_mse + args.pde_lambda * avg_pde

        # ── Validation ────────────────────────────────────────────────
        val_loss_str = ""
        val_mse, val_pde, val_rl2 = "", "", ""

        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)

        if do_val:
            model.eval()
            with torch.no_grad():
                v_mse_sum = 0.0
                v_pde_sum = 0.0
                v_l2_num  = 0.0
                v_l2_den  = 0.0
                v_batches = 0

                for vs in range(0, len(val_idx), B):
                    vi = val_idx[vs : vs + B]
                    xv, yv = make_batch(bdy_n, bulk_n, t_coord, z_coord, vi)
                    xv, yv = xv.to(device), yv.to(device)
                    pv = model(xv)

                    v_mse_sum += mse_fn(pv, yv).item()
                    v_pde_sum += pde_fn(pv).item()
                    v_l2_num  += torch.norm(pv - yv).item() ** 2
                    v_l2_den  += torch.norm(yv).item() ** 2
                    v_batches += 1

                val_mse = v_mse_sum / v_batches
                val_pde = v_pde_sum / v_batches
                val_rl2 = math.sqrt(v_l2_num / (v_l2_den + 1e-12))
                val_loss = val_mse + args.pde_lambda * val_pde

                if val_rl2 < best_val_rl2:
                    best_val_rl2 = val_rl2
                    torch.save(model.state_dict(), args.save_path)

                val_loss_str = f"{val_rl2:.6f}"
        else:
            val_loss, val_loss_str = "", ""

        # Log every epoch
        log_writer.writerow([
            epoch,
            f"{avg_loss:.8f}", f"{avg_mse:.8f}", f"{avg_pde:.8f}",
            f"{lr_now:.8f}",
            f"{val_loss:.8f}" if do_val else "",
            f"{val_mse:.8f}" if do_val else "",
            f"{val_pde:.8f}" if do_val else "",
            f"{val_rl2:.8f}" if do_val else "",
        ])
        log_file.flush()

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            vr = val_loss_str if val_loss_str else "   —   "
            print(f"   {epoch:5d} | {avg_loss:10.6f} | {avg_mse:10.6f} | "
                  f"{avg_pde:10.6f} | {lr_now:10.2e} | {vr:>10}")

    elapsed = time.time() - t0
    log_file.close()

    print(f"   {'─' * 74}")
    print(f"\n   Training complete in {elapsed/60:.1f} min")
    print(f"   Best val Rel L2: {best_val_rl2:.6f}")
    print(f"   Model saved:     {args.save_path}")
    print(f"   Training log:    {args.log_path}")


if __name__ == "__main__":
    train(parse_args())
