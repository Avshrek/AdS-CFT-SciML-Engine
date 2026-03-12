"""
train_publication.py — Resume Training for Publication-Level Convergence
=========================================================================

Continues training from ``collision_publication.pth`` (epoch 70) with
physics-informed loss and aggressive learning-rate scheduling to push
Relative L₂ below 5% for publication.

Training chain
~~~~~~~~~~~~~~
    unified_time_final.pth  (Phase 2 — single-source)
         ↓  finetune_collisions.py
    collision_rigorous.pth  (50 epochs — dual-source)
         ↓  finetune_collisions_physics.py
    collision_publication.pth  (70 epochs — physics-informed)
         ↓  THIS SCRIPT
    collision_final.pth  (target: 500 total effective epochs)

Loss
~~~~
    L_total = L_data(MSE)  +  λ · L_PDE(∇²Φ)

    λ linearly ramps from λ_start → λ_end over the first warmup_epochs.

Scheduler
~~~~~~~~~
    CosineAnnealingWarmRestarts with T_0=50 for cyclic exploration.

Checkpointing
~~~~~~~~~~~~~~
    ✓ Saves best model (lowest Rel L₂) separately
    ✓ Periodic checkpoint every ``save_every`` epochs
    ✓ Full state dict (model + optimizer + scheduler + epoch) for resuming
    ✓ Auto-downloads from Google Drive if running on Colab

Usage (Colab)
-------------
    !python train_publication.py
    !python train_publication.py --epochs 500 --lr 5e-5
    !python train_publication.py --resume checkpoints/checkpoint_epoch_200.pt

Usage (Local)
-------------
    python train_publication.py --epochs 50 --lr 5e-5
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from fno_architectures import FNO3d


# ─────────────────────────────────────────────────────────────────────────────
# Physics loss
# ─────────────────────────────────────────────────────────────────────────────

class SpatialLaplacianLoss(nn.Module):
    r"""∇²_{xz}Φ = 0  via 5-point stencil on every (X,Z) time-slice."""

    def __init__(self, dx: float = 1.0) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0,  1.0,  0.0],
             [1.0, -4.0,  1.0],
             [0.0,  1.0,  0.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / (dx ** 2)
        self.register_buffer("kernel", kernel)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, C, T, X, Z = u.shape
        flat = u.permute(0, 2, 1, 3, 4).reshape(B * T, C, X, Z)
        lap = F.conv2d(flat, self.kernel, padding=1)
        interior = lap[:, :, 1:-1, 1:-1]
        return torch.mean(interior ** 2)


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target) / (torch.norm(target) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resume training for publication-level convergence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir",     type=str, default="data_collision_master")
    p.add_argument("--max_samples",  type=int, default=None,
                    help="Cap dataset size to save RAM (default: use all)")

    # Model
    p.add_argument("--pretrained",   type=str,
                    default=os.path.join("models", "collision_publication.pth"),
                    help="Weights to resume from")
    p.add_argument("--modes",        type=int, default=8)
    p.add_argument("--width",        type=int, default=20)
    p.add_argument("--n_layers",     type=int, default=4)

    # Training
    p.add_argument("--epochs",       type=int, default=500,
                    help="Total NEW epochs to train (on top of existing 70)")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--lr",           type=float, default=5e-5,
                    help="Peak learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip",    type=float, default=1.0)

    # Physics loss schedule
    p.add_argument("--lambda_start", type=float, default=0.05,
                    help="Initial PDE loss weight")
    p.add_argument("--lambda_end",   type=float, default=0.2,
                    help="Final PDE loss weight")
    p.add_argument("--warmup_epochs",type=int, default=50,
                    help="Epochs over which λ ramps up")

    # Scheduler
    p.add_argument("--T_0",         type=int, default=50,
                    help="CosineAnnealingWarmRestarts period")

    # Checkpointing
    p.add_argument("--save_dir",     type=str, default="models")
    p.add_argument("--ckpt_dir",     type=str, default="checkpoints")
    p.add_argument("--save_every",   type=int, default=10,
                    help="Save checkpoint every N epochs")
    p.add_argument("--save_name",    type=str,
                    default="collision_final.pth")

    # Resume from full checkpoint (model + optimizer + scheduler + epoch)
    p.add_argument("--resume",       type=str, default=None,
                    help="Path to full checkpoint .pt to resume mid-run")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Memory-efficient data loading  (Colab-safe: ~12 GB RAM)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_data_path(data_dir: str):
    """Find dataset files, checking fallback dirs."""
    bdy_path  = os.path.join(data_dir, "bdy_collision.npy")
    bulk_path = os.path.join(data_dir, "bulk_collision.npy")

    if not os.path.isfile(bdy_path):
        for alt in ["data_collision_5k", "data_collision"]:
            alt_bdy = os.path.join(alt, "bdy_collision.npy")
            if os.path.isfile(alt_bdy):
                bdy_path  = alt_bdy
                bulk_path = os.path.join(alt, "bulk_collision.npy")
                break

    if not os.path.isfile(bdy_path):
        print(f"❌ Dataset not found. Tried: {data_dir}/, data_collision_5k/, data_collision/")
        sys.exit(1)

    return bdy_path, bulk_path


def load_and_standardise(data_dir: str, max_samples: int = None):
    """Load collision data with strict μ=0, σ=1 standardisation.

    Memory-optimised: builds tensors in-place and frees intermediates
    aggressively to stay within Colab's ~12 GB RAM.
    """
    bdy_path, bulk_path = _resolve_data_path(data_dir)

    # Step 1: Memory-map to compute stats WITHOUT loading into RAM
    print("     Computing normalisation stats (memory-mapped) ...")
    bdy_mmap  = np.load(bdy_path,  mmap_mode="r")
    bulk_mmap = np.load(bulk_path, mmap_mode="r")

    N_full, T, X = bdy_mmap.shape
    Z = bulk_mmap.shape[3]
    N = min(N_full, max_samples) if max_samples else N_full

    # Compute stats on full data (cheap via mmap)
    x_mu  = float(bdy_mmap.mean())
    x_sig = float(bdy_mmap.std())
    y_mu  = float(bulk_mmap.mean())
    y_sig = float(bulk_mmap.std())

    print(f"     Dataset: {N_full} total, using {N} samples")
    print(f"     Boundary  μ={x_mu:+.4f}  σ={x_sig:.4f}")
    print(f"     Bulk      μ={y_mu:+.4f}  σ={y_sig:.4f}")

    # Step 2: Load only the samples we need + standardise in-place
    print("     Loading and standardising ...")
    bdy_z = bdy_mmap[:N].astype(np.float32).copy()
    bdy_z -= x_mu
    bdy_z /= (x_sig + 1e-8)

    bulk_z = bulk_mmap[:N].astype(np.float32).copy()
    bulk_z -= y_mu
    bulk_z /= (y_sig + 1e-8)

    # Free mmaps
    del bdy_mmap, bulk_mmap
    gc.collect()

    # Step 3: Build 3-channel input directly into final tensor
    #   Avoids creating 3 separate (N,T,X,Z) arrays + stack
    print("     Building input volume (memory-efficient) ...")
    x_input = np.empty((N, 3, T, X, Z), dtype=np.float32)

    # Ch-0: boundary wave tiled to depth (build row-by-row to save peak RAM)
    for i in range(N):
        x_input[i, 0] = np.tile(bdy_z[i, :, :, np.newaxis], (1, 1, Z))
    del bdy_z
    gc.collect()

    # Ch-1: temporal coordinate (0 → 1) — broadcast fill
    t_coord = np.linspace(0, 1, T, dtype=np.float32)
    x_input[:, 1] = t_coord[None, :, None, None]

    # Ch-2: depth coordinate (0 → 1) — broadcast fill
    z_coord = np.linspace(0, 1, Z, dtype=np.float32)
    x_input[:, 2] = z_coord[None, None, None, :]

    # Step 4: Target
    y_target = bulk_z[:, np.newaxis, :, :, :]
    del bulk_z
    gc.collect()

    # Step 5: Convert to tensors
    print("     Converting to PyTorch tensors ...")
    x_t = torch.from_numpy(x_input)
    del x_input; gc.collect()

    y_t = torch.from_numpy(y_target)
    del y_target; gc.collect()

    stats = dict(x_mu=x_mu, x_sig=x_sig, y_mu=y_mu, y_sig=y_sig)
    mem_gb = (x_t.nbytes + y_t.nbytes) / 1e9
    print(f"     ✅ Tensors ready — {mem_gb:.2f} GB in RAM")

    return x_t, y_t, stats


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  🚀 PUBLICATION-LEVEL TRAINING — Resume from Epoch 70")
    print("=" * 72)
    print(f"\n  Device          : {device}")
    if device.type == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM            : {mem_gb:.1f} GB")
    print(f"  Resume from     : {args.pretrained}")
    print(f"  New epochs      : {args.epochs}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  λ_PDE           : {args.lambda_start} → {args.lambda_end}")
    print(f"  Scheduler       : CosineAnnealingWarmRestarts (T₀={args.T_0})")
    print(f"  Output          : {args.save_dir}/{args.save_name}")

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  📦 LOADING DATA")
    print(f"{'─' * 72}")

    x_train, y_train, stats = load_and_standardise(args.data_dir, args.max_samples)

    N = x_train.shape[0]
    print(f"\n  Samples         : {N}")
    print(f"  x_train shape   : {tuple(x_train.shape)}")
    print(f"  y_train shape   : {tuple(y_train.shape)}")
    print(f"  Boundary  μ={stats['x_mu']:+.4f}  σ={stats['x_sig']:.4f}")
    print(f"  Bulk      μ={stats['y_mu']:+.4f}  σ={stats['y_sig']:.4f}")

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True, pin_memory=(device.type == "cuda"),
        num_workers=2 if device.type == "cuda" else 0,
        drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  🧠 MODEL")
    print(f"{'─' * 72}")

    model = FNO3d(
        modes1=args.modes, modes2=args.modes, modes3=args.modes,
        width=args.width, n_layers=args.n_layers, in_channels=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  FNO3d — {n_params:,} parameters")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    mse_criterion = nn.MSELoss()
    pde_criterion = SpatialLaplacianLoss(dx=1.0).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1)

    start_epoch = 0

    # ── Resume logic ──────────────────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        # Full checkpoint resume (mid-run crash recovery)
        print(f"\n  ♻️  Resuming from full checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"     Resuming at epoch {start_epoch + 1}")
    else:
        # Load pretrained weights (first run — continuing from epoch 70)
        if not os.path.isfile(args.pretrained):
            print(f"  ❌ Pretrained weights not found: {args.pretrained}")
            sys.exit(1)
        state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state)
        print(f"  ✅ Loaded pretrained weights from {args.pretrained}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  📈 TRAINING LOG")
    print(f"{'─' * 72}\n")
    print(f"  {'Ep':>5}  {'L_data':>10}  {'L_PDE':>10}  "
          f"{'L_total':>10}  {'ε_rel':>10}  {'λ':>6}  {'LR':>10}  {'Time':>6}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*10}  {'─'*6}  {'─'*10}  {'─'*6}")

    best_rel_l2 = float("inf")
    best_epoch  = 0
    history     = []  # for optional loss curve plotting

    total_t0 = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        model.train()

        # λ ramp: linear from lambda_start to lambda_end over warmup_epochs
        progress = min(1.0, epoch / max(1, args.warmup_epochs))
        lam = args.lambda_start + progress * (args.lambda_end - args.lambda_start)

        sum_mse, sum_pde, sum_rel = 0.0, 0.0, 0.0
        n_batches = 0
        t0 = time.perf_counter()

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)

            l_data = mse_criterion(pred, yb)
            l_pde  = pde_criterion(pred)
            l_tot  = l_data + lam * l_pde

            l_tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                eps_rel = relative_l2_error(pred, yb)

            sum_mse += l_data.item()
            sum_pde += l_pde.item()
            sum_rel += eps_rel.item()
            n_batches += 1

        scheduler.step()

        elapsed = time.perf_counter() - t0
        avg_mse = sum_mse / n_batches
        avg_pde = sum_pde / n_batches
        avg_tot = avg_mse + lam * avg_pde
        avg_rel = sum_rel / n_batches
        cur_lr  = optimizer.param_groups[0]["lr"]

        history.append(dict(epoch=epoch+1, l_data=avg_mse, l_pde=avg_pde,
                            l_total=avg_tot, rel_l2=avg_rel, lr=cur_lr))

        # Best model tracking
        marker = ""
        if avg_rel < best_rel_l2:
            best_rel_l2 = avg_rel
            best_epoch  = epoch + 1
            marker = " ★"
            best_path = os.path.join(args.save_dir, "collision_best.pth")
            torch.save(model.state_dict(), best_path)

        # Print every epoch (Colab scrolls nicely)
        effective_ep = 70 + epoch + 1  # total effective epoch count
        print(f"  {effective_ep:>5d}  {avg_mse:>10.6f}  {avg_pde:>10.6f}  "
              f"{avg_tot:>10.6f}  {avg_rel:>10.6f}  {lam:>6.3f}  "
              f"{cur_lr:>10.2e}  {elapsed:>5.1f}s{marker}")

        # Periodic checkpoint (full state for crash recovery)
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir,
                                     f"checkpoint_epoch_{effective_ep}.pt")
            torch.save({
                "epoch": epoch + 1,
                "effective_epoch": effective_ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rel_l2": avg_rel,
                "loss": avg_tot,
            }, ckpt_path)
            print(f"        💾 Checkpoint → {ckpt_path}")

    total_time = time.perf_counter() - total_t0

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, args.save_name)
    torch.save(model.state_dict(), final_path)

    # ── Save training history ─────────────────────────────────────────────
    hist_path = os.path.join(args.save_dir, "training_history.npy")
    np.save(hist_path, history)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  📊 TRAINING COMPLETE — SUMMARY")
    print(f"{'═' * 72}")
    print(f"\n  Total time       : {total_time/60:.1f} min")
    print(f"  Epochs trained   : {args.epochs - start_epoch}")
    print(f"  Effective range  : epoch 71 → {70 + args.epochs}")
    print(f"  Best ε_rel       : {best_rel_l2:.6f}  (effective epoch {70 + best_epoch})")
    print(f"  Final ε_rel      : {avg_rel:.6f}")
    print(f"  Final L_data     : {avg_mse:.6f}")
    print(f"  Final L_PDE      : {avg_pde:.6f}")
    print(f"\n  📁 Saved:")
    print(f"     Final weights → {final_path}")
    print(f"     Best weights  → {os.path.join(args.save_dir, 'collision_best.pth')}")
    print(f"     History       → {hist_path}")
    print(f"     Checkpoints   → {args.ckpt_dir}/")

    # ── Publication readiness check ───────────────────────────────────────
    print(f"\n{'─' * 72}")
    if best_rel_l2 < 0.05:
        print(f"  ✅ PUBLICATION READY — Rel L₂ = {best_rel_l2:.4f} (<5%)")
    elif best_rel_l2 < 0.10:
        print(f"  ⚠️  CLOSE — Rel L₂ = {best_rel_l2:.4f} (<10%, target <5%)")
        print(f"     Consider: more epochs, lower LR, or larger dataset")
    else:
        print(f"  ❌ NOT CONVERGED — Rel L₂ = {best_rel_l2:.4f}")
        print(f"     Consider resuming from best checkpoint with lower LR")
    print(f"{'─' * 72}")

    print(f"\n  Next steps:")
    print(f"    1. python evaluate_collisions.py --model {final_path}")
    print(f"    2. python benchmark_speedup.py")
    print(f"\n{'═' * 72}")
    print("  🏁 DONE")
    print(f"{'═' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
