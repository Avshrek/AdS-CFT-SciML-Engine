"""
finetune_collisions_physics.py — Rigorous Physics-Informed Fine-Tuning
========================================================================

Fine-tunes the pre-trained FNO3d holographic surrogate on dual-source
collision data with a combined data + physics loss and academic-grade
logging including the Relative L₂ Error norm.

Loss
~~~~
    L_total  =  L_data(MSE)  +  λ · L_PDE(∇²Φ)

The PDE residual enforces ∇²_{xz}Φ = 0 on every (X, Z) time-slice using
the standard 5-point finite-difference stencil applied via ``F.conv2d``.

Metrics (per epoch)
~~~~~~~~~~~~~~~~~~~
    L_data   — Mean Squared Error between prediction and ground truth.
    L_PDE    — Mean Squared Laplacian residual over interior points.
    ε_rel    — Relative L₂ Error:  ‖Φ_pred − Φ_true‖₂ / ‖Φ_true‖₂

Usage
-----
    python finetune_collisions_physics.py
    python finetune_collisions_physics.py --epochs 100 --lr 5e-5
    python finetune_collisions_physics.py --help
"""

from __future__ import annotations

import argparse
import math
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
# Custom loss components
# ─────────────────────────────────────────────────────────────────────────────

class SpatialLaplacianLoss(nn.Module):
    r"""Quasi-static PDE residual: ∇²_{xz}Φ = 0 per time-slice.

    Applies the 5-point stencil::

        [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]]  / dx²

    to every ``(X, Z)`` slice via a single batched ``conv2d`` call.
    """

    def __init__(self, dx: float = 1.0) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / (dx ** 2)
        self.register_buffer("kernel", kernel)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        u : ``(B, 1, T, X, Z)``

        Returns
        -------
        Scalar — MSE of the Laplacian over interior grid points.
        """
        B, C, T, X, Z = u.shape
        flat = u.permute(0, 2, 1, 3, 4).reshape(B * T, C, X, Z)
        lap = F.conv2d(flat, self.kernel, padding=1)
        interior = lap[:, :, 1:-1, 1:-1]
        return torch.mean(interior ** 2)


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Relative L₂ error:  ‖pred − target‖₂  /  ‖target‖₂."""
    diff_norm = torch.norm(pred - target)
    true_norm = torch.norm(target) + 1e-8
    return diff_norm / true_norm


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rigorous physics-informed fine-tuning on collision data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    type=str, default="data_collision_master")
    p.add_argument("--pretrained",  type=str,
                    default=os.path.join("models", "unified_time_final.pth"))
    p.add_argument("--modes",       type=int, default=8)
    p.add_argument("--width",       type=int, default=20)
    p.add_argument("--n_layers",    type=int, default=4)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--lambda_pde",  type=float, default=0.1,
                    help="Fixed PDE loss weight λ")
    p.add_argument("--save_dir",    type=str, default="models")
    p.add_argument("--save_name",   type=str, default="collision_rigorous.pth")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Strict standardisation & 3-channel construction
# ─────────────────────────────────────────────────────────────────────────────

def load_and_standardise(data_dir: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load collision data, enforce μ=0 σ=1 standardisation, build volume.

    Strict standardisation (z-score) is essential here because the
    dual-source superposition produces higher amplitude waves than the
    single-source Phase 2 data the model was originally trained on.

    Returns
    -------
    x_train : ``(N, 3, T, X, Z)``
    y_train : ``(N, 1, T, X, Z)``
    stats   : normalisation parameters for later de-normalisation
    """
    bdy_raw  = np.load(os.path.join(data_dir, "bdy_collision.npy")).astype(np.float32)
    bulk_raw = np.load(os.path.join(data_dir, "bulk_collision.npy")).astype(np.float32)

    N, T, X = bdy_raw.shape
    Z = bulk_raw.shape[3]

    # ── Strict standardisation (μ=0, σ=1) ────────────────────────────────
    x_mu, x_sig = float(bdy_raw.mean()),  float(bdy_raw.std())
    y_mu, y_sig = float(bulk_raw.mean()), float(bulk_raw.std())

    bdy_z  = (bdy_raw  - x_mu) / (x_sig + 1e-8)
    bulk_z = (bulk_raw - y_mu) / (y_sig + 1e-8)

    # Verify standardisation
    assert abs(bdy_z.mean())  < 1e-5, f"Boundary mean {bdy_z.mean():.2e} ≠ 0"
    assert abs(bdy_z.std() - 1.0) < 1e-3, f"Boundary std {bdy_z.std():.4f} ≠ 1"

    # ── Ch-0: wave tiled over depth ───────────────────────────────────────
    wave_3d = np.tile(bdy_z[:, :, :, np.newaxis], (1, 1, 1, Z))

    # ── Ch-1: temporal coordinate (0 → 1) ────────────────────────────────
    t_coord = np.linspace(0.0, 1.0, T, dtype=np.float32)
    time_3d = np.broadcast_to(
        t_coord[np.newaxis, :, np.newaxis, np.newaxis], (N, T, X, Z)
    ).copy()

    # ── Ch-2: depth coordinate (0 → 1) ───────────────────────────────────
    z_coord = np.linspace(0.0, 1.0, Z, dtype=np.float32)
    depth_3d = np.broadcast_to(
        z_coord[np.newaxis, np.newaxis, np.newaxis, :], (N, T, X, Z)
    ).copy()

    x_input  = np.stack([wave_3d, time_3d, depth_3d], axis=1)
    y_target = bulk_z[:, np.newaxis, :, :, :]

    stats = dict(x_mu=x_mu, x_sig=x_sig, y_mu=y_mu, y_sig=y_sig)
    return torch.from_numpy(x_input), torch.from_numpy(y_target), stats


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning loop with academic logging
# ─────────────────────────────────────────────────────────────────────────────

def finetune(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  RIGOROUS PHYSICS-INFORMED FINE-TUNING — FNO3d")
    print("=" * 72)
    print(f"\n  Device          : {device}")
    print(f"  Pre-trained     : {args.pretrained}")
    print(f"  Data            : {args.data_dir}/")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  λ_PDE           : {args.lambda_pde}")
    print(f"  Output          : {args.save_dir}/{args.save_name}")

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  DATA INGESTION & STANDARDISATION")
    print(f"{'─' * 72}")

    x_train, y_train, stats = load_and_standardise(args.data_dir)

    print(f"\n  Collision dataset statistics (raw):")
    print(f"    Boundary   μ = {stats['x_mu']:+.6f}    σ = {stats['x_sig']:.6f}")
    print(f"    Bulk       μ = {stats['y_mu']:+.6f}    σ = {stats['y_sig']:.6f}")
    print(f"\n  After standardisation:")
    print(f"    Boundary   μ ≈ {float(x_train[:, 0].mean()):.2e}    "
          f"σ ≈ {float(x_train[:, 0].std()):.4f}")
    print(f"    Bulk       μ ≈ {float(y_train.mean()):.2e}    "
          f"σ ≈ {float(y_train.std()):.4f}")
    print(f"\n  x_train : {tuple(x_train.shape)}")
    print(f"  y_train : {tuple(y_train.shape)}")

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  MODEL INITIALISATION")
    print(f"{'─' * 72}")

    model = FNO3d(
        modes1=args.modes, modes2=args.modes, modes3=args.modes,
        width=args.width, n_layers=args.n_layers, in_channels=3,
    ).to(device)

    state = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(state)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  FNO3d architecture loaded")
    print(f"    Parameters    : {n_params:,}")
    print(f"    Fourier modes : {args.modes}³")
    print(f"    Width         : {args.width}")
    print(f"    Layers        : {args.n_layers}")

    # ── Loss, optimiser ───────────────────────────────────────────────────
    mse_criterion = nn.MSELoss()
    pde_criterion = SpatialLaplacianLoss(dx=1.0).to(device)
    lam = args.lambda_pde

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  TRAINING LOG")
    print(f"{'─' * 72}\n")
    print(f"  {'Epoch':>5}  {'L_data':>10}  {'L_PDE':>10}  "
          f"{'L_total':>10}  {'ε_rel':>10}  {'Time':>7}")
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 10}  "
          f"{'─' * 10}  {'─' * 10}  {'─' * 7}")

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        sum_mse, sum_pde, sum_rel = 0.0, 0.0, 0.0
        n_batches = 0
        t0 = time.perf_counter()

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)                           # (B, 1, T, X, Z)

            l_data = mse_criterion(pred, yb)
            l_pde  = pde_criterion(pred)
            l_tot  = l_data + lam * l_pde

            l_tot.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_clip
            )
            optimizer.step()

            with torch.no_grad():
                eps_rel = relative_l2_error(pred, yb)

            sum_mse += l_data.item()
            sum_pde += l_pde.item()
            sum_rel += eps_rel.item()
            n_batches += 1

        elapsed  = time.perf_counter() - t0
        avg_mse  = sum_mse / n_batches
        avg_pde  = sum_pde / n_batches
        avg_tot  = avg_mse + lam * avg_pde
        avg_rel  = sum_rel / n_batches

        marker = ""
        if avg_tot < best_loss:
            best_loss = avg_tot
            best_epoch = epoch + 1
            marker = " ★"

        print(f"  {epoch + 1:>5d}  {avg_mse:>10.6f}  {avg_pde:>10.6f}  "
              f"{avg_tot:>10.6f}  {avg_rel:>10.6f}  {elapsed:>6.1f}s{marker}")

    # ── Save ──────────────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, args.save_name)
    torch.save(model.state_dict(), final_path)

    print(f"\n{'─' * 72}")
    print("  RESULTS SUMMARY")
    print(f"{'─' * 72}")
    print(f"\n  Best L_total     : {best_loss:.6f}  (epoch {best_epoch})")
    print(f"  Final ε_rel      : {avg_rel:.6f}")
    print(f"  Final L_data     : {avg_mse:.6f}")
    print(f"  Final L_PDE      : {avg_pde:.6f}")
    print(f"  Saved weights    → {final_path}")
    print(f"  Parameters       : {n_params:,}")
    print(f"\n{'=' * 72}")
    print("  ✅ RIGOROUS FINE-TUNING COMPLETE")
    print(f"{'=' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    finetune(parse_args())
