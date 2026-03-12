#!/usr/bin/env python3
"""
================================================================
 FULL TEST-SET EVALUATION — Quantum Neural-AdS
================================================================
Runs the trained HybridQuantumAdS model on ALL 1000 samples,
computing per-sample and aggregate statistics for publication.

Outputs:
  results/quantum_full_evaluation.csv   — per-sample metrics
  results/quantum_aggregate_stats.txt   — summary table
================================================================
"""

import os
import sys
import time
import numpy as np
import torch
from hybrid_autoencoder import HybridQuantumAdS

# ── Config ────────────────────────────────────────────────────
DATA_DIR   = "data_collision_master"
MODEL_PATH = "models/NATURE_QUANTUM_MODEL.pth"
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 16  # inference batch size


def load_model():
    model = HybridQuantumAdS(in_channels=1, out_channels=1)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    print(f"  Model loaded from {MODEL_PATH}  →  {DEVICE}")
    return model


def load_data():
    bdy = np.load(os.path.join(DATA_DIR, "bdy_collision.npy"))   # (N, 20, 64)
    blk = np.load(os.path.join(DATA_DIR, "bulk_collision.npy"))   # (N, 20, 64, 64)
    print(f"  Data: {bdy.shape[0]} samples, boundary {bdy.shape[1:]}, bulk {blk.shape[1:]}")
    return bdy, blk


def prepare_input(bdy_sample):
    """Tile boundary [20,64] → [1,1,20,64,64] for Conv3d encoder."""
    t = torch.from_numpy(bdy_sample).float()          # (20, 64)
    t = t.unsqueeze(-1).repeat(1, 1, 64)               # (20, 64, 64)
    return t.unsqueeze(0).unsqueeze(0)                  # (1, 1, 20, 64, 64)


@torch.no_grad()
def evaluate_all(model, bdy, blk):
    """Run inference on every sample and collect per-sample metrics."""
    N = bdy.shape[0]
    records = []
    all_preds = []

    t0 = time.time()
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        # Build batch
        batch_in = torch.cat([prepare_input(bdy[i]) for i in range(start, end)], dim=0).to(DEVICE)
        batch_gt = torch.from_numpy(blk[start:end]).float().unsqueeze(1).to(DEVICE)  # (B,1,20,64,64)

        pred = model(batch_in)
        
        for j in range(pred.shape[0]):
            idx = start + j
            p = pred[j, 0].cpu().numpy()   # (20, 64, 64)
            g = blk[idx]                     # (20, 64, 64)
            all_preds.append(p)

            err = np.abs(p - g)
            mae         = float(np.mean(err))
            max_err     = float(np.max(err))
            mse         = float(np.mean((p - g) ** 2))
            rmse        = float(np.sqrt(mse))
            norm_gt     = float(np.linalg.norm(g.ravel()))
            rel_l2      = float(np.linalg.norm((p - g).ravel()) / (norm_gt + 1e-12))
            data_range  = float(g.max() - g.min())
            mae_pct     = mae / (data_range + 1e-12) * 100

            # Boundary preservation: prediction at z=0 vs input boundary
            pred_bdy  = p[:, :, 0]           # (20, 64) — first z-slice
            bdy_gt    = bdy[idx]              # (20, 64)
            bdy_mae   = float(np.mean(np.abs(pred_bdy - bdy_gt)))

            # PDE residual (5-point Laplacian on each time-slice)
            pde_res = _pde_residual(p)

            records.append({
                "sample_idx": idx,
                "mae": mae,
                "mae_pct": mae_pct,
                "mse": mse,
                "rmse": rmse,
                "max_abs_error": max_err,
                "relative_l2": rel_l2,
                "boundary_mae": bdy_mae,
                "pde_residual": pde_res,
            })

        if (end) % 100 == 0 or end == N:
            elapsed = time.time() - t0
            print(f"  [{end:>4d}/{N}]  elapsed {elapsed:.1f}s")

    return records, np.array(all_preds)


def _pde_residual(vol):
    """Mean |∇²Φ| via 5-point Laplacian stencil on each (x,z) slice."""
    # vol: (T, X, Z)
    residuals = []
    for t in range(vol.shape[0]):
        s = vol[t]  # (X, Z)
        lap = np.zeros_like(s)
        lap[1:-1, :] += s[2:, :] + s[:-2, :] - 2 * s[1:-1, :]
        lap[:, 1:-1] += s[:, 2:] + s[:, :-2] - 2 * s[:, 1:-1]
        residuals.append(np.mean(np.abs(lap[1:-1, 1:-1])))
    return float(np.mean(residuals))


def print_and_save_summary(records):
    """Print aggregate statistics and save to file."""
    import pandas as pd

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "quantum_full_evaluation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Per-sample CSV → {csv_path}")

    # ── Aggregate ─────────────────────────────────────────────
    lines = []
    def p(s):
        lines.append(s)
        print(s)

    p("")
    p("=" * 68)
    p("  QUANTUM NEURAL-ADS — FULL TEST-SET EVALUATION")
    p("=" * 68)
    p(f"  Samples evaluated : {len(df)}")
    p(f"  Model             : {MODEL_PATH}")
    p(f"  Device            : {DEVICE}")
    p("")
    p("  ┌──────────────────────┬────────────────────────────────┐")
    p("  │  Metric              │  Mean ± Std                    │")
    p("  ├──────────────────────┼────────────────────────────────┤")

    for col, label, fmt in [
        ("mae",          "MAE",              ".6f"),
        ("mae_pct",      "MAE (% of range)", ".4f"),
        ("mse",          "MSE",              ".8f"),
        ("rmse",         "RMSE",             ".6f"),
        ("max_abs_error","Max |Error|",      ".6f"),
        ("relative_l2",  "Relative L₂",     ".6f"),
        ("boundary_mae", "Boundary MAE z=0", ".6f"),
        ("pde_residual", "PDE Residual ∇²Φ", ".6f"),
    ]:
        m = df[col].mean()
        s = df[col].std()
        val = f"{m:{fmt}} ± {s:{fmt}}"
        p(f"  │  {label:<20s}│  {val:<30s}│")

    p("  └──────────────────────┴────────────────────────────────┘")
    p("")

    # Percentile breakdown
    p("  PERCENTILE BREAKDOWN (Relative L₂):")
    for thr in [0.01, 0.02, 0.05, 0.10, 0.20]:
        frac = (df["relative_l2"] < thr).mean() * 100
        p(f"    Rel L₂ < {thr:.0%}  :  {frac:5.1f}% of samples")
    p("")

    # Best / Worst
    best = df.loc[df["mae"].idxmin()]
    worst = df.loc[df["mae"].idxmax()]
    median_idx = (df["mae"] - df["mae"].median()).abs().idxmin()
    med = df.loc[median_idx]
    p(f"  BEST   sample #{int(best['sample_idx']):>4d}  MAE={best['mae']:.6f}  Rel L₂={best['relative_l2']:.6f}")
    p(f"  MEDIAN sample #{int(med['sample_idx']):>4d}  MAE={med['mae']:.6f}  Rel L₂={med['relative_l2']:.6f}")
    p(f"  WORST  sample #{int(worst['sample_idx']):>4d}  MAE={worst['mae']:.6f}  Rel L₂={worst['relative_l2']:.6f}")
    p("=" * 68)

    txt_path = os.path.join(OUT_DIR, "quantum_aggregate_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Summary → {txt_path}")

    return df


if __name__ == "__main__":
    print("\n" + "=" * 68)
    print("  QUANTUM NEURAL-ADS  —  FULL TEST-SET EVALUATION")
    print("=" * 68 + "\n")
    model = load_model()
    bdy, blk = load_data()
    records, all_preds = evaluate_all(model, bdy, blk)
    df = print_and_save_summary(records)

    # Save predictions for figure script
    np.save(os.path.join(OUT_DIR, "quantum_all_predictions.npy"), all_preds)
    print(f"  Predictions saved → {os.path.join(OUT_DIR, 'quantum_all_predictions.npy')}")
    print("\n  ✅ Done.\n")
