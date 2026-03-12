"""
evaluate_collisions.py — Rigorous Statistical Evaluation for Collision Model
==============================================================================

Runs inference on ALL samples (or a specified subset) of the collision
dataset and produces:

1. **Aggregate statistical report** — mean ± std of MAE, Relative L₂, and
   per-sample error across the entire dataset.

2. **Publication figure** — 4-panel layout:
   - Panel A: Best-case sample (Ground Truth | Prediction | Error)
   - Panel B: Median-case sample
   - Panel C: Worst-case sample
   - Panel D: Error distribution histogram across all samples

3. **Per-sample CSV** — full metrics for every sample (importable to LaTeX).

Saved to ``results/collision_proof.png`` and ``results/collision_metrics.csv``.

Usage
-----
    python evaluate_collisions.py
    python evaluate_collisions.py --n_eval 100
    python evaluate_collisions.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fno_architectures import FNO3d


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rigorous collision-regime evaluation (multi-sample).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", type=str,
                    default=os.path.join("models", "collision_publication.pth"))
    p.add_argument("--data_dir", type=str, default="data_collision_master")
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_eval", type=int, default=None,
                    help="Number of samples to evaluate (default: all)")
    p.add_argument("--time_slice", type=int, default=None,
                    help="Time-slice for figure panels (default: middle)")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data + Model
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str):
    """Load full collision dataset and compute norm stats."""
    bdy  = np.load(os.path.join(data_dir, "bdy_collision.npy")).astype(np.float32)
    bulk = np.load(os.path.join(data_dir, "bulk_collision.npy")).astype(np.float32)

    stats = dict(
        x_mean=float(bdy.mean()),  x_std=float(bdy.std()),
        y_mean=float(bulk.mean()), y_std=float(bulk.std()),
    )
    return bdy, bulk, stats


def build_model(path, device, modes=8, width=20, n_layers=4):
    model = FNO3d(modes1=modes, modes2=modes, modes3=modes,
                  width=width, n_layers=n_layers, in_channels=3).to(device)
    if not os.path.isfile(path):
        print(f"Not found: {path}"); sys.exit(1)
    state = torch.load(path, map_location=device)
    # Handle full checkpoint dict vs raw state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        eff = state.get("effective_epoch", state.get("epoch", "?"))
        rl2 = state.get("rel_l2", None)
        extra = f"  epoch={eff}"
        if rl2 is not None:
            extra += f"  rel_l2={rl2:.6f}"
    else:
        model.load_state_dict(state)
        extra = ""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO3d: {n_params:,} parameters  ({path}){extra}")
    return model


def construct_input(bdy_sample, stats, Z):
    """Build (1, 3, T, X, Z) from a single boundary (T, X)."""
    T, X = bdy_sample.shape
    bdy_n = (bdy_sample - stats["x_mean"]) / (stats["x_std"] + 1e-8)
    wave  = np.tile(bdy_n[:, :, None], (1, 1, Z))
    tc    = np.linspace(0, 1, T, dtype=np.float32)
    time  = np.broadcast_to(tc[:, None, None], (T, X, Z)).copy()
    zc    = np.linspace(0, 1, Z, dtype=np.float32)
    depth = np.broadcast_to(zc[None, None, :], (T, X, Z)).copy()
    return np.stack([wave, time, depth], axis=0)[None].astype(np.float32)


@torch.no_grad()
def predict_single(model, x_input, device, y_mean, y_std):
    x = torch.from_numpy(x_input).to(device)
    pred = model(x).squeeze().cpu().numpy()
    return pred * (y_std + 1e-8) + y_mean


# ─────────────────────────────────────────────────────────────────────────────
# Publication figure
# ─────────────────────────────────────────────────────────────────────────────

def _add_cbar(im, ax):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.08)
    plt.colorbar(im, cax=cax)


def _draw_triplet(axes, truth_t, pred_t, row_label, vmin, vmax):
    """Draw one row of [Truth | Pred | Error] on 3 axes."""
    err_t = np.abs(truth_t - pred_t)
    mae_t = np.mean(err_t)

    im0 = axes[0].imshow(truth_t, cmap="inferno", vmin=vmin, vmax=vmax,
                          origin="upper", aspect="equal")
    axes[0].set_ylabel(row_label, fontsize=11, fontweight="bold")
    _add_cbar(im0, axes[0])

    im1 = axes[1].imshow(pred_t, cmap="inferno", vmin=vmin, vmax=vmax,
                          origin="upper", aspect="equal")
    _add_cbar(im1, axes[1])

    im2 = axes[2].imshow(err_t, cmap="hot", vmin=0, vmax=err_t.max(),
                          origin="upper", aspect="equal")
    axes[2].set_title(f"MAE: {mae_t:.4f}", fontsize=9, pad=4)
    _add_cbar(im2, axes[2])


def create_publication_figure(
    bdy_all, bulk_all, model, device, stats,
    best_idx, median_idx, worst_idx,
    per_sample_mae, per_sample_rl2,
    time_slice, save_path, dpi,
):
    """4-row publication figure: Best | Median | Worst | Histogram."""
    Z = bulk_all.shape[3]

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("white")

    # Create grid: 3 rows of triplets + 1 row of histograms
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3,
                           height_ratios=[1, 1, 1, 0.9])

    cases = [
        (best_idx,   "BEST CASE",   "#1B5E20"),
        (median_idx, "MEDIAN CASE", "#1565C0"),
        (worst_idx,  "WORST CASE",  "#B71C1C"),
    ]

    for row, (idx, label, colour) in enumerate(cases):
        bdy_s  = bdy_all[idx]
        bulk_s = bulk_all[idx]
        x_in   = construct_input(bdy_s, stats, Z)
        pred_s = predict_single(model, x_in, device,
                                 stats["y_mean"], stats["y_std"])

        t = time_slice
        truth_t = bulk_s[t]
        pred_t  = pred_s[t]
        vmin = min(truth_t.min(), pred_t.min())
        vmax = max(truth_t.max(), pred_t.max())

        err_full = np.abs(bulk_s - pred_s)
        mae_full = np.mean(err_full)
        rl2_full = np.linalg.norm(pred_s.ravel() - bulk_s.ravel()) / (
            np.linalg.norm(bulk_s.ravel()) + 1e-12)

        axes = [fig.add_subplot(gs[row, c]) for c in range(3)]

        row_label = (f"{label}\nSample #{idx}\n"
                     f"MAE: {mae_full:.4f}\nRel L₂: {rl2_full:.4f}")
        _draw_triplet(axes, truth_t, pred_t, row_label, vmin, vmax)

        if row == 0:
            axes[0].set_title("Ground Truth\n(LU Solver)", fontsize=12,
                               fontweight="bold", pad=10, color="#1B5E20")
            axes[1].set_title("Neural-AdS\n(FNO3d)", fontsize=12,
                               fontweight="bold", pad=10, color="#1565C0")
            axes[2].set_title("Absolute Error", fontsize=12,
                               fontweight="bold", pad=10, color="#B71C1C")

    # ── Row 4: Histograms ────────────────────────────────────────────────
    ax_mae = fig.add_subplot(gs[3, 0])
    ax_rl2 = fig.add_subplot(gs[3, 1])
    ax_txt = fig.add_subplot(gs[3, 2])

    # MAE histogram
    ax_mae.hist(per_sample_mae, bins=30, color="#1565C0", alpha=0.85,
                edgecolor="white", linewidth=0.5)
    ax_mae.axvline(np.mean(per_sample_mae), color="#B71C1C", lw=2,
                    linestyle="--", label=f"μ = {np.mean(per_sample_mae):.4f}")
    ax_mae.set_xlabel("MAE per sample", fontsize=11)
    ax_mae.set_ylabel("Count", fontsize=11)
    ax_mae.set_title("MAE Distribution", fontsize=12, fontweight="bold")
    ax_mae.legend(fontsize=10)

    # Rel L₂ histogram
    ax_rl2.hist(per_sample_rl2, bins=30, color="#7C4DFF", alpha=0.85,
                edgecolor="white", linewidth=0.5)
    ax_rl2.axvline(np.mean(per_sample_rl2), color="#B71C1C", lw=2,
                    linestyle="--", label=f"μ = {np.mean(per_sample_rl2):.4f}")
    ax_rl2.set_xlabel("Relative L₂ per sample", fontsize=11)
    ax_rl2.set_ylabel("Count", fontsize=11)
    ax_rl2.set_title("Relative L₂ Distribution", fontsize=12, fontweight="bold")
    ax_rl2.legend(fontsize=10)

    # Summary stats text box
    ax_txt.axis("off")
    summary = (
        f"AGGREGATE STATISTICS\n"
        f"{'─' * 36}\n"
        f"Samples evaluated:  {len(per_sample_mae)}\n\n"
        f"MAE\n"
        f"  Mean:   {np.mean(per_sample_mae):.6f}\n"
        f"  Std:    {np.std(per_sample_mae):.6f}\n"
        f"  Min:    {np.min(per_sample_mae):.6f}\n"
        f"  Max:    {np.max(per_sample_mae):.6f}\n\n"
        f"Relative L₂\n"
        f"  Mean:   {np.mean(per_sample_rl2):.6f}\n"
        f"  Std:    {np.std(per_sample_rl2):.6f}\n"
        f"  Min:    {np.min(per_sample_rl2):.6f}\n"
        f"  Max:    {np.max(per_sample_rl2):.6f}\n\n"
        f"% samples with Rel L₂ < 5%:  "
        f"{100 * np.mean(per_sample_rl2 < 0.05):.1f}%\n"
        f"% samples with Rel L₂ < 10%: "
        f"{100 * np.mean(per_sample_rl2 < 0.10):.1f}%"
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=10.5, fontfamily="monospace", va="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5",
                          edgecolor="#BDBDBD", alpha=0.95))

    fig.suptitle(
        "Dual-Source Collision Regime — Neural-AdS vs Exact Solver\n"
        "Comprehensive Statistical Evaluation",
        fontsize=17, fontweight="bold", y=0.995,
    )

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n🖼️  Publication figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}\n")

    # ── Load ──────────────────────────────────────────────────────────
    print("📦 Loading collision dataset …")
    bdy_all, bulk_all, stats = load_dataset(args.data_dir)
    N, T, X = bdy_all.shape
    Z = bulk_all.shape[3]
    n_eval = min(args.n_eval or N, N)
    print(f"   {N} samples × {T} time-steps × {X}×{Z} grid")
    print(f"   Evaluating: {n_eval} samples\n")

    model = build_model(args.model_path, device,
                        args.modes, args.width, args.n_layers)

    time_slice = args.time_slice if args.time_slice is not None else T // 2

    # ── Per-sample evaluation ─────────────────────────────────────────
    print(f"\n⚡ Running inference on {n_eval} samples …\n")

    per_sample_mae = np.empty(n_eval, dtype=np.float64)
    per_sample_max = np.empty(n_eval, dtype=np.float64)
    per_sample_rl2 = np.empty(n_eval, dtype=np.float64)

    for i in range(n_eval):
        bdy_s  = bdy_all[i]
        bulk_s = bulk_all[i]

        x_in  = construct_input(bdy_s, stats, Z)
        pred  = predict_single(model, x_in, device,
                                stats["y_mean"], stats["y_std"])

        err = np.abs(bulk_s - pred)
        per_sample_mae[i] = np.mean(err)
        per_sample_max[i] = np.max(err)
        per_sample_rl2[i] = (np.linalg.norm(pred.ravel() - bulk_s.ravel())
                             / (np.linalg.norm(bulk_s.ravel()) + 1e-12))

        if (i + 1) % max(1, n_eval // 10) == 0 or i == n_eval - 1:
            print(f"   [{i+1:4d}/{n_eval}]  "
                  f"MAE: {per_sample_mae[i]:.5f}  "
                  f"Rel L₂: {per_sample_rl2[i]:.5f}")

    # ── Identify best / median / worst ────────────────────────────────
    sorted_idx = np.argsort(per_sample_rl2)
    best_idx   = sorted_idx[0]
    median_idx = sorted_idx[n_eval // 2]
    worst_idx  = sorted_idx[-1]

    # ── Aggregate report ──────────────────────────────────────────────
    data_range = float(bulk_all.max() - bulk_all.min()) + 1e-8

    print(f"\n{'═' * 64}")
    print(f"  📊 COLLISION-REGIME STATISTICAL EVALUATION")
    print(f"  Model: {args.model_path}")
    print(f"  Samples: {n_eval}   Grid: {T}×{X}×{Z}")
    print(f"{'═' * 64}")
    print(f"  MAE")
    print(f"    Mean ± Std : {np.mean(per_sample_mae):.6f} ± "
          f"{np.std(per_sample_mae):.6f}")
    print(f"    Min / Max  : {np.min(per_sample_mae):.6f} / "
          f"{np.max(per_sample_mae):.6f}")
    print(f"    As % range : {np.mean(per_sample_mae)/data_range*100:.2f}%")
    print(f"  Relative L₂")
    print(f"    Mean ± Std : {np.mean(per_sample_rl2):.6f} ± "
          f"{np.std(per_sample_rl2):.6f}")
    print(f"    Min / Max  : {np.min(per_sample_rl2):.6f} / "
          f"{np.max(per_sample_rl2):.6f}")
    print(f"  Max Absolute Error")
    print(f"    Mean ± Std : {np.mean(per_sample_max):.6f} ± "
          f"{np.std(per_sample_max):.6f}")
    print(f"  Sample Quality")
    print(f"    Rel L₂ < 1%  : {100*np.mean(per_sample_rl2<0.01):.1f}%")
    print(f"    Rel L₂ < 5%  : {100*np.mean(per_sample_rl2<0.05):.1f}%")
    print(f"    Rel L₂ < 10% : {100*np.mean(per_sample_rl2<0.10):.1f}%")
    print(f"  Reference Samples")
    print(f"    Best   (#{best_idx})   : MAE={per_sample_mae[best_idx]:.6f}  "
          f"Rel L₂={per_sample_rl2[best_idx]:.6f}")
    print(f"    Median (#{median_idx})  : MAE={per_sample_mae[median_idx]:.6f}  "
          f"Rel L₂={per_sample_rl2[median_idx]:.6f}")
    print(f"    Worst  (#{worst_idx})   : MAE={per_sample_mae[worst_idx]:.6f}  "
          f"Rel L₂={per_sample_rl2[worst_idx]:.6f}")
    print(f"{'═' * 64}")

    # ── Save CSV ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "collision_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "mae", "max_abs_error", "relative_l2"])
        for i in range(n_eval):
            writer.writerow([i, f"{per_sample_mae[i]:.8f}",
                             f"{per_sample_max[i]:.8f}",
                             f"{per_sample_rl2[i]:.8f}"])
    print(f"\n📄 Metrics CSV → {csv_path}")

    # ── Publication figure ────────────────────────────────────────────
    fig_path = os.path.join(args.output_dir, "collision_proof.png")
    print("\n🎨 Rendering publication figure …")
    create_publication_figure(
        bdy_all, bulk_all, model, device, stats,
        best_idx, median_idx, worst_idx,
        per_sample_mae, per_sample_rl2,
        time_slice, fig_path, args.dpi,
    )

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
