#!/usr/bin/env python3
"""
================================================================
 PUBLICATION FIGURES — Quantum Neural-AdS
================================================================
Generates a single 2×2 (or 2×3) multi-panel figure suitable for
journal submission (Nature / PRL / PRX Quantum style).

  Panel A — Training convergence (MSE & MAE vs epoch)
  Panel B — Entanglement entropy → Page limit
  Panel C — 3D bulk reconstruction (best sample GT vs Pred vs Error)
  Panel D — Error distribution across all test samples

Requires:
  results/nature_quantum_metrics.csv   (from training)
  results/quantum_full_evaluation.csv  (from evaluate_quantum_full.py)
  data_collision_master/               (for ground truth)
  results/quantum_all_predictions.npy  (from evaluate_quantum_full.py)

Output:
  figures/publication_figure_main.pdf
  figures/publication_figure_main.png
================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "legend.fontsize":  8.5,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.5,
})

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_training_metrics():
    df = pd.read_csv("results/nature_quantum_metrics.csv")
    return df


def load_eval_metrics():
    df = pd.read_csv("results/quantum_full_evaluation.csv")
    return df


def load_data_and_preds():
    blk  = np.load("data_collision_master/bulk_collision.npy")
    pred = np.load("results/quantum_all_predictions.npy")
    return blk, pred


def panel_a_convergence(ax_mse, ax_mae, df_train):
    """Panel A: Training convergence — MSE (left axis) and MAE."""
    epochs = df_train["Epoch"].values
    mse    = df_train["MSE_Loss"].values
    mae    = df_train["Bulk_MAE"].values

    # --- MSE subplot ---
    ax_mse.semilogy(epochs, mse, color="#1565C0", linewidth=1.0, alpha=0.85)
    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("MSE Loss (log scale)")
    ax_mse.set_title("(a) Training Convergence", fontweight="bold", loc="left")

    # Annotate final
    ax_mse.annotate(f"Final: {mse[-1]:.2e}",
                    xy=(epochs[-1], mse[-1]),
                    xytext=(-80, 20), textcoords="offset points",
                    fontsize=8, color="#1565C0",
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8))

    # --- MAE subplot ---
    ax_mae.plot(epochs, mae, color="#E65100", linewidth=1.0, alpha=0.85)
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("Bulk MAE")
    ax_mae.set_title("", fontweight="bold")  # shared with (a)

    ax_mae.annotate(f"Best: {min(mae):.4f}",
                    xy=(np.argmin(mae) + 1, min(mae)),
                    xytext=(30, 20), textcoords="offset points",
                    fontsize=8, color="#E65100",
                    arrowprops=dict(arrowstyle="->", color="#E65100", lw=0.8))

    # 16x improvement annotation
    ratio = mse[0] / mse[-1]
    ax_mse.text(0.98, 0.92, f"{ratio:.0f}× improvement",
                transform=ax_mse.transAxes, ha="right", fontsize=9,
                fontweight="bold", color="#1B5E20",
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#66BB6A", alpha=0.9))


def panel_b_entropy(ax, df_train):
    """Panel B: von Neumann entropy converging to Page limit."""
    epochs  = df_train["Epoch"].values
    entropy = df_train["von_Neumann_Entropy"].values

    # Page limit for 5+5 bipartition of 10 qubits
    d_A = 2**5
    d_B = 2**5
    page_limit = np.log(d_A) - d_A / (2 * d_B)  # corrected Page value

    ax.plot(epochs, entropy, color="#6A1B9A", linewidth=1.0, alpha=0.85,
            label=r"$S_A$ (von Neumann)")
    ax.axhline(page_limit, color="#B71C1C", linestyle="--", linewidth=1.2,
               label=f"Page limit = {page_limit:.4f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Entanglement Entropy $S_A$")
    ax.set_title("(b) Ryu-Takayanagi Convergence", fontweight="bold", loc="left")
    ax.legend(loc="lower right", framealpha=0.9)

    # Annotate deviation
    last_100 = entropy[-100:]
    mean_ent = np.mean(last_100)
    dev_pct  = abs(mean_ent - page_limit) / page_limit * 100
    ax.text(0.98, 0.15,
            f"Last 100 epochs:\n"
            f"$\\langle S_A \\rangle$ = {mean_ent:.4f} ± {np.std(last_100):.4f}\n"
            f"Deviation: {dev_pct:.2f}%",
            transform=ax.transAxes, ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F3E5F5", ec="#AB47BC", alpha=0.9))


def panel_c_reconstruction(ax_gt, ax_pred, ax_err, blk, pred, df_eval):
    """Panel C: Best-case 3D bulk reconstruction (single time-slice)."""
    best_idx = int(df_eval.loc[df_eval["mae"].idxmin(), "sample_idx"])

    gt = blk[best_idx]    # (20, 64, 64)
    pr = pred[best_idx]   # (20, 64, 64)

    # Pick middle time-slice
    t_mid = gt.shape[0] // 2
    gt_s = gt[t_mid]
    pr_s = pr[t_mid]
    er_s = np.abs(pr_s - gt_s)

    vmin = min(gt_s.min(), pr_s.min())
    vmax = max(gt_s.max(), pr_s.max())

    im0 = ax_gt.imshow(gt_s, cmap="inferno", vmin=vmin, vmax=vmax,
                        origin="lower", aspect="equal")
    ax_gt.set_title("(c) Ground Truth", fontweight="bold", loc="left")
    ax_gt.set_xlabel("x (boundary)")
    ax_gt.set_ylabel("z (radial depth)")
    plt.colorbar(im0, ax=ax_gt, fraction=0.046, pad=0.04)

    im1 = ax_pred.imshow(pr_s, cmap="inferno", vmin=vmin, vmax=vmax,
                          origin="lower", aspect="equal")
    ax_pred.set_title("Prediction", fontweight="bold", loc="left")
    ax_pred.set_xlabel("x (boundary)")
    plt.colorbar(im1, ax=ax_pred, fraction=0.046, pad=0.04)

    im2 = ax_err.imshow(er_s, cmap="hot", vmin=0, vmax=er_s.max(),
                         origin="lower", aspect="equal")
    mae_val = float(np.mean(er_s))
    ax_err.set_title(f"|Error| (MAE={mae_val:.5f})", fontweight="bold", loc="left")
    ax_err.set_xlabel("x (boundary)")
    plt.colorbar(im2, ax=ax_err, fraction=0.046, pad=0.04)

    # Sample label
    ax_gt.text(0.02, 0.95, f"Sample #{best_idx}, t={t_mid}",
               transform=ax_gt.transAxes, fontsize=7, color="white",
               va="top", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))


def panel_d_error_distribution(ax_mae, ax_rl2, df_eval):
    """Panel D: Error distribution histograms across all samples."""
    maes = df_eval["mae"].values
    rl2s = df_eval["relative_l2"].values

    # MAE histogram
    ax_mae.hist(maes, bins=40, color="#1565C0", alpha=0.85,
                edgecolor="white", linewidth=0.4)
    ax_mae.axvline(np.mean(maes), color="#B71C1C", lw=1.5, ls="--",
                   label=f"μ = {np.mean(maes):.5f}")
    ax_mae.axvline(np.median(maes), color="#FF6F00", lw=1.2, ls=":",
                   label=f"median = {np.median(maes):.5f}")
    ax_mae.set_xlabel("MAE per sample")
    ax_mae.set_ylabel("Count")
    ax_mae.set_title("(d) MAE Distribution", fontweight="bold", loc="left")
    ax_mae.legend(fontsize=7.5, loc="upper right")

    # Rel L2 histogram
    ax_rl2.hist(rl2s, bins=40, color="#2E7D32", alpha=0.85,
                edgecolor="white", linewidth=0.4)
    ax_rl2.axvline(np.mean(rl2s), color="#B71C1C", lw=1.5, ls="--",
                   label=f"μ = {np.mean(rl2s):.4f}")
    ax_rl2.axvline(np.median(rl2s), color="#FF6F00", lw=1.2, ls=":",
                   label=f"median = {np.median(rl2s):.4f}")
    ax_rl2.set_xlabel("Relative L₂ per sample")
    ax_rl2.set_ylabel("Count")
    ax_rl2.set_title("Rel L₂ Distribution", fontweight="bold", loc="left")
    ax_rl2.legend(fontsize=7.5, loc="upper right")


def build_figure():
    """Compose the full publication figure."""
    df_train = load_training_metrics()
    df_eval  = load_eval_metrics()
    blk, pred = load_data_and_preds()

    # ── Layout: 3 rows ──────────────────────────────────────
    #  Row 0: [Convergence MSE] [Convergence MAE] [Entropy → Page]
    #  Row 1: [GT]              [Prediction]       [Error map]
    #  Row 2: [MAE histogram]   [Rel L2 histogram] [Stats box]
    fig = plt.figure(figsize=(16, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.35,
                  height_ratios=[1, 1, 0.9])

    # Row 0
    ax_mse     = fig.add_subplot(gs[0, 0])
    ax_mae_tr  = fig.add_subplot(gs[0, 1])
    ax_entropy = fig.add_subplot(gs[0, 2])

    # Row 1
    ax_gt   = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_err  = fig.add_subplot(gs[1, 2])

    # Row 2
    ax_hist_mae = fig.add_subplot(gs[2, 0])
    ax_hist_rl2 = fig.add_subplot(gs[2, 1])
    ax_stats    = fig.add_subplot(gs[2, 2])

    # ── Draw panels ──────────────────────────────────────────
    panel_a_convergence(ax_mse, ax_mae_tr, df_train)
    panel_b_entropy(ax_entropy, df_train)
    panel_c_reconstruction(ax_gt, ax_pred, ax_err, blk, pred, df_eval)
    panel_d_error_distribution(ax_hist_mae, ax_hist_rl2, df_eval)

    # ── Stats summary box ────────────────────────────────────
    ax_stats.axis("off")
    maes = df_eval["mae"].values
    rl2s = df_eval["relative_l2"].values
    mses = df_eval["mse"].values
    bdy  = df_eval["boundary_mae"].values
    pde  = df_eval["pde_residual"].values

    # Page entropy
    ent = df_train["von_Neumann_Entropy"].values
    page_lim = np.log(2**5) - 2**5 / (2 * 2**5)
    ent_mean = np.mean(ent[-100:])
    ent_dev  = abs(ent_mean - page_lim) / page_lim * 100

    stats_text = (
        f"AGGREGATE STATISTICS  (N = {len(df_eval)})\n"
        f"{'─' * 42}\n"
        f"\n"
        f"{'Metric':<22s} {'Mean':>10s}  {'± Std':>10s}\n"
        f"{'─' * 42}\n"
        f"{'MAE':<22s} {np.mean(maes):>10.6f}  {np.std(maes):>10.6f}\n"
        f"{'MSE':<22s} {np.mean(mses):>10.8f}  {np.std(mses):>10.8f}\n"
        f"{'Relative L₂':<22s} {np.mean(rl2s):>10.6f}  {np.std(rl2s):>10.6f}\n"
        f"{'Boundary MAE (z=0)':<22s} {np.mean(bdy):>10.6f}  {np.std(bdy):>10.6f}\n"
        f"{'PDE Residual |∇²Φ|':<22s} {np.mean(pde):>10.6f}  {np.std(pde):>10.6f}\n"
        f"{'─' * 42}\n"
        f"\n"
        f"QUANTUM PHYSICS\n"
        f"  Qubits: 10 (5+5 bipartition)\n"
        f"  Layers: 3 StronglyEntanglingLayers\n"
        f"  ⟨S_A⟩ (last 100): {ent_mean:.4f} ± {np.std(ent[-100:]):.4f}\n"
        f"  Page limit:        {page_lim:.4f}\n"
        f"  Deviation:         {ent_dev:.2f}%\n"
    )

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily="monospace", va="top",
                  bbox=dict(boxstyle="round,pad=0.6", fc="#FAFAFA",
                            ec="#BDBDBD", alpha=0.95))

    # ── Supertitle ───────────────────────────────────────────
    fig.suptitle("Quantum Neural-AdS: Holographic Bulk Reconstruction\n"
                 "via Hybrid Quantum-Classical Autoencoder",
                 fontsize=15, fontweight="bold", y=0.995)

    # ── Save ─────────────────────────────────────────────────
    for ext in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"publication_figure_main.{ext}")
        fig.savefig(path, facecolor="white")
        print(f"  Saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 60 + "\n")
    build_figure()
    print("\n  ✅ Done.\n")
