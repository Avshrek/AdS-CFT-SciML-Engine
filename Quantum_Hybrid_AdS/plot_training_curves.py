"""
plot_training_curves.py — Training Convergence Visualisation
==============================================================

Reads the CSV log from ``finetune_publication.py`` and produces a
4-panel convergence figure:

  (A) Training Loss vs Epoch
  (B) Validation Loss vs Epoch
  (C) Learning Rate Schedule
  (D) Validation Relative L2 vs Epoch

Usage
-----
    python plot_training_curves.py
    python plot_training_curves.py --log results/training_log.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot training convergence curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log", type=str,
                    default=os.path.join("results", "training_log.csv"))
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Parse CSV ─────────────────────────────────────────────────────
    import csv
    with open(args.log, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    epochs     = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    train_mse  = [float(r["train_mse"]) for r in rows]
    train_pde  = [float(r["train_pde"]) for r in rows]
    lr         = [float(r["lr"]) for r in rows]

    # Validation is only logged every N epochs
    val_epochs = []
    val_loss   = []
    val_mse    = []
    val_pde    = []
    val_rl2    = []
    for r in rows:
        if r["val_rel_l2"]:
            val_epochs.append(int(r["epoch"]))
            val_loss.append(float(r["val_loss"]))
            val_mse.append(float(r["val_mse"]))
            val_pde.append(float(r["val_pde"]))
            val_rl2.append(float(r["val_rel_l2"]))

    # ── 2x2 Figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor("white")
    fig.suptitle("Training Convergence — Collision Fine-Tuning",
                 fontsize=16, fontweight="bold", y=0.98)

    # (A) Training Loss
    ax = axes[0, 0]
    ax.semilogy(epochs, train_loss, color="#1565C0", lw=1.2, alpha=0.8,
                label="Total Loss")
    ax.semilogy(epochs, train_mse, color="#43A047", lw=1, alpha=0.6,
                label="MSE (data)")
    ax.semilogy(epochs, train_pde, color="#EF6C00", lw=1, alpha=0.6,
                label="PDE (physics)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("(A) Training Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (B) Validation Loss
    ax = axes[0, 1]
    if val_loss:
        ax.semilogy(val_epochs, val_loss, "s-", color="#7B1FA2", lw=1.5,
                     markersize=4, label="Total Val Loss")
        ax.semilogy(val_epochs, val_mse, "o--", color="#43A047", lw=1,
                     markersize=3, alpha=0.7, label="Val MSE")
        ax.semilogy(val_epochs, val_pde, "^--", color="#EF6C00", lw=1,
                     markersize=3, alpha=0.7, label="Val PDE")
        ax.legend(fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("(B) Validation Loss", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (C) Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, lr, color="#D32F2F", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("(C) Learning Rate Schedule (Cosine Annealing)", fontsize=13,
                  fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    ax.grid(True, alpha=0.3)

    # (D) Validation Rel L2
    ax = axes[1, 1]
    if val_rl2:
        ax.plot(val_epochs, val_rl2, "s-", color="#1565C0", lw=2,
                 markersize=5)
        ax.axhline(0.05, color="#43A047", ls="--", lw=1.5, alpha=0.7,
                    label="5% target")
        ax.axhline(0.10, color="#EF6C00", ls="--", lw=1.5, alpha=0.7,
                    label="10% threshold")
        best = min(val_rl2)
        best_ep = val_epochs[val_rl2.index(best)]
        ax.annotate(f"Best: {best:.4f}\n(epoch {best_ep})",
                     xy=(best_ep, best), fontsize=10,
                     arrowprops=dict(arrowstyle="->", color="black"),
                     xytext=(best_ep + len(epochs)*0.1, best + 0.02),
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.legend(fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("(D) Validation Relative L2", fontsize=13,
                  fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "training_curves.png")
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Print summary
    print(f"Training log: {len(epochs)} epochs")
    print(f"Final train loss: {train_loss[-1]:.6f}")
    if val_rl2:
        print(f"Best val Rel L2:  {min(val_rl2):.6f} (epoch {val_epochs[val_rl2.index(min(val_rl2))]})")
        print(f"Final val Rel L2: {val_rl2[-1]:.6f}")
    print(f"Figure saved: {save_path}")


if __name__ == "__main__":
    main()
