"""
Publication-Quality 4-Panel Figure — Quantum Neural-AdS
Panel A: Training Convergence (MSE + MAE)
Panel B: Entanglement Entropy → Page Limit
Panel C: Best-Sample 3D Bulk Reconstruction (GT vs Pred vs Error)
Panel D: Error Distribution (MAE histogram over full test set)

Requires: results/nature_quantum_metrics.csv, results/full_testset_metrics.csv
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BLUE   = "#1565C0"
RED    = "#B71C1C"
GREEN  = "#2E7D32"
ORANGE = "#E65100"
PURPLE = "#6A1B9A"

OUT_PATH = "results/publication_figure.png"


def load_training_metrics():
    rows = list(csv.DictReader(open("results/nature_quantum_metrics.csv")))
    epochs   = [int(r["Epoch"]) for r in rows]
    mse_loss = [float(r["MSE_Loss"]) for r in rows]
    bulk_mae = [float(r["Bulk_MAE"]) for r in rows]
    entropy  = [float(r["von_Neumann_Entropy"]) for r in rows]
    return epochs, mse_loss, bulk_mae, entropy


def load_testset_metrics():
    rows = list(csv.DictReader(open("results/full_testset_metrics.csv")))
    maes = [float(r["mae"]) for r in rows]
    rl2s = [float(r["rel_l2"]) for r in rows]
    mses = [float(r["mse"]) for r in rows]
    return np.array(maes), np.array(rl2s), np.array(mses)


def get_best_worst_sample(maes):
    return int(np.argmin(maes)), int(np.argmax(maes))


def load_sample_prediction(idx):
    """Run model on sample idx and return (truth, pred, error)."""
    import torch
    from hybrid_autoencoder import HybridQuantumAdS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridQuantumAdS(in_channels=1, out_channels=1).to(device)
    state = torch.load("models/NATURE_QUANTUM_MODEL.pth", map_location=device)
    if "quantum_layer.q_weights" in state and "quantum_layer.q_layer.weights" not in state:
        state["quantum_layer.q_layer.weights"] = state.pop("quantum_layer.q_weights")
    model.load_state_dict(state)
    model.eval()

    bdy = np.load("data_collision_master/bdy_collision.npy")
    blk = np.load("data_collision_master/bulk_collision.npy")

    b = torch.from_numpy(bdy[idx]).float().unsqueeze(-1).repeat(1, 1, 64)
    x = b.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy().squeeze()
    truth = blk[idx]
    return truth, pred, np.abs(pred - truth)


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading training metrics ...")
    epochs, mse_loss, bulk_mae, entropy = load_training_metrics()

    print("Loading test-set metrics ...")
    maes, rl2s, mses = load_testset_metrics()
    best_idx, worst_idx = get_best_worst_sample(maes)

    print(f"Loading best sample #{best_idx} for reconstruction panel ...")
    truth, pred, error = load_sample_prediction(best_idx)
    # pick middle time-slice for 2D visualization
    t_mid = truth.shape[0] // 2

    # ── Build figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.30)

    # ═══════════════════════════════════════════════════════════════════
    # PANEL A: Training Convergence
    # ═══════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a2 = ax_a.twinx()

    ln1 = ax_a.plot(epochs, mse_loss, color=BLUE, lw=1.2, alpha=0.85, label="MSE Loss")
    ln2 = ax_a2.plot(epochs, bulk_mae, color=RED, lw=1.2, alpha=0.85, label="Bulk MAE")

    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("MSE Loss", color=BLUE)
    ax_a2.set_ylabel("Bulk MAE", color=RED)
    ax_a.tick_params(axis="y", labelcolor=BLUE)
    ax_a2.tick_params(axis="y", labelcolor=RED)

    # combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_a.legend(lns, labs, loc="upper right", framealpha=0.9)

    ax_a.set_title("(A)  Training Convergence", fontweight="bold", pad=10)
    ax_a.annotate(f"Final MSE: {mse_loss[-1]:.6f}\nFinal MAE: {bulk_mae[-1]:.4f}",
                  xy=(0.55, 0.55), xycoords="axes fraction",
                  fontsize=9, color="#333",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

    # ═══════════════════════════════════════════════════════════════════
    # PANEL B: Entanglement Entropy → Page Limit
    # ═══════════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])

    # Page limit for 5+5 bipartition of 10 qubits
    d_A = 2**5
    d_B = 2**5
    page_limit = np.log(d_A) - d_A / (2 * d_B)
    s_max = np.log(d_A)  # ln(32)

    ax_b.plot(epochs, entropy, color=PURPLE, lw=1.0, alpha=0.7, label="$S_A$ (von Neumann)")
    ax_b.axhline(page_limit, color=GREEN, ls="--", lw=2, label=f"Page limit = {page_limit:.4f}")
    ax_b.axhline(s_max, color=ORANGE, ls=":", lw=1.5, alpha=0.6, label=f"$S_{{max}}$ = ln(32) = {s_max:.4f}")

    # shade convergence zone
    ent_last100 = entropy[-100:]
    ent_mean = np.mean(ent_last100)
    ent_std  = np.std(ent_last100)
    ax_b.axhspan(ent_mean - 2*ent_std, ent_mean + 2*ent_std,
                 color=PURPLE, alpha=0.08, label=f"Last 100: {ent_mean:.4f} $\\pm$ {ent_std:.4f}")

    dev_pct = abs(ent_mean - page_limit) / page_limit * 100
    ax_b.set_xlabel("Epoch")
    ax_b.set_ylabel("Entanglement Entropy $S_A$")
    ax_b.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax_b.set_title("(B)  Ryu-Takayanagi Convergence", fontweight="bold", pad=10)
    ax_b.annotate(f"Deviation from Page: {dev_pct:.2f}%",
                  xy=(0.03, 0.92), xycoords="axes fraction",
                  fontsize=10, fontweight="bold", color=GREEN,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN, alpha=0.9))

    # ═══════════════════════════════════════════════════════════════════
    # PANEL C: Best-Sample Bulk Reconstruction (t = T/2)
    # ═══════════════════════════════════════════════════════════════════
    gs_c = gs[1, 0].subgridspec(1, 3, wspace=0.05)

    vmin = min(truth[t_mid].min(), pred[t_mid].min())
    vmax = max(truth[t_mid].max(), pred[t_mid].max())
    err_slice = error[t_mid]

    ax_gt = fig.add_subplot(gs_c[0])
    im0 = ax_gt.imshow(truth[t_mid], cmap="inferno", vmin=vmin, vmax=vmax,
                        origin="upper", aspect="equal")
    ax_gt.set_title("Ground Truth", fontsize=10, fontweight="bold", color=GREEN)
    ax_gt.set_ylabel("Radial depth $z$")
    ax_gt.set_xlabel("Boundary $x$")

    ax_pr = fig.add_subplot(gs_c[1])
    ax_pr.imshow(pred[t_mid], cmap="inferno", vmin=vmin, vmax=vmax,
                 origin="upper", aspect="equal")
    ax_pr.set_title("Quantum Neural-AdS", fontsize=10, fontweight="bold", color=BLUE)
    ax_pr.set_yticks([])
    ax_pr.set_xlabel("Boundary $x$")

    ax_er = fig.add_subplot(gs_c[2])
    im2 = ax_er.imshow(err_slice, cmap="hot", vmin=0, vmax=err_slice.max(),
                        origin="upper", aspect="equal")
    mae_slice = np.mean(err_slice)
    ax_er.set_title(f"|Error|  MAE={mae_slice:.5f}", fontsize=10, fontweight="bold", color=RED)
    ax_er.set_yticks([])
    ax_er.set_xlabel("Boundary $x$")

    # colorbars
    fig.colorbar(im0, ax=ax_gt, fraction=0.046, pad=0.04, format="%.2f")
    fig.colorbar(im2, ax=ax_er, fraction=0.046, pad=0.04, format="%.4f")

    # panel label
    ax_gt.text(-0.15, 1.15, "(C)  Best-Sample Bulk Reconstruction",
               transform=ax_gt.transAxes, fontsize=12, fontweight="bold")

    # ═══════════════════════════════════════════════════════════════════
    # PANEL D: Error Distribution
    # ═══════════════════════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])

    ax_d.hist(maes, bins=40, color=BLUE, alpha=0.8, edgecolor="white", lw=0.5,
              label=f"MAE ($\\mu$={maes.mean():.5f})")
    ax_d.axvline(maes.mean(), color=RED, lw=2, ls="--",
                 label=f"Mean = {maes.mean():.5f}")
    ax_d.axvline(np.median(maes), color=ORANGE, lw=1.5, ls=":",
                 label=f"Median = {np.median(maes):.5f}")

    ax_d.set_xlabel("Mean Absolute Error (MAE)")
    ax_d.set_ylabel("Count")
    ax_d.legend(loc="upper right", framealpha=0.9)
    ax_d.set_title("(D)  Error Distribution (Full Test Set)", fontweight="bold", pad=10)

    # Stats text box
    stats_text = (
        f"$N$ = {len(maes)} samples\n"
        f"MAE: {maes.mean():.5f} $\\pm$ {maes.std():.5f}\n"
        f"MSE: {mses.mean():.7f} $\\pm$ {mses.std():.7f}\n"
        f"Rel $L_2$: {rl2s.mean():.5f} $\\pm$ {rl2s.std():.5f}\n"
        f"< 5% Rel $L_2$: {100*np.mean(rl2s<0.05):.0f}%"
    )
    ax_d.text(0.97, 0.60, stats_text, transform=ax_d.transAxes,
              fontsize=8.5, va="top", ha="right", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#BDBDBD", alpha=0.95))

    # ── Super title ───────────────────────────────────────────────────
    fig.suptitle("Quantum Neural-AdS: Holographic Bulk Reconstruction\n"
                 "Hybrid 10-Qubit Autoencoder  |  3-Layer StronglyEntanglingLayers  |  1000 Epochs",
                 fontsize=14, fontweight="bold", y=0.98)

    fig.savefig(OUT_PATH, facecolor="white")
    plt.close(fig)
    print(f"\nPublication figure saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
