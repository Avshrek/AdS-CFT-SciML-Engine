"""
Full Test-Set Evaluation — Quantum Neural-AdS
Runs inference on ALL 1000 samples, computes aggregate statistics.
Outputs: results/full_testset_metrics.csv + printed summary table.
"""
import os, sys, time, csv
import numpy as np
import torch
from hybrid_autoencoder import HybridQuantumAdS

# ── Config ────────────────────────────────────────────────────────────
DATA_DIR    = "data_collision_master"
MODEL_PATH  = "models/NATURE_QUANTUM_MODEL.pth"
OUT_CSV     = "results/full_testset_metrics.csv"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    os.makedirs("results", exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model from {MODEL_PATH} ...")
    model = HybridQuantumAdS(in_channels=1, out_channels=1).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    # handle key migration in either direction
    if "quantum_layer.q_weights" in state and "quantum_layer.q_layer.weights" not in state:
        state["quantum_layer.q_layer.weights"] = state.pop("quantum_layer.q_weights")
    elif "quantum_layer.q_layer.weights" in state and "quantum_layer.q_weights" not in state:
        pass  # already correct
    model.load_state_dict(state)
    model.eval()
    print(f"  Model loaded on {DEVICE}")

    # ── Load data ─────────────────────────────────────────────────────
    bdy = np.load(os.path.join(DATA_DIR, "bdy_collision.npy"))   # (N,20,64)
    blk = np.load(os.path.join(DATA_DIR, "bulk_collision.npy"))  # (N,20,64,64)
    N = bdy.shape[0]
    print(f"  Dataset: {N} samples  bdy={bdy.shape}  blk={blk.shape}")

    # ── Inference loop ────────────────────────────────────────────────
    per_sample = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(N):
            # tile boundary (20,64) -> (1,1,20,64,64)
            b = torch.from_numpy(bdy[i]).float().unsqueeze(-1).repeat(1, 1, 64)
            x = b.unsqueeze(0).unsqueeze(0).to(DEVICE)

            pred = model(x).cpu().numpy().squeeze()          # (20,64,64)
            truth = blk[i]                                    # (20,64,64)

            err = np.abs(pred - truth)
            mae    = float(np.mean(err))
            mse    = float(np.mean((pred - truth) ** 2))
            max_ae = float(np.max(err))
            rel_l2 = float(np.linalg.norm(pred.ravel() - truth.ravel())
                           / (np.linalg.norm(truth.ravel()) + 1e-12))

            per_sample.append({
                "sample": i,
                "mae": mae,
                "mse": mse,
                "max_abs_error": max_ae,
                "rel_l2": rel_l2,
            })

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1:>4d}/{N}]  MAE={mae:.6f}  MSE={mse:.8f}  RelL2={rel_l2:.6f}")

    elapsed = time.time() - t0
    print(f"\n  Inference complete: {elapsed:.1f}s  ({elapsed/N*1000:.1f} ms/sample)")

    # ── Save per-sample CSV ───────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "mae", "mse", "max_abs_error", "rel_l2"])
        w.writeheader()
        w.writerows(per_sample)
    print(f"  Per-sample CSV saved to {OUT_CSV}")

    # ── Aggregate statistics ──────────────────────────────────────────
    maes   = np.array([s["mae"]    for s in per_sample])
    mses   = np.array([s["mse"]    for s in per_sample])
    maxes  = np.array([s["max_abs_error"] for s in per_sample])
    rl2s   = np.array([s["rel_l2"] for s in per_sample])

    data_range = blk.max() - blk.min()
    mae_pct = maes.mean() / data_range * 100

    print(f"\n{'='*64}")
    print(f"  FULL TEST-SET EVALUATION  ({N} samples)")
    print(f"{'='*64}")
    print(f"  MAE           : {maes.mean():.6f} +/- {maes.std():.6f}")
    print(f"  MAE (% range) : {mae_pct:.4f}%")
    print(f"  MSE           : {mses.mean():.8f} +/- {mses.std():.8f}")
    print(f"  Max Abs Error : {maxes.mean():.6f} +/- {maxes.std():.6f}")
    print(f"  Rel L2        : {rl2s.mean():.6f} +/- {rl2s.std():.6f}")
    print(f"  Samples < 1% RelL2 : {100*np.mean(rl2s < 0.01):.1f}%")
    print(f"  Samples < 5% RelL2 : {100*np.mean(rl2s < 0.05):.1f}%")
    print(f"  Samples <10% RelL2 : {100*np.mean(rl2s < 0.10):.1f}%")
    print(f"  Best  sample  : #{np.argmin(maes)}  MAE={maes.min():.6f}")
    print(f"  Worst sample  : #{np.argmax(maes)}  MAE={maes.max():.6f}")
    print(f"  Median MAE    : {np.median(maes):.6f}")
    print(f"{'='*64}")

if __name__ == "__main__":
    main()
