"""
visualize_quantum_bulk.py — 3-D Bulk Reconstruction Visualizer
===============================================================

Side-by-side animated 3-D comparison of:
  • Ground Truth  (physics simulation)
  • Hybrid Quantum Neural Network prediction

Outputs
-------
    quantum_ground_truth.html   — standalone interactive ground truth
    quantum_prediction.html     — standalone interactive quantum prediction
    quantum_comparison.html     — dual-panel iframe page for direct comparison

Usage
-----
    python visualize_quantum_bulk.py
    python visualize_quantum_bulk.py --sample_idx 42
    python visualize_quantum_bulk.py --colorscale Plasma --enhance 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch

# ── Local imports ────────────────────────────────────────────────────
from render_3d_universe import build_hologram_figure, _enhance_contrast
from relative_l2_error import relative_l2_error


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-D Bulk Reconstruction: Ground Truth vs Quantum Hybrid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default="data_collision_master",
                    help="Directory containing bdy_collision.npy and bulk_collision.npy.")
    p.add_argument("--checkpoint", type=str, default="models/NATURE_QUANTUM_MODEL.pth",
                    help="Path to the trained HybridQuantumAdS checkpoint.")
    p.add_argument("--sample_idx", type=int, default=-1,
                    help="Sample index to visualise (-1 = random).")
    p.add_argument("--enhance", type=float, default=0.35,
                    help="Power-law contrast exponent (0-1). Lower = more bulk detail.")
    p.add_argument("--colorscale", type=str, default="Viridis",
                    help="Plotly colorscale (Viridis, Plasma, Inferno, Magma …).")
    p.add_argument("--fps", type=int, default=4,
                    help="Animation speed in frames per second.")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# DATA LOADING  (mirrors train_nature_quantum.py exactly)
# ═════════════════════════════════════════════════════════════════════

def load_sample(data_dir: str, idx: int):
    """
    Load a single (boundary, bulk) pair from the collision dataset.

    Returns
    -------
    x_input : torch.Tensor, shape (1, 1, 20, 64, 64)
        Tiled boundary ready for HybridQuantumAdS.
    y_truth : np.ndarray, shape (20, 64, 64)
        Ground truth bulk field in physical units.
    """
    bdy_path  = os.path.join(data_dir, "bdy_collision.npy")
    bulk_path = os.path.join(data_dir, "bulk_collision.npy")

    # Fallback directories
    if not os.path.isfile(bdy_path):
        for alt in ["data_collision_5k", "data_collision"]:
            alt_bdy = os.path.join(alt, "bdy_collision.npy")
            if os.path.isfile(alt_bdy):
                bdy_path  = alt_bdy
                bulk_path = os.path.join(alt, "bulk_collision.npy")
                break

    if not os.path.isfile(bdy_path):
        print("❌ [ERROR] Cannot find bdy_collision.npy in any known directory.")
        sys.exit(1)

    print(f"  [>] Loading data from: {os.path.dirname(bdy_path)}")
    bdy_all  = np.load(bdy_path,  mmap_mode="r")
    bulk_all = np.load(bulk_path, mmap_mode="r")

    N = bdy_all.shape[0]
    idx = min(idx, N - 1)
    print(f"     Dataset: {N} samples  |  Using sample #{idx}")

    # Boundary: [20, 64] → tile → [1, 1, 20, 64, 64]
    bdy_sample = bdy_all[idx].astype(np.float32).copy()          # (20, 64)
    bdy_tiled  = np.tile(bdy_sample[:, :, np.newaxis], (1, 1, 64))  # (20, 64, 64)
    x_input    = torch.from_numpy(bdy_tiled).float().unsqueeze(0).unsqueeze(0)

    # Bulk ground truth: [20, 64, 64]
    y_truth = bulk_all[idx].astype(np.float32).copy()

    print(f"     Input shape : {list(x_input.shape)}")
    print(f"     Truth shape : {list(y_truth.shape)}")
    print(f"     Truth Φ range: [{y_truth.min():.4f}, {y_truth.max():.4f}]")

    return x_input, y_truth


# ═════════════════════════════════════════════════════════════════════
# MODEL LOADING & INFERENCE
# ═════════════════════════════════════════════════════════════════════

def run_quantum_inference(checkpoint_path: str, x_input: torch.Tensor) -> np.ndarray:
    """
    Load HybridQuantumAdS_SIREN (or ClassicalAdS for Phase 1) from checkpoint.

    Returns
    -------
    field : np.ndarray, shape (20, 64, 64)
    """
    from siren_decoder import HybridQuantumAdS_SIREN
    from train_kaggle_threephase import QuantumLatentLayer, ClassicalAdS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [*] Loading Hybrid Quantum SIREN Model from: {checkpoint_path}")
    print(f"     Device: {device.type.upper()}")

    if "phase1" in checkpoint_path.lower() or "classical" in checkpoint_path.lower():
        print(f"  [!] Detected Phase 1: Loading ClassicalAdS (no quantum bottleneck)")
        model = ClassicalAdS().to(device)
    else:
        model = HybridQuantumAdS_SIREN().to(device)
        # Inject quantum layer (decoupled to avoid circular imports)
        model.quantum_layer = QuantumLatentLayer(n_quantum_layers=3).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle full checkpoint dict vs raw state_dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Model loaded — {n_params:,} parameters")

    # Forward pass
    print("  [~] Running quantum inference ...")
    x_input = x_input.to(device)
    with torch.no_grad():
        pred = model(x_input)  # (1, 1, 20, 64, 64)

    field = pred[0, 0].cpu().numpy()  # (20, 64, 64)
    print(f"  [OK] Prediction shape: {list(field.shape)}")
    print(f"     Φ range: [{field.min():.4f}, {field.max():.4f}]")

    return field


# ═════════════════════════════════════════════════════════════════════
# COMPARISON PAGE BUILDER
# ═════════════════════════════════════════════════════════════════════

def build_comparison_html(
    gt_file: str,
    pred_file: str,
    sample_idx: int,
    rel_l2_pct: float,
    output_path: str = "quantum_comparison.html",
):
    """Write a premium dual-panel iframe comparison page."""

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural-AdS · Quantum Bulk Reconstruction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #07070f;
    color: #e0e0f0;
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── Header ────────────────────────────────────── */
  .header {{
    text-align: center;
    padding: 28px 20px 18px;
    background: linear-gradient(180deg, #0d0d1e 0%, #07070f 100%);
    border-bottom: 1px solid rgba(100,100,180,0.15);
  }}
  .header h1 {{
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
  }}
  .header .meta {{
    display: flex;
    justify-content: center;
    gap: 28px;
    flex-wrap: wrap;
    font-size: 0.82rem;
    color: #888;
  }}
  .header .meta span {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }}
  .header .meta .value {{
    color: #c4b5fd;
    font-weight: 600;
  }}
  .header .badge {{
    display: inline-block;
    margin-top: 12px;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #34d399;
  }}

  /* ── Panels ────────────────────────────────────── */
  .panels {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    height: calc(100vh - 140px);
    min-height: 600px;
  }}
  .panel {{
    position: relative;
    border: 1px solid rgba(100,100,180,0.1);
  }}
  .panel-label {{
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 10;
    padding: 6px 18px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    pointer-events: none;
  }}
  .panel-label.gt {{
    background: rgba(96, 165, 250, 0.15);
    border: 1px solid rgba(96, 165, 250, 0.3);
    color: #93c5fd;
  }}
  .panel-label.pred {{
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.3);
    color: #c4b5fd;
  }}
  .panel iframe {{
    width: 100%;
    height: 100%;
    border: none;
  }}

  /* ── Footer ────────────────────────────────────── */
  .footer {{
    text-align: center;
    padding: 10px;
    font-size: 0.7rem;
    color: #444;
    border-top: 1px solid rgba(100,100,180,0.1);
  }}

  @media (max-width: 900px) {{
    .panels {{ grid-template-columns: 1fr; height: auto; }}
    .panel {{ min-height: 500px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Quantum Bulk Reconstruction — AdS/CFT Correspondence</h1>
  <div class="meta">
    <span>Model <span class="value">Hybrid Quantum Neural Network</span></span>
    <span>Sample <span class="value">#{sample_idx}</span></span>
    <span>Qubits <span class="value">10</span></span>
    <span>Architecture <span class="value">Encoder → Quantum Bottleneck → Decoder</span></span>
  </div>
  <div class="badge">Relative L₂ Error: {rel_l2_pct:.2f}%</div>
</div>

<div class="panels">
  <div class="panel">
    <div class="panel-label gt">Ground Truth — Physics Simulation</div>
    <iframe src="{gt_file}" loading="eager"></iframe>
  </div>
  <div class="panel">
    <div class="panel-label pred">Quantum Hybrid Prediction</div>
    <iframe src="{pred_file}" loading="eager"></iframe>
  </div>
</div>

<div class="footer">
  Neural-AdS &middot; AdS/CFT Holographic Simulator &middot;
  Hybrid Quantum-Classical Architecture (10-qubit PennyLane)
</div>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [OK] Comparison page written: {output_path} ({size_kb:.1f} KB)")


# ═════════════════════════════════════════════════════════════════════
# EXPORT HELPER
# ═════════════════════════════════════════════════════════════════════

def export_figure(fig, path: str):
    """Write a Plotly figure to an HTML file."""
    fig.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
        config=dict(
            displaylogo=False,
            modeBarButtonsToRemove=["toImage", "sendDataToCloud"],
            responsive=True,
        ),
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [OK] Exported: {path} ({size_mb:.1f} MB)")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print()
    print("═" * 72)
    print("  Neural-AdS  |  Quantum Bulk Reconstruction Visualizer")
    print("═" * 72)

    # ── Pick sample ──────────────────────────────────────────────────
    if args.sample_idx < 0:
        args.sample_idx = np.random.randint(0, 1000)
        print(f"\n🎲 Random universe selected: Sample #{args.sample_idx}")

    # ── Load data ────────────────────────────────────────────────────
    print("\n── Step 1: Loading Data ──")
    x_input, y_truth = load_sample(args.data_dir, args.sample_idx)

    # ── Run quantum inference ────────────────────────────────────────
    print("\n── Step 2: Quantum Inference ──")
    y_pred = run_quantum_inference(args.checkpoint, x_input)

    # ── Compute Relative L2 Error ────────────────────────────────────
    print("\n── Step 3: Metrics ──")
    # Add batch dim for the metric function: [1, 20, 64, 64]
    rel_l2_pct = relative_l2_error(
        y_truth[np.newaxis],
        y_pred[np.newaxis],
    )
    print(f"  ✅ Relative L₂ Error = {rel_l2_pct:.4f} %")

    # ── Apply contrast enhancement ───────────────────────────────────
    gamma = args.enhance
    if gamma < 1.0:
        print(f"\n  [*] Contrast enhancement (γ = {gamma:.2f})")
        y_truth_viz = _enhance_contrast(y_truth.copy(), gamma)
        y_pred_viz  = _enhance_contrast(y_pred.copy(),  gamma)
    else:
        y_truth_viz = y_truth.copy()
        y_pred_viz  = y_pred.copy()

    # ── Lock z-axis to same range for fair comparison ────────────────
    global_max = max(
        float(np.abs(y_truth_viz).max()),
        float(np.abs(y_pred_viz).max()),
    ) * 1.05
    print(f"  [*] Locked z-axis: ±{global_max:.4f}")

    # ── Build figures ────────────────────────────────────────────────
    print(f"\n── Step 4: Building 3-D Holograms ──")
    print(f"     Colorscale : {args.colorscale}")
    print(f"     FPS        : {args.fps}")

    fig_gt = build_hologram_figure(
        y_truth_viz,
        colorscale=args.colorscale,
        fps=args.fps,
        title_suffix=f"Ground Truth (Sample #{args.sample_idx})",
        zmax_override=global_max,
    )

    fig_pred = build_hologram_figure(
        y_pred_viz,
        colorscale=args.colorscale,
        fps=args.fps,
        title_suffix=f"Quantum Hybrid Prediction (Rel L₂ = {rel_l2_pct:.2f}%)",
        zmax_override=global_max,
    )

    # ── Export ────────────────────────────────────────────────────────
    print(f"\n── Step 5: Exporting ──")
    gt_file   = "quantum_ground_truth.html"
    pred_file = "quantum_prediction.html"
    comp_file = "quantum_comparison.html"

    export_figure(fig_gt,   gt_file)
    export_figure(fig_pred, pred_file)
    build_comparison_html(gt_file, pred_file, args.sample_idx, rel_l2_pct, comp_file)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("═" * 72)
    print(f"  ✅ DONE — Open {comp_file} for side-by-side comparison")
    print(f"     Relative L₂ Error: {rel_l2_pct:.4f} %")
    print("═" * 72)
    print()


if __name__ == "__main__":
    main()
