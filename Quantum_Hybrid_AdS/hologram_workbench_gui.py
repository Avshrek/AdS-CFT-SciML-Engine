"""
hologram_workbench_gui.py — Phase 3: Physicist's Interactive Workbench (Gradio)
================================================================================

A browser-based GUI for designing two-source quantum boundary interference
patterns and watching the Neural-AdS engine reconstruct the bulk
geometry — validated against the **exact LU-factorised Laplace solver**.

    Φ_total(x, t) = A1·sin(k1·x + ω1·t)  +  A2·sin(k2·x + ω2·t + φ2)

Output
~~~~~~
* **1×3 animated GIF**: Ground Truth (LU Solver) | Neural-AdS Prediction | Error
* **Quantitative metrics**: MAE, Max Error, Relative L₂

Model Selector
~~~~~~~~~~~~~~
* **Phase 2 (Single-Source)** — ``unified_time_final.pth``
* **Phase 3 (Collision)** — ``collision_rigorous.pth``

Launch
------
    python hologram_workbench_gui.py
    # Opens at http://localhost:7860
"""

from __future__ import annotations

import os
import sys
import time
import tempfile

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gradio as gr

from fno_architectures import FNO3d


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE   = 64
TIME_STEPS  = 20

MODELS = {
    "Phase 2 (Single-Source)": {
        "path":      os.path.join("models", "unified_time_final.pth"),
        "data_dir":  "data_holography_time",
        "bdy_file":  "boundary_time.npy",
        "bulk_file": "bulk_time.npy",
    },
    "Phase 3 (Collision)": {
        "path":      os.path.join("models", "collision_rigorous.pth"),
        "data_dir":  "data_collision_master",
        "bdy_file":  "bdy_collision.npy",
        "bulk_file": "bulk_collision.npy",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Exact Laplace Solver (LU-factorised sparse system)
# ─────────────────────────────────────────────────────────────────────────────

def _build_laplace_system(n: int):
    """Construct and LU-factorise the 2-D Laplacian with Dirichlet BCs.

    Returns
    -------
    solve    : callable — solve(b) returns the solution vector.
    top_row  : 1-D array of flattened indices for the top boundary row.
    """
    N = n * n
    main_diag    = -4.0 * np.ones(N)
    side_diag    = np.ones(N - 1)
    side_diag[np.arange(1, N) % n == 0] = 0
    up_down_diag = np.ones(N - n)

    A = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -n, n],
        shape=(N, N), format="lil",
    )

    grid_idx = np.arange(N).reshape(n, n)
    all_bdy = np.unique(np.concatenate((
        grid_idx[0, :], grid_idx[-1, :],
        grid_idx[:, 0], grid_idx[:, -1],
    )))
    for idx in all_bdy:
        A[idx, :] = 0
        A[idx, idx] = 1

    return spla.factorized(A.tocsc()), grid_idx[0, :]


def solve_ground_truth(boundary: np.ndarray) -> np.ndarray:
    """Solve ∇²Φ = 0 with Dirichlet BCs for every time-slice.

    Parameters
    ----------
    boundary : ``(T, X)`` — raw boundary wave.

    Returns
    -------
    bulk_truth : ``(T, X, Z)`` — exact solution via LU factorisation.
    """
    T, X = boundary.shape
    solve, top_row_idx = _build_laplace_system(X)

    bulk = np.empty((T, X, X), dtype=np.float32)
    b    = np.zeros(X * X, dtype=np.float64)

    for t_idx in range(T):
        b[:] = 0.0
        b[top_row_idx] = boundary[t_idx]
        sol = solve(b).reshape(X, X)
        bulk[t_idx] = sol.astype(np.float32)

    return bulk


# ─────────────────────────────────────────────────────────────────────────────
# Startup: models + normalisation stats
# ─────────────────────────────────────────────────────────────────────────────

print("🚀 Initialising Holographic Workbench …\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Device: {DEVICE}")

# Pre-factorise the LU solver at startup (takes ~0.3s)
print("🔧 Pre-factorising LU Laplace solver …")
_LU_SOLVE, _LU_TOP_ROW = _build_laplace_system(GRID_SIZE)
print("   ✔ Exact solver ready.\n")

LOADED_MODELS = {}
NORM_STATS    = {}

for label, cfg in MODELS.items():
    data_dir = cfg["data_dir"]
    bdy_path  = os.path.join(data_dir, cfg["bdy_file"])
    bulk_path = os.path.join(data_dir, cfg["bulk_file"])

    if os.path.isfile(bdy_path) and os.path.isfile(bulk_path):
        print(f"📦 Loading norm stats for {label} …")
        _bdy  = np.load(bdy_path,  mmap_mode="r")
        _bulk = np.load(bulk_path, mmap_mode="r")
        NORM_STATS[label] = dict(
            x_mean=float(_bdy.mean()),  x_std=float(_bdy.std()),
            y_mean=float(_bulk.mean()), y_std=float(_bulk.std()),
        )
        del _bdy, _bulk
        ns = NORM_STATS[label]
        print(f"   x̄={ns['x_mean']:.4f}  σx={ns['x_std']:.4f}  "
              f"ȳ={ns['y_mean']:.4f}  σy={ns['y_std']:.4f}")
    else:
        print(f"⚠  Data not found for {label} — skipping.")

    model_path = cfg["path"]
    if os.path.isfile(model_path):
        print(f"🧠 Loading FNO3d from {model_path} …")
        model = FNO3d(
            modes1=8, modes2=8, modes3=8,
            width=20, n_layers=4, in_channels=3,
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        LOADED_MODELS[label] = model
        _n = sum(p.numel() for p in model.parameters())
        print(f"   ✔ {_n:,} parameters loaded.\n")
    else:
        print(f"⚠  Checkpoint not found: {model_path}\n")

AVAILABLE_MODELS = sorted(
    [k for k in LOADED_MODELS if k in NORM_STATS],
    reverse=True,
)
if not AVAILABLE_MODELS:
    print("❌ No models available.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_two_source_boundary(
    k1: float, omega1: float, amp1: float,
    k2: float, omega2: float, amp2: float, phi2: float,
) -> np.ndarray:
    """Φ(x,t) = A1·sin(k1·x + ω1·t) + A2·sin(k2·x + ω2·t + φ2)."""
    x = np.linspace(0.0, 2 * np.pi, GRID_SIZE, dtype=np.float64)
    t = np.linspace(0.0, 2 * np.pi, TIME_STEPS, dtype=np.float64)
    t_col, x_row = t[:, np.newaxis], x[np.newaxis, :]
    wave = (amp1 * np.sin(k1 * x_row + omega1 * t_col)
          + amp2 * np.sin(k2 * x_row + omega2 * t_col + phi2))
    return wave.astype(np.float32)


def construct_volume(boundary: np.ndarray, norm: dict) -> np.ndarray:
    """Build (1, 3, T, X, Z) model input from a raw boundary."""
    T, X = boundary.shape
    Z = GRID_SIZE
    bdy_norm = (boundary - norm["x_mean"]) / (norm["x_std"] + 1e-8)
    wave_3d  = np.tile(bdy_norm[:, :, np.newaxis], (1, 1, Z))
    t_coord  = np.linspace(0.0, 1.0, T, dtype=np.float32)
    time_3d  = np.broadcast_to(t_coord[:, None, None], (T, X, Z)).copy()
    z_coord  = np.linspace(0.0, 1.0, Z, dtype=np.float32)
    depth_3d = np.broadcast_to(z_coord[None, None, :], (T, X, Z)).copy()
    return np.stack([wave_3d, time_3d, depth_3d], axis=0)[np.newaxis].astype(np.float32)


@torch.no_grad()
def run_inference(volume: np.ndarray, model_label: str) -> np.ndarray:
    """Forward pass → denormalised bulk prediction (T, X, Z)."""
    model = LOADED_MODELS[model_label]
    norm  = NORM_STATS[model_label]
    x = torch.from_numpy(volume).to(DEVICE)
    pred = model(x).squeeze().cpu().numpy()
    return pred * (norm["y_std"] + 1e-8) + norm["y_mean"]


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation → 1×3 Animated GIF (Ground Truth | Prediction | Error)
# ─────────────────────────────────────────────────────────────────────────────

def render_animation(
    boundary: np.ndarray,
    bulk_truth: np.ndarray,
    bulk_pred: np.ndarray,
    metrics: dict,
    model_label: str,
    fps: int = 5,
) -> str:
    """Create a 1×3 animated GIF with professional scientific layout."""
    T, X, Z = bulk_truth.shape
    error = np.abs(bulk_truth - bulk_pred)

    # Fixed colour limits across all frames
    bulk_vmin = min(bulk_truth.min(), bulk_pred.min())
    bulk_vmax = max(bulk_truth.max(), bulk_pred.max())
    err_vmax  = error.max()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6),
                              gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.patch.set_facecolor("white")

    # Title
    fig.suptitle(
        "Neural-AdS Holographic Bulk Reconstruction vs Exact Solver",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Metrics subtitle
    metrics_text = (
        f"MAE: {metrics['mae']:.4f} ({metrics['mae_pct']:.2f}%)   │   "
        f"Max Error: {metrics['max_err']:.4f} ({metrics['max_pct']:.2f}%)   │   "
        f"Rel L₂: {metrics['rel_l2']:.4f}   │   "
        f"Model: {model_label}"
    )
    fig.text(0.5, 0.93, metrics_text, ha="center", fontsize=10,
             fontstyle="italic", color="#263238",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                       edgecolor="#81C784", alpha=0.95))

    # --- Panel 0: Ground Truth ────────────────────────────────────────────
    im0 = axes[0].imshow(
        bulk_truth[0], cmap="inferno", vmin=bulk_vmin, vmax=bulk_vmax,
        origin="upper", aspect="equal",
    )
    axes[0].set_title("Ground Truth (LU Solver)", fontsize=13,
                       fontweight="bold", pad=10, color="#1B5E20")
    axes[0].set_xlabel("Boundary (x)")
    axes[0].set_ylabel("Radial Depth (z)")
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im0, cax=cax0)

    # --- Panel 1: Neural-AdS Prediction ──────────────────────────────────
    im1 = axes[1].imshow(
        bulk_pred[0], cmap="inferno", vmin=bulk_vmin, vmax=bulk_vmax,
        origin="upper", aspect="equal",
    )
    axes[1].set_title("Neural-AdS Prediction (FNO3d)", fontsize=13,
                       fontweight="bold", pad=10, color="#1565C0")
    axes[1].set_xlabel("Boundary (x)")
    axes[1].set_ylabel("Radial Depth (z)")
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax1)

    # --- Panel 2: Absolute Error ─────────────────────────────────────────
    im2 = axes[2].imshow(
        error[0], cmap="hot", vmin=0, vmax=err_vmax,
        origin="upper", aspect="equal",
    )
    err_title = axes[2].set_title("Absolute Error", fontsize=13,
                                    fontweight="bold", pad=10, color="#B71C1C")
    axes[2].set_xlabel("Boundary (x)")
    axes[2].set_ylabel("Radial Depth (z)")
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax2)

    # Time badge (lower-left corner)
    time_badge = fig.text(
        0.02, 0.02, "", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.95),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.90], w_pad=2.5)

    def update(t):
        im0.set_data(bulk_truth[t])
        im1.set_data(bulk_pred[t])
        im2.set_data(error[t])
        mae_t = np.mean(error[t])
        axes[2].set_title(f"Absolute Error  (MAE: {mae_t:.4f})",
                           fontsize=13, fontweight="bold", pad=10, color="#B71C1C")
        time_badge.set_text(f"  t = {t + 1}/{T}  ")
        return im0, im1, im2, time_badge

    anim = animation.FuncAnimation(fig, update, frames=T,
                                    interval=1000 // fps, blit=False)

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    anim.save(tmp.name, writer="pillow", fps=fps, dpi=130)
    plt.close(fig)
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Gradio callback
# ─────────────────────────────────────────────────────────────────────────────

def holographic_inference(
    model_label: str,
    k1: float, omega1: float, amp1: float,
    k2: float, omega2: float, amp2: float, phi2: float,
) -> tuple[str, str]:
    """End-to-end: boundary → LU ground truth → FNO inference → comparison."""
    if model_label not in LOADED_MODELS:
        return None, f"❌ Model **{model_label}** is not available."

    norm = NORM_STATS[model_label]

    # ── Generate boundary ─────────────────────────────────────────────
    boundary = generate_two_source_boundary(k1, omega1, amp1,
                                             k2, omega2, amp2, phi2)

    # ── Exact ground truth via LU solver ──────────────────────────────
    t0 = time.perf_counter()
    bulk_truth = solve_ground_truth(boundary)
    solver_time = time.perf_counter() - t0

    # ── Neural-AdS inference ──────────────────────────────────────────
    t1 = time.perf_counter()
    volume    = construct_volume(boundary, norm)
    bulk_pred = run_inference(volume, model_label)
    nn_time   = time.perf_counter() - t1

    speedup = solver_time / (nn_time + 1e-9)

    # ── Quantitative metrics ──────────────────────────────────────────
    error     = np.abs(bulk_truth - bulk_pred)
    mae       = float(np.mean(error))
    max_err   = float(np.max(error))
    data_range = float(bulk_truth.max() - bulk_truth.min()) + 1e-8
    mae_pct   = (mae / data_range) * 100
    max_pct   = (max_err / data_range) * 100
    rel_l2    = float(
        np.linalg.norm(bulk_pred.ravel() - bulk_truth.ravel())
        / (np.linalg.norm(bulk_truth.ravel()) + 1e-12)
    )

    metrics = dict(mae=mae, max_err=max_err, mae_pct=mae_pct,
                   max_pct=max_pct, rel_l2=rel_l2)

    print(f"⚡ [{model_label}] Solver: {solver_time:.3f}s  |  "
          f"FNO: {nn_time:.3f}s  |  Speedup: {speedup:.0f}×  |  "
          f"MAE: {mae:.4f} ({mae_pct:.2f}%)  |  Rel L₂: {rel_l2:.4f}")

    # ── Render 1×3 animation ──────────────────────────────────────────
    gif_path = render_animation(boundary, bulk_truth, bulk_pred,
                                 metrics, model_label)

    # ── Metrics card (Markdown) ───────────────────────────────────────
    metrics_md = (
        f"## 📊 Accuracy Report — Neural-AdS vs Exact Solver\n\n"
        f"| Metric | Value |\n"
        f"|:---|:---|\n"
        f"| **Model** | `{model_label}` |\n"
        f"| **Mean Absolute Error** | `{mae:.6f}`  ({mae_pct:.2f}% of range) |\n"
        f"| **Max Absolute Error** | `{max_err:.6f}`  ({max_pct:.2f}% of range) |\n"
        f"| **Relative L₂ Error** | `{rel_l2:.6f}` |\n"
        f"| **LU Solver Time** | `{solver_time:.3f}s` |\n"
        f"| **FNO Inference Time** | `{nn_time:.3f}s` |\n"
        f"| **Speedup** | `{speedup:.0f}×` |\n"
        f"\n> The ground truth is computed by solving ∇²Φ = 0 via "
        f"LU-factorised sparse matrix decomposition at every time-slice. "
        f"The Neural-AdS prediction is a single forward pass through the "
        f"Fourier Neural Operator.\n"
    )

    return gif_path, metrics_md


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Holographic Workbench",
        theme=gr.themes.Soft(primary_hue="purple"),
    ) as app:

        gr.Markdown(
            "# 🌌 Holographic Workbench\n"
            "Design two interfering quantum sources. The engine computes the "
            "**exact ground truth** via LU sparse solver and the **Neural-AdS "
            "prediction** via FNO3d — side by side.\n\n"
            "**Φ(x, t) = A₁·sin(k₁x + ω₁t)  +  A₂·sin(k₂x + ω₂t + φ₂)**"
        )

        # ── Model Selector ───────────────────────────────────────────────
        with gr.Row():
            model_selector = gr.Radio(
                choices=AVAILABLE_MODELS,
                value=AVAILABLE_MODELS[0],
                label="🧠 Model",
                info="Phase 2 = single-source  |  Phase 3 = dual-source collision",
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔵 Source A")
                k1    = gr.Slider(1.0, 8.0, value=2.0, step=0.1,
                                   label="k₁  (Wavenumber)")
                omega1 = gr.Slider(0.5, 4.0, value=1.0, step=0.1,
                                    label="ω₁  (Frequency)")
                amp1   = gr.Slider(0.1, 0.75, value=0.5, step=0.05,
                                    label="A₁  (Amplitude)")

            with gr.Column():
                gr.Markdown("### 🟣 Source B")
                k2    = gr.Slider(1.0, 8.0, value=5.0, step=0.1,
                                   label="k₂  (Wavenumber)")
                omega2 = gr.Slider(0.5, 4.0, value=2.5, step=0.1,
                                    label="ω₂  (Frequency)")
                amp2   = gr.Slider(0.1, 0.75, value=0.4, step=0.05,
                                    label="A₂  (Amplitude)")
                phi2   = gr.Slider(0.0, 6.283, value=0.0, step=0.05,
                                    label="φ₂  (Phase Offset)")

        generate_btn = gr.Button("🚀 Generate & Compare",
                                  variant="primary", size="lg")

        with gr.Row():
            output_gif = gr.Image(
                label="Ground Truth  |  Neural-AdS Prediction  |  Error",
                type="filepath",
            )
            metrics_panel = gr.Markdown(
                value="*Press 🚀 to compute ground truth and run Neural-AdS inference.*",
                label="Accuracy Report",
            )

        generate_btn.click(
            fn=holographic_inference,
            inputs=[model_selector, k1, omega1, amp1, k2, omega2, amp2, phi2],
            outputs=[output_gif, metrics_panel],
        )

        gr.Markdown(
            "---\n"
            "*Unified Neural-AdS · Phase 3 — Physicist's Workbench*  \n"
            "*Ground truth: LU-factorised ∇²Φ = 0  ·  Surrogate: FNO3d*"
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    print("🌐 Launching Holographic Workbench …\n")
    app.launch(share=False)
