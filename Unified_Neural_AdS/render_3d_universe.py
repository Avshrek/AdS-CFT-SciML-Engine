"""
render_3d_universe.py -- Interactive 3-D Holographic Bulk Projection
====================================================================

Visualises the output of the FNO3d Neural-AdS model as an animated 3-D
surface plot using Plotly.  The prediction tensor has shape:

    (1, 1, 20, 64, 64) -> (Batch, Channel, Time, X_Boundary, Z_Depth)

The surface is:
    - X-axis  = Boundary Space (X)
    - Y-axis  = Holographic Depth (Z)
    - Z-axis  = Scalar Field Amplitude (Phi)

The 20 time-slices are animated via Plotly animation frames so you can
watch the dual-source constructive interference erupt in real time.

Requirements
------------
    pip install torch plotly numpy

Usage
-----
    python render_3d_universe.py                                # model prediction
    python render_3d_universe.py --ground_truth                 # actual physics data
    python render_3d_universe.py --ground_truth --sample_idx 42 # different collision sample
    python render_3d_universe.py --dummy                        # synthetic demo
    python render_3d_universe.py --colorscale Plasma
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render FNO3d prediction as animated 3-D holographic surface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model .pth/.pt checkpoint. "
                         "If omitted, auto-selects latest from checkpoints/ or models/.")
    p.add_argument("--data_dir", type=str, default="data_collision_master",
                    help="Dataset directory (for real inference input).")
    p.add_argument("--sample_idx", type=int, default=0,
                    help="Sample index to load from the dataset.")
    p.add_argument("--ground_truth", action="store_true",
                    help="Visualise the GROUND TRUTH bulk field instead of model prediction. "
                         "Shows the actual physics simulation data.")
    p.add_argument("--dummy", action="store_true",
                    help="Skip model loading -- use synthetic dual-source collision data.")
    p.add_argument("--output", type=str, default="hologram_3d_bulk.html",
                    help="Output HTML filename.")
    p.add_argument("--colorscale", type=str, default="Viridis",
                    help="Plotly colorscale (Viridis, Plasma, Inferno, Magma, Cividis ...).")
    p.add_argument("--modes", type=int, default=8,
                    help="FNO3d Fourier modes (must match checkpoint).")
    p.add_argument("--width", type=int, default=20,
                    help="FNO3d hidden width (must match checkpoint).")
    p.add_argument("--n_layers", type=int, default=4,
                    help="FNO3d number of Fourier layers.")
    p.add_argument("--fps", type=int, default=4,
                    help="Animation speed in frames per second.")
    p.add_argument("--enhance", type=float, default=0.35,
                    help="Power-law contrast exponent (0.0-1.0). Lower = more bulk "
                         "detail visible. Set to 1.0 for raw/linear values.")
    p.add_argument("--res", type=int, default=64,
                    help="Output spatial resolution (X and Z). Values > 64 trigger "
                         "zero-shot super-resolution via interpolation.")
    p.add_argument("--smooth", action="store_true",
                    help="Apply linear interpolation smoothing to the visualization mesh.")
    p.add_argument("--zmax", type=float, default=None,
                    help="Lock the z-axis (amplitude) range to [-zmax, +zmax]. "
                         "Use the same value for ground truth and prediction "
                         "to get matching scales for visual comparison.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(explicit: str | None) -> str | None:
    """Return the path to the best available checkpoint."""
    if explicit and os.path.isfile(explicit):
        return explicit

    # Priority 1: checkpoints/ directory (full training checkpoints)
    ckpt_files = sorted(glob.glob(os.path.join("checkpoints", "*.pt")))
    if ckpt_files:
        return ckpt_files[-1]

    # Priority 2: any .pt checkpoint file in models/ (newest epoch wins)
    pt_files = sorted(glob.glob(os.path.join("models", "checkpoint_epoch_*.pt")))
    if pt_files:
        return pt_files[-1]

    # Priority 3: named weight files in models/
    for name in ["collision_final.pth", "collision_best.pth",
                 "collision_publication.pth", "collision_rigorous.pth",
                 "unified_time_final.pth"]:
        path = os.path.join("models", name)
        if os.path.isfile(path):
            return path

    return None


# ---------------------------------------------------------------------------
# Data loading -- matches train_publication.py pipeline
# ---------------------------------------------------------------------------

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

    return bdy_path, bulk_path


def _build_input_tensor(args, device):
    """
    Build a (1, 3, T, R, R) input tensor using GLOBAL standardisation
    that matches train_publication.py exactly.

    When args.res > 64, the boundary signal is upsampled via scipy
    interp1d so the FNO3d evaluates the learned operator at a higher
    spatial density (zero-shot super-resolution).

    Returns (input_tensor, bulk_stats) where bulk_stats = (y_mu, y_sig).
    """
    import torch
    from scipy.interpolate import interp1d

    T = 20
    R = args.res  # target resolution (may differ from native 64)

    bdy_path, bulk_path = _resolve_data_path(args.data_dir)

    if not os.path.isfile(bdy_path):
        print("  [!] No dataset found -- using dummy input tensor.")
        x_input = torch.randn(1, 3, T, R, R, device=device)
        return x_input, (0.0, 1.0)

    print("  [>] Loading data from: %s" % os.path.dirname(bdy_path))

    # Memory-map to compute GLOBAL stats (matches training exactly)
    bdy_mmap  = np.load(bdy_path,  mmap_mode="r")
    bulk_mmap = np.load(bulk_path, mmap_mode="r")

    N_full = bdy_mmap.shape[0]
    X_native = bdy_mmap.shape[2]  # 64

    # Global stats over the ENTIRE dataset (same as training)
    x_mu  = float(bdy_mmap.mean())
    x_sig = float(bdy_mmap.std())
    y_mu  = float(bulk_mmap.mean())
    y_sig = float(bulk_mmap.std())

    print("     Dataset: %d samples" % N_full)
    print("     Boundary  mu=%+.4f  sigma=%.4f" % (x_mu, x_sig))
    print("     Bulk      mu=%+.4f  sigma=%.4f" % (y_mu, y_sig))

    # Load single sample and standardise with GLOBAL stats
    idx = min(args.sample_idx, N_full - 1)
    print("     Using sample #%d" % idx)

    bdy_sample = bdy_mmap[idx].astype(np.float32).copy()  # (T, X_native)
    bdy_sample = (bdy_sample - x_mu) / (x_sig + 1e-8)

    # Zero-shot upscaling: interpolate boundary from X_native -> R
    if R != X_native:
        print("     Zero-shot upscale: %d -> %d" % (X_native, R))
        x_old = np.linspace(0, 1, X_native, dtype=np.float32)
        x_new = np.linspace(0, 1, R, dtype=np.float32)
        interp_fn = interp1d(x_old, bdy_sample, axis=1, kind="linear")
        bdy_sample = interp_fn(x_new).astype(np.float32)  # (T, R)

    # Tile to (T, R, R) and build 3-channel input
    wave = np.tile(bdy_sample[:, :, np.newaxis], (1, 1, R))  # (T, R, R)

    t_coord = np.linspace(0, 1, T, dtype=np.float32)[:, None, None]
    t_grid  = np.broadcast_to(t_coord, (T, R, R)).copy()

    z_coord = np.linspace(0, 1, R, dtype=np.float32)[None, None, :]
    z_grid  = np.broadcast_to(z_coord, (T, R, R)).copy()

    x_input = np.stack([wave, t_grid, z_grid], axis=0)  # (3, T, R, R)
    x_input = x_input[np.newaxis]                        # (1, 3, T, R, R)

    return torch.from_numpy(x_input).to(device), (y_mu, y_sig)


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(args: argparse.Namespace) -> np.ndarray:
    """
    Load the actual physics simulation bulk field directly from disk.
    No model needed -- this is what the FNO3d is *supposed* to predict.

    If args.res != 64, the field is upsampled via scipy.ndimage.zoom
    so it matches the requested visualisation resolution.

    Returns
    -------
    field : np.ndarray, shape (T, R, R)
    """
    from scipy.ndimage import zoom

    bdy_path, bulk_path = _resolve_data_path(args.data_dir)

    if not os.path.isfile(bulk_path):
        print("  [!] bulk_collision.npy not found -- falling back to dummy.")
        return _generate_synthetic_field()

    print("  [>] Loading GROUND TRUTH from: %s" % bulk_path)
    bulk_mmap = np.load(bulk_path, mmap_mode="r")
    N_full = bulk_mmap.shape[0]
    idx = min(args.sample_idx, N_full - 1)

    field = bulk_mmap[idx].astype(np.float32).copy()  # (T, 64, 64)
    print("     Dataset: %d samples, using #%d" % (N_full, idx))
    print("  [OK] Ground truth shape: %s" % str(field.shape))
    print("     Phi range: [%.4f, %.4f]" % (field.min(), field.max()))

    # Upscale to target resolution if needed
    R = args.res
    native = field.shape[1]  # 64
    if R != native:
        scale = R / native
        print("     Upscaling ground truth: %d -> %d (zoom=%.2fx)" % (native, R, scale))
        field = zoom(field, (1, scale, scale), order=1).astype(np.float32)
        print("     Upscaled shape: %s" % str(field.shape))

    # Per-depth diagnostics
    T, X, Z = field.shape
    print("     Amplitude by depth:")
    for z_idx in [0, Z // 4, Z // 2, 3 * Z // 4, Z - 1]:
        slc = field[:, :, z_idx]
        print("       Z=%2d: std=%.4f  range=[%+.4f, %+.4f]" % (
            z_idx, slc.std(), slc.min(), slc.max()))

    return field


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> np.ndarray:
    """
    Load model, build or load input, run forward pass,
    de-standardise output to physical amplitude.

    Returns
    -------
    field : np.ndarray, shape (T, R, R) where R = args.res
    """
    import torch
    from fno_architectures import FNO3d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = FNO3d(
        modes1=args.modes, modes2=args.modes, modes3=args.modes,
        width=args.width, n_layers=args.n_layers, in_channels=3,
    ).to(device)
    model.eval()

    # Load weights
    ckpt_path = _find_latest_checkpoint(args.checkpoint)
    if ckpt_path is None:
        print("  [!] No checkpoint found -- falling back to dummy data mode.")
        return _generate_synthetic_field()

    print("  [*] Loading checkpoint: %s" % ckpt_path)
    state = torch.load(ckpt_path, map_location=device)

    # Handle full-checkpoint dict vs raw state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        eff_epoch = state.get("effective_epoch", state.get("epoch", "?"))
        rel_l2 = state.get("rel_l2", None)
        print("     Effective epoch: %s" % eff_epoch)
        if rel_l2 is not None:
            print("     Checkpoint Rel L2: %.6f" % rel_l2)
    else:
        model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    print("  [OK] Model loaded -- %s params" % "{:,}".format(n_params))

    # Build input tensor (with global standardisation matching training)
    x_input, (y_mu, y_sig) = _build_input_tensor(args, device)

    # Forward pass
    print("  [~] Running inference ...")
    with torch.no_grad():
        pred = model(x_input)  # (1, 1, T, X, Z)

    # De-standardise to physical amplitude
    field = pred[0, 0].cpu().numpy()  # (T, X, Z)
    field = field * (y_sig + 1e-8) + y_mu

    print("  [OK] Prediction shape: %s" % str(field.shape))
    print("     Phi range (physical): [%.4f, %.4f]" % (field.min(), field.max()))

    # Per-depth diagnostics
    T, X, Z = field.shape
    print("     Amplitude by depth:")
    for z_idx in [0, Z // 4, Z // 2, 3 * Z // 4, Z - 1]:
        slc = field[:, :, z_idx]
        print("       Z=%2d: std=%.4f  range=[%+.4f, %+.4f]" % (
            z_idx, slc.std(), slc.min(), slc.max()))

    return field


# ---------------------------------------------------------------------------
# Synthetic dual-source collision (for --dummy mode)
# ---------------------------------------------------------------------------

def _generate_synthetic_field() -> np.ndarray:
    """
    Generate a beautiful synthetic dual-source collision field.

    Two Gaussian wave-packets propagate from opposite boundaries of
    the AdS bulk, collide at the centre, and produce constructive
    interference -- an eruption of amplitude in 3-D space.

    Returns
    -------
    field : np.ndarray, shape (20, 64, 64)
    """
    print("  [~] Generating synthetic dual-source collision field ...")

    T, X, Z = 20, 64, 64
    x = np.linspace(-1, 1, X, dtype=np.float64)
    z = np.linspace(-1, 1, Z, dtype=np.float64)
    xx, zz = np.meshgrid(x, z, indexing="ij")  # (X, Z)

    field = np.zeros((T, X, Z), dtype=np.float64)

    for t_idx in range(T):
        t = t_idx / (T - 1)  # 0 -> 1

        # Source A: starts at (x=-0.7, z=-0.7), propagates toward centre
        cx_a = -0.7 + 0.7 * t
        cz_a = -0.7 + 0.7 * t
        r2_a = (xx - cx_a) ** 2 + (zz - cz_a) ** 2
        sig_a = 0.12 + 0.04 * t
        wave_a = 1.2 * np.exp(-r2_a / (2 * sig_a ** 2)) * np.cos(8 * np.pi * t - 4 * np.pi * np.sqrt(r2_a + 0.01))

        # Source B: starts at (x=+0.7, z=+0.7), propagates toward centre
        cx_b = 0.7 - 0.7 * t
        cz_b = 0.7 - 0.7 * t
        r2_b = (xx - cx_b) ** 2 + (zz - cz_b) ** 2
        sig_b = 0.12 + 0.04 * t
        wave_b = 1.2 * np.exp(-r2_b / (2 * sig_b ** 2)) * np.cos(8 * np.pi * t - 4 * np.pi * np.sqrt(r2_b + 0.01))

        # Constructive interference
        field[t_idx] = wave_a + wave_b

        # Add collision splash at midpoint after t > 0.4
        if t > 0.4:
            splash_amp = 0.8 * (t - 0.4) / 0.6
            r2_c = xx ** 2 + zz ** 2
            splash = splash_amp * np.exp(-r2_c / 0.06) * np.cos(12 * np.pi * t)
            field[t_idx] += splash

    print("  [OK] Synthetic field generated: shape %s" % str(field.shape))
    print("     Phi range: [%.4f, %.4f]" % (field.min(), field.max()))
    return field.astype(np.float32)


# ---------------------------------------------------------------------------
# Contrast enhancement
# ---------------------------------------------------------------------------

def _enhance_contrast(field: np.ndarray, gamma: float = 0.35) -> np.ndarray:
    """
    Power-law contrast enhancement:  sign(x) * |x|^gamma

    With gamma < 1 this compresses the dynamic range, making small
    bulk-interior oscillations visible alongside the large boundary
    peaks. Without this the boundary spikes dominate the z-axis and
    the interior looks completely flat.

    gamma = 1.0  ->  no change (linear)
    gamma = 0.5  ->  square-root compression
    gamma = 0.35 ->  strong enhancement (default, reveals bulk structure)
    """
    if gamma >= 1.0:
        return field
    return np.sign(field) * np.abs(field) ** gamma


# ---------------------------------------------------------------------------
# 3-D Plotly Visualisation
# ---------------------------------------------------------------------------

def build_hologram_figure(
    field: np.ndarray,
    colorscale: str = "Viridis",
    fps: int = 4,
    title_suffix: str = "",
    zmax_override: float | None = None,
) -> go.Figure:
    """
    Build a publication-grade animated 3-D surface plot.

    Parameters
    ----------
    field : (T, X, Z) -- the spatiotemporal scalar field Phi.
    colorscale : Plotly colorscale name.
    fps : animation frame rate.
    title_suffix : extra text for the subtitle.
    """
    T, NX, NZ = field.shape
    frame_ms = int(1000 / fps)

    # Coordinate grids
    x_vals = np.linspace(0, 1, NX)
    z_vals = np.linspace(0, 1, NZ)

    # Global |Phi| range for locked z-axis
    if zmax_override is not None:
        phi_max = zmax_override
    else:
        phi_max = float(np.abs(field).max()) * 1.15
    phi_min = -phi_max

    # Initial surface (t = 0)
    surface_t0 = field[0].T  # (NZ, NX)

    # Use HTML entities for Greek letters in labels (rendered in browser)
    hover_tpl = (
        "<b>X (Boundary)</b>: %{x:.3f}<br>"
        "<b>Z (Depth)</b>: %{y:.3f}<br>"
        "<b>&#934;</b>: %{z:.4f}<br>"
        "<extra></extra>"
    )

    initial_surface = go.Surface(
        x=x_vals,
        y=z_vals,
        z=surface_t0,
        colorscale=colorscale,
        cmin=phi_min,
        cmax=phi_max,
        colorbar=dict(
            title=dict(
                text="&#934; (scalar field)",
                font=dict(size=14, color="#e0e0e0"),
            ),
            tickfont=dict(size=11, color="#cccccc"),
            len=0.6,
            thickness=18,
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        opacity=0.95,
        lighting=dict(
            ambient=0.35,
            diffuse=0.65,
            specular=0.3,
            roughness=0.4,
            fresnel=0.2,
        ),
        lightposition=dict(x=0, y=0, z=2),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="#ffffff",
                project_z=True,
            ),
        ),
        hovertemplate=hover_tpl,
    )

    # Animation frames
    frames = []
    slider_steps = []

    for t in range(T):
        surface_t = field[t].T  # (NZ, NX)

        frame = go.Frame(
            data=[go.Surface(
                x=x_vals,
                y=z_vals,
                z=surface_t,
                colorscale=colorscale,
                cmin=phi_min,
                cmax=phi_max,
                opacity=0.95,
                lighting=dict(
                    ambient=0.35,
                    diffuse=0.65,
                    specular=0.3,
                    roughness=0.4,
                    fresnel=0.2,
                ),
                lightposition=dict(x=0, y=0, z=2),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="#ffffff",
                        project_z=True,
                    ),
                ),
                hovertemplate=hover_tpl,
            )],
            name=str(t),
        )
        frames.append(frame)

        slider_steps.append(dict(
            args=[[str(t)], dict(
                frame=dict(duration=frame_ms, redraw=True),
                mode="immediate",
                transition=dict(duration=frame_ms // 2, easing="cubic-in-out"),
            )],
            label="t=%d" % t,
            method="animate",
        ))

    # Layout
    fig = go.Figure(data=[initial_surface], frames=frames)

    # Subtitle
    subtitle = "Dual-Source Constructive Interference"
    if title_suffix:
        subtitle += " &middot; " + title_suffix
    subtitle += " &middot; T &#8712; [0, 19]"

    # Play / Pause buttons
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            x=0.05,
            y=0.05,
            xanchor="left",
            yanchor="bottom",
            pad=dict(t=50, r=10),
            font=dict(color="#e0e0e0", size=13),
            bgcolor="rgba(40, 40, 60, 0.8)",
            bordercolor="rgba(100, 100, 140, 0.6)",
            buttons=[
                dict(
                    label="> Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_ms, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=frame_ms // 2,
                                            easing="cubic-in-out"),
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="|| Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        ),
    ]

    # Time slider
    sliders = [
        dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                prefix="Time Step: ",
                visible=True,
                xanchor="center",
                font=dict(size=14, color="#e0e0e0"),
            ),
            transition=dict(duration=frame_ms // 2, easing="cubic-in-out"),
            pad=dict(b=10, t=40),
            len=0.9,
            x=0.05,
            y=0,
            steps=slider_steps,
            bgcolor="rgba(40, 40, 60, 0.6)",
            activebgcolor="rgba(80, 80, 140, 0.9)",
            bordercolor="rgba(100, 100, 140, 0.4)",
            borderwidth=1,
            ticklen=4,
            font=dict(size=10, color="#aaaaaa"),
        ),
    ]

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                "<b>Neural-AdS Holographic Bulk Projection</b><br>"
                "<span style='font-size:13px; color:#999'>"
                + subtitle +
                "</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=20, color="#e8e8f0"),
        ),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="X  (Boundary Space)",
                    font=dict(size=13, color="#b0b0d0"),
                ),
                backgroundcolor="rgba(10, 10, 25, 0.9)",
                gridcolor="rgba(60, 60, 100, 0.35)",
                showbackground=True,
                tickfont=dict(size=10, color="#888"),
                range=[0, 1],
            ),
            yaxis=dict(
                title=dict(
                    text="Z  (Holographic Depth)",
                    font=dict(size=13, color="#b0b0d0"),
                ),
                backgroundcolor="rgba(10, 10, 25, 0.9)",
                gridcolor="rgba(60, 60, 100, 0.35)",
                showbackground=True,
                tickfont=dict(size=10, color="#888"),
                range=[0, 1],
            ),
            zaxis=dict(
                title=dict(
                    text="&#934;  (Amplitude)",
                    font=dict(size=13, color="#b0b0d0"),
                ),
                backgroundcolor="rgba(10, 10, 25, 0.9)",
                gridcolor="rgba(60, 60, 100, 0.35)",
                showbackground=True,
                tickfont=dict(size=10, color="#888"),
                range=[phi_min, phi_max],  # LOCKED -- no camera bounce
            ),
            camera=dict(
                eye=dict(x=2.1, y=-2.1, z=1.4),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.65),
            domain=dict(x=[0, 1], y=[0, 1])
        ),
        updatemenus=updatemenus,
        sliders=sliders,
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        margin=dict(l=0, r=0, t=50, b=0),
        height=750,
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif"),
        annotations=[
            dict(
                text=(
                    "Neural-AdS &middot; AdS/CFT Holographic Simulator &middot; "
                    "Fourier Neural Operator"
                ),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.02,
                xanchor="center",
                font=dict(size=10, color="#555566"),
            ),
        ],
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print()
    print("=" * 72)
    print("  Neural-AdS  |  3-D Holographic Bulk Renderer")
    print("=" * 72)

    import numpy as np
    if args.sample_idx == 0:  # Or whatever Opus set as the default
        args.sample_idx = np.random.randint(0, 1000)
        print(f"\n🎲 RANDOM UNIVERSE SELECTED: Sample #{args.sample_idx}")

    # Obtain the prediction field
    if args.dummy:
        field = _generate_synthetic_field()
        subtitle = "Synthetic Demo"
    elif args.ground_truth:
        field = load_ground_truth(args)
        subtitle = "Ground Truth (Sample #%d)" % args.sample_idx
    else:
        field = run_inference(args)
        subtitle = "FNO3d Prediction"

    # ── Step 1: Smooth BEFORE enhancement ──────────────────────────────
    # Critical: smoothing must happen on the raw physical-amplitude field
    # BEFORE contrast enhancement, because gamma < 1 amplifies noise.
    #   Raw noise ~0.01 → smoothed to ~0.002 → enhanced@0.7: 0.005 (invisible)
    # vs the wrong order:
    #   Raw noise ~0.01 → enhanced@0.7: 0.05  → smoothed: 0.04  (still visible!)
    if args.smooth:
        from scipy.ndimage import gaussian_filter
        sigma = max(1.0, args.res / 40.0)  # gentle: 1.6@64, 3.2@128
        print("")
        print("  [*] Gentle denoise on raw field (sigma=%.1f)" % sigma)
        print("     Before: peak=%.4f  range=[%.4f, %.4f]" % (
            np.abs(field).max(), field.min(), field.max()))
        for t in range(field.shape[0]):
            field[t] = gaussian_filter(field[t], sigma=sigma)
        print("     After:  peak=%.4f  range=[%.4f, %.4f]" % (
            np.abs(field).max(), field.min(), field.max()))

    # ── Step 2: Contrast enhancement ─────────────────────────────────
    gamma = args.enhance
    if gamma < 1.0:
        print("")
        print("  [*] Applying contrast enhancement (gamma=%.2f)" % gamma)
        print("     Before: Phi range [%.4f, %.4f]" % (field.min(), field.max()))
        field = _enhance_contrast(field, gamma)
        print("     After:  Phi range [%.4f, %.4f]" % (field.min(), field.max()))
        print("     (use --enhance 1.0 for raw linear values)")

    # Build the figure
    print("")
    print("  [~] Building 3-D animated hologram ...")
    print("     Resolution : %d x %d" % (field.shape[1], field.shape[2]))
    print("     Colorscale : %s" % args.colorscale)
    print("     FPS        : %d" % args.fps)
    print("     Time steps : %d" % field.shape[0])

    fig = build_hologram_figure(
        field,
        colorscale=args.colorscale,
        fps=args.fps,
        title_suffix=subtitle,
        zmax_override=args.zmax,
    )

    # Export
    output_path = args.output
    print("")
    print("  [>] Exporting interactive animation -> %s" % output_path)

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
        config=dict(
            displaylogo=False,
            modeBarButtonsToRemove=["toImage", "sendDataToCloud"],
            responsive=True,
        ),
    )

    file_size_mb = os.path.getsize(output_path) / 1e6
    print("  [OK] Done!  (%.1f MB)" % file_size_mb)

    # Also show in browser
    print("")
    print("  [>] Opening in browser ...")
    fig.show()

    print("")
    print("=" * 72)
    print("  [DONE] Hologram exported to: %s" % output_path)
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()