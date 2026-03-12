"""
app.py  --  Neural-AdS Professor Dashboard  (3-Tab Edition)
============================================================

A sleek, dark-mode Streamlit dashboard showcasing a 3D Fourier Neural
Operator trained on AdS/CFT dual-source collision physics.

Run:
    pip install streamlit
    cd "d:\\Unified Neural-AdS"
    streamlit run app.py

Files expected in the project root:
    ground_truth.html    - Plotly 3D surface (exact solver)
    prediction.html      - Plotly 3D surface (FNO3d prediction)
    results/collision_proof.png   - statistical proof figure
    gt_animation.gif     - ground truth time evolution
    pred_animation.gif   - FNO3d predicted time evolution
    fno_architectures.py - model architecture
    models/checkpoint_epoch_470.pt  - trained weights
    data_collision_master/         - training data (for standardisation stats)
"""

from __future__ import annotations

import os
import sys
import glob
import time

import numpy as np
import scipy
import torch
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Make sure the project root is on PYTHONPATH so we can import
# fno_architectures from the same directory as app.py.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fno_architectures import FNO3d


# ═══════════════════════════════════════════════════════════════════════
# Page config  (MUST be the very first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Neural-AdS  |  Holographic Bulk Kinematics",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADER  --  loaded ONCE, stays in GPU/CPU memory
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading FNO3d model weights…")
def load_model(
    checkpoint: str = "models/collision_best.pth",
    fallback: str = "models/checkpoint_epoch_470.pt",
):
    """
    Instantiate FNO3d and load trained weights.

    Tries the epoch-style checkpoint first (which wraps the state dict
    inside a dict with key ``model_state_dict``), then falls back to a
    raw state-dict ``.pth`` file.
    """
    device = torch.device("cpu")  # CPU for the dashboard

    model = FNO3d(
        modes1=8, modes2=8, modes3=8,
        width=20, n_layers=4, in_channels=3,
    ).to(device)

    # Resolve path
    ckpt_path = checkpoint if os.path.isfile(checkpoint) else fallback
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint} or {fallback}")

    state = torch.load(ckpt_path, map_location=device)

    # Handle full-checkpoint dict vs raw state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, device, n_params, os.path.basename(ckpt_path)


@st.cache_resource(show_spinner="Loading dataset statistics…")
def load_norm_stats(data_dir: str = "data_collision_master", collision: bool = True) -> dict:
    bdy_file  = "bdy_collision.npy" if collision else "boundary_time.npy"
    bulk_file = "bulk_collision.npy" if collision else "bulk_time.npy"
    
    bdy_path = os.path.join(data_dir, bdy_file)
    bulk_path = os.path.join(data_dir, bulk_file)
    
    if not os.path.isfile(bdy_path) or not os.path.isfile(bulk_path):
        return {"x_mean": 0.0006, "x_std": 0.4668,
                "y_mean": -0.0002, "y_std": 0.0834}
                
    bdy  = np.load(bdy_path,  mmap_mode="r")
    bulk = np.load(bulk_path, mmap_mode="r")
    stats = dict(x_mean=float(bdy.mean()), x_std=float(bdy.std()),
                 y_mean=float(bulk.mean()), y_std=float(bulk.std()))
    del bdy, bulk
    return stats


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════

TIME_STEPS = 20

def generate_collision_boundary(k1, omega1, amp1, k2, omega2, amp2,
                                 phi2, grid_size=64) -> np.ndarray:
    x = np.linspace(0.0, 2 * np.pi, grid_size, dtype=np.float64)
    t = np.linspace(0.0, 2 * np.pi, TIME_STEPS, dtype=np.float64)
    t_col, x_row = t[:, None], x[None, :]
    wave = (amp1 * np.sin(k1 * x_row + omega1 * t_col)
          + amp2 * np.sin(k2 * x_row + omega2 * t_col + phi2))
    return wave.astype(np.float32)

def construct_input_volume(boundary: np.ndarray, norm: dict, grid_size=64) -> np.ndarray:
    T, X = boundary.shape
    Z = grid_size
    bdy_norm = (boundary - norm["x_mean"]) / (norm["x_std"] + 1e-8)
    wave_3d  = np.tile(bdy_norm[:, :, None], (1, 1, Z))
    t_coord  = np.linspace(0.0, 1.0, T, dtype=np.float32)
    time_3d  = np.broadcast_to(t_coord[:, None, None], (T, X, Z)).copy()
    z_coord  = np.linspace(0.0, 1.0, Z, dtype=np.float32)
    depth_3d = np.broadcast_to(z_coord[None, None, :], (T, X, Z)).copy()
    return np.stack([wave_3d, time_3d, depth_3d], axis=0)[None].astype(np.float32)

@torch.no_grad()
def infer(model, volume, device, y_mean, y_std):
    x = torch.from_numpy(volume).to(device)
    pred = model(x).squeeze().cpu().numpy()
    return pred * (y_std + 1e-8) + y_mean


# ═══════════════════════════════════════════════════════════════════════
# HELPER  --  embed Plotly HTML with proper sizing
# ═══════════════════════════════════════════════════════════════════════

def embed_plotly(filepath: str, label: str, css_class: str, height: int = 750):
    """Read an exported Plotly HTML file and embed it interactively."""
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as fh:
            html = fh.read()
        st.markdown(
            f'<div class="plotly-label {css_class}">{label}</div>'
            f'<div class="plotly-wrap">',
            unsafe_allow_html=True,
        )
        components.html(html, height=height, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"**{label}** — file not found: `{filepath}`")
        st.caption("Generate it with `render_3d_universe.py`.")


# ═══════════════════════════════════════════════════════════════════════
# HELPER  --  build a Plotly surface figure
# ═══════════════════════════════════════════════════════════════════════

def make_animated_surface_figure(
    field_3d: np.ndarray,
    title: str = "",
    zmin: float = -1.5,
    zmax: float = 1.5,
) -> go.Figure:
    """Create a publication-grade 3D animated surface from a (T, NX, NZ) array."""
    T, NX, NZ = field_3d.shape
    x_vals = np.linspace(0, 1, NX)
    z_vals = np.linspace(0, 1, NZ)

    hover_tpl = (
        "<b>X</b>: %{x:.3f}<br>"
        "<b>Z</b>: %{y:.3f}<br>"
        "<b>Φ</b>: %{z:.4f}<br>"
        "<extra></extra>"
    )

    initial_surface = go.Surface(
        z=field_3d[0].T,  # (NZ, NX)
        x=x_vals,
        y=z_vals,
        colorscale="Viridis",
        cmin=zmin, cmax=zmax,
        showscale=True,
        colorbar=dict(
            title=dict(text="Φ", font=dict(size=14, color="#ccc")),
            tickfont=dict(size=10, color="#aaa"),
            len=0.75,
        ),
        lighting=dict(
            ambient=0.35, diffuse=0.6, specular=0.3,
            roughness=0.7, fresnel=0.3,
        ),
        contours_z=dict(show=True, usecolormap=True, project_z=True),
        hovertemplate=hover_tpl,
    )

    frames = []
    slider_steps = []
    frame_ms = 150

    for t in range(T):
        surface_t = field_3d[t].T
        frame = go.Frame(
            data=[go.Surface(
                z=surface_t,
                x=x_vals,
                y=z_vals,
                colorscale="Viridis",
                cmin=zmin, cmax=zmax,
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
            label=f"t={t}",
            method="animate",
        ))

    fig = go.Figure(data=[initial_surface], frames=frames)

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
                            transition=dict(duration=frame_ms // 2, easing="cubic-in-out"),
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
            font=dict(size=10, color="#aaa"),
        ),
    ]

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#e4e4f0"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="X (Boundary Space)",
                range=[0, 1],
                backgroundcolor="rgba(10,10,25,0.9)",
                gridcolor="rgba(60,60,100,0.3)",
                showbackground=True,
                tickfont=dict(size=10, color="#888"),
            ),
            yaxis=dict(
                title="Z (Holographic Depth)",
                range=[0, 1],
                backgroundcolor="rgba(10,10,25,0.9)",
                gridcolor="rgba(60,60,100,0.3)",
                showbackground=True,
                tickfont=dict(size=10, color="#888"),
            ),
            zaxis=dict(
                title="Φ (Amplitude)",
                range=[zmin, zmax],
                backgroundcolor="rgba(10,10,25,0.9)",
                gridcolor="rgba(60,60,100,0.3)",
                showbackground=True,
                tickfont=dict(size=10, color="#888"),
            ),
            camera=dict(
                eye=dict(x=2.1, y=-2.1, z=1.4),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.65),
        ),
        updatemenus=updatemenus,
        sliders=sliders,
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        margin=dict(l=0, r=0, t=50, b=10),
        height=700,
        font=dict(family="Inter, sans-serif", color="#ccc"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# CSS  --  dark scientific aesthetic
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-void:       #060612;
    --bg-primary:    #0a0a1a;
    --bg-card:       #111128;
    --border:        #222255;
    --border-glow:   rgba(0,200,255,0.12);
    --text-primary:  #e4e4f0;
    --text-muted:    #8888aa;
    --cyan:          #00d4ff;
    --purple:        #8855ff;
    --green:         #00ff88;
    --gold:          #ffd700;
}

.stApp {
    background: linear-gradient(160deg, #060612 0%, #0e0830 45%, #0a1828 100%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.6rem; max-width: 1500px; }

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: var(--bg-card); border-radius: 12px;
    padding: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    height: 44px; border-radius: 8px !important;
    color: var(--text-muted) !important; font-weight: 500 !important;
    font-size: 0.92rem !important; padding: 0 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(136,85,255,0.15)) !important;
    color: var(--cyan) !important; font-weight: 600 !important;
    border-bottom: none !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.25s, box-shadow 0.25s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 12px 36px rgba(0,180,255,0.12);
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important; font-size: 0.78rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.9rem !important; font-weight: 700 !important;
}

/* Plotly embed */
.plotly-wrap {
    border: 1px solid var(--border); border-radius: 14px;
    overflow: hidden; box-shadow: 0 6px 28px rgba(0,0,0,0.5);
    margin-bottom: 0.6rem;
}
.plotly-label {
    text-align: center; font-weight: 600; font-size: 0.95rem;
    padding: 12px 0 8px; letter-spacing: 0.05em;
}
.label-truth { color: var(--green); }
.label-pred  { color: var(--cyan);  }

/* Hero */
.hero { text-align: center; padding: 1.6rem 1rem 1rem; }
.hero-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.02em; margin-bottom: 0.15rem;
}
.hero-sub { font-size: 1.1rem; color: var(--text-muted); margin-bottom: 0.8rem; }
.hero-badge {
    display: inline-block; background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2); color: var(--cyan);
    font-size: 0.72rem; font-weight: 600; padding: 4px 16px;
    border-radius: 20px; margin-bottom: 0.8rem;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.hero-abstract {
    font-size: 0.98rem; color: var(--text-muted);
    max-width: 900px; margin: 0 auto; line-height: 1.7;
}
.hero-abstract strong { color: var(--text-primary); }

/* Utility */
.divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1rem 0;
}
.sec-title {
    font-size: 1.2rem; font-weight: 600; color: var(--cyan);
    margin: 1.2rem 0 0.4rem; letter-spacing: 0.03em;
}
.sec-desc {
    font-size: 0.88rem; color: var(--text-muted);
    margin-bottom: 1rem; line-height: 1.6;
}
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 22px 26px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 0.8rem;
}
.card h4 { color: var(--purple); margin-bottom: 0.5rem; }
.card code {
    background: rgba(136,85,255,0.1); padding: 2px 8px;
    border-radius: 4px; font-family: 'JetBrains Mono', monospace;
    color: var(--purple); font-size: 0.86rem;
}
.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; background: var(--bg-card) !important;
}
.stForm {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important; background: var(--bg-card) !important;
    padding: 16px !important;
}
.footer {
    text-align: center; color: var(--text-muted);
    font-size: 0.75rem; padding: 2rem 0 1rem; opacity: 0.5;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# HERO HEADER  +  METRICS
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-badge">Undergraduate Research &middot; Manipal University Jaipur &middot; Computer Science</div>
    <div class="hero-title">Neural-AdS</div>
    <p class="hero-sub">
        Learning Holographic Bulk Kinematics with 3D Fourier Neural Operators
    </p>
    <p class="hero-abstract">
        A <strong>3D Fourier Neural Operator (FNO3d)</strong> trained as a
        surrogate model for holographic bulk reconstruction under the
        <strong>AdS/CFT correspondence</strong>. It maps 2D boundary quantum-state
        collisions directly into a 3D gravitational bulk, bypassing the
        <strong>extremely slow LU Factorisation</strong> required by traditional exact
        solvers &mdash; achieving a <strong>1,000&times; computational speedup</strong>
        while maintaining high fidelity to the governing curved-spacetime physics.
    </p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("⚡  Speedup", "1,000×", "vs LU Factorisation")
with m2:
    st.metric("📉  Rel L₂ Error", "6.84 %", "Validation")
with m3:
    st.metric("🧮  Physics Loss", "0.00356", "Curved Laplacian")
with m4:
    st.metric("🔬  Zero-Shot SR", "64 → 128", "Super-Resolution")

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "🌌  3D Spacetime Validation",
    "🎬  Temporal Dynamics",
    "🧪  Interactive Holographic Sandbox",
])


# ─────────────────────────────────────────────────────────────  TAB 1
with tab1:
    st.markdown(
        '<div class="sec-title">Side-by-Side Holographic Bulk Projection</div>'
        '<div class="sec-desc">'
        'Fully interactive 3D surfaces &mdash; drag to rotate, scroll to zoom. '
        'Both plots share a locked z-axis for direct comparison.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_gt, col_pr = st.columns(2)
    with col_gt:
        embed_plotly("ground_truth.html",
                     "Exact Solver (Ground Truth)", "label-truth", 750)
    with col_pr:
        embed_plotly("prediction.html",
                     "FNO3d Surrogate (Epoch 553)", "label-pred", 750)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.expander("📊  Statistical Proof (1,000 Samples)", expanded=False):
        proof = os.path.join("results", "collision_proof.png")
        if os.path.isfile(proof):
            st.image(proof, use_container_width=True,
                     caption="Aggregate evaluation: MAE, Rel L₂, representative slices.")
        else:
            st.info(f"`{proof}` not found.")

        phys = os.path.join("results", "physics_consistency.png")
        if os.path.isfile(phys):
            st.image(phys, use_container_width=True,
                     caption="Physics consistency: PDE residual & energy diagnostics.")


# ─────────────────────────────────────────────────────────────  TAB 2
with tab2:
    st.markdown(
        '<div class="sec-title">Temporal Dynamics — Wave Collision in AdS Bulk</div>'
        '<div class="sec-desc">'
        'Watch boundary quantum states collapse and interfere as they '
        'propagate into the holographic bulk over 20 discrete time steps.'
        '</div>',
        unsafe_allow_html=True,
    )

    anim_gt   = "gt_animation.gif"
    anim_pred = "pred_animation.gif"

    # Auto-detect fallback GIFs from results/
    if not os.path.isfile(anim_gt):
        for c in ["results/collision_evolution.gif",
                   "results/spacetime_evolution.gif",
                   "results/interference.gif"]:
            if os.path.isfile(c):
                anim_gt = c; break

    if not os.path.isfile(anim_pred):
        for c in ["results/deep_pulse.gif",
                   "results/collision_sandbox.gif",
                   "results/speed_test.gif"]:
            if os.path.isfile(c):
                anim_pred = c; break

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown('<div class="plotly-label label-truth">Ground Truth Dynamics</div>',
                    unsafe_allow_html=True)
        if os.path.isfile(anim_gt):
            st.image(anim_gt, use_container_width=True)
        else:
            st.warning("Place `gt_animation.gif` in the project root.")
    with ac2:
        st.markdown('<div class="plotly-label label-pred">FNO3d Predicted Dynamics</div>',
                    unsafe_allow_html=True)
        if os.path.isfile(anim_pred):
            st.image(anim_pred, use_container_width=True)
        else:
            st.warning("Place `pred_animation.gif` in the project root.")

    # Extra gallery
    extra_gifs = sorted(glob.glob("results/*.gif"))
    shown = {os.path.abspath(anim_gt), os.path.abspath(anim_pred)}
    extra_gifs = [g for g in extra_gifs if os.path.abspath(g) not in shown]
    if extra_gifs:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        with st.expander("🎞️  Additional Animations", expanded=False):
            for i in range(0, len(extra_gifs), 3):
                row = st.columns(3)
                for j, col in enumerate(row):
                    idx = i + j
                    if idx < len(extra_gifs):
                        with col:
                            name = os.path.splitext(
                                os.path.basename(extra_gifs[idx])
                            )[0].replace("_", " ").title()
                            st.caption(name)
                            st.image(extra_gifs[idx], use_container_width=True)


# ─────────────────────────────────────────────────────────────  TAB 3
# INTERACTIVE HOLOGRAPHIC SANDBOX  --  Real PyTorch inference
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        '<div class="sec-title">Interactive Holographic Sandbox</div>'
        '<div class="sec-desc">'
        'Configure boundary wave parameters and run the trained FNO3d '
        'through real PyTorch inference. The model evaluates the learned '
        "Green's function at arbitrary boundary conditions in &lt; 50 ms."
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Load model & data stats (cached, runs only once) ─────────
    try:
        model, device, n_params, ckpt_name = load_model()
        stats = load_norm_stats()
        model_ready = True
    except Exception as e:
        model_ready = False
        model_error = str(e)

    ctrl_col, out_col = st.columns([1, 2])

    # ── Left: parameter controls ───────────────────────────────────
    with ctrl_col:
        with st.form("boundary_params", clear_on_submit=False):
            st.markdown("##### Boundary Wave Parameters")
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            st.markdown("**Source 1**")
            a1 = st.slider("Amplitude  $A_1$", 0.0, 2.0, 1.0, 0.05, key="a1")
            k1 = st.slider("Frequency  $k_1$", 1, 20, 5, key="k1")
            w1 = st.slider("Phase  $\\omega_1$", 0.0, 6.28, 0.0, 0.1, key="w1")

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            st.markdown("**Source 2**")
            a2 = st.slider("Amplitude  $A_2$", 0.0, 2.0, 0.8, 0.05, key="a2")
            k2 = st.slider("Frequency  $k_2$", 1, 20, 7, key="k2")
            w2 = st.slider("Phase  $\\omega_2$", 0.0, 6.28, 3.14, 0.1, key="w2")

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            st.markdown("**Rendering Options**")
            res = st.select_slider("Resolution (Zero-Shot SR)", options=[64, 128, 256], value=64)
            enhance = st.slider("Contrast Enhancement", 0.05, 1.0, 0.35, 0.05)
            smooth = st.checkbox("Gaussian Smoothing", value=False)

            # The UI projection is fully animated across 20 frames so we don't need a static t_step slider!

            submitted = st.form_submit_button(
                "⚡  Generate Bulk Prediction",
                use_container_width=True,
            )

    # ── Right: output area ─────────────────────────────────────────
    with out_col:
        if not model_ready:
            st.error(f"**Model failed to load:** {model_error}")
            st.info("Ensure `fno_architectures.py` and the checkpoint exist.")

        elif not submitted:
            st.markdown("""
            <div class="card" style="margin-top:0.8rem;">
                <h4>🔮  Live Inference</h4>
                <p style="color:#8888aa; line-height:1.7;">
                    Adjust the boundary wave parameters on the left and click
                    <strong style="color:#00d4ff;">Generate Bulk Prediction</strong>
                    to push data through the real PyTorch FNO3d weights.
                </p>
                <p style="color:#8888aa; line-height:1.7; margin-top:0.6rem;">
                    The neural operator evaluates in
                    <strong style="color:#00ff88;">&lt; 50 ms</strong>
                    (vs ~60 s for an exact LU solver). No synthetic
                    placeholders &mdash; this is real model inference.
                </p>
                <table style="color:#888; font-size:0.85rem; margin-top:0.8rem; line-height:1.8;">
                    <tr><td>Checkpoint</td><td><code>""" + ckpt_name + """</code></td></tr>
                    <tr><td>Parameters</td><td><code>""" + f"{n_params:,}" + """</code></td></tr>
                    <tr><td>Device</td><td><code>""" + str(device) + """</code></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── REAL PYTORCH INFERENCE ──────────────────────────────
            with st.spinner("Running FNO3d Inference…"):
                t0 = time.perf_counter()

                # 1. Build boundary tensor
                boundary = generate_collision_boundary(
                    k1, w1, a1, k2, w2, a2, w2, grid_size=res
                )
                
                # 2. Construct volume
                volume = construct_input_volume(boundary, stats, grid_size=res)

                # 3. Forward pass through the trained model
                bulk_pred = infer(model, volume, device, stats["y_mean"], stats["y_std"])

                elapsed_ms = (time.perf_counter() - t0) * 1000

                # 4. Optional Gaussian Smoothing
                if smooth:
                    from scipy.ndimage import gaussian_filter
                    sigma = max(1.0, res / 40.0)
                    for t_idx in range(bulk_pred.shape[0]):
                        bulk_pred[t_idx] = gaussian_filter(bulk_pred[t_idx], sigma=sigma)
                        
                # 5. Contrast Enhancement
                if enhance < 1.0:
                    bulk_pred = np.sign(bulk_pred) * np.abs(bulk_pred) ** enhance

            # ── Success banner ──────────────────────────────────────
            st.success(
                f"**Inference complete** — {elapsed_ms:.1f} ms on "
                f"`{device}` using `{ckpt_name}` "
                f"({n_params:,} params)"
            )

            # ── Display info ────────────────────────────────────────
            c_info1, c_info2, c_info3 = st.columns(3)
            with c_info1:
                st.metric("Inference Time", f"{elapsed_ms:.1f} ms")
            with c_info2:
                st.metric("Output Shape", f"{bulk_pred.shape}")
            with c_info3:
                st.metric("Φ Range",
                          f"[{bulk_pred.min():.3f}, {bulk_pred.max():.3f}]")

            # ── 3D surface of the selected time step ────────────────
            st.markdown(
                '<div class="sec-title">Bulk Field Φ(x, z) at t = 10</div>',
                unsafe_allow_html=True,
            )

            # User explicitly requested t=10
            plot_data = bulk_pred[10, :, :]  

            # Plotting with plotly.graph_objects.Surface
            NX, NZ = plot_data.shape
            x_vals = np.linspace(0, 1, NX)
            z_vals = np.linspace(0, 1, NZ)

            fig = go.Figure(data=[go.Surface(
                z=plot_data.T,
                x=x_vals,
                y=z_vals,
                colorscale="Viridis",
                cmin=-1.5,
                cmax=1.5,
                showscale=True,
                colorbar=dict(title=dict(text="Φ", font=dict(color="#ccc")), tickfont=dict(color="#aaa"))
            )])

            title_str = f"FNO3d Prediction  |  t = 10  |  A₁={a1:.1f} k₁={k1} ω₁={w1:.1f}  ·  A₂={a2:.1f} k₂={k2} ω₂={w2:.1f}"
            
            fig.update_layout(
                title=dict(text=title_str, font=dict(color="#e4e4f0", size=16), x=0.5),
                scene=dict(
                    xaxis=dict(title="X (Boundary)", range=[0, 1],
                               backgroundcolor="rgba(10,10,25,0.9)", gridcolor="rgba(60,60,100,0.3)"),
                    yaxis=dict(title="Z (Depth)", range=[0, 1],
                               backgroundcolor="rgba(10,10,25,0.9)", gridcolor="rgba(60,60,100,0.3)"),
                    zaxis=dict(title="Φ (Amplitude)", range=[-1.5, 1.5],
                               backgroundcolor="rgba(10,10,25,0.9)", gridcolor="rgba(60,60,100,0.3)"),
                    camera=dict(eye=dict(x=2.1, y=-2.1, z=1.4)),
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=0.65),
                ),
                margin=dict(l=0, r=0, t=50, b=10),
                height=700,
                paper_bgcolor="#0a0a1a",
                plot_bgcolor="#0a0a1a",
                font=dict(color="#ccc")
            )

            st.plotly_chart(fig, use_container_width=True)

            # ── Parameter summary table ─────────────────────────────
            st.markdown(f"""
            | Parameter | Source 1 | Source 2 |
            |-----------|----------|----------|
            | Amplitude | `{a1:.2f}` | `{a2:.2f}` |
            | Frequency | `{k1}` | `{k2}` |
            | Phase     | `{w1:.2f}` | `{w2:.2f}` |
            """)


# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE  &  FUTURE WORK  (always visible below tabs)
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<hr class="divider">', unsafe_allow_html=True)

arch_c, future_c = st.columns(2)

with arch_c:
    st.markdown("""
    <div class="card">
        <h4>🏗️  FNO3d Architecture</h4>
        <table style="width:100%; color:#ccc; font-size:0.9rem; line-height:2.1;">
            <tr><td style="color:#888;">Model</td>
                <td><code>FNO3d</code></td></tr>
            <tr><td style="color:#888;">Fourier Modes</td>
                <td><code>modes = 8</code> per dimension</td></tr>
            <tr><td style="color:#888;">Channel Width</td>
                <td><code>width = 20</code></td></tr>
            <tr><td style="color:#888;">Spectral Layers</td>
                <td><code>4</code></td></tr>
            <tr><td style="color:#888;">Parameters</td>
                <td><code>3,281,377</code></td></tr>
            <tr><td style="color:#888;">Input</td>
                <td><code>(B, 3, 20, 64, 64)</code></td></tr>
            <tr><td style="color:#888;">Output</td>
                <td><code>(B, 1, 20, 64, 64)</code></td></tr>
            <tr><td style="color:#888;">Training</td>
                <td><code>553 epochs</code> &middot; AdamW &middot; OneCycleLR</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with future_c:
    st.markdown("""
    <div class="card">
        <h4>🔭  Future Work</h4>
        <ul style="color:#ccc; font-size:0.9rem; line-height:2.1; padding-left:1.2rem;">
            <li><strong style="color:#00d4ff;">Hard Boundary Constraints (Ansatz)</strong> —
                Enforce exact Dirichlet conditions via architectural ansatz
                to push Rel L₂ <strong>&lt; 5%</strong>.</li>
            <li><strong style="color:#8855ff;">Multi-Resolution Training</strong> —
                Joint training at 64 / 128 / 256 for improved zero-shot
                generalisation.</li>
            <li><strong style="color:#00ff88;">Autoregressive Rollout</strong> —
                Extend prediction beyond the 20-step training window.</li>
            <li><strong style="color:#ffd700;">GPU Benchmarks</strong> —
                Rigorous A100 / H100 timing vs LU factorisation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<hr class="divider">
<div class="footer">
    Neural-AdS &middot; Department of Computer Science &middot;
    Manipal University Jaipur<br>
    PyTorch &middot; Plotly &middot; Streamlit &middot;
    Fourier Neural Operators
</div>
""", unsafe_allow_html=True)
