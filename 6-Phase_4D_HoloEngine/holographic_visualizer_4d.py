"""
holographic_visualizer_4d.py
============================
4D Holographic Spacetime Visualizer for Binary Black Hole Merger (AdS/CFT)

High-fidelity Python visualizer using Plotly and Matplotlib to render the
output of a trained 4D Physics-Informed Neural Network.  The PINN simulates
a binary scalar-field merger in the AdS bulk, dual to a Quark-Gluon Plasma
collision on the CFT boundary.

Components
----------
1.  Primary View — 3D Volumetric Isosurface (Plotly)
    Multi-layered opacity (0.2 → 0.8) showing BH cores + gravitational radiation.
2.  Secondary View — Holographic Projection (Matplotlib)
    Boundary fluid density vs HRT area growth over time.
3.  QNM Chirp Plot — Peak bulk |φ| tracking inspiral chirp → ring-down.
4.  Sommerfeld Boundary Glow — Edge-coloured radiation absorption.
5.  Causality Masking — Red-shift overlay near local light cones.
6.  Full Dashboard — 8-panel combined view at a single time slice.
7.  Time Animation — GIF of bulk + boundary evolution across 50 frames.
8.  Ground Truth Comparison — Side-by-side with apex_master_dataset.npz.

Usage on Kaggle
---------------
    %%writefile /kaggle/working/holographic_visualizer_4d.py
    <paste this file>

    # Then in the next cell:
    from holographic_visualizer_4d import HolographicVisualizer
    viz = HolographicVisualizer('/kaggle/working/checkpoints/best_model.pt')
    viz.render_all(t_idx=25, output_dir='/kaggle/working/viz_output')

Architecture Auto-Detection
---------------------------
LATENT_DIM, SIREN_HIDDEN, SIREN_LAYERS are inferred from the checkpoint
state dict, so the visualizer works regardless of training configuration
(LATENT_DIM=10, 128, etc.).
"""

# ====================================================================== #
#  IMPORTS                                                                 #
# ====================================================================== #
import os
import math
import warnings
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")                     # headless-safe; call plt.show() after
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3D projection)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[warn] Plotly not installed — 3D isosurface will fall back to Matplotlib."
          "  Install: pip install plotly kaleido")


# ====================================================================== #
#  1.  CONFIGURATION (mirrors training; auto-overridden at load time)      #
# ====================================================================== #
class VizConfig:
    Z_MIN           = 1e-4
    U_MIN           = math.log(1e-4)        # ≈ −9.2103  (UV boundary)
    U_MAX           = 0.0                   # IR deep bulk
    U_BOUNDARY      = math.log(1e-4)
    X_RANGE         = (-1.0, 1.0)
    Y_RANGE         = (-1.0, 1.0)
    T_RANGE         = (0.0, 1.0)

    # Architecture (overwritten by auto-detect)
    LATENT_DIM      = 10
    SIREN_HIDDEN    = 256
    SIREN_LAYERS    = 5
    SIREN_OMEGA_0   = 30.0
    ENCODER_CHANNELS        = [1, 16, 32, 64]
    ENCODER_TEMPORAL_FRAMES = 16
    ENCODER_SPATIAL_RES     = 32

    # Physics
    DELTA           = 3.0
    KAPPA           = 0.1
    KAPPA_MAX       = 0.1

    # Inference mesh
    NX  = 100       # spatial x grid points
    NY  = 100       # spatial y grid points
    NU  = 50        # bulk u grid points
    NT  = 50        # temporal frames for animation

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ====================================================================== #
#  2.  SELF-CONTAINED MODEL DEFINITIONS                                    #
# ====================================================================== #
class SineLayer(nn.Module):
    def __init__(self, din, dout, w0=30.0, first=False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(din, dout)
        with torch.no_grad():
            b = 1.0 / din if first else math.sqrt(6.0 / din) / w0
            self.linear.weight.uniform_(-b, b)

    def forward(self, x, gamma=None, beta=None):
        h = self.w0 * self.linear(x)
        if gamma is not None:
            h = gamma * h + beta
        return torch.sin(h)


class FiLMSiren(nn.Module):
    def __init__(self, latent_dim=10, hidden=256, n_layers=5, omega_0=30.0):
        super().__init__()
        self.first = SineLayer(4, hidden, omega_0, first=True)
        self.hidden = nn.ModuleList(
            [SineLayer(hidden, hidden, omega_0) for _ in range(n_layers - 1)]
        )
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            b = math.sqrt(6.0 / hidden) / omega_0
            self.out.weight.uniform_(-b, b)
        self.film = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * hidden),
            )
            for _ in range(n_layers)
        ])

    def forward(self, coords, z):
        fp = []
        for g in self.film:
            o = g(z)
            ga, be = o.chunk(2, dim=-1)
            fp.append((ga + 1.0, be))
        h = self.first(coords, fp[0][0], fp[0][1])
        for i, ly in enumerate(self.hidden):
            h = ly(h, fp[i + 1][0], fp[i + 1][1])
        return self.out(h)


class ConvEncoder3D(nn.Module):
    def __init__(self, latent_dim=10, channels=None, T=16, H=32, W=32,
                 flat_size=None, fc_hidden=128):
        super().__init__()
        if channels is None:
            channels = [1, 16, 32, 64]
        blk = []
        for i in range(len(channels) - 1):
            blk += [
                nn.Conv3d(channels[i], channels[i + 1], 3, 2, 1),
                nn.BatchNorm3d(channels[i + 1]),
                nn.LeakyReLU(0.2, True),
            ]
        self.conv = nn.Sequential(*blk)
        if flat_size is None:
            t, h, w = T, H, W
            for _ in range(len(channels) - 1):
                t = (t + 1) // 2
                h = (h + 1) // 2
                w = (w + 1) // 2
            flat_size = channels[-1] * t * h * w
        self.fc = nn.Sequential(
            nn.Linear(flat_size, fc_hidden),
            nn.SiLU(),
            nn.Linear(fc_hidden, latent_dim),
        )

    def forward(self, x):
        h = self.conv(x)
        return self.fc(h.reshape(h.size(0), -1))


# ====================================================================== #
#  3.  SYNTHETIC COLLISION FLUID (matches training data generator)         #
# ====================================================================== #
def generate_collision_data(T=16, H=32, W=32):
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    data = torch.zeros(T, H, W)
    for ti in range(T):
        t = ti / max(T - 1, 1)
        sep = 0.6 * (1 - t)
        ang = 2.5 * np.pi * t
        cx1, cy1 = sep * np.cos(ang), sep * np.sin(ang)
        sig = 0.15 + 0.10 * (1 - t)
        amp = 1.0 + 2.0 * t
        b1 = amp * torch.exp(-((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2 * sig ** 2))
        b2 = amp * torch.exp(-((xx + cx1) ** 2 + (yy + cy1) ** 2) / (2 * sig ** 2))
        frame = b1 + b2
        if t > 0.7:
            rf = (t - 0.7) / 0.3
            r = torch.sqrt(xx ** 2 + yy ** 2 + 1e-8)
            frame += rf * 0.5 * torch.sin(12 * r - 8 * t) * torch.exp(-r ** 2 / 0.25)
        data[ti] = frame
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return data


# ====================================================================== #
#  4.  HOLOGRAPHIC VISUALIZER                                              #
# ====================================================================== #
class HolographicVisualizer:
    """
    High-fidelity 4D Holographic Spacetime Visualizer.

    Loads a trained FiLM-SIREN + ConvEncoder3D checkpoint and renders:
      - 3D volumetric isosurfaces (Plotly)
      - Holographic boundary↔bulk projection (Matplotlib)
      - QNM chirp / ring-down tracking
      - Sommerfeld boundary glow
      - Causality masking (red-shift)
      - Metric backreaction
      - HRT shadow on the bulk floor

    Physics formulas implemented for rendering:
      ke      = clamp(κ · |T_tt − T_xx|, max = K_max)
      HRT     = 1.0 + (√(1 + ke) − 1).mean()           [Differential Area Fix]
      V_max   = percentile(|φ|, 98)                      [Adaptive Normalisation]
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        dataset_path: str = None,
        sim_idx: int = 0,
        device: str = None,
        nx: int = 100,
        ny: int = 100,
        nu: int = 50,
        nt: int = 50,
    ):
        self.cfg = VizConfig()
        self.cfg.NX, self.cfg.NY, self.cfg.NU, self.cfg.NT = nx, ny, nu, nt
        self.device = device or self.cfg.DEVICE
        self.sim_idx = sim_idx

        # --- Load checkpoint & auto-detect architecture -----------------
        print(f"[viz] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self._autodetect_architecture(ckpt)

        # --- Build & load models ----------------------------------------
        self.encoder = ConvEncoder3D(
            latent_dim=self.cfg.LATENT_DIM,
            channels=self.cfg.ENCODER_CHANNELS,
            T=self.cfg.ENCODER_TEMPORAL_FRAMES,
            H=self.cfg.ENCODER_SPATIAL_RES,
            W=self.cfg.ENCODER_SPATIAL_RES,
            flat_size=getattr(self.cfg, '_enc_flat_size', None),
            fc_hidden=getattr(self.cfg, '_enc_fc_hidden', 128),
        ).to(self.device)

        self.siren = FiLMSiren(
            latent_dim=self.cfg.LATENT_DIM,
            hidden=self.cfg.SIREN_HIDDEN,
            n_layers=self.cfg.SIREN_LAYERS,
            omega_0=self.cfg.SIREN_OMEGA_0,
        ).to(self.device)

        self.encoder.load_state_dict(ckpt["encoder"])
        self.siren.load_state_dict(ckpt["siren"])
        self.encoder.eval()
        self.siren.eval()
        ep = ckpt.get("epoch", "?")
        bl = ckpt.get("best_loss", "?")
        print(f"[viz] Models loaded  (epoch {ep}, best_loss {bl})")

        # --- CNN input → latent vector ----------------------------------
        self._prepare_cnn_input(dataset_path)

        # --- Coordinate arrays ------------------------------------------
        self.x_lin = np.linspace(*self.cfg.X_RANGE, self.cfg.NX)
        self.y_lin = np.linspace(*self.cfg.Y_RANGE, self.cfg.NY)
        self.u_lin = np.linspace(self.cfg.U_MIN, self.cfg.U_MAX, self.cfg.NU)
        self.t_lin = np.linspace(*self.cfg.T_RANGE, self.cfg.NT)

        self._field_cache: dict = {}

        pts_per_frame = self.cfg.NX * self.cfg.NY * self.cfg.NU
        print(f"[viz] Mesh: {self.cfg.NX}×{self.cfg.NY}×{self.cfg.NU} "
              f"× {self.cfg.NT} frames  ({pts_per_frame:,} pts/frame)")

    # ------------------------------------------------------------------ #
    #  Architecture auto-detection from checkpoint state_dict              #
    # ------------------------------------------------------------------ #
    def _autodetect_architecture(self, ckpt):
        sd_s = ckpt.get("siren", {})
        sd_e = ckpt.get("encoder", {})

        # LATENT_DIM: FiLM generator first linear weight → shape (hidden, latent)
        for k in sorted(sd_s.keys()):
            if "film" in k and "0.weight" in k:
                self.cfg.LATENT_DIM = sd_s[k].shape[1]
                break

        # SIREN_HIDDEN: first layer linear weight → shape (hidden, 4)
        for k in sorted(sd_s.keys()):
            if "first" in k and "linear.weight" in k:
                self.cfg.SIREN_HIDDEN = sd_s[k].shape[0]
                break

        # SIREN_LAYERS: count hidden.{i}.linear.weight keys + 1 (first)
        hidden_count = sum(
            1 for k in sd_s if k.startswith("hidden.") and k.endswith(".linear.weight")
        )
        self.cfg.SIREN_LAYERS = hidden_count + 1

        # Encoder FC auto-detect from state dict:
        #   fc.0.weight → (fc_hidden, flat_size)
        #   fc.2.weight → (latent_dim, fc_hidden)
        self.cfg._enc_flat_size = None
        self.cfg._enc_fc_hidden = 128      # default
        if "fc.0.weight" in sd_e:
            self.cfg._enc_flat_size = sd_e["fc.0.weight"].shape[1]
            self.cfg._enc_fc_hidden = sd_e["fc.0.weight"].shape[0]
        if "fc.2.weight" in sd_e:
            self.cfg.LATENT_DIM = sd_e["fc.2.weight"].shape[0]

        # Reverse-engineer T, H, W from flat_size if possible
        if self.cfg._enc_flat_size is not None:
            ch_last = self.cfg.ENCODER_CHANNELS[-1]
            thw = self.cfg._enc_flat_size // ch_last
            # Try common (T, H, W) combos from the dataset generator
            found = False
            for T_try, HW_try in [(100, 64), (16, 32), (50, 48), (32, 32)]:
                t, h, w = T_try, HW_try, HW_try
                for _ in range(len(self.cfg.ENCODER_CHANNELS) - 1):
                    t = (t + 1) // 2
                    h = (h + 1) // 2
                    w = (w + 1) // 2
                if t * h * w == thw:
                    self.cfg.ENCODER_TEMPORAL_FRAMES = T_try
                    self.cfg.ENCODER_SPATIAL_RES = HW_try
                    found = True
                    break
            if not found:
                print(f"[viz] WARNING: Could not reverse-engineer T/H/W from "
                      f"flat_size={self.cfg._enc_flat_size}. Using flat_size override.")

        print(f"[viz] Auto-detected: LATENT_DIM={self.cfg.LATENT_DIM}, "
              f"SIREN_HIDDEN={self.cfg.SIREN_HIDDEN}, "
              f"SIREN_LAYERS={self.cfg.SIREN_LAYERS}")
        if self.cfg._enc_flat_size is not None:
            print(f"[viz] Encoder FC: flat_size={self.cfg._enc_flat_size}, "
                  f"fc_hidden={self.cfg._enc_fc_hidden}, "
                  f"T={self.cfg.ENCODER_TEMPORAL_FRAMES}, "
                  f"H=W={self.cfg.ENCODER_SPATIAL_RES}")

    # ------------------------------------------------------------------ #
    #  CNN encoder input                                                   #
    # ------------------------------------------------------------------ #
    def _prepare_cnn_input(self, dataset_path):
        if dataset_path and os.path.exists(dataset_path):
            print(f"[viz] Loading CNN volume from dataset: {dataset_path}")
            data = np.load(dataset_path)
            vol = data["cnn_volumes"][self.sim_idx]
            cnn_in = (
                torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(self.device)
            )
            self.cnn_data = vol  # (T, H, W)
        else:
            print("[viz] Generating synthetic collision fluid for encoder")
            bd = generate_collision_data(
                self.cfg.ENCODER_TEMPORAL_FRAMES,
                self.cfg.ENCODER_SPATIAL_RES,
                self.cfg.ENCODER_SPATIAL_RES,
            )
            cnn_in = bd.unsqueeze(0).unsqueeze(0).to(self.device)
            self.cnn_data = bd.numpy()  # (T, H, W)

        with torch.no_grad():
            self.z_latent = self.encoder(cnn_in)  # (1, LATENT_DIM)
        print(f"[viz] Latent z:  shape={tuple(self.z_latent.shape)}, "
              f"‖z‖={self.z_latent.norm().item():.4f}")

    # ================================================================== #
    #  FIELD INFERENCE                                                     #
    # ================================================================== #
    @torch.no_grad()
    def infer_field_at_time(self, t_val: float, batch_size: int = 65536):
        """
        Evaluate φ_renorm(t, x, y, u) on the full (NX, NY, NU) mesh.

        Returns
        -------
        field : ndarray of shape (NX, NY, NU)
        """
        key = round(t_val, 8)
        if key in self._field_cache:
            return self._field_cache[key]

        xx, yy, uu = np.meshgrid(
            self.x_lin, self.y_lin, self.u_lin, indexing="ij"
        )
        tt = np.full_like(xx, t_val)
        coords_np = np.stack(
            [tt.ravel(), xx.ravel(), yy.ravel(), uu.ravel()], axis=1
        ).astype(np.float32)

        N = coords_np.shape[0]
        phi_all = np.zeros(N, dtype=np.float32)
        z_lat = self.z_latent

        for i in range(0, N, batch_size):
            batch = torch.from_numpy(coords_np[i : i + batch_size]).to(self.device)
            z_exp = z_lat.expand(batch.shape[0], -1)
            phi = self.siren(batch, z_exp).squeeze(-1).cpu().numpy()
            phi_all[i : i + batch_size] = phi

        field = phi_all.reshape(self.cfg.NX, self.cfg.NY, self.cfg.NU)
        self._field_cache[key] = field
        return field

    def infer_all_frames(self):
        """Pre-compute φ for every time frame (cached)."""
        print("[viz] Inferring field over all time frames …")
        for i, t in enumerate(self.t_lin):
            self.infer_field_at_time(t)
            if (i + 1) % 10 == 0 or (i + 1) == len(self.t_lin):
                print(f"      frame {i + 1}/{self.cfg.NT}")
        print("[viz] Inference complete.")

    # ================================================================== #
    #  PHYSICS COMPUTATIONS  (finite-difference — render-time only)        #
    # ================================================================== #
    def _time_derivative(self, t_val: float, eps: float = 0.005):
        """Central-difference ∂φ/∂t."""
        t_fwd = min(t_val + eps, self.cfg.T_RANGE[1])
        t_bwd = max(t_val - eps, self.cfg.T_RANGE[0])
        dt = t_fwd - t_bwd
        if dt < 1e-12:
            return np.zeros((self.cfg.NX, self.cfg.NY, self.cfg.NU))
        return (
            self.infer_field_at_time(t_fwd) - self.infer_field_at_time(t_bwd)
        ) / dt

    def compute_backreaction(self, field, t_val):
        r"""
        Metric backreaction:
            ke = \text{clamp}(\kappa \cdot |T_{tt} - T_{xx}|, \max = K_{max})
        """
        dx = self.x_lin[1] - self.x_lin[0]
        dy = self.y_lin[1] - self.y_lin[0]

        dphi_dx = np.gradient(field, dx, axis=0)
        dphi_dy = np.gradient(field, dy, axis=1)
        dphi_dt = self._time_derivative(t_val)

        e2u = np.exp(2.0 * self.u_lin)[np.newaxis, np.newaxis, :]
        T_tt = 0.5 * e2u * dphi_dt ** 2
        T_xx = 0.5 * e2u * dphi_dx ** 2
        diff = np.abs(T_tt - T_xx)

        ke = np.clip(self.cfg.KAPPA * diff, 0.0, self.cfg.KAPPA_MAX)
        return ke

    def compute_hrt_shadow(self, field, t_val):
        """
        Differential Area Fix:
            hrt_density(x, y) = 1.0 + mean_u[ √(1 + ke) − 1 ]
        Projected as a 2D «shadow» on the bulk floor (u = 0).
        """
        ke = self.compute_backreaction(field, t_val)
        area_excess = np.sqrt(1.0 + ke) - 1.0          # (NX, NY, NU)
        shadow = 1.0 + area_excess.mean(axis=2)         # (NX, NY)
        return shadow

    def compute_global_hrt(self, ke):
        """Global scalar:  HRT = 1.0 + ⟨√(1 + ke) − 1⟩  [Differential Area Fix]"""
        return 1.0 + (np.sqrt(1.0 + ke) - 1.0).mean()

    def compute_causality_mask(self, field, t_val):
        r"""
        Causality masking:
            g^{\mu\nu} \partial_\mu \phi \partial_\nu \phi

        Returns
        -------
        redshift  : (NX, NY, NU)  — near-lightcone intensity ∈ [0, 1]
        grad_norm : (NX, NY, NU)  — signed light-cone norm
        """
        dx = self.x_lin[1] - self.x_lin[0]
        dy = self.y_lin[1] - self.y_lin[0]
        du = self.u_lin[1] - self.u_lin[0] if self.cfg.NU > 1 else 1.0

        dphi_dx = np.gradient(field, dx, axis=0)
        dphi_dy = np.gradient(field, dy, axis=1)
        dphi_du = np.gradient(field, du, axis=2)
        dphi_dt = self._time_derivative(t_val)

        e2u = np.exp(2.0 * self.u_lin)[np.newaxis, np.newaxis, :]

        # g^{μν} = diag(−e^{2u}, e^{2u}, e^{2u}, 1)
        grad_norm = (
            -e2u * dphi_dt ** 2
            + e2u * dphi_dx ** 2
            + e2u * dphi_dy ** 2
            + dphi_du ** 2
        )

        # Speed ratio  |spatial gradient| / (e^u |∂_t φ|)
        eu = np.exp(self.u_lin)[np.newaxis, np.newaxis, :]
        spatial = np.sqrt(e2u * (dphi_dx ** 2 + dphi_dy ** 2) + dphi_du ** 2 + 1e-12)
        temporal = eu * (np.abs(dphi_dt) + 1e-12)
        speed_ratio = spatial / temporal

        # Peaks at speed_ratio → 1  (approaching local light cone)
        redshift = np.exp(-2.0 * (speed_ratio - 1.0) ** 2)
        redshift = np.clip(redshift, 0.0, 1.0)

        return redshift, grad_norm

    def compute_sommerfeld_glow(self, field, t_val):
        """
        Sommerfeld Boundary Glow — wave intensity at spatial arena edges.

            x = +1 :  |∂_t φ + ∂_x φ|     x = −1 :  |∂_t φ − ∂_x φ|
            y = +1 :  |∂_t φ + ∂_y φ|     y = −1 :  |∂_t φ − ∂_y φ|
        """
        dx = self.x_lin[1] - self.x_lin[0]
        dy = self.y_lin[1] - self.y_lin[0]

        dphi_dx = np.gradient(field, dx, axis=0)
        dphi_dy = np.gradient(field, dy, axis=1)
        dphi_dt = self._time_derivative(t_val)

        glow = {
            "x+": np.abs(dphi_dt[-1, :, :] + dphi_dx[-1, :, :]),   # (NY, NU)
            "x-": np.abs(dphi_dt[0, :, :]  - dphi_dx[0, :, :]),    # (NY, NU)
            "y+": np.abs(dphi_dt[:, -1, :]  + dphi_dy[:, -1, :]),   # (NX, NU)
            "y-": np.abs(dphi_dt[:, 0, :]   - dphi_dy[:, 0, :]),    # (NX, NU)
        }
        return glow

    def adaptive_normalize(self, field, percentile=98):
        """98th-percentile dynamic V_max for adaptive contrast."""
        v_max = np.percentile(np.abs(field), percentile)
        if v_max < 1e-8:
            v_max = np.abs(field).max() + 1e-8
        return np.clip(field / v_max, -1.0, 1.0), v_max

    def compute_qnm_series(self):
        """Quasi-Normal Mode series: peak / mean / RMS of |φ| at each time frame."""
        print("[viz] Computing QNM time series …")
        peak, mean, rms = [], [], []
        for t in self.t_lin:
            f = self.infer_field_at_time(t)
            peak.append(np.max(np.abs(f)))
            mean.append(np.mean(np.abs(f)))
            rms.append(np.sqrt(np.mean(f ** 2)))
        return dict(
            t=self.t_lin,
            peak=np.array(peak),
            mean=np.array(mean),
            rms=np.array(rms),
        )

    # ================================================================== #
    #  PRIMARY VIEW — 3D ISOSURFACE  (Plotly)                              #
    # ================================================================== #
    def render_3d_isosurface(self, t_idx=25, save_path="isosurface_3d.html"):
        """
        3D volumetric render of φ(x, y, u) at a single time slice.

        Multi-layered isosurface with opacities 0.15 → 0.70.
        Overlays:  HRT shadow on the bulk floor,  Sommerfeld glow on edges.
        """
        if not HAS_PLOTLY:
            print("[warn] Plotly unavailable — falling back to Matplotlib scatter.")
            return self._render_3d_matplotlib(t_idx, save_path)

        t_val = self.t_lin[t_idx]
        field = self.infer_field_at_time(t_val)
        field_norm, v_max = self.adaptive_normalize(field)
        abs_f = np.abs(field_norm)

        xx, yy, uu = np.meshgrid(self.x_lin, self.y_lin, self.u_lin, indexing="ij")

        hrt_shadow = self.compute_hrt_shadow(field, t_val)
        glow = self.compute_sommerfeld_glow(field, t_val)

        fig = go.Figure()

        # ------ Multi-layered isosurface volume ------
        iso_levels = np.linspace(0.15, 0.90, 5) * abs_f.max()
        opacities = [0.12, 0.20, 0.35, 0.55, 0.75]

        for level, opa in zip(iso_levels[::-1], opacities):
            fig.add_trace(go.Isosurface(
                x=xx.ravel(), y=yy.ravel(), z=uu.ravel(),
                value=abs_f.ravel(),
                isomin=float(level),
                isomax=float(abs_f.max()),
                opacity=opa,
                surface_count=1,
                colorscale="Inferno",
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=f"iso >= {level:.2f}",
            ))

        # ------ HRT Shadow on the bulk floor (u = 0) ------
        xx_2d, yy_2d = np.meshgrid(self.x_lin, self.y_lin, indexing="ij")
        sn = (hrt_shadow - hrt_shadow.min()) / (hrt_shadow.max() - hrt_shadow.min() + 1e-8)
        fig.add_trace(go.Surface(
            x=xx_2d, y=yy_2d,
            z=np.full_like(xx_2d, self.cfg.U_MAX),
            surfacecolor=sn,
            colorscale="Viridis",
            opacity=0.55,
            showscale=True,
            colorbar=dict(title="HRT Area", x=1.02, len=0.35, y=0.25),
            name="HRT Shadow",
        ))

        # ------ Sommerfeld Boundary Glow on the 4 vertical faces ------
        som_cmap = [[0, "rgba(0,40,200,0)"], [0.4, "rgba(0,180,255,0.25)"],
                     [1, "rgba(255,255,50,0.70)"]]

        yy_f, uu_f = np.meshgrid(self.y_lin, self.u_lin, indexing="ij")
        for edge, xv in [("x+", 1.0), ("x-", -1.0)]:
            gi = glow[edge]
            gn = gi / (gi.max() + 1e-8)
            fig.add_trace(go.Surface(
                x=np.full_like(yy_f, xv), y=yy_f, z=uu_f,
                surfacecolor=gn, colorscale=som_cmap, opacity=0.30,
                showscale=False, name=f"Sommerfeld {edge}",
            ))

        xx_f, uu_f2 = np.meshgrid(self.x_lin, self.u_lin, indexing="ij")
        for edge, yv in [("y+", 1.0), ("y-", -1.0)]:
            gi = glow[edge]
            gn = gi / (gi.max() + 1e-8)
            fig.add_trace(go.Surface(
                x=xx_f, y=np.full_like(xx_f, yv), z=uu_f2,
                surfacecolor=gn, colorscale=som_cmap, opacity=0.30,
                showscale=False, name=f"Sommerfeld {edge}",
            ))

        # ------ Layout ------
        fig.update_layout(
            title=dict(
                text=(f"4D Holographic Bulk: φ(x, y, u)  at  t = {t_val:.3f}<br>"
                      f"<sub>Binary BH Merger in AdS₄  |  V_max = {v_max:.4f} (p98)"
                      f"  |  LATENT_DIM = {self.cfg.LATENT_DIM}</sub>"),
                font=dict(size=15),
            ),
            scene=dict(
                xaxis_title="x  (transverse)",
                yaxis_title="y  (transverse)",
                zaxis_title="u = ln(z)  (bulk depth)",
                zaxis=dict(range=[self.cfg.U_MIN, self.cfg.U_MAX]),
                aspectratio=dict(x=1, y=1, z=0.7),
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
            ),
            width=1050, height=820,
            paper_bgcolor="black",
            font_color="white",
        )

        if save_path:
            html_str = fig.to_html(include_plotlyjs="cdn")
            with open(save_path, "w", encoding="utf-8") as fh:
                fh.write(html_str)
            print(f"[viz] 3D isosurface saved -> {save_path}")
        return fig

    def _render_3d_matplotlib(self, t_idx, save_path):
        """Fallback scatter plot when Plotly is unavailable."""
        t_val = self.t_lin[t_idx]
        field = self.infer_field_at_time(t_val)
        fn, v_max = self.adaptive_normalize(field)

        xx, yy, uu = np.meshgrid(self.x_lin, self.y_lin, self.u_lin, indexing="ij")
        mask = np.abs(fn) > 0.15  # only show noticeable signal

        fig = plt.figure(figsize=(12, 9), facecolor="black")
        ax = fig.add_subplot(111, projection="3d", facecolor="black")
        sc = ax.scatter(
            xx[mask], yy[mask], uu[mask],
            c=fn[mask], cmap="inferno", alpha=0.35, s=1,
        )
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        ax.set_zlabel("u = ln(z)", color="white")
        ax.set_title(f"φ(x, y, u)  t = {t_val:.3f}  V_max = {v_max:.4f}",
                      color="white", fontsize=12)
        ax.tick_params(colors="white")
        plt.colorbar(sc, ax=ax, shrink=0.5, label="φ_renorm (p98 normed)")
        out = save_path.replace(".html", ".png") if save_path.endswith(".html") else save_path
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[viz] 3D scatter saved → {out}")
        plt.show()

    # ================================================================== #
    #  SECONDARY VIEW — HOLOGRAPHIC PROJECTION  (Matplotlib)               #
    # ================================================================== #
    def render_holographic_projection(self, save_path="holographic_projection.png"):
        """
        2D heatmap:
            Row 1 — Boundary fluid density  φ_R(t, x, y = 0, u = UV)
            Row 2 — HRT area density  S_RT(t, x, y = 0)
            Row 3 — Global HRT area over time (Differential Area Fix)
        """
        n_disp = min(self.cfg.NT, 50)
        t_idx = np.linspace(0, self.cfg.NT - 1, n_disp, dtype=int)

        bnd_evo = np.zeros((self.cfg.NX, n_disp))
        hrt_evo = np.zeros((self.cfg.NX, n_disp))
        hrt_global = np.zeros(n_disp)

        print("[viz] Computing holographic projection evolution …")
        for j, ti in enumerate(t_idx):
            t_val = self.t_lin[ti]
            field = self.infer_field_at_time(t_val)

            # Boundary slice: u index 0 = U_MIN (UV boundary), y = 0
            bnd_evo[:, j] = field[:, self.cfg.NY // 2, 0]

            # HRT shadow
            shadow = self.compute_hrt_shadow(field, t_val)
            hrt_evo[:, j] = shadow[:, self.cfg.NY // 2]

            # Global HRT with Differential Area Fix
            ke = self.compute_backreaction(field, t_val)
            hrt_global[j] = self.compute_global_hrt(ke)

        fig, axes = plt.subplots(3, 1, figsize=(14, 13),
                                 gridspec_kw={"height_ratios": [1, 1, 0.45]},
                                 facecolor="black")
        text_c = "white"
        ext = [self.t_lin[0], self.t_lin[-1], self.x_lin[0], self.x_lin[-1]]

        im0 = axes[0].imshow(bnd_evo, aspect="auto", cmap="magma",
                             extent=ext, origin="lower")
        axes[0].set_ylabel("x  (boundary)", color=text_c)
        axes[0].set_title("CFT Boundary Fluid:  φ_ren(t, x, y=0, u=UV)",
                          color="gold", fontsize=13)
        axes[0].tick_params(colors=text_c)
        plt.colorbar(im0, ax=axes[0], label="φ_renorm")

        im1 = axes[1].imshow(hrt_evo, aspect="auto", cmap="viridis",
                             extent=ext, origin="lower")
        axes[1].set_ylabel("x  (boundary)", color=text_c)
        axes[1].set_title("HRT Area Density:  S_RT(t, x, y=0)",
                          color="gold", fontsize=13)
        axes[1].tick_params(colors=text_c)
        plt.colorbar(im1, ax=axes[1], label="HRT Area")

        axes[2].plot(self.t_lin[t_idx], hrt_global, "c-", lw=2.5,
                     label="Global HRT Area")
        axes[2].axhline(1.0, color="gray", ls="--", alpha=0.5,
                        label="Vacuum baseline")
        axes[2].axvline(0.5, color="gold", ls=":", alpha=0.6,
                        label="Coalescence (t ≈ 0.5)")
        axes[2].set_xlabel("t  (merger time)", color=text_c)
        axes[2].set_ylabel("S_RT", color=text_c)
        axes[2].set_title("Global HRT:  1 + ⟨√(1+κe) − 1⟩  [Differential Area Fix]",
                          color="gold", fontsize=12)
        axes[2].legend(facecolor="black", edgecolor="gray", labelcolor="white")
        axes[2].set_facecolor("black")
        axes[2].tick_params(colors=text_c)
        axes[2].grid(True, alpha=0.2, color="gray")

        for ax in axes:
            ax.set_facecolor("black")

        fig.suptitle("Holographic Projection:  Boundary ↔ Bulk Duality",
                     color="gold", fontsize=15, y=0.99)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[viz] Holographic projection saved → {save_path}")
        plt.show()

    # ================================================================== #
    #  QNM CHIRP PLOT                                                      #
    # ================================================================== #
    def render_qnm_plot(self, save_path="qnm_chirp.png"):
        """
        Quasi-Normal Mode line graph:
            Top    — peak |φ| in bulk  (chirp → ring-down envelope)
            Bottom — instantaneous frequency via Hilbert transform
        """
        qnm = self.compute_qnm_series()

        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                 facecolor="black")
        text_c = "white"

        # --- amplitude envelope ---
        axes[0].plot(qnm["t"], qnm["peak"], "r-", lw=2.2, label="Peak |φ|")
        axes[0].plot(qnm["t"], qnm["rms"], "b--", lw=1.5, alpha=0.7,
                     label="RMS φ")
        axes[0].fill_between(qnm["t"], 0, qnm["peak"], alpha=0.12, color="red")
        axes[0].axvline(0.5, color="gold", ls=":", lw=1.2,
                        label="Coalescence (t ≈ 0.5)")
        axes[0].set_ylabel("Field Amplitude", color=text_c)
        axes[0].set_title("Quasi-Normal Mode:  Bulk Scalar Chirp & Ring-Down",
                          color="gold", fontsize=14)
        axes[0].legend(facecolor="black", edgecolor="gray", labelcolor="white")
        axes[0].set_facecolor("black")
        axes[0].tick_params(colors=text_c)
        axes[0].grid(True, alpha=0.2, color="gray")

        # --- instantaneous frequency ---
        try:
            from scipy.signal import hilbert
            analytic = hilbert(qnm["peak"])
            inst_phase = np.unwrap(np.angle(analytic))
            dt = self.t_lin[1] - self.t_lin[0] if len(self.t_lin) > 1 else 1.0
            inst_freq = np.gradient(inst_phase, dt) / (2 * np.pi)
            axes[1].plot(qnm["t"], inst_freq, "g-", lw=2, label="Instant. Frequency")
            axes[1].set_ylabel("Frequency  (cycles / t)", color=text_c)
            axes[1].set_title("QNM Frequency:  Chirp → Ringdown",
                              color="gold", fontsize=13)
        except ImportError:
            dt = self.t_lin[1] - self.t_lin[0] if len(self.t_lin) > 1 else 1.0
            dpeak = np.gradient(qnm["peak"], dt)
            axes[1].plot(qnm["t"], dpeak, "g-", lw=2, label="d|φ_peak|/dt")
            axes[1].set_ylabel("Rate of change", color=text_c)
            axes[1].set_title("Peak Amplitude Rate of Change",
                              color="gold", fontsize=13)

        axes[1].axvline(0.5, color="gold", ls=":", lw=1.2)
        axes[1].legend(facecolor="black", edgecolor="gray", labelcolor="white")
        axes[1].set_facecolor("black")
        axes[1].tick_params(colors=text_c)
        axes[1].grid(True, alpha=0.2, color="gray")
        axes[1].set_xlabel("t  (merger time)", color=text_c)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[viz] QNM chirp plot saved → {save_path}")
        plt.show()

    # ================================================================== #
    #  SOMMERFELD BOUNDARY GLOW                                            #
    # ================================================================== #
    def render_sommerfeld_glow(self, t_idx=25, save_path="sommerfeld_glow.png"):
        """4-panel heatmap of Sommerfeld radiation flux at each arena edge."""
        t_val = self.t_lin[t_idx]
        field = self.infer_field_at_time(t_val)
        glow = self.compute_sommerfeld_glow(field, t_val)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="black")
        text_c = "white"

        panels = [
            ("x+", "x = +1  (Right)",   "y", self.y_lin),
            ("x-", "x = −1  (Left)",    "y", self.y_lin),
            ("y+", "y = +1  (Top)",      "x", self.x_lin),
            ("y-", "y = −1  (Bottom)",   "x", self.x_lin),
        ]
        for ax, (edge, title, ylabel, yvals) in zip(axes.flat, panels):
            data = glow[edge]
            ext = [self.u_lin[0], self.u_lin[-1], yvals[0], yvals[-1]]
            im = ax.imshow(data, aspect="auto", cmap="hot", origin="lower",
                           extent=ext)
            ax.set_xlabel("u  (bulk depth)", color=text_c)
            ax.set_ylabel(ylabel, color=text_c)
            ax.set_title(f"{title}\n|∂_t φ ± ∂_n φ|", color="gold", fontsize=11)
            ax.tick_params(colors=text_c)
            ax.set_facecolor("black")
            plt.colorbar(im, ax=ax, label="Sommerfeld Flux")

        fig.suptitle(f"Sommerfeld Boundary Glow   t = {t_val:.3f}\n"
                     f"Radiation Absorption at Arena Edges",
                     color="gold", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[viz] Sommerfeld glow saved → {save_path}")
        plt.show()

    # ================================================================== #
    #  CAUSALITY MAP                                                       #
    # ================================================================== #
    def render_causality_map(self, t_idx=25, save_path="causality_map.png"):
        r"""
        Three-panel causality diagnostic at a mid-bulk slice:
            1. φ_renorm
            2. Near-lightcone red-shift intensity
            3. Sign of g^{\mu\nu}\partial_\mu\phi \partial_\nu\phi
        """
        t_val = self.t_lin[t_idx]
        field = self.infer_field_at_time(t_val)
        fn, _ = self.adaptive_normalize(field)
        redshift, grad_norm = self.compute_causality_mask(field, t_val)

        mid_u = self.cfg.NU // 2
        ext = [*self.cfg.X_RANGE, *self.cfg.Y_RANGE]

        fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor="black")
        text_c = "white"

        im0 = axes[0].imshow(fn[:, :, mid_u].T, origin="lower", cmap="RdBu_r",
                             extent=ext, vmin=-1, vmax=1)
        axes[0].set_title(f"φ_renorm  (u = {self.u_lin[mid_u]:.1f})",
                          color="gold", fontsize=12)
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(redshift[:, :, mid_u].T, origin="lower", cmap="hot",
                             extent=ext, vmin=0, vmax=1)
        axes[1].set_title("Causality Red-Shift Intensity",
                          color="gold", fontsize=12)
        plt.colorbar(im1, ax=axes[1], label="Near light-cone")

        ln = grad_norm[:, :, mid_u]
        im2 = axes[2].imshow(np.sign(ln).T, origin="lower", cmap="bwr",
                             extent=ext, vmin=-1, vmax=1)
        axes[2].set_title("g^{μν}∂_μφ ∂_νφ  Sign\n(Red = Timelike VIOLATION)",
                          color="gold", fontsize=11)
        plt.colorbar(im2, ax=axes[2], label="−1: spacelike  +1: timelike")

        for ax in axes:
            ax.set_xlabel("x", color=text_c)
            ax.set_ylabel("y", color=text_c)
            ax.tick_params(colors=text_c)
            ax.set_facecolor("black")

        fig.suptitle(f"Causality Masking   t = {t_val:.3f},  u = {self.u_lin[mid_u]:.2f}",
                     color="gold", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[viz] Causality map saved → {save_path}")
        plt.show()

    # ================================================================== #
    #  RADIAL PROFILE                                                      #
    # ================================================================== #
    def render_radial_profile(self, t_idx=25, save_path="radial_profile.png"):
        """φ_renorm and φ_bulk = φ_R · e^{3u}  vs u at spatial centre (x=0, y=0)."""
        t_val = self.t_lin[t_idx]
        field = self.infer_field_at_time(t_val)

        cx, cy = self.cfg.NX // 2, self.cfg.NY // 2
        phi_R = field[cx, cy, :]
        phi_bulk = phi_R * np.exp(3.0 * self.u_lin)

        fig, ax1 = plt.subplots(figsize=(11, 6), facecolor="black")
        ax2 = ax1.twinx()

        ax1.plot(self.u_lin, phi_R, "c-", lw=2.5, label="φ_renorm")
        ax2.plot(self.u_lin, phi_bulk, "r-", lw=2.5, label="φ_bulk = φ_R · e^{3u}")

        ax1.axvline(self.cfg.U_BOUNDARY, color="purple", ls=":", alpha=0.6,
                    label=f"UV boundary  u = {self.cfg.U_BOUNDARY:.1f}")
        ax1.axvline(0, color="lime", ls=":", alpha=0.6, label="IR deep bulk  u = 0")

        ax1.set_xlabel("u = ln(z)", color="white")
        ax1.set_ylabel("φ_renorm", color="cyan")
        ax2.set_ylabel("φ_bulk  (true field)", color="red")
        ax1.set_title(f"Holographic Radial Profile   t = {t_val:.3f},  (x, y) = (0, 0)",
                      color="gold", fontsize=13)
        ax1.set_facecolor("black")
        ax1.tick_params(colors="white")
        ax2.tick_params(colors="red")
        ax1.grid(True, alpha=0.15, color="gray")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   facecolor="black", edgecolor="gray", labelcolor="white",
                   loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[viz] Radial profile saved → {save_path}")
        plt.show()

    # ================================================================== #
    #  FULL 8-PANEL DASHBOARD                                              #
    # ================================================================== #
    def render_full_dashboard(self, t_idx=25, save_path="holographic_dashboard.png"):
        """Combined 8-panel dashboard at a single time slice."""
        t_val = self.t_lin[t_idx]
        field = self.infer_field_at_time(t_val)
        fn, v_max = self.adaptive_normalize(field)

        ke = self.compute_backreaction(field, t_val)
        hrt_shadow = self.compute_hrt_shadow(field, t_val)
        redshift, _ = self.compute_causality_mask(field, t_val)
        glow = self.compute_sommerfeld_glow(field, t_val)
        hrt_val = self.compute_global_hrt(ke)

        mid_u = self.cfg.NU // 2
        ext = [*self.cfg.X_RANGE, *self.cfg.Y_RANGE]
        tc = "white"

        fig = plt.figure(figsize=(22, 16), facecolor="black")
        gs = gridspec.GridSpec(3, 3, hspace=0.38, wspace=0.32)

        # ---- Row 0 : field slices ----
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(fn[:, :, 0].T, origin="lower", cmap="inferno",
                         extent=ext)
        ax1.set_title("UV Boundary  φ_R", color=tc, fontsize=11)
        ax1.set_xlabel("x", color=tc); ax1.set_ylabel("y", color=tc)
        ax1.tick_params(colors=tc); ax1.set_facecolor("black")
        plt.colorbar(im1, ax=ax1, shrink=0.82)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(fn[:, :, mid_u].T, origin="lower", cmap="RdBu_r",
                         extent=ext, vmin=-1, vmax=1)
        ax2.set_title(f"Mid-Bulk  φ_R  (u = {self.u_lin[mid_u]:.1f})",
                      color=tc, fontsize=11)
        ax2.set_xlabel("x", color=tc); ax2.set_ylabel("y", color=tc)
        ax2.tick_params(colors=tc); ax2.set_facecolor("black")
        plt.colorbar(im2, ax=ax2, shrink=0.82)

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(fn[:, :, -1].T, origin="lower", cmap="plasma",
                         extent=ext)
        ax3.set_title("IR Deep Bulk  φ_R  (u = 0)", color=tc, fontsize=11)
        ax3.set_xlabel("x", color=tc); ax3.set_ylabel("y", color=tc)
        ax3.tick_params(colors=tc); ax3.set_facecolor("black")
        plt.colorbar(im3, ax=ax3, shrink=0.82)

        # ---- Row 1 : physics diagnostics ----
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(hrt_shadow.T, origin="lower", cmap="viridis",
                         extent=ext)
        ax4.set_title("HRT Shadow  (S_RT)", color=tc, fontsize=11)
        ax4.set_xlabel("x", color=tc); ax4.set_ylabel("y", color=tc)
        ax4.tick_params(colors=tc); ax4.set_facecolor("black")
        plt.colorbar(im4, ax=ax4, shrink=0.82, label="Area")

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(redshift[:, :, mid_u].T, origin="lower", cmap="hot",
                         extent=ext, vmin=0, vmax=1)
        ax5.set_title("Causality Red-Shift", color=tc, fontsize=11)
        ax5.set_xlabel("x", color=tc); ax5.set_ylabel("y", color=tc)
        ax5.tick_params(colors=tc); ax5.set_facecolor("black")
        plt.colorbar(im5, ax=ax5, shrink=0.82, label="Near light-cone")

        ax6 = fig.add_subplot(gs[1, 2])
        ke_s = ke[:, :, mid_u]
        im6 = ax6.imshow(ke_s.T, origin="lower", cmap="magma", extent=ext)
        ax6.set_title("Metric Backreaction  κe", color=tc, fontsize=11)
        ax6.set_xlabel("x", color=tc); ax6.set_ylabel("y", color=tc)
        ax6.tick_params(colors=tc); ax6.set_facecolor("black")
        plt.colorbar(im6, ax=ax6, shrink=0.82)

        # ---- Row 2 left : Sommerfeld panoramic strip ----
        ax7 = fig.add_subplot(gs[2, 0:2])
        glow_strip = np.concatenate([
            glow["x+"].mean(axis=-1),
            glow["y+"].mean(axis=-1),
            glow["x-"].mean(axis=-1)[::-1],
            glow["y-"].mean(axis=-1)[::-1],
        ])
        ax7.fill_between(range(len(glow_strip)), glow_strip, alpha=0.55,
                         color="cyan")
        ax7.plot(glow_strip, "w-", lw=1)
        n = len(self.y_lin)
        for pos, lab in [(0, "x+"), (n, "y+"), (2 * n, "x−"), (3 * n, "y−")]:
            ax7.axvline(pos, color="yellow", ls=":", alpha=0.5)
            ax7.text(pos + 2, glow_strip.max() * 0.92, lab,
                     color="yellow", fontsize=9)
        ax7.set_xlabel("Boundary Perimeter  (unrolled)", color=tc)
        ax7.set_ylabel("Sommerfeld Flux", color=tc)
        ax7.set_title("Sommerfeld Boundary Glow:  Radiation Absorption",
                      color=tc, fontsize=11)
        ax7.set_facecolor("black")
        ax7.tick_params(colors=tc)
        ax7.grid(True, alpha=0.15, color="gray")

        # ---- Row 2 right : radial profile ----
        ax8 = fig.add_subplot(gs[2, 2])
        cx, cy = self.cfg.NX // 2, self.cfg.NY // 2
        phi_R_c = field[cx, cy, :]
        phi_B_c = phi_R_c * np.exp(3.0 * self.u_lin)
        ax8.plot(self.u_lin, phi_R_c, "c-", lw=2, label="φ_R")
        ax8t = ax8.twinx()
        ax8t.plot(self.u_lin, phi_B_c, "r-", lw=2, label="φ_bulk")
        ax8.set_xlabel("u = ln(z)", color=tc)
        ax8.set_ylabel("φ_renorm", color="cyan")
        ax8t.set_ylabel("φ_bulk", color="red")
        ax8.set_title("Radial Profile  (x=0, y=0)", color=tc, fontsize=11)
        ax8.set_facecolor("black")
        ax8.tick_params(colors=tc)
        ax8t.tick_params(colors="red")
        ax8.grid(True, alpha=0.15, color="gray")

        fig.suptitle(
            f"╔══════════════  4D HOLOGRAPHIC SPACETIME DASHBOARD  ══════════════╗\n"
            f"t = {t_val:.3f}   |   V_max = {v_max:.4f} (p98)   |   "
            f"HRT = {hrt_val:.4f}   |   LATENT_DIM = {self.cfg.LATENT_DIM}",
            color="gold", fontsize=16, y=0.99,
        )

        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[viz] Dashboard saved → {save_path}")
        plt.show()

    # ================================================================== #
    #  TIME EVOLUTION ANIMATION                                            #
    # ================================================================== #
    def render_time_animation(self, save_path="holographic_evolution.gif", fps=8):
        """
        Animate UV boundary + mid-bulk + IR bulk slices across all time frames.
        Saves as GIF (pillow writer).
        """
        self.infer_all_frames()

        mid_u = self.cfg.NU // 2
        all_bnd, all_mid, all_ir = [], [], []

        for t in self.t_lin:
            f = self._field_cache[round(t, 8)]
            all_bnd.append(f[:, :, 0])
            all_mid.append(f[:, :, mid_u])
            all_ir.append(f[:, :, -1])

        # Global dynamic range via 98th percentile across all frames
        g_max = max(
            np.percentile(np.abs(np.array(all_bnd)), 98),
            np.percentile(np.abs(np.array(all_mid)), 98),
            np.percentile(np.abs(np.array(all_ir)), 98),
            1e-8,
        )

        ext = [*self.cfg.X_RANGE, *self.cfg.Y_RANGE]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="black")

        im0 = axes[0].imshow(all_bnd[0].T, origin="lower", cmap="inferno",
                             extent=ext, vmin=-g_max, vmax=g_max, animated=True)
        im1 = axes[1].imshow(all_mid[0].T, origin="lower", cmap="RdBu_r",
                             extent=ext, vmin=-g_max, vmax=g_max, animated=True)
        im2 = axes[2].imshow(all_ir[0].T, origin="lower", cmap="plasma",
                             extent=ext, vmin=-g_max, vmax=g_max, animated=True)

        titles = ["UV Boundary", f"Mid-Bulk (u={self.u_lin[mid_u]:.1f})",
                  "IR Deep (u=0)"]
        for ax, title in zip(axes, titles):
            ax.set_title(title, color="white", fontsize=11)
            ax.set_xlabel("x", color="white")
            ax.set_ylabel("y", color="white")
            ax.tick_params(colors="white")
            ax.set_facecolor("black")

        time_text = fig.text(0.5, 0.01, "", ha="center", color="gold",
                             fontsize=13)

        def update(frame):
            im0.set_data(all_bnd[frame].T)
            im1.set_data(all_mid[frame].T)
            im2.set_data(all_ir[frame].T)
            time_text.set_text(f"t = {self.t_lin[frame]:.3f}")
            return im0, im1, im2, time_text

        anim = animation.FuncAnimation(fig, update, frames=len(self.t_lin),
                                       interval=1000 // fps, blit=True)
        anim.save(save_path, writer="pillow", fps=fps, dpi=100)
        print(f"[viz] Time animation saved → {save_path}")
        plt.close(fig)

    # ================================================================== #
    #  GROUND-TRUTH COMPARISON  (if apex_master_dataset.npz available)     #
    # ================================================================== #
    def render_ground_truth_comparison(self, dataset_path, sim_idx=None,
                                       save_path="gt_comparison.png"):
        """Side-by-side: ground truth  vs  model reconstruction  vs  |residual|."""
        if sim_idx is None:
            sim_idx = self.sim_idx

        if not os.path.exists(dataset_path):
            print(f"[warn] Dataset not found at {dataset_path}")
            return

        data = np.load(dataset_path)
        pts = data["pinn_points"]

        # Slice near coalescence t ≈ 0.5
        mask = (pts[:, 0] == sim_idx) & (np.abs(pts[:, 1] - 0.5) < 0.1)
        gt = pts[mask]
        if len(gt) == 0:
            print("[warn] No points near t=0.5. Using first 5000 bulk samples.")
            gt = pts[pts[:, 0] == sim_idx][:5000]

        coords = torch.from_numpy(gt[:, 1:5]).float().to(self.device)
        with torch.no_grad():
            z_exp = self.z_latent.expand(coords.shape[0], -1)
            pred = self.siren(coords, z_exp).cpu().numpy().flatten()

        gt_phi = gt[:, 5]
        residual = np.abs(gt_phi - pred)

        fig = plt.figure(figsize=(20, 7), facecolor="black")
        tc = "white"

        ax1 = fig.add_subplot(131, projection="3d", facecolor="black")
        sc1 = ax1.scatter(gt[:, 2], gt[:, 3], gt[:, 4],
                          c=gt_phi, cmap="viridis", alpha=0.5, s=2)
        ax1.set_title("Ground Truth", color=tc)
        ax1.set_xlabel("x", color=tc); ax1.set_ylabel("y", color=tc)
        ax1.set_zlabel("u", color=tc)
        ax1.tick_params(colors=tc)
        plt.colorbar(sc1, ax=ax1, shrink=0.5)

        ax2 = fig.add_subplot(132, projection="3d", facecolor="black")
        sc2 = ax2.scatter(gt[:, 2], gt[:, 3], gt[:, 4],
                          c=pred, cmap="plasma", alpha=0.5, s=2)
        ax2.set_title("Model Reconstruction", color=tc)
        ax2.set_xlabel("x", color=tc); ax2.set_ylabel("y", color=tc)
        ax2.set_zlabel("u", color=tc)
        ax2.tick_params(colors=tc)
        plt.colorbar(sc2, ax=ax2, shrink=0.5)

        ax3 = fig.add_subplot(133, projection="3d", facecolor="black")
        sc3 = ax3.scatter(gt[:, 2], gt[:, 3], gt[:, 4],
                          c=residual, cmap="hot", alpha=0.5, s=2)
        mse = np.mean(residual ** 2)
        ax3.set_title(f"|Residual|  (MSE = {mse:.6f})", color=tc)
        ax3.set_xlabel("x", color=tc); ax3.set_ylabel("y", color=tc)
        ax3.set_zlabel("u", color=tc)
        ax3.tick_params(colors=tc)
        plt.colorbar(sc3, ax=ax3, shrink=0.5)

        fig.suptitle(f"Ground Truth vs Model — Sim {sim_idx}  (t ≈ 0.5)",
                     color="gold", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[viz] GT comparison saved → {save_path}")
        plt.show()

    # ================================================================== #
    #  MASTER — RENDER ALL                                                 #
    # ================================================================== #
    def render_all(self, t_idx=25, output_dir="viz_output"):
        """Generate every visualisation and save to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        hdr = (
            "\n" + "=" * 64 + "\n"
            "   4D HOLOGRAPHIC SPACETIME VISUALIZER\n"
            "   Binary Black Hole Merger — AdS/CFT Correspondence\n"
            + "=" * 64 + "\n"
        )
        print(hdr)

        j = os.path.join  # shorthand

        # 1. Dashboard
        self.render_full_dashboard(t_idx, j(output_dir, "dashboard.png"))

        # 2. 3D Isosurface (Plotly)
        ext = "html" if HAS_PLOTLY else "png"
        self.render_3d_isosurface(t_idx, j(output_dir, f"isosurface_3d.{ext}"))

        # 3. Holographic projection
        self.render_holographic_projection(j(output_dir, "holographic_projection.png"))

        # 4. QNM chirp
        self.render_qnm_plot(j(output_dir, "qnm_chirp.png"))

        # 5. Sommerfeld glow
        self.render_sommerfeld_glow(t_idx, j(output_dir, "sommerfeld_glow.png"))

        # 6. Causality map
        self.render_causality_map(t_idx, j(output_dir, "causality_map.png"))

        # 7. Radial profile
        self.render_radial_profile(t_idx, j(output_dir, "radial_profile.png"))

        # 8. Time animation (GIF)
        self.render_time_animation(j(output_dir, "holographic_evolution.gif"))

        print(f"\n[viz] ✓  All outputs saved to: {output_dir}/")
        print("=" * 64)


# ====================================================================== #
#  ENTRY POINT                                                             #
# ====================================================================== #
if __name__ == "__main__":

    # ------ Paths (auto-discover on Kaggle) ------
    CKPT = "/kaggle/working/checkpoints/best_model.pt"
    DATASET = None

    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "apex_master_dataset.npz":
                DATASET = os.path.join(root, f)
                break
        if DATASET:
            break

    if not os.path.exists(CKPT):
        # Fallback for local dev
        for alt in ["checkpoints/best_model.pt", "best_model.pt"]:
            if os.path.exists(alt):
                CKPT = alt
                break

    print(f"Checkpoint : {CKPT}")
    print(f"Dataset    : {DATASET or '(synthetic)'}")

    viz = HolographicVisualizer(
        checkpoint_path=CKPT,
        dataset_path=DATASET,
        sim_idx=0,
        nx=100, ny=100, nu=50, nt=50,
    )

    # Coalescence frame (t ≈ 0.5)
    coal_idx = viz.cfg.NT // 2

    viz.render_all(t_idx=coal_idx, output_dir="/kaggle/working/viz_output")

    # Ground truth comparison if dataset exists
    if DATASET:
        viz.render_ground_truth_comparison(
            DATASET,
            save_path="/kaggle/working/viz_output/gt_comparison.png",
        )
