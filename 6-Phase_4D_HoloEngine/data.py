"""
data.py - Data Loading & Synthetic Collision-Fluid Generation
==============================================================

Two data pipelines:

1. **Master Dataset** (apex_master_dataset.npz)
   Loads pre-computed relativistic collision data with Lorentz contraction,
   critical collapse amplitude, Sommerfeld radiation, and entropy targets.
   Used for production training on Kaggle.

2. **Synthetic Fallback**
   Generates a simplified binary-merger collision sequence on the fly.
   Used for local development and testing without the .npz file.

Both return a unified tuple:
    (cnn_volume, boundary_coords, boundary_values, entropy_target, time_ticks)
"""

import os
import math
import torch
import numpy as np
from config import Config


# ====================================================================== #
#  MASTER DATASET LOADER                                                   #
# ====================================================================== #
def load_master_dataset(config: type = Config, sim_idx: int = 1):
    """
    Load the Holographic Boundary Dataset from apex_master_dataset.npz.

    Searches multiple candidate paths (Kaggle input dirs + local).

    Parameters
    ----------
    config  : Config class
    sim_idx : int – which simulation to use (0, 1, or 2 for v=0.4, 0.6, 0.8)

    Returns
    -------
    Tuple of (cnn_volume, boundary_coords, boundary_values, entropy_target, time_ticks)
    or None if the file is not found.
    """
    candidates = [
        '/kaggle/input/datasets/avshrek/holographic-4d-engine/apex_master_dataset.npz',
        '/kaggle/input/holographic-4d-engine/apex_master_dataset.npz',
        'apex_master_dataset.npz',
        os.path.join(os.path.dirname(__file__), 'apex_master_dataset.npz'),
    ]

    dataset_path = None
    for p in candidates:
        if os.path.exists(p):
            dataset_path = p
            break

    if dataset_path is None:
        return None

    data = np.load(dataset_path)

    # CNN volume: (1, T, H, W)  for the selected simulation
    cnn_vol = torch.from_numpy(data['cnn_volumes'][sim_idx]).float()

    # PINN collocation table: filter for this simulation
    pinn_pts = data['pinn_points']
    sim_mask = (pinn_pts[:, 0] == sim_idx)
    sim_pts  = pinn_pts[sim_mask]

    # Columns: [sim_id, t, x, y, u, phi, dphi_du]
    boundary_coords = torch.from_numpy(sim_pts[:, 1:5]).float()   # (N, 4)
    boundary_values = torch.from_numpy(sim_pts[:, 5]).float()     # (N,)

    # Entropy targets over time
    entropy_target = torch.from_numpy(data['entropy_targets'][sim_idx]).float()
    time_ticks     = torch.from_numpy(data['time_ticks']).float()

    return cnn_vol, boundary_coords, boundary_values, entropy_target, time_ticks


# ====================================================================== #
#  SYNTHETIC DATA GENERATOR                                                #
# ====================================================================== #
def generate_synthetic_data(config: type = Config):
    """
    Generate a synthetic binary-merger collision fluid sequence.

    Physics picture
    ---------------
    Two Gaussian energy concentrations ("black holes") orbit each other
    on shrinking inspiral trajectories, coalesce at t ~ 0.7, and
    produce a quasi-normal ringdown afterward.

    Returns
    -------
    cnn_volume      : Tensor (T, H, W)  with values in [0, 1]
    boundary_coords : Tensor (T*H*W, 4) with columns [t, x, y, u_boundary]
    boundary_values : Tensor (T*H*W,)
    entropy_target  : Tensor scalar (approximate S_max)
    time_ticks      : Tensor (T,)
    """
    T = config.ENCODER_TEMPORAL_FRAMES
    H = config.ENCODER_SPATIAL_RES
    W = config.ENCODER_SPATIAL_RES

    x = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], W)
    y = torch.linspace(config.Y_RANGE[0], config.Y_RANGE[1], H)
    yy, xx = torch.meshgrid(y, x, indexing="ij")          # (H, W) each

    data = torch.zeros(T, H, W)

    for t_idx in range(T):
        t = t_idx / max(T - 1, 1)                         # normalised [0, 1]

        # --- inspiral trajectory ---
        separation = 0.6 * (1.0 - t)                      # monotonically closing
        angle      = 2.5 * math.pi * t                    # ~1.25 orbits
        cx1 = separation * math.cos(angle)
        cy1 = separation * math.sin(angle)
        cx2, cy2 = -cx1, -cy1

        # --- blob parameters ---
        sigma = 0.15 + 0.10 * (1.0 - t)                   # tightens as merge
        amp   = 1.0  + 2.0  * t                            # focusing amplification

        blob1 = amp * torch.exp(-((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2.0 * sigma ** 2))
        blob2 = amp * torch.exp(-((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2.0 * sigma ** 2))

        frame = blob1 + blob2

        # --- post-merger ringdown ---
        if t > 0.7:
            ringdown_frac = (t - 0.7) / 0.3
            r = torch.sqrt(xx ** 2 + yy ** 2 + 1e-8)
            ring = ringdown_frac * 0.5 * torch.sin(12.0 * r - 8.0 * t) \
                   * torch.exp(-r ** 2 / 0.25)
            frame = frame + ring

        data[t_idx] = frame

    # Normalise globally to [0, 1]
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    # Build boundary coordinate table
    t_vals = torch.linspace(config.T_RANGE[0], config.T_RANGE[1], T)
    x_vals = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], W)
    y_vals = torch.linspace(config.Y_RANGE[0], config.Y_RANGE[1], H)

    tt, yy_c, xx_c = torch.meshgrid(t_vals, y_vals, x_vals, indexing="ij")
    boundary_coords = torch.stack([
        tt.flatten(),
        xx_c.flatten(),
        yy_c.flatten(),
        torch.full((T * H * W,), config.U_BOUNDARY),
    ], dim=-1)

    boundary_values = data.flatten()
    entropy_target  = torch.tensor(3.0)    # approximate S_max for v=0.6

    return data, boundary_coords, boundary_values, entropy_target, t_vals


# ====================================================================== #
#  UNIFIED LOADER                                                          #
# ====================================================================== #
def load_data(config: type = Config, sim_idx: int = 1):
    """
    Try master dataset first; fall back to synthetic.

    Returns
    -------
    cnn_volume       : Tensor — CNN encoder input volume
    boundary_coords  : Tensor (N, 4)
    boundary_values  : Tensor (N,)
    entropy_target   : Tensor — scalar or (T,) entropy targets
    time_ticks       : Tensor (T,)
    source           : str — 'master' or 'synthetic'
    """
    result = load_master_dataset(config, sim_idx)
    if result is not None:
        cnn_vol, bnd_c, bnd_v, ent, tt = result
        return cnn_vol, bnd_c, bnd_v, ent, tt, "master"

    cnn_vol, bnd_c, bnd_v, ent, tt = generate_synthetic_data(config)
    return cnn_vol, bnd_c, bnd_v, ent, tt, "synthetic"
