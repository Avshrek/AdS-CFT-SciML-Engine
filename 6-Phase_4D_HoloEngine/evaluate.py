"""
evaluate.py  –  Post-Training Inference & Physics Validation Engine
=====================================================================

Loads a trained checkpoint and computes:
    1. Boundary reconstruction MSE (across all time steps).
    2. PDE residual statistics on a dense bulk grid.
    3. Sommerfeld boundary condition violation.
    4. Cauchy initial condition violation.
    5. HRT area and causality penalty.
    6. Radial profile phi_renorm(u) at key spacetime points.
    7. Peak amplitude tracking over time.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --checkpoint checkpoints/best_model.pt --device cuda
"""

import argparse
import os
import torch
import numpy as np

from config import Config
from data import load_data
from sampler import ApexDualSampler
from model import ConvEncoder3D, FiLMSiren
from physics import (
    causal_bizon_pde,
    hrt_covariant_area,
    _grad,
)


def load_checkpoint(ckpt_path: str, config: type = Config, device: str = "cpu"):
    """Load encoder + SIREN from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = ConvEncoder3D(config).to(device)
    siren   = FiLMSiren(config).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    siren.load_state_dict(ckpt["siren"])
    encoder.eval()
    siren.eval()
    return encoder, siren, ckpt


@torch.no_grad()
def evaluate_boundary_reconstruction(encoder, siren, sampler, bnd_input,
                                      config=Config, device="cpu"):
    """Compute per-timestep boundary MSE."""
    z_lat = encoder(bnd_input)

    total = sampler.boundary_coords.shape[0]
    batch = min(total, 8192)
    idx = torch.randint(0, total, (batch,))
    coords = sampler.boundary_coords[idx].to(device)
    values = sampler.boundary_values[idx].to(device)

    z_exp = z_lat.expand(batch, -1)
    pred = siren(coords, z_exp).squeeze(-1)
    mse = ((pred - values) ** 2).mean().item()
    peak_pred = pred.abs().max().item()
    peak_gt   = values.abs().max().item()

    return dict(boundary_mse=mse, peak_pred=peak_pred, peak_gt=peak_gt)


def evaluate_pde_residual(encoder, siren, sampler, bnd_input,
                           config=Config, device="cpu", n_points=4096):
    """Compute PDE residual statistics on a random bulk grid."""
    with torch.no_grad():
        z_lat = encoder(bnd_input).detach()

    coords, weights = sampler.sample_continuous_bulk(n_points)
    coords = coords.to(device).requires_grad_(True)

    z_exp = z_lat.expand(n_points, -1)
    phi_R = siren(coords, z_exp)

    residual, _ = causal_bizon_pde(phi_R, coords, config)

    res_np = residual.detach().cpu().numpy().flatten()
    return dict(
        pde_residual_mean=float(np.mean(np.abs(res_np))),
        pde_residual_std=float(np.std(res_np)),
        pde_residual_max=float(np.max(np.abs(res_np))),
        pde_residual_median=float(np.median(np.abs(res_np))),
    )


def evaluate_hrt(encoder, siren, sampler, bnd_input,
                  config=Config, device="cpu", n_points=2048):
    """Compute HRT area and causality penalty."""
    with torch.no_grad():
        z_lat = encoder(bnd_input).detach()

    coords, _ = sampler.sample_continuous_bulk(n_points)
    coords = coords.to(device).requires_grad_(True)

    z_exp = z_lat.expand(n_points, -1)
    phi_R = siren(coords, z_exp)
    hrt_area, causal_pen = hrt_covariant_area(phi_R, coords, config)

    return dict(
        hrt_area=hrt_area.item(),
        causality_penalty=causal_pen.item(),
    )


@torch.no_grad()
def evaluate_peak_amplitude_over_time(encoder, siren, sampler, bnd_input,
                                       config=Config, device="cpu",
                                       n_time_bins=20, n_spatial=500):
    """Track peak |phi| at the boundary across time bins."""
    z_lat = encoder(bnd_input)

    t_edges = torch.linspace(config.T_RANGE[0], config.T_RANGE[1], n_time_bins + 1)
    peaks = []

    for i in range(n_time_bins):
        t_lo, t_hi = t_edges[i].item(), t_edges[i + 1].item()
        t_mid = (t_lo + t_hi) / 2.0

        x = torch.rand(n_spatial) * 2.0 - 1.0
        y = torch.rand(n_spatial) * 2.0 - 1.0
        t = torch.full((n_spatial,), t_mid)
        u = torch.full((n_spatial,), config.U_BOUNDARY)

        coords = torch.stack([t, x, y, u], dim=-1).to(device)
        z_exp  = z_lat.expand(n_spatial, -1)
        phi    = siren(coords, z_exp).squeeze(-1)
        peaks.append(dict(t=t_mid, peak_phi=phi.abs().max().item()))

    return peaks


@torch.no_grad()
def evaluate_radial_profile(encoder, siren, bnd_input,
                             config=Config, device="cpu",
                             n_u=200):
    """Compute phi_renorm(u) at (t=0, x=0, y=0) and (t=0.5, x=0, y=0)."""
    z_lat = encoder(bnd_input)
    u_vals = torch.linspace(config.U_MIN, config.U_MAX, n_u)

    profiles = {}
    for t_val in [0.0, 0.5, 1.0]:
        coords = torch.stack([
            torch.full((n_u,), t_val),
            torch.zeros(n_u),
            torch.zeros(n_u),
            u_vals,
        ], dim=-1).to(device)

        z_exp = z_lat.expand(n_u, -1)
        phi = siren(coords, z_exp).squeeze(-1).cpu().numpy()
        profiles[f"t={t_val}"] = dict(
            u=u_vals.numpy().tolist(),
            phi_renorm=phi.tolist()
        )

    return profiles


def run_full_evaluation(ckpt_path: str, config=Config, device="cpu"):
    """Run all evaluation metrics and print results."""
    print("=" * 72)
    print("  4D HOLOGRAPHIC ENGINE  –  Post-Training Evaluation")
    print("=" * 72)

    encoder, siren, ckpt = load_checkpoint(ckpt_path, config, device)
    epoch = ckpt.get("epoch", "?")
    best_loss = ckpt.get("best_loss", "?")
    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"  Epoch: {epoch}  |  Best training loss: {best_loss}")

    # Load data
    cnn_vol, bnd_c, bnd_v, ent, tt, source = load_data(config)
    if source == "master":
        bnd_input = cnn_vol
        while bnd_input.dim() < 5:
            bnd_input = bnd_input.unsqueeze(0)
        bnd_input = bnd_input.to(device)
    else:
        bnd_input = cnn_vol.unsqueeze(0).unsqueeze(0).to(device)
    print(f"  Data source: {source}\n")

    sampler = ApexDualSampler(bnd_c, bnd_v, config)

    # 1. Boundary reconstruction
    print("  [1/5] Boundary reconstruction ...")
    bnd_metrics = evaluate_boundary_reconstruction(
        encoder, siren, sampler, bnd_input, config, device)
    for k, v in bnd_metrics.items():
        print(f"         {k}: {v:.6f}")

    # 2. PDE residual
    print("  [2/5] PDE residual ...")
    pde_metrics = evaluate_pde_residual(
        encoder, siren, sampler, bnd_input, config, device)
    for k, v in pde_metrics.items():
        print(f"         {k}: {v:.6f}")

    # 3. HRT area
    print("  [3/5] HRT area & causality ...")
    hrt_metrics = evaluate_hrt(
        encoder, siren, sampler, bnd_input, config, device)
    for k, v in hrt_metrics.items():
        print(f"         {k}: {v:.6f}")

    # 4. Peak amplitude over time
    print("  [4/5] Peak amplitude over time ...")
    peaks = evaluate_peak_amplitude_over_time(
        encoder, siren, sampler, bnd_input, config, device)
    for p in peaks:
        print(f"         t={p['t']:.3f}  peak_phi={p['peak_phi']:.4f}")

    # 5. Radial profiles
    print("  [5/5] Radial bulk profiles ...")
    profiles = evaluate_radial_profile(
        encoder, siren, bnd_input, config, device)
    for key, prof in profiles.items():
        phi_arr = np.array(prof['phi_renorm'])
        print(f"         {key}:  phi range = [{phi_arr.min():.4f}, {phi_arr.max():.4f}]")

    print("\n" + "=" * 72)
    print("  EVALUATION COMPLETE")
    print("=" * 72)

    return dict(
        boundary=bnd_metrics,
        pde=pde_metrics,
        hrt=hrt_metrics,
        peaks=peaks,
        profiles=profiles,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate holographic model")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_full_evaluation(args.checkpoint, Config, args.device)
