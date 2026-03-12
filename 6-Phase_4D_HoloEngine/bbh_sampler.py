"""
bbh_sampler.py — Coordinate Sampler for the BBH Characteristic Formulation
============================================================================

Generates training collocation points in the 3D domain (v, x, u) with
specialized sampling for:

1. **Boundary**  (u ≈ u_min):  Enforce holographic dictionary
2. **Bulk**:  Random collocation for Einstein equation residuals
3. **Cauchy surface**  (v = 0):  Enforce initial data
4. **Radial lines**  (fixed v, x; sweep u):  Enforce nested ODE structure
5. **Horizon region**:  Dense sampling near A ≈ 0 for regularity
6. **Causal time chunks**:  v-axis partitioned for causal PINN weighting
"""

import math
import torch
from ads_config import BBHConfig


def sample_boundary(n: int, cfg: type = BBHConfig,
                    device: str = None) -> torch.Tensor:
    """
    Points on the AdS boundary  u = u_min  (z → 0).

    Returns
    -------
    coords : (n, 3) — [v, x, u_boundary]
    """
    device = device or cfg.DEVICE
    v = torch.rand(n, device=device) * (cfg.V_RANGE[1] - cfg.V_RANGE[0]) + cfg.V_RANGE[0]
    x = torch.rand(n, device=device) * (cfg.X_RANGE[1] - cfg.X_RANGE[0]) + cfg.X_RANGE[0]
    u = torch.full((n,), cfg.U_BOUNDARY, device=device)
    return torch.stack([v, x, u], dim=-1)


def sample_bulk(n: int, cfg: type = BBHConfig,
                device: str = None, excision_u: float = None) -> torch.Tensor:
    """
    Uniform random points in the full (v, x, u) domain.
    Slightly biased toward the boundary (small u) where gradients matter more.
    Optionally excises points inside an estimated horizon.

    Returns
    -------
    coords : (n, 3) — [v, x, u]
    """
    device = device or cfg.DEVICE
    v = torch.rand(n, device=device) * (cfg.V_RANGE[1] - cfg.V_RANGE[0]) + cfg.V_RANGE[0]
    x = torch.rand(n, device=device) * (cfg.X_RANGE[1] - cfg.X_RANGE[0]) + cfg.X_RANGE[0]

    # Bias toward boundary: sample uniformly in z, then convert to u = ln(z)
    z = torch.rand(n, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN
    u = torch.log(z)

    coords = torch.stack([v, x, u], dim=-1)

    # Excision: remove points inside estimated horizon
    if excision_u is not None:
        mask = u < (excision_u - cfg.EXCISION_BUFFER)
        if mask.sum() > n // 2:
            coords = coords[mask]

    return coords


def sample_cauchy(n: int, cfg: type = BBHConfig,
                  device: str = None) -> torch.Tensor:
    """
    Points on the initial time slice  v = 0.

    Returns
    -------
    coords : (n, 3) — [0, x, u]
    """
    device = device or cfg.DEVICE
    v = torch.zeros(n, device=device)
    x = torch.rand(n, device=device) * (cfg.X_RANGE[1] - cfg.X_RANGE[0]) + cfg.X_RANGE[0]
    z = torch.rand(n, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN
    u = torch.log(z)
    return torch.stack([v, x, u], dim=-1)


def sample_radial_lines(n_lines: int, n_per_line: int,
                        cfg: type = BBHConfig,
                        device: str = None) -> torch.Tensor:
    """
    Lines at fixed (v, x) sweeping through u.

    These enforce the NESTED ODE structure: at each (v, x),
    the radial equations R1-R4 must be satisfied along the
    entire u-coordinate simultaneously.

    Returns
    -------
    coords : (n_lines * n_per_line, 3) — [v, x, u]
    """
    device = device or cfg.DEVICE

    # Pick random (v, x) anchor points
    v_anchors = torch.rand(n_lines, device=device) * (cfg.V_RANGE[1] - cfg.V_RANGE[0]) + cfg.V_RANGE[0]
    x_anchors = torch.rand(n_lines, device=device) * (cfg.X_RANGE[1] - cfg.X_RANGE[0]) + cfg.X_RANGE[0]

    # Create uniform u grid for each line
    u_line = torch.linspace(cfg.U_MIN, cfg.U_MAX, n_per_line, device=device)

    # Broadcast: (n_lines, 1) x (1, n_per_line) -> flatten
    v_all = v_anchors.unsqueeze(1).expand(n_lines, n_per_line).reshape(-1)
    x_all = x_anchors.unsqueeze(1).expand(n_lines, n_per_line).reshape(-1)
    u_all = u_line.unsqueeze(0).expand(n_lines, n_per_line).reshape(-1)

    return torch.stack([v_all, x_all, u_all], dim=-1)


def sample_horizon_region(n: int, z_horizon_estimate: float = 0.5,
                          cfg: type = BBHConfig,
                          device: str = None) -> torch.Tensor:
    """
    Dense sampling around the estimated horizon location.

    The horizon is where  A(v, x, z) = 0,  typically somewhere
    around z ~ z_h.  We concentrate points near this surface.

    Returns
    -------
    coords : (n, 3) — [v, x, u]
    """
    device = device or cfg.DEVICE
    v = torch.rand(n, device=device) * (cfg.V_RANGE[1] - cfg.V_RANGE[0]) + cfg.V_RANGE[0]
    x = torch.rand(n, device=device) * (cfg.X_RANGE[1] - cfg.X_RANGE[0]) + cfg.X_RANGE[0]

    # Gaussian-concentrate around z_horizon
    z_center = z_horizon_estimate
    z_spread = 0.15  # spread around horizon
    z = z_center + z_spread * torch.randn(n, device=device)
    z = z.clamp(cfg.Z_MIN, cfg.Z_MAX)
    u = torch.log(z)

    return torch.stack([v, x, u], dim=-1)


def sample_time_chunk(n: int, chunk_idx: int,
                      cfg: type = BBHConfig,
                      device: str = None) -> torch.Tensor:
    """
    Points within a specific time chunk for causal PINN training.

    The v-domain is split into NUM_TIME_CHUNKS slabs.
    Chunk i covers v ∈ [v_min + i·Δv, v_min + (i+1)·Δv].

    Returns
    -------
    coords : (n, 3) — [v, x, u]
    """
    device = device or cfg.DEVICE
    v_min, v_max = cfg.V_RANGE
    dv = (v_max - v_min) / cfg.NUM_TIME_CHUNKS

    v_lo = v_min + chunk_idx * dv
    v_hi = v_lo + dv

    v = torch.rand(n, device=device) * (v_hi - v_lo) + v_lo
    x = torch.rand(n, device=device) * (cfg.X_RANGE[1] - cfg.X_RANGE[0]) + cfg.X_RANGE[0]
    z = torch.rand(n, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN
    u = torch.log(z)

    return torch.stack([v, x, u], dim=-1)


def causal_weight(v: torch.Tensor, chunk_idx: int,
                  cfg: type = BBHConfig) -> torch.Tensor:
    """
    Causal weighting factor w(v) for a time chunk.

    Points at earlier times get exponentially HIGHER weight,
    enforcing that the network learns accurate physics at early
    times before attempting later evolution.

    w(v) = exp(-ε (v - v_chunk_start))

    Returns
    -------
    weights : (N,) — per-point causal weights
    """
    v_min = cfg.V_RANGE[0]
    dv = (cfg.V_RANGE[1] - v_min) / cfg.NUM_TIME_CHUNKS
    v_start = v_min + chunk_idx * dv

    return torch.exp(-cfg.CAUSAL_EPSILON * (v - v_start))


class BBHBatchSampler:
    """
    Unified batch generator for BBH training.

    Each call to `sample()` returns a dict with all coordinate batches
    needed for one training step:
        - boundary:  for holographic dictionary enforcement
        - bulk:      for Einstein equation residuals (with excision)
        - cauchy:    for initial data at v=0
        - radial:    for nested ODE structure
        - horizon:   for horizon regularity (Phase C only)
        - adaptive:  for near-horizon PDE enforcement (Phase C only)
    """

    def __init__(self, cfg: type = BBHConfig, device: str = None):
        self.cfg = cfg
        self.device = device or cfg.DEVICE
        self.z_horizon_est = 0.5  # Updated during training from horizon finder
        self.u_horizon_est = None  # ln(z_horizon_est) when available

    def update_horizon_estimate(self, z_h: float):
        """Update the estimated horizon location."""
        self.z_horizon_est = z_h
        self.u_horizon_est = math.log(max(z_h, self.cfg.Z_MIN))

    def _adaptive_near_horizon(self, n):
        """Sample points concentrated near the estimated horizon."""
        cfg, dev = self.cfg, self.device
        v = torch.rand(n, device=dev) * (cfg.V_RANGE[1] - cfg.V_RANGE[0]) + cfg.V_RANGE[0]
        x = torch.rand(n, device=dev) * (cfg.X_RANGE[1] - cfg.X_RANGE[0]) + cfg.X_RANGE[0]
        u_center = self.u_horizon_est if self.u_horizon_est is not None else math.log(0.5)
        u = u_center + 0.1 * torch.randn(n, device=dev)
        u = u.clamp(cfg.U_MIN, cfg.U_MAX)
        return torch.stack([v, x, u], dim=-1)

    def sample(self, phase: str = 'A') -> dict:
        cfg = self.cfg
        dev = self.device

        # In Phase C with excision, pass horizon estimate
        excision_u = self.u_horizon_est if phase == 'C' and self.u_horizon_est is not None else None

        batch = {
            'boundary': sample_boundary(cfg.BOUNDARY_BATCH, cfg, dev),
            'bulk': sample_bulk(cfg.BULK_BATCH, cfg, dev, excision_u=excision_u),
            'cauchy': sample_cauchy(getattr(cfg, 'CAUCHY_BATCH', cfg.BOUNDARY_BATCH // 2), cfg, dev),
        }

        if phase in ('B', 'C'):
            n_lines = cfg.RADIAL_BATCH // 32
            batch['radial'] = sample_radial_lines(n_lines, 32, cfg, dev)

        if phase == 'C':
            batch['horizon'] = sample_horizon_region(
                cfg.HORIZON_BATCH, self.z_horizon_est, cfg, dev)
            # Adaptive near-horizon PDE enforcement
            n_adp = int(cfg.BULK_BATCH * getattr(cfg, 'ADAPTIVE_FRAC', 0.25))
            if n_adp > 0:
                batch['adaptive'] = self._adaptive_near_horizon(n_adp)

        for key in batch:
            batch[key] = batch[key].requires_grad_(True)

        return batch
