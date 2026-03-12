"""
physics.py - Relativistic Dynamics, Causal PINN, Boundary Conditions, HRT
==========================================================================

Implements Phases 3-6 of the blueprint with ALL numerical fixes:

    • causal_bizon_pde()          – exact u-metric d'Alembertian + backreaction
    • causal_pinn_weights()       – CFL-aware causal decay with log1p (FIX 10)
    • sommerfeld_radiative_loss() – orthogonal absorbing BCs with log1p (FIX 11)
    • bulk_cauchy_loss()          – bulk-interior-only Cauchy (FIX 12)
    • hrt_covariant_area()        – differential swelling + log1p causality (FIX 13)

All PDE / HRT paths use  z_latent.detach()  to sever the CNN encoder from
the physics autograd graph (prevents VRAM death & empty-universe cheating).

The Bianchi identity guarantees ∇_μ T^{μν} = 0 when the scalar EOM is
satisfied, so NO explicit 3rd-order divergence loss is computed.
"""

import math
import torch
import torch.nn.functional as F

from config import Config


# ====================================================================== #
#  UTILITY:  2nd-order autograd derivatives                               #
# ====================================================================== #
def _grad(outputs, inputs, create_graph=True):
    """Thin wrapper around torch.autograd.grad with sane defaults."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def compute_derivatives(phi_renorm: torch.Tensor, coords: torch.Tensor):
    """
    Compute all first and second partial derivatives of φ_renorm
    with respect to the 4-D coordinate vector (t, x, y, u).

    Parameters
    ----------
    phi_renorm : (B, 1) – network output (finite renormalised mode)
    coords     : (B, 4) – with requires_grad=True

    Returns
    -------
    dict with keys: phi, dt, dx, dy, du, dt2, dx2, dy2, du2
    """
    g = _grad(phi_renorm, coords)                   # (B, 4)
    dt, dx, dy, du = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]

    dt2 = _grad(dt, coords)[:, 0:1]
    dx2 = _grad(dx, coords)[:, 1:2]
    dy2 = _grad(dy, coords)[:, 2:3]
    du2 = _grad(du, coords)[:, 3:4]

    return dict(phi=phi_renorm,
                dt=dt, dx=dx, dy=dy, du=du,
                dt2=dt2, dx2=dx2, dy2=dy2, du2=du2)


# ====================================================================== #
#  PHASE 4-A :  CAUSAL BIZON PDE                                          #
# ====================================================================== #
def causal_bizon_pde(phi_renorm: torch.Tensor,
                     coords:     torch.Tensor,
                     config:     type = Config):
    """
    Exact u-metric d'Alembertian PDE residual with:
        • Conformal-frame backreaction  (1 + clamp(κ T_conf, max=κ_max))
        • Non-linear φ³ self-interaction
        • Algebraic mass-gap annihilation shift

    The residual that must vanish:

        -(1 + κ_eff) e^{2u} ∂²_t φ_R
        + e^{2u} (∂²_x φ_R + ∂²_y φ_R)
        + ∂²_u φ_R
        + 3 ∂_u φ_R
        - λ φ_R³ e^{6u}
        = 0

    FIX 3: Backreaction uses conformal-frame stress-energy (no e^{6u}
    amplification) and is clamped to max KAPPA_MAX = 5 → max 6× PDE
    amplification instead of the old 51×.

    Parameters
    ----------
    phi_renorm : (B, 1)   – SIREN output (requires_grad via coords)
    coords     : (B, 4)   – (t, x, y, u) with requires_grad=True

    Returns
    -------
    residual : (B, 1)
    derivs   : dict of derivative tensors
    """
    u   = coords[:, 3:4]
    e2u = torch.exp(2.0 * u)
    e6u = torch.exp(6.0 * u)

    derivs = compute_derivatives(phi_renorm, coords)

    # --- Conformal-frame stress-energy proxy for backreaction ---
    # FIX 3: Use conformal-frame T (no e^{6u}), clamp to KAPPA_MAX
    with torch.no_grad():
        T_conformal = 0.5 * (
            derivs['dt'] ** 2
            + derivs['dx'] ** 2
            + derivs['dy'] ** 2
            + derivs['du'] ** 2
        )
        backreaction = 1.0 + torch.clamp(
            config.KAPPA * T_conformal,
            max=config.KAPPA_MAX
        )                                                       # ∈ [1.0, 6.0]

    # --- PDE residual ---
    residual = (
        - backreaction * e2u * derivs['dt2']                    # time
        + e2u * (derivs['dx2'] + derivs['dy2'])                 # transverse
        + derivs['du2']                                          # radial 2nd
        + 3.0 * derivs['du']                                    # radial 1st (drift)
        - config.LAMBDA_NL * (phi_renorm ** 3) * e6u            # non-linear
    )

    return residual, derivs


# ====================================================================== #
#  PHASE 5-A :  CAUSAL PINN WEIGHTS  (FIX 10: log1p)                      #
# ====================================================================== #
def causal_pinn_weights(residuals:    torch.Tensor,
                        t_coords:     torch.Tensor,
                        config:       type = Config,
                        volume_w:     torch.Tensor = None) -> torch.Tensor:
    """
    Time-sorted, CFL-aware, chunk-binned causal PINN loss with log1p.

    FIX 10: log1p(res²) maps ANY residual to O(1)–O(50).
    Old res² with SIREN derivatives O(ω₀²)=3600 and backreaction=51×
    gave total loss O(10^29). Now gradient = 2r/(1+r²) · dr/dθ which
    naturally dampens enormous residuals while treating small ones equally.

        W_k = exp( -ε  ·  Σ_{j<k}  Mean( log1p(L²_res_chunk_j) ) )

    Parameters
    ----------
    residuals : (B, 1)
    t_coords  : (B,)     detached time coordinates
    config    : Config
    volume_w  : (B,)     optional AdS volume measure weights  e^{-3u}

    Returns
    -------
    weighted_loss : scalar
    """
    device = residuals.device
    B = residuals.shape[0]

    # Sort by time
    idx    = torch.argsort(t_coords)
    res_sq = torch.log1p(residuals[idx].squeeze(-1) ** 2)       # FIX 10

    # Apply AdS volume measure: emphasise boundary-adjacent points
    if volume_w is not None:
        w_sorted = volume_w[idx]
        w_sorted = w_sorted / (w_sorted.mean() + 1e-8)
        res_sq   = res_sq * w_sorted

    # Adaptive chunk count respecting CFL:  Δt_chunk ≤ Δx_min ≈ 2 / B^{1/4}
    dx_min   = 2.0 / (B ** 0.25 + 1e-8)
    dt_total = config.T_RANGE[1] - config.T_RANGE[0]
    n_chunks = max(int(math.ceil(dt_total / dx_min)), config.NUM_TIME_CHUNKS)
    n_chunks = min(n_chunks, config.NUM_TIME_CHUNKS * 2)        # cap for memory

    chunk_sz = max(B // n_chunks, 1)

    # Mean residual² per chunk
    chunk_losses = []
    for k in range(n_chunks):
        lo = k * chunk_sz
        hi = min(lo + chunk_sz, B) if k < n_chunks - 1 else B
        chunk_losses.append(res_sq[lo:hi].mean())
    chunk_losses = torch.stack(chunk_losses)                    # (n_chunks,)

    # Causal weights  W_k = exp( -ε Σ_{j<k} L_j )
    cumsum        = torch.cumsum(chunk_losses, dim=0).detach()
    shifted       = torch.cat([torch.zeros(1, device=device), cumsum[:-1]])
    causal_w      = torch.exp(-config.CAUSAL_EPSILON * shifted) # (n_chunks,)

    # Weighted mean
    return (causal_w * chunk_losses).sum() / (causal_w.sum() + 1e-8)


# ====================================================================== #
#  PHASE 5-B :  SOMMERFELD RADIATIVE BCs  (FIX 11: log1p)                  #
# ====================================================================== #
def sommerfeld_radiative_loss(model, sampler, z_latent_det: torch.Tensor,
                              config: type = Config) -> torch.Tensor:
    """
    Orthogonal Sommerfeld absorbing boundaries at the spatial arena edges.

        x = +1 :  (∂_t φ_R + ∂_x φ_R)² = 0
        x = -1 :  (∂_t φ_R − ∂_x φ_R)² = 0
        y = +1 :  (∂_t φ_R + ∂_y φ_R)² = 0
        y = -1 :  (∂_t φ_R − ∂_y φ_R)² = 0

    FIX 11: log1p stabilisation prevents O(10^13) loss domination.
    """
    device = z_latent_det.device
    coords, edge_types = sampler.sample_sommerfeld_boundary(config.BOUNDARY_BATCH)
    coords     = coords.to(device).requires_grad_(True)
    edge_types = edge_types.to(device)

    B = coords.shape[0]
    z_lat = z_latent_det[:1].expand(B, -1)

    phi = model(coords, z_lat)
    g   = _grad(phi, coords)
    dt, dx, dy = g[:, 0:1], g[:, 1:2], g[:, 2:3]

    loss = torch.zeros(1, device=device)

    # x = +1
    m = (edge_types == 0)
    if m.any():
        loss = loss + ((dt[m] + dx[m]) ** 2).mean()
    # x = -1
    m = (edge_types == 1)
    if m.any():
        loss = loss + ((dt[m] - dx[m]) ** 2).mean()
    # y = +1
    m = (edge_types == 2)
    if m.any():
        loss = loss + ((dt[m] + dy[m]) ** 2).mean()
    # y = -1
    m = (edge_types == 3)
    if m.any():
        loss = loss + ((dt[m] - dy[m]) ** 2).mean()

    return torch.log1p(loss).squeeze()                          # FIX 11


# ====================================================================== #
#  PHASE 5-C :  BULK CAUCHY  (t = 0)  (FIX 12: interior only)             #
# ====================================================================== #
def bulk_cauchy_loss(model, sampler, z_latent_det: torch.Tensor,
                     config: type = Config):
    """
    Cauchy Initial Value constraints at t = 0 — BULK INTERIOR ONLY.

    FIX 12: Only enforce for u > U_MIN + 2.0 (deep in AdS bulk where
    vacuum conditions are physically correct). The old code enforced φ=0
    at ALL depths including u ≈ U_BOUNDARY where the boundary data has
    non-zero initial conditions (Gaussian BH bumps). This contradiction
    + the momentum term drove ml to 10^12.

    Returns
    -------
    field_loss    : scalar   (MSE of φ_R² at t = 0, bulk interior)
    momentum_loss : scalar   (MSE of (∂_t φ_R)² at t = 0, bulk interior)
    """
    device = z_latent_det.device
    coords = sampler.sample_cauchy_surface(config.BOUNDARY_BATCH).to(device)
    coords.requires_grad_(True)

    B = coords.shape[0]
    z_lat = z_latent_det[:1].expand(B, -1)

    phi = model(coords, z_lat)
    g   = _grad(phi, coords)

    # FIX 12: Restrict to bulk interior (away from AdS boundary)
    u_vals = coords[:, 3].detach()
    mask = (u_vals > config.U_MIN + 2.0)

    if mask.any():
        field_loss = (phi[mask] ** 2).mean()
        mom_loss   = (g[mask, 0:1] ** 2).mean()
    else:
        field_loss = torch.tensor(0.0, device=device, requires_grad=True)
        mom_loss   = torch.tensor(0.0, device=device, requires_grad=True)

    return field_loss, mom_loss


# ====================================================================== #
#  PHASE 6 :  HRT COVARIANT AREA + LORENTZIAN PENALTY  (FIX 13)           #
# ====================================================================== #
def hrt_covariant_area(phi_renorm: torch.Tensor,
                       coords:     torch.Tensor,
                       config:     type = Config):
    """
    Computes:

    1.  HRT extremal surface area via Differential Swelling.

        Uses conformal-frame energy differences to bypass the 1.0 baseline
        washout in FP32. Returns 1 + excess_growth so that hrt_area >= 1.

    2.  Lorentzian causality penalty with log1p (FIX 13).

        Reconstructs true phi_bulk = phi_R * e^{3u} and rejects superluminal
        surface normals.  log1p prevents O(10^14) penalty domination.

    Parameters
    ----------
    phi_renorm : (B, 1)
    coords     : (B, 4)  with requires_grad=True

    Returns
    -------
    hrt_area          : scalar (>= 1.0)
    causality_penalty : scalar
    """
    u   = coords[:, 3:4]
    e2u = torch.exp(2.0 * u)
    e3u = torch.exp(3.0 * u)

    derivs = compute_derivatives(phi_renorm, coords)

    # --- Conformal-frame energy difference for differential swelling ---
    T_tt_conf = 0.5 * derivs['dt'] ** 2
    T_xx_conf = 0.5 * derivs['dx'] ** 2
    diff_conf = torch.abs(T_tt_conf - T_xx_conf)

    # kappa_eff clamped at KAPPA_MAX
    kappa_eff = torch.clamp(config.KAPPA * diff_conf, max=config.KAPPA_MAX)

    # Differential swelling: (sqrt(1 + kappa) - 1) extracts geometry-sensitive part
    excess_growth = (torch.sqrt(1.0 + kappa_eff + 1e-8) - 1.0).mean()
    hrt_area = 1.0 + excess_growth

    # --- Lorentzian causality penalty using phi_bulk ---
    # Reconstruct true bulk field:  phi_bulk = phi_R * e^{3u}
    dt_b = e3u * derivs['dt']
    dx_b = e3u * derivs['dx']
    dy_b = e3u * derivs['dy']
    du_b = e3u * (derivs['du'] + 3.0 * derivs['phi'])

    # g^{mu,nu}: diag(-e^{2u}, e^{2u}, e^{2u}, 1)
    grad_norm = (-e2u * dt_b ** 2
                 + e2u * dx_b ** 2
                 + e2u * dy_b ** 2
                 + du_b ** 2)

    # FIX 13: log1p prevents O(10^14) penalty
    causality_penalty = torch.log1p(1000.0 * torch.relu(grad_norm).mean())

    return hrt_area, causality_penalty
