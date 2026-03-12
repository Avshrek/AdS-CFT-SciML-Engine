"""
bbh_initial_data.py — Boosted Binary Black Hole Initial Data on v=0
====================================================================

Generates the Cauchy data for two boosted AdS-Schwarzschild black holes
at the initial advanced-time slice v=0.

Physics
-------
Each black hole is an AdS-Schwarzschild solution with mass M located at
transverse position x₀ and boosted with velocity β along the x-axis.
In the null characteristic formulation the initial data is specified
on the v = 0 slice.  We prescribe the metric functions A, Σ, B, V, φ
and their first v-derivatives.

A single AdS-Schwarzschild BH in Poincaré-EF coordinates:
    A_BH(z) = 1 - μ z³       with μ = 2M

The two-BH superposition uses Gaussian energy profiles localized at
x₁ and x₂ with Lorentz-contracted widths σ/γ:

    μ(x) = μ₁(x) + μ₂(x)
    μ_i(x) = 2M_i · exp(-(x - x_i)² / (2 σ_i²))

This is exact to leading order in the post-Newtonian expansion and
becomes a genuine BBH initial configuration once the Einstein solver
finds the self-consistent metric.
"""

import math
import torch
from ads_config import BBHConfig


def _lorentz_gamma(beta: float) -> float:
    """Lorentz factor γ = 1/√(1 - β²)."""
    return 1.0 / math.sqrt(1.0 - beta * beta)


def energy_profile(x: torch.Tensor, mass: float, x0: float,
                   beta: float, width: float) -> torch.Tensor:
    """
    Gaussian energy profile for a single boosted BH.

    Parameters
    ----------
    x     : (N,) transverse coordinate
    mass  : BH mass parameter
    x0    : Center position at v=0
    beta  : Boost velocity
    width : Proper width of the energy profile

    Returns
    -------
    mu : (N,) mass function μ(x) = 2M · G(x; x₀, σ/γ)
    """
    gamma = _lorentz_gamma(beta)
    sigma = width / gamma  # Lorentz-contracted width
    return 2.0 * mass * torch.exp(-0.5 * ((x - x0) / sigma) ** 2)


def compute_initial_metric(coords: torch.Tensor,
                           cfg: type = BBHConfig) -> dict:
    """
    Compute the initial metric functions at v = 0.

    Parameters
    ----------
    coords : (N, 3) tensor with columns [v, x, u]
             v should be ≈ 0 for initial data
    cfg    : BBHConfig class

    Returns
    -------
    dict with keys 'A', 'Sigma', 'B', 'V', 'phi', 'A_dot'
    each (N,) tensors of initial values.

    A_dot = ∂_v A |_{v=0} encodes the initial boost momentum.
    """
    x = coords[:, 1]
    u = coords[:, 2]
    z = torch.exp(u)  # u = ln(z)

    # Total mass function from two BHs
    mu1 = energy_profile(x, cfg.BH_MASS_1, cfg.BH_POSITION_1,
                         cfg.BH_BOOST_1, cfg.BH_WIDTH)
    mu2 = energy_profile(x, cfg.BH_MASS_2, cfg.BH_POSITION_2,
                         cfg.BH_BOOST_2, cfg.BH_WIDTH)
    mu = mu1 + mu2  # (N,)

    # ----- Metric functions at v=0 -----
    # A = 1 - μ(x) z³  (planar AdS-Schwarzschild in Poincaré-EF gauge)
    z3 = z.pow(3)
    A_init = 1.0 - mu * z3

    # Σ = 1 + correction from superposed masses
    # To leading order, Σ deviates at O(z³):
    # Σ = 1 + (1/6) μ(x) z³  (from Einstein constraint on v=0)
    Sigma_init = 1.0 + (1.0 / 6.0) * mu * z3

    # B = 0 at v=0 for head-on collision (no initial GW anisotropy)
    # The anisotropy develops dynamically during the merger
    B_init = torch.zeros_like(x)

    # V_shift encodes the initial momentum from the boost
    # V ~ β · ∂_x μ · z² at leading order
    # Using finite differences for ∂_x μ:
    gamma1 = _lorentz_gamma(cfg.BH_BOOST_1)
    gamma2 = _lorentz_gamma(cfg.BH_BOOST_2)
    sigma1 = cfg.BH_WIDTH / gamma1
    sigma2 = cfg.BH_WIDTH / gamma2

    dmu1_dx = -((x - cfg.BH_POSITION_1) / sigma1**2) * mu1
    dmu2_dx = -((x - cfg.BH_POSITION_2) / sigma2**2) * mu2

    # Total x-momentum density
    p_x = cfg.BH_BOOST_1 * mu1 + cfg.BH_BOOST_2 * mu2
    V_init = p_x * z.pow(2)

    # phi = 0 at v=0 (no initial scalar excitation)
    phi_init = torch.zeros_like(x)

    # Time derivative of A at v=0 (needed for evolution)
    # ∂_v A ~ -2 (β₁ dμ₁/dx + β₂ dμ₂/dx) z³
    A_dot_init = -(cfg.BH_BOOST_1 * dmu1_dx
                   + cfg.BH_BOOST_2 * dmu2_dx) * z3

    return {
        'A': A_init,
        'Sigma': Sigma_init,
        'B': B_init,
        'V': V_init,
        'phi': phi_init,
        'A_dot': A_dot_init,
        'mu': mu,  # mass function for diagnostics
    }


def initial_data_loss(predicted: torch.Tensor,
                      coords: torch.Tensor,
                      cfg: type = BBHConfig) -> torch.Tensor:
    """
    Loss that enforces the v=0 Cauchy data.

    Parameters
    ----------
    predicted : (N, 5) — SIREN output for initial-time points
                columns: [A, Σ, B, V, φ]
    coords    : (N, 3) — coordinates [v, x, u], v ≈ 0

    Returns
    -------
    loss : scalar — MSE between predicted and analytic initial data
    """
    target = compute_initial_metric(coords, cfg)

    loss = (
        (predicted[:, cfg.IDX_A] - target['A']).pow(2).mean()
        + (predicted[:, cfg.IDX_SIGMA] - target['Sigma']).pow(2).mean()
        + (predicted[:, cfg.IDX_B] - target['B']).pow(2).mean()
        + (predicted[:, cfg.IDX_V] - target['V']).pow(2).mean()
        + (predicted[:, cfg.IDX_PHI] - target['phi']).pow(2).mean()
    )

    return loss


def generate_initial_slice(n_x: int = 200, n_u: int = 200,
                           cfg: type = BBHConfig) -> dict:
    """
    Generate a dense initial-time slice for visualization / validation.

    Returns
    -------
    dict with 'x', 'z', 'A', 'Sigma', 'B', 'V', 'phi' 2D arrays.
    """
    x = torch.linspace(cfg.X_RANGE[0], cfg.X_RANGE[1], n_x)
    u = torch.linspace(cfg.U_MIN, cfg.U_MAX, n_u)
    xx, uu = torch.meshgrid(x, u, indexing='ij')

    coords = torch.stack([
        torch.zeros_like(xx.flatten()),  # v = 0
        xx.flatten(),
        uu.flatten(),
    ], dim=-1)

    fields = compute_initial_metric(coords, cfg)

    return {
        'x': xx.numpy(),
        'z': torch.exp(uu).numpy(),
        'A': fields['A'].reshape(n_x, n_u).detach().numpy(),
        'Sigma': fields['Sigma'].reshape(n_x, n_u).detach().numpy(),
        'B': fields['B'].reshape(n_x, n_u).detach().numpy(),
        'V': fields['V'].reshape(n_x, n_u).detach().numpy(),
        'phi': fields['phi'].reshape(n_x, n_u).detach().numpy(),
        'mu': fields['mu'].reshape(n_x, n_u).detach().numpy(),
    }
