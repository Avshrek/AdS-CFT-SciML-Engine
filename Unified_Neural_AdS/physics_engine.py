"""
physics_engine.py — PINO PDE Residual Loss for the Unified Neural-AdS Pipeline
================================================================================

AdS/CFT Physics Context
------------------------
In the AdS/CFT (Anti-de Sitter / Conformal Field Theory) correspondence, the
bulk spacetime geometry is governed by field equations that reduce to the Laplace
equation ∇²φ = 0 in the static, linearised regime.  A Physics-Informed Neural
Operator (PINO) augments the pure data-fitting loss with a PDE residual term so
the learned operator *inherently* respects the governing equation:

    L_total = L_data  +  λ(epoch) · L_PDE

where L_PDE = MSE(∇²u_pred) over the interior of the bulk grid.

Implementation
--------------
* LaplacianLoss  — computes L_PDE via a fixed 2D finite-difference convolution
                   kernel (no autograd required).
* PINOScheduler — smoothly ramps λ from 0 → λ_target over a configurable number
                   of warm-up epochs so the PDE loss doesn't explode gradients
                   when predictions are still far from physical.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LaplacianLoss — PDE residual via fixed conv2d stencil
# ---------------------------------------------------------------------------

class LaplacianLoss(nn.Module):
    """Compute the Laplace-equation residual ∇²u ≈ 0 on a 2-D grid.

    Uses the standard 5-point finite-difference stencil::

        [[0,  1,  0],
         [1, -4,  1],
         [0,  1,  0]]

    applied as a *non-trainable* convolution so the computation is fully
    vectorised on the GPU and incurs zero autograd overhead for the stencil
    weights.

    Parameters
    ----------
    dx : float, optional
        Grid spacing (assumed uniform in both directions).  The residual is
        normalised by 1/dx² so that the loss magnitude is independent of
        resolution.  Default ``1.0`` (dimensionless grid).

    Returns
    -------
    loss : torch.Tensor (scalar)
        Mean-Squared PDE residual over the *interior* bulk points (boundary
        pixels are excluded via a 1-pixel crop after convolution).
    """

    def __init__(self, dx: float = 1.0) -> None:
        super().__init__()

        # 5-point Laplacian stencil  —  shape (1, 1, 3, 3)
        kernel = torch.tensor(
            [[0.0,  1.0,  0.0],
             [1.0, -4.0,  1.0],
             [0.0,  1.0,  0.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / (dx ** 2)

        # register_buffer → moves with .to(device), saved in state_dict,
        # but never updated by the optimiser.
        self.register_buffer("kernel", kernel)

    def forward(self, u_pred: torch.Tensor) -> torch.Tensor:
        """Evaluate L_PDE = MSE(∇²u_pred) on interior grid points.

        Parameters
        ----------
        u_pred : Tensor of shape ``(B, 1, H, W)``
            Predicted bulk scalar field.

        Returns
        -------
        Scalar tensor (mean squared Laplacian residual).
        """
        # Apply the stencil with padding=1 so output keeps (H, W)
        laplacian = F.conv2d(u_pred, self.kernel, padding=1)

        # Crop the 1-pixel boundary ring — boundary values are prescribed by
        # Dirichlet conditions and must not contribute to the PDE residual.
        interior = laplacian[:, :, 1:-1, 1:-1]

        return torch.mean(interior ** 2)


# ---------------------------------------------------------------------------
# PINOScheduler — smooth λ warm-up
# ---------------------------------------------------------------------------

class PINOScheduler:
    r"""Ramp the PDE-loss weight λ from an initial value to a target value.

    The default strategy is a **linear** ramp:

    .. math::

        \lambda(e) = \lambda_{\text{start}}
        + (\lambda_{\text{target}} - \lambda_{\text{start}})
          \cdot \min\!\bigl(e / N_{\text{ramp}},\; 1\bigr)

    An optional ``strategy='cosine'`` uses a half-cosine curve for a gentler
    onset and sharper saturation.

    Parameters
    ----------
    lambda_target : float
        Asymptotic value of λ after warm-up.
    ramp_epochs : int
        Number of epochs over which to ramp.
    lambda_start : float, optional
        Starting value of λ (default ``0.0``).
    strategy : str, optional
        ``'linear'`` (default) or ``'cosine'``.
    """

    def __init__(
        self,
        lambda_target: float = 1.0,
        ramp_epochs: int = 50,
        lambda_start: float = 0.0,
        strategy: str = "linear",
    ) -> None:
        if strategy not in ("linear", "cosine"):
            raise ValueError(f"Unknown strategy '{strategy}'; use 'linear' or 'cosine'.")
        self.lambda_start = lambda_start
        self.lambda_target = lambda_target
        self.ramp_epochs = max(ramp_epochs, 1)
        self.strategy = strategy

    def get_lambda(self, epoch: int) -> float:
        """Return the λ value for a given *epoch* (0-indexed)."""
        t = min(epoch / self.ramp_epochs, 1.0)

        if self.strategy == "cosine":
            # Smooth half-cosine: slow start → fast middle → gentle landing
            t = 0.5 * (1.0 - math.cos(math.pi * t))

        return self.lambda_start + (self.lambda_target - self.lambda_start) * t

    def __repr__(self) -> str:
        return (
            f"PINOScheduler(target={self.lambda_target}, "
            f"ramp={self.ramp_epochs}, strategy='{self.strategy}')"
        )
