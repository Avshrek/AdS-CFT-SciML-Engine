"""
sampler.py - ApexDualSampler  (Phase 1 Implementation)
========================================================
Dual-Sampler Architecture that resolves the Dirichlet Interpolation Disconnect:

1. **Boundary Sampler** – yields exact (t_i, x_i, y_i) coordinates matching
   the discrete 2D collision fluid dataset, pinned at u = U_BOUNDARY = ln(1e-4).
   Feeds into the Boundary Anchor Data Loss.

2. **Bulk Collocation Sampler** – generates random, continuous (t, x, y, u) points
   for PDE / Energy / HRT evaluation.  Weighted by the true AdS volume determinant
   1/z^3 = e^{-3u}.

3. **Cauchy Sampler** – bulk points at t = 0 for the initial-value constraint.

4. **Sommerfeld Sampler** – edge points at x = +/-1, y = +/-1 for radiative BCs.

Supports both:
  - Pre-computed boundary grids (synthetic data)
  - Direct coordinate tensors (master dataset .npz)
"""

import torch
from config import Config


class ApexDualSampler:
    """
    Flexible dual sampler that accepts either:
      (a) boundary_coords + boundary_values tensors  (from master dataset)
      (b) a (T, H, W) boundary_data volume            (from synthetic generator)
    """

    def __init__(self, boundary_coords: torch.Tensor,
                 boundary_values: torch.Tensor,
                 config: type = Config):
        """
        Parameters
        ----------
        boundary_coords : Tensor (N, 4)
            Pre-built boundary coordinate table [t, x, y, u].
        boundary_values : Tensor (N,)
            Corresponding scalar field values.
        config : Config class with all hyperparameters.
        """
        self.cfg = config
        self.boundary_coords = boundary_coords
        self.boundary_values = boundary_values

    @classmethod
    def from_volume(cls, boundary_data: torch.Tensor, config: type = Config):
        """Construct from a (T, H, W) collision-fluid volume."""
        n_time, H, W = boundary_data.shape

        t_vals = torch.linspace(config.T_RANGE[0], config.T_RANGE[1], n_time)
        x_vals = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], W)
        y_vals = torch.linspace(config.Y_RANGE[0], config.Y_RANGE[1], H)

        tt, yy, xx = torch.meshgrid(t_vals, y_vals, x_vals, indexing="ij")

        coords = torch.stack(
            [
                tt.flatten(),
                xx.flatten(),
                yy.flatten(),
                torch.full_like(tt.flatten(), config.U_BOUNDARY),
            ],
            dim=-1,
        )
        values = boundary_data.flatten()
        return cls(coords, values, config)

    # --------------------------------------------------------------------- #
    #  1.  Boundary Sampler
    # --------------------------------------------------------------------- #
    def sample_discrete_boundary(self, batch_size: int):
        """
        Returns
        -------
        coords : (B, 4)   exact (t, x, y, u=U_BOUNDARY) from the dataset grid.
        values : (B,)      corresponding scalar-field values.
        """
        total = self.boundary_coords.shape[0]
        idx = torch.randint(0, total, (batch_size,))
        return self.boundary_coords[idx], self.boundary_values[idx]

    # --------------------------------------------------------------------- #
    #  2.  Bulk Collocation Sampler
    # --------------------------------------------------------------------- #
    def sample_continuous_bulk(self, batch_size: int):
        """
        Returns
        -------
        coords  : (B, 4)  random (t, x, y, u).
        weights : (B,)    AdS volume measure  e^{-3u}.
        """
        t = torch.rand(batch_size) * (self.cfg.T_RANGE[1] - self.cfg.T_RANGE[0]) \
            + self.cfg.T_RANGE[0]
        x = torch.rand(batch_size) * (self.cfg.X_RANGE[1] - self.cfg.X_RANGE[0]) \
            + self.cfg.X_RANGE[0]
        y = torch.rand(batch_size) * (self.cfg.Y_RANGE[1] - self.cfg.Y_RANGE[0]) \
            + self.cfg.Y_RANGE[0]
        u = torch.rand(batch_size) * (self.cfg.U_MAX - self.cfg.U_MIN) \
            + self.cfg.U_MIN

        coords  = torch.stack([t, x, y, u], dim=-1)
        weights = torch.exp(-3.0 * u)          # 1/z^3 * Jacobian(z->u) = e^{-3u}
        return coords, weights

    # --------------------------------------------------------------------- #
    #  3.  Cauchy Surface Sampler  (t = 0)
    # --------------------------------------------------------------------- #
    def sample_cauchy_surface(self, batch_size: int):
        """Random bulk points with t strictly = 0."""
        x = torch.rand(batch_size) * (self.cfg.X_RANGE[1] - self.cfg.X_RANGE[0]) \
            + self.cfg.X_RANGE[0]
        y = torch.rand(batch_size) * (self.cfg.Y_RANGE[1] - self.cfg.Y_RANGE[0]) \
            + self.cfg.Y_RANGE[0]
        u = torch.rand(batch_size) * (self.cfg.U_MAX - self.cfg.U_MIN) \
            + self.cfg.U_MIN
        t = torch.zeros(batch_size)
        return torch.stack([t, x, y, u], dim=-1)

    # --------------------------------------------------------------------- #
    #  4.  Sommerfeld Boundary Sampler  (x = +/-1  or  y = +/-1)
    # --------------------------------------------------------------------- #
    def sample_sommerfeld_boundary(self, batch_size: int):
        """
        Returns
        -------
        coords     : (B, 4)
        edge_types : (B,)  int tensor
            0 -> x = +1,  1 -> x = -1,  2 -> y = +1,  3 -> y = -1
        """
        n = batch_size // 4                     # per edge
        remainder = batch_size - 4 * n

        t_all = torch.rand(batch_size) * (self.cfg.T_RANGE[1] - self.cfg.T_RANGE[0]) \
                + self.cfg.T_RANGE[0]
        u_all = torch.rand(batch_size) * (self.cfg.U_MAX - self.cfg.U_MIN) \
                + self.cfg.U_MIN

        coords_list, edge_list = [], []

        for i, (fixed_val, dim_name) in enumerate(
            [(1.0, "x+"), (-1.0, "x-"), (1.0, "y+"), (-1.0, "y-")]
        ):
            lo = i * n
            hi = lo + n + (remainder if i == 3 else 0)
            t_e = t_all[lo:hi]
            u_e = u_all[lo:hi]
            m   = t_e.shape[0]

            if dim_name.startswith("x"):
                x_e = torch.full((m,), fixed_val)
                y_e = torch.rand(m) * 2.0 - 1.0
            else:
                x_e = torch.rand(m) * 2.0 - 1.0
                y_e = torch.full((m,), fixed_val)

            coords_list.append(torch.stack([t_e, x_e, y_e, u_e], dim=-1))
            edge_list.append(torch.full((m,), i, dtype=torch.long))

        return torch.cat(coords_list, dim=0), torch.cat(edge_list, dim=0)
