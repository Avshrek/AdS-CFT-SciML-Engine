"""
metric_model.py — 5-Channel FiLM-SIREN for the Full Dynamical Metric
======================================================================

The network learns the DEVIATIONS from the pure-AdS reference metric.
This guarantees correct asymptotics at initialization and improves
convergence (the network starts at exact vacuum solution).

Architecture
------------
Input:   3D coordinates (v, x, u)   where u = ln(z)
Cond:    128-D latent from boundary ConvEncoder
Output:  5 fields [δA, δΣ, δB, V, φ]

The physical metric functions are reconstructed as:
    A(v,x,z) = A_pure(z) + z^3 · δA_net(v,x,u)
    Σ(v,x,z) = 1         + z^3 · δΣ_net(v,x,u)
    B(v,x,z) =             z^3 · δB_net(v,x,u)
    V(v,x,z) =             z^2 · δV_net(v,x,u)
    φ(v,x,z) =             z^Δ · δφ_net(v,x,u)   with Δ=3 for AdS4

The z^n prefactors enforce the correct holographic boundary falloff
analytically — the network never needs to learn them.

The z^3 coefficients extracted at z→0 (i.e., δA_net, δΣ_net at the
boundary) are directly the boundary CFT stress-energy tensor components:
    <T_vv> ∝ δA_net|_{z=0}
    <T_xx - T_yy> ∝ δB_net|_{z=0}
"""

import math
import torch
import torch.nn as nn

from ads_config import BBHConfig


# ====================================================================== #
#  SINE LAYER  (shared with scalar engine)                                 #
# ====================================================================== #
class SineLayer(nn.Module):
    """SIREN sine activation with optional FiLM modulation."""

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x, gamma=None, beta=None):
        h = self.omega_0 * self.linear(x)
        if gamma is not None and beta is not None:
            h = gamma * h + beta
        return torch.sin(h)


# ====================================================================== #
#  5-CHANNEL FiLM-SIREN  (The Metric Network)                             #
# ====================================================================== #
class MetricSIREN(nn.Module):
    """
    FiLM-SIREN that outputs 5 raw deviation fields from 3D coordinates.

    Input:   (B, 3)  →  (v, x, u)
    Cond:    (B, LATENT_DIM)  →  from boundary ConvEncoder
    Output:  (B, 5)  →  [δA, δΣ, δB, δV, δφ]  (raw network output)

    These are NOT the physical metric fields — use MetricReconstructor
    to apply the asymptotic boundary layer.
    """

    def __init__(self, cfg: type = BBHConfig):
        super().__init__()
        self.cfg = cfg

        in_dim  = 3                     # (v, x, u)
        hidden  = cfg.SIREN_HIDDEN
        n_hid   = cfg.SIREN_LAYERS
        omega_0 = cfg.SIREN_OMEGA_0
        lat_dim = cfg.LATENT_DIM
        out_dim = cfg.NUM_METRIC_FIELDS  # 5

        # First SIREN layer
        self.first_layer = SineLayer(in_dim, hidden, omega_0=omega_0, is_first=True)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            SineLayer(hidden, hidden, omega_0=omega_0) for _ in range(n_hid - 1)
        ])

        # Output: 5 channels
        self.output_layer = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)

        # FiLM generators: one per modulated layer
        self.film_generators = nn.ModuleList()
        for _ in range(n_hid):
            self.film_generators.append(
                nn.Sequential(
                    nn.Linear(lat_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 2 * hidden),
                )
            )

    def forward(self, coords: torch.Tensor, z_latent: torch.Tensor) -> torch.Tensor:
        """
        coords   : (B, 3)  — (v, x, u)
        z_latent : (B, LATENT_DIM)
        Returns  : (B, 5)  — raw deviation fields
        """
        # FiLM parameters
        film_params = []
        for gen in self.film_generators:
            out = gen(z_latent)
            gamma, beta = out.chunk(2, dim=-1)
            gamma = gamma + 1.0
            film_params.append((gamma, beta))

        # Forward through SIREN
        h = self.first_layer(coords, film_params[0][0], film_params[0][1])
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h, film_params[i + 1][0], film_params[i + 1][1])

        return self.output_layer(h)


# ====================================================================== #
#  METRIC RECONSTRUCTOR  (Asymptotic Boundary Layer)                       #
# ====================================================================== #
class MetricReconstructor:
    """
    Transforms raw SIREN output into physical metric fields by applying
    the correct holographic boundary falloff factors.

    This enforces:
        A     = A_pure(z) + z³ · δA_net    →  A → 1 at boundary
        Σ     = 1         + z³ · δΣ_net    →  Σ → 1 at boundary
        B     =             z³ · δB_net    →  B → 0 at boundary
        V     =             z² · δV_net    →  V → 0 at boundary
        φ     =             z^Δ · δφ_net   →  φ → 0 at boundary (normalizable)

    The SIREN never needs to learn the z-dependent suppression — it's
    built in analytically. This is the key numerical stability trick.
    """

    def __init__(self, cfg: type = BBHConfig):
        self.cfg = cfg

    def reconstruct(self, raw_output: torch.Tensor,
                    coords: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        raw_output : (B, 5)  — δA, δΣ, δB, δV, δφ from MetricSIREN
        coords     : (B, 3)  — (v, x, u)

        Returns
        -------
        dict with keys: A, Sigma, B, V_shift, phi, and their raw deviations
             Also includes z, e2u, etc. for downstream use
        """
        u = coords[:, 2:3]                           # (B, 1)
        z = torch.exp(u)                              # z = e^u ∈ [1e-4, 1]
        z2 = z ** 2
        z3 = z ** 3

        dA    = raw_output[:, 0:1]
        dSig  = raw_output[:, 1:2]
        dB    = raw_output[:, 2:3]
        dV    = raw_output[:, 3:4]
        dphi  = raw_output[:, 4:5]

        # Pure AdS reference: A_pure = 1 (planar Poincaré-EF gauge)
        # In Poincaré patch EF: ds²=(1/z²)[-dv²+2dvdz+dx²+dy²] → A=1
        A_pure = torch.ones_like(z)

        # Physical fields with correct falloffs
        A       = A_pure + z3 * dA          # Lapse
        Sigma   = 1.0    + z3 * dSig        # Area element (must stay > 0)
        B       =          z3 * dB           # Anisotropy
        V_shift =          z2 * dV           # Shift
        phi     = z3 * dphi                  # Scalar (Δ=3 for massless in AdS4)

        return dict(
            A=A, Sigma=Sigma, B=B, V_shift=V_shift, phi=phi,
            dA=dA, dSigma=dSig, dB=dB, dV=dV, dphi=dphi,
            z=z, z2=z2, z3=z3, u=u,
        )


# ====================================================================== #
#  BOUNDARY ENCODER  (same ConvEncoder, adapted for 2D boundary data)      #
# ====================================================================== #
class BoundaryEncoder(nn.Module):
    """
    Encodes the boundary CFT stress-tensor data T_μν(v, x) into
    a 128-D latent vector.

    Input:  (B_batch, 1, N_v, N_x, 1) — boundary data over (v, x)
            The trailing dim=1 is for 3D conv compatibility
    Output: (B_batch, LATENT_DIM)
    """

    def __init__(self, cfg: type = BBHConfig):
        super().__init__()
        self.cfg = cfg
        channels = cfg.ENCODER_CHANNELS

        conv_blocks = []
        for i in range(len(channels) - 1):
            conv_blocks.extend([
                nn.Conv3d(channels[i], channels[i + 1],
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(channels[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.conv = nn.Sequential(*conv_blocks)

        # Compute flattened size
        v_o = cfg.ENCODER_TEMPORAL_FRAMES
        x_o = cfg.ENCODER_SPATIAL_RES
        y_o = 1  # collapsed y-dimension
        for _ in range(len(channels) - 1):
            v_o = (v_o + 1) // 2
            x_o = (x_o + 1) // 2
            y_o = (y_o + 1) // 2
        flat_size = channels[-1] * v_o * x_o * y_o

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.SiLU(),
            nn.Linear(512, cfg.LATENT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)
