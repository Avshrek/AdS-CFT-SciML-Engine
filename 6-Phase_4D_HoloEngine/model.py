"""
model.py - Neural Representation  (Phase 2 Implementation)
============================================================

ConvEncoder3D
    Ingests the 2D collision-fluid temporal sequence -> 128-D Quantum Latent Vector.

FiLMSiren
    Takes 4-D spacetime coordinates (t, x, y, u) + 128-D latent (via Feature-wise
    Linear Modulation) -> phi_renorm  (the bounded, finite renormalised scalar mode).

Key architectural safety:
    * The CNN encoder is ONLY optimised by the Boundary Anchor Data Loss.
    * For all PDE / HRT / energy computations the latent is passed through
      z_latent.detach()  to sever the autograd graph and prevent VRAM explosion
      and "empty-universe" cheating.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from config import Config


# ========================================================================== #
#                        SIREN SINE LAYER                                     #
# ========================================================================== #
class SineLayer(nn.Module):
    """Single SIREN layer with optional FiLM modulation.

    y = sin(gamma * (omega_0 * Linear(x)) + beta)

    When gamma, beta are not supplied, defaults to standard SIREN.
    """

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

    def forward(self, x: torch.Tensor,
                gamma: torch.Tensor = None,
                beta:  torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : (B, in)
        gamma : (B, out) - FiLM multiplicative modulation
        beta  : (B, out) - FiLM additive modulation
        """
        h = self.omega_0 * self.linear(x)
        if gamma is not None and beta is not None:
            h = gamma * h + beta
        return torch.sin(h)


# ========================================================================== #
#                        FiLM-SIREN NETWORK                                   #
# ========================================================================== #
class FiLMSiren(nn.Module):
    """
    FiLM-modulated SIREN.

    Input  : coords (B, 4)  ->  (t, x, y, u)
    Cond.  : z_latent (B, LATENT_DIM)
    Output : phi_renorm (B, 1)
    """

    def __init__(self, config: type = Config):
        super().__init__()
        self.config = config

        in_dim  = 4                 # (t, x, y, u)
        hidden  = config.SIREN_HIDDEN
        n_hid   = config.SIREN_LAYERS
        omega_0 = config.SIREN_OMEGA_0
        lat_dim = config.LATENT_DIM

        # First SIREN layer
        self.first_layer = SineLayer(in_dim, hidden, omega_0=omega_0, is_first=True)

        # Hidden SIREN layers
        self.hidden_layers = nn.ModuleList([
            SineLayer(hidden, hidden, omega_0=omega_0) for _ in range(n_hid - 1)
        ])

        # Final linear -> scalar
        self.output_layer = nn.Linear(hidden, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)

        # --- FiLM generators: one per modulated layer (first + hidden) ---
        self.film_generators = nn.ModuleList()
        for _ in range(n_hid):
            self.film_generators.append(
                nn.Sequential(
                    nn.Linear(lat_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 2 * hidden),         # [gamma | beta]
                )
            )

    def forward(self, coords: torch.Tensor, z_latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords   : (B, 4)
        z_latent : (B, LATENT_DIM)

        Returns
        -------
        phi_renorm : (B, 1)
        """
        # Pre-compute FiLM params for every layer
        film_params = []
        for gen in self.film_generators:
            out = gen(z_latent)                         # (B, 2*H)
            gamma, beta = out.chunk(2, dim=-1)          # (B, H) each
            gamma = gamma + 1.0                         # centre around identity
            film_params.append((gamma, beta))

        # Forward through SIREN
        h = self.first_layer(coords, film_params[0][0], film_params[0][1])
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h, film_params[i + 1][0], film_params[i + 1][1])

        return self.output_layer(h)                     # (B, 1)


# ========================================================================== #
#                   3-D CONVOLUTIONAL ENCODER                                 #
# ========================================================================== #
class ConvEncoder3D(nn.Module):
    """
    Encodes the temporal 2-D collision-fluid sequence into a 128-D Quantum
    Latent Vector.

    Input  : (B, 1, T, H, W)
    Output : (B, LATENT_DIM)
    """

    def __init__(self, config: type = Config):
        super().__init__()
        self.config = config
        channels = config.ENCODER_CHANNELS              # e.g. [1, 16, 32, 64]

        conv_blocks = []
        for i in range(len(channels) - 1):
            conv_blocks.extend([
                nn.Conv3d(channels[i], channels[i + 1],
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(channels[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.conv = nn.Sequential(*conv_blocks)

        # Compute flattened feature size after conv layers
        t_o = config.ENCODER_TEMPORAL_FRAMES
        h_o = config.ENCODER_SPATIAL_RES
        w_o = config.ENCODER_SPATIAL_RES
        for _ in range(len(channels) - 1):
            t_o = (t_o + 1) // 2
            h_o = (h_o + 1) // 2
            w_o = (w_o + 1) // 2
        flat_size = channels[-1] * t_o * h_o * w_o

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.SiLU(),
            nn.Linear(512, config.LATENT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, T, H, W)
        Returns z_latent : (B, LATENT_DIM)
        """
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)
