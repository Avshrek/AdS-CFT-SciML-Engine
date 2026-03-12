"""
siren_decoder.py -- SIREN Implicit Neural Representation Decoder
================================================================

Replaces the ConvTranspose3d decoder in HybridQuantumAdS with a
Sinusoidal Representation Network (SIREN) that naturally captures
high-frequency gravitational wave geometry.

The SIREN decoder mirrors HKLL bulk reconstruction:
    Input:  [10 quantum latent variables] + [x, y, z coordinates] = 13-dim
    Output: [scalar gravity amplitude at that exact coordinate]   = 1-dim

Reference: Sitzmann et al., "Implicit Neural Representations with
           Periodic Activation Functions" (NeurIPS 2020)
"""

import math
import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# SIREN LAYER  (with mathematically correct initialization)
# =====================================================================

class SirenLayer(nn.Module):
    """
    Single SIREN layer:  y = sin(omega_0 * (Wx + b))

    Weight initialization follows Sitzmann et al. 2020:
      - First layer:  W ~ U(-1/fan_in, +1/fan_in)
      - Hidden layers: W ~ U(-sqrt(6/fan_in)/omega_0, +sqrt(6/fan_in)/omega_0)
    """

    def __init__(self, in_features, out_features, omega_0=30.0,
                 is_first=False, is_last=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)

        # SIREN-specific initialization
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        if self.is_last:
            # Final layer: no sine, raw linear output for amplitude
            return self.linear(x)
        return torch.sin(self.omega_0 * self.linear(x))


# =====================================================================
# SIREN DECODER
# =====================================================================

class SirenDecoder(nn.Module):
    """
    SIREN-based implicit decoder for holographic bulk reconstruction.

    Maps (latent_z, x, y, z) -> scalar amplitude Phi.

    Architecture:
        13 -> 256 -> 256 -> 256 -> 256 -> 1

    Parameters
    ----------
    latent_dim : int
        Dimension of the quantum latent vector (10).
    coord_dim : int
        Number of spatial coordinates (3: x, y, z).
    hidden_dim : int
        Width of hidden SIREN layers.
    n_layers : int
        Number of hidden layers.
    omega_0 : float
        Frequency parameter for SIREN activations. Higher = more
        high-frequency detail. 30.0 is the standard from the paper.
    """

    def __init__(self, latent_dim=10, coord_dim=3, hidden_dim=256,
                 n_layers=4, omega_0=30.0):
        super().__init__()

        in_dim = latent_dim + coord_dim  # 13

        layers = []

        # First SIREN layer (special initialization)
        self.first_layer = SirenLayer(in_dim, hidden_dim, omega_0=omega_0,
                                      is_first=True)

        # Hidden SIREN layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.hidden_layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))

        # Final layer (linear output, no sine)
        self.final_layer = SirenLayer(hidden_dim, 1, omega_0=omega_0,
                                      is_last=True)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (N, 13)
            Concatenation of [latent_z (10), coords (3)].

        Returns
        -------
        Tensor, shape (N, 1)
            Scalar amplitude at each query point.
        """
        # x is [latent_z (10), z_depth, y, x]
        
        # Hyperbolic Positional Encoding (HPE)
        # We multiply the raw coordinates by the holographic depth `z` 
        # BEFORE they hit the first sine wave, stretching frequencies in deep bulk.
        
        z_depth = x[:, 10:11] 
        x_scaled = x.clone()
        # The True AdS Hyperbolic Positional Encoding: divide by z
        # High frequency at the boundary (z->0), smooth in the deep bulk (z->1)
        x_scaled[:, 11:] = x_scaled[:, 11:] / z_depth

        out = self.first_layer(x_scaled)
        for layer in self.hidden_layers:
            out = layer(out)
        return self.final_layer(out)


# =====================================================================
# COORDINATE GRID GENERATOR (GPU-efficient)
# =====================================================================

def make_coord_grid(depth=20, height=64, width=64, device="cpu", use_hyperbolic=True):
    """
    Generate a 3D coordinate grid. If `use_hyperbolic` is True,
    the depth coordinate (z) is transformed to represent the AdS metric,
    stretching wavelengths in the deep bulk to prevent Fourier artifacts.

    Returns
    -------
    coords : Tensor, shape (D*H*W, 3)
    """
    # AdS Holographic Depth: z_min=1e-4 is boundary (UV cutoff), z > 0 is deep bulk.
    if use_hyperbolic:
        z_vals = torch.linspace(1e-4, 1.0, depth, device=device)
    else:
        # Standard linear [-1, 1] Cartesian
        z_vals = torch.linspace(-1, 1, depth, device=device)

    h = torch.linspace(-1, 1, height, device=device)
    w = torch.linspace(-1, 1, width,  device=device)

    # Meshgrid: indexing='ij' gives (D, H, W) ordering
    gd, gh, gw = torch.meshgrid(z_vals, h, w, indexing="ij")

    # Stack and flatten to (D*H*W, 3)
    coords = torch.stack([gd, gh, gw], dim=-1).reshape(-1, 3)

    return coords


# =====================================================================
# UPDATED HYBRID QUANTUM MODEL (Encoder + Quantum + SIREN Decoder)
# =====================================================================

class HybridQuantumAdS_SIREN(nn.Module):
    """
    Hybrid Quantum-Classical model with SIREN implicit decoder.

    Architecture:
        1. Encoder:  [B,1,20,64,64] -> [B,10]  (Conv3d compression)
        2. Quantum:  [B,10] -> [B,10]           (10-qubit entanglement)
        3. SIREN:    [B*D*H*W, 13] -> [B*D*H*W, 1] -> [B,1,20,64,64]

    The SIREN decoder queries every (x,y,z) point in the bulk volume,
    conditioned on the quantum latent vector, producing the scalar
    field amplitude at each coordinate. This mirrors the HKLL bulk
    reconstruction operator in AdS/CFT.
    """

    def __init__(self, in_channels=1, hidden_dim=256, n_siren_layers=4,
                 omega_0=30.0):
        super().__init__()

        # Volume dimensions (fixed for this dataset)
        self.D, self.H, self.W = 20, 64, 64

        # ---------------------------------------------------------
        # 1. ENCODER (IDENTICAL to original -- DO NOT MODIFY)
        # ---------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Tanh(),
        )

        # ---------------------------------------------------------
        # 2. QUANTUM BOTTLENECK (IDENTICAL -- DO NOT MODIFY)
        #    Imported at runtime to avoid circular dependency
        # ---------------------------------------------------------
        self.quantum_layer = None  # set externally or in subclass

        # ---------------------------------------------------------
        # 3. SIREN IMPLICIT DECODER (replaces ConvTranspose3d)
        # ---------------------------------------------------------
        self.siren_decoder = SirenDecoder(
            latent_dim=10,
            coord_dim=3,
            hidden_dim=hidden_dim,
            n_layers=n_siren_layers,
            omega_0=omega_0,
        )

        # Pre-compute and cache the coordinate grid
        self._cached_coords = None
        self._cached_device = None

    def _get_coords(self, device):
        """Get or create cached coordinate grid on the correct device."""
        if self._cached_coords is None or self._cached_device != device:
            self._cached_coords = make_coord_grid(
                self.D, self.H, self.W, device=device
            )
            self._cached_device = device
        return self._cached_coords

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        N = self.D * self.H * self.W  # 20 * 64 * 64 = 81920

        # Step 1: Encode boundary -> 10-dim latent
        latent_z = self.encoder(x)                          # [B, 10]

        # Step 2: Quantum entanglement mapping
        latent_z = self.quantum_layer(latent_z)              # [B, 10]

        # Step 3: Get coordinate grid (cached, GPU-resident)
        coords = self._get_coords(device)                    # [N, 3]

        # Step 4: Expand latent to match every coordinate query
        #   latent_z: [B, 10] -> [B, 1, 10] -> [B, N, 10]
        #   coords:   [N, 3]  -> [1, N, 3]  -> [B, N, 3]
        latent_expanded = latent_z.unsqueeze(1).expand(B, N, 10)
        coords_expanded = coords.unsqueeze(0).expand(B, N, 3)

        # Step 5: Concatenate [quantum_latent, coordinates] -> [B, N, 13]
        queries = torch.cat([latent_expanded, coords_expanded], dim=-1)

        # Step 6: Flatten batch for SIREN -> [B*N, 13]
        queries_flat = queries.reshape(B * N, 13)

        # Step 7: SIREN forward pass -> [B*N, 1]
        phi_flat = self.siren_decoder(queries_flat)

        # Step 8: Reshape to volume -> [B, 1, D, H, W]
        phi = phi_flat.reshape(B, 1, self.D, self.H, self.W)

        return phi


# =====================================================================
# QUICK TEST
# =====================================================================

if __name__ == "__main__":
    print("Testing SIREN Decoder components...\n")

    # Test SirenDecoder standalone
    decoder = SirenDecoder(latent_dim=10, coord_dim=3, hidden_dim=256, n_layers=4)
    n = sum(p.numel() for p in decoder.parameters())
    print(f"SirenDecoder parameters: {n:,}")

    dummy_input = torch.randn(100, 13)
    out = decoder(dummy_input)
    print(f"SirenDecoder: {list(dummy_input.shape)} -> {list(out.shape)}")

    # Test coordinate grid
    coords = make_coord_grid(20, 64, 64)
    print(f"\nCoordinate grid: {list(coords.shape)}")
    print(f"  Range: [{coords.min():.1f}, {coords.max():.1f}]")

    # Test full model (with dummy quantum layer)
    print("\nTesting HybridQuantumAdS_SIREN...")

    class DummyQuantum(nn.Module):
        def forward(self, x): return torch.tanh(x)

    model = HybridQuantumAdS_SIREN(hidden_dim=256, n_siren_layers=4)
    model.quantum_layer = DummyQuantum()

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {n_total:,}")

    dummy_vol = torch.randn(2, 1, 20, 64, 64)
    output = model(dummy_vol)
    print(f"Forward pass: {list(dummy_vol.shape)} -> {list(output.shape)}")

    # Verify output range
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("\nAll tests passed!")
