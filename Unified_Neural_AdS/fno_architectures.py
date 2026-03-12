"""
fno_architectures.py — Fourier Neural Operator Cores for Unified Neural-AdS
=============================================================================

AdS/CFT Physics Context
------------------------
The Fourier Neural Operator (FNO) learns mappings between *function spaces* by
parameterising the integral kernel in Fourier space.  This is a natural fit for
AdS/CFT holography because:

  1. The boundary CFT data is naturally decomposed into Fourier modes.
  2. Each retained Fourier mode acts as a "frequency channel" that encodes how
     a particular harmonic of the boundary condition propagates into the bulk.
  3. Truncating to the first ``modes`` frequencies is the *spectral* analogue of
     the UV/IR connection in holography — high-frequency (UV) boundary data
     decays exponentially into the bulk (IR).

Provided architectures
----------------------
* ``SpectralConv2d`` / ``FNO2d``  — 2-D (static bulk slice, Phase 1).
* ``SpectralConv3d`` / ``FNO3d``  — 3-D (2+1 D spatiotemporal bulk, Phase 2).

Coordinate-Channel Convention
-----------------------------
+----------+------------+-----------------------------------------------+
| Model    | in_channels| Channel semantics                              |
+==========+============+===============================================+
| FNO2d    | 2          | ch-0:  propagated boundary wave (tiled to 2-D) |
|          |            | ch-1:  spatial Y-depth coordinate grid          |
+----------+------------+-----------------------------------------------+
| FNO3d    | 3          | ch-0:  boundary wave (tiled to 3-D)             |
|          |            | ch-1:  temporal coordinate grid                 |
|          |            | ch-2:  spatial depth coordinate grid            |
+----------+------------+-----------------------------------------------+
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2-D  COMPONENTS  (Phase 1 — static bulk slice)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SpectralConv2d(nn.Module):
    """Spectral convolution in 2-D Fourier space.

    Retains the lowest ``(modes1, modes2)`` frequencies from both the
    positive- and negative-frequency halves of the first spatial axis (the
    second axis is real-FFT'd, so only positive frequencies exist).

    Parameters
    ----------
    in_channels, out_channels : int
        Number of input / output feature channels.
    modes1, modes2 : int
        Number of Fourier modes to retain along each spatial axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batched complex multiplication via einsum: (B,I,X,Y)×(I,O,X,Y)→(B,O,X,Y)."""
        return torch.einsum("bixy,ioxy->boxy", a, b)

    # -- forward ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(B, C_in, H, W)``

        Returns
        -------
        ``(B, C_out, H, W)``
        """
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]

        # 1. Forward 2-D real FFT  →  (B, C_in, H, W//2+1)  complex
        x_ft = torch.fft.rfft2(x)

        # 2. Allocate output spectrum
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # 3. Multiply the retained low-frequency modes by learned weights
        #    — positive-frequency block (top rows)
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1,
        )
        #    — negative-frequency block (bottom rows, conjugate-symmetric half)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2,
        )

        # 4. Inverse FFT back to spatial domain
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2-D static bulk reconstruction.

    Architecture::

        Lift(in_channels → width)
          → [SpectralConv2d + Conv1x1 + GELU] × n_layers
          → Project(width → 128 → 1)

    The first ``n_layers - 1`` blocks apply GELU; the final block is linear
    (no activation) following the original FNO convention.

    Parameters
    ----------
    modes1, modes2 : int
        Fourier modes retained per spatial axis (default 12).
    width : int
        Hidden channel dimension (default 32).
    n_layers : int
        Number of Fourier layers (default 4).
    in_channels : int
        Number of input channels (default **2**: wave + Y-depth coordinate).
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        # Lifting layer: project input channels to high-dim feature space
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # Fourier layers  (spectral conv  +  pointwise bypass)
        self.spectral_convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.bypass_convs = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )

        # Projection head: width → 128 → 1
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(B, in_channels, H, W)``
            For the default AdS/CFT setup ``in_channels = 2``:
            channel 0 = boundary wave tiled over the grid,
            channel 1 = normalised Y-depth coordinate.

        Returns
        -------
        ``(B, 1, H, W)`` — predicted bulk scalar field.
        """
        # Lift
        x = self.lift(x)  # (B, width, H, W)

        # Fourier layers
        for i, (spec, byp) in enumerate(zip(self.spectral_convs, self.bypass_convs)):
            x1 = spec(x)
            x2 = byp(x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)  # no activation on the last layer

        # Project to output
        x = x.permute(0, 2, 3, 1)   # (B, H, W, width)
        x = F.gelu(self.proj1(x))   # (B, H, W, 128)
        x = self.proj2(x)           # (B, H, W, 1)
        x = x.permute(0, 3, 1, 2)   # (B, 1, H, W)
        return x


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3-D  COMPONENTS  (Phase 2 — 2+1 D spatiotemporal evolution)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SpectralConv3d(nn.Module):
    """Spectral convolution in 3-D Fourier space (time × x × y).

    Uses ``torch.fft.rfftn`` with ``dim=(-3, -2, -1)`` —  the *last* spatial
    axis is real-FFT'd (halved), while the first two axes retain both positive
    and negative frequency blocks.

    Four weight tensors cover the four relevant quadrants of the 3-D
    half-complex spectrum ``(±modes1, ±modes2, +modes3)``.

    Parameters
    ----------
    in_channels, out_channels : int
    modes1, modes2, modes3 : int
        Retained Fourier modes along time, x, and y respectively.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2, modes3)

        # Four quadrant weights: (±T, ±X, +Y_rfft)
        self.w1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))  # (+T, +X, +Y)
        self.w2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))  # (+T, -X, +Y)
        self.w3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))  # (-T, +X, +Y)
        self.w4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))  # (-T, -X, +Y)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(B,I,T,X,Y) × (I,O,T,X,Y) → (B,O,T,X,Y)"""
        return torch.einsum("bitxy,iotxy->botxy", a, b)

    # -- forward ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(B, C_in, T, X, Y)``

        Returns
        -------
        ``(B, C_out, T, X, Y)``
        """
        B = x.shape[0]
        T, X, Y = x.shape[-3], x.shape[-2], x.shape[-1]

        # 3-D real FFT  →  last axis halved
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            B, self.out_channels, T, X, Y // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        m1, m2, m3 = self.modes1, self.modes2, self.modes3

        # Quadrant 1  (+T, +X, +Y)
        out_ft[:, :, :m1, :m2, :m3] = self._compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.w1,
        )
        # Quadrant 2  (+T, -X, +Y)
        out_ft[:, :, :m1, -m2:, :m3] = self._compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.w2,
        )
        # Quadrant 3  (-T, +X, +Y)
        out_ft[:, :, -m1:, :m2, :m3] = self._compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.w3,
        )
        # Quadrant 4  (-T, -X, +Y)
        out_ft[:, :, -m1:, -m2:, :m3] = self._compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.w4,
        )

        return torch.fft.irfftn(out_ft, s=(T, X, Y))


class FNO3d(nn.Module):
    """Fourier Neural Operator for 2+1 D spatiotemporal bulk evolution.

    Extends the FNO2d architecture to three dimensions (time × radial-x ×
    boundary-y) for learning how the bulk AdS geometry evolves under time-
    dependent boundary perturbations.

    Architecture::

        Lift(in_channels → width)
          → [SpectralConv3d + Conv1×1×1 + GELU] × n_layers
          → Project(width → 128 → 1)

    Parameters
    ----------
    modes1, modes2, modes3 : int
        Fourier modes retained along time, x, and y (default 8 each).
    width : int
        Hidden channel dimension (default 20).
    n_layers : int
        Number of Fourier layers (default 4).
    in_channels : int
        Number of input channels (default **3**: wave + time coord + depth coord).
    """

    def __init__(
        self,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 20,
        n_layers: int = 4,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers

        # Lifting
        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)

        # Fourier layers
        self.spectral_convs = nn.ModuleList(
            [SpectralConv3d(width, width, modes1, modes2, modes3) for _ in range(n_layers)]
        )
        self.bypass_convs = nn.ModuleList(
            [nn.Conv3d(width, width, kernel_size=1) for _ in range(n_layers)]
        )

        # Projection
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(B, in_channels, T, X, Y)``
            For the default setup ``in_channels = 3``:
            ch-0 = boundary wave tiled, ch-1 = time coord, ch-2 = depth coord.

        Returns
        -------
        ``(B, 1, T, X, Y)`` — predicted spatiotemporal bulk field.
        """
        # Lift
        x = self.lift(x)  # (B, width, T, X, Y)

        # Fourier layers
        for i, (spec, byp) in enumerate(zip(self.spectral_convs, self.bypass_convs)):
            x1 = spec(x)
            x2 = byp(x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # Project
        x = x.permute(0, 2, 3, 4, 1)   # (B, T, X, Y, width)
        x = F.gelu(self.proj1(x))       # (B, T, X, Y, 128)
        x = self.proj2(x)               # (B, T, X, Y, 1)
        x = x.permute(0, 4, 1, 2, 3)   # (B, 1, T, X, Y)
        return x
