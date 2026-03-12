"""
classical_autoencoder.py -- Classical 10-dim Bottleneck Autoencoder
===================================================================

Identical encoder/decoder architecture to HybridQuantumAdS, but the
10-qubit QuantumLatentLayer is replaced with a classical Linear(10,10)
+ Tanh bottleneck.  Used for Phase 1 pre-training so the classical
layers learn a stable compression/decompression before the quantum
circuit is introduced.

The bottleneck output is clamped to [-1, 1] via Tanh to match the
quantum Pauli-Z expectation value range.
"""

import torch
import torch.nn as nn


class ClassicalAdS(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # ---------------------------------------------------------
        # 1. ENCODER  (identical to HybridQuantumAdS)
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
        # 2. CLASSICAL BOTTLENECK  (replaces quantum layer)
        #    Same 10 -> 10 dimensionality.
        #    Tanh keeps output in [-1, 1] like Pauli-Z expectation.
        # ---------------------------------------------------------
        self.classical_bottleneck = nn.Sequential(
            nn.Linear(10, 10),
            nn.Tanh(),
        )

        # ---------------------------------------------------------
        # 3. DECODER  (identical to HybridQuantumAdS)
        # ---------------------------------------------------------
        self.decoder_projection = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 2 * 8 * 8),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1,
                               output_padding=(1, 0, 0)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1,
                               output_padding=(0, 0, 0)),
            nn.ReLU(),
            nn.ConvTranspose3d(16, out_channels, kernel_size=4, stride=2,
                               padding=1, output_padding=(0, 0, 0)),
        )

    def forward(self, x):
        latent = self.encoder(x)
        bottleneck_out = self.classical_bottleneck(latent)
        x_expanded = self.decoder_projection(bottleneck_out)
        x_reshaped = x_expanded.view(-1, 64, 2, 8, 8)
        return self.decoder_conv(x_reshaped)

    # ---------------------------------------------------------
    # Weight transfer: extract encoder + decoder for quantum model
    # ---------------------------------------------------------
    def get_transferable_weights(self):
        """
        Return a state_dict containing ONLY encoder + decoder weights,
        keyed exactly as HybridQuantumAdS expects them.

        The classical_bottleneck weights are discarded -- the quantum
        layer will replace them.
        """
        transfer = {}
        for name, param in self.state_dict().items():
            if name.startswith("classical_bottleneck"):
                continue  # discard -- quantum layer replaces this
            transfer[name] = param
        return transfer


# --- Quick Test ---
if __name__ == "__main__":
    print("Initializing Classical 10-dim Autoencoder...")
    model = ClassicalAdS()
    dummy = torch.randn(2, 1, 20, 64, 64)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Transferable keys: {list(model.get_transferable_weights().keys())[:5]} ...")
    n = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n:,}")
