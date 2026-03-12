import torch
import torch.nn as nn
from quantum_architecture import QuantumLatentLayer

class HybridQuantumAdS(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # ---------------------------------------------------------
        # 1. THE CLASSICAL ENCODER (The Compressor)
        # Crushes the 20x64x64 volume down to bypass the QRAM bottleneck
        # ---------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=4, stride=2, padding=1), # -> 10x32x32
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),          # -> 5x16x16
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),          # -> 2x8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10), # The exact 10 numbers for our qubits
            nn.Tanh() # Tanh keeps numbers between -1 and 1 (perfect for quantum rotations)
        )
        
        # ---------------------------------------------------------
        # 2. THE QUANTUM BOTTLENECK (The Physics Engine)
        # 10 Qubits entangle to map the boundary math to the bulk math
        # ---------------------------------------------------------
        self.quantum_layer = QuantumLatentLayer(n_quantum_layers=3)
        
        # ---------------------------------------------------------
        # 3. THE CLASSICAL DECODER (The Expander)
        # Projects the quantum output back into the 3D spacetime volume
        # ---------------------------------------------------------
        self.decoder_projection = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 2 * 8 * 8),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(1,0,0)), # -> 5x16x16
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=(0,0,0)), # -> 10x32x32
            nn.ReLU(),
            nn.ConvTranspose3d(16, out_channels, kernel_size=4, stride=2, padding=1, output_padding=(0,0,0)) # -> 20x64x64
        )

    def forward(self, x):
        # Step 1: Compress classical spacetime
        latent_classical = self.encoder(x)
        
        # Step 2: Quantum Entanglement Mapping
        latent_quantum = self.quantum_layer(latent_classical)
        
        # Step 3: Expand back to classical 3D bulk
        x_expanded = self.decoder_projection(latent_quantum)
        x_reshaped = x_expanded.view(-1, 64, 2, 8, 8) 
        out_volume = self.decoder_conv(x_reshaped)
        
        return out_volume

# --- Quick Test ---
if __name__ == "__main__":
    print("🔬 Initializing Hybrid Quantum-Classical Engine...")
    model = HybridQuantumAdS()
    
    # Simulating one batch of your 20x64x64 boundary wave
    dummy_wave = torch.randn(1, 1, 20, 64, 64)
    
    print("⏳ Executing Forward Pass (Compress -> Quantum Compute -> Expand)...")
    output = model(dummy_wave)
    
    print(f"✅ Success! Output Volume Shape: {output.shape}")
