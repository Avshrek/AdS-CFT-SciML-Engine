import scipy.constants  # ADD THIS LINE FIRST
import pennylane as qml
import torch
import torch.nn as nn

# 1. Initialize the High-Speed Quantum Simulator
n_qubits = 10

# 🔥 FORCE QUANTUM MATH 🔥
dev = qml.device("default.qubit", wires=n_qubits)
print("⚡ [PHYSICS ENGINE] Quantum Math forced to PyTorch backend")


# 2. Define the Quantum Entanglement Circuit
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    # Phase A: Amplitude Encoding (The Pro Way)
    # AngleEmbedding automatically handles PyTorch batches (like our 5 universes)
    # and maps them perfectly to the Y-rotation of the qubits without manual loops!
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        
    # Phase B: The "Brain" of the QML
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Phase C: Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
# 3. The PyTorch Wrapper
# This makes the quantum circuit act exactly like a standard PyTorch nn.Module layer
class QuantumLatentLayer(nn.Module):
    def __init__(self, n_quantum_layers=3):
        super().__init__()
        # Define the shape of the learnable weights for the quantum gates
        weight_shapes = {"weights": (n_quantum_layers, n_qubits, 3)}
        
        # Wrap the circuit so it can receive PyTorch tensors and compute gradients
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        # x is the [Batch, 10] tensor coming from your classical Encoder
        return self.q_layer(x).float()

# -------------------------------------------------------------------
# 4. QUANTUM OBSERVER NODE (Nature Metric Extraction)
# -------------------------------------------------------------------

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_observer_circuit(inputs, weights):
    """
    Identical to the training circuit, but returns the full Density Matrix (ρ)
    instead of measured Pauli-Z observables.
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # We return the density matrix of the entire 10-qubit system
    return qml.density_matrix(wires=range(n_qubits))

import numpy as np

class QuantumEntropyExtractor:
    """
    Takes the precise weights learned by `QuantumLatentLayer` and extracts
    the von Neumann Entanglement Entropy from the resulting density matrix.
    S = -Tr(ρ ln ρ)
    """
    @staticmethod
    def calculate_von_neumann_entropy(density_matrix):
        # Convert to numpy for stable eigenvalue calculation and to avoid PyTorch LAPACK issues on Windows
        if isinstance(density_matrix, torch.Tensor):
            density_matrix = density_matrix.detach().cpu().numpy()
            
        # Get eigenvalues (probabilities of the quantum microstates) using numpy
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        
        # Filter out 0s/negatives to avoid log(0) issues due to float precision
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        # S = - sum(p * ln(p)) using numpy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return entropy.item()

    @staticmethod
    def extract_entropy(inputs, q_layer_module):
        """
        Pass a batch of inputs and the trained QuantumLatentLayer to
        physically calculate the entanglement entropy of the bottleneck.
        """
        # We need the exact weights the training circuit is currently using
        current_weights = q_layer_module.q_layer.weights.detach()
        
        # We will loop through the batch and compute entropy per sample
        batch_entropy = []
        for i in range(inputs.shape[0]):
            single_input = inputs[i]
            
            # Execute observer circuit
            rho = quantum_observer_circuit(single_input, current_weights)
            
            # Calculate Entropy
            S = QuantumEntropyExtractor.calculate_von_neumann_entropy(rho)
            batch_entropy.append(S)
            
        return sum(batch_entropy) / len(batch_entropy)

# --- Quick Test ---
if __name__ == "__main__":
    print("🔬 Booting Quantum Simulator...")
    q_model = QuantumLatentLayer()
    
    # Fake input vector representing the compressed boundary wave
    dummy_input = torch.rand((2, 10)) 
    
    print("⏳ Executing Forward Pass...")
    out = q_model(dummy_input)
    print(f"✅ Quantum Forward pass successful. Shape: {out.shape}")
    
    print("⏳ Extracting Entanglement Entropy...")
    entropy = QuantumEntropyExtractor.extract_entropy(dummy_input, q_model)
    print(f"✅ von Neumann Entropy calculated: S = {entropy:.4f}")
