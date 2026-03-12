"""
quantum_tether.py - PennyLane Quantum Entropy Tether  (Phase 6)
=================================================================

Computes the von Neumann entanglement entropy of a 10-qubit parameterised
quantum circuit.  The parameter-shift rule needs 20 full backward passes
for a 10-qubit gradient, so we use **Alternating Optimization**:

    1. Compute  S_quantum(t)  once per epoch (or every N epochs).
    2. Detach / cache the scalar target.
    3. Use analytic PyTorch gradients to force the geometric HRT Area
       to match the cached entropy.

If PennyLane is not installed, a smooth classical entropy proxy is used
so that the rest of the pipeline still runs.
"""

import math
import numpy as np
import torch

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False

from config import Config


class QuantumEntropyTether:
    """
    Manages the quantum circuit, its cached entropy value, and the
    latent -> circuit-parameter mapping.
    """

    def __init__(self, config: type = Config):
        self.cfg = config
        self.cached_entropy = None   # scalar, detached

        if HAS_PENNYLANE:
            self.dev = qml.device("default.qubit", wires=config.NUM_QUBITS)
            self._build_circuit()
        else:
            print("[QuantumTether] PennyLane not found – falling back to "
                  "classical entropy proxy.")

    # ------------------------------------------------------------------ #
    #  Circuit construction                                                #
    # ------------------------------------------------------------------ #
    def _build_circuit(self):
        nq = self.cfg.NUM_QUBITS
        nl = self.cfg.QUANTUM_LAYERS

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def _circuit(params):
            # Initial superposition
            for i in range(nq):
                qml.Hadamard(wires=i)
            # Parameterised variational layers
            for layer in range(nl):
                for i in range(nq):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)
                # Entangling CNOT ladder + circular bond
                for i in range(nq - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[nq - 1, 0])
            return qml.state()

        self._circuit = _circuit

    # ------------------------------------------------------------------ #
    #  Latent  ->  circuit parameters                                      #
    # ------------------------------------------------------------------ #
    def _latent_to_params(self, z_latent: torch.Tensor) -> torch.Tensor:
        """Map a 1-D latent vector to circuit rotation angles."""
        n_total = self.cfg.QUANTUM_LAYERS * self.cfg.NUM_QUBITS * 2
        # Tile the latent to fill all parameters, then reshape
        expanded = z_latent.detach().repeat(math.ceil(n_total / self.cfg.LATENT_DIM))
        expanded = expanded[:n_total]
        return (expanded * math.pi).reshape(
            self.cfg.QUANTUM_LAYERS, self.cfg.NUM_QUBITS, 2
        )

    # ------------------------------------------------------------------ #
    #  Entropy computation                                                 #
    # ------------------------------------------------------------------ #
    def compute_entropy(self, z_latent: torch.Tensor) -> torch.Tensor:
        """
        Compute the von Neumann entanglement entropy  S = -Tr(rho log rho)
        of the reduced density matrix for the first half of qubits.

        Parameters
        ----------
        z_latent : 1-D tensor of size LATENT_DIM

        Returns
        -------
        entropy : scalar tensor
        """
        if not HAS_PENNYLANE:
            return self._classical_proxy(z_latent)

        params = self._latent_to_params(z_latent)
        state  = self._circuit(params)                        # (2^nq,) complex

        nq   = self.cfg.NUM_QUBITS
        n_A  = nq // 2
        dim_A, dim_B = 2 ** n_A, 2 ** (nq - n_A)

        psi   = state.reshape(dim_A, dim_B)
        rho_A = psi @ psi.conj().T                            # reduced rho_A

        # Eigenvalues (Hermitian matrix)
        eigvals = torch.linalg.eigvalsh(rho_A.real)
        eigvals = torch.clamp(eigvals, min=1e-12)
        entropy = -torch.sum(eigvals * torch.log(eigvals))
        return entropy

    def _classical_proxy(self, z_latent: torch.Tensor) -> torch.Tensor:
        """Softmax-based entropy proxy when PennyLane is absent."""
        p = torch.softmax(z_latent.detach().float(), dim=-1)
        p = torch.clamp(p, min=1e-12)
        return -torch.sum(p * torch.log(p))

    # ------------------------------------------------------------------ #
    #  Cache management  (Alternating Optimisation)                        #
    # ------------------------------------------------------------------ #
    def update_cache(self, z_latent: torch.Tensor):
        """Recompute and cache entropy (call once every N epochs)."""
        with torch.no_grad():
            self.cached_entropy = self.compute_entropy(z_latent).detach()

    def get_cached_entropy(self) -> torch.Tensor:
        """Return the cached entropy scalar (detached, safe for loss)."""
        if self.cached_entropy is None:
            return torch.tensor(0.0)
        return self.cached_entropy


# ====================================================================== #
#  Loss function                                                           #
# ====================================================================== #
def quantum_entropy_tether_loss(hrt_area: torch.Tensor,
                                 cached_entropy: torch.Tensor) -> torch.Tensor:
    """
    L_quantum = ( HRT_area  -  S_quantum )^2

    hrt_area       – differentiable scalar from the geometric HRT pipeline
    cached_entropy – detached scalar from the quantum circuit (fixed target)
    """
    return (hrt_area - cached_entropy) ** 2
