# AdS-CFT-SciML-Engine
**Accelerating AdS/CFT Boundary-Value Problems via SciML.**

This engine bypasses iterative O(N^3) numerical PDE solvers using advanced Operator Learning, moving beyond static 3D snapshots to forge a continuous, mathematically rigorous pipeline driven by General Relativity.

### Directory Overview

#### [Milestone 1: Unified_Neural_AdS](./Unified_Neural_AdS) (The Proven Surrogate)
* Status: Proven Baseline
* Key Metrics: >500x inference speedup over classical LU factorization solvers, ~0.02 Mean Absolute Error (MAE), and a 3% relative L2 error baseline on dual-source quantum collisions.
* Focus: Frequency domain operations mapping 1D+Time boundaries to a 2D+Time bulk using FNO architectures.

#### [Milestone 2: Quantum_Hybrid_AdS](./Quantum_Hybrid_AdS) (The Architecture Flex)
* Status: Experimental Hybrid
* Focus: Bridging quantum-classical pipelines by embedding a 10-Qubit PennyLane Variational Quantum Circuit within the latent space, utilizing FiLM-Conditioned SIREN decoders with Hyperbolic Positional Encoding to dynamically guide macroscopic Ryu-Takayanagi minimal area loss.

#### [Milestone 3: 6-Phase_4D_HoloEngine](./6-Phase_4D_HoloEngine) (The Active Frontier)
* Status: Active Development / Seeking Mentorship
* Focus: Solving continuous coordinate samplers and PyTorch Autograd PDEs.
* Roadblock: While the architecture handles the exact PDEs computationally via Conformal Compactification and Inverted Skenderis Renormalization, it is currently struggling to learn the correct non-linear physical convergence and gravitational backreaction.
