# AdS-CFT-SciML-Engine
**Accelerating AdS/CFT Boundary-Value Problems via SciML.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12nzZXFXGGYJryjT6n0VQGWy3gW1rS6Fg?usp=sharing)

This engine bypasses iterative O(N^3) numerical PDE solvers using advanced Operator Learning, moving beyond static 3D snapshots to forge a continuous, mathematically rigorous pipeline driven by General Relativity.

### Directory Overview

#### [Milestone 1: Unified_Neural_AdS](./Unified_Neural_AdS) (The Proven Surrogate)
* **Status:** Proven Baseline
* **Key Metrics:** >500x inference speedup, 3% relative L2 error on dual-source quantum collisions.
* **Live Demo:** [Interactive Colab with t=8 & t=14 Reconstructions](https://colab.research.google.com/drive/12nzZXFXGGYJryjT6n0VQGWy3gW1rS6Fg?usp=sharing)

#### [Milestone 2: Quantum_Hybrid_AdS](./Quantum_Hybrid_AdS) (The Architecture Flex)
* **Status:** Experimental Hybrid
* **Focus:** Embedding 10-Qubit PennyLane VQCs within FiLM-Conditioned SIREN decoders.

#### [Milestone 3: 6-Phase_4D_HoloEngine](./6-Phase_4D_HoloEngine) (The Active Frontier)
* **Status:** Active Development / Seeking Mentorship
* **Focus:** Solving continuous coordinate samplers and PyTorch Autograd PDEs.
