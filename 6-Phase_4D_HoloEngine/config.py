"""
config.py - Centralized Configuration for the 4D Holographic Quantum Gravity Simulator
========================================================================================
All hyperparameters, physical constants, coordinate domains, and training schedules.
Tuned for Kaggle T4 GPU (16 GB VRAM).

Incorporates all 15 battle-tested numerical fixes:
  FIX  1: omega_0 = 30    — halves derivative magnitude, quarters d²
  FIX  2: epsilon = 0.1   — gentle causal PINN weighting (all chunks contribute)
  FIX  3: kappa = 10/5    — backreaction max 6× (not 51×)
  FIX  4: LR_B = 1e-4     — adequate Phase B learning rate
  FIX  5: patience = 200  — prevents premature LR decay
  FIX  6: W_PDE = 1.0     — retuned for log-space residuals
  FIX  7: W_CAUCHY_DT=0.1 — very soft momentum, Phase B only
  FIX  8: PDE_WARMUP=200  — gradual physics introduction
  FIX  9: NaN recovery    — checkpoint reload on divergence
  FIX 10: log1p(res²)     — maps any residual to O(1)–O(50)
  FIX 11: log1p Sommerfeld — prevents O(10^13) domination
  FIX 12: Cauchy interior — prevents contradiction with boundary data
  FIX 13: log1p causality — bounded acausal penalty
  FIX 14: Phase A = PURE DATA — no cauchy, no PDE
  FIX 15: Phase B warmup  — prevents gradient shock at transition
"""

import math
import torch


class Config:
    # =========================================================================
    # COORDINATE DOMAIN  (Phase 1 - Conformal Compactification)
    # =========================================================================
    Z_MIN       = 1e-4                          # AdS boundary cutoff
    Z_MAX       = 1.0                           # Deep bulk
    U_MIN       = math.log(Z_MIN)               # ln(1e-4) ≈ -9.2103
    U_MAX       = 0.0                           # ln(1)    = 0
    U_BOUNDARY  = math.log(Z_MIN)               # Boundary pinned at u = ln(z_min)
    X_RANGE     = (-1.0, 1.0)
    Y_RANGE     = (-1.0, 1.0)
    T_RANGE     = (0.0, 1.0)

    # =========================================================================
    # NEURAL ARCHITECTURE  (Phase 2 - FiLM-SIREN)
    # =========================================================================
    LATENT_DIM          = 128       # Quantum latent vector dimension
    SIREN_HIDDEN        = 256       # Width of every SIREN hidden layer
    SIREN_LAYERS        = 5         # Number of hidden SIREN layers
    SIREN_OMEGA_0       = 30.0      # FIX 1: was 60 → halves ∂ magnitude

    # 3D Convolutional Encoder
    ENCODER_CHANNELS        = [1, 16, 32, 64]
    ENCODER_TEMPORAL_FRAMES = 100   # Frames in the collision fluid sequence
    ENCODER_SPATIAL_RES     = 64    # H = W of each frame

    # =========================================================================
    # PHYSICS CONSTANTS  (Phases 3-4)
    # =========================================================================
    DELTA       = 3.0       # Boundary scaling dimension (massless scalar in AdS4)
    LAMBDA_NL   = 1.0       # Non-linear self-coupling  λ φ³
    KAPPA       = 10.0      # FIX 3: was 150 → bounded backreaction
    KAPPA_MAX   = 5.0       # FIX 3: was 50 → max 6× amplification

    # =========================================================================
    # SAMPLING  (Phase 1 - Dual Sampler)
    # =========================================================================
    BOUNDARY_BATCH  = 1024  # Discrete boundary points per step
    BULK_BATCH      = 2048  # Continuous bulk collocation points per step

    # =========================================================================
    # CAUSAL PINN  (Phase 5)
    # =========================================================================
    NUM_TIME_CHUNKS     = 16        # Base chunk count (adjusted by CFL)
    CAUSAL_EPSILON      = 0.1       # FIX 2: was 10 → all time chunks contribute

    # =========================================================================
    # TRAINING SCHEDULE  (Phase 5 - Curriculum)
    # =========================================================================
    TOTAL_EPOCHS                = 2000
    CURRICULUM_PHASE_A_EPOCHS   = 800   # FIX 14: pure data fitting phase
    LR_PHASE_A                  = 5e-4
    LR_PHASE_B                  = 1e-4  # FIX 4: was 5e-5 → adequate physics LR
    GRAD_CLIP                   = 1.0
    GRADIENT_ACCUMULATION_STEPS = 1     # Increase if OOM on T4
    SCHEDULER_PATIENCE          = 200   # FIX 5: was 50 → max 3 halvings per phase
    SCHEDULER_FACTOR            = 0.5

    # =========================================================================
    # LOSS WEIGHTS
    # =========================================================================
    W_DATA          = 500.0     # Boundary anchor data loss (high priority)
    W_PDE           = 1.0       # FIX 6: retuned for log-space residuals
    W_CAUCHY        = 1.0       # Phase B only, bulk interior (FIX 12/14)
    W_CAUCHY_DT     = 0.1       # FIX 7: very soft momentum, Phase B only
    W_SOMMERFELD    = 0.1       # Radiative boundary absorption
    W_HRT           = 0.0       # HRT area (0 = no direct area penalty)
    W_CAUSALITY_HRT = 1.0       # Lorentzian causality penalty
    W_QUANTUM       = 10.0      # Quantum entropy tether

    # =========================================================================
    # WARMUP & STABILITY  (FIX 8, 9)
    # =========================================================================
    PDE_WARMUP_EPOCHS   = 200       # FIX 8: gradual physics introduction
    NAN_MAX_RECOVERIES  = 5         # FIX 9: auto-reload on divergence

    # =========================================================================
    # QUANTUM CIRCUIT  (Phase 6)
    # =========================================================================
    NUM_QUBITS      = 10
    QUANTUM_LAYERS  = 3
    QUANTUM_UPDATE_EVERY = 10   # Epochs between quantum entropy re-evaluation

    # =========================================================================
    # HARDWARE / KAGGLE T4
    # =========================================================================
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION = False     # Keep False: 2nd-order autograd needs fp32
    CHECKPOINT_DIR  = "checkpoints"
    LOG_EVERY       = 10
    SAVE_EVERY      = 200
