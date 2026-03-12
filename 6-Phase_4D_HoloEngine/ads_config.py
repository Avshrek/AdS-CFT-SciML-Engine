"""
ads_config.py — Configuration for the Full Einstein BBH Merger Engine
======================================================================

Characteristic formulation of GR in AdS4 with Z2 (head-on) symmetry.
Metric ansatz in infalling Eddington-Finkelstein coordinates (v, x, z):

    ds² = (1/z²)[ -A dv² + Σ²(e^B dx² + e^{-B} dy²) + 2 dv dz + 2V dv dx ]

Network outputs 5 scalar fields:  A, Σ, B, V, φ
Coordinates:  v (advanced time), x (transverse), u = ln(z) (holographic radial)

References
----------
  Chesler & Yaffe, JHEP 1407 (2014) 086
  Bantilan, Figueras & Kunesch, PRL 124 (2020) 191101
  Chesler & Yaffe, PRL 106 (2011) 021601
"""

import math
import torch


class BBHConfig:
    # =====================================================================
    #  COORDINATE DOMAIN  (Characteristic null slicing)
    # =====================================================================
    Z_MIN       = 1e-4              # AdS boundary cutoff
    Z_MAX       = 1.0               # Deep bulk / near-horizon
    U_MIN       = math.log(Z_MIN)   # ≈ -9.2103
    U_MAX       = 0.0               # ln(1) = 0
    U_BOUNDARY  = math.log(Z_MIN)   # Boundary pinned at u = ln(z_min)

    X_RANGE     = (-1.0, 1.0)       # Transverse spatial coordinate
    V_RANGE     = (0.0, 1.0)        # Advanced time (null)

    # Y-direction is suppressed by Z2 symmetry (head-on collision)
    # The metric encodes y-dependence through the anisotropy B

    # =====================================================================
    #  METRIC DEGREES OF FREEDOM
    # =====================================================================
    # The SIREN outputs 5 channels: [A, Sigma, B, V_shift, phi]
    NUM_METRIC_FIELDS   = 5
    FIELD_NAMES         = ['A', 'Sigma', 'B', 'V_shift', 'phi']

    # Indices into the 5-channel output
    IDX_A       = 0     # Lapse / redshift function
    IDX_SIGMA   = 1     # Area element
    IDX_B       = 2     # Anisotropy (GW polarization)
    IDX_V       = 3     # Shift vector (x-momentum)
    IDX_PHI     = 4     # Scalar field (matter)

    # =====================================================================
    #  NEURAL ARCHITECTURE
    # =====================================================================
    LATENT_DIM          = 128       # Quantum latent vector dimension
    SIREN_HIDDEN        = 256       # Width of SIREN hidden layers
    SIREN_LAYERS        = 6         # Deeper for 5-channel coupled output
    SIREN_OMEGA_0       = 30.0      # First-layer frequency

    # 3D Conv Encoder  (encodes boundary stress-tensor sequence)
    # Input: (B, 1, N_v, N_x, 1) -> boundary data over (v, x)
    # We treat the boundary data as a 2D sequence with channel dim
    ENCODER_CHANNELS        = [1, 16, 32, 64]
    ENCODER_TEMPORAL_FRAMES = 100   # v-slices
    ENCODER_SPATIAL_RES     = 64    # x-resolution

    # =====================================================================
    #  PHYSICAL CONSTANTS
    # =====================================================================
    # AdS4 cosmological constant: Λ = -3/L² with L=1
    LAMBDA_COSMOLOGICAL = -3.0
    NEWTON_G            = 1.0 / (16.0 * math.pi)  # 8πG = 1 convention
    DELTA               = 3.0       # Boundary scaling dimension (AdS4)

    # Scalar field coupling
    SCALAR_MASS_SQ      = 0.0       # Massless conformally coupled scalar
    LAMBDA_NL           = 0.0       # No self-interaction (pure GR + minimal scalar)

    # =====================================================================
    #  PURE ADS REFERENCE METRIC  (what the network deviations are from)
    # =====================================================================
    # In Poincaré patch EF coords, pure (planar) AdS4 has:
    #   A = 1   (NOT 1+z² — that's global AdS)
    #   Σ = 1
    #   B = 0
    #   V = 0
    #   φ = 0
    # The network learns DEVIATIONS from this background

    # =====================================================================
    #  BOUNDARY ASYMPTOTICS  (holographic renormalization)
    # =====================================================================
    # Near z→0, metric functions must have specific falloffs:
    #   A(v,x,z) = 1 + a₃(v,x) z³ + O(z⁴)         (a₃ encodes <T_vv>)
    #   Σ(v,x,z) = 1 + σ₃(v,x) z³ + O(z⁴)
    #   B(v,x,z) = b₃(v,x) z³ + O(z⁴)               (b₃ encodes <T_xx-T_yy>)
    #   V(v,x,z) = v₂(v,x) z² + O(z³)               (shift falls as z²)
    # The z³ coefficients map to <T_μν> of the boundary CFT
    ASYMPTOTIC_ORDER    = 3         # Power of z at which CFT data appears

    # =====================================================================
    #  INITIAL DATA  (boosted black holes)
    # =====================================================================
    # Two boosted AdS-Schwarzschild BHs at v=0
    BH_MASS_1           = 0.5       # First BH Schwarzschild mass parameter
    BH_MASS_2           = 0.5       # Second BH (equal mass)
    BH_POSITION_1       = -0.4      # x-coordinate of BH 1 at v=0
    BH_POSITION_2       = 0.4       # x-coordinate of BH 2 at v=0
    BH_BOOST_1          = 0.3       # Boost velocity of BH 1 (toward center)
    BH_BOOST_2          = -0.3      # Boost velocity of BH 2 (toward center)
    BH_WIDTH            = 0.15      # Gaussian width of energy profile

    # =====================================================================
    #  SAMPLING
    # =====================================================================
    BOUNDARY_BATCH      = 1024      # Points on z=z_min boundary per step
    BULK_BATCH          = 2048      # Bulk collocation points per step
    HORIZON_BATCH       = 512       # Points for horizon-finding
    RADIAL_BATCH        = 256       # Points per radial ODE line
    CAUCHY_BATCH        = 512       # Points on initial surface v=0

    # =====================================================================
    #  CAUSAL PINN  (advanced-time ordering)
    # =====================================================================
    NUM_TIME_CHUNKS     = 16        # Chunks along v-axis
    CAUSAL_EPSILON      = 0.1       # Gentle causal weighting

    # =====================================================================
    #  TRAINING SCHEDULE  (3-Phase Curriculum)
    # =====================================================================
    TOTAL_EPOCHS                = 3000
    PHASE_A_EPOCHS              = 500   # Pure AdS + boundary data
    PHASE_B_EPOCHS              = 1000  # Nested radial Einstein eqs
    # Phase C: epochs PHASE_A + PHASE_B + 1 → TOTAL  (full evolution)

    LR_PHASE_A                  = 5e-4
    LR_PHASE_B                  = 2e-4
    LR_PHASE_C                  = 1e-4
    GRAD_CLIP                   = 1.0
    SCHEDULER_PATIENCE          = 400
    SCHEDULER_FACTOR            = 0.5

    # =====================================================================
    #  LOSS WEIGHTS
    # =====================================================================
    # Phase A
    W_BOUNDARY          = 500.0     # Boundary metric falloff data
    W_PURE_ADS          = 100.0     # Enforce pure AdS at v=0 interior

    # Phase B (nested radial equations)
    W_EINSTEIN_RADIAL   = 10.0      # Nested ODE residuals
    W_SCALAR_EQ         = 5.0       # Klein-Gordon residual
    W_CONSTRAINT        = 10.0      # Algebraic constraint residual

    # Phase C (evolution + everything)
    W_EVOLUTION         = 10.0      # d_+(Sigma_u) evolution equation
    W_EVOLUTION_B       = 8.0       # d_+(B) anisotropy evolution
    W_EVOLUTION_V       = 8.0       # d_+(V) shift evolution
    W_EVOLUTION_A       = 8.0       # d_+(A) lapse evolution
    W_HORIZON_REG       = 1.0       # Horizon regularity condition
    W_CAUSALITY         = 1.0       # Lorentzian signature enforcement
    W_QUANTUM           = 5.0       # Quantum entropy tether
    W_CONSTRAINT_DAMP   = 20.0      # Extra constraint weight near v=0
    EXCISION_BUFFER     = 0.05      # Buffer outside AH for excision
    ADAPTIVE_FRAC       = 0.25      # Fraction of bulk batch for adaptive near-horizon

    # =====================================================================
    #  WARMUP & STABILITY
    # =====================================================================
    PDE_WARMUP_EPOCHS   = 200       # Gradual Einstein equation introduction
    NAN_MAX_RECOVERIES  = 5
    LOG1P_STABILIZE     = True      # log1p on all residuals

    # =====================================================================
    #  QUANTUM CIRCUIT  (CFT state preparation)
    # =====================================================================
    NUM_QUBITS          = 12        # 12 qubits for boundary CFT state
    QUANTUM_LAYERS      = 4         # Variational depth
    QUANTUM_UPDATE_EVERY = 10       # Epochs between entropy re-evaluation
    SUBSYSTEM_QUBITS    = 6         # Bipartition: 6|6 for entanglement

    # =====================================================================
    #  HARDWARE
    # =====================================================================
    DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION     = False     # fp32 required for 2nd-order autograd
    CHECKPOINT_DIR      = "checkpoints/bbh"
    LOG_EVERY           = 10
    SAVE_EVERY          = 200
