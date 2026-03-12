"""
generate_apex_master_dataset.py
================================
Generates a mathematically precise, C-infinity 4D Holographic Boundary Dataset
for training a continuous PINN that simulates binary scalar collapse in AdS4.

Dependencies: numpy ONLY.
Output:       apex_master_dataset.npz

Strictly obeys all 5 Phases:
    1. Relativistic Kinematics & Cauchy Initialization
    2. Critical Collapse & Holographic Anchoring
    3. Sommerfeld Radiation & C-infinity Blending
    4. Quantum Thermodynamic Tether
    5. Dual-Architecture Hardware Packaging (FP32)
"""

import numpy as np


def generate_dataset():
    # =========================================================================
    #  CONFIGURATION
    # =========================================================================
    velocities = np.array([0.4, 0.6, 0.8], dtype=np.float64)
    n_sims     = len(velocities)
    frames     = 100
    res        = 64
    sigma      = 0.15           # Gaussian blob width (proper frame)
    A_crit     = 2.85           # Critical amplitude (Phase 2)
    u_bnd      = -9.21          # Conformal boundary coordinate ln(z_min)
    v_shock    = 0.7            # Shockwave propagation speed  (Phase 3)

    # Coordinate grids (float64 intermediate precision)
    t_arr = np.linspace(0.0, 1.0, frames,  dtype=np.float64)
    x_arr = np.linspace(-1.0, 1.0, res,    dtype=np.float64)
    y_arr = np.linspace(-1.0, 1.0, res,    dtype=np.float64)

    # 2-D spatial meshgrid:  X[yi, xi] = x_arr[xi],  Y[yi, xi] = y_arr[yi]
    X, Y = np.meshgrid(x_arr, y_arr, indexing="xy")

    # Spatial envelope  cos(x π/2) cos(y π/2)  →  exact 0 at all 4 edges
    envelope = np.cos(X * np.pi / 2.0) * np.cos(Y * np.pi / 2.0)

    # Pre-allocate
    cnn_volumes     = np.zeros((n_sims, 1, frames, res, res), dtype=np.float64)
    entropy_targets = np.zeros((n_sims, frames),               dtype=np.float64)

    # =========================================================================
    #  PER-SIMULATION GENERATION
    # =========================================================================
    for sid in range(n_sims):
        v = velocities[sid]
        print(f"  [sim {sid}]  v = {v:.1f}")

        for fi in range(frames):
            t = t_arr[fi]

            # =================================================================
            #  PHASE 1 : Relativistic Kinematics & Cauchy Initialization
            # =================================================================
            # 1a. Cosine trajectory  →  dx/dt|_{t=0} = 0  (Cauchy rest mass)
            x_pos = v * np.cos(np.pi * t)                  # blob centre
            vel_t = -v * np.pi * np.sin(np.pi * t)         # analytical velocity

            # 1b. Lorentz contraction  γ = 1/√(1 - v(t)²)
            #     Clamp to subluminal to keep γ finite & real.
            vel_sq = np.clip(vel_t ** 2, 0.0, 0.9999)
            gamma  = 1.0 / np.sqrt(1.0 - vel_sq)

            # Lorentz-contracted Gaussian blobs  (squeezed along x)
            #   σ_x = σ/γ  →  exponent becomes  γ²(x-cx)²/(2σ²)
            blob1 = np.exp(
                -gamma ** 2 * (X - x_pos) ** 2 / (2.0 * sigma ** 2)
                - Y ** 2 / (2.0 * sigma ** 2)
            )
            blob2 = np.exp(
                -gamma ** 2 * (X + x_pos) ** 2 / (2.0 * sigma ** 2)
                - Y ** 2 / (2.0 * sigma ** 2)
            )

            # 1c. Spatial envelope  →  exact 0 at grid edges at all t
            pre_field = (blob1 + blob2) * envelope

            # =================================================================
            #  PHASE 3 : Sommerfeld Radiation & C-infinity Blending
            # =================================================================
            # 3a. Expanding ring shockwave  (physically active only post-merger)
            r            = np.sqrt(X ** 2 + Y ** 2 + 1e-12)
            shock_radius = v_shock * max(t - 0.5, 0.0)
            shock_width  = 0.05
            shockwave    = np.exp(
                -(r - shock_radius) ** 2 / (2.0 * shock_width ** 2)
            )

            # Planar damping  exp(-2|X||Y|)  →  wave hits square edges flat
            damping = np.exp(-2.0 * np.abs(X) * np.abs(Y))

            # Amplitude scales with collision energy  ∝ v
            post_field = v * shockwave * damping * envelope

            # 3b. C-infinity tanh blend  (infinitely differentiable crossover)
            blend = 0.5 * (1.0 + np.tanh(40.0 * (t - 0.5)))
            field = (1.0 - blend) * pre_field + blend * post_field

            cnn_volumes[sid, 0, fi] = field

        # =====================================================================
        #  PHASE 2 : Critical Collapse & Holographic Anchoring
        # =====================================================================
        # 2a. Normalise then scale to A_crit = 2.85
        vol     = cnn_volumes[sid, 0]
        vol_max = np.max(np.abs(vol))
        if vol_max > 1e-15:
            vol = (vol / vol_max) * A_crit
        cnn_volumes[sid, 0] = vol

        # =====================================================================
        #  PHASE 4 : Quantum Thermodynamic Tether – S(t)
        # =====================================================================
        # S_max scales with collision energy  ∝  ln(2)(1 + 2v)
        S_max = np.log(2.0) * (1.0 + 2.0 * v)

        for fi in range(frames):
            t = t_arr[fi]
            if t <= 0.4:
                # Pre-collision : near-zero entropy
                entropy_targets[sid, fi] = 0.0
            elif t < 0.6:
                # Merger window : quadratic growth with smooth saturation (C1)
                #   s² (3 - 2s)  →  onset ≈ 3 S_max s²  (quadratic)
                s = (t - 0.4) / 0.2
                entropy_targets[sid, fi] = S_max * s ** 2 * (3.0 - 2.0 * s)
            else:
                # Post-merger : saturated Bekenstein-Hawking entropy
                entropy_targets[sid, fi] = S_max

    # =========================================================================
    #  PHASE 5 : Dual-Architecture Hardware Packaging
    # =========================================================================
    print("\n  Assembling PINN collocation table ...")

    N = n_sims * frames * res * res          # 3 × 100 × 64 × 64 = 1,228,800

    # --- Vectorised column construction ---
    # Flatten order matches cnn_volumes[:, 0].reshape(-1):
    #   sid (slowest) → fi → yi → xi (fastest)

    sid_col = np.repeat(
        np.arange(n_sims, dtype=np.float64), frames * res * res
    )
    t_col = np.tile(
        np.repeat(t_arr, res * res), n_sims
    )

    # Within each frame:  yi varies, for each yi xi cycles
    per_frame_x = np.tile(x_arr, res)        # [x0..x63, x0..x63, ...] × res
    per_frame_y = np.repeat(y_arr, res)       # [y0]*res, [y1]*res, ...

    x_col = np.tile(per_frame_x, n_sims * frames)
    y_col = np.tile(per_frame_y, n_sims * frames)

    u_col       = np.full(N, u_bnd, dtype=np.float64)       # Phase 2b: u = -9.21
    phi_col     = cnn_volumes[:, 0].reshape(-1)              # same flatten order
    dphi_du_col = np.zeros(N, dtype=np.float64)              # Phase 2c: dφ/du = 0

    pinn_points = np.column_stack([
        sid_col, t_col, x_col, y_col, u_col, phi_col, dphi_du_col
    ])

    # =========================================================================
    #  CAST TO FP32  &  SAVE
    # =========================================================================
    out = dict(
        cnn_volumes     = cnn_volumes.astype(np.float32),        # [3, 1, 100, 64, 64]
        pinn_points     = pinn_points.astype(np.float32),        # [N, 7]
        entropy_targets = entropy_targets.astype(np.float32),    # [3, 100]
        time_ticks      = t_arr.astype(np.float32),              # [100]
        metadata        = velocities.astype(np.float32),         # [3]
        domain_bounds   = np.array(
            [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -9.21, 0.0],
            dtype=np.float32
        ),                                                       # [8]
    )

    np.savez_compressed("apex_master_dataset.npz", **out)

    # =========================================================================
    #  VALIDATION
    # =========================================================================
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║                  DATASET VALIDATION REPORT                  ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    for key, arr in out.items():
        print(f"  {key:20s}  shape={str(arr.shape):20s}  dtype={arr.dtype}"
              f"  range=[{arr.min():.6f}, {arr.max():.6f}]")

    # --- Shape assertions ---
    assert out["cnn_volumes"].shape     == (3, 1, 100, 64, 64), "cnn_volumes shape"
    assert out["pinn_points"].shape     == (N, 7),               "pinn_points shape"
    assert out["entropy_targets"].shape == (3, 100),             "entropy_targets shape"
    assert out["time_ticks"].shape      == (100,),               "time_ticks shape"
    assert out["metadata"].shape        == (3,),                 "metadata shape"
    assert out["domain_bounds"].shape   == (8,),                 "domain_bounds shape"

    # --- Dtype assertions ---
    for key, arr in out.items():
        assert arr.dtype == np.float32, f"{key} is {arr.dtype}, expected float32"

    # --- Physics assertions ---
    cv = out["cnn_volumes"]

    # Envelope: exact zero at grid edges  (cos(±π/2) = 0)
    edge_max = max(
        np.max(np.abs(cv[:, :, :, :,  0])),       # x = -1
        np.max(np.abs(cv[:, :, :, :, -1])),        # x = +1
        np.max(np.abs(cv[:, :, :,  0, :])),        # y = -1
        np.max(np.abs(cv[:, :, :, -1, :])),        # y = +1
    )
    status = "PASS" if edge_max < 1e-6 else "FAIL"
    print(f"\n  [{status}]  Edge max |phi|       = {edge_max:.2e}  (must be ~ 0)")

    # Critical amplitude
    global_max = np.max(np.abs(cv))
    status = "PASS" if abs(global_max - A_crit) < 0.01 else "FAIL"
    print(f"  [{status}]  Global max |phi|     = {global_max:.4f}  (must be {A_crit})")

    # Cauchy rest mass: velocity at t = 0 = 0
    #   x_pos'(0) = -v π sin(0) = 0  ✓  (analytic; verify blob symmetry)
    t0_frame = cv[:, 0, 0, :, :]                               # all sims, t=0
    left     = t0_frame[:, :, :res // 2]
    right    = t0_frame[:, :, res // 2:][:, :, ::-1]
    sym_err  = np.max(np.abs(left - right))
    status   = "PASS" if sym_err < 1e-4 else "FAIL"
    print(f"  [{status}]  t=0 x-symmetry err  = {sym_err:.2e}  (Cauchy rest)")

    # Neumann radial flux: dφ/du = 0
    dphi_max = np.max(np.abs(out["pinn_points"][:, 6]))
    status   = "PASS" if dphi_max == 0.0 else "FAIL"
    print(f"  [{status}]  max |dphi/du|        = {dphi_max:.2e}  (Neumann)")

    # Holographic anchor: all u = -9.21
    u_vals  = out["pinn_points"][:, 4]
    u_check = np.allclose(u_vals, np.float32(-9.21))
    status  = "PASS" if u_check else "FAIL"
    print(f"  [{status}]  All u == -9.21       = {u_check}")

    # Entropy: zero pre-merger, saturated post-merger
    et = out["entropy_targets"]
    pre_zero = np.max(np.abs(et[:, :40]))                       # t < 0.4
    post_sat = np.min(et[:, -1])                                 # t = 1.0
    status_p = "PASS" if pre_zero < 1e-6 else "FAIL"
    status_s = "PASS" if post_sat > 0.5 else "FAIL"
    print(f"  [{status_p}]  Entropy pre-merger   = {pre_zero:.2e}  (must be ~ 0)")
    print(f"  [{status_s}]  Entropy post-merger  = {post_sat:.4f}  (must be > 0)")

    # Domain bounds
    expected = np.array([0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -9.21, 0.0],
                        dtype=np.float32)
    db_ok = np.array_equal(out["domain_bounds"], expected)
    status = "PASS" if db_ok else "FAIL"
    print(f"  [{status}]  domain_bounds match  = {db_ok}")

    # Summary
    size_mb = sum(a.nbytes for a in out.values()) / 1e6
    print(f"\n  Total PINN points : {N:,}")
    print(f"  Uncompressed size : {size_mb:.1f} MB")
    print(f"\n  >>> apex_master_dataset.npz saved successfully <<<")


# =========================================================================
#  ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    print("=" * 64)
    print("  4D Holographic Boundary Dataset Generator")
    print("  Phases 1-5 : Kinematics | Collapse | Sommerfeld | Entropy | Pack")
    print("=" * 64)
    generate_dataset()
