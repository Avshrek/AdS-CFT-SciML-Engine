"""
generate_time_physics.py — Spatiotemporal Data Generator for Unified Neural-AdS Phase 2
=========================================================================================

AdS/CFT Physics Context
------------------------
Phase 1 solved a *static* holographic map  Φ_boundary(x) → Φ_bulk(x, z).
Phase 2 promotes this to a *time-dependent* (2+1-D) map:

    Φ_boundary(x, t)  →  Φ_bulk(x, z, t)

At each discrete time step t_k the boundary wave evolves via a superposition
of standing / travelling modes with random frequencies ω and wavenumbers k.
The interior bulk is then obtained by solving the 2-D Laplace equation
∇²Φ = 0 (Dirichlet BVP) independently for every t_k, treating the
time-varying boundary as a sequence of quasi-static snapshots — the standard
adiabatic approximation valid when ω << c/L_AdS.

Output shapes
-------------
    boundary_time.npy  →  (N_samples, T, X)       = (1000, 20, 64)
    bulk_time.npy      →  (N_samples, T, X, Z)    = (1000, 20, 64, 64)

Usage
-----
    python generate_time_physics.py
    python generate_time_physics.py --n_samples 500 --time_steps 10
    python generate_time_physics.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 — Spatiotemporal holographic data generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_samples",   type=int, default=1000,
                    help="Number of independent boundary configurations")
    p.add_argument("--grid_size",   type=int, default=64,
                    help="Spatial resolution (X = Z = grid_size)")
    p.add_argument("--time_steps",  type=int, default=20,
                    help="Number of discrete time steps T")
    p.add_argument("--save_dir",    type=str, default="data_holography_time",
                    help="Output directory for .npy files")
    p.add_argument("--seed",        type=int, default=None,
                    help="Optional RNG seed for reproducibility")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Laplace solver setup  (reuses the Phase 1 finite-difference pattern)
# ─────────────────────────────────────────────────────────────────────────────

def build_laplace_system(n: int):
    """Construct and LU-factorise the 2-D Laplace operator with Dirichlet BCs.

    Parameters
    ----------
    n : spatial grid size (the operator acts on an n×n grid).

    Returns
    -------
    solve : callable — ``solve(b)`` returns the solution vector directly.
    top_row_indices : 1-D array of flattened indices for the top boundary
                      (row 0 of the n×n grid), where the input wave is applied.
    """
    N = n * n

    # --- 5-point Laplacian stencil -------------------------------------------
    main_diag = -4.0 * np.ones(N)
    side_diag = np.ones(N - 1)
    side_diag[np.arange(1, N) % n == 0] = 0          # kill row-wrapping
    up_down_diag = np.ones(N - n)

    A = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -n, n],
        shape=(N, N),
        format="lil",
    )

    # --- Dirichlet BCs: top = input wave; left, right, bottom = 0 ------------
    grid_idx = np.arange(N).reshape(n, n)
    top_row      = grid_idx[0, :]
    bottom_row   = grid_idx[-1, :]
    left_col     = grid_idx[:, 0]
    right_col    = grid_idx[:, -1]
    all_bdy = np.unique(np.concatenate((top_row, bottom_row, left_col, right_col)))

    for idx in all_bdy:
        A[idx, :] = 0
        A[idx, idx] = 1

    # --- LU factorisation (O(N²) once, then each solve is O(N)) --------------
    solve = spla.factorized(A.tocsc())
    return solve, top_row


# ─────────────────────────────────────────────────────────────────────────────
# Boundary wave generator (time-dependent)
# ─────────────────────────────────────────────────────────────────────────────

def generate_boundary_sequence(
    x: np.ndarray,
    t_steps: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a time-evolving boundary wave as a superposition of modes.

    The wave at time t is:

        Φ(x, t) = Σ_i  A_i · sin(k_i · x  +  φ_i  +  ω_i · t)

    where k (wavenumber), φ (phase), ω (angular frequency), and A (amplitude)
    are randomly sampled per mode.  This yields a rich mixture of standing and
    travelling wave patterns across the time axis.

    Parameters
    ----------
    x       : spatial grid, shape ``(grid_size,)``
    t_steps : normalised time array, shape ``(T,)``
    rng     : NumPy Generator for reproducible randomness

    Returns
    -------
    boundary : ``(T, grid_size)`` — boundary wave at each time step.
    """
    n_modes = rng.integers(2, 5)                     # 2–4 superposed modes

    k     = rng.integers(1, 8, size=n_modes)                 # wavenumbers
    phi   = rng.uniform(0.0, 2 * np.pi, size=n_modes)       # spatial phases
    omega = rng.uniform(0.5, 4.0, size=n_modes)             # temporal freq
    amp   = rng.uniform(0.4, 1.2, size=n_modes)             # amplitudes

    # Vectorised: (T, 1) vs (n_modes,) broadcasts → (T, n_modes)
    # then (T, n_modes, 1) · (1, 1, X) → (T, n_modes, X) → sum → (T, X)
    t_col = t_steps[:, np.newaxis, np.newaxis]               # (T, 1, 1)
    x_row = x[np.newaxis, np.newaxis, :]                     # (1, 1, X)
    k_    = k[np.newaxis, :, np.newaxis]                     # (1, M, 1)
    phi_  = phi[np.newaxis, :, np.newaxis]
    omega_= omega[np.newaxis, :, np.newaxis]
    amp_  = amp[np.newaxis, :, np.newaxis]

    modes = amp_ * np.sin(k_ * x_row + phi_ + omega_ * t_col)  # (T, M, X)
    boundary = modes.sum(axis=1)                                # (T, X)

    # Global amplitude jitter for diversity
    boundary *= rng.uniform(0.8, 1.3)

    return boundary.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate(args: argparse.Namespace) -> None:
    N        = args.n_samples
    G        = args.grid_size
    T        = args.time_steps
    save_dir = args.save_dir

    rng = np.random.default_rng(args.seed)
    os.makedirs(save_dir, exist_ok=True)

    # ── Spatial / temporal grids ──────────────────────────────────────────
    x_grid = np.linspace(0.0, 2 * np.pi, G, dtype=np.float64)
    t_grid = np.linspace(0.0, 2 * np.pi, T, dtype=np.float64)  # one full period

    # ── Solver setup (factorise once) ─────────────────────────────────────
    print(f"🚀 Building Sparse Laplacian ({G}×{G}) & LU Factorisation …")
    solve, top_row_idx = build_laplace_system(G)
    print("   ✔ Solver ready.\n")

    # ── Pre-allocate output arrays ────────────────────────────────────────
    all_boundary = np.empty((N, T, G),    dtype=np.float32)
    all_bulk     = np.empty((N, T, G, G), dtype=np.float32)

    print(f"⚡ Generating {N} spatiotemporal universes  "
          f"({T} time-steps × {G}×{G} bulk each) …\n")

    for i in tqdm(range(N), desc="Samples", unit="sample", ncols=88):
        # --- Time-evolving boundary for this sample --------------------------
        bdy_seq = generate_boundary_sequence(x_grid, t_grid, rng)  # (T, X)
        all_boundary[i] = bdy_seq

        # --- Solve the 2-D Laplace BVP at every time step --------------------
        b = np.zeros(G * G, dtype=np.float64)                # reusable RHS

        for t in range(T):
            b[:] = 0.0
            b[top_row_idx] = bdy_seq[t]                      # apply boundary
            sol = solve(b).reshape(G, G)
            all_bulk[i, t] = sol.astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────
    bdy_path  = os.path.join(save_dir, "boundary_time.npy")
    bulk_path = os.path.join(save_dir, "bulk_time.npy")

    print(f"\n💾 Saving datasets to '{save_dir}/' …")
    print(f"   boundary_time.npy  →  {all_boundary.shape}")
    print(f"   bulk_time.npy      →  {all_bulk.shape}")

    np.save(bdy_path,  all_boundary)
    np.save(bulk_path, all_bulk)

    bdy_mb  = os.path.getsize(bdy_path)  / (1024 ** 2)
    bulk_mb = os.path.getsize(bulk_path) / (1024 ** 2)

    print(f"\n   📦 boundary_time.npy : {bdy_mb:,.1f} MB")
    print(f"   📦 bulk_time.npy     : {bulk_mb:,.1f} MB")
    print(f"   📦 Total             : {bdy_mb + bulk_mb:,.1f} MB")
    print("\n✅ Phase 2 data generation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate(parse_args())
