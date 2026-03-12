"""
generate_dual_source_data.py — Dual-Source Collision Dataset Generator
=======================================================================

Generates a spatiotemporal holographic dataset where each boundary is the
superposition of two independent sine-wave sources:

    Φ_total(x, t) = A₁·sin(k₁·x + ω₁·t) + A₂·sin(k₂·x + ω₂·t + φ₂)

with randomised parameters per sample.  The 2-D bulk is solved via the
LU-factorised Laplace operator (∇²Φ = 0, Dirichlet BCs) at every time step.

Output shapes
-------------
    bdy_collision.npy   →  (N, T, X)       = (1000, 20, 64)
    bulk_collision.npy  →  (N, T, X, Z)    = (1000, 20, 64, 64)

Usage
-----
    python generate_dual_source_data.py
    python generate_dual_source_data.py --n_samples 500 --seed 42
    python generate_dual_source_data.py --help
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dual-source collision dataset generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_samples",  type=int, default=1000)
    p.add_argument("--grid_size",  type=int, default=64)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--save_dir",   type=str, default="data_collision_master")
    p.add_argument("--seed",       type=int, default=None,
                    help="Optional RNG seed for reproducibility")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Laplace solver (same LU-factorised sparse system as Phase 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────

def build_laplace_system(n: int):
    """Construct and LU-factorise the 2-D Laplacian with Dirichlet BCs.

    Returns
    -------
    solve : callable — ``solve(b)`` returns the solution vector.
    top_row : 1-D array of flattened indices for the top boundary row.
    """
    N = n * n
    main_diag = -4.0 * np.ones(N)
    side_diag = np.ones(N - 1)
    side_diag[np.arange(1, N) % n == 0] = 0
    up_down_diag = np.ones(N - n)

    A = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -n, n],
        shape=(N, N), format="lil",
    )

    grid_idx = np.arange(N).reshape(n, n)
    all_bdy = np.unique(np.concatenate((
        grid_idx[0, :], grid_idx[-1, :],
        grid_idx[:, 0], grid_idx[:, -1],
    )))
    for idx in all_bdy:
        A[idx, :] = 0
        A[idx, idx] = 1

    return spla.factorized(A.tocsc()), grid_idx[0, :]


# ─────────────────────────────────────────────────────────────────────────────
# Dual-source boundary generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dual_boundary(
    x: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Φ(x,t) = A₁·sin(k₁x + ω₁t) + A₂·sin(k₂x + ω₂t + φ₂).

    Parameter ranges
    ~~~~~~~~~~~~~~~~
    k   : 1.0 – 8.0
    ω   : 0.5 – 4.0
    A   : 0.1 – 0.75  (per source, so total peak ≈ 1.5)
    φ₂  : 0   – 2π

    Returns ``(T, X)`` boundary wave.
    """
    k1    = rng.uniform(1.0, 8.0)
    k2    = rng.uniform(1.0, 8.0)
    omega1 = rng.uniform(0.5, 4.0)
    omega2 = rng.uniform(0.5, 4.0)
    amp1   = rng.uniform(0.1, 0.75)
    amp2   = rng.uniform(0.1, 0.75)
    phi2   = rng.uniform(0.0, 2 * np.pi)

    t_col, x_row = t[:, np.newaxis], x[np.newaxis, :]   # (T,1), (1,X)

    wave = (amp1 * np.sin(k1 * x_row + omega1 * t_col)
          + amp2 * np.sin(k2 * x_row + omega2 * t_col + phi2))
    return wave.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate(args: argparse.Namespace) -> None:
    N = args.n_samples
    G = args.grid_size
    T = args.time_steps

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    x_grid = np.linspace(0.0, 2 * np.pi, G, dtype=np.float64)
    t_grid = np.linspace(0.0, 2 * np.pi, T, dtype=np.float64)

    print(f"🚀 Building Sparse Laplacian ({G}×{G}) & LU Factorisation …")
    solve, top_row_idx = build_laplace_system(G)
    print("   ✔ Solver ready.\n")

    all_bdy  = np.empty((N, T, G),    dtype=np.float32)
    all_bulk = np.empty((N, T, G, G), dtype=np.float32)

    print(f"⚡ Generating {N} dual-source collision universes  "
          f"({T} time-steps × {G}×{G} bulk each) …\n")

    for i in tqdm(range(N), desc="Samples", unit="sample", ncols=88):
        bdy_seq = generate_dual_boundary(x_grid, t_grid, rng)  # (T, X)
        all_bdy[i] = bdy_seq

        b = np.zeros(G * G, dtype=np.float64)
        for t_idx in range(T):
            b[:] = 0.0
            b[top_row_idx] = bdy_seq[t_idx]
            sol = solve(b).reshape(G, G)
            all_bulk[i, t_idx] = sol.astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────
    bdy_path  = os.path.join(args.save_dir, "bdy_collision.npy")
    bulk_path = os.path.join(args.save_dir, "bulk_collision.npy")

    print(f"\n💾 Saving to '{args.save_dir}/' …")
    print(f"   bdy_collision.npy   →  {all_bdy.shape}")
    print(f"   bulk_collision.npy  →  {all_bulk.shape}")

    np.save(bdy_path,  all_bdy)
    np.save(bulk_path, all_bulk)

    bdy_mb  = os.path.getsize(bdy_path)  / (1024 ** 2)
    bulk_mb = os.path.getsize(bulk_path) / (1024 ** 2)

    print(f"\n   📦 bdy_collision.npy  : {bdy_mb:,.1f} MB")
    print(f"   📦 bulk_collision.npy : {bulk_mb:,.1f} MB")
    print(f"   📦 Total              : {bdy_mb + bulk_mb:,.1f} MB")
    print("\n✅ Dual-source collision dataset generation complete.")


if __name__ == "__main__":
    generate(parse_args())
