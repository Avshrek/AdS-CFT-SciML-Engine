"""
benchmark_speedup.py — FNO vs LU Solver Speed Benchmark
=========================================================

Measures inference time for:
  * Exact LU-factorised Laplace solver  (CPU-only)
  * Neural-AdS FNO3d                    (CPU and/or GPU)

across different batch sizes (1, 10, 50, 100) and prints a publication-
quality timing table.

Usage
-----
    python benchmark_speedup.py
    python benchmark_speedup.py --n_warmup 5 --n_repeat 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from fno_architectures import FNO3d


GRID_SIZE  = 64
TIME_STEPS = 20


def parse_args():
    p = argparse.ArgumentParser(
        description="FNO vs LU solver speed benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", type=str,
                    default=os.path.join("models", "collision_rigorous.pth"))
    p.add_argument("--data_dir", type=str, default="data_collision_5k",
                    help="Dataset dir (for norm stats + real inputs)")
    p.add_argument("--batch_sizes", type=int, nargs="+",
                    default=[1, 10, 50, 100])
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_repeat", type=int, default=10)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LU solver (same as data generation)
# ─────────────────────────────────────────────────────────────────────────────

def build_laplace_system(n: int):
    N = n * n
    main = -4.0 * np.ones(N)
    side = np.ones(N - 1)
    side[np.arange(1, N) % n == 0] = 0
    ud = np.ones(N - n)
    A = sp.diags([main, side, side, ud, ud],
                 [0, -1, 1, -n, n], shape=(N, N), format="lil")
    grid = np.arange(N).reshape(n, n)
    bdy = np.unique(np.concatenate((
        grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1])))
    for i in bdy:
        A[i, :] = 0; A[i, i] = 1
    return spla.factorized(A.tocsc()), grid[0, :]


def lu_solve_batch(boundaries: np.ndarray, solve_fn, top_row):
    """Solve for a batch of boundaries. Returns (B, T, X, Z)."""
    B, T, X = boundaries.shape
    results = np.empty((B, T, X, X), dtype=np.float32)
    b = np.zeros(X * X, dtype=np.float64)
    for i in range(B):
        for t in range(T):
            b[:] = 0.0
            b[top_row] = boundaries[i, t]
            results[i, t] = solve_fn(b).reshape(X, X).astype(np.float32)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FNO inference
# ─────────────────────────────────────────────────────────────────────────────

def construct_batch_input(boundaries, stats, Z):
    B, T, X = boundaries.shape
    bdy_n = (boundaries - stats["x_mean"]) / (stats["x_std"] + 1e-8)
    wave  = np.tile(bdy_n[:, :, :, None], (1, 1, 1, Z))
    tc    = np.linspace(0, 1, T, dtype=np.float32)
    time  = np.broadcast_to(tc[None, :, None, None], (B, T, X, Z)).copy()
    zc    = np.linspace(0, 1, Z, dtype=np.float32)
    depth = np.broadcast_to(zc[None, None, None, :], (B, T, X, Z)).copy()
    return np.stack([wave, time, depth], axis=1).astype(np.float32)


@torch.no_grad()
def fno_infer_batch(model, volume_np, device):
    x = torch.from_numpy(volume_np).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    pred = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data for real boundaries
    bdy_path = os.path.join(args.data_dir, "bdy_collision.npy")
    if not os.path.isfile(bdy_path):
        bdy_path = os.path.join("data_collision_master", "bdy_collision.npy")
    bdy_all = np.load(bdy_path).astype(np.float32)

    stats = dict(
        x_mean=float(bdy_all.mean()), x_std=float(bdy_all.std()),
        y_mean=0.0, y_std=1.0,
    )

    # LU solver setup
    print("Pre-factorising LU solver ...")
    solve_fn, top_row = build_laplace_system(GRID_SIZE)

    # FNO model
    print(f"Loading FNO3d from {args.model_path} ...")
    model = FNO3d(modes1=args.modes, modes2=args.modes, modes3=args.modes,
                  width=args.width, n_layers=args.n_layers,
                  in_channels=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ── Benchmark ─────────────────────────────────────────────────────
    max_b = max(args.batch_sizes)
    boundaries = bdy_all[:max_b]
    volumes = construct_batch_input(boundaries, stats, GRID_SIZE)

    results = []
    header = f"{'Batch':>6} | {'LU Solver (s)':>14} | {'FNO (s)':>14} | {'Speedup':>10}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(f"  SPEED BENCHMARK — LU Solver vs Neural-AdS FNO3d")
    print(f"  Grid: {TIME_STEPS}x{GRID_SIZE}x{GRID_SIZE}  |  Device: {device}")
    print(f"  Warmup: {args.n_warmup}  Repeats: {args.n_repeat}")
    print(f"{sep}")
    print(header)
    print(sep)

    for bs in args.batch_sizes:
        bdy_b = boundaries[:bs]
        vol_b = volumes[:bs]

        # Warmup
        for _ in range(args.n_warmup):
            lu_solve_batch(bdy_b[:1], solve_fn, top_row)
            fno_infer_batch(model, vol_b[:1], device)

        # Time LU solver
        lu_times = []
        for _ in range(args.n_repeat):
            t0 = time.perf_counter()
            lu_solve_batch(bdy_b, solve_fn, top_row)
            lu_times.append(time.perf_counter() - t0)
        lu_mean = np.mean(lu_times)

        # Time FNO
        fno_times = []
        for _ in range(args.n_repeat):
            t0 = time.perf_counter()
            fno_infer_batch(model, vol_b, device)
            fno_times.append(time.perf_counter() - t0)
        fno_mean = np.mean(fno_times)

        speedup = lu_mean / (fno_mean + 1e-12)
        results.append((bs, lu_mean, fno_mean, speedup))
        print(f"{bs:>6} | {lu_mean:>14.4f} | {fno_mean:>14.4f} | {speedup:>9.1f}x")

    print(sep)

    # Per-sample times for batch=1
    bs1 = [r for r in results if r[0] == 1]
    if bs1:
        _, lu1, fno1, sp1 = bs1[0]
        print(f"\n  Per-sample inference:")
        print(f"    LU Solver: {lu1*1000:.1f} ms")
        print(f"    FNO3d:     {fno1*1000:.1f} ms")
        print(f"    Speedup:   {sp1:.1f}x")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "benchmark_speedup.txt")
    with open(out_path, "w") as f:
        f.write(f"Speed Benchmark — LU Solver vs Neural-AdS FNO3d\n")
        f.write(f"Device: {device}\n")
        if device.type == "cuda":
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Grid: {TIME_STEPS}x{GRID_SIZE}x{GRID_SIZE}\n\n")
        f.write(f"{header}\n{sep}\n")
        for bs, lu, fno, sp in results:
            f.write(f"{bs:>6} | {lu:>14.4f} | {fno:>14.4f} | {sp:>9.1f}x\n")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
