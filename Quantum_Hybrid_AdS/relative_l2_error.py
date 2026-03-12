"""
Relative L2 Error Metric
─────────────────────────
Per-sample Relative L2 Error for comparing predicted bulk geometry
against ground truth, used in FNO vs Hybrid Quantum Neural Network
benchmarks for the AdS/CFT holographic bulk reconstruction paper.

    RelL2(i) = ||y_true[i] - y_pred[i]||_2  /  ||y_true[i]||_2

Returns the mean over the batch, expressed as a percentage.

Supports both PyTorch tensors and NumPy arrays transparently.
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def relative_l2_error(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    Compute the mean Relative L2 Error (%) over a batch.

    Parameters
    ----------
    y_true : ndarray or Tensor, shape [B, ...]
        Ground-truth bulk geometry.
    y_pred : ndarray or Tensor, shape [B, ...]
        Model prediction (same shape as y_true).
    eps : float
        Small constant to guard against division by zero
        for near-vacuum truth samples.

    Returns
    -------
    float
        Mean Relative L2 Error as a percentage (0–100 scale).
    """
    # ── Detect backend ──────────────────────────────────────────────
    is_torch = _HAS_TORCH and torch.is_tensor(y_true)

    if is_torch:
        # Detach from graph & move to CPU for safe numerics
        yt = y_true.detach().float()
        yp = y_pred.detach().float()

        batch = yt.shape[0]
        # Flatten every dimension except the batch dim → [B, -1]
        yt_flat = yt.reshape(batch, -1)
        yp_flat = yp.reshape(batch, -1)

        diff_norm = torch.linalg.norm(yt_flat - yp_flat, dim=1)   # [B]
        true_norm = torch.linalg.norm(yt_flat, dim=1)             # [B]

        rel_l2 = diff_norm / (true_norm + eps)                    # [B]
        mean_pct = rel_l2.mean().item() * 100.0

    else:
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)

        batch = yt.shape[0]
        yt_flat = yt.reshape(batch, -1)
        yp_flat = yp.reshape(batch, -1)

        diff_norm = np.linalg.norm(yt_flat - yp_flat, axis=1)    # [B]
        true_norm = np.linalg.norm(yt_flat, axis=1)              # [B]

        rel_l2 = diff_norm / (true_norm + eps)                   # [B]
        mean_pct = float(rel_l2.mean()) * 100.0

    return mean_pct


def print_relative_l2(y_true, y_pred, label: str = "Model", eps: float = 1e-8):
    """Convenience wrapper that computes and prints the result."""
    pct = relative_l2_error(y_true, y_pred, eps=eps)
    print(f"[{label}]  Relative L2 Error  =  {pct:.4f} %")
    return pct


# ── Quick self-test ─────────────────────────────────────────────────
if __name__ == "__main__":
    # NumPy sanity check  (B=4, T=10, H=32, W=32)
    rng = np.random.default_rng(42)
    truth = rng.standard_normal((4, 10, 32, 32))
    pred  = truth + 0.05 * rng.standard_normal(truth.shape)   # ~5 % noise

    print_relative_l2(truth, pred, label="NumPy test")

    if _HAS_TORCH:
        t_truth = torch.tensor(truth)
        t_pred  = torch.tensor(pred)
        print_relative_l2(t_truth, t_pred, label="Torch test")
