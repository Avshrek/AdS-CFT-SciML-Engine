"""
train_twophase_quantum.py -- Three-Phase Quantum Training Pipeline
==================================================================

Phase 1: Classical Foundation
    Train ClassicalAdS (10-dim bottleneck) with physics-aware loss.
    Teaches encoder+decoder stable compression before quantum injection.

Phase 2: Quantum Fine-Tuning (Differential LR)
    Transfer encoder+decoder weights into HybridQuantumAdS.
    Freeze encoder. Quantum layer trains at high LR, decoder at micro-LR.

Phase 3: Hardware-Aware Noise Training
    Switch to default.mixed with DepolarizingChannel to simulate
    IBM Torino gate errors. Forces decoder to learn noise correction.

Physics-Aware Loss:
    L = MSE + lambda_s * L_spectral + lambda_g * L_gradient + lambda_b * L_boundary

Usage:
    python train_twophase_quantum.py --phase all --data_dir /path/to/data
    python train_twophase_quantum.py --phase 1 --phase1_epochs 200
    python train_twophase_quantum.py --phase 2
    python train_twophase_quantum.py --phase 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-Phase Quantum Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", type=str, default="all",
                   choices=["1", "2", "3", "all"],
                   help="Which phase to run.")
    p.add_argument("--data_dir", type=str, default="data_collision_master",
                   help="Dataset directory.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--phase1_epochs", type=int, default=200)
    p.add_argument("--phase2_epochs", type=int, default=100)
    p.add_argument("--phase3_epochs", type=int, default=30)
    p.add_argument("--lambda_spectral", type=float, default=0.1)
    p.add_argument("--lambda_gradient", type=float, default=0.05)
    p.add_argument("--lambda_boundary", type=float, default=0.5)
    p.add_argument("--noise_rate", type=float, default=0.005,
                   help="Depolarizing noise per gate for Phase 3.")
    p.add_argument("--phase2_unfreeze_epoch", type=int, default=50,
                   help="Epoch at which decoder is unfrozen in Phase 2.")
    return p.parse_args()


# =====================================================================
# DATASET  (mirrors train_nature_quantum.py)
# =====================================================================

class UniverseDataset(Dataset):
    def __init__(self, data_dir):
        bdy_path = os.path.join(data_dir, "bdy_collision.npy")
        blk_path = os.path.join(data_dir, "bulk_collision.npy")

        # Fallback directories
        if not os.path.exists(bdy_path):
            for alt in ["data_collision_5k", "data_collision"]:
                alt_bdy = os.path.join(alt, "bdy_collision.npy")
                if os.path.exists(alt_bdy):
                    bdy_path = alt_bdy
                    blk_path = os.path.join(alt, "bulk_collision.npy")
                    break

        if not os.path.exists(bdy_path) or not os.path.exists(blk_path):
            print(f"[ERROR] Dataset not found: {bdy_path}")
            self.length = 0
            return

        print(f"[DATA] Loading from {os.path.dirname(bdy_path)} ...")
        self.bdy = np.load(bdy_path)
        self.blk = np.load(blk_path)
        self.length = self.bdy.shape[0]
        print(f"[DATA] {self.length} samples loaded")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        bdy = torch.from_numpy(self.bdy[idx]).float()         # (20, 64)
        blk = torch.from_numpy(self.blk[idx]).float()          # (20, 64, 64)

        # Tile boundary to match encoder input: (20, 64) -> (20, 64, 64)
        bdy_tiled = bdy.unsqueeze(-1).repeat(1, 1, 64)

        # Add channel dim: (1, 20, 64, 64)
        return bdy_tiled.unsqueeze(0), blk.unsqueeze(0)


# =====================================================================
# PHYSICS-AWARE LOSS FUNCTION
# =====================================================================

class PhysicsAwareLoss(nn.Module):
    """
    L = MSE + ls * L_spectral + lg * L_gradient + lb * L_boundary

    - L_spectral: 3D FFT magnitude difference across (Depth, Height, Width)
    - L_gradient: spatial gradient (finite difference) difference
    - L_boundary: AdS/CFT constraint -- bulk[:,:,0,:,:] must match boundary
    """

    def __init__(self, lambda_s=0.1, lambda_g=0.05, lambda_b=0.5):
        super().__init__()
        self.lambda_s = lambda_s
        self.lambda_g = lambda_g
        self.lambda_b = lambda_b
        self.mse = nn.MSELoss()

    def spectral_loss(self, pred, truth):
        """L2 in 3D Fourier space across dims (Depth, Height, Width)."""
        # pred, truth: (B, 1, D, H, W)
        fft_pred  = torch.fft.fftn(pred,  dim=(2, 3, 4))
        fft_truth = torch.fft.fftn(truth, dim=(2, 3, 4))
        return torch.mean(torch.abs(fft_pred - fft_truth) ** 2)

    def gradient_loss(self, pred, truth):
        """L2 of spatial gradient difference (finite differences)."""
        # Gradient along Depth (dim 2)
        grad_pred_d  = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        grad_truth_d = truth[:, :, 1:, :, :] - truth[:, :, :-1, :, :]
        loss_d = torch.mean((grad_pred_d - grad_truth_d) ** 2)

        # Gradient along Height (dim 3)
        grad_pred_h  = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        grad_truth_h = truth[:, :, :, 1:, :] - truth[:, :, :, :-1, :]
        loss_h = torch.mean((grad_pred_h - grad_truth_h) ** 2)

        # Gradient along Width (dim 4)
        grad_pred_w  = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        grad_truth_w = truth[:, :, :, :, 1:] - truth[:, :, :, :, :-1]
        loss_w = torch.mean((grad_pred_w - grad_truth_w) ** 2)

        return (loss_d + loss_h + loss_w) / 3.0

    def boundary_loss(self, pred, boundary_input):
        """
        AdS/CFT boundary constraint: the depth=0 slice of the predicted
        bulk must match the input CFT boundary data.

        pred:           (B, 1, 20, 64, 64) -- predicted bulk
        boundary_input: (B, 1, 20, 64, 64) -- tiled boundary (all depth
                        slices are identical, so any slice works as truth)
        """
        # Extract outermost depth slice: pred[:, :, 0, :, :] -> (B, 1, 64, 64)
        # This corresponds to the first "time/depth" plane (z = 0)
        pred_boundary = pred[:, :, 0, :, :]

        # The boundary input is tiled identically across the last dim,
        # so slice any depth: boundary_input[:, :, 0, :, :]
        # But the actual raw boundary is the (20, 64) signal tiled across width.
        # We compare the first spatial slice of the prediction to the input.
        truth_boundary = boundary_input[:, :, 0, :, :]

        return torch.mean((pred_boundary - truth_boundary) ** 2)

    def forward(self, pred, truth, boundary_input):
        l_mse = self.mse(pred, truth)
        l_spec = self.spectral_loss(pred, truth)
        l_grad = self.gradient_loss(pred, truth)
        l_bdy  = self.boundary_loss(pred, boundary_input)

        total = (l_mse
                 + self.lambda_s * l_spec
                 + self.lambda_g * l_grad
                 + self.lambda_b * l_bdy)

        return total, {
            "MSE": l_mse.item(),
            "Spectral": l_spec.item(),
            "Gradient": l_grad.item(),
            "Boundary": l_bdy.item(),
            "Total": total.item(),
        }


# =====================================================================
# RELATIVE L2 METRIC
# =====================================================================

def compute_rel_l2(pred, truth):
    """Per-sample Relative L2 %, averaged over batch."""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = truth.reshape(B, -1)
    diff_norm = torch.linalg.norm(p - t, dim=1)
    true_norm = torch.linalg.norm(t, dim=1)
    return (diff_norm / (true_norm + 1e-8)).mean().item() * 100.0


# =====================================================================
# CHECKPOINT UTILITIES
# =====================================================================

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_ckpt(tag, epoch, model, optimizer, scheduler, metrics):
    path = os.path.join(CHECKPOINT_DIR, f"twophase_{tag}_latest.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }, path)
    if epoch % 10 == 0:
        backup = os.path.join(CHECKPOINT_DIR, f"twophase_{tag}_epoch_{epoch}.pth")
        torch.save(torch.load(path, map_location="cpu"), backup)


def load_ckpt(tag, model, optimizer, scheduler, device):
    path = os.path.join(CHECKPOINT_DIR, f"twophase_{tag}_latest.pth")
    if not os.path.exists(path):
        return 1
    print(f"[RESUME] Loading {path} ...")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start = ckpt["epoch"] + 1
    print(f"[RESUME] Continuing from epoch {start}")
    return start


# =====================================================================
# PHASE 1: CLASSICAL FOUNDATION
# =====================================================================

def run_phase1(args, device):
    from classical_autoencoder import ClassicalAdS

    print("\n" + "=" * 72)
    print("  PHASE 1: Classical Foundation Training")
    print("=" * 72)

    model = ClassicalAdS(in_channels=1, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ClassicalAdS -- {n_params:,} parameters")

    criterion = PhysicsAwareLoss(
        lambda_s=args.lambda_spectral,
        lambda_g=args.lambda_gradient,
        lambda_b=args.lambda_boundary,
    )
    scaler = GradScaler()

    dataset = UniverseDataset(args.data_dir)
    if len(dataset) == 0:
        return
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=True,
        persistent_workers=True if os.name != "nt" else False,
    )

    epochs = args.phase1_epochs
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs,
        steps_per_epoch=len(loader), pct_start=0.2,
    )

    start_epoch = load_ckpt("phase1", model, optimizer, scheduler, device)

    # Metrics log
    metrics_path = "results/twophase_metrics.csv"
    os.makedirs("results", exist_ok=True)
    metrics_log = []
    if os.path.exists(metrics_path) and start_epoch > 1:
        metrics_log = pd.read_csv(metrics_path).to_dict("records")

    print(f"\n[PHASE 1] Training epochs {start_epoch} -> {epochs}")
    print(f"[LOSS] MSE + {args.lambda_spectral}*Spectral"
          f" + {args.lambda_gradient}*Gradient"
          f" + {args.lambda_boundary}*Boundary\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses = {"MSE": 0, "Spectral": 0, "Gradient": 0,
                        "Boundary": 0, "Total": 0}
        n_batches = 0

        for x_bdy, y_bulk in loader:
            x_bdy  = x_bdy.to(device, non_blocking=True)
            y_bulk = y_bulk.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                pred = model(x_bdy)
                loss, comps = criterion(pred, y_bulk, x_bdy)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += comps[k]
            n_batches += 1

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        # Validation Relative L2
        model.eval()
        with torch.no_grad():
            val_x, val_y = next(iter(loader))
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_pred = model(val_x)
            rel_l2 = compute_rel_l2(val_pred, val_y)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  P1 Epoch [{epoch:3d}/{epochs}] | "
              f"Total={epoch_losses['Total']:.6f} | "
              f"MSE={epoch_losses['MSE']:.6f} | "
              f"Spec={epoch_losses['Spectral']:.6f} | "
              f"Grad={epoch_losses['Gradient']:.6f} | "
              f"Bdy={epoch_losses['Boundary']:.6f} | "
              f"RelL2={rel_l2:.2f}% | "
              f"lr={lr_now:.2e}")

        row = {"Phase": 1, "Epoch": epoch, "RelL2_pct": rel_l2,
               "lr": lr_now, **epoch_losses}
        metrics_log.append(row)

        save_ckpt("phase1", epoch, model, optimizer, scheduler, row)
        pd.DataFrame(metrics_log).to_csv(metrics_path, index=False)

    # Save final Phase 1 weights
    out_path = "models/classical_phase1.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"\n[PHASE 1 DONE] Saved: {out_path}")
    return model


# =====================================================================
# PHASE 2: QUANTUM FINE-TUNING (Differential LR)
# =====================================================================

def run_phase2(args, device):
    from hybrid_autoencoder import HybridQuantumAdS
    from classical_autoencoder import ClassicalAdS

    print("\n" + "=" * 72)
    print("  PHASE 2: Quantum Fine-Tuning (Differential LR)")
    print("=" * 72)

    # --- Build quantum model ---
    model = HybridQuantumAdS(in_channels=1, out_channels=1).to(device)

    # --- Transfer classical weights ---
    phase1_path = "models/classical_phase1.pth"
    if os.path.exists(phase1_path):
        print(f"[TRANSFER] Loading Phase 1 weights from {phase1_path}")
        classical_model = ClassicalAdS()
        classical_model.load_state_dict(torch.load(phase1_path, map_location=device))
        transferable = classical_model.get_transferable_weights()

        # Load matching keys (encoder + decoder)
        model_sd = model.state_dict()
        transferred = 0
        for k, v in transferable.items():
            if k in model_sd:
                model_sd[k] = v
                transferred += 1
        model.load_state_dict(model_sd)
        print(f"[TRANSFER] {transferred} parameter tensors transferred")
        del classical_model
    else:
        print(f"[WARNING] No Phase 1 checkpoint found at {phase1_path}")
        print("[WARNING] Starting Phase 2 from scratch (not recommended)")

    model = model.to(device)

    # --- Freeze encoder ---
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("[FREEZE] Encoder frozen (will not train)")

    # --- Differential learning rates ---
    quantum_params = list(model.quantum_layer.parameters())
    decoder_params = (list(model.decoder_projection.parameters())
                      + list(model.decoder_conv.parameters()))

    optimizer = optim.AdamW([
        {"params": quantum_params,  "lr": 1e-3},    # aggressive quantum search
        {"params": decoder_params,  "lr": 1e-6},    # micro-adjustments only
    ], weight_decay=1e-4)

    print(f"[LR] Quantum layer: 1e-3 | Decoder: 1e-6")

    criterion = PhysicsAwareLoss(
        lambda_s=args.lambda_spectral,
        lambda_g=args.lambda_gradient,
        lambda_b=args.lambda_boundary,
    )
    scaler = GradScaler()

    dataset = UniverseDataset(args.data_dir)
    if len(dataset) == 0:
        return
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=True,
        persistent_workers=True if os.name != "nt" else False,
    )

    epochs = args.phase2_epochs
    scheduler = None  # manual LR management for differential rates

    start_epoch = load_ckpt("phase2", model, optimizer, scheduler, device)

    # Metrics
    metrics_path = "results/twophase_metrics.csv"
    os.makedirs("results", exist_ok=True)
    metrics_log = []
    if os.path.exists(metrics_path):
        metrics_log = pd.read_csv(metrics_path).to_dict("records")

    print(f"\n[PHASE 2] Training epochs {start_epoch} -> {epochs}")

    for epoch in range(start_epoch, epochs + 1):

        # --- Unfreeze decoder at specified epoch (with micro-LR) ---
        if epoch == args.phase2_unfreeze_epoch:
            print(f"\n[UNFREEZE] Epoch {epoch}: Decoder unfrozen at lr=5e-6")
            optimizer.param_groups[1]["lr"] = 5e-6

        model.train()
        epoch_losses = {"MSE": 0, "Spectral": 0, "Gradient": 0,
                        "Boundary": 0, "Total": 0}
        n_batches = 0

        for x_bdy, y_bulk in loader:
            x_bdy  = x_bdy.to(device, non_blocking=True)
            y_bulk = y_bulk.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                pred = model(x_bdy)
                loss, comps = criterion(pred, y_bulk, x_bdy)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k in epoch_losses:
                epoch_losses[k] += comps[k]
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x, val_y = next(iter(loader))
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_pred = model(val_x)
            rel_l2 = compute_rel_l2(val_pred, val_y)

        # Quantum entropy (every 5 epochs to save time)
        entropy = 0.0
        if epoch % 5 == 0:
            try:
                from quantum_architecture import QuantumEntropyExtractor
                with torch.no_grad():
                    latent = model.encoder(val_x)
                entropy = QuantumEntropyExtractor.extract_entropy(
                    latent, model.quantum_layer)
            except Exception:
                entropy = -1.0

        lr_q = optimizer.param_groups[0]["lr"]
        lr_d = optimizer.param_groups[1]["lr"]
        print(f"  P2 Epoch [{epoch:3d}/{epochs}] | "
              f"Total={epoch_losses['Total']:.6f} | "
              f"MSE={epoch_losses['MSE']:.6f} | "
              f"RelL2={rel_l2:.2f}% | "
              f"S={entropy:.4f} | "
              f"lr_q={lr_q:.1e} lr_d={lr_d:.1e}")

        row = {"Phase": 2, "Epoch": epoch, "RelL2_pct": rel_l2,
               "Entropy": entropy, "lr_q": lr_q, "lr_d": lr_d,
               **epoch_losses}
        metrics_log.append(row)

        save_ckpt("phase2", epoch, model, optimizer, scheduler, row)
        pd.DataFrame(metrics_log).to_csv(metrics_path, index=False)

    out_path = "models/quantum_phase2.pth"
    torch.save(model.state_dict(), out_path)
    print(f"\n[PHASE 2 DONE] Saved: {out_path}")
    return model


# =====================================================================
# PHASE 3: HARDWARE-AWARE NOISE TRAINING
# =====================================================================

def run_phase3(args, device):
    print("\n" + "=" * 72)
    print("  PHASE 3: Hardware-Aware Noise Training")
    print("=" * 72)

    # --- Build noisy quantum device + circuit ---
    import pennylane as qml

    n_qubits = 10
    noise_rate = args.noise_rate

    # Noisy simulator mimicking IBM Torino gate errors
    noisy_dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(noisy_dev, interface="torch", diff_method="backprop")
    def noisy_quantum_circuit(inputs, weights):
        """Same circuit as quantum_architecture.py but with depolarizing noise."""
        # Amplitude encoding
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # Entangling layers with noise injection after each layer
        n_layers = weights.shape[0]
        for layer_idx in range(n_layers):
            # Apply one layer of StronglyEntanglingLayers manually
            for qubit in range(n_qubits):
                qml.Rot(weights[layer_idx, qubit, 0],
                        weights[layer_idx, qubit, 1],
                        weights[layer_idx, qubit, 2],
                        wires=qubit)
            # CNOT entanglement pattern (same as StronglyEntanglingLayers)
            for qubit in range(n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1 + layer_idx) % n_qubits])

            # Inject depolarizing noise on every qubit after each layer
            for qubit in range(n_qubits):
                qml.DepolarizingChannel(noise_rate, wires=qubit)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # --- Build a noise-injecting quantum layer ---
    class NoisyQuantumLayer(nn.Module):
        def __init__(self, n_quantum_layers=3):
            super().__init__()
            self.q_weights = nn.Parameter(
                torch.randn(n_quantum_layers, n_qubits, 3) * 0.1
            )

        def forward(self, x):
            results = []
            for i in range(x.shape[0]):
                out = noisy_quantum_circuit(x[i], self.q_weights)
                results.append(torch.stack(out))
            return torch.stack(results).float()

    # --- Load Phase 2 model ---
    from hybrid_autoencoder import HybridQuantumAdS

    model = HybridQuantumAdS(in_channels=1, out_channels=1)
    phase2_path = "models/quantum_phase2.pth"
    if os.path.exists(phase2_path):
        print(f"[LOAD] Phase 2 weights from {phase2_path}")
        model.load_state_dict(torch.load(phase2_path, map_location="cpu"))
    else:
        print(f"[WARNING] No Phase 2 checkpoint at {phase2_path}")

    # Replace quantum layer with noisy version, transferring trained weights
    noisy_layer = NoisyQuantumLayer(n_quantum_layers=3)

    # Transfer learned quantum weights from Phase 2
    with torch.no_grad():
        src_weights = None
        for name, param in model.quantum_layer.named_parameters():
            if "weight" in name.lower():
                src_weights = param
                break
        if src_weights is not None and src_weights.shape == noisy_layer.q_weights.shape:
            noisy_layer.q_weights.copy_(src_weights)
            print("[TRANSFER] Quantum weights transferred to noisy layer")
        else:
            print("[WARNING] Could not transfer quantum weights -- shape mismatch")

    model.quantum_layer = noisy_layer
    model = model.to(device)

    print(f"[NOISE] Depolarizing rate: {noise_rate} per gate per qubit")

    # --- Freeze encoder ---
    for param in model.encoder.parameters():
        param.requires_grad = False

    # --- Differential LR ---
    quantum_params = list(model.quantum_layer.parameters())
    decoder_params = (list(model.decoder_projection.parameters())
                      + list(model.decoder_conv.parameters()))

    optimizer = optim.AdamW([
        {"params": quantum_params, "lr": 5e-4},
        {"params": decoder_params, "lr": 1e-6},
    ], weight_decay=1e-4)
    print(f"[LR] Quantum: 5e-4 | Decoder: 1e-6")

    criterion = PhysicsAwareLoss(
        lambda_s=args.lambda_spectral,
        lambda_g=args.lambda_gradient,
        lambda_b=args.lambda_boundary,
    )

    dataset = UniverseDataset(args.data_dir)
    if len(dataset) == 0:
        return
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=True,
        persistent_workers=True if os.name != "nt" else False,
    )

    epochs = args.phase3_epochs
    start_epoch = load_ckpt("phase3", model, optimizer, None, device)

    metrics_path = "results/twophase_metrics.csv"
    os.makedirs("results", exist_ok=True)
    metrics_log = []
    if os.path.exists(metrics_path):
        metrics_log = pd.read_csv(metrics_path).to_dict("records")

    print(f"\n[PHASE 3] Training epochs {start_epoch} -> {epochs}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses = {"MSE": 0, "Spectral": 0, "Gradient": 0,
                        "Boundary": 0, "Total": 0}
        n_batches = 0

        for x_bdy, y_bulk in loader:
            x_bdy  = x_bdy.to(device, non_blocking=True)
            y_bulk = y_bulk.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # No AMP for Phase 3 -- mixed device needs float32 for stability
            pred = model(x_bdy)
            loss, comps = criterion(pred, y_bulk, x_bdy)

            loss.backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += comps[k]
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x, val_y = next(iter(loader))
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_pred = model(val_x)
            rel_l2 = compute_rel_l2(val_pred, val_y)

        lr_q = optimizer.param_groups[0]["lr"]
        lr_d = optimizer.param_groups[1]["lr"]
        print(f"  P3 Epoch [{epoch:3d}/{epochs}] | "
              f"Total={epoch_losses['Total']:.6f} | "
              f"MSE={epoch_losses['MSE']:.6f} | "
              f"RelL2={rel_l2:.2f}% | "
              f"lr_q={lr_q:.1e} lr_d={lr_d:.1e}")

        row = {"Phase": 3, "Epoch": epoch, "RelL2_pct": rel_l2,
               "lr_q": lr_q, "lr_d": lr_d, **epoch_losses}
        metrics_log.append(row)

        save_ckpt("phase3", epoch, model, optimizer, None, row)
        pd.DataFrame(metrics_log).to_csv(metrics_path, index=False)

    # --- Save final model ---
    # Transfer noisy weights back to a clean HybridQuantumAdS for deployment
    final_model = HybridQuantumAdS(in_channels=1, out_channels=1)
    final_sd = final_model.state_dict()

    # Copy encoder + decoder from noise-trained model
    for k, v in model.state_dict().items():
        if k.startswith("quantum_layer"):
            continue  # handle separately
        if k in final_sd:
            final_sd[k] = v

    # Copy noisy quantum weights -> clean quantum layer
    # Noisy layer stores weights as q_weights, clean stores as q_layer.weights
    noisy_weights = model.quantum_layer.q_weights.detach().cpu()
    for k in final_sd:
        if "q_layer.weights" in k or "q_weights" in k:
            if final_sd[k].shape == noisy_weights.shape:
                final_sd[k] = noisy_weights
                print(f"[TRANSFER] Noisy quantum weights -> clean model key '{k}'")

    final_model.load_state_dict(final_sd)

    out_path = "models/QUANTUM_TWOPHASE_MODEL.pth"
    torch.save(final_model.state_dict(), out_path)
    print(f"\n[PHASE 3 DONE] Final model saved: {out_path}")
    return final_model


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[HARDWARE] Device: {device.type.upper()}")

    phases = [args.phase] if args.phase != "all" else ["1", "2", "3"]

    for phase in phases:
        t0 = time.time()
        if phase == "1":
            run_phase1(args, device)
        elif phase == "2":
            run_phase2(args, device)
        elif phase == "3":
            run_phase3(args, device)
        elapsed = time.time() - t0
        print(f"\n[TIME] Phase {phase} completed in {elapsed/60:.1f} minutes")

    print("\n" + "=" * 72)
    print("  ALL PHASES COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
