"""
train_kaggle_threephase.py
==========================
SELF-CONTAINED Three-Phase Quantum Training Pipeline for Kaggle GPU.

Upload this single file to a Kaggle notebook, attach your dataset as
    /kaggle/input/datasets/avshrek/collision-master-data/
and run:
    !pip install pennylane -q
    %run train_kaggle_threephase.py --phase all

Phase 1: Classical Foundation (200 epochs)
Phase 2: Quantum Fine-Tuning with Differential LR (100 epochs)
Phase 3: Hardware-Aware Noise Training (30 epochs)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import scipy.constants
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import the new AdS Physics Engine components
from ads_physics import (
    sample_ads_collocation,
    klein_gordon_loss,
    eikonal_loss,
    ryu_takayanagi_loss,
    get_loss_weights
)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-Phase Quantum Training for Kaggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", type=str, default="all",
                   choices=["1", "2", "3", "all"])
    p.add_argument("--data_dir", type=str,
                   default="/kaggle/input/datasets/avshrek/collision-master-data",
                   help="Dataset directory.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--phase1_epochs", type=int, default=200)
    p.add_argument("--phase2_epochs", type=int, default=100)
    p.add_argument("--phase3_epochs", type=int, default=30)
    p.add_argument("--lambda_spectral", type=float, default=1.0)
    p.add_argument("--lambda_gradient", type=float, default=0.05)
    p.add_argument("--lambda_boundary", type=float, default=0.1)
    p.add_argument("--noise_rate", type=float, default=0.005)
    p.add_argument("--phase2_unfreeze_epoch", type=int, default=50)
    return p.parse_args()


# =====================================================================
# DATASET
# =====================================================================

class UniverseDataset(Dataset):
    def __init__(self, data_dir):
        bdy_path = os.path.join(data_dir, "bdy_collision.npy")
        blk_path = os.path.join(data_dir, "bulk_collision.npy")

        if not os.path.exists(bdy_path) or not os.path.exists(blk_path):
            print(f"[ERROR] Dataset not found at {data_dir}")
            self.length = 0
            return

        print(f"[DATA] Loading from {data_dir} ...")
        self.bdy = np.load(bdy_path)
        self.blk = np.load(blk_path)
        self.length = self.bdy.shape[0]
        print(f"[DATA] {self.length} samples | "
              f"bdy={self.bdy.shape} | blk={self.blk.shape}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        bdy = torch.from_numpy(self.bdy[idx]).float()       # (20, 64)
        blk = torch.from_numpy(self.blk[idx]).float()        # (20, 64, 64)
        bdy_tiled = bdy.unsqueeze(-1).repeat(1, 1, 64)      # (20, 64, 64)
        return bdy_tiled.unsqueeze(0), blk.unsqueeze(0)      # (1, 20, 64, 64)


import math

# =====================================================================
# SIREN IMPLICIT NEURAL REPRESENTATION DECODER
# =====================================================================

class SirenLayer(nn.Module):
    """Single SIREN layer: y = sin(omega_0 * (Wx + b))
    Initialization follows Sitzmann et al. NeurIPS 2020."""

    def __init__(self, in_features, out_features, omega_0=30.0,
                 is_first=False, is_last=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        if self.is_last:
            return self.linear(x)
        return torch.sin(self.omega_0 * self.linear(x))


class SirenDecoder(nn.Module):
    """SIREN decoder: (latent_z, x, y, z) -> scalar amplitude.
    Architecture: 13 -> 256 -> 256 -> 256 -> 256 -> 1"""

    def __init__(self, latent_dim=10, coord_dim=3, hidden_dim=256,
                 n_layers=4, omega_0=30.0):
        super().__init__()
        in_dim = latent_dim + coord_dim
        
        self.first_layer = SirenLayer(in_dim, hidden_dim, omega_0=omega_0, is_first=True)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.hidden_layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.final_layer = SirenLayer(hidden_dim, 1, omega_0=omega_0, is_last=True)

    def forward(self, x):
        # The True AdS Hyperbolic Positional Encoding: divide by z
        # High frequency at the boundary (z->0), smooth in the deep bulk (z->1)
        z_depth = x[:, 10:11] 
        x_scaled = x.clone()
        x_scaled[:, 11:] = x_scaled[:, 11:] / z_depth

        out = self.first_layer(x_scaled)
        for layer in self.hidden_layers:
            out = layer(out)
        return self.final_layer(out)


def make_coord_grid(depth=20, height=64, width=64, device="cpu", use_hyperbolic=True):
    """Generate 3D coordinate grid. Z is stretched hyperbolically."""
    if use_hyperbolic:
        z_vals = torch.linspace(1e-4, 1.0, depth, device=device)
    else:
        z_vals = torch.linspace(-1, 1, depth, device=device)

    h = torch.linspace(-1, 1, height, device=device)
    w = torch.linspace(-1, 1, width,  device=device)
    gd, gh, gw = torch.meshgrid(z_vals, h, w, indexing="ij")
    return torch.stack([gd, gh, gw], dim=-1).reshape(-1, 3)


# =====================================================================
# CLASSICAL AUTOENCODER (Phase 1 -- SIREN decoder)
# =====================================================================

class ClassicalAdS(nn.Module):
    """Encoder + classical 10-dim bottleneck + SIREN decoder."""

    def __init__(self, in_channels=1, hidden_dim=256, n_siren_layers=4, omega_0=30.0):
        super().__init__()
        self.D, self.H, self.W = 20, 64, 64
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Tanh(),
        )
        self.classical_bottleneck = nn.Sequential(nn.Linear(10, 10), nn.Tanh())
        self.siren_decoder = SirenDecoder(latent_dim=10, coord_dim=3,
                                          hidden_dim=hidden_dim,
                                          n_layers=n_siren_layers,
                                          omega_0=omega_0)
        self._cached_coords = None
        self._cached_device = None

    def _get_coords(self, device):
        if self._cached_coords is None or self._cached_device != device:
            self._cached_coords = make_coord_grid(self.D, self.H, self.W, device=device)
            self._cached_device = device
        return self._cached_coords

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        N = self.D * self.H * self.W
        z = self.classical_bottleneck(self.encoder(x))   # [B, 10]
        coords = self._get_coords(device)                 # [N, 3]
        z_exp = z.unsqueeze(1).expand(B, N, 10)
        c_exp = coords.unsqueeze(0).expand(B, N, 3)
        queries = torch.cat([z_exp, c_exp], dim=-1).reshape(B * N, 13)
        return self.siren_decoder(queries).reshape(B, 1, self.D, self.H, self.W)

    def get_transferable_weights(self):
        """Return encoder + siren_decoder state_dict (drop bottleneck)."""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("classical_bottleneck")}


# =====================================================================
# HYBRID QUANTUM MODEL (Encoder + Quantum + SIREN Decoder)
# =====================================================================

try:
    import pennylane as qml
except ImportError:
    print("[INSTALL] PennyLane not found. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pennylane", "-q"])
    print("[INSTALL] PennyLane installed successfully!")
    import pennylane as qml

N_QUBITS = 10
_clean_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_clean_dev, interface="torch", diff_method="adjoint")
def _quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumLatentLayer(nn.Module):
    def __init__(self, n_quantum_layers=3):
        super().__init__()
        weight_shapes = {"weights": (n_quantum_layers, N_QUBITS, 3)}
        self.q_layer = qml.qnn.TorchLayer(_quantum_circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x).float()


class HybridQuantumAdS(nn.Module):
    """Encoder + 10-qubit QuantumLatentLayer + SIREN implicit decoder."""

    def __init__(self, in_channels=1, hidden_dim=256, n_siren_layers=4, omega_0=30.0):
        super().__init__()
        self.D, self.H, self.W = 20, 64, 64
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Tanh(),
        )
        self.quantum_layer = QuantumLatentLayer(n_quantum_layers=3)
        self.siren_decoder = SirenDecoder(latent_dim=10, coord_dim=3,
                                          hidden_dim=hidden_dim,
                                          n_layers=n_siren_layers,
                                          omega_0=omega_0)
        self._cached_coords = None
        self._cached_device = None

    def _get_coords(self, device):
        if self._cached_coords is None or self._cached_device != device:
            self._cached_coords = make_coord_grid(self.D, self.H, self.W, device=device)
            self._cached_device = device
        return self._cached_coords

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        N = self.D * self.H * self.W  # 81920
        z = self.quantum_layer(self.encoder(x))           # [B, 10]
        coords = self._get_coords(device)                  # [N, 3]
        z_exp = z.unsqueeze(1).expand(B, N, 10)
        c_exp = coords.unsqueeze(0).expand(B, N, 3)
        queries = torch.cat([z_exp, c_exp], dim=-1).reshape(B * N, 13)
        return self.siren_decoder(queries).reshape(B, 1, self.D, self.H, self.W)

    @autocast(enabled=False)
    def forward_collocation(self, x, coords):
        """
        Evaluate the physics model specifically at randomly sampled
        AdS collocation points for computing the Klein-Gordon loss,
        drastically reducing VRAM overhead vs full volume evaluation.
        """
        B = x.shape[0]
        N = coords.shape[1]
        z = self.quantum_layer(self.encoder(x))           # [B, 10]
        z_exp = z.unsqueeze(1).expand(B, N, 10)
        
        # coords is already shape [B, N, 3], no need to expand 
        # because sample_ads_collocation yields [B, N, 3].
        queries = torch.cat([z_exp, coords], dim=-1).reshape(B * N, 13)
        
        # Returns [B, N, 1] matching the collocation point count
        return self.siren_decoder(queries).reshape(B, N, 1)


# =====================================================================
# QUANTUM ENTROPY EXTRACTOR (for Phase 2 metrics)
# =====================================================================

@qml.qnode(_clean_dev, interface="torch", diff_method="adjoint")
def _observer_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.density_matrix(wires=range(N_QUBITS))


def extract_entropy(inputs, q_layer_module):
    """Compute mean von Neumann entropy across a batch."""
    current_weights = q_layer_module.q_layer.weights.detach()
    batch_S = []
    for i in range(min(inputs.shape[0], 4)):  # cap at 4 samples for speed
        rho = _observer_circuit(inputs[i], current_weights)
        if isinstance(rho, torch.Tensor):
            rho = rho.detach().cpu().numpy()
        eigs = np.linalg.eigvalsh(rho)
        eigs = eigs[eigs > 1e-12]
        S = -np.sum(eigs * np.log(eigs))
        batch_S.append(float(S))
    return sum(batch_S) / len(batch_S) if batch_S else 0.0


# =====================================================================
# PHYSICS-AWARE LOSS FUNCTION
# =====================================================================

class PhysicsAwareLoss(nn.Module):
    """
    L = MSE + ls * L_spectral + lg * L_gradient + lb * L_boundary

    L_spectral : 3D FFT magnitude difference (dim 2,3,4)
    L_gradient : finite-difference spatial gradient difference
    L_boundary : AdS/CFT boundary condition at depth z=0
    """

    def __init__(self, ls=0.1, lg=0.05, lb=0.5):
        super().__init__()
        self.ls, self.lg, self.lb = ls, lg, lb
        self.mse = nn.MSELoss()

    def spectral_loss(self, pred, truth):
        # norm="forward" divides by N so FFT values match MSE scale
        # .float() needed because cuFFT rejects half-precision on non-power-of-2
        fp = torch.fft.fftn(pred.float(),  dim=(2, 3, 4), norm="forward")
        ft = torch.fft.fftn(truth.float(), dim=(2, 3, 4), norm="forward")
        return torch.mean(torch.abs(fp - ft) ** 2)

    def gradient_loss(self, pred, truth):
        ld = torch.mean((pred[:,:,1:,:,:] - pred[:,:,:-1,:,:] -
                          truth[:,:,1:,:,:] + truth[:,:,:-1,:,:]) ** 2)
        lh = torch.mean((pred[:,:,:,1:,:] - pred[:,:,:,:-1,:] -
                          truth[:,:,:,1:,:] + truth[:,:,:,:-1,:]) ** 2)
        lw = torch.mean((pred[:,:,:,:,1:] - pred[:,:,:,:,:-1] -
                          truth[:,:,:,:,1:] + truth[:,:,:,:,:-1]) ** 2)
        return (ld + lh + lw) / 3.0

    def boundary_loss(self, pred, boundary_input):
        # Depth axis is dim 2: pred[:, :, 0, :, :] = outermost boundary
        return torch.mean((pred[:, :, 0, :, :] -
                           boundary_input[:, :, 0, :, :]) ** 2)

    def forward(self, pred, truth, boundary_input):
        l_mse  = self.mse(pred, truth)
        l_spec = self.spectral_loss(pred, truth)
        l_grad = self.gradient_loss(pred, truth)
        l_bdy  = self.boundary_loss(pred, boundary_input)
        total  = l_mse + self.ls * l_spec + self.lg * l_grad + self.lb * l_bdy
        return total, {
            "MSE": l_mse.item(), "Spectral": l_spec.item(),
            "Gradient": l_grad.item(), "Boundary": l_bdy.item(),
            "Total": total.item(),
        }


# =====================================================================
# RELATIVE L2 METRIC
# =====================================================================

def compute_rel_l2(pred, truth):
    B = pred.shape[0]
    p, t = pred.reshape(B, -1), truth.reshape(B, -1)
    return (torch.linalg.norm(p - t, dim=1) /
            (torch.linalg.norm(t, dim=1) + 1e-8)).mean().item() * 100.0


# =====================================================================
# CHECKPOINT SYSTEM
# =====================================================================

CKPT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
MODEL_DIR = "/kaggle/working/models"
os.makedirs(MODEL_DIR, exist_ok=True)
RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_ckpt(tag, epoch, model, optimizer, scheduler, metrics):
    path = os.path.join(CKPT_DIR, f"twophase_{tag}_latest.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }, path)
    if epoch % 10 == 0:
        backup = os.path.join(CKPT_DIR, f"twophase_{tag}_epoch_{epoch}.pth")
        torch.save(torch.load(path, map_location="cpu"), backup)
        print(f"       Backup: {backup}")


def load_ckpt(tag, model, optimizer, scheduler, device):
    path = os.path.join(CKPT_DIR, f"twophase_{tag}_latest.pth")
    if not os.path.exists(path):
        return 1
    print(f"[RESUME] Loading {path}")
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
    print("\n" + "=" * 72)
    print("  PHASE 1: Classical Foundation Training")
    print("  Loss: MSE + Spectral + Gradient + Boundary")
    print("=" * 72)

    model = ClassicalAdS().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ClassicalAdS -- {n:,} params")

    criterion = PhysicsAwareLoss(args.lambda_spectral, args.lambda_gradient,
                                 args.lambda_boundary)
    scaler = GradScaler()

    dataset = UniverseDataset(args.data_dir)

    epochs = args.phase1_epochs
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs,
        steps_per_epoch=len(loader), pct_start=0.2)

    start_epoch = load_ckpt("phase1", model, optimizer, scheduler, device)

    csv_path = os.path.join(RESULTS_DIR, "twophase_metrics.csv")
    log = []
    if os.path.exists(csv_path) and start_epoch > 1:
        log = pd.read_csv(csv_path).to_dict("records")

    print(f"\n[GO] Epochs {start_epoch} -> {epochs}\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        sums = {k: 0.0 for k in ["MSE","Spectral","Gradient","Boundary","Total"]}
        nb = 0

        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(x)
                loss, c = criterion(pred, y, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for k in sums: sums[k] += c[k]
            nb += 1

        for k in sums: sums[k] /= max(1, nb)

        model.eval()
        with torch.no_grad():
            vx, vy = next(iter(loader))
            vx, vy = vx.to(device), vy.to(device)
            rl2 = compute_rel_l2(model(vx), vy)

        lr = optimizer.param_groups[0]["lr"]
        print(f"  P1 [{epoch:3d}/{epochs}] "
              f"Total={sums['Total']:.5f} MSE={sums['MSE']:.5f} "
              f"Spec={sums['Spectral']:.4f} Grad={sums['Gradient']:.5f} "
              f"Bdy={sums['Boundary']:.5f} RelL2={rl2:.2f}% lr={lr:.2e}")

        row = {"Phase": 1, "Epoch": epoch, "RelL2": rl2, "lr": lr, **sums}
        log.append(row)
        save_ckpt("phase1", epoch, model, optimizer, scheduler, row)
        pd.DataFrame(log).to_csv(csv_path, index=False)

    out = os.path.join(MODEL_DIR, "classical_phase1.pth")
    torch.save(model.state_dict(), out)
    print(f"\n[PHASE 1 DONE] {out}")
    return model


# =====================================================================
# PHASE 2: QUANTUM FINE-TUNING (Differential LR)
# =====================================================================

def run_phase2(args, device):
    print("\n" + "=" * 72)
    print("  PHASE 2: Quantum Fine-Tuning (Differential LR)")
    print("  Quantum lr=1e-3 | Decoder lr=1e-6 | Encoder FROZEN")
    print("=" * 72)

    model = HybridQuantumAdS().to(device)

    # Transfer Phase 1 weights
    p1_path = os.path.join(MODEL_DIR, "classical_phase1.pth")
    if os.path.exists(p1_path):
        print(f"[TRANSFER] Loading {p1_path}")
        c_model = ClassicalAdS()
        c_model.load_state_dict(torch.load(p1_path, map_location=device))
        transfer = c_model.get_transferable_weights()
        sd = model.state_dict()
        count = 0
        for k, v in transfer.items():
            if k in sd:
                sd[k] = v
                count += 1
        model.load_state_dict(sd)
        print(f"[TRANSFER] {count} tensors transferred")
        del c_model
    else:
        print(f"[WARNING] {p1_path} not found -- training from scratch")

    model = model.to(device)

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("[FREEZE] Encoder frozen")

    # Differential LR
    q_params = list(model.quantum_layer.parameters())
    d_params = list(model.siren_decoder.parameters())

    optimizer = optim.AdamW([
        {"params": q_params, "lr": 1e-3},
        {"params": d_params, "lr": 1e-4},
    ], weight_decay=1e-4)
    print("[LR] Quantum=1e-3 | Decoder=1e-4")

    criterion = PhysicsAwareLoss(args.lambda_spectral, args.lambda_gradient,
                                 args.lambda_boundary)
    scaler = GradScaler()

    dataset = UniverseDataset(args.data_dir)

    epochs = args.phase2_epochs
    start_epoch = load_ckpt("phase2", model, optimizer, None, device)

    csv_path = os.path.join(RESULTS_DIR, "twophase_metrics.csv")
    log = []
    if os.path.exists(csv_path):
        log = pd.read_csv(csv_path).to_dict("records")

    print(f"\n[GO] Epochs {start_epoch} -> {epochs}\n")

    for epoch in range(start_epoch, epochs + 1):
        # Unfreeze decoder at specified epoch
        if epoch == args.phase2_unfreeze_epoch:
            print(f"\n  [UNFREEZE] Epoch {epoch}: Decoder lr -> 5e-6")
            optimizer.param_groups[1]["lr"] = 5e-6

        # Get Sigmoid Curriculum Loss Weights for this epoch
        w_curriculum = get_loss_weights(epoch, epochs)
        
        model.train()
        sums = {k: 0.0 for k in ["MSE","Spectral","Gradient","Boundary",
                                 "KG_PDE","Eikonal","RT","Total"]}
        nb = 0

        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # --- 1. Classical MSE & Boundary Anchoring ---
            with autocast():
                # Phase 2 now evaluates the Dirichlet boundary specifically at z=1e-4 
                # (which maps to slice 0 in the y tensor if generated correctly). 
                pred = model(x)
                loss_classic, c_classic = criterion(pred, y, x)
            # --- 2. AdS Physics PDE Computing (Autograd-safe) ---
            # We must compute autograd outside `autocast` for 2nd order stability.
            loss_pde = torch.tensor(0.0, device=device)
            loss_eik = torch.tensor(0.0, device=device)
            loss_rt  = torch.tensor(0.0, device=device)
            
            if w_curriculum["pde"] > 0 or w_curriculum["rt"] > 0:
                # Sample 4096 collocation points per batch logarithmically
                col_coords, measure = sample_ads_collocation(
                    batch_size=x.shape[0], num_points=4096, 
                    z_min=1e-4, z_max=1.0, device=device
                )
                
                # Forward pass explicitly at collocation coords
                phi = model.forward_collocation(x, col_coords)
                
                # Klein-Gordon & Eikonal (Curved space d'Alembertian)
                if w_curriculum["pde"] > 0:
                    loss_pde = klein_gordon_loss(phi, col_coords, measure, mass_sq=0.0)
                    loss_eik = eikonal_loss(phi, col_coords)
                
                # Ryu-Takayanagi tether constraints
                if w_curriculum["rt"] > 0:
                    # In true formulation, area is integrated level set. 
                    # We compute a proxy area as the integral of the measure bounds 
                    # and tether it to entanglement entropy.
                    area_proxy = torch.tensor(1.0, device=device, requires_grad=True) * sum(measure.mean(dim=1)) 
                    # We fetch static entropy periodically to guide the Area Proxy
                    with torch.no_grad():
                        z_latent = model.encoder(x)
                    current_entropy = extract_entropy(z_latent, model.quantum_layer)
                    if current_entropy > 0:
                        loss_rt = ryu_takayanagi_loss(area_proxy, current_entropy, G_N=1.0)
                    
            # Combine losses via curriculum (Topological Scaffolding)
            # 3D Data Scaffold decays to zero, 2D Boundary Anchor remains permanent
            loss_3d_scaffold = (c_classic["MSE"] + 
                                args.lambda_spectral * c_classic["Spectral"] + 
                                args.lambda_gradient * c_classic["Gradient"])
            loss_2d_anchor   = args.lambda_boundary * c_classic["Boundary"]
            
            total_loss = (
                w_curriculum["3d_data"] * loss_3d_scaffold + 
                w_curriculum["boundary"] * loss_2d_anchor +
                w_curriculum["pde"] * (loss_pde + loss_eik) +
                w_curriculum["rt"] * loss_rt
            )
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            for k in ["MSE","Spectral","Gradient","Boundary"]:
                sums[k] += c_classic[k]
            sums["KG_PDE"]  += loss_pde.item()
            sums["Eikonal"] += loss_eik.item()
            sums["RT"]      += loss_rt.item()
            sums["Total"]   += total_loss.item()
            nb += 1

        for k in sums: sums[k] /= max(1, nb)

        model.eval()
        with torch.no_grad():
            vx, vy = next(iter(loader))
            vx, vy = vx.to(device), vy.to(device)
            vp = model(vx)
            rl2 = compute_rel_l2(vp, vy)

        # Entropy every 10 epochs
        entropy = 0.0
        if epoch % 10 == 0:
            try:
                with torch.no_grad():
                    lat = model.encoder(vx)
                entropy = extract_entropy(lat, model.quantum_layer)
            except Exception as e:
                entropy = -1.0
                print(f"  [WARN] Entropy failed: {e}")

        lr_q = optimizer.param_groups[0]["lr"]
        lr_d = optimizer.param_groups[1]["lr"]
        print(f"  P2 [{epoch:3d}/{epochs}] "
              f"Tot={sums['Total']:.3f} MSE={sums['MSE']:.5f} "
              f"PDE={sums['KG_PDE']:.3f} RT={sums['RT']:.4f} "
              f"RelL2={rl2:.1f}% S={entropy:.3f} "
              f"lr_q={lr_q:.0e} lr_d={lr_d:.0e}")

        row = {"Phase": 2, "Epoch": epoch, "RelL2": rl2,
               "Entropy": entropy, "lr_q": lr_q, "lr_d": lr_d,
               "w_pde": w_curriculum["pde"], "w_rt": w_curriculum["rt"], **sums}
        log.append(row)
        save_ckpt("phase2", epoch, model, optimizer, None, row)
        pd.DataFrame(log).to_csv(csv_path, index=False)

    out = os.path.join(MODEL_DIR, "quantum_phase2.pth")
    torch.save(model.state_dict(), out)
    print(f"\n[PHASE 2 DONE] {out}")
    return model


# =====================================================================
# PHASE 3: HARDWARE-AWARE NOISE TRAINING
# =====================================================================

def run_phase3(args, device):
    print("\n" + "=" * 72)
    print("  PHASE 3: Hardware-Aware Noise Training")
    print(f"  Depolarizing noise rate: {args.noise_rate} per gate")
    print("=" * 72)

    noise_rate = args.noise_rate

    # --- Noisy quantum device ---
    noisy_dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(noisy_dev, interface="torch", diff_method="backprop")
    def noisy_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        n_layers = weights.shape[0]
        for li in range(n_layers):
            for q in range(N_QUBITS):
                qml.Rot(weights[li, q, 0], weights[li, q, 1],
                        weights[li, q, 2], wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1 + li) % N_QUBITS])
            for q in range(N_QUBITS):
                qml.DepolarizingChannel(noise_rate, wires=q)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    class NoisyQuantumLayer(nn.Module):
        def __init__(self, n_layers=3):
            super().__init__()
            self.q_weights = nn.Parameter(torch.randn(n_layers, N_QUBITS, 3) * 0.1)

        def forward(self, x):
            results = []
            for i in range(x.shape[0]):
                out = noisy_circuit(x[i], self.q_weights)
                results.append(torch.stack(out))
            return torch.stack(results).float()

    # --- Load Phase 2 model ---
    model = HybridQuantumAdS()
    p2_path = os.path.join(MODEL_DIR, "quantum_phase2.pth")
    if os.path.exists(p2_path):
        print(f"[LOAD] {p2_path}")
        model.load_state_dict(torch.load(p2_path, map_location="cpu"))
    else:
        print(f"[WARNING] {p2_path} not found")

    # Swap in noisy layer, transfer weights
    noisy_layer = NoisyQuantumLayer(n_layers=3)
    with torch.no_grad():
        src = None
        for name, param in model.quantum_layer.named_parameters():
            if "weight" in name.lower():
                src = param
                break
        if src is not None and src.shape == noisy_layer.q_weights.shape:
            noisy_layer.q_weights.copy_(src)
            print("[TRANSFER] Quantum weights -> noisy layer")

    model.quantum_layer = noisy_layer
    model = model.to(device)

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Differential LR
    q_params = list(model.quantum_layer.parameters())
    d_params = list(model.siren_decoder.parameters())
    optimizer = optim.AdamW([
        {"params": q_params, "lr": 5e-4},
        {"params": d_params, "lr": 1e-4},
    ], weight_decay=1e-4)
    print("[LR] Quantum=5e-4 | Decoder=1e-4")

    criterion = PhysicsAwareLoss(args.lambda_spectral, args.lambda_gradient,
                                 args.lambda_boundary)

    dataset = UniverseDataset(args.data_dir)
    if len(dataset) == 0: return
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    epochs = args.phase3_epochs
    start_epoch = load_ckpt("phase3", model, optimizer, None, device)

    csv_path = os.path.join(RESULTS_DIR, "twophase_metrics.csv")
    log = []
    if os.path.exists(csv_path):
        log = pd.read_csv(csv_path).to_dict("records")

    print(f"\n[GO] Epochs {start_epoch} -> {epochs}\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        sums = {k: 0.0 for k in ["MSE","Spectral","Gradient","Boundary","Total"]}
        nb = 0

        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # No AMP for mixed device -- needs float32
            pred = model(x)
            loss, c = criterion(pred, y, x)
            loss.backward()
            optimizer.step()
            for k in sums: sums[k] += c[k]
            nb += 1

        for k in sums: sums[k] /= max(1, nb)

        model.eval()
        with torch.no_grad():
            vx, vy = next(iter(loader))
            vx, vy = vx.to(device), vy.to(device)
            rl2 = compute_rel_l2(model(vx), vy)

        lr_q = optimizer.param_groups[0]["lr"]
        lr_d = optimizer.param_groups[1]["lr"]
        print(f"  P3 [{epoch:3d}/{epochs}] "
              f"Total={sums['Total']:.5f} MSE={sums['MSE']:.5f} "
              f"RelL2={rl2:.2f}% lr_q={lr_q:.1e} lr_d={lr_d:.1e}")

        row = {"Phase": 3, "Epoch": epoch, "RelL2": rl2,
               "lr_q": lr_q, "lr_d": lr_d, **sums}
        log.append(row)
        save_ckpt("phase3", epoch, model, optimizer, None, row)
        pd.DataFrame(log).to_csv(csv_path, index=False)

    # --- Save final model back into clean HybridQuantumAdS ---
    final = HybridQuantumAdS()
    fsd = final.state_dict()
    for k, v in model.state_dict().items():
        if k.startswith("quantum_layer"):
            continue
        if k in fsd:
            fsd[k] = v
    # Transfer noisy weights to clean layer
    nw = model.quantum_layer.q_weights.detach().cpu()
    for k in fsd:
        if "q_layer.weights" in k and fsd[k].shape == nw.shape:
            fsd[k] = nw
            print(f"[TRANSFER] Noisy weights -> '{k}'")
    final.load_state_dict(fsd)

    out = os.path.join(MODEL_DIR, "QUANTUM_TWOPHASE_MODEL.pth")
    torch.save(final.state_dict(), out)
    print(f"\n[PHASE 3 DONE] Final model: {out}")
    return final


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print(f"  THREE-PHASE QUANTUM TRAINING PIPELINE")
    print(f"  Device: {device.type.upper()}")
    print(f"{'='*72}")

    phases = [args.phase] if args.phase != "all" else ["1", "2", "3"]

    for phase in phases:
        t0 = time.time()
        if   phase == "1": run_phase1(args, device)
        elif phase == "2": run_phase2(args, device)
        elif phase == "3": run_phase3(args, device)
        mins = (time.time() - t0) / 60
        print(f"\n  Phase {phase} completed in {mins:.1f} minutes")

    print(f"\n{'='*72}")
    print(f"  ALL PHASES COMPLETE")
    print(f"  Metrics: {RESULTS_DIR}/twophase_metrics.csv")
    print(f"  Models:  {MODEL_DIR}/")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
