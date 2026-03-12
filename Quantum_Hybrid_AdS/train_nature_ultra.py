import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from hybrid_autoencoder import HybridQuantumAdS

# ---------------------------------------------------------
# 1. DISCOVERY HOOKS
# ---------------------------------------------------------
def detect_holographic_event_horizon(bulk_tensor, threshold=10.0):
    """
    Monitors the 3D bulk output for high-energy density regions, 
    preparing us for black hole formation analysis.
    """
    max_energy = torch.max(torch.abs(bulk_tensor))
    if max_energy > threshold:
        print(f"⚠️ [DISCOVERY] Holographic Event Horizon Detected! High-energy density region: {max_energy.item():.2f}")

# ---------------------------------------------------------
# 2. INDUSTRIAL DATA PIPELINE
# ---------------------------------------------------------
class UniverseDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        if not self.files:
            print(f"⚠️ [WARNING] No .pt files found in {data_dir}!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return data['boundary'], data['bulk']

# ---------------------------------------------------------
# 3. TRAINING EPOCH LOGIC
# ---------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, scaler, device, scheduler=None):
    model.train()
    
    epoch_loss = 0.0
    epoch_l2_error = 0.0
    num_batches = 0
    
    for x_boundary_batch, y_bulk_truth in loader:
        # non_blocking=True for asynchronous GPU transfers
        x_boundary_batch = x_boundary_batch.to(device, non_blocking=True)
        y_bulk_truth = y_bulk_truth.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True) # slightly faster than .zero_grad()
        
        # Mixed Precision Forward Pass
        with autocast():
            # Forward Pass: Crush -> Entangle -> Expand
            pred_bulk = model(x_boundary_batch)
            
            # Discovery Hook
            detect_holographic_event_horizon(pred_bulk)
            
            # Calculate Loss
            loss = criterion(pred_bulk, y_bulk_truth)
        
        # Mixed Precision Backward Pass: The Heavy Compute
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
            
        epoch_loss += loss.item()
        
        with torch.no_grad():
            batch_l2 = torch.norm(pred_bulk - y_bulk_truth) / torch.norm(y_bulk_truth)
            epoch_l2_error += batch_l2.item()
            
        num_batches += 1
        
    if num_batches == 0:
        return 0.0, 0.0
        
    avg_loss = epoch_loss / num_batches
    avg_l2_error = epoch_l2_error / num_batches
    
    return avg_loss, avg_l2_error

# ---------------------------------------------------------
# 4. THE SONIC BOOM EXECUTION ORCHESTRATOR
# ---------------------------------------------------------
def run_sonic_boom_training():
    # Accelerate everything
    torch.backends.cudnn.benchmark = True
    
    # Automatically binds to the Kaggle P100 or Colab T4 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ [HARDWARE] Neural Engine bound to: {device.type.upper()}")
    if device.type == "cuda":
        print(f"⚡ [GPU DETECTED] {torch.cuda.get_device_name(0)}")

    # Initialize Hybrid Architecture
    model = HybridQuantumAdS(in_channels=1, out_channels=1).to(device)
    
    # Load previously learned physics from collision_best.pth
    pretrained_path = os.path.join("models", "collision_best.pth")
    if os.path.exists(pretrained_path):
        print(f"\n🔄 [SYSTEM] Loading learned physics from {pretrained_path}...")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✅ [SYSTEM] Loaded successfully.")
        except Exception as e:
            print(f"⚠️ [WARNING] Failed to load checkpoint: {e}")
    else:
        print(f"\n⚠️ [WARNING] Checkpoint not found at {pretrained_path}. Starting from scratch.")
        
    # Hardware Optimization: Fuse 3D CNN and Quantum kernels
    if hasattr(torch, "compile"):
        print(f"⚡ [COMPILER] Fusing modules with torch.compile...")
        # note: compile not supported on all platforms/python versions, try-catch just in case
        try:
            model = torch.compile(model)
            print(f"✅ [COMPILER] Fusion successful.")
        except Exception as e:
            print(f"⚠️ [WARNING] torch.compile failed or unsupported, skipping fusion: {e}")
        
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # Dataset Preparation
    dataset = UniverseDataset("data_holography")
    if len(dataset) == 0:
        print("❌ [ERROR] Dataset is empty.")
        return
        
    # ---------------------------------------------------------
    # STAGE 1: THE WARP SLINGSHOT (2.5 Hours / 250 Epochs)
    # ---------------------------------------------------------
    print("\n🚀 [STAGE 1] Initiating The Warp Slingshot...")
    
    # Note: persistent_workers=True requires num_workers > 0.
    # On Windows, num_workers > 0 sometimes causes multiprocessing issues in scripts
    # not guarded by __main__ block nicely, but we are inside one.
    loader_stage_1 = DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0, # safe fallback for windows
        pin_memory=True,
        persistent_workers=True if os.name != 'nt' else False
    )
    
    epochs_stage_1 = 250
    optimizer_stage_1 = optim.AdamW(model.parameters(), lr=5e-3)
    
    # Slingshot to bypass local minima
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer_stage_1, 
        max_lr=5e-3, 
        epochs=epochs_stage_1, 
        steps_per_epoch=len(loader_stage_1),
        pct_start=0.2
    )

    for epoch in range(1, epochs_stage_1 + 1):
        loss, l2_err = train_epoch(model, loader_stage_1, optimizer_stage_1, criterion, scaler, device, scheduler)
        if epoch % 5 == 0 or epoch == 1:
            # We use optimizer param group for generic LR because OneCycleLR updates it there
            current_lr = optimizer_stage_1.param_groups[0]['lr']
            print(f"S1 Epoch [{epoch}/{epochs_stage_1}] | MSE: {loss:.6f} | L2 Error: {l2_err * 100:.3f}% | LR: {current_lr:.6f}")

    # ---------------------------------------------------------
    # STAGE 2: THE CRYOGENIC POLISH (0.5 Hours / 50 Epochs)
    # ---------------------------------------------------------
    print("\n❄️ [STAGE 2] Initiating The Cryogenic Polish...")
    loader_stage_2 = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True,
        persistent_workers=True if os.name != 'nt' else False
    )
    
    epochs_stage_2 = 50
    # Static, ultra-low learning rate
    optimizer_stage_2 = optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(1, epochs_stage_2 + 1):
        loss, l2_err = train_epoch(model, loader_stage_2, optimizer_stage_2, criterion, scaler, device, scheduler=None)
        
        print(f"S2 Epoch [{epoch}/{epochs_stage_2}] | MSE: {loss:.6f} | L2 Error: {l2_err * 100:.3f}%")
        
        # Convergence Target
        if l2_err < 0.001:
            print(f"\n🏆 [VICTORY] Achieved <0.1% Relative L2 Error ({l2_err * 100:.3f}%) at Epoch {epoch} of Stage 2!")
            final_path = "FINAL_NATURE_MODEL.pth"
            
            # Save the unwrapped model state (in case torch.compile wrapped it)
            state_dict_to_save = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save(state_dict_to_save, final_path)
            
            print(f"💾 [SYSTEM] Nature-ready model secured at: {final_path}")
            print("Terminating run. Ready for publication.")
            return

    print("\n✅ [SYSTEM] Sonic Boom Protocol Exhausted. Target <0.1% not reached, but pipeline finished gracefully.")

if __name__ == "__main__":
    run_sonic_boom_training()
