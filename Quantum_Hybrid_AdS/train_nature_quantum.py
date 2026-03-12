import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from hybrid_autoencoder import HybridQuantumAdS
from quantum_architecture import QuantumEntropyExtractor

# ---------------------------------------------------------
# 1. INDUSTRIAL DATA PIPELINE
# ---------------------------------------------------------
class UniverseDataset(Dataset):
    def __init__(self, data_dir):
        bdy_path = os.path.join(data_dir, "bdy_collision.npy")
        blk_path = os.path.join(data_dir, "bulk_collision.npy")
        
        if not os.path.exists(bdy_path) or not os.path.exists(blk_path):
            print(f"⚠️ [WARNING] Dataset missing! Looked for {bdy_path} and {blk_path}")
            self.length = 0
            return
            
        print(f"📦 Loading datasets into RAM from {data_dir}...")
        self.bdy = np.load(bdy_path)
        self.blk = np.load(blk_path)
        self.length = self.bdy.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # hybrid_autoencoder.py expects input [Batch, Channels, T, X, Y] -> [1, 20, 64, 64]
        # data_collision_master is shape [20, 64] for boundary and [20, 64, 64] for bulk
        
        bdy_tensor = torch.from_numpy(self.bdy[idx]).float()
        blk_tensor = torch.from_numpy(self.blk[idx]).float()
        
        # Add channel dimension so it becomes [1, 20, 64] and [1, 20, 64, 64]
        # Note: the current hybrid_autoencoder.py expects a 3D input volume [20, 64, 64] 
        # for BOTH boundary and bulk because of the Conv3d layers in `self.encoder`.
        # We need to tile the boundary [20, 64] across the Y dimension to match [20, 64, 64] 
        # just like train_publication.py does it!
        
        bdy_tiled = bdy_tensor.unsqueeze(-1).repeat(1, 1, 64) # [20, 64] -> [20, 64, 1] -> [20, 64, 64]
        
        # Add the 1 channel dimension for conv3d 
        return bdy_tiled.unsqueeze(0), blk_tensor.unsqueeze(0)


# ---------------------------------------------------------
# 2. CHECKPOINT AND RECOVERY PROTOCOL
# ---------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "quantum_nature_latest.pth")

def save_checkpoint(epoch, model, optimizer, scheduler, loss, entropy):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'entropy': entropy
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    
    # Keep discrete historical backups every 10 epochs
    if epoch % 10 == 0:
        backup_path = os.path.join(CHECKPOINT_DIR, f"quantum_nature_epoch_{epoch}.pth")
        torch.save(checkpoint, backup_path)
        print(f"        💾 Backup securely written to {backup_path}")

def load_checkpoint(model, optimizer, scheduler, device):
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n🔄 [SYSTEM] Found interrupted training state at {CHECKPOINT_PATH}. Waking up AI...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # 💥 HOTFIX: Migrate legacy TorchLayer weights to the new GPU Parameter architecture
        if 'quantum_layer.q_layer.weights' in state_dict:
            state_dict['quantum_layer.q_weights'] = state_dict.pop('quantum_layer.q_layer.weights')
            print("🔄 [SYSTEM] Migrated legacy quantum weights to the new GPU architecture.")
            
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🚀 [SYSTEM] Resuming training exactly from Epoch {start_epoch} with preserved momentum.")
        return start_epoch
    return 1

# ---------------------------------------------------------
# 3. THE NATURE QUANTUM ORCHESTRATOR
# ---------------------------------------------------------
def run_quantum_nature_training():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ [HARDWARE] Neural Engine bound to: {device.type.upper()}")

    # Initialize Hybrid Architecture from scratch
    # We purposefully DO NOT load classical weights here.
    model = HybridQuantumAdS(in_channels=1, out_channels=1).to(device)
    
    # (torch.compile removed because Kaggle P100 is Pascal Architecture / Compute 6.0)
    
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    dataset = UniverseDataset("/kaggle/input/datasets/avshrek/collision-master-data")
    if len(dataset) == 0:
        print("❌ [ERROR] Dataset is empty.")
        return
        
    loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0, 
        pin_memory=True,
        persistent_workers=True if os.name != 'nt' else False
    )
    
    epochs = 100
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Slingshot to bypass local minima quickly
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        epochs=epochs, 
        steps_per_epoch=len(loader),
        pct_start=0.2
    )

    # Load previous memory if doing Kaggle "Account Switch" or recovering from crash
    start_epoch = load_checkpoint(model, optimizer, scheduler, device)

    metrics_log = []
    
    # If resuming, load previous metrics so they aren't overwritten
    metrics_path = "results/nature_quantum_metrics.csv"
    if os.path.exists(metrics_path) and start_epoch > 1:
        existing_metrics = pd.read_csv(metrics_path)
        metrics_log = existing_metrics.to_dict('records')

    print(f"\n🚀 [NATURE RUN] Initiating Quantum Hybrid Training from Epoch {start_epoch}...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for x_boundary_batch, y_bulk_truth in loader:
            x_boundary_batch = x_boundary_batch.to(device, non_blocking=True)
            y_bulk_truth = y_bulk_truth.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                pred_bulk = model(x_boundary_batch)
                loss = criterion(pred_bulk, y_bulk_truth)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
                
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / max(1, num_batches)
        
        # --- QUANTUM METRICS EVALUATION ---
        model.eval()
        with torch.no_grad():
            # Grab just one validation batch to compute entropy
            val_bdy, val_blk = next(iter(loader))
            val_bdy = val_bdy.to(device)
            val_blk = val_blk.to(device)
            
            val_pred = model(val_bdy)
            
            # 1. Classical Geometry Metric
            # MAE physically represents the average deviation of the scalar field (geometry)
            val_mae = torch.mean(torch.abs(val_pred - val_blk)).item()
            
            # 2. Quantum Entanglement Metric
            # Run the Classical Boundary through the CNN Encoder first
            latent_classical = model.encoder(val_bdy)
            # Pass to our Entropy Extractor
            entropy = QuantumEntropyExtractor.extract_entropy(latent_classical, model.quantum_layer)
            
        print(f"Epoch [{epoch}/{epochs}] | MSE: {avg_loss:.6f} | Bulk MAE: {val_mae:.6f} | Q-Entropy S: {entropy:.6f}")
        
        metrics_log.append({
            "Epoch": epoch,
            "MSE_Loss": avg_loss,
            "Bulk_MAE": val_mae,
            "von_Neumann_Entropy": entropy
        })
        
        # Trigger Checkpoint System
        save_checkpoint(epoch, model, optimizer, scheduler, avg_loss, entropy)
        
        # Save Metrics incrementally just in case of abrupt termination
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(metrics_log).to_csv(metrics_path, index=False)
        
    print("\n✅ [NATURE RUN] Training Successfully Concluded.")
    
    # Save Metrics for the Paper
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(metrics_log)
    df.to_csv("results/nature_quantum_metrics.csv", index=False)
    print("💾 [SYSTEM] Metric history securely saved to results/nature_quantum_metrics.csv")
    
    # Save the Quantum Model
    torch.save(model.state_dict(), "NATURE_QUANTUM_MODEL.pth")
    print("💾 [SYSTEM] Model secured at NATURE_QUANTUM_MODEL.pth")

if __name__ == "__main__":
    run_quantum_nature_training()
