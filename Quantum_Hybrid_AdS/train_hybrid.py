import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
from torch.utils.data import Dataset, DataLoader
from hybrid_autoencoder import HybridQuantumAdS

def detect_holographic_event_horizon(bulk_tensor, threshold=10.0):
    """
    Monitors the 3D bulk output for high-energy density regions, 
    preparing us for black hole formation analysis.
    """
    max_energy = torch.max(torch.abs(bulk_tensor))
    if max_energy > threshold:
        print(f"⚠️ [DISCOVERY] Holographic Event Horizon Detected! High-energy density region: {max_energy.item():.2f}")


# ---------------------------------------------------------
# 1. CLOUD STORAGE & CHECKPOINT CONFIGURATION
# ---------------------------------------------------------
# Change this path to your mounted Google Drive or Kaggle Working Directory
CHECKPOINT_DIR = "/kaggle/working/" # Use "/content/drive/MyDrive/Quantum_AdS/" for Colab
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "quantum_nature_checkpoint.pth")

def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # CRITICAL: Saves Adam's momentum!
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"\n💾 [SYSTEM] Checkpoint securely written to {path} at Epoch {epoch}.")
    print("Safe to disconnect or switch cloud accounts.")

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        print(f"\n🔄 [SYSTEM] Found existing checkpoint at {path}. Waking up AI...")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🚀 [SYSTEM] Resuming training exactly from Epoch {start_epoch} with preserved momentum.")
        return start_epoch
    else:
        print("\n🌱 [SYSTEM] No checkpoint found. Initializing fresh quantum entanglement mapping...")
        return 0

# ---------------------------------------------------------
# 2. HARDWARE ACCELERATION SETUP
# ---------------------------------------------------------
# Automatically binds to the Kaggle P100 or Colab T4 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ [HARDWARE] Neural Engine bound to: {device.type.upper()}")
if device.type == "cuda":
    print(f"⚡ [GPU DETECTED] {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------
# 3. INITIALIZE HYBRID ARCHITECTURE
# ---------------------------------------------------------
model = HybridQuantumAdS(in_channels=1, out_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Load previous memory if doing the "Account Switch" strategy
start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_PATH)

# ---------------------------------------------------------
# 4. THE DATA PIPELINE 
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

print("\n📦 [DATA] Loading Spacetime Universe Batch...")
dataset = UniverseDataset("data_holography")
if len(dataset) == 0:
    print("❌ [ERROR] Dataset is empty. Please populate 'data_holography' with .pt files.")
    exit(1)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# ---------------------------------------------------------
# 5. THE ENDURANCE TRAINING LOOP
# ---------------------------------------------------------
total_epochs = 1000
save_interval = 10 # Saves to cloud every 10 epochs

print("\n🌌 [TRAINING] Commencing Quantum-Classical Optimization...")

for epoch in range(start_epoch, total_epochs):
    model.train()
    
    epoch_loss = 0.0
    epoch_l2_error = 0.0
    num_batches = 0
    
    for x_boundary_batch, y_bulk_truth in dataloader:
        x_boundary_batch = x_boundary_batch.to(device)
        y_bulk_truth = y_bulk_truth.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass: Crush -> Entangle -> Expand
        pred_bulk = model(x_boundary_batch)
        
        # Discovery Hook
        detect_holographic_event_horizon(pred_bulk)
        
        # Calculate Loss
        loss = criterion(pred_bulk, y_bulk_truth)
        
        # Backward Pass: The Heavy Compute
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        with torch.no_grad():
            batch_l2 = torch.norm(pred_bulk - y_bulk_truth) / torch.norm(y_bulk_truth)
            epoch_l2_error += batch_l2.item()
            
        num_batches += 1
        
    if num_batches == 0:
        print("⚠️ [WARNING] Dataloader is empty, skipping epoch!")
        continue
        
    avg_loss = epoch_loss / num_batches
    avg_l2_error = epoch_l2_error / num_batches
    
    # Progress Monitor
    if epoch % 2 == 0:
        print(f"Epoch [{epoch}/{total_epochs}] | MSE Loss: {avg_loss:.6f} | Relative L2 Error: {avg_l2_error * 100:.2f}%")
        
    # Cloud Checkpoint Trigger
    if epoch > 0 and epoch % save_interval == 0:
        save_checkpoint(epoch, model, optimizer, avg_loss, CHECKPOINT_PATH)

print("\n✅ [SYSTEM] Quantum Training Protocol Successfully Concluded.")
