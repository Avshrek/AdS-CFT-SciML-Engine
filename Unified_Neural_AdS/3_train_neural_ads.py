import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_fno import FNO2d
import numpy as np
from tqdm import tqdm

# --- 0.05% ERROR CONFIG ---
BATCH_SIZE = 64
EPOCHS = 150           # OneCycle needs fewer epochs to converge
MAX_LR = 0.005         # Peak learning rate
WEIGHT_DECAY = 1e-4    # Prevents overfitting on high frequencies
# --------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Training on {device} with Super-Convergence Protocol")

# 1. Load the EXACT Data
print("Loading Data...")
x_data = np.load('data_holography/boundary_train.npy').astype(np.float32)
y_data = np.load('data_holography/bulk_train.npy').astype(np.float32)

# Normalization (Critical for <1% error)
x_mean, x_std = x_data.mean(), x_data.std()
y_mean, y_std = y_data.mean(), y_data.std()
x_data = (x_data - x_mean) / (x_std + 1e-8)
y_data = (y_data - y_mean) / (y_std + 1e-8)

# To Tensor
x_train = torch.from_numpy(x_data).unsqueeze(1).unsqueeze(-1) # (N, 1, 64, 1)
y_train = torch.from_numpy(y_data).unsqueeze(1)               # (N, 1, 64, 64)

# FNO expects (N, 1, 64, 64) for input too (we broadcast the 1D boundary to 2D grid hint)
# But our current model takes (N, 1, 64, 64). Let's just repeat the boundary to fill the input 
# This helps FNO "see" the boundary everywhere.
x_train = x_train.repeat(1, 1, 1, 64) 

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# 2. Model
model = FNO2d(modes1=24, modes2=24, width=64).to(device)

# 3. Super-Convergence Optimizer
optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
criterion = nn.MSELoss()

# 4. Training
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    loop = tqdm(train_loader, leave=True)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        # Gradient Clipping (Prevents the 91% spikes)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        scheduler.step() # Update LR every batch
        
        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    # Checkpoint
    if (epoch+1) % 50 == 0:
        torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "models/neural_ads_final.pth")
print("✅ Training Complete.")
