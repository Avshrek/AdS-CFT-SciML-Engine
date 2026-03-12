import torch
import numpy as np
from hybrid_autoencoder import HybridQuantumAdS
from fno_architectures import FNO3d
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def construct_fno_input(bdy_s, x_mean, x_std, Z):
    # bdy_s is (T, X)
    T, X = bdy_s.shape
    bdy_n = (bdy_s - x_mean) / (x_std + 1e-8)
    wave  = np.tile(bdy_n[:, :, None], (1, 1, Z))
    tc    = np.linspace(0, 1, T, dtype=np.float32)
    time  = np.broadcast_to(tc[:, None, None], (T, X, Z)).copy()
    zc    = np.linspace(0, 1, Z, dtype=np.float32)
    depth = np.broadcast_to(zc[None, None, :], (T, X, Z)).copy()
    return np.stack([wave, time, depth], axis=0).astype(np.float32)

def main():
    print("Loading Data...")
    bdy_all = np.load('data_collision_master/bdy_collision.npy')[:100]
    blk_all = np.load('data_collision_master/bulk_collision.npy')[:100]

    bdy_mean = bdy_all.mean()
    bdy_std = bdy_all.std()
    
    # We will compute metrics per-sample to avoid memory issues and print proper Relative L2
    Z_dim = 64
    fno_rl2_list = []
    fno_mae_list = []
    q_rl2_list = []
    q_mae_list = []

    print("Loading FNO3d (Classical)...")
    fno = FNO3d(modes1=8, modes2=8, modes3=8, width=20, in_channels=3)
    if os.path.exists('models/collision_publication.pth'):
        fno_state = torch.load('models/collision_publication.pth', map_location='cpu')
        if "model_state_dict" in fno_state:
            fno.load_state_dict(fno_state["model_state_dict"])
        else:
            fno.load_state_dict(fno_state)
    elif os.path.exists('models/collision_final.pth'):
        fno.load_state_dict(torch.load('models/collision_final.pth', map_location='cpu'))
    fno.eval()

    print("Loading HybridQuantumAdS (Quantum)...")
    q_model = HybridQuantumAdS(in_channels=1, out_channels=1)
    q_model.load_state_dict(torch.load('models/NATURE_QUANTUM_MODEL.pth', map_location='cpu'), strict=False)
    q_model.eval()

    print("Running Inference over 100 samples...")
    for i in range(100):
        bdy_s = bdy_all[i]
        blk_truth = blk_all[i]
        
        # FNO preparation
        fno_in = construct_fno_input(bdy_s, bdy_mean, bdy_std, Z_dim)
        fno_in_t = torch.from_numpy(fno_in).unsqueeze(0).float()
        
        # Quantum preparation
        bdy_t = torch.from_numpy(bdy_s).float()
        q_in_t = bdy_t.unsqueeze(-1).repeat(1, 1, 64).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            fno_out = fno(fno_in_t).squeeze(0).squeeze(0).numpy() # (T, X, Z)
            
            # FNO output is normalized to the bulk mean/std.
            # But the dataset is collision_master so we need to reverse normalize if trained that way.
            # evaluate_collisions.py reverses normalization: pred * (y_std + 1e-8) + y_mean
            bulk_mean = blk_all.mean()
            bulk_std = blk_all.std()
            fno_pred = fno_out * (bulk_std + 1e-8) + bulk_mean
            
            q_out = q_model(q_in_t).squeeze(0).squeeze(0).numpy()
            # Quantum output usually matches truth directly depending on how it was trained.
            # In train_nature_quantum.py, we feed the raw bulk `y_bulk_truth` directly.
            q_pred = q_out
            
        # Error computation
        fno_mae_list.append(np.mean(np.abs(fno_pred - blk_truth)))
        q_mae_list.append(np.mean(np.abs(q_pred - blk_truth)))
        
        blk_norm = np.linalg.norm(blk_truth.ravel()) + 1e-12
        fno_rl2 = np.linalg.norm(fno_pred.ravel() - blk_truth.ravel()) / blk_norm
        q_rl2 = np.linalg.norm(q_pred.ravel() - blk_truth.ravel()) / blk_norm
        
        fno_rl2_list.append(fno_rl2)
        q_rl2_list.append(q_rl2)

    # Averages
    fno_mae = np.mean(fno_mae_list)
    q_mae = np.mean(q_mae_list)
    fno_rl2 = np.mean(fno_rl2_list)
    q_rl2 = np.mean(q_rl2_list)

    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'Classical FNO3d':<12} | {'Quantum Hybrid':<12}")
    print("-" * 50)
    print(f"{'Parameters':<20} | {count_parameters(fno):<12,} | {count_parameters(q_model):<12,}")
    print(f"{'Latent Dimension':<20} | {'Extensive':<12} | {'10 Qubits':<12}")
    print(f"{'MAE':<20} | {fno_mae*100:<11.2f}% | {q_mae*100:<11.2f}%")
    print(f"{'Relative L2':<20} | {fno_rl2*100:<11.2f}% | {q_rl2*100:<11.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
