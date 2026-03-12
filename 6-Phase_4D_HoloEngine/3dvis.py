import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import numpy as np

def visualize_comparison(model, cfg, dataset_path, time_step=0.5):
    # 1. Load Ground Truth from Master Dataset
    data = np.load(dataset_path)
    # Filter points for the selected time step
    pinn_pts = data['pinn_points']
    mask = (np.abs(pinn_pts[:, 1] - time_step) < 0.05)
    gt_pts = pinn_pts[mask]
    
    # 2. Generate Model Predictions
    model.eval()
    coords = torch.from_numpy(gt_pts[:, 1:5]).float().to(cfg.DEVICE)
    with torch.no_grad():
        z_latent = model.encoder(torch.from_numpy(data['cnn_volumes'][1]).unsqueeze(0).unsqueeze(0).float().to(cfg.DEVICE))
        pred_phi = model.siren(coords, z_latent.expand(coords.shape[0], -1)).cpu().numpy()

    # 3. Create Subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Ground Truth (Master Dataset)", "Model Prediction (Holographic Engine)")
    )

    # Ground Truth Plot
    fig.add_trace(go.Isosurface(
        x=gt_pts[:, 2], y=gt_pts[:, 3], z=gt_pts[:, 4],
        value=gt_pts[:, 5],
        isomin=0.5, isomax=2.0, surface_count=5,
        colorscale='Viridis', opacity=0.6, showscale=False
    ), row=1, col=1)

    # Model Prediction Plot
    fig.add_trace(go.Isosurface(
        x=gt_pts[:, 2], y=gt_pts[:, 3], z=gt_pts[:, 4],
        value=pred_phi.flatten(),
        isomin=0.5, isomax=2.0, surface_count=5,
        colorscale='Plasma', opacity=0.6,
    ), row=1, col=2)

    fig.update_layout(
        title=f"Holographic Bulk Merger at t={time_step}",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='U (Bulk Depth)'),
        scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='U (Bulk Depth)'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

# Usage:
# visualize_comparison(engine_model, Config, '/kaggle/input/holographic-4d-engine/apex_master_dataset.npz')
