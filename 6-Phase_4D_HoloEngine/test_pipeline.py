"""Full pipeline validation — imports, forward pass, losses, gradients."""
import torch
import sys

print("=" * 60)
print("  FULL PIPELINE VALIDATION")
print("=" * 60)

# 1) Import all modules
print("\n[1/6] Importing modules...", end=" ")
try:
    from ads_config import BBHConfig as cfg
    from metric_model import MetricSIREN, MetricReconstructor, BoundaryEncoder
    from einstein_equations import (
        compute_all_einstein_residuals,
        einstein_residual_loss,
        boundary_regularity_loss,
        metric_positivity_loss,
    )
    from bbh_initial_data import compute_initial_metric, initial_data_loss
    from bbh_sampler import BBHBatchSampler
    from horizon import find_apparent_horizon, horizon_regularity_loss
    from observables import extract_boundary_stress_tensor
    from cft_quantum_state import CFTQuantumState
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# 2) Build models
print("[2/6] Building models...", end=" ")
try:
    siren = MetricSIREN(cfg)
    encoder = BoundaryEncoder(cfg)
    recon = MetricReconstructor(cfg)
    n_params = sum(p.numel() for p in siren.parameters()) + \
               sum(p.numel() for p in encoder.parameters())
    print(f"OK  ({n_params:,} total parameters)")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# 3) Forward pass
print("[3/6] Forward pass...", end=" ")
try:
    # Boundary input
    bnd_input = torch.randn(1, 1, cfg.ENCODER_TEMPORAL_FRAMES,
                            cfg.ENCODER_SPATIAL_RES, 1)
    z_latent = encoder(bnd_input)

    # Bulk coords
    N = 128
    coords = torch.rand(N, 3, requires_grad=True)
    coords_scaled = coords.clone().detach()
    coords_scaled[:, 0] *= 0.5
    coords_scaled[:, 1] = coords_scaled[:, 1] * 2 - 1
    coords_scaled[:, 2] = coords_scaled[:, 2] * (cfg.U_MAX - cfg.U_MIN) + cfg.U_MIN
    coords_scaled = coords_scaled.requires_grad_(True)

    z_lat_exp = z_latent.expand(N, -1)
    raw = siren(coords_scaled, z_lat_exp)
    metric = recon.reconstruct(raw, coords_scaled)
    print(f"OK  (A shape: {metric['A'].shape})")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# 4) Einstein residuals
print("[4/6] Einstein residuals...", end=" ")
try:
    residuals, derivs = compute_all_einstein_residuals(metric, coords_scaled)
    res_losses = einstein_residual_loss(residuals)
    print("OK")
    for name, loss in res_losses.items():
        print(f"       {name:20s} = {loss.item():.6f}")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5) Total loss
print("[5/6] Total loss computation...", end=" ")
try:
    bnd_loss = boundary_regularity_loss(metric, coords_scaled)
    pos_loss = metric_positivity_loss(metric)
    total = sum(res_losses.values()) + bnd_loss + pos_loss
    print(f"OK  (total = {total.item():.4f})")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# 6) Gradient flow
print("[6/6] Gradient flow...", end=" ")
try:
    total.backward()
    grad_norms = {}
    for name, p in siren.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    n_with_grad = sum(1 for v in grad_norms.values() if v > 0)
    n_total = len(grad_norms)
    print(f"OK  ({n_with_grad}/{n_total} params have non-zero gradients)")
    max_grad_name = max(grad_norms, key=grad_norms.get)
    print(f"       Max grad norm: {grad_norms[max_grad_name]:.4f} ({max_grad_name})")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("  ALL 6 VALIDATION CHECKS PASSED")
print("=" * 60)
