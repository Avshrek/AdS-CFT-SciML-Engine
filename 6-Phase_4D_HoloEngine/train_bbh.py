"""
train_bbh.py — 3-Phase Curriculum Training for Full Einstein BBH Merger
========================================================================

Phase A  (epochs 0–500):     Pure AdS metric + boundary data enforcement
Phase B  (epochs 500–1500):  Nested radial Einstein equations (R1-R4, KG)
Phase C  (epochs 1500–3000): Full evolution (E1) + horizon + quantum tether

The curriculum ensures the network first learns the correct vacuum solution
and boundary conditions, then gradually activates the Einstein equations,
and finally turns on the time-evolution and horizon physics.

This is the TRAINING LOOP for the full characteristic-formulation
Einstein solver — NOT the scalar engine trainer.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from ads_config import BBHConfig as cfg
from metric_model import MetricSIREN, MetricReconstructor, BoundaryEncoder
from einstein_equations import (
    compute_all_einstein_residuals,
    einstein_residual_loss,
    boundary_regularity_loss,
    metric_positivity_loss,
)
from bbh_initial_data import initial_data_loss
from bbh_sampler import BBHBatchSampler, causal_weight
from horizon import find_apparent_horizon, horizon_regularity_loss
from observables import (
    extract_boundary_stress_tensor,
    extract_gravitational_waveform,
    check_energy_conservation,
)
from cft_quantum_state import CFTQuantumState, hrt_quantum_consistency_loss


# ====================================================================== #
#  PHASE DETERMINATION                                                     #
# ====================================================================== #
def get_phase(epoch: int) -> str:
    """Return current training phase A/B/C based on epoch."""
    if epoch < cfg.PHASE_A_EPOCHS:
        return 'A'
    elif epoch < cfg.PHASE_A_EPOCHS + cfg.PHASE_B_EPOCHS:
        return 'B'
    else:
        return 'C'


def get_lr(phase: str) -> float:
    """Return learning rate for current phase."""
    return {'A': cfg.LR_PHASE_A, 'B': cfg.LR_PHASE_B, 'C': cfg.LR_PHASE_C}[phase]


def pde_warmup_factor(epoch: int) -> float:
    """Gradual ramp-up of PDE loss weight during first PDE_WARMUP_EPOCHS."""
    phase = get_phase(epoch)
    if phase == 'A':
        return 0.0  # No PDE in Phase A
    phase_b_start = cfg.PHASE_A_EPOCHS
    pde_epoch = epoch - phase_b_start
    if pde_epoch < cfg.PDE_WARMUP_EPOCHS:
        return pde_epoch / cfg.PDE_WARMUP_EPOCHS
    return 1.0


# ====================================================================== #
#  COMPUTE TOTAL LOSS FOR ONE STEP                                         #
# ====================================================================== #
def compute_loss(siren: MetricSIREN, encoder: BoundaryEncoder,
                 reconstructor: MetricReconstructor,
                 batch: dict, epoch: int,
                 boundary_input: torch.Tensor,
                 quantum_state: CFTQuantumState = None) -> dict:
    """
    Compute all loss terms for the current training phase.

    Returns
    -------
    losses : dict  {name: scalar tensor}
    total  : scalar tensor (weighted sum)
    """
    phase = get_phase(epoch)
    warmup = pde_warmup_factor(epoch)
    losses = {}

    z_latent = encoder(boundary_input)
    z_latent_expanded = z_latent

    # ---- BOUNDARY LOSS (all phases) ----
    bnd_coords = batch['boundary']
    z_lat_bnd = z_latent_expanded.expand(bnd_coords.shape[0], -1)
    raw_bnd = siren(bnd_coords, z_lat_bnd)
    metric_bnd = reconstructor.reconstruct(raw_bnd, bnd_coords)
    losses['boundary'] = boundary_regularity_loss(metric_bnd, bnd_coords) * cfg.W_BOUNDARY

    # ---- CAUCHY / INITIAL DATA LOSS (all phases) ----
    cauchy_coords = batch['cauchy']
    z_lat_cau = z_latent_expanded.expand(cauchy_coords.shape[0], -1)
    raw_cau = siren(cauchy_coords, z_lat_cau)
    metric_cau = reconstructor.reconstruct(raw_cau, cauchy_coords)
    predicted_cau = torch.cat([
        metric_cau['A'], metric_cau['Sigma'], metric_cau['B'],
        metric_cau['V_shift'], metric_cau['phi']
    ], dim=-1)
    losses['initial_data'] = initial_data_loss(predicted_cau, cauchy_coords) * cfg.W_PURE_ADS

    # ---- CONSTRAINT DAMPING at initial surface (v ≈ 0) ----
    v_cau = cauchy_coords[:, 0:1]
    cau_near_zero = (v_cau < 0.1).float().mean()
    if cau_near_zero > 0 and warmup > 0:
        res_cau, _ = compute_all_einstein_residuals(metric_cau, cauchy_coords)
        losses['constraint_damp'] = (
            torch.log1p(res_cau['constraint_res'].pow(2).mean())
            * cfg.W_CONSTRAINT_DAMP * warmup
        )

    # ---- POSITIVITY CONSTRAINT (all phases) ----
    losses['positivity'] = metric_positivity_loss(metric_bnd)

    # ---- LORENTZIAN SIGNATURE ENFORCEMENT (Phase B + C) ----
    if phase in ('B', 'C'):
        A_bnd = metric_bnd['A']
        S_bnd = metric_bnd['Sigma']
        sig_viol = torch.relu(-A_bnd).pow(2).mean() + torch.relu(-S_bnd + 1e-3).pow(2).mean()
        losses['causality'] = torch.log1p(sig_viol * 100.0) * cfg.W_CAUSALITY

    # ---- BULK EINSTEIN EQUATIONS with per-point causal weighting (Phase B + C) ----
    res_losses = None
    if phase in ('B', 'C') and warmup > 0:
        bulk_coords = batch['bulk']
        z_lat_bulk = z_latent_expanded.expand(bulk_coords.shape[0], -1)
        raw_bulk = siren(bulk_coords, z_lat_bulk)
        metric_bulk = reconstructor.reconstruct(raw_bulk, bulk_coords)

        residuals, _ = compute_all_einstein_residuals(metric_bulk, bulk_coords)

        v_bulk = bulk_coords[:, 0].detach()
        chunk_idx = min(int(epoch / (cfg.TOTAL_EPOCHS / cfg.NUM_TIME_CHUNKS)),
                        cfg.NUM_TIME_CHUNKS - 1)
        cw = causal_weight(v_bulk, chunk_idx, cfg)  # per-point [N]
        res_losses = einstein_residual_loss(residuals, weights=cw)

        for name in ['sigma_res', 'aniso_res', 'shift_res', 'lapse_res']:
            losses[name] = res_losses[name] * cfg.W_EINSTEIN_RADIAL * warmup
        losses['kg_res'] = res_losses['kg_res'] * cfg.W_SCALAR_EQ * warmup
        losses['constraint_res'] = res_losses['constraint_res'] * cfg.W_CONSTRAINT * warmup

    # ---- RADIAL LINE ENFORCEMENT (Phase B + C) ----
    if phase in ('B', 'C') and warmup > 0 and 'radial' in batch:
        rad_coords = batch['radial']
        z_lat_rad = z_latent_expanded.expand(rad_coords.shape[0], -1)
        raw_rad = siren(rad_coords, z_lat_rad)
        metric_rad = reconstructor.reconstruct(raw_rad, rad_coords)

        rad_residuals, _ = compute_all_einstein_residuals(metric_rad, rad_coords)
        rad_losses = einstein_residual_loss(rad_residuals)
        rad_total = sum(rad_losses.values()) * cfg.W_EINSTEIN_RADIAL * warmup * 0.5
        losses['radial_einstein'] = torch.nan_to_num(rad_total, nan=0.0, posinf=1e4, neginf=0.0)

    # ---- EVOLUTION EQUATIONS — ALL 4 with causal weighting (Phase C) ----
    if phase == 'C' and res_losses is not None:
        losses['evolution_res']   = res_losses['evolution_res']   * cfg.W_EVOLUTION
        losses['evolution_B_res'] = res_losses['evolution_B_res'] * cfg.W_EVOLUTION_B
        losses['evolution_V_res'] = res_losses['evolution_V_res'] * cfg.W_EVOLUTION_V
        losses['evolution_A_res'] = res_losses['evolution_A_res'] * cfg.W_EVOLUTION_A

    # ---- ADAPTIVE near-horizon PDE enforcement (Phase C) ----
    if phase == 'C' and warmup > 0 and 'adaptive' in batch:
        adp_coords = batch['adaptive']
        z_lat_adp = z_latent_expanded.expand(adp_coords.shape[0], -1)
        raw_adp = siren(adp_coords, z_lat_adp)
        metric_adp = reconstructor.reconstruct(raw_adp, adp_coords)
        adp_res, _ = compute_all_einstein_residuals(metric_adp, adp_coords)
        adp_losses = einstein_residual_loss(adp_res)
        adp_total = sum(adp_losses.values()) * cfg.W_EINSTEIN_RADIAL * warmup * 0.3
        losses['adaptive_pde'] = torch.nan_to_num(adp_total, nan=0.0, posinf=1e4, neginf=0.0)

    # ---- HORIZON REGULARITY (Phase C only) ----
    if phase == 'C' and 'horizon' in batch:
        hor_coords = batch['horizon']
        z_lat_hor = z_latent_expanded.expand(hor_coords.shape[0], -1)
        raw_hor = siren(hor_coords, z_lat_hor)
        metric_hor = reconstructor.reconstruct(raw_hor, hor_coords)
        losses['horizon_reg'] = horizon_regularity_loss(
            metric_hor, hor_coords
        ) * cfg.W_HORIZON_REG

    # ---- QUANTUM ENTROPY TETHER (Phase C, every N epochs) ----
    if phase == 'C' and quantum_state is not None:
        cached_S = quantum_state.get_cached_entropy()
        if cached_S.abs() > 1e-8:
            losses['quantum_tether'] = cached_S * cfg.W_QUANTUM * 0.01

    # ---- TOTAL ----
    total = sum(losses.values())

    return losses, total


# ====================================================================== #
#  SYNTHETIC BOUNDARY DATA  (placeholder until real CFT data exists)       #
# ====================================================================== #
def generate_synthetic_boundary(cfg_cls: type = cfg) -> torch.Tensor:
    """
    Create synthetic boundary stress-tensor data for the encoder.

    In a production setup, this would come from the holographic
    dictionary or observational data. Here we use the analytic
    BBH initial data extrapolated to the boundary.

    Returns: (1, 1, N_v, N_x, 1) tensor
    """
    n_v = cfg_cls.ENCODER_TEMPORAL_FRAMES
    n_x = cfg_cls.ENCODER_SPATIAL_RES

    v = torch.linspace(cfg_cls.V_RANGE[0], cfg_cls.V_RANGE[1], n_v)
    x = torch.linspace(cfg_cls.X_RANGE[0], cfg_cls.X_RANGE[1], n_x)

    # Two-BH energy profile at boundary (z → 0):
    # <T_vv> ∝ μ₁(x) + μ₂(x) at v=0, evolving in time
    from bbh_initial_data import energy_profile
    mu_x = (energy_profile(x, cfg_cls.BH_MASS_1, cfg_cls.BH_POSITION_1,
                           cfg_cls.BH_BOOST_1, cfg_cls.BH_WIDTH)
            + energy_profile(x, cfg_cls.BH_MASS_2, cfg_cls.BH_POSITION_2,
                             cfg_cls.BH_BOOST_2, cfg_cls.BH_WIDTH))

    # Time evolution: BHs move toward each other
    data = torch.zeros(n_v, n_x)
    for i, vi in enumerate(v):
        shift = vi * 0.5  # merger timescale
        mu_evolved = (
            energy_profile(x, cfg_cls.BH_MASS_1,
                           cfg_cls.BH_POSITION_1 + cfg_cls.BH_BOOST_1 * vi.item(),
                           cfg_cls.BH_BOOST_1, cfg_cls.BH_WIDTH)
            + energy_profile(x, cfg_cls.BH_MASS_2,
                             cfg_cls.BH_POSITION_2 + cfg_cls.BH_BOOST_2 * vi.item(),
                             cfg_cls.BH_BOOST_2, cfg_cls.BH_WIDTH)
        )
        data[i, :] = mu_evolved

    # Normalize
    data = data / (data.max() + 1e-8)

    return data.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,N_v,N_x,1)


# ====================================================================== #
#  CHECKPOINT MANAGEMENT                                                   #
# ====================================================================== #
def save_checkpoint(siren, encoder, optimizer, epoch, loss,
                    path: str):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'siren_state': siren.state_dict(),
        'encoder_state': encoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(path: str, siren, encoder, optimizer=None):
    """Load checkpoint if it exists. Returns epoch to resume from."""
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location=cfg.DEVICE, weights_only=False)
    siren.load_state_dict(ckpt['siren_state'])
    encoder.load_state_dict(ckpt['encoder_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']}, loss={ckpt['loss']:.6f}")
    return ckpt['epoch']


# ====================================================================== #
#  MAIN TRAINING LOOP                                                      #
# ====================================================================== #
def train():
    """Full 3-phase BBH training."""
    device = cfg.DEVICE
    print("=" * 70)
    print("  FULL EINSTEIN BBH MERGER ENGINE — Characteristic Formulation")
    print(f"  Device: {device}")
    print(f"  Phases: A({cfg.PHASE_A_EPOCHS}) → B({cfg.PHASE_B_EPOCHS}) → "
          f"C({cfg.TOTAL_EPOCHS - cfg.PHASE_A_EPOCHS - cfg.PHASE_B_EPOCHS})")
    print("=" * 70)

    # ---- Build models ----
    siren = MetricSIREN(cfg).to(device)
    encoder = BoundaryEncoder(cfg).to(device)
    reconstructor = MetricReconstructor(cfg)

    n_params = sum(p.numel() for p in siren.parameters())
    n_enc = sum(p.numel() for p in encoder.parameters())
    print(f"[Model] MetricSIREN: {n_params:,} params")
    print(f"[Model] BoundaryEncoder: {n_enc:,} params")
    print(f"[Model] Total: {n_params + n_enc:,} params")

    # ---- Optimizer ----
    all_params = list(siren.parameters()) + list(encoder.parameters())
    optimizer = optim.AdamW(all_params, lr=cfg.LR_PHASE_A, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg.SCHEDULER_PATIENCE,
        factor=cfg.SCHEDULER_FACTOR, min_lr=1e-6,
    )

    # ---- Generate boundary data ----
    boundary_input = generate_synthetic_boundary().to(device)
    print(f"[Data] Boundary input shape: {boundary_input.shape}")

    # ---- Sampler ----
    sampler = BBHBatchSampler(cfg, device)

    # ---- Quantum state (Phase C) ----
    quantum_state = CFTQuantumState(cfg)

    # ---- Checkpoint resume ----
    ckpt_dir = cfg.CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    start_epoch = load_checkpoint(best_path, siren, encoder, optimizer)

    # ---- Training state ----
    best_loss = float('inf')
    nan_recoveries = 0
    current_phase = 'A'

    print(f"\n[Train] Starting from epoch {start_epoch}")
    t0 = time.time()

    for epoch in range(start_epoch, cfg.TOTAL_EPOCHS):
        phase = get_phase(epoch)

        # ---- Phase transition ----
        if phase != current_phase:
            new_lr = get_lr(phase)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            # Reset scheduler on phase transition so it doesn't carry stale state
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=cfg.SCHEDULER_PATIENCE,
                factor=cfg.SCHEDULER_FACTOR, min_lr=1e-6,
            )
            print(f"\n{'='*60}")
            print(f"  PHASE {phase} ACTIVATED  (epoch {epoch}, lr={new_lr:.1e})")
            print(f"{'='*60}")
            current_phase = phase

        # ---- Sample batch ----
        batch = sampler.sample(phase)

        # ---- Forward + loss ----
        optimizer.zero_grad()

        try:
            losses, total = compute_loss(
                siren, encoder, reconstructor, batch, epoch,
                boundary_input, quantum_state
            )
        except RuntimeError as e:
            if 'nan' in str(e).lower() or 'inf' in str(e).lower():
                nan_recoveries += 1
                if nan_recoveries > cfg.NAN_MAX_RECOVERIES:
                    print(f"[FATAL] Too many NaN recoveries ({nan_recoveries})")
                    break
                print(f"[NaN Recovery #{nan_recoveries}] Skipping epoch {epoch}")
                optimizer.zero_grad()
                continue
            raise

        # ---- NaN check ----
        if torch.isnan(total) or torch.isinf(total):
            nan_recoveries += 1
            if nan_recoveries > cfg.NAN_MAX_RECOVERIES:
                print(f"[FATAL] Too many NaN occurrences ({nan_recoveries})")
                break
            print(f"[NaN #{nan_recoveries}] Loss={total.item():.4e} at epoch {epoch}")
            optimizer.zero_grad()
            continue

        # ---- Backward ----
        total.backward()
        nn.utils.clip_grad_norm_(all_params, cfg.GRAD_CLIP)
        optimizer.step()
        # Only step scheduler after warmup completes
        if pde_warmup_factor(epoch) >= 1.0 or phase == 'A':
            scheduler.step(total.item())

        # ---- Quantum entropy update (Phase C, every N epochs) ----
        if phase == 'C' and epoch % cfg.QUANTUM_UPDATE_EVERY == 0:
            with torch.no_grad():
                z_lat = encoder(boundary_input)
            quantum_state.update_cache(z_lat.squeeze(0))

        # ---- Horizon estimate update (Phase C, every 50 epochs) ----
        if phase == 'C' and epoch % 50 == 0:
            try:
                with torch.no_grad():
                    horizon_info = find_apparent_horizon(
                        siren, encoder, boundary_input, reconstructor, cfg
                    )
                    if horizon_info['found']:
                        valid = ~torch.isnan(horizon_info['z_AH'])
                        if valid.any():
                            z_h_mean = horizon_info['z_AH'][valid].mean().item()
                            sampler.update_horizon_estimate(z_h_mean)
            except Exception:
                pass  # Horizon finding can fail early in training

        # ---- Logging ----
        if epoch % cfg.LOG_EVERY == 0:
            lr_now = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0
            loss_str = " | ".join(
                f"{k}={v.item():.3e}" for k, v in sorted(losses.items())
            )
            print(f"[{phase}] E{epoch:05d} | total={total.item():.4e} | "
                  f"lr={lr_now:.1e} | {elapsed:.0f}s | {loss_str}")

        # ---- Best model + checkpoints ----
        if total.item() < best_loss:
            best_loss = total.item()
            save_checkpoint(siren, encoder, optimizer, epoch, best_loss,
                            best_path)

        if epoch > 0 and epoch % cfg.SAVE_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{epoch}.pt")
            save_checkpoint(siren, encoder, optimizer, epoch, total.item(),
                            ckpt_path)

    # ---- Final save ----
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    save_checkpoint(siren, encoder, optimizer, epoch, total.item(), final_path)

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — Best loss: {best_loss:.6e}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"{'='*70}")

    # ---- Final diagnostics ----
    print("\n[Diagnostics] Post-training analysis...")
    with torch.no_grad():
        # Stress tensor
        try:
            T = extract_boundary_stress_tensor(
                siren, encoder, boundary_input, reconstructor, cfg
            )
            print(f"  <T_vv> range: [{T['T_vv'].min():.4f}, {T['T_vv'].max():.4f}]")
        except Exception as e:
            print(f"  Stress tensor extraction failed: {e}")

        # Gravitational waveform
        try:
            wf = extract_gravitational_waveform(
                siren, encoder, boundary_input, reconstructor, cfg
            )
            print(f"  h+ waveform: {wf['h_plus'].shape}, "
                  f"peak |h+| = {wf['h_plus'].abs().max():.6f}")
        except Exception as e:
            print(f"  Waveform extraction failed: {e}")

        # Ward identity (requires stress tensor dict)
        try:
            if 'T' in dir():
                ward = check_energy_conservation(T, cfg)
                print(f"  Ward identity max violation: {ward['max_violation']:.6e}")
                print(f"  Ward identity mean violation: {ward['mean_violation']:.6e}")
        except Exception as e:
            print(f"  Ward identity check failed: {e}")

        # Horizon + surface gravity + HRT
        h_info = None
        try:
            h_info = find_apparent_horizon(
                siren, encoder, boundary_input, reconstructor, cfg
            )
            if h_info['found']:
                print(f"  Horizon: {h_info['n_components']} component(s), "
                      f"entropy = {h_info['entropy']:.6f}")
                try:
                    from horizon import compute_surface_gravity
                    T_BH = compute_surface_gravity(
                        siren, encoder, boundary_input, reconstructor, h_info, cfg)
                    print(f"  Surface gravity → T_BH = {T_BH:.6f}")
                except Exception as e:
                    print(f"  Surface gravity failed: {e}")
            else:
                print("  Horizon: not found (network may need more training)")
        except Exception as e:
            print(f"  Horizon finder failed: {e}")

        try:
            from horizon import hrt_entanglement_entropy
            s_ee = hrt_entanglement_entropy(
                siren, encoder, boundary_input, reconstructor,
                cfg=cfg, horizon_data=h_info)
            print(f"  HRT entanglement entropy: {s_ee:.4f}")
        except Exception as e:
            print(f"  HRT failed: {e}")


# ====================================================================== #
#  ENTRY POINT                                                             #
# ====================================================================== #
if __name__ == "__main__":
    train()
