"""
train_apex_4d.py  –  The Ultimate 4D Holographic Quantum Gravity Training Pipeline
====================================================================================

Implements the **Two-Phase Curriculum Training Loop** mandated by Phase 5,
with ALL 15 battle-tested numerical fixes:

  Phase A  (Epoch 1 -> CURRICULUM_PHASE_A_EPOCHS)
  ───────────────────────────────────────────────
      FIX 14: PURE DATA FITTING — no Cauchy, no PDE, no Sommerfeld.
      * CNN Data Loss with amplitude weighting (20x on peaks).
      * t=0 emphasis lock (10x extra weight on t < 0.1 points).
      * NO physics constraints — prevents contradiction between
        boundary data (non-zero Gaussians) and Cauchy vacuum.

  Phase B  (Epoch CURRICULUM_PHASE_A_EPOCHS+1 -> TOTAL_EPOCHS)
  ────────────────────────────────────────────────────────────
      * Freeze the 3D CNN Encoder (preserves latent mapping).
      * FIX 15: Gradual warmup of physics losses over PDE_WARMUP_EPOCHS.
      * Activate Causal PDE with log1p (FIX 10) and time-dilation.
      * Activate Sommerfeld BCs with log1p (FIX 11).
      * Activate bulk-interior-only Cauchy (FIX 12).
      * Activate HRT + log1p causality (FIX 13).
      * Activate Alternating Quantum Entropy Tether.
      * FIX 9: NaN recovery — reloads best checkpoint, halves LR.

Designed for **Kaggle T4 GPU** (16 GB VRAM).  Mixed precision is OFF because
2nd-order autograd requires float32.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from data import load_data
from sampler import ApexDualSampler
from model import ConvEncoder3D, FiLMSiren
from physics import (
    causal_bizon_pde,
    causal_pinn_weights,
    sommerfeld_radiative_loss,
    bulk_cauchy_loss,
    hrt_covariant_area,
)
from quantum_tether import QuantumEntropyTether, quantum_entropy_tether_loss


# ====================================================================== #
#  PHASE A LOSS  (FIX 14: Pure Data)                                       #
# ====================================================================== #
def apex_loss_phase_a(encoder:   ConvEncoder3D,
                      siren:     FiLMSiren,
                      sampler:   ApexDualSampler,
                      bnd_input: torch.Tensor,
                      config:    type = Config):
    """
    Phase A: PURE DATA FITTING — no Cauchy, no momentum, no PDE.

    FIX 14: The old Phase A had bulk_cauchy_loss which enforced dt_phi(t=0)=0
    across all depths including the AdS boundary. But the boundary data at
    t=0 has non-zero initial conditions (Gaussian BH bumps). This contradiction
    caused the momentum term to grow to 10^12, drowning data gradients.

    Now: Phase A trains ONLY on boundary data + t=0 emphasis, letting the
    encoder and SIREN fully learn the waveform before physics constraints.
    """
    device = config.DEVICE

    # ---- Boundary Anchor Data Loss ----
    bnd_coords, bnd_vals = sampler.sample_discrete_boundary(config.BOUNDARY_BATCH)
    bnd_coords = bnd_coords.to(device)
    bnd_vals   = bnd_vals.to(device)

    z_latent     = encoder(bnd_input)                        # (1, LATENT_DIM)
    z_lat_expand = z_latent.expand(bnd_coords.shape[0], -1)

    phi_bnd   = siren(bnd_coords, z_lat_expand).squeeze(-1)  # (B,)

    # Amplitude weighting: 20x penalty on merger peaks
    err = (phi_bnd - bnd_vals) ** 2
    amp_weights = 1.0 + 20.0 * torch.abs(bnd_vals)
    data_loss = (err * amp_weights).mean()

    # t=0 emphasis lock: extra 10x weight on early timestep points
    t0_mask = (bnd_coords[:, 0] < 0.1)
    if t0_mask.any():
        t0_err = (phi_bnd[t0_mask] - bnd_vals[t0_mask]) ** 2
        t0_w   = 1.0 + 20.0 * torch.abs(bnd_vals[t0_mask])
        t0_loss = (t0_err * t0_w).mean() * 10.0
    else:
        t0_loss = torch.tensor(0.0, device=device)

    total = config.W_DATA * data_loss + t0_loss

    info = dict(data=data_loss.item(),
                t0_lock=t0_loss.item() if torch.is_tensor(t0_loss) else t0_loss,
                total=total.item())
    return total, info


# ====================================================================== #
#  PHASE B LOSS  (FIX 15: Warmup)                                         #
# ====================================================================== #
def apex_loss_phase_b(encoder:         ConvEncoder3D,
                      siren:           FiLMSiren,
                      sampler:         ApexDualSampler,
                      bnd_input:       torch.Tensor,
                      quantum_tether:  QuantumEntropyTether,
                      config:          type = Config,
                      epoch:           int = 0):
    """
    Phase B: Full physics with frozen CNN and gradual warmup.

    FIX 15: Physics constraints are multiplied by a warmup factor:
      warmup = min(1.0, (epoch - Phase_A_end) / PDE_WARMUP_EPOCHS)

    This prevents the gradient shock that caused PDE=10^29 at transition.
    z_latent is always detached for physics computations.
    """
    device = config.DEVICE
    phase_b_ep = max(1, epoch - config.CURRICULUM_PHASE_A_EPOCHS)
    warmup = min(1.0, phase_b_ep / max(config.PDE_WARMUP_EPOCHS, 1))

    with torch.no_grad():
        z_latent = encoder(bnd_input)
    z_det = z_latent.detach()

    # ---- 1. Boundary Anchor Data Loss (always full weight) ----
    bnd_coords, bnd_vals = sampler.sample_discrete_boundary(config.BOUNDARY_BATCH)
    bnd_coords = bnd_coords.to(device)
    bnd_vals   = bnd_vals.to(device)

    z_bnd = z_det.expand(bnd_coords.shape[0], -1)
    phi_bnd   = siren(bnd_coords, z_bnd).squeeze(-1)

    err = (phi_bnd - bnd_vals) ** 2
    amp_weights = 1.0 + 20.0 * torch.abs(bnd_vals)
    data_loss = (err * amp_weights).mean()

    # ---- 2. Causal PDE Residual (warmed up, log1p) ----
    bulk_coords, bulk_w = sampler.sample_continuous_bulk(config.BULK_BATCH)
    bulk_coords = bulk_coords.to(device).requires_grad_(True)
    bulk_w      = bulk_w.to(device)

    z_bulk      = z_det.expand(bulk_coords.shape[0], -1)
    phi_bulk_R  = siren(bulk_coords, z_bulk)

    residual, derivs = causal_bizon_pde(phi_bulk_R, bulk_coords, config)
    pde_loss = causal_pinn_weights(residual,
                                   bulk_coords[:, 0].detach(),
                                   config,
                                   volume_w=bulk_w.detach())

    # ---- 3. Cauchy (warmed up, bulk interior only) ----
    field_loss, mom_loss = bulk_cauchy_loss(siren, sampler, z_det, config)

    # ---- 4. Sommerfeld (warmed up, log1p) ----
    somm_loss = sommerfeld_radiative_loss(siren, sampler, z_det, config)

    # ---- 5. HRT Covariant Area + Causality (warmed up causality) ----
    hrt_coords, _ = sampler.sample_continuous_bulk(config.BULK_BATCH // 2)
    hrt_coords = hrt_coords.to(device).requires_grad_(True)
    z_hrt      = z_det.expand(hrt_coords.shape[0], -1)
    phi_hrt    = siren(hrt_coords, z_hrt)

    hrt_area, causal_pen = hrt_covariant_area(phi_hrt, hrt_coords, config)

    # ---- 6. Quantum Entropy Tether ----
    cached_S      = quantum_tether.get_cached_entropy().to(device)
    quantum_loss  = quantum_entropy_tether_loss(hrt_area, cached_S)

    # ---- Total: data at full weight, physics multiplied by warmup ----
    total = (config.W_DATA          * data_loss
             + warmup * (
                 config.W_PDE           * pde_loss
                 + config.W_CAUCHY      * field_loss
                 + config.W_CAUCHY_DT   * mom_loss
                 + config.W_SOMMERFELD  * somm_loss
                 + config.W_CAUSALITY_HRT * causal_pen
             )
             + config.W_HRT           * hrt_area
             + config.W_QUANTUM       * quantum_loss)

    info = dict(data=data_loss.item(),
                pde=pde_loss.item(),
                cauchy=field_loss.item(),
                momentum=mom_loss.item(),
                sommerfeld=somm_loss.item(),
                hrt=hrt_area.item(),
                causality=causal_pen.item(),
                quantum=quantum_loss.item(),
                warmup=warmup,
                total=total.item())
    return total, info


# ====================================================================== #
#  MAIN CURRICULUM TRAINING LOOP                                           #
# ====================================================================== #
def train_apex_curriculum(config: type = Config):
    device = config.DEVICE
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ================================================================== #
    #  1. DATA                                                             #
    # ================================================================== #
    print("=" * 72)
    print("  4D HOLOGRAPHIC QUANTUM GRAVITY ENGINE  –  6-Phase Pipeline")
    print("  All 15 numerical fixes active")
    print("=" * 72)
    print("\n[1/6] Loading collision-fluid boundary data ...")

    cnn_vol, bnd_coords, bnd_values, entropy_target, time_ticks, source = \
        load_data(config)

    if source == "master":
        bnd_input = cnn_vol
        while bnd_input.dim() < 5:
            bnd_input = bnd_input.unsqueeze(0)
        bnd_input = bnd_input.to(device)
        print(f"       Source: master dataset (apex_master_dataset.npz)")
    else:
        bnd_input = cnn_vol.unsqueeze(0).unsqueeze(0).to(device)
        print(f"       Source: synthetic (no .npz found)")

    print(f"       Encoder input shape: {tuple(bnd_input.shape)}")

    # ================================================================== #
    #  2. MODELS                                                           #
    # ================================================================== #
    print("\n[2/6] Building neural architecture ...")
    encoder = ConvEncoder3D(config).to(device)
    siren   = FiLMSiren(config).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_sir = sum(p.numel() for p in siren.parameters())
    print(f"       ConvEncoder3D  : {n_enc:>10,} params")
    print(f"       FiLM-SIREN     : {n_sir:>10,} params")
    print(f"       Total          : {n_enc + n_sir:>10,} params")

    # ================================================================== #
    #  3. SAMPLER                                                          #
    # ================================================================== #
    print("\n[3/6] Initialising ApexDualSampler ...")
    sampler = ApexDualSampler(bnd_coords, bnd_values, config)

    # ================================================================== #
    #  4. QUANTUM TETHER                                                   #
    # ================================================================== #
    print("\n[4/6] Initialising Quantum Entropy Tether ...")
    qtether = QuantumEntropyTether(config)

    # ================================================================== #
    #  5. OPTIMISERS & SCHEDULERS                                          #
    # ================================================================== #
    print("\n[5/6] Setting up optimisers ...")
    print(f"       FIX summary: omega_0={config.SIREN_OMEGA_0}  "
          f"kappa={config.KAPPA}/{config.KAPPA_MAX}  "
          f"eps={config.CAUSAL_EPSILON}  patience={config.SCHEDULER_PATIENCE}")
    print(f"       Phase A = PURE DATA (no cauchy/momentum/PDE)")
    print(f"       Phase B = log1p PDE + {config.PDE_WARMUP_EPOCHS}-epoch warmup "
          f"+ NaN recovery ({config.NAN_MAX_RECOVERIES} max)")

    opt_a = optim.Adam(list(encoder.parameters()) + list(siren.parameters()),
                       lr=config.LR_PHASE_A)
    opt_b = optim.Adam(siren.parameters(), lr=config.LR_PHASE_B)

    sched_a = optim.lr_scheduler.ReduceLROnPlateau(
        opt_a, patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR)
    sched_b = optim.lr_scheduler.ReduceLROnPlateau(
        opt_b, patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR)

    # ================================================================== #
    #  6. TRAINING LOOP                                                    #
    # ================================================================== #
    print("\n[6/6] Starting curriculum training ...\n")
    print("=" * 72)
    print(f"  PHASE A  –  Pure Data Fitting  (epochs 1 – "
          f"{config.CURRICULUM_PHASE_A_EPOCHS})")
    print("=" * 72)

    history    = []
    best_loss  = float("inf")
    nan_count  = 0
    accum      = config.GRADIENT_ACCUMULATION_STEPS

    for epoch in range(1, config.TOTAL_EPOCHS + 1):
        t0 = time.time()

        # ---------- Phase transition ----------
        if epoch == config.CURRICULUM_PHASE_A_EPOCHS + 1:
            print("\n" + "=" * 72)
            print(f"  PHASE B  –  Full Causal PDE + HRT + Quantum Tether  "
                  f"(epochs {epoch} – {config.TOTAL_EPOCHS})")
            print("  Freezing CNN Encoder weights ...")
            print("=" * 72 + "\n")

            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

            # Seed the quantum entropy cache
            with torch.no_grad():
                z0 = encoder(bnd_input).squeeze(0)
            qtether.update_cache(z0)

        # ---------- Phase A ----------
        if epoch <= config.CURRICULUM_PHASE_A_EPOCHS:
            encoder.train()
            siren.train()
            opt_a.zero_grad()

            loss, info = apex_loss_phase_a(
                encoder, siren, sampler, bnd_input, config)

            # FIX 9: NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  [!NaN] Phase A epoch {epoch}, "
                      f"recovery {nan_count}/{config.NAN_MAX_RECOVERIES}")
                if nan_count > config.NAN_MAX_RECOVERIES:
                    print("  [FATAL] Too many NaN events. Stopping.")
                    break
                ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=device,
                                      weights_only=False)
                    encoder.load_state_dict(ckpt['encoder'])
                    siren.load_state_dict(ckpt['siren'])
                for pg in opt_a.param_groups:
                    pg['lr'] *= 0.5
                opt_a.zero_grad()
                continue

            (loss / accum).backward()
            if epoch % accum == 0 or epoch == config.CURRICULUM_PHASE_A_EPOCHS:
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(siren.parameters()),
                    config.GRAD_CLIP)
                opt_a.step()
                opt_a.zero_grad()
            sched_a.step(info['total'])

        # ---------- Phase B ----------
        else:
            siren.train()
            opt_b.zero_grad()

            # Refresh quantum entropy cache periodically
            if (epoch - config.CURRICULUM_PHASE_A_EPOCHS) \
                    % config.QUANTUM_UPDATE_EVERY == 1:
                with torch.no_grad():
                    z0 = encoder(bnd_input).squeeze(0)
                qtether.update_cache(z0)

            loss, info = apex_loss_phase_b(
                encoder, siren, sampler, bnd_input, qtether, config, epoch=epoch)

            # FIX 9: NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  [!NaN] Phase B epoch {epoch}, "
                      f"recovery {nan_count}/{config.NAN_MAX_RECOVERIES}")
                if nan_count > config.NAN_MAX_RECOVERIES:
                    print("  [FATAL] Too many NaN events. Stopping.")
                    break
                ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=device,
                                      weights_only=False)
                    siren.load_state_dict(ckpt['siren'])
                for pg in opt_b.param_groups:
                    pg['lr'] *= 0.5
                opt_b.zero_grad()
                continue

            (loss / accum).backward()
            if epoch % accum == 0:
                nn.utils.clip_grad_norm_(siren.parameters(), config.GRAD_CLIP)
                opt_b.step()
                opt_b.zero_grad()
            sched_b.step(info['total'])

        dt = time.time() - t0
        history.append(info)

        # ---------- Logging ----------
        if epoch <= 5 or epoch % config.LOG_EVERY == 0:
            phase = "A" if epoch <= config.CURRICULUM_PHASE_A_EPOCHS else "B"
            lr_now = opt_a.param_groups[0]['lr'] if phase == "A" \
                     else opt_b.param_groups[0]['lr']

            # Diagnostic: peak phi amplitude
            with torch.no_grad():
                z_sample = encoder(bnd_input)
                sample_c, _ = sampler.sample_discrete_boundary(1000)
                phi_peak = siren(sample_c.to(device),
                                 z_sample.expand(1000, -1)).abs().max().item()

            parts = " | ".join(f"{k}={v:.5f}" for k, v in info.items())
            print(f"[{phase}] E{epoch:4d}/{config.TOTAL_EPOCHS}  "
                  f"lr={lr_now:.2e}  peak_phi={phi_peak:.3f}  |  "
                  f"{parts}  ({dt:.1f}s)")

        # ---------- Checkpointing ----------
        if not (torch.isnan(loss) or torch.isinf(loss)):
            if info['total'] < best_loss:
                best_loss = info['total']
                torch.save(dict(
                    epoch=epoch,
                    encoder=encoder.state_dict(),
                    siren=siren.state_dict(),
                    best_loss=best_loss,
                ), os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))

        if epoch % config.SAVE_EVERY == 0:
            torch.save(dict(
                epoch=epoch,
                encoder=encoder.state_dict(),
                siren=siren.state_dict(),
                opt_a=opt_a.state_dict(),
                opt_b=opt_b.state_dict(),
                history=history,
            ), os.path.join(config.CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pt"))

    # ================================================================== #
    #  DONE                                                                #
    # ================================================================== #
    print("\n" + "=" * 72)
    print(f"  TRAINING COMPLETE   |   best total loss = {best_loss:.8f}")
    print(f"  NaN recoveries used: {nan_count}/{config.NAN_MAX_RECOVERIES}")
    print("=" * 72)

    # Save final model
    torch.save(dict(
        epoch=config.TOTAL_EPOCHS,
        encoder=encoder.state_dict(),
        siren=siren.state_dict(),
        history=history,
    ), os.path.join(config.CHECKPOINT_DIR, "final_model.pt"))

    return encoder, siren, history


# ====================================================================== #
#  ENTRY POINT                                                             #
# ====================================================================== #
if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU detected : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"VRAM         : {mem:.1f} GB")
    else:
        print("WARNING: No CUDA GPU detected – falling back to CPU.")
        Config.DEVICE = "cpu"

    train_apex_curriculum(Config)
