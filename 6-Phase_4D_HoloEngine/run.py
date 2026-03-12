"""
run.py  –  Unified Entry Point for the 4D Holographic Quantum Gravity Engine
==============================================================================

Commands  (Scalar Engine)
-------------------------
    python run.py train       – Run the full 2-phase curriculum training loop.
    python run.py evaluate    – Load best checkpoint and evaluate all metrics.
    python run.py visualize   – Generate all diagnostic plots from checkpoint.
    python run.py generate    – Regenerate the apex_master_dataset.npz.
    python run.py all         – Train, then evaluate, then visualize.

Commands  (Full Einstein BBH Engine)
------------------------------------
    python run.py bbh_train   – 3-phase curriculum: Phase A→B→C.
    python run.py bbh_status  – Show BBH checkpoint info + diagnostics.

Options
-------
    --device cpu|cuda         – Force device (auto-detected by default).
    --checkpoint PATH         – Checkpoint file for evaluate/visualize.
    --epochs N                – Override TOTAL_EPOCHS.
"""

import argparse
import sys
import torch

from config import Config


def cmd_train(args, config):
    """Run the full curriculum training pipeline."""
    from train_apex_4d import train_apex_curriculum
    if args.epochs:
        config.TOTAL_EPOCHS = args.epochs
    train_apex_curriculum(config)


def cmd_evaluate(args, config):
    """Run post-training evaluation."""
    from evaluate import run_full_evaluation
    run_full_evaluation(args.checkpoint, config, config.DEVICE)


def cmd_visualize(args, config):
    """Generate all diagnostic visualizations."""
    from visualize import (
        load_models, plot_loss_curves, plot_boundary_reconstruction,
        plot_radial_slices, plot_bulk_cross_sections, plot_chirp,
        animate_boundary,
    )
    from data import load_data

    encoder, siren, ckpt = load_models(args.checkpoint, config, config.DEVICE)
    history = ckpt.get("history", [])

    cnn_vol, _, _, _, _, source = load_data(config)
    if source == "master":
        bnd_input = cnn_vol
        while bnd_input.dim() < 5:
            bnd_input = bnd_input.unsqueeze(0)
        bnd_input = bnd_input.to(config.DEVICE)
    else:
        bnd_input = cnn_vol.unsqueeze(0).unsqueeze(0).to(config.DEVICE)

    if history:
        plot_loss_curves(history)
    plot_boundary_reconstruction(encoder, siren, config, config.DEVICE)
    plot_radial_slices(encoder, siren, bnd_input, config, config.DEVICE)
    plot_bulk_cross_sections(encoder, siren, bnd_input, config, config.DEVICE)
    plot_chirp(encoder, siren, bnd_input, config, config.DEVICE)
    animate_boundary(encoder, siren, bnd_input, config, config.DEVICE)
    print("\n[run] All visualizations saved to plots/")


def cmd_generate(args, config):
    """Regenerate the master dataset."""
    from generate_apex_master_dataset import generate_dataset
    generate_dataset()


def cmd_all(args, config):
    """Train, evaluate, then visualize."""
    cmd_train(args, config)
    args.checkpoint = "checkpoints/best_model.pt"
    cmd_evaluate(args, config)
    cmd_visualize(args, config)


# ====================================================================== #
#  BBH ENGINE COMMANDS                                                     #
# ====================================================================== #
def cmd_bbh_train(args, config):
    """Run the full 3-phase Einstein BBH training."""
    from train_bbh import train
    train()


def cmd_bbh_status(args, config):
    """Show BBH checkpoint info and diagnostics."""
    import os
    from ads_config import BBHConfig
    ckpt_dir = BBHConfig.CHECKPOINT_DIR
    best = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best):
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        print(f"[BBH] Best checkpoint: epoch {ckpt['epoch']}, "
              f"loss = {ckpt['loss']:.6e}")
    else:
        print(f"[BBH] No checkpoint found at {best}")
    # List all checkpoints
    if os.path.isdir(ckpt_dir):
        files = sorted(os.listdir(ckpt_dir))
        print(f"[BBH] Checkpoints ({len(files)}): {', '.join(files)}")


def main():
    parser = argparse.ArgumentParser(
        description="4D Holographic Quantum Gravity Engine")
    parser.add_argument("command", choices=["train", "evaluate", "visualize",
                                            "generate", "all",
                                            "bbh_train", "bbh_status"],
                        help="Action to perform")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu or cuda)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model.pt",
                        help="Checkpoint path for evaluate/visualize")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total epochs for training")
    args = parser.parse_args()

    config = Config

    # Device selection
    if args.device:
        config.DEVICE = args.device
    elif not torch.cuda.is_available():
        config.DEVICE = "cpu"

    if config.DEVICE == "cuda" and torch.cuda.is_available():
        print(f"[run] GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
    else:
        if config.DEVICE == "cuda":
            print("[run] WARNING: CUDA requested but not available. Using CPU.")
            config.DEVICE = "cpu"
        print("[run] Device: CPU")

    dispatch = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "generate": cmd_generate,
        "all": cmd_all,
        "bbh_train": cmd_bbh_train,
        "bbh_status": cmd_bbh_status,
    }

    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
