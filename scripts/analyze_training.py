#!/usr/bin/env python3.11
"""
Analyze Training Results

Examines checkpoint files and training metrics to understand model behavior.

Usage:
    python3.11 scripts/analyze_training.py checkpoints/real_mediapipe_cached
"""

import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def analyze_checkpoint(checkpoint_path: str):
    """Analyze a single checkpoint file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {Path(checkpoint_path).name}")
    print(f"{'='*70}\n")

    # Load checkpoint (weights_only=False since we trust our own checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Basic info
    print("Checkpoint Contents:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")

    # Loss history
    if 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"\nLoss History:")
        print(f"  Train losses: {len(train_losses)} epochs")
        print(f"  Val losses: {len(val_losses)} epochs")
        print(f"  First train loss: {train_losses[0]:.4f}")
        print(f"  Last train loss: {train_losses[-1]:.4f}")
        print(f"  Best val loss: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")

    # Model info
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\nModel Parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Number of layers: {len(state_dict)}")

    # Config info
    if 'encoder_config' in checkpoint:
        enc_config = checkpoint['encoder_config']
        print(f"\nEncoder Configuration:")
        print(f"  Input type: {enc_config.input_type}")
        print(f"  Input dim: {enc_config.input_dim}")
        print(f"  Hidden dim: {enc_config.hidden_dim}")
        print(f"  Num layers: {enc_config.num_layers}")
        print(f"  Bidirectional: {enc_config.bidirectional}")

    if 'ctc_config' in checkpoint:
        ctc_config = checkpoint['ctc_config']
        print(f"\nCTC Configuration:")
        print(f"  Vocabulary size: {ctc_config.vocab_size}")
        print(f"  Blank ID: {ctc_config.blank_id}")

    return checkpoint


def plot_training_curves(checkpoint_dir: str, output_path: str = None):
    """Plot training and validation loss curves."""
    checkpoint_dir = Path(checkpoint_dir)

    # Find all checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Load the latest checkpoint for full history
    latest_checkpoint = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)

    train_losses = latest_checkpoint['train_losses']
    val_losses = latest_checkpoint['val_losses']
    epochs = list(range(1, len(train_losses) + 1))

    # Create plot
    plt.figure(figsize=(12, 6))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Log scale (helpful for CTC loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Loss Curves (Log Scale)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_path}")
    else:
        # Save to checkpoint directory
        output_path = checkpoint_dir / "training_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_path}")

    # Also display statistics
    print(f"\n{'='*70}")
    print("Training Statistics")
    print(f"{'='*70}")
    print(f"Total epochs: {len(epochs)}")
    print(f"\nTrain Loss:")
    print(f"  Initial: {train_losses[0]:.4f}")
    print(f"  Final: {train_losses[-1]:.4f}")
    print(f"  Reduction: {train_losses[0] - train_losses[-1]:.4f} ({(1 - train_losses[-1]/train_losses[0])*100:.1f}%)")
    print(f"  Min: {min(train_losses):.4f} (epoch {train_losses.index(min(train_losses)) + 1})")

    print(f"\nVal Loss:")
    print(f"  Initial: {val_losses[0]:.4f}")
    print(f"  Final: {val_losses[-1]:.4f}")
    print(f"  Best: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")

    # Check for overfitting
    best_val_idx = val_losses.index(min(val_losses))
    if best_val_idx < len(val_losses) - 3:
        print(f"\n⚠ Warning: Val loss hasn't improved since epoch {best_val_idx + 1}")
        print(f"  This suggests overfitting or learning plateau")
    else:
        print(f"\n✓ Model still improving (best val loss in recent epochs)")

    print(f"{'='*70}\n")


def compare_checkpoints(checkpoint_dir: str):
    """Compare all checkpoints to find the best one."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"\n{'='*70}")
    print("Checkpoint Comparison")
    print(f"{'='*70}\n")

    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status'}")
    print("-" * 60)

    best_val_loss = float('inf')
    best_epoch = -1

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        epoch = ckpt['epoch']
        train_loss = ckpt['train_losses'][-1] if 'train_losses' in ckpt else 0
        val_loss = ckpt['val_loss']

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            status = "← BEST"

        print(f"{epoch:<8} {train_loss:<15.4f} {val_loss:<15.4f} {status}")

    print(f"\n✓ Best checkpoint: epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"{'='*70}\n")


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('checkpoint_dir', type=str,
                       help='Directory containing checkpoints')
    parser.add_argument('--plot', action='store_true',
                       help='Generate loss curve plots')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all checkpoints')
    parser.add_argument('--inspect', type=str, default=None,
                       help='Inspect specific checkpoint file')

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"Error: Directory not found: {checkpoint_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Training Analysis")
    print(f"{'='*70}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Default: show everything
    if not (args.plot or args.compare or args.inspect):
        args.plot = True
        args.compare = True

    if args.compare:
        compare_checkpoints(checkpoint_dir)

    if args.plot:
        plot_training_curves(checkpoint_dir)

    if args.inspect:
        checkpoint_path = args.inspect
        if not Path(checkpoint_path).is_absolute():
            checkpoint_path = checkpoint_dir / checkpoint_path
        analyze_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()
