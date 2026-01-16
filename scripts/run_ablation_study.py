#!/usr/bin/env python3.11
"""
Run ablation study to validate pre-training benefit.

Experiments:
- 3 baseline runs (random encoder initialization) with seeds 42, 43, 44
- 3 pre-trained runs (Stage 1 encoder) with seeds 42, 43, 44

Total: 6 runs × ~30 min = ~3 hours
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import time

# Configuration
SEEDS = [42, 43, 44]
NUM_EPOCHS = 30
DEVICE = "cuda"
HIDDEN_DIM = 128  # Must match Stage 1 for pre-trained loading
NUM_LAYERS = 2     # Must match Stage 1 for pre-trained loading
BATCH_SIZE = 8

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_CACHE = PROJECT_ROOT / "data/cache/phase1/train_20glosses.pkl"
VAL_CACHE = PROJECT_ROOT / "data/cache/phase1/val_20glosses.pkl"
PRETRAINED_ENCODER = PROJECT_ROOT / "checkpoints/stage1/checkpoint_best.pt"
RESULTS_DIR = PROJECT_ROOT / "checkpoints/ablation"

# Check prerequisites
if not TRAIN_CACHE.exists():
    print(f"❌ Error: Training cache not found at {TRAIN_CACHE}")
    sys.exit(1)

if not VAL_CACHE.exists():
    print(f"❌ Error: Validation cache not found at {VAL_CACHE}")
    sys.exit(1)

if not PRETRAINED_ENCODER.exists():
    print(f"❌ Error: Pre-trained encoder not found at {PRETRAINED_ENCODER}")
    sys.exit(1)

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_experiment(condition: str, seed: int, use_pretrained: bool) -> dict:
    """Run a single training experiment."""

    checkpoint_dir = RESULTS_DIR / f"{condition}_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python3.11",
        str(PROJECT_ROOT / "src/training/stage2_ctc.py"),
        "--feature-cache", str(TRAIN_CACHE),
        "--val-feature-cache", str(VAL_CACHE),
        "--checkpoint-dir", str(checkpoint_dir),
        "--device", DEVICE,
        "--num-epochs", str(NUM_EPOCHS),
        "--hidden-dim", str(HIDDEN_DIM),
        "--num-layers", str(NUM_LAYERS),
        "--batch-size", str(BATCH_SIZE),
        "--seed", str(seed),
    ]

    if use_pretrained:
        cmd.extend(["--pretrained-encoder", str(PRETRAINED_ENCODER)])

    print(f"\n{'='*70}")
    print(f"Running: {condition} (seed={seed})")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time

        # Try to read final results from checkpoint
        best_checkpoint = checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            import torch
            checkpoint = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
            val_loss = checkpoint.get('val_loss', None)

            # Try to get accuracy from validation (need to parse from logs or checkpoint)
            # For now, we'll read from the checkpoint directory later

            return {
                'condition': condition,
                'seed': seed,
                'use_pretrained': use_pretrained,
                'success': True,
                'elapsed_time': elapsed,
                'checkpoint_dir': str(checkpoint_dir),
                'val_loss': val_loss,
            }
        else:
            return {
                'condition': condition,
                'seed': seed,
                'use_pretrained': use_pretrained,
                'success': True,
                'elapsed_time': elapsed,
                'checkpoint_dir': str(checkpoint_dir),
                'val_loss': None,
            }

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Experiment failed: {e}")
        return {
            'condition': condition,
            'seed': seed,
            'use_pretrained': use_pretrained,
            'success': False,
            'elapsed_time': elapsed,
            'error': str(e),
        }

def main():
    """Run all ablation experiments."""

    print(f"""
{'='*70}
ABLATION STUDY: Pre-training Benefit Validation
{'='*70}

Configuration:
  Seeds: {SEEDS}
  Epochs per run: {NUM_EPOCHS}
  Device: {DEVICE}
  Total experiments: {len(SEEDS) * 2} (3 baseline + 3 pre-trained)

Data:
  Train cache: {TRAIN_CACHE}
  Val cache: {VAL_CACHE}
  Pre-trained encoder: {PRETRAINED_ENCODER}

Results directory: {RESULTS_DIR}

Estimated time: ~3 hours (30 min per experiment)

{'='*70}
""")

    input("Press Enter to start experiments (or Ctrl+C to cancel)...")

    all_results = []

    # Run baseline experiments (random initialization)
    print(f"\n{'#'*70}")
    print(f"# BASELINE EXPERIMENTS (Random Initialization)")
    print(f"{'#'*70}\n")

    for seed in SEEDS:
        result = run_experiment(f"baseline", seed, use_pretrained=False)
        all_results.append(result)

        # Save intermediate results
        results_file = RESULTS_DIR / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Run pre-trained experiments
    print(f"\n{'#'*70}")
    print(f"# PRE-TRAINED EXPERIMENTS (Stage 1 Initialization)")
    print(f"{'#'*70}\n")

    for seed in SEEDS:
        result = run_experiment(f"pretrained", seed, use_pretrained=True)
        all_results.append(result)

        # Save intermediate results
        results_file = RESULTS_DIR / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Save final results
    results_file = RESULTS_DIR / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"\nTo analyze results, run:")
    print(f"  python3.11 scripts/analyze_ablation.py")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
