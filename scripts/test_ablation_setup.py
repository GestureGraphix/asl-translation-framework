#!/usr/bin/env python3.11
"""
Quick test to verify ablation study setup works.

Runs 2 quick experiments (1 epoch each):
1. Baseline (random init)
2. Pre-trained (Stage 1 init)

Total time: ~2 minutes
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_CACHE = PROJECT_ROOT / "data/cache/phase1/train_20glosses.pkl"
VAL_CACHE = PROJECT_ROOT / "data/cache/phase1/val_20glosses.pkl"
PRETRAINED_ENCODER = PROJECT_ROOT / "checkpoints/stage1/checkpoint_best.pt"
TEST_DIR = PROJECT_ROOT / "checkpoints/ablation_test"

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

print(f"\n{'='*70}")
print(f"ABLATION SETUP TEST")
print(f"{'='*70}\n")
print("This will run 2 quick experiments (1 epoch each) to verify setup:")
print("  1. Baseline (random initialization)")
print("  2. Pre-trained (Stage 1 initialization)")
print()
print(f"Estimated time: ~2 minutes")
print(f"{'='*70}\n")

# Test 1: Baseline
print("Test 1: Baseline (random initialization)")
print("-" * 70)

baseline_dir = TEST_DIR / "baseline"
baseline_dir.mkdir(parents=True, exist_ok=True)

cmd_baseline = [
    "python3.11",
    str(PROJECT_ROOT / "src/training/stage2_ctc.py"),
    "--feature-cache", str(TRAIN_CACHE),
    "--val-feature-cache", str(VAL_CACHE),
    "--checkpoint-dir", str(baseline_dir),
    "--device", "cuda",
    "--num-epochs", "1",  # Just 1 epoch for testing
    "--hidden-dim", "128",  # Must match Stage 1
    "--num-layers", "2",     # Must match Stage 1
    "--batch-size", "8",
    "--seed", "42",
]

print(f"Running: {' '.join(cmd_baseline[-8:])}")
result = subprocess.run(cmd_baseline, capture_output=True, text=True)

if result.returncode != 0:
    print("\n❌ Baseline test FAILED")
    print("\nStderr:")
    print(result.stderr)
    sys.exit(1)
else:
    # Check for validation accuracy in output
    if "Val Acc:" in result.stdout:
        for line in result.stdout.split('\n'):
            if "Val Acc:" in line:
                print(f"✓ {line.strip()}")
                break
    else:
        print("✓ Training completed")

print()

# Test 2: Pre-trained
print("Test 2: Pre-trained (Stage 1 initialization)")
print("-" * 70)

pretrained_dir = TEST_DIR / "pretrained"
pretrained_dir.mkdir(parents=True, exist_ok=True)

cmd_pretrained = [
    "python3.11",
    str(PROJECT_ROOT / "src/training/stage2_ctc.py"),
    "--feature-cache", str(TRAIN_CACHE),
    "--val-feature-cache", str(VAL_CACHE),
    "--checkpoint-dir", str(pretrained_dir),
    "--device", "cuda",
    "--num-epochs", "1",  # Just 1 epoch for testing
    "--hidden-dim", "128",  # Must match Stage 1
    "--num-layers", "2",     # Must match Stage 1
    "--batch-size", "8",
    "--seed", "42",
    "--pretrained-encoder", str(PRETRAINED_ENCODER),
]

print(f"Running: {' '.join(cmd_pretrained[-10:])}")
result = subprocess.run(cmd_pretrained, capture_output=True, text=True)

if result.returncode != 0:
    print("\n❌ Pre-trained test FAILED")
    print("\nStderr:")
    print(result.stderr)
    sys.exit(1)
else:
    # Check for pre-trained loading message
    if "Successfully loaded pre-trained encoder" in result.stdout:
        print("✓ Pre-trained encoder loaded successfully")

    # Check for validation accuracy in output
    if "Val Acc:" in result.stdout:
        for line in result.stdout.split('\n'):
            if "Val Acc:" in line:
                print(f"✓ {line.strip()}")
                break
    else:
        print("✓ Training completed")

print()
print(f"{'='*70}")
print(f"✅ ABLATION SETUP TEST PASSED")
print(f"{'='*70}\n")
print("Everything works! You can now run the full ablation study:")
print(f"  python3.11 scripts/run_ablation_study.py")
print()
print(f"Note: Test results saved to {TEST_DIR}")
print(f"      (You can delete this directory)")
print(f"{'='*70}\n")
