#!/usr/bin/env python3.11
"""
Cache Features for Phase 1 Training

Precomputes and caches phonological features for training efficiency.
Run this BEFORE train_phase1.py for fast training.

Usage:
    python3.11 scripts/cache_phase1_features.py --max-glosses 20

This will:
1. Load 20 most common sign classes
2. Extract features from ~1000 videos
3. Cache to disk for fast loading during training
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.wlasl_dataset_filtered import FilteredWLASLDataset
import argparse


def main():
    parser = argparse.ArgumentParser(description='Cache features for Phase 1')

    parser.add_argument('--data-root', type=str,
                       default='/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100',
                       help='Root directory for videos')
    parser.add_argument('--metadata', type=str,
                       default='/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json',
                       help='Path to metadata.json')
    parser.add_argument('--max-glosses', type=int, default=20,
                       help='Maximum number of sign classes')
    parser.add_argument('--samples-per-gloss', type=int, default=50,
                       help='Maximum samples per class')

    args = parser.parse_args()

    cache_dir = Path("data/cache/phase1")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / f"train_{args.max_glosses}glosses.pkl"
    val_cache = cache_dir / f"val_{args.max_glosses}glosses.pkl"

    print(f"\n{'='*70}")
    print(f"Phase 1 Feature Caching")
    print(f"{'='*70}")
    print(f"Max glosses: {args.max_glosses}")
    print(f"Samples per gloss: {args.samples_per_gloss}")
    print(f"Train cache: {train_cache}")
    print(f"Val cache: {val_cache}")
    print(f"{'='*70}\n")

    # Create training dataset
    print("Creating training dataset...")
    train_dataset = FilteredWLASLDataset(
        data_root=args.data_root,
        metadata_path=args.metadata,
        split='train',
        max_glosses=args.max_glosses,
        samples_per_gloss=args.samples_per_gloss,
        extract_features=True,
    )

    # Precompute train features
    print("\n" + "="*70)
    print("Extracting Training Features")
    print("="*70 + "\n")
    train_dataset.precompute_features(save_path=str(train_cache))

    # Create validation dataset (use same glosses as train)
    print("\n\nCreating validation dataset...")
    val_dataset = FilteredWLASLDataset(
        data_root=args.data_root,
        metadata_path=args.metadata,
        split='val',
        selected_glosses=list(train_dataset.included_glosses),  # Same glosses
        samples_per_gloss=args.samples_per_gloss // 2,
        extract_features=True,
    )

    # Precompute val features
    print("\n" + "="*70)
    print("Extracting Validation Features")
    print("="*70 + "\n")
    val_dataset.precompute_features(save_path=str(val_cache))

    print("\n" + "="*70)
    print("✓ Feature Caching Complete!")
    print("="*70)
    print(f"\nCached files:")
    print(f"  Train: {train_cache} ({train_cache.stat().st_size / 1e6:.1f} MB)")
    print(f"  Val:   {val_cache} ({val_cache.stat().st_size / 1e6:.1f} MB)")
    print(f"\nNext step:")
    print(f"  python3.11 scripts/train_phase1.py --use-cache --max-glosses {args.max_glosses}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
