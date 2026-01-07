#!/usr/bin/env python3.11
"""
Precompute and cache phonological features for WLASL dataset.

This script extracts features from all videos ONCE and saves them to disk.
Training can then load cached features instantly without re-processing videos.

This dramatically speeds up training and reduces CPU load.

Usage:
    python3.11 scripts/precompute_features.py --split train
    python3.11 scripts/precompute_features.py --split val
    python3.11 scripts/precompute_features.py --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.wlasl_dataset import WLASLDataset
import argparse


def main():
    parser = argparse.ArgumentParser(description='Precompute features for WLASL')
    parser.add_argument('--data-root', type=str,
                        default='/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100',
                        help='Root directory for videos')
    parser.add_argument('--metadata', type=str,
                        default='/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json',
                        help='Path to metadata.json')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--output-dir', type=str,
                        default='/home/alex/Documents/asl-translation-framework/data/processed',
                        help='Output directory for cached features')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to process (for testing)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"features_{args.split}.pkl"

    print(f"\n{'='*70}")
    print(f"Precomputing Features - {args.split.upper()} Split")
    print(f"{'='*70}\n")

    # Create dataset
    dataset = WLASLDataset(
        data_root=args.data_root,
        metadata_path=args.metadata,
        split=args.split,
        max_samples=args.max_samples,
        extract_features=True,
    )

    # Precompute and save
    dataset.precompute_features(save_path=str(output_path))

    print(f"\n{'='*70}")
    print(f"✓ Features saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'='*70}\n")

    print("Next steps:")
    print(f"  1. Run for other splits: --split val, --split test")
    print(f"  2. Use in training with: --feature-cache {output_path}")
    print()


if __name__ == "__main__":
    main()
