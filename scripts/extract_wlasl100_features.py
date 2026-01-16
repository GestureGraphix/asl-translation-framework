#!/usr/bin/env python3.11
"""
Extract MediaPipe features for WLASL100 dataset.

Processes 100 most common signs (~1,348 videos) with train/val/test splits.
"""

import json
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phonology.mediapipe_extractor_v2 import MediaPipeExtractor
from phonology.features import FeatureExtractor

# Paths
METADATA_PATH = PROJECT_ROOT / "data/raw/wlasl/metadata.json"
VIDEO_DIR = Path("/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100")
OUTPUT_DIR = PROJECT_ROOT / "data/cache/wlasl100"

def load_metadata():
    """Load and parse WLASL metadata."""
    print(f"Loading metadata from {METADATA_PATH}...")

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    # Count videos per gloss
    gloss_data = {}
    for entry in metadata:
        gloss = entry['gloss']
        gloss_data[gloss] = entry['instances']

    # Sort by video count (top 100)
    sorted_glosses = sorted(gloss_data.items(),
                           key=lambda x: len(x[1]),
                           reverse=True)[:100]

    print(f"✓ Loaded metadata for 100 glosses")
    print(f"  Total videos: {sum(len(v) for _, v in sorted_glosses)}")

    return dict(sorted_glosses)

def find_video_file(video_id: str, gloss: str, video_dir: Path) -> Path:
    """Find video file by ID in gloss subdirectory."""
    # Videos are organized as videos_100/{gloss}/{video_id}.mp4
    gloss_dir = video_dir / gloss
    if not gloss_dir.exists():
        return None

    video_file = gloss_dir / f"{video_id}.mp4"
    if video_file.exists():
        return video_file

    return None

def extract_features_for_dataset(gloss_data: dict, split: str):
    """Extract features for a specific split."""

    mp_extractor = MediaPipeExtractor()
    feature_extractor = FeatureExtractor()

    cache_data = {}
    gloss_to_id = {}
    next_gloss_id = 1  # 0 reserved for blank

    split_instances = []
    for gloss, instances in gloss_data.items():
        for instance in instances:
            if instance['split'] == split:
                split_instances.append((gloss, instance))

    print(f"\n{'='*70}")
    print(f"Extracting {split.upper()} features")
    print(f"{'='*70}")
    print(f"Videos to process: {len(split_instances)}")

    success_count = 0
    skip_count = 0
    error_count = 0

    pbar = tqdm(split_instances, desc=f"{split.capitalize()} extraction")

    for gloss, instance in pbar:
        video_id = instance['video_id']

        # Find video file
        video_path = find_video_file(video_id, gloss, VIDEO_DIR)

        if not video_path or not video_path.exists():
            skip_count += 1
            pbar.set_postfix({'success': success_count, 'skip': skip_count, 'error': error_count})
            continue

        try:
            # Extract landmarks
            landmarks_sequence = mp_extractor.extract_video(str(video_path), max_frames=None)

            if len(landmarks_sequence) == 0:
                skip_count += 1
                pbar.set_postfix({'success': success_count, 'skip': skip_count, 'error': error_count})
                continue

            # Extract features for each frame
            feature_sequence = []
            for landmarks in landmarks_sequence:
                feats = feature_extractor.extract_features(landmarks, include_temporal=True)
                feature_sequence.append(feats.concatenate())

            features = np.array(feature_sequence, dtype=np.float32)

            if features is None or len(features) == 0:
                skip_count += 1
                pbar.set_postfix({'success': success_count, 'skip': skip_count, 'error': error_count})
                continue

            # Assign gloss ID
            if gloss not in gloss_to_id:
                gloss_to_id[gloss] = next_gloss_id
                next_gloss_id += 1

            gloss_id = gloss_to_id[gloss]

            # Store
            cache_data[video_id] = {
                'features': features,
                'gloss': gloss,
                'gloss_id': gloss_id,
                'video_id': video_id,
                'split': split,
            }

            success_count += 1
            pbar.set_postfix({'success': success_count, 'skip': skip_count, 'error': error_count})

        except Exception as e:
            error_count += 1
            pbar.set_postfix({'success': success_count, 'skip': skip_count, 'error': error_count})
            continue

    print(f"\n{'='*70}")
    print(f"{split.upper()} Summary")
    print(f"{'='*70}")
    print(f"  Success: {success_count}")
    print(f"  Skipped (no video): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Success rate: {success_count / len(split_instances) * 100:.1f}%")
    print(f"{'='*70}\n")

    return cache_data, gloss_to_id

def main():
    """Main extraction function."""

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check video directory exists
    if not VIDEO_DIR.exists():
        print(f"❌ Error: Video directory not found: {VIDEO_DIR}")
        sys.exit(1)

    # Count videos
    video_count = len(list(VIDEO_DIR.glob("*.mp4")))
    print(f"✓ Found {video_count} video files in {VIDEO_DIR}")

    # Load metadata
    gloss_data = load_metadata()

    print(f"\n{'='*70}")
    print(f"WLASL100 Feature Extraction")
    print(f"{'='*70}")
    print(f"Glosses: 100")
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    # Extract features for each split
    all_gloss_to_id = {}

    for split in ['train', 'val', 'test']:
        cache_data, gloss_to_id = extract_features_for_dataset(gloss_data, split)

        # Merge gloss_to_id mappings
        all_gloss_to_id.update(gloss_to_id)

        # Save cache
        output_file = OUTPUT_DIR / f"features_{split}_wlasl100.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"✓ Saved {len(cache_data)} cached features to {output_file}")

        # Print gloss distribution
        gloss_counts = defaultdict(int)
        for data in cache_data.values():
            gloss_counts[data['gloss']] += 1

        print(f"\nTop 10 glosses in {split}:")
        sorted_counts = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)
        for gloss, count in sorted_counts[:10]:
            print(f"  {gloss}: {count} videos")
        print()

    # Save vocabulary mapping
    vocab_file = OUTPUT_DIR / "vocabulary.json"
    id_to_gloss = {v: k for k, v in all_gloss_to_id.items()}
    vocab_data = {
        'gloss_to_id': all_gloss_to_id,
        'id_to_gloss': id_to_gloss,
        'vocab_size': len(all_gloss_to_id) + 1,  # +1 for blank
        'num_glosses': len(all_gloss_to_id),
    }

    with open(vocab_file, 'w') as f:
        json.dump(vocab_data, f, indent=2)

    print(f"✓ Saved vocabulary mapping to {vocab_file}")
    print(f"  Vocabulary size: {vocab_data['vocab_size']} (including blank)")

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Files created:")
    print(f"  {OUTPUT_DIR}/features_train_wlasl100.pkl")
    print(f"  {OUTPUT_DIR}/features_val_wlasl100.pkl")
    print(f"  {OUTPUT_DIR}/features_test_wlasl100.pkl")
    print(f"  {OUTPUT_DIR}/vocabulary.json")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
