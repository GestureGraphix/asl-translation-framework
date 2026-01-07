"""
WLASL Dataset with Filtered Vocabulary

Extension of wlasl_dataset.py that supports:
- Filtering to specific glosses
- Balanced sampling across classes
- Better vocabulary management for training efficiency
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
from collections import defaultdict
import pickle

from .wlasl_dataset import collate_fn  # Reuse collate function


class FilteredWLASLDataset(Dataset):
    """
    WLASL dataset with filtered vocabulary.

    Only includes specified glosses to reduce vocabulary size
    and prevent blank collapse.
    """

    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        split: str = 'train',
        max_glosses: Optional[int] = None,
        selected_glosses: Optional[List[str]] = None,
        samples_per_gloss: Optional[int] = None,
        feature_cache_path: Optional[str] = None,
        extract_features: bool = True,
    ):
        """
        Initialize filtered WLASL dataset.

        Args:
            data_root: Root directory containing videos
            metadata_path: Path to metadata.json
            split: Dataset split ('train', 'val', 'test')
            max_glosses: Maximum number of glosses to include (most common)
            selected_glosses: Specific glosses to include (overrides max_glosses)
            samples_per_gloss: Maximum samples per gloss (for balancing)
            feature_cache_path: Path to cached features
            extract_features: Whether to extract features on-the-fly
        """
        self.data_root = Path(data_root)
        self.split = split
        self.extract_features = extract_features
        self.feature_cache_path = feature_cache_path

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Determine which glosses to include
        if selected_glosses is not None:
            self.included_glosses = set(selected_glosses)
        elif max_glosses is not None:
            # Select most common glosses
            gloss_counts = self._count_glosses_in_split(split)
            top_glosses = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)
            self.included_glosses = set([g for g, _ in top_glosses[:max_glosses]])
        else:
            # Include all glosses
            self.included_glosses = set(item['gloss'] for item in self.metadata)

        print(f"Including {len(self.included_glosses)} glosses: {sorted(list(self.included_glosses))[:10]}...")

        # Build vocabulary from included glosses only
        self.gloss_to_id = {}
        self.id_to_gloss = {}

        for idx, gloss in enumerate(sorted(self.included_glosses), start=1):
            self.gloss_to_id[gloss] = idx
            self.id_to_gloss[idx] = gloss

        self.vocab_size = len(self.gloss_to_id) + 1  # +1 for blank at 0

        # Build sample list with balancing
        self.samples = self._build_sample_list(samples_per_gloss)

        # Load or create feature cache
        self.feature_cache = {}
        if feature_cache_path and Path(feature_cache_path).exists():
            self._load_feature_cache(feature_cache_path)

        print(f"\nFiltered WLASL Dataset ({split}):")
        print(f"  Glosses: {len(self.included_glosses)}")
        print(f"  Vocabulary: {self.vocab_size} (including blank)")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Feature cache: {len(self.feature_cache)} samples")

        # Print class distribution
        self._print_class_distribution()

    def _count_glosses_in_split(self, split: str) -> Dict[str, int]:
        """Count samples per gloss in a split."""
        counts = defaultdict(int)
        for item in self.metadata:
            gloss = item['gloss']
            for instance in item['instances']:
                if instance['split'] == split:
                    video_path = self.data_root / gloss / f"{instance['video_id']}.mp4"
                    if video_path.exists():
                        counts[gloss] += 1
        return counts

    def _build_sample_list(self, samples_per_gloss: Optional[int]) -> List[Dict]:
        """Build sample list with optional balancing."""
        samples_by_gloss = defaultdict(list)

        # Group samples by gloss
        for item in self.metadata:
            gloss = item['gloss']

            # Skip if not in included glosses
            if gloss not in self.included_glosses:
                continue

            gloss_id = self.gloss_to_id[gloss]

            for instance in item['instances']:
                # Filter by split
                if instance['split'] != self.split:
                    continue

                video_id = instance['video_id']
                video_path = self.data_root / gloss / f"{video_id}.mp4"

                # Check if video exists
                if not video_path.exists():
                    continue

                samples_by_gloss[gloss].append({
                    'video_path': str(video_path),
                    'gloss': gloss,
                    'gloss_id': gloss_id,
                    'video_id': video_id,
                    'instance_id': instance['instance_id'],
                })

        # Balance samples if requested
        all_samples = []
        for gloss, gloss_samples in samples_by_gloss.items():
            if samples_per_gloss is not None:
                gloss_samples = gloss_samples[:samples_per_gloss]
            all_samples.extend(gloss_samples)

        return all_samples

    def _print_class_distribution(self):
        """Print distribution of samples across classes."""
        from collections import Counter
        gloss_counts = Counter(s['gloss'] for s in self.samples)

        print(f"\nClass Distribution:")
        print(f"  Min samples/class: {min(gloss_counts.values())}")
        print(f"  Max samples/class: {max(gloss_counts.values())}")
        print(f"  Avg samples/class: {np.mean(list(gloss_counts.values())):.1f}")

        print(f"\nTop 10 classes:")
        for gloss, count in gloss_counts.most_common(10):
            print(f"    {gloss}: {count} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.samples[idx]
        video_id = sample['video_id']

        # Try to load from cache
        if video_id in self.feature_cache:
            features = self.feature_cache[video_id]
        elif self.extract_features:
            # Extract features on-the-fly
            features = self._extract_features(sample['video_path'])
            self.feature_cache[video_id] = features
        else:
            features = None

        return {
            'features': features,
            'gloss_id': sample['gloss_id'],
            'gloss': sample['gloss'],
            'video_path': sample['video_path'],
            'video_id': video_id,
            'length': len(features) if features is not None else 0,
        }

    def _extract_features(self, video_path: str) -> np.ndarray:
        """Extract phonological features from video."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from phonology.mediapipe_extractor_v2 import MediaPipeExtractor
        from phonology.features import FeatureExtractor

        # Extract landmarks
        mp_extractor = MediaPipeExtractor()
        landmarks_sequence = mp_extractor.extract_video(video_path, max_frames=None)

        if len(landmarks_sequence) == 0:
            return np.zeros((1, 36), dtype=np.float32)

        # Extract features
        feature_extractor = FeatureExtractor()
        feature_sequence = []

        for landmarks in landmarks_sequence:
            features = feature_extractor.extract_features(landmarks, include_temporal=True)
            feature_sequence.append(features.concatenate())

        return np.array(feature_sequence, dtype=np.float32)

    def _load_feature_cache(self, cache_path: str):
        """Load cached features from disk."""
        print(f"Loading feature cache from {cache_path}...")
        with open(cache_path, 'rb') as f:
            self.feature_cache = pickle.load(f)

    def save_feature_cache(self, cache_path: str):
        """Save feature cache to disk."""
        print(f"Saving feature cache to {cache_path}...")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.feature_cache, f)
        print(f"✓ Saved {len(self.feature_cache)} cached features")

    def precompute_features(self, save_path: Optional[str] = None):
        """Precompute features for all samples."""
        from tqdm import tqdm

        print(f"Precomputing features for {len(self)} samples...")

        for idx in tqdm(range(len(self)), desc="Extracting features"):
            _ = self[idx]  # Trigger feature extraction

        if save_path:
            self.save_feature_cache(save_path)

        print(f"✓ Precomputed {len(self.feature_cache)} features")
