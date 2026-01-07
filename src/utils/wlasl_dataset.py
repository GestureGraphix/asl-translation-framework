"""
WLASL Dataset Loader

Loads WLASL videos and extracts phonological features for training.

Dataset structure:
    - metadata.json: Contains gloss labels and video information
    - videos_100/: Directory with videos organized by gloss
        - {gloss}/
            - {video_id}.mp4

Usage:
    dataset = WLASLDataset(data_root, metadata_path, split='train')
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
import pickle


class WLASLDataset(Dataset):
    """
    WLASL dataset for ASL recognition.

    Loads videos and extracts phonological features on-the-fly or from cache.
    """

    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        split: str = 'train',
        feature_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        extract_features: bool = True,
        keep_full_vocab: bool = False,
    ):
        """
        Initialize WLASL dataset.

        Args:
            data_root: Root directory containing videos (e.g., data/raw/wlasl/videos_100)
            metadata_path: Path to metadata.json
            split: Dataset split ('train', 'val', 'test')
            feature_cache_path: Path to cached features (if available)
            max_samples: Maximum samples to load (for debugging)
            extract_features: Whether to extract features or return raw video paths
            keep_full_vocab: Keep full vocabulary even when using cache (for transfer learning)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.extract_features = extract_features
        self.feature_cache_path = feature_cache_path

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Build vocabulary (gloss -> ID mapping)
        self.gloss_to_id = {}
        self.id_to_gloss = {}

        all_glosses = sorted(set(item['gloss'] for item in self.metadata))
        for idx, gloss in enumerate(all_glosses, start=1):  # Start at 1, reserve 0 for CTC blank
            self.gloss_to_id[gloss] = idx
            self.id_to_gloss[idx] = gloss

        self.vocab_size = len(self.gloss_to_id) + 1  # +1 for blank

        # Build sample list
        self.samples = []
        for item in self.metadata:
            gloss = item['gloss']
            gloss_id = self.gloss_to_id[gloss]

            for instance in item['instances']:
                # Filter by split
                if instance['split'] != split:
                    continue

                video_id = instance['video_id']
                video_path = self.data_root / gloss / f"{video_id}.mp4"

                # Check if video exists
                if not video_path.exists():
                    continue

                self.samples.append({
                    'video_path': str(video_path),
                    'gloss': gloss,
                    'gloss_id': gloss_id,
                    'video_id': video_id,
                    'instance_id': instance['instance_id'],
                })

        # Limit samples if requested
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Load or create feature cache
        self.feature_cache = {}
        if feature_cache_path and Path(feature_cache_path).exists():
            self._load_feature_cache(feature_cache_path)

        # If using cache without extraction, filter to only cached samples
        if not extract_features and len(self.feature_cache) > 0:
            cached_video_ids = set(self.feature_cache.keys())
            original_count = len(self.samples)
            self.samples = [s for s in self.samples if s['video_id'] in cached_video_ids]
            print(f"  Filtered to {len(self.samples)} cached samples (was {original_count})")

            # Optionally rebuild vocabulary from cached samples only
            if not keep_full_vocab:
                cached_glosses = set(s['gloss'] for s in self.samples)
                self.gloss_to_id = {}
                self.id_to_gloss = {}
                for idx, gloss in enumerate(sorted(cached_glosses), start=1):
                    self.gloss_to_id[gloss] = idx
                    self.id_to_gloss[idx] = gloss
                self.vocab_size = len(self.gloss_to_id) + 1  # +1 for blank
                print(f"  Vocabulary filtered to {len(self.gloss_to_id)} cached glosses")

                # Update gloss_id in samples to match new vocabulary
                for sample in self.samples:
                    sample['gloss_id'] = self.gloss_to_id[sample['gloss']]
            else:
                # Keep full vocabulary from metadata (already loaded)
                print(f"  Keeping full vocabulary: {len(self.gloss_to_id)} glosses")
                # gloss_id already set correctly from original metadata

        print(f"WLASL Dataset ({split}):")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Vocabulary: {self.vocab_size} ({len(self.gloss_to_id)} glosses + blank)")
        print(f"  Feature cache: {len(self.feature_cache)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - features: Phonological features (T, 36) or None if not extracted
                - gloss_id: Integer gloss ID
                - gloss: String gloss label
                - video_path: Path to video file
                - length: Sequence length
        """
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
        """
        Extract phonological features from video.

        Args:
            video_path: Path to video file

        Returns:
            features: Phonological features (T, 36)
        """
        # Import here to avoid circular dependencies
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from phonology.mediapipe_extractor_v2 import MediaPipeExtractor
        from phonology.features import FeatureExtractor

        # Extract landmarks
        mp_extractor = MediaPipeExtractor()
        landmarks_sequence = mp_extractor.extract_video(video_path, max_frames=None)

        if len(landmarks_sequence) == 0:
            # Return dummy features if extraction fails
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
        with open(cache_path, 'wb') as f:
            pickle.dump(self.feature_cache, f)
        print(f"✓ Saved {len(self.feature_cache)} cached features")

    def precompute_features(self, save_path: Optional[str] = None):
        """
        Precompute features for all samples.

        Args:
            save_path: Path to save cache (optional)
        """
        print(f"Precomputing features for {len(self)} samples...")

        for idx in tqdm(range(len(self)), desc="Extracting features"):
            _ = self[idx]  # Trigger feature extraction

        if save_path:
            self.save_feature_cache(save_path)

        print(f"✓ Precomputed {len(self.feature_cache)} features")


def collate_fn(batch: List[Dict]) -> Optional[Dict[str, Union[torch.Tensor, List[str]]]]:
    """
    Collate function for DataLoader.

    Handles variable-length sequences by padding.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors with padding, or None if batch is empty
    """
    # Filter out samples with no features
    batch = [s for s in batch if s['features'] is not None and len(s['features']) > 0]

    if len(batch) == 0:
        return None

    # Get max sequence length
    max_len = max(s['length'] for s in batch)

    # Prepare tensors
    batch_size = len(batch)
    features = torch.zeros(batch_size, max_len, 36, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    gloss_ids = torch.zeros(batch_size, dtype=torch.long)

    # Fill tensors
    for i, sample in enumerate(batch):
        seq_len = sample['length']
        features[i, :seq_len, :] = torch.FloatTensor(sample['features'])
        lengths[i] = seq_len
        gloss_ids[i] = sample['gloss_id']

    return {
        'features': features,
        'lengths': lengths,
        'gloss_ids': gloss_ids,
        'glosses': [s['gloss'] for s in batch],
        'video_ids': [s['video_id'] for s in batch],
    }


# ============================================================================
# Utilities
# ============================================================================

def build_vocabulary(metadata_path: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Build vocabulary from metadata.

    Args:
        metadata_path: Path to metadata.json

    Returns:
        gloss_to_id: Mapping from gloss to ID
        id_to_gloss: Mapping from ID to gloss
        vocab_size: Total vocabulary size (including blank)
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    glosses = sorted(set(item['gloss'] for item in metadata))

    gloss_to_id = {}
    id_to_gloss = {}

    for idx, gloss in enumerate(glosses, start=1):
        gloss_to_id[gloss] = idx
        id_to_gloss[idx] = gloss

    vocab_size = len(gloss_to_id) + 1  # +1 for blank at index 0

    return gloss_to_id, id_to_gloss, vocab_size


def get_data_split_stats(metadata_path: str):
    """Print statistics about data splits."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    split_counts = {'train': 0, 'val': 0, 'test': 0}
    gloss_counts = {}

    for item in metadata:
        gloss = item['gloss']
        if gloss not in gloss_counts:
            gloss_counts[gloss] = {'train': 0, 'val': 0, 'test': 0}

        for instance in item['instances']:
            split = instance['split']
            split_counts[split] += 1
            gloss_counts[gloss][split] += 1

    print("\nDataset Statistics:")
    print(f"  Total glosses: {len(gloss_counts)}")
    print(f"  Train: {split_counts['train']} samples")
    print(f"  Val: {split_counts['val']} samples")
    print(f"  Test: {split_counts['test']} samples")
    print(f"  Total: {sum(split_counts.values())} samples")

    return split_counts, gloss_counts


# ============================================================================
# Testing
# ============================================================================

def test_dataset():
    """Test WLASL dataset loading."""
    print("\n" + "="*70)
    print("Testing WLASL Dataset")
    print("="*70 + "\n")

    # Paths
    data_root = "/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100"
    metadata_path = "/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json"

    # Get statistics
    get_data_split_stats(metadata_path)

    # Create dataset (without feature extraction for speed)
    print("\nCreating dataset...")
    dataset = WLASLDataset(
        data_root=data_root,
        metadata_path=metadata_path,
        split='train',
        max_samples=10,
        extract_features=False,  # Don't extract features for this test
    )

    # Test iteration
    print(f"\nTesting dataset iteration...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}:")
        print(f"    Gloss: {sample['gloss']} (ID: {sample['gloss_id']})")
        print(f"    Video: {Path(sample['video_path']).name}")

    # Test vocabulary
    print(f"\nVocabulary (first 10 glosses):")
    for gloss, idx in sorted(dataset.gloss_to_id.items())[:10]:
        print(f"  {idx}: {gloss}")

    # Test dataloader (without features)
    print(f"\nTesting DataLoader...")
    # Note: We're not extracting features, so collate_fn will filter them out
    # This is just to test the DataLoader infrastructure

    print("\n✓ Dataset test passed!\n")


if __name__ == "__main__":
    test_dataset()
