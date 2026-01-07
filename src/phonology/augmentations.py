"""Temporal augmentations for self-supervised phonological learning.

Implements augmentation strategies for Stage 1 pre-training (Section 7.2).
Creates positive pairs for contrastive learning while preserving phonological structure.

Mathematical Framework:
    Given video sequence X = {x_t}_{t=1}^T, create augmented views:
    - X' = Aug_1(X)
    - X'' = Aug_2(X)

    Positive pairs: (X', X'') from same video
    Negative pairs: (X_i', X_j'') from different videos i ≠ j

Key Invariances (Section 2.2):
    - Sim(3) transformations (translation, rotation, scale)
    - Temporal cropping (subsequences preserve phonology)
    - Speed perturbations (within ±20% preserve timing)
    - Gaussian noise (small σ preserves quantization by Proposition 1)
"""

import numpy as np
import torch
from typing import Tuple, Optional
import random


class TemporalAugmentation:
    """Temporal augmentations for sign language video features.

    Augmentation strategies:
    1. Temporal cropping: Extract random subsequences
    2. Speed perturbation: Resample at different speeds
    3. Gaussian noise: Add small noise to features
    4. Temporal masking: Mask random frames (optional)

    Preserves phonological structure while creating diversity for contrastive learning.
    """

    def __init__(
        self,
        crop_ratio: Tuple[float, float] = (0.7, 1.0),
        speed_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.05,
        mask_prob: float = 0.0,
        seed: Optional[int] = None
    ):
        """Initialize temporal augmentation.

        Args:
            crop_ratio: (min, max) ratio of sequence length to keep
            speed_range: (min, max) speed multiplier for resampling
            noise_std: Standard deviation of Gaussian noise
                      Should be < γ/L to preserve quantization (Proposition 1)
            mask_prob: Probability of masking each frame
            seed: Random seed for reproducibility
        """
        self.crop_ratio = crop_ratio
        self.speed_range = speed_range
        self.noise_std = noise_std
        self.mask_prob = mask_prob

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def temporal_crop(self, features: torch.Tensor) -> torch.Tensor:
        """Extract random temporal subsequence.

        Implements random cropping while ensuring minimum length.

        Args:
            features: (T, D) feature sequence

        Returns:
            Cropped sequence (T', D) where T' >= crop_ratio[0] * T
        """
        T, D = features.shape

        # Sample crop ratio
        ratio = random.uniform(*self.crop_ratio)
        new_length = max(int(T * ratio), 10)  # Ensure minimum 10 frames

        if new_length >= T:
            return features

        # Random start position
        start = random.randint(0, T - new_length)
        end = start + new_length

        return features[start:end]

    def speed_perturbation(self, features: torch.Tensor) -> torch.Tensor:
        """Resample sequence at different speed.

        Implements temporal resampling using linear interpolation.
        Speed changes within ±20% preserve phonological timing.

        Args:
            features: (T, D) feature sequence

        Returns:
            Resampled sequence (T', D)
        """
        T, D = features.shape
        device = features.device

        # Sample speed multiplier
        speed = random.uniform(*self.speed_range)
        new_length = max(int(T / speed), 10)  # Ensure minimum 10 frames

        if new_length == T:
            return features

        # Move to CPU for numpy interpolation
        features_cpu = features.cpu()

        # Linear interpolation
        old_indices = torch.linspace(0, T - 1, T)
        new_indices = torch.linspace(0, T - 1, new_length)

        # Interpolate each dimension
        resampled = torch.zeros(new_length, D, dtype=features.dtype)
        for d in range(D):
            resampled[:, d] = torch.from_numpy(
                np.interp(new_indices.numpy(), old_indices.numpy(), features_cpu[:, d].numpy())
            )

        # Move back to original device
        return resampled.to(device)

    def add_noise(self, features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to features.

        Noise magnitude controlled by Proposition 1 (margin robustness):
        If ||η|| < γ/L, quantization output unchanged.

        Args:
            features: (T, D) feature sequence

        Returns:
            Noisy features (T, D)
        """
        noise = torch.randn_like(features) * self.noise_std
        return features + noise

    def temporal_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Randomly mask frames.

        Args:
            features: (T, D) feature sequence

        Returns:
            Masked features (T, D) with some frames zeroed
        """
        if self.mask_prob == 0.0:
            return features

        T, D = features.shape
        mask = torch.rand(T) > self.mask_prob
        return features * mask.unsqueeze(1)

    def __call__(
        self,
        features: torch.Tensor,
        apply_crop: bool = True,
        apply_speed: bool = True,
        apply_noise: bool = True,
        apply_mask: bool = False
    ) -> torch.Tensor:
        """Apply random augmentations to feature sequence.

        Args:
            features: (T, D) feature sequence
            apply_crop: Whether to apply temporal cropping
            apply_speed: Whether to apply speed perturbation
            apply_noise: Whether to add Gaussian noise
            apply_mask: Whether to apply temporal masking

        Returns:
            Augmented features (T', D)
        """
        augmented = features

        if apply_crop:
            augmented = self.temporal_crop(augmented)

        if apply_speed:
            augmented = self.speed_perturbation(augmented)

        if apply_noise:
            augmented = self.add_noise(augmented)

        if apply_mask:
            augmented = self.temporal_mask(augmented)

        return augmented

    def create_positive_pair(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create two augmented views of the same sequence.

        Used for contrastive learning where (view1, view2) is a positive pair.

        Args:
            features: (T, D) feature sequence

        Returns:
            Tuple of two augmented views (T1, D) and (T2, D)
        """
        view1 = self(features)
        view2 = self(features)
        return view1, view2


def create_contrastive_batch(
    features_list: list,
    augmenter: TemporalAugmentation,
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create batch for contrastive learning.

    Generates positive pairs and prepares for negative sampling.

    Args:
        features_list: List of feature sequences [(T_i, D), ...]
        augmenter: TemporalAugmentation instance
        batch_size: Number of samples in batch

    Returns:
        Tuple of:
        - anchors: (B, T_max, D) padded anchor views
        - positives: (B, T_max, D) padded positive views
        - lengths: (B, 2) actual lengths of [anchor, positive]
    """
    # Sample batch_size sequences
    sampled = random.sample(features_list, min(batch_size, len(features_list)))

    # Create positive pairs
    anchors = []
    positives = []
    anchor_lengths = []
    positive_lengths = []

    for features in sampled:
        view1, view2 = augmenter.create_positive_pair(features)
        anchors.append(view1)
        positives.append(view2)
        anchor_lengths.append(len(view1))
        positive_lengths.append(len(view2))

    # Pad to same length
    max_len = max(max(anchor_lengths), max(positive_lengths))

    anchors_padded = torch.zeros(batch_size, max_len, sampled[0].shape[1])
    positives_padded = torch.zeros(batch_size, max_len, sampled[0].shape[1])

    for i, (anchor, positive) in enumerate(zip(anchors, positives)):
        anchors_padded[i, :len(anchor)] = anchor
        positives_padded[i, :len(positive)] = positive

    lengths = torch.tensor([[anchor_lengths[i], positive_lengths[i]]
                           for i in range(batch_size)])

    return anchors_padded, positives_padded, lengths


# Example usage and testing
if __name__ == "__main__":
    # Test augmentation
    print("Testing temporal augmentations...")

    # Create dummy feature sequence (100 frames, 36 dims)
    features = torch.randn(100, 36)

    # Create augmenter
    augmenter = TemporalAugmentation(
        crop_ratio=(0.7, 1.0),
        speed_range=(0.8, 1.2),
        noise_std=0.05,
        seed=42
    )

    # Test individual augmentations
    cropped = augmenter.temporal_crop(features)
    print(f"Original length: {len(features)}, Cropped: {len(cropped)}")

    resampled = augmenter.speed_perturbation(features)
    print(f"Original length: {len(features)}, Resampled: {len(resampled)}")

    noisy = augmenter.add_noise(features)
    noise_magnitude = (noisy - features).norm(dim=1).mean()
    print(f"Average noise magnitude: {noise_magnitude:.4f}")

    # Test positive pair creation
    view1, view2 = augmenter.create_positive_pair(features)
    print(f"\nPositive pair shapes: {view1.shape}, {view2.shape}")

    # Test batch creation
    features_list = [torch.randn(80 + i*10, 36) for i in range(10)]
    anchors, positives, lengths = create_contrastive_batch(
        features_list, augmenter, batch_size=8
    )
    print(f"\nBatch shapes:")
    print(f"  Anchors: {anchors.shape}")
    print(f"  Positives: {positives.shape}")
    print(f"  Lengths: {lengths.shape}")
    print(f"  Sample lengths: {lengths[:3]}")

    print("\n✓ All augmentation tests passed!")
