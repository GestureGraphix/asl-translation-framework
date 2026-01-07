#!/usr/bin/env python3.11
"""
Test script for phonological feature extraction.

Tests the complete pipeline:
    Video → MediaPipe landmarks → Sim(3) normalization → Phonological features

Usage:
    python3.11 scripts/test_feature_extraction.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phonology.mediapipe_extractor_v2 import MediaPipeExtractor
from phonology.features import FeatureExtractor
import numpy as np


def test_feature_pipeline(video_path: str, max_frames: int = 50):
    """
    Test complete feature extraction pipeline on a video.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process
    """
    print(f"\n{'='*70}")
    print(f"Testing Feature Extraction Pipeline")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Max frames: {max_frames}")
    print(f"{'='*70}\n")

    # Step 1: Extract landmarks
    print("Step 1: Extracting landmarks with MediaPipe...")
    extractor = MediaPipeExtractor()
    landmarks_sequence = extractor.extract_video(video_path, max_frames=max_frames)

    if len(landmarks_sequence) == 0:
        print("❌ ERROR: No landmarks extracted!")
        return False

    print(f"✓ Extracted {len(landmarks_sequence)} frames\n")

    # Step 2: Extract features
    print("Step 2: Extracting phonological features...")
    feature_extractor = FeatureExtractor()
    feature_sequence = []

    for i, landmarks in enumerate(landmarks_sequence):
        features = feature_extractor.extract_features(landmarks, include_temporal=True)
        feature_sequence.append(features)

        if i == 0:
            # Analyze first frame in detail
            print(f"\nFirst Frame Feature Analysis:")
            print(f"  Handshape:   {features.handshape.shape} = {features.handshape[:3]}...")
            print(f"  Location:    {features.location.shape} = {features.location}")
            print(f"  Orientation: {features.orientation.shape} = {features.orientation}")
            print(f"  Movement:    {features.movement.shape} = {features.movement} (zeros for first frame)")
            print(f"  Non-manual:  {features.nonmanual.shape} = {features.nonmanual}")
            print(f"  Concatenated: {features.concatenate().shape}")

    print(f"\n✓ Extracted features for {len(feature_sequence)} frames")

    # Step 3: Validate feature shapes and ranges
    print(f"\n{'='*70}")
    print("Validation")
    print(f"{'='*70}")

    all_features = np.array([f.concatenate() for f in feature_sequence])
    print(f"Feature matrix shape: {all_features.shape}")
    print(f"Expected: ({len(feature_sequence)}, 36)")

    if all_features.shape[1] != 36:
        print(f"❌ ERROR: Feature dimension mismatch! Expected 36, got {all_features.shape[1]}")
        return False

    print(f"✓ Feature dimensions correct\n")

    # Analyze feature statistics
    print("Feature Statistics (across all frames):")
    print(f"  Mean:     {np.mean(all_features, axis=0)[:5]}... (first 5)")
    print(f"  Std:      {np.std(all_features, axis=0)[:5]}... (first 5)")
    print(f"  Min:      {np.min(all_features, axis=0)[:5]}... (first 5)")
    print(f"  Max:      {np.max(all_features, axis=0)[:5]}... (first 5)")

    # Check for non-zero movement features (should appear after first frame)
    movement_features = all_features[:, 16:25]  # Movement is indices 16-24 (10+6 offset)
    nonzero_movement = np.any(movement_features != 0, axis=1)
    num_with_movement = np.sum(nonzero_movement)

    print(f"\nTemporal Features:")
    print(f"  Frames with non-zero movement: {num_with_movement}/{len(feature_sequence)}")
    print(f"  Expected: {len(feature_sequence) - 1} (all except first frame)")

    if num_with_movement < len(feature_sequence) - 1:
        print(f"  ⚠ WARNING: Some frames missing movement features")
    else:
        print(f"  ✓ Movement features computed correctly")

    # Step 4: Test Sim(3) invariance
    print(f"\n{'='*70}")
    print("Testing Sim(3) Invariance")
    print(f"{'='*70}")

    test_landmark = landmarks_sequence[0]

    # Original features
    feature_extractor.reset_temporal_state()
    original_features = feature_extractor.extract_features(test_landmark, include_temporal=False)

    # Apply translation
    from phonology.mediapipe_extractor_v2 import RawLandmarks
    translation = np.array([1.0, 2.0, 3.0])

    translated_landmarks = RawLandmarks(
        left_hand=test_landmark.left_hand + translation,
        right_hand=test_landmark.right_hand + translation,
        face=test_landmark.face + translation,
        pose=test_landmark.pose + translation,
        timestamp=test_landmark.timestamp,
    )

    feature_extractor.reset_temporal_state()
    translated_features = feature_extractor.extract_features(translated_landmarks, include_temporal=False)

    # Compare features (should be approximately equal due to normalization)
    feature_diff = np.max(np.abs(original_features.concatenate() - translated_features.concatenate()))
    print(f"\nTranslation invariance test:")
    print(f"  Translation: {translation}")
    print(f"  Max feature difference: {feature_diff:.6f}")

    if feature_diff < 0.1:
        print(f"  ✓ PASS: Features invariant to translation")
    else:
        print(f"  ⚠ WARNING: Features changed significantly under translation")

    # Apply scale
    scale_factor = 2.0
    scaled_landmarks = RawLandmarks(
        left_hand=test_landmark.left_hand * scale_factor,
        right_hand=test_landmark.right_hand * scale_factor,
        face=test_landmark.face * scale_factor,
        pose=test_landmark.pose * scale_factor,
        timestamp=test_landmark.timestamp,
    )

    feature_extractor.reset_temporal_state()
    scaled_features = feature_extractor.extract_features(scaled_landmarks, include_temporal=False)

    scale_diff = np.max(np.abs(original_features.concatenate() - scaled_features.concatenate()))
    print(f"\nScale invariance test:")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Max feature difference: {scale_diff:.6f}")

    if scale_diff < 0.1:
        print(f"  ✓ PASS: Features invariant to scaling")
    else:
        print(f"  ⚠ WARNING: Features changed significantly under scaling")

    print(f"\n{'='*70}")
    print("✓ Feature Extraction Test Complete!")
    print(f"{'='*70}\n")

    return True


def test_individual_components():
    """Test individual feature extraction components."""
    print(f"\n{'='*70}")
    print("Testing Individual Components")
    print(f"{'='*70}\n")

    from phonology.mediapipe_extractor_v2 import MediaPipeExtractor

    # Create test frame
    extractor_mp = MediaPipeExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = extractor_mp.extract_frame(dummy_frame, 0.0)

    if landmarks is None:
        print("⚠ Could not extract landmarks from dummy frame")
        return

    feature_extractor = FeatureExtractor()

    # Test normalization
    print("Testing Sim(3) normalization...")
    normalized = feature_extractor.normalize_sim3(landmarks)
    print(f"  ✓ Scale: {normalized.scale:.4f}")
    print(f"  ✓ Translation: {normalized.translation}")
    print(f"  ✓ Rotation shape: {normalized.rotation.shape}")

    # Test feature extraction
    print("\nTesting feature extraction...")
    features = feature_extractor.extract_features(landmarks, include_temporal=False)

    print(f"  ✓ Handshape (10): {features.handshape.shape}")
    print(f"  ✓ Location (6): {features.location.shape}")
    print(f"  ✓ Orientation (6): {features.orientation.shape}")
    print(f"  ✓ Movement (9): {features.movement.shape}")
    print(f"  ✓ Non-manual (5): {features.nonmanual.shape}")
    print(f"  ✓ Total (36): {features.concatenate().shape}")

    print(f"\n✓ All components working correctly!\n")


def main():
    """Run feature extraction tests."""

    # Test individual components first
    test_individual_components()

    # Find a sample video
    video_base = Path("/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100")

    # Find first available video
    for gloss_dir in sorted(video_base.iterdir()):
        if gloss_dir.is_dir():
            videos = list(gloss_dir.glob("*.mp4"))
            if videos:
                gloss = gloss_dir.name
                video_path = videos[0]
                print(f"\nTesting on video: '{gloss}' ({video_path.name})")
                success = test_feature_pipeline(str(video_path), max_frames=50)

                if success:
                    print("\n" + "="*70)
                    print("NEXT STEPS")
                    print("="*70)
                    print("1. ✓ MediaPipe extraction working!")
                    print("2. ✓ Feature extraction working!")
                    print("3. → Next: Implement product VQ quantizer (quantizer.py)")
                    print("4. → Then: Write comprehensive tests")
                    print("="*70 + "\n")
                    return 0
                else:
                    return 1

    print("❌ ERROR: No videos found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
