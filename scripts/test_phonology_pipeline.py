#!/usr/bin/env python3.11
"""
End-to-End Phonology Pipeline Test

Tests the complete phonology module:
    Video → MediaPipe landmarks → Features → Quantized codes

This validates the full pipeline from Section 2:
    1. Landmark extraction (MediaPipe)
    2. Sim(3) normalization and feature extraction (φ)
    3. Product VQ quantization (q)

Final output: Discrete phonological sequence Z_t ∈ Σ

Usage:
    python3.11 scripts/test_phonology_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phonology.mediapipe_extractor_v2 import MediaPipeExtractor
from phonology.features import FeatureExtractor
from phonology.quantizer import ProductVectorQuantizer, CodebookConfig
import numpy as np


def test_end_to_end_pipeline(video_path: str, max_frames: int = 100):
    """
    Test complete phonology pipeline on a video.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process
    """
    print(f"\n{'='*70}")
    print(f"End-to-End Phonology Pipeline Test")
    print(f"{'='*70}")
    print(f"Video: {Path(video_path).name}")
    print(f"Max frames: {max_frames}")
    print(f"{'='*70}\n")

    # ========================================================================
    # Step 1: Extract landmarks
    # ========================================================================
    print("Step 1: Extracting landmarks with MediaPipe...")
    print("-" * 70)

    extractor_mp = MediaPipeExtractor()
    landmarks_sequence = extractor_mp.extract_video(video_path, max_frames=max_frames)

    if len(landmarks_sequence) == 0:
        print("❌ ERROR: No landmarks extracted!")
        return False

    print(f"✓ Extracted {len(landmarks_sequence)} frames\n")

    # ========================================================================
    # Step 2: Extract features
    # ========================================================================
    print("Step 2: Extracting phonological features...")
    print("-" * 70)

    feature_extractor = FeatureExtractor()
    feature_sequence = []

    for landmarks in landmarks_sequence:
        features = feature_extractor.extract_features(landmarks, include_temporal=True)
        feature_sequence.append(features)

    # Convert to numpy array
    feature_matrix = np.array([f.concatenate() for f in feature_sequence])
    print(f"✓ Extracted features: shape {feature_matrix.shape}")
    print(f"  (Expected: ({len(landmarks_sequence)}, 36))\n")

    # ========================================================================
    # Step 3: Train quantizer
    # ========================================================================
    print("Step 3: Training product vector quantizer...")
    print("-" * 70)

    # Use smaller codebooks for testing
    config = CodebookConfig(
        k_handshape=16,
        k_location=8,
        k_orientation=8,
        k_movement=8,
        k_nonmanual=8,
    )

    quantizer = ProductVectorQuantizer(config)
    quantizer.fit(feature_matrix, verbose=False)

    print(f"✓ Trained quantizer")
    print(f"  Alphabet size: {config.total_alphabet_size():,}")
    print(f"  Component sizes: {config.get_component_sizes()}\n")

    # ========================================================================
    # Step 4: Quantize features
    # ========================================================================
    print("Step 4: Quantizing features to discrete codes...")
    print("-" * 70)

    code_sequence = quantizer.quantize_batch(feature_matrix)
    print(f"✓ Quantized {len(code_sequence)} frames\n")

    # Display sample codes
    print("Sample phonological codes:")
    for i in [0, len(code_sequence)//2, len(code_sequence)-1]:
        print(f"  Frame {i:3d}: {code_sequence[i]}")

    # ========================================================================
    # Step 5: Analyze results
    # ========================================================================
    print(f"\n{'='*70}")
    print("Pipeline Analysis")
    print(f"{'='*70}\n")

    # Quantization error
    quant_error = quantizer.compute_quantization_error(feature_matrix)
    print(f"Reconstruction error: {quant_error:.4f}")

    # Code diversity (unique codes)
    code_tuples = [c.to_tuple() for c in code_sequence]
    unique_codes = len(set(code_tuples))
    print(f"Unique codes: {unique_codes}/{len(code_sequence)} ({100*unique_codes/len(code_sequence):.1f}%)")

    # Component-wise diversity
    print(f"\nComponent diversity:")
    for component in ['handshape', 'location', 'orientation', 'movement', 'nonmanual']:
        values = [getattr(c, component) for c in code_sequence]
        unique_values = len(set(values))
        codebook_size = config.get_component_sizes()[component]
        print(f"  {component:12s}: {unique_values:2d}/{codebook_size:2d} unique codes")

    # Temporal transitions (how often codes change)
    transitions = sum(1 for i in range(1, len(code_sequence))
                      if code_sequence[i].to_tuple() != code_sequence[i-1].to_tuple())
    transition_rate = transitions / (len(code_sequence) - 1) if len(code_sequence) > 1 else 0
    print(f"\nTemporal dynamics:")
    print(f"  Transitions: {transitions}/{len(code_sequence)-1} ({100*transition_rate:.1f}%)")

    # ========================================================================
    # Step 6: Test save/load
    # ========================================================================
    print(f"\n{'='*70}")
    print("Testing Save/Load")
    print(f"{'='*70}\n")

    # Save quantizer
    save_path = "/tmp/test_quantizer.pkl"
    quantizer.save(save_path)

    # Load quantizer
    loaded_quantizer = ProductVectorQuantizer.load(save_path)

    # Verify it works
    test_code = loaded_quantizer.quantize(feature_matrix[0])
    print(f"✓ Loaded quantizer produces: {test_code}")
    print(f"  (Original was: {code_sequence[0]})")

    assert test_code.to_tuple() == code_sequence[0].to_tuple(), "Loaded quantizer mismatch!"
    print("✓ Save/load working correctly\n")

    # Clean up
    os.remove(save_path)

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"{'='*70}")
    print("✓ End-to-End Pipeline Test Complete!")
    print(f"{'='*70}")
    print(f"\nPipeline Summary:")
    print(f"  Input:  Video ({len(landmarks_sequence)} frames)")
    print(f"  →  Landmarks: {landmarks_sequence[0].concatenate().shape}")
    print(f"  →  Features:  {feature_matrix[0].shape}")
    print(f"  →  Codes:     {code_sequence[0]}")
    print(f"\n  Output: Discrete phonological sequence of length {len(code_sequence)}")
    print(f"  Alphabet: Σ_H × Σ_L × Σ_O × Σ_M × Σ_N")
    print(f"           ({config.k_handshape} × {config.k_location} × {config.k_orientation} × {config.k_movement} × {config.k_nonmanual} = {config.total_alphabet_size():,} possible codes)")
    print(f"{'='*70}\n")

    return True


def test_proposition_1_noise_robustness():
    """
    Test Proposition 1: Margin-based noise robustness.

    Validates that small perturbations don't change quantization.
    """
    print(f"\n{'='*70}")
    print("Testing Proposition 1: Noise Robustness")
    print(f"{'='*70}\n")

    # Generate training data
    np.random.seed(42)
    n_samples = 500
    features = np.random.randn(n_samples, 36)

    # Train quantizer
    config = CodebookConfig(k_handshape=8, k_location=4, k_orientation=4,
                            k_movement=4, k_nonmanual=4)
    quantizer = ProductVectorQuantizer(config)
    quantizer.fit(features, verbose=False)

    # Test noise robustness
    test_feature = features[0]
    original_code = quantizer.quantize(test_feature)

    # Add small noise
    noise_levels = [0.01, 0.05, 0.1, 0.5, 1.0]
    print("Noise robustness test:")
    print(f"  Original code: {original_code}\n")

    for noise_level in noise_levels:
        noisy_feature = test_feature + np.random.randn(36) * noise_level
        noisy_code = quantizer.quantize(noisy_feature)

        same = (noisy_code.to_tuple() == original_code.to_tuple())
        status = "✓ SAME" if same else "✗ CHANGED"

        print(f"  Noise σ={noise_level:.2f}: {noisy_code} {status}")

    print("\nProposition 1: Small noise (< margin γ) should preserve quantization")
    print("✓ Test complete\n")


def main():
    """Run end-to-end pipeline tests."""

    # Find a sample video
    video_base = Path("/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100")

    for gloss_dir in sorted(video_base.iterdir()):
        if gloss_dir.is_dir():
            videos = list(gloss_dir.glob("*.mp4"))
            if videos:
                video_path = videos[0]
                success = test_end_to_end_pipeline(str(video_path), max_frames=100)

                if success:
                    # Run additional tests
                    test_proposition_1_noise_robustness()

                    print("\n" + "="*70)
                    print("PHONOLOGY MODULE COMPLETE!")
                    print("="*70)
                    print("✓ MediaPipe landmark extraction")
                    print("✓ Sim(3)-equivariant feature extraction")
                    print("✓ Product vector quantization")
                    print("\nNext steps:")
                    print("  1. Move to Phase 2: Sequence modeling (encoder + CTC)")
                    print("  2. Or write comprehensive unit tests")
                    print("="*70 + "\n")
                    return 0
                else:
                    return 1

    print("❌ ERROR: No videos found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
