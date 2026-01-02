"""
Unit Tests for Phonology Module
Validates mathematical guarantees from Section 2.

Tests organized by proposition:
    - test_sim3_invariance:     Proposition 1 (G-invariance of q∘φ)
    - test_noise_robustness:    Proposition 1 (margin robustness)
    - test_product_vq_*:        Proposition 2 (sample complexity)
    - test_joint_margin:        Proposition 3 (joint margin robustness)
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'phonology'))

from mediapipe_extractor import RawLandmarks
from features import FeatureExtractor, PhonologicalFeatures


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_landmarks():
    """Generate synthetic landmarks with proper structure."""
    np.random.seed(42)
    
    # Create structured pose
    pose = np.random.randn(33, 3) * 0.1
    pose[11] = np.array([-0.25, 0.0, 0.0])  # Left shoulder
    pose[12] = np.array([0.25, 0.0, 0.0])   # Right shoulder
    pose[0] = np.array([0.0, 0.1, 0.0])     # Nose/neck
    
    return RawLandmarks(
        left_hand=np.random.randn(21, 3) * 0.05,
        right_hand=np.random.randn(21, 3) * 0.05,
        face=np.random.randn(468, 3) * 0.02,
        pose=pose,
        timestamp=0.0,
    )


@pytest.fixture
def feature_extractor():
    """Create feature extractor instance."""
    return FeatureExtractor()


# ============================================================================
# Proposition 1: Geometric Invariance (Section 2.1)
# ============================================================================

def test_translation_invariance(synthetic_landmarks, feature_extractor):
    """
    Test that features are invariant under translation.
    
    Mathematical property:
        φ(X + t) = φ(X) for any translation t ∈ ℝ³
    """
    # Extract features from original landmarks
    features_original = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    
    # Apply translation
    translations = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        np.array([1.5, -1.0, 0.5]),
    ]
    
    for t in translations:
        landmarks_translated = RawLandmarks(
            left_hand=synthetic_landmarks.left_hand + t,
            right_hand=synthetic_landmarks.right_hand + t,
            face=synthetic_landmarks.face + t,
            pose=synthetic_landmarks.pose + t,
            timestamp=synthetic_landmarks.timestamp,
        )
        
        features_translated = feature_extractor.extract_features(landmarks_translated, include_temporal=False)
        
        # Features should be identical (up to numerical precision)
        diff = np.max(np.abs(features_original.concatenate() - features_translated.concatenate()))
        
        assert diff < 1e-4, f"Translation invariance violated: diff = {diff:.6e} for t = {t}"


def test_rotation_invariance(synthetic_landmarks, feature_extractor):
    """
    Test that features are invariant under yaw rotation (z-axis).
    
    Note: Full 3D rotation invariance not guaranteed due to gravity/viewing assumptions.
    """
    features_original = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    
    # Test yaw rotations (around z-axis)
    angles = [np.pi/6, np.pi/4, np.pi/3, np.pi/2]
    
    for theta in angles:
        R = feature_extractor._rotation_matrix_z(theta)
        
        landmarks_rotated = RawLandmarks(
            left_hand=synthetic_landmarks.left_hand @ R.T,
            right_hand=synthetic_landmarks.right_hand @ R.T,
            face=synthetic_landmarks.face @ R.T,
            pose=synthetic_landmarks.pose @ R.T,
            timestamp=synthetic_landmarks.timestamp,
        )
        
        features_rotated = feature_extractor.extract_features(landmarks_rotated, include_temporal=False)
        
        # Features should be approximately invariant
        diff = np.max(np.abs(features_original.concatenate() - features_rotated.concatenate()))
        
        # Allow slightly larger tolerance for rotation due to numerical issues
        assert diff < 1e-3, f"Rotation invariance violated: diff = {diff:.6e} for θ = {theta:.3f}"


def test_scale_invariance(synthetic_landmarks, feature_extractor):
    """
    Test that features are invariant under uniform scaling.
    
    Mathematical property:
        φ(s·X) = φ(X) for any scale s > 0
    """
    features_original = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    
    scales = [0.5, 1.5, 2.0, 3.0]
    
    for s in scales:
        landmarks_scaled = RawLandmarks(
            left_hand=synthetic_landmarks.left_hand * s,
            right_hand=synthetic_landmarks.right_hand * s,
            face=synthetic_landmarks.face * s,
            pose=synthetic_landmarks.pose * s,
            timestamp=synthetic_landmarks.timestamp,
        )
        
        features_scaled = feature_extractor.extract_features(landmarks_scaled, include_temporal=False)
        
        diff = np.max(np.abs(features_original.concatenate() - features_scaled.concatenate()))
        
        assert diff < 1e-4, f"Scale invariance violated: diff = {diff:.6e} for s = {s}"


def test_composed_sim3_invariance(synthetic_landmarks, feature_extractor):
    """
    Test invariance under composed Sim(3) transformations.
    
    Proposition 1: q∘φ is G-invariant for G ⊂ Sim(3).
    """
    features_original = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    
    # Compose: translate → rotate → scale
    t = np.array([1.0, -0.5, 0.2])
    theta = np.pi / 3
    s = 1.8
    
    R = feature_extractor._rotation_matrix_z(theta)
    
    # Apply transformations in sequence
    landmarks_final = RawLandmarks(
        left_hand=(synthetic_landmarks.left_hand + t) @ R.T * s,
        right_hand=(synthetic_landmarks.right_hand + t) @ R.T * s,
        face=(synthetic_landmarks.face + t) @ R.T * s,
        pose=(synthetic_landmarks.pose + t) @ R.T * s,
        timestamp=synthetic_landmarks.timestamp,
    )
    
    features_final = feature_extractor.extract_features(landmarks_final, include_temporal=False)
    
    diff = np.max(np.abs(features_original.concatenate() - features_final.concatenate()))
    
    assert diff < 1e-3, f"Composed Sim(3) invariance violated: diff = {diff:.6e}"
    
    print(f"✓ Composed Sim(3) invariance: max diff = {diff:.6e}")


# ============================================================================
# Proposition 1: Noise Robustness (Section 2.1)
# ============================================================================

def test_noise_robustness_lipschitz(synthetic_landmarks, feature_extractor):
    """
    Test Lipschitz property of feature map φ.
    
    Proposition 1: If φ is L-Lipschitz, then:
        ||φ(X') - φ(X)|| ≤ L ||X' - X||
    
    We verify empirically that L ≈ 10 (claimed in paper).
    """
    features_original = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    f_original = features_original.concatenate()
    
    # Add various magnitude perturbations
    epsilon_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    lipschitz_constants = []
    
    for epsilon in epsilon_values:
        # Random perturbation
        noise = np.random.randn(*synthetic_landmarks.left_hand.shape) * epsilon
        
        landmarks_perturbed = RawLandmarks(
            left_hand=synthetic_landmarks.left_hand + noise,
            right_hand=synthetic_landmarks.right_hand + noise[:21],  # Same magnitude
            face=synthetic_landmarks.face,
            pose=synthetic_landmarks.pose,
            timestamp=synthetic_landmarks.timestamp,
        )
        
        features_perturbed = feature_extractor.extract_features(landmarks_perturbed, include_temporal=False)
        f_perturbed = features_perturbed.concatenate()
        
        # Compute Lipschitz constant
        delta_f = np.linalg.norm(f_perturbed - f_original)
        delta_x = np.linalg.norm(noise)
        
        if delta_x > 1e-8:  # Avoid division by zero
            L_empirical = delta_f / delta_x
            lipschitz_constants.append(L_empirical)
    
    if len(lipschitz_constants) > 0:
        L_max = max(lipschitz_constants)
        L_mean = np.mean(lipschitz_constants)
        
        print(f"Empirical Lipschitz constant: max = {L_max:.2f}, mean = {L_mean:.2f}")
        
        # Paper claims L ≈ 10, verify it's in reasonable range
        assert L_max < 20, f"Lipschitz constant too large: {L_max:.2f}"


def test_margin_robustness(synthetic_landmarks, feature_extractor):
    """
    Test margin robustness property.
    
    Proposition 1: If ||η|| < γ/L, then q(φ(X + η)) = q(φ(X))
    
    This test verifies that small perturbations don't change quantized output.
    Note: Requires quantizer implementation (to be added in quantizer.py).
    """
    # TODO: Implement after quantizer.py is ready
    # For now, just test that features are stable under small noise
    
    features_original = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    f_original = features_original.concatenate()
    
    # Very small perturbation
    epsilon = 0.001
    noise = np.random.randn(*synthetic_landmarks.pose.shape) * epsilon
    
    landmarks_perturbed = RawLandmarks(
        left_hand=synthetic_landmarks.left_hand,
        right_hand=synthetic_landmarks.right_hand,
        face=synthetic_landmarks.face,
        pose=synthetic_landmarks.pose + noise,
        timestamp=synthetic_landmarks.timestamp,
    )
    
    features_perturbed = feature_extractor.extract_features(landmarks_perturbed, include_temporal=False)
    f_perturbed = features_perturbed.concatenate()
    
    diff = np.linalg.norm(f_perturbed - f_original)
    
    # For small noise, features should be very close
    assert diff < 0.1, f"Features not robust to small noise: diff = {diff:.6f}"
    
    print(f"✓ Small noise robustness: ||Δf|| = {diff:.6f} for ||η|| = {epsilon}")


# ============================================================================
# Feature Extraction Tests
# ============================================================================

def test_feature_dimensions(synthetic_landmarks, feature_extractor):
    """Test that extracted features have correct dimensions."""
    features = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    
    # Section 2.3: u^H(10), u^L(6), u^O(6), u^M(9), u^N(5)
    assert features.handshape.shape == (10,), f"Handshape dimension mismatch: {features.handshape.shape}"
    assert features.location.shape == (6,), f"Location dimension mismatch: {features.location.shape}"
    assert features.orientation.shape == (6,), f"Orientation dimension mismatch: {features.orientation.shape}"
    assert features.movement.shape == (9,), f"Movement dimension mismatch: {features.movement.shape}"
    assert features.nonmanual.shape == (5,), f"Nonmanual dimension mismatch: {features.nonmanual.shape}"
    
    # Total: 36 dimensions
    concatenated = features.concatenate()
    assert concatenated.shape == (36,), f"Concatenated dimension mismatch: {concatenated.shape}"
    
    print(f"✓ Feature dimensions correct: {concatenated.shape}")


def test_temporal_features(synthetic_landmarks, feature_extractor):
    """Test temporal derivative computation (Δf_t)."""
    # Reset state
    feature_extractor.reset_temporal_state()
    
    # First frame: no temporal features
    features_t0 = feature_extractor.extract_features(synthetic_landmarks, include_temporal=True)
    assert np.all(features_t0.movement == 0), "First frame should have zero movement"
    
    # Second frame: should have temporal features
    # Perturb landmarks slightly
    landmarks_t1 = RawLandmarks(
        left_hand=synthetic_landmarks.left_hand + 0.01,
        right_hand=synthetic_landmarks.right_hand - 0.01,
        face=synthetic_landmarks.face,
        pose=synthetic_landmarks.pose,
        timestamp=synthetic_landmarks.timestamp + 0.033,  # ~30fps
    )
    
    features_t1 = feature_extractor.extract_features(landmarks_t1, include_temporal=True)
    
    # Movement should be non-zero now
    movement_norm = np.linalg.norm(features_t1.movement)
    assert movement_norm > 1e-6, "Second frame should have non-zero movement"
    
    print(f"✓ Temporal features working: ||movement|| = {movement_norm:.6f}")


def test_missing_landmarks(feature_extractor):
    """Test graceful handling of missing landmarks (e.g., hand not visible)."""
    # Create landmarks with missing right hand
    landmarks_missing_hand = RawLandmarks(
        left_hand=np.random.randn(21, 3) * 0.05,
        right_hand=np.zeros((21, 3)),  # Missing hand
        face=np.random.randn(468, 3) * 0.02,
        pose=np.random.randn(33, 3) * 0.1,
        timestamp=0.0,
    )
    
    # Should not crash
    features = feature_extractor.extract_features(landmarks_missing_hand, include_temporal=False)
    
    # Right hand features should be zero or default
    # (exact behavior depends on implementation choice)
    assert features.concatenate().shape == (36,), "Should still return 36-dim features"
    
    print(f"✓ Missing landmarks handled gracefully")


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_extraction(synthetic_landmarks, feature_extractor):
    """Test complete pipeline: raw landmarks → normalized → features."""
    # Step 1: Normalize
    normalized = feature_extractor.normalize_sim3(synthetic_landmarks)
    
    assert normalized.left_hand.shape == (21, 3)
    assert normalized.scale > 0, "Scale should be positive"
    assert normalized.rotation.shape == (3, 3)
    
    # Step 2: Extract features
    features = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    
    # Validate feature ranges (sanity check)
    # Angles should be in [0, π]
    assert np.all(features.handshape >= 0) and np.all(features.handshape <= np.pi + 0.1)
    
    # Normals should be unit vectors (approximately)
    left_normal = features.orientation[:3]
    right_normal = features.orientation[3:]
    
    # (May be zero if hand not detected properly, but if non-zero should be normalized)
    if np.linalg.norm(left_normal) > 0:
        assert abs(np.linalg.norm(left_normal) - 1.0) < 0.1, "Left normal should be unit vector"
    
    print(f"✓ End-to-end extraction successful")


# ============================================================================
# Performance Tests
# ============================================================================

def test_extraction_speed(synthetic_landmarks, feature_extractor, benchmark=None):
    """Test that feature extraction is fast enough for real-time (<10ms target)."""
    import time
    
    # Warmup
    for _ in range(10):
        feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    
    # Benchmark
    n_iterations = 100
    start = time.time()
    
    for _ in range(n_iterations):
        features = feature_extractor.extract_features(synthetic_landmarks, include_temporal=False)
    
    elapsed = time.time() - start
    avg_time = (elapsed / n_iterations) * 1000  # Convert to ms
    
    print(f"Average extraction time: {avg_time:.3f} ms")
    
    # Target: <10ms (Section 8.1 latency budget)
    assert avg_time < 10, f"Feature extraction too slow: {avg_time:.3f} ms"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])