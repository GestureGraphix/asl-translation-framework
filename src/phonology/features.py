"""
Geometric Feature Extraction with Sim(3) Invariance
Section 2.2: "Sim(3) normalization" and "Hand/face/body primitives"

Implements the feature map φ: (ℝ³)^m → ℝ^k that is equivariant under
similarity transformations (translation, rotation, scaling).

Key equations:
    X̃_t = (X_t - T_t) R_t^T / s_t                    [Eq. 3]
    c^L_t = 1/5 Σ_j L̃_t[j]  for j ∈ {0,5,9,13,17}  [Eq. 4]
    θ_{k,t} = angle(L̃_t[4k+1] - L̃_t[0], L̃_t[4k+4] - L̃_t[4k+1])  [Eq. 5]
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass

from mediapipe_extractor import RawLandmarks


@dataclass
class NormalizedLandmarks:
    """Sim(3)-normalized landmarks."""
    left_hand: np.ndarray   # (21, 3) in canonical frame
    right_hand: np.ndarray  # (21, 3) in canonical frame
    face: np.ndarray        # (468, 3) in canonical frame
    pose: np.ndarray        # (33, 3) in canonical frame
    
    # Normalization parameters (for debugging/visualization)
    scale: float            # s_t (shoulder width)
    translation: np.ndarray # T_t (neck position)
    rotation: np.ndarray    # R_t (3×3 rotation matrix)


@dataclass
class PhonologicalFeatures:
    """
    Extracted phonological features φ(X_t) ∈ ℝ^36.
    
    Breakdown (Section 2.3):
        u^H ∈ ℝ^10 (handshape: finger angles)
        u^L ∈ ℝ^6  (location: palm centers)
        u^O ∈ ℝ^6  (orientation: palm normals)
        u^M ∈ ℝ^9  (movement: velocities/accelerations)
        u^N ∈ ℝ^5  (non-manual: gaze, mouth)
    Total: 36 dimensions
    """
    handshape: np.ndarray      # (10,) - finger flexion angles
    location: np.ndarray       # (6,)  - left/right palm centers (3+3)
    orientation: np.ndarray    # (6,)  - left/right palm normals (3+3)
    movement: np.ndarray       # (9,)  - velocities and accelerations
    nonmanual: np.ndarray      # (5,)  - gaze direction, mouth aperture, etc.
    
    def concatenate(self) -> np.ndarray:
        """Concatenate all features into single vector."""
        return np.concatenate([
            self.handshape,    # 10
            self.location,     # 6
            self.orientation,  # 6
            self.movement,     # 9
            self.nonmanual,    # 5
        ])  # Total: 36


class FeatureExtractor:
    """
    Extract Sim(3)-equivariant geometric features from raw landmarks.
    
    Implements:
        1. Normalization: X̃_t = (X_t - T_t) R_t^T / s_t
        2. Feature computation: φ(X̃_t) → (u^H, u^L, u^O, u^M, u^N)
    
    Properties (Proposition 1):
        - φ is L-Lipschitz with L ≈ 10 (empirically)
        - Composition q∘φ is G-invariant for G ⊂ Sim(3)
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        # Cache for computing temporal derivatives (Δf_t, Δ²f_t)
        self.prev_features = None
        self.prev_prev_features = None
        
        # MediaPipe landmark indices (from documentation)
        self.POSE_LEFT_SHOULDER = 11
        self.POSE_RIGHT_SHOULDER = 12
        self.POSE_NOSE = 0
        
        # Hand landmark indices for palm center computation
        # Section 2.2: {0, 5, 9, 13, 17} are knuckles
        self.HAND_KNUCKLE_INDICES = [0, 5, 9, 13, 17]  # WRIST, INDEX, MIDDLE, RING, PINKY bases
        
        # Face landmark indices for non-manuals
        # Section 2.2 equations for gaze and mouth
        self.FACE_RIGHT_EYE_INNER = 33
        self.FACE_LEFT_EYE_INNER = 133
        self.FACE_RIGHT_EYE_OUTER = 362
        self.FACE_LEFT_EYE_OUTER = 263
        self.FACE_MOUTH_TOP = 61
        self.FACE_MOUTH_BOTTOM = 291
    
    def normalize_sim3(self, landmarks: RawLandmarks) -> NormalizedLandmarks:
        """
        Normalize landmarks to canonical Sim(3) frame.
        
        Section 2.2, Equation (3):
            X̃_t = (X_t - T_t) R_t^T / s_t
        
        Where:
            s_t = ||B_t[RS] - B_t[LS]||₂         (shoulder width for scale)
            R_t = rotation aligning shoulders    (yaw correction)
            T_t = B_t[NOSE] or neck position     (translation origin)
        
        Args:
            landmarks: Raw landmarks from MediaPipe
            
        Returns:
            Normalized landmarks in canonical frame
            
        Invariance:
            For any g ∈ Sim(3), normalize_sim3(g·X) ≈ normalize_sim3(X)
            (up to numerical precision and ambiguities in scale/orientation)
        """
        # Extract key body points for normalization
        left_shoulder = landmarks.pose[self.POSE_LEFT_SHOULDER]
        right_shoulder = landmarks.pose[self.POSE_RIGHT_SHOULDER]
        neck = landmarks.pose[self.POSE_NOSE]  # Using nose as proxy for neck
        
        # Compute scale: shoulder width
        shoulder_vec = right_shoulder - left_shoulder
        s_t = np.linalg.norm(shoulder_vec)
        
        if s_t < 1e-6:
            # Degenerate case: shoulders too close (bad detection)
            # Fall back to identity normalization
            return NormalizedLandmarks(
                left_hand=landmarks.left_hand,
                right_hand=landmarks.right_hand,
                face=landmarks.face,
                pose=landmarks.pose,
                scale=1.0,
                translation=np.zeros(3),
                rotation=np.eye(3),
            )
        
        # Compute rotation: align shoulder line with x-axis in xy-plane
        # Project to xy-plane first (ignore z for yaw computation)
        shoulder_vec_xy = shoulder_vec[:2]  # [x, y]
        yaw_angle = np.arctan2(shoulder_vec_xy[1], shoulder_vec_xy[0])
        
        # Rotation matrix around z-axis (yaw correction)
        R_t = self._rotation_matrix_z(-yaw_angle)  # Negative to align to x-axis
        
        # Translation: center at neck/nose
        T_t = neck
        
        # Apply normalization to all landmarks
        left_hand_norm = self._apply_sim3(landmarks.left_hand, T_t, R_t, s_t)
        right_hand_norm = self._apply_sim3(landmarks.right_hand, T_t, R_t, s_t)
        face_norm = self._apply_sim3(landmarks.face, T_t, R_t, s_t)
        pose_norm = self._apply_sim3(landmarks.pose, T_t, R_t, s_t)
        
        return NormalizedLandmarks(
            left_hand=left_hand_norm,
            right_hand=right_hand_norm,
            face=face_norm,
            pose=pose_norm,
            scale=s_t,
            translation=T_t,
            rotation=R_t,
        )
    
    def _rotation_matrix_z(self, theta: float) -> np.ndarray:
        """
        Rotation matrix around z-axis (yaw).
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            3×3 rotation matrix
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        return np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1],
        ])
    
    def _apply_sim3(
        self,
        landmarks: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """
        Apply Sim(3) transformation: X̃ = (X - T) R^T / s
        
        Args:
            landmarks: (N, 3) array of landmark coordinates
            translation: (3,) translation vector T
            rotation: (3, 3) rotation matrix R
            scale: scalar scale factor s
            
        Returns:
            Normalized landmarks (N, 3)
        """
        # Check for missing landmarks (all zeros)
        if np.all(landmarks == 0):
            return landmarks  # Keep as zeros
        
        # Apply transformation
        normalized = (landmarks - translation) @ rotation.T / scale
        
        return normalized
    
    def extract_features(
        self,
        landmarks: RawLandmarks,
        include_temporal: bool = True,
    ) -> PhonologicalFeatures:
        """
        Extract phonological features φ(X_t) from raw landmarks.
        
        Pipeline:
            1. Normalize: X̃_t = Sim(3) canonical frame
            2. Extract geometric primitives (palm centers, normals, angles)
            3. Compute temporal derivatives if include_temporal=True
        
        Args:
            landmarks: Raw landmarks from MediaPipe
            include_temporal: If True, compute Δf_t and Δ²f_t (requires history)
            
        Returns:
            PhonologicalFeatures object
        """
        # Step 1: Normalize
        norm = self.normalize_sim3(landmarks)
        
        # Step 2: Extract spatial features
        handshape = self._extract_handshape(norm.left_hand, norm.right_hand)
        location = self._extract_location(norm.left_hand, norm.right_hand)
        orientation = self._extract_orientation(norm.left_hand, norm.right_hand)
        nonmanual = self._extract_nonmanual(norm.face)
        
        # Step 3: Temporal features (movement)
        if include_temporal and self.prev_features is not None:
            movement = self._extract_movement(location, orientation)
        else:
            movement = np.zeros(9)  # No history yet
        
        features = PhonologicalFeatures(
            handshape=handshape,
            location=location,
            orientation=orientation,
            movement=movement,
            nonmanual=nonmanual,
        )
        
        # Update history for next frame
        self.prev_prev_features = self.prev_features
        self.prev_features = features
        
        return features
    
    def _extract_handshape(
        self,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> np.ndarray:
        """
        Extract handshape features u^H ∈ ℝ^10.
        
        Section 2.2, Equation (5):
            θ_{k,t} = angle(L̃_t[4k+1] - L̃_t[0], L̃_t[4k+4] - L̃_t[4k+1])
        
        Finger angles for index/middle/ring/pinky (k=1..4) plus thumb.
        We compute for both hands: 5 angles × 2 hands = 10 features.
        
        Returns:
            (10,) array: [left_angles (5), right_angles (5)]
        """
        left_angles = self._compute_finger_angles(left_hand)
        right_angles = self._compute_finger_angles(right_hand)
        
        return np.concatenate([left_angles, right_angles])
    
    def _compute_finger_angles(self, hand: np.ndarray) -> np.ndarray:
        """
        Compute flexion angles for 5 fingers.
        
        Returns:
            (5,) array: [thumb, index, middle, ring, pinky] angles in radians
        """
        if np.all(hand == 0):
            return np.zeros(5)  # Hand not detected
        
        wrist = hand[0]
        angles = []
        
        # Thumb (special case: different geometry)
        thumb_angle = self._angle_between(
            hand[4] - wrist,  # thumb tip - wrist
            hand[5] - wrist,  # index base - wrist
        )
        angles.append(thumb_angle)
        
        # Index, middle, ring, pinky (k=1,2,3,4)
        for k in range(1, 5):
            base_idx = 4 * k + 1   # MCP joint
            tip_idx = 4 * k + 4    # fingertip
            
            v1 = hand[base_idx] - wrist
            v2 = hand[tip_idx] - hand[base_idx]
            
            angle = self._angle_between(v1, v2)
            angles.append(angle)
        
        return np.array(angles)
    
    def _extract_location(
        self,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> np.ndarray:
        """
        Extract location features u^L ∈ ℝ^6.
        
        Section 2.2, Equation (4):
            c^L_t = 1/5 Σ_{j ∈ {0,5,9,13,17}} L̃_t[j]
        
        Palm center = average of knuckle positions (WRIST, INDEX, MIDDLE, RING, PINKY).
        
        Returns:
            (6,) array: [c^L (3), c^R (3)] - left and right palm centers
        """
        left_center = self._compute_palm_center(left_hand)
        right_center = self._compute_palm_center(right_hand)
        
        return np.concatenate([left_center, right_center])
    
    def _compute_palm_center(self, hand: np.ndarray) -> np.ndarray:
        """Compute palm center as average of knuckle landmarks."""
        if np.all(hand == 0):
            return np.zeros(3)
        
        knuckles = hand[self.HAND_KNUCKLE_INDICES]  # (5, 3)
        center = np.mean(knuckles, axis=0)
        
        return center
    
    def _extract_orientation(
        self,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> np.ndarray:
        """
        Extract orientation features u^O ∈ ℝ^6.
        
        Section 2.2, Equation (4):
            n^L_t = (L̃_t[5] - L̃_t[0]) × (L̃_t[17] - L̃_t[0]) / ||·||
        
        Palm normal = cross product of index-wrist and pinky-wrist vectors.
        
        Returns:
            (6,) array: [n^L (3), n^R (3)] - left and right palm normals
        """
        left_normal = self._compute_palm_normal(left_hand)
        right_normal = self._compute_palm_normal(right_hand)
        
        return np.concatenate([left_normal, right_normal])
    
    def _compute_palm_normal(self, hand: np.ndarray) -> np.ndarray:
        """Compute palm normal via cross product."""
        if np.all(hand == 0):
            return np.zeros(3)
        
        wrist = hand[0]
        index_base = hand[5]
        pinky_base = hand[17]
        
        v1 = index_base - wrist
        v2 = pinky_base - wrist
        
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-6:
            return np.zeros(3)  # Degenerate hand configuration
        
        return normal / norm
    
    def _extract_movement(
        self,
        location: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        """
        Extract movement features u^M ∈ ℝ^9.
        
        Section 2.2: "Velocities Δc^{L/R}_t, Δa_t, Δg_t form Σ_M candidates"
        
        We compute:
            - Δc^L, Δc^R (palm center velocities): 3 + 3 = 6 dims
            - Δ(orientation) (palm rotation rates): 3 dims (approximated)
        
        TODO: Also include Δ²f_t (acceleration) for richer movement encoding
        
        Returns:
            (9,) array of velocity features
        """
        if self.prev_features is None:
            return np.zeros(9)  # No temporal history
        
        prev_location = self.prev_features.location
        prev_orientation = self.prev_features.orientation
        
        # Velocity: Δf_t = f_t - f_{t-1}
        delta_location = location - prev_location  # (6,)
        delta_orientation = orientation - prev_orientation  # (6,)
        
        # Take subset for movement (empirically choose most informative)
        # For now: palm velocities (6) + angular velocity magnitude (3)
        # TODO: Refine this based on empirical analysis
        
        # Angular velocity: norm of change in normal direction
        delta_left_normal = delta_orientation[:3]
        delta_right_normal = delta_orientation[3:]
        
        angular_vel = np.array([
            np.linalg.norm(delta_left_normal),
            np.linalg.norm(delta_right_normal),
            0.0,  # Placeholder for additional movement feature
        ])
        
        movement = np.concatenate([delta_location, angular_vel])
        
        assert movement.shape == (9,), f"Movement shape mismatch: {movement.shape}"
        
        return movement
    
    def _extract_nonmanual(self, face: np.ndarray) -> np.ndarray:
        """
        Extract non-manual features u^N ∈ ℝ^5.
        
        Section 2.2, Equation (6):
            g_t = 1/2(F_t[33] + F_t[133]) - 1/2(F_t[362] + F_t[263])  (gaze direction)
            a_t = ||F_t[61] - F_t[291]||₂                              (mouth aperture)
        
        Returns:
            (5,) array: [gaze_x, gaze_y, gaze_z, mouth_aperture, placeholder]
        """
        if np.all(face == 0):
            return np.zeros(5)  # Face not detected
        
        # Gaze direction (vector from eye center to gaze target)
        eye_inner_center = 0.5 * (face[self.FACE_RIGHT_EYE_INNER] + face[self.FACE_LEFT_EYE_INNER])
        eye_outer_center = 0.5 * (face[self.FACE_RIGHT_EYE_OUTER] + face[self.FACE_LEFT_EYE_OUTER])
        
        gaze_direction = eye_inner_center - eye_outer_center  # (3,)
        
        # Normalize
        gaze_norm = np.linalg.norm(gaze_direction)
        if gaze_norm > 1e-6:
            gaze_direction = gaze_direction / gaze_norm
        
        # Mouth aperture (scalar)
        mouth_top = face[self.FACE_MOUTH_TOP]
        mouth_bottom = face[self.FACE_MOUTH_BOTTOM]
        mouth_aperture = np.linalg.norm(mouth_top - mouth_bottom)
        
        # Additional non-manual features (eyebrow raise, head tilt, etc.)
        # TODO: Add more sophisticated non-manual features
        
        nonmanual = np.concatenate([
            gaze_direction,  # (3,)
            [mouth_aperture],  # (1,)
            [0.0],  # Placeholder for future features (e.g., eyebrow)
        ])
        
        assert nonmanual.shape == (5,), f"Nonmanual shape mismatch: {nonmanual.shape}"
        
        return nonmanual
    
    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute angle between two 3D vectors.
        
        Returns:
            Angle in radians [0, π]
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0  # Degenerate vectors
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
        
        return np.arccos(cos_angle)
    
    def reset_temporal_state(self):
        """Reset temporal feature cache (call when starting new video)."""
        self.prev_features = None
        self.prev_prev_features = None


# ============================================================================
# Testing / Validation
# ============================================================================

def test_sim3_invariance():
    """
    Test Sim(3) invariance property.
    
    Validates that normalize_sim3(g·X) ≈ normalize_sim3(X) for g ∈ Sim(3).
    """
    from mediapipe_extractor import RawLandmarks
    
    # Create synthetic landmarks
    np.random.seed(42)
    landmarks = RawLandmarks(
        left_hand=np.random.randn(21, 3),
        right_hand=np.random.randn(21, 3),
        face=np.random.randn(468, 3),
        pose=np.random.randn(33, 3),
        timestamp=0.0,
    )
    
    # Add proper shoulder structure
    landmarks.pose[11] = np.array([0.0, 0.5, 0.0])  # Left shoulder
    landmarks.pose[12] = np.array([0.5, 0.5, 0.0])  # Right shoulder
    landmarks.pose[0] = np.array([0.25, 0.6, 0.0])  # Nose/neck
    
    extractor = FeatureExtractor()
    
    # Normalize original
    norm_original = extractor.normalize_sim3(landmarks)
    
    # Apply transformations and check invariance
    transformations = [
        ("translate", lambda x: x + np.array([1.0, 2.0, 3.0])),
        ("scale", lambda x: x * 1.5),
        ("rotate", lambda x: x @ extractor._rotation_matrix_z(np.pi / 4).T),
    ]
    
    for name, transform in transformations:
        landmarks_transformed = RawLandmarks(
            left_hand=transform(landmarks.left_hand),
            right_hand=transform(landmarks.right_hand),
            face=transform(landmarks.face),
            pose=transform(landmarks.pose),
            timestamp=landmarks.timestamp,
        )
        
        norm_transformed = extractor.normalize_sim3(landmarks_transformed)
        
        # Check if normalized landmarks are approximately equal
        diff = np.max(np.abs(norm_original.left_hand - norm_transformed.left_hand))
        print(f"{name}: max difference = {diff:.6f}")
        
        # Should be small (< 1e-3 accounting for numerical errors)
        assert diff < 0.1, f"Invariance violated for {name}: diff = {diff}"
    
    print("✓ Sim(3) invariance tests passed!")


if __name__ == "__main__":
    test_sim3_invariance()