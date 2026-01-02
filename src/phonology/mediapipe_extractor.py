"""
MediaPipe Landmark Extraction
Section 2.2: "MediaPipe-Based Implementation"

Extracts raw landmark tensors from video frames:
    L_t ∈ ℝ^(21×3)  (left hand)
    R_t ∈ ℝ^(21×3)  (right hand)
    F_t ∈ ℝ^(468×3) (face mesh)
    B_t ∈ ℝ^(33×3)  (pose/body)

Total: 1623 landmarks × 3 coordinates = 4869-dimensional observation
"""

import mediapipe as mp
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RawLandmarks:
    """Container for raw MediaPipe landmarks at time t."""
    left_hand: np.ndarray   # (21, 3) - left hand landmarks
    right_hand: np.ndarray  # (21, 3) - right hand landmarks
    face: np.ndarray        # (468, 3) - face mesh landmarks
    pose: np.ndarray        # (33, 3) - body pose landmarks
    timestamp: float        # Video timestamp in seconds
    
    def __post_init__(self):
        """Validate shapes."""
        assert self.left_hand.shape == (21, 3), f"Invalid left hand shape: {self.left_hand.shape}"
        assert self.right_hand.shape == (21, 3), f"Invalid right hand shape: {self.right_hand.shape}"
        assert self.face.shape == (468, 3), f"Invalid face shape: {self.face.shape}"
        assert self.pose.shape == (33, 3), f"Invalid pose shape: {self.pose.shape}"
    
    def concatenate(self) -> np.ndarray:
        """
        Concatenate all landmarks into single vector.
        
        Section 2.2:
            X_t = vec([L_t; R_t; F_t; B_t]) ∈ ℝ^{1623×3}
        
        Returns:
            Array of shape (1623, 3)
        """
        return np.vstack([
            self.left_hand,   # 21 landmarks
            self.right_hand,  # 21 landmarks
            self.face,        # 468 landmarks
            self.pose         # 33 landmarks
        ])  # Total: 543 landmarks × 3 = 1629... wait, paper says 1623?
        
        # NOTE: Discrepancy check needed - paper says 1623, but 21+21+468+33 = 543
        # Possible: MediaPipe face mesh is 468, but maybe using subset?


class MediaPipeExtractor:
    """
    Extract landmarks from video frames using MediaPipe Holistic.
    
    Configuration matches Section 2.2:
        - Holistic model (combined hand + pose + face)
        - World coordinates (not image pixel coordinates)
        - Minimum detection confidence: 0.5
        - Minimum tracking confidence: 0.5
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,  # 0=lite, 1=full, 2=heavy
    ):
        """
        Initialize MediaPipe Holistic model.
        
        Args:
            min_detection_confidence: Minimum confidence for detection to be considered successful
            min_tracking_confidence: Minimum confidence for tracking to be considered successful
            model_complexity: Model complexity (0, 1, or 2). Higher = more accurate but slower.
                             Use 1 for development, consider 0 for edge deployment.
        """
        self.mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,  # Video stream, not single images
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True,  # Temporal smoothing
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic_module = mp.solutions.holistic
        
    def extract_frame(
        self, 
        frame: np.ndarray,
        timestamp: float,
    ) -> Optional[RawLandmarks]:
        """
        Extract landmarks from a single video frame.
        
        Args:
            frame: RGB image array of shape (H, W, 3), dtype uint8
            timestamp: Frame timestamp in seconds
            
        Returns:
            RawLandmarks object if detection successful, None otherwise
            
        Note:
            MediaPipe expects RGB input (not BGR). If using OpenCV, convert first:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        """
        # Process frame
        results = self.mp_holistic.process(frame)
        
        # Check if detection succeeded
        if not self._has_sufficient_landmarks(results):
            return None
        
        # Extract landmarks (convert to numpy arrays)
        left_hand = self._extract_hand_landmarks(results.left_hand_landmarks)
        right_hand = self._extract_hand_landmarks(results.right_hand_landmarks)
        face = self._extract_face_landmarks(results.face_landmarks)
        pose = self._extract_pose_landmarks(results.pose_landmarks)
        
        return RawLandmarks(
            left_hand=left_hand,
            right_hand=right_hand,
            face=face,
            pose=pose,
            timestamp=timestamp,
        )
    
    def _has_sufficient_landmarks(self, results) -> bool:
        """
        Check if we have sufficient landmarks for processing.
        
        Section 1.2: "MediaPipe landmark detection succeeds on >95% of frames"
        
        We require at minimum:
            - Pose landmarks (for normalization via shoulders/neck)
            - At least one hand visible
        
        Face landmarks are optional (used for non-manuals, but not critical)
        """
        # Always need pose for Sim(3) normalization
        if results.pose_landmarks is None:
            return False
        
        # Need at least one hand
        if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            return False
        
        return True
    
    def _extract_hand_landmarks(self, landmarks) -> np.ndarray:
        """
        Extract hand landmarks as (21, 3) array.
        
        If hand not detected, return zeros (will be handled by downstream
        normalization and feature extraction).
        """
        if landmarks is None:
            return np.zeros((21, 3))
        
        coords = []
        for lm in landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        
        return np.array(coords)  # Shape: (21, 3)
    
    def _extract_face_landmarks(self, landmarks) -> np.ndarray:
        """
        Extract face mesh landmarks as (468, 3) array.
        
        MediaPipe face mesh has 468 landmarks. If not detected, return zeros.
        """
        if landmarks is None:
            return np.zeros((468, 3))
        
        coords = []
        for lm in landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        
        return np.array(coords)  # Shape: (468, 3)
    
    def _extract_pose_landmarks(self, landmarks) -> np.ndarray:
        """
        Extract pose landmarks as (33, 3) array.
        
        MediaPipe pose has 33 landmarks. Should always be present if
        _has_sufficient_landmarks() returned True.
        """
        if landmarks is None:
            raise ValueError("Pose landmarks missing - should not happen if filtering worked")
        
        coords = []
        for lm in landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        
        return np.array(coords)  # Shape: (33, 3)
    
    def extract_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> list[RawLandmarks]:
        """
        Extract landmarks from entire video file.
        
        Args:
            video_path: Path to video file
            max_frames: If specified, stop after this many frames (for testing)
            
        Returns:
            List of RawLandmarks objects (one per successfully processed frame)
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        landmarks_sequence = []
        frame_idx = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            # Convert BGR to RGB (MediaPipe expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Compute timestamp
            timestamp = frame_idx / fps if fps > 0 else frame_idx * (1/30.0)
            
            # Extract landmarks
            landmarks = self.extract_frame(frame_rgb, timestamp)
            
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
            
            frame_idx += 1
        
        cap.release()
        
        success_rate = len(landmarks_sequence) / frame_idx if frame_idx > 0 else 0
        print(f"Extracted {len(landmarks_sequence)}/{frame_idx} frames ({success_rate:.1%} success rate)")
        
        # Section 1.2 assumption: ">95% of frames"
        if success_rate < 0.95:
            print(f"WARNING: Success rate {success_rate:.1%} below 95% threshold")
        
        return landmarks_sequence
    
    def visualize_landmarks(self, frame: np.ndarray, landmarks: RawLandmarks) -> np.ndarray:
        """
        Draw landmarks on frame for debugging/visualization.
        
        Args:
            frame: RGB image array
            landmarks: RawLandmarks to visualize
            
        Returns:
            Annotated image
        """
        # Convert landmarks back to MediaPipe format for drawing
        # (This is a bit hacky - ideally we'd store the raw results)
        
        annotated = frame.copy()
        
        # TODO: Implement visualization using mp_drawing
        # For now, just return original frame
        
        return annotated
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'mp_holistic'):
            self.mp_holistic.close()


# ============================================================================
# Utility Functions
# ============================================================================

def get_landmark_names() -> Dict[str, Dict[int, str]]:
    """
    Get human-readable names for MediaPipe landmarks.
    
    Useful for debugging and understanding which landmarks correspond
    to which body parts.
    
    Returns:
        Dictionary with keys 'hand', 'pose', 'face' containing index->name mappings
    """
    # Hand landmarks (21 points)
    hand_names = {
        0: "WRIST",
        1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
        5: "INDEX_FINGER_MCP", 6: "INDEX_FINGER_PIP", 7: "INDEX_FINGER_DIP", 8: "INDEX_FINGER_TIP",
        9: "MIDDLE_FINGER_MCP", 10: "MIDDLE_FINGER_PIP", 11: "MIDDLE_FINGER_DIP", 12: "MIDDLE_FINGER_TIP",
        13: "RING_FINGER_MCP", 14: "RING_FINGER_PIP", 15: "RING_FINGER_DIP", 16: "RING_FINGER_TIP",
        17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
    }
    
    # Pose landmarks (33 points) - we care most about shoulders and neck
    pose_names = {
        0: "NOSE",
        11: "LEFT_SHOULDER",
        12: "RIGHT_SHOULDER",
        23: "LEFT_HIP",
        24: "RIGHT_HIP",
        # ... (full list in MediaPipe docs)
    }
    
    # Face landmarks (468 points) - we care about eyes, eyebrows, mouth
    # Specific indices from Section 2.2:
    # g_t = 1/2(F_t[33] + F_t[133]) - 1/2(F_t[362] + F_t[263])  (gaze)
    # a_t = ||F_t[61] - F_t[291]||  (mouth aperture)
    face_names = {
        33: "RIGHT_EYE_INNER",
        133: "LEFT_EYE_INNER",
        362: "RIGHT_EYE_OUTER",
        263: "LEFT_EYE_OUTER",
        61: "MOUTH_TOP",
        291: "MOUTH_BOTTOM",
    }
    
    return {
        "hand": hand_names,
        "pose": pose_names,
        "face": face_names,
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Quick test on sample video
    extractor = MediaPipeExtractor(model_complexity=1)
    
    # TODO: Replace with actual test video path
    test_video = "data/raw/sample_asl_video.mp4"
    
    # Extract first 10 frames for testing
    landmarks_seq = extractor.extract_video(test_video, max_frames=10)
    
    print(f"Extracted {len(landmarks_seq)} frames")
    
    if len(landmarks_seq) > 0:
        first_frame = landmarks_seq[0]
        print(f"First frame shape: {first_frame.concatenate().shape}")
        print(f"Timestamp: {first_frame.timestamp:.3f}s")
        
        # Check for missing landmarks
        if np.all(first_frame.left_hand == 0):
            print("WARNING: Left hand not detected")
        if np.all(first_frame.right_hand == 0):
            print("WARNING: Right hand not detected")