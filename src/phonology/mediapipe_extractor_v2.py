"""
Real MediaPipe Landmark Extraction (MediaPipe 0.10+ Tasks API)
Section 2.2: "MediaPipe-Based Implementation"

Uses MediaPipe 0.10+ Tasks API with separate landmarkers for:
    - Hand landmarks (left + right)
    - Face mesh landmarks
    - Pose landmarks

This replaces the synthetic extractor with actual MediaPipe detection.
"""

import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class RawLandmarks:
    """Container for raw MediaPipe landmarks at time t."""
    left_hand: np.ndarray   # (21, 3) - left hand landmarks
    right_hand: np.ndarray  # (21, 3) - right hand landmarks
    face: np.ndarray        # (478, 3) - face mesh landmarks (MediaPipe 0.10+)
    pose: np.ndarray        # (33, 3) - body pose landmarks
    timestamp: float        # Video timestamp in seconds

    def __post_init__(self):
        """Validate shapes."""
        assert self.left_hand.shape == (21, 3), f"Invalid left hand shape: {self.left_hand.shape}"
        assert self.right_hand.shape == (21, 3), f"Invalid right hand shape: {self.right_hand.shape}"
        assert self.face.shape == (478, 3), f"Invalid face shape: {self.face.shape}"
        assert self.pose.shape == (33, 3), f"Invalid pose shape: {self.pose.shape}"

    def concatenate(self) -> np.ndarray:
        """
        Concatenate all landmarks into single vector.

        Returns:
            Array of shape (553, 3) - total landmarks
        """
        return np.vstack([
            self.left_hand,   # 21 landmarks
            self.right_hand,  # 21 landmarks
            self.face,        # 478 landmarks
            self.pose         # 33 landmarks
        ])  # Total: 553 landmarks × 3


class MediaPipeExtractor:
    """
    Real MediaPipe extractor using Tasks API (v0.10+).

    Uses separate landmarkers for hands, face, and pose.
    Processes each frame independently (no state maintenance needed).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        """
        Initialize MediaPipe landmarkers.

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        # Initialize landmarkers
        self._init_hand_landmarker()
        self._init_face_landmarker()
        self._init_pose_landmarker()

        print(f"✓ Real MediaPipe extractor initialized")
        print(f"  Hand detection confidence: {min_detection_confidence}")
        print(f"  Face detection confidence: {min_detection_confidence}")
        print(f"  Pose detection confidence: {min_detection_confidence}")

    def _init_hand_landmarker(self):
        """Initialize hand landmarker."""
        from pathlib import Path
        model_path = str(Path(__file__).parent.parent.parent / "models" / "mediapipe" / "hand_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,  # Detect both hands
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    def _init_face_landmarker(self):
        """Initialize face landmarker."""
        from pathlib import Path
        model_path = str(Path(__file__).parent.parent.parent / "models" / "mediapipe" / "face_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def _init_pose_landmarker(self):
        """Initialize pose landmarker."""
        from pathlib import Path
        model_path = str(Path(__file__).parent.parent.parent / "models" / "mediapipe" / "pose_landmarker_full.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)

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
        """
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect hands
        hand_result = self.hand_landmarker.detect(mp_image)

        # Detect face
        face_result = self.face_landmarker.detect(mp_image)

        # Detect pose
        pose_result = self.pose_landmarker.detect(mp_image)

        # Check if we have minimum required landmarks (pose + at least one hand)
        if not pose_result.pose_landmarks:
            return None

        if not hand_result.hand_landmarks and not hand_result.hand_landmarks:
            return None  # Need at least one hand

        # Extract landmarks
        left_hand = self._extract_hand_landmarks(hand_result, handedness='Left')
        right_hand = self._extract_hand_landmarks(hand_result, handedness='Right')
        face = self._extract_face_landmarks(face_result)
        pose = self._extract_pose_landmarks(pose_result)

        return RawLandmarks(
            left_hand=left_hand,
            right_hand=right_hand,
            face=face,
            pose=pose,
            timestamp=timestamp,
        )

    def _extract_hand_landmarks(
        self,
        result,
        handedness: str
    ) -> np.ndarray:
        """
        Extract hand landmarks for specific hand (Left or Right).

        Args:
            result: Hand detection result from MediaPipe
            handedness: 'Left' or 'Right'

        Returns:
            (21, 3) array of hand landmarks or zeros if not detected
        """
        if not result.hand_landmarks or not result.handedness:
            return np.zeros((21, 3), dtype=np.float32)

        # Find the hand with matching handedness
        for i, hand_handedness in enumerate(result.handedness):
            if hand_handedness[0].category_name == handedness:
                landmarks = result.hand_landmarks[i]
                coords = np.array([
                    [lm.x, lm.y, lm.z] for lm in landmarks
                ], dtype=np.float32)
                return coords

        # Hand not detected
        return np.zeros((21, 3), dtype=np.float32)

    def _extract_face_landmarks(self, result) -> np.ndarray:
        """
        Extract face mesh landmarks.

        Returns:
            (478, 3) array of face landmarks or zeros if not detected
        """
        if not result.face_landmarks:
            return np.zeros((478, 3), dtype=np.float32)

        landmarks = result.face_landmarks[0]  # Take first face
        coords = np.array([
            [lm.x, lm.y, lm.z] for lm in landmarks
        ], dtype=np.float32)

        return coords

    def _extract_pose_landmarks(self, result) -> np.ndarray:
        """
        Extract pose landmarks.

        Returns:
            (33, 3) array of pose landmarks
        """
        if not result.pose_landmarks:
            raise ValueError("Pose landmarks missing - should not happen if filtering worked")

        landmarks = result.pose_landmarks[0]  # Take first pose
        coords = np.array([
            [lm.x, lm.y, lm.z] for lm in landmarks
        ], dtype=np.float32)

        return coords

    def extract_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> list:
        """
        Extract landmarks from entire video file.

        Args:
            video_path: Path to video file
            max_frames: If specified, stop after this many frames

        Returns:
            List of RawLandmarks objects
        """
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

        if success_rate < 0.95:
            print(f"⚠ WARNING: Success rate {success_rate:.1%} below 95% threshold")

        return landmarks_sequence

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()


# ============================================================================
# Testing
# ============================================================================

def test_real_extractor():
    """Test real MediaPipe extraction on a video."""
    from pathlib import Path

    print("\n" + "="*70)
    print("Testing Real MediaPipe Extractor")
    print("="*70 + "\n")

    # Find a test video
    video_base = Path("/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100")

    for gloss_dir in sorted(video_base.iterdir()):
        if gloss_dir.is_dir():
            videos = list(gloss_dir.glob("*.mp4"))
            if videos:
                test_video = videos[0]
                print(f"Testing on: {test_video}")
                print(f"Gloss: {gloss_dir.name}\n")

                # Create extractor
                extractor = MediaPipeExtractor()

                # Extract from first 30 frames
                landmarks_seq = extractor.extract_video(str(test_video), max_frames=30)

                if len(landmarks_seq) > 0:
                    print(f"\n✓ Successfully extracted {len(landmarks_seq)} frames")

                    # Analyze first frame
                    first = landmarks_seq[0]
                    print(f"\nFirst frame analysis:")
                    print(f"  Left hand detected: {not np.all(first.left_hand == 0)}")
                    print(f"  Right hand detected: {not np.all(first.right_hand == 0)}")
                    print(f"  Face detected: {not np.all(first.face == 0)}")
                    print(f"  Pose detected: {not np.all(first.pose == 0)}")

                    # Check landmarks are in valid range [0, 1]
                    all_landmarks = first.concatenate()
                    print(f"\nLandmark statistics:")
                    print(f"  Min: {all_landmarks.min():.4f}")
                    print(f"  Max: {all_landmarks.max():.4f}")
                    print(f"  Mean: {all_landmarks.mean():.4f}")

                    print("\n✓ Real MediaPipe extraction working!")
                else:
                    print("❌ No landmarks extracted")

                break

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_real_extractor()
