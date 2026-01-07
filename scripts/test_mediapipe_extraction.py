#!/usr/bin/env python3.11
"""
Test script for MediaPipe landmark extraction on WLASL videos.

Usage:
    python3.11 scripts/test_mediapipe_extraction.py

This script will:
1. Load a sample video from the WLASL dataset
2. Extract landmarks using MediaPipe
3. Print statistics and verify the extraction pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phonology.mediapipe_extractor_v2 import MediaPipeExtractor
import numpy as np


def test_single_video(video_path: str, max_frames: int = 30):
    """
    Test MediaPipe extraction on a single video.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process (for quick testing)
    """
    print(f"\n{'='*70}")
    print(f"Testing MediaPipe Extraction")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Max frames: {max_frames}")
    print(f"{'='*70}\n")

    # Initialize extractor
    extractor = MediaPipeExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,  # Full model
    )

    # Extract landmarks
    print("Extracting landmarks...")
    landmarks_sequence = extractor.extract_video(video_path, max_frames=max_frames)

    if len(landmarks_sequence) == 0:
        print("❌ ERROR: No landmarks extracted!")
        return False

    print(f"\n✓ Successfully extracted {len(landmarks_sequence)} frames")

    # Analyze first frame
    print("\n" + "="*70)
    print("First Frame Analysis")
    print("="*70)

    first_frame = landmarks_sequence[0]
    print(f"Timestamp: {first_frame.timestamp:.3f}s")
    print(f"\nLandmark shapes:")
    print(f"  Left hand:  {first_frame.left_hand.shape}")
    print(f"  Right hand: {first_frame.right_hand.shape}")
    print(f"  Face:       {first_frame.face.shape}")
    print(f"  Pose:       {first_frame.pose.shape}")
    print(f"  Concatenated: {first_frame.concatenate().shape}")

    # Check for missing landmarks
    print(f"\nLandmark detection status:")
    left_detected = not np.all(first_frame.left_hand == 0)
    right_detected = not np.all(first_frame.right_hand == 0)
    face_detected = not np.all(first_frame.face == 0)

    print(f"  Left hand:  {'✓ Detected' if left_detected else '✗ Not detected'}")
    print(f"  Right hand: {'✓ Detected' if right_detected else '✗ Not detected'}")
    print(f"  Face:       {'✓ Detected' if face_detected else '✗ Not detected'}")
    print(f"  Pose:       ✓ Always present (required)")

    # Statistics across all frames
    print(f"\n{'='*70}")
    print("Sequence Statistics")
    print(f"{'='*70}")

    left_hand_frames = sum(1 for lm in landmarks_sequence if not np.all(lm.left_hand == 0))
    right_hand_frames = sum(1 for lm in landmarks_sequence if not np.all(lm.right_hand == 0))
    face_frames = sum(1 for lm in landmarks_sequence if not np.all(lm.face == 0))

    total_frames = len(landmarks_sequence)

    print(f"Left hand detected:  {left_hand_frames}/{total_frames} frames ({100*left_hand_frames/total_frames:.1f}%)")
    print(f"Right hand detected: {right_hand_frames}/{total_frames} frames ({100*right_hand_frames/total_frames:.1f}%)")
    print(f"Face detected:       {face_frames}/{total_frames} frames ({100*face_frames/total_frames:.1f}%)")

    # Validate Section 1.2 assumption (>95% success rate already validated by extract_video)
    print(f"\n{'='*70}")
    print("Validation")
    print(f"{'='*70}")

    # Check that at least one hand is visible in most frames (ASL requirement)
    hand_visible_frames = sum(1 for lm in landmarks_sequence
                              if not np.all(lm.left_hand == 0) or not np.all(lm.right_hand == 0))
    hand_visibility = hand_visible_frames / total_frames

    print(f"Frames with ≥1 hand visible: {hand_visible_frames}/{total_frames} ({100*hand_visibility:.1f}%)")

    if hand_visibility >= 0.95:
        print("✓ PASS: Hand visibility meets 95% threshold")
    else:
        print(f"⚠ WARNING: Hand visibility {100*hand_visibility:.1f}% below 95% threshold")

    # Sample some landmark values
    print(f"\n{'='*70}")
    print("Sample Landmark Values (First Frame)")
    print(f"{'='*70}")

    if left_detected:
        print(f"\nLeft wrist (landmark 0): {first_frame.left_hand[0]}")
        print(f"Left index tip (landmark 8): {first_frame.left_hand[8]}")

    if right_detected:
        print(f"\nRight wrist (landmark 0): {first_frame.right_hand[0]}")
        print(f"Right index tip (landmark 8): {first_frame.right_hand[8]}")

    # Pose landmarks (shoulders for normalization)
    print(f"\nLeft shoulder (landmark 11): {first_frame.pose[11]}")
    print(f"Right shoulder (landmark 12): {first_frame.pose[12]}")

    print(f"\n{'='*70}")
    print("✓ Extraction Test Complete!")
    print(f"{'='*70}\n")

    return True


def main():
    """Run extraction tests on sample WLASL videos."""

    # Find a sample video
    video_base = Path("/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100")

    # Try to find the first available video
    sample_videos = []
    for gloss_dir in sorted(video_base.iterdir()):
        if gloss_dir.is_dir():
            videos = list(gloss_dir.glob("*.mp4"))
            if videos:
                sample_videos.append((gloss_dir.name, videos[0]))
                if len(sample_videos) >= 3:  # Get 3 samples
                    break

    if not sample_videos:
        print("❌ ERROR: No videos found in", video_base)
        return 1

    print(f"\nFound {len(sample_videos)} sample videos to test:")
    for gloss, path in sample_videos:
        print(f"  - {gloss}: {path.name}")

    # Test first video
    gloss, video_path = sample_videos[0]
    print(f"\n\nTesting video for gloss: '{gloss}'")

    success = test_single_video(str(video_path), max_frames=30)

    if success:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. ✓ MediaPipe extraction working!")
        print("2. → Next: Implement feature extraction (features.py)")
        print("3. → Then: Implement product VQ quantizer (quantizer.py)")
        print("="*70 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
