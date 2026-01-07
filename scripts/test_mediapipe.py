#!/usr/bin/env python3.11
"""Test MediaPipe extraction on a few WLASL videos"""

import sys
sys.path.insert(0, 'src/phonology')

from pathlib import Path
from mediapipe_extractor import MediaPipeExtractor

# Get first 3 videos
VIDEO_DIR = Path('data/raw/wlasl/videos')
videos = list(VIDEO_DIR.glob('*/*.mp4'))[:3]

print(f"Testing MediaPipe on {len(videos)} videos...\n")

extractor = MediaPipeExtractor(model_complexity=1)

for video_path in videos:
    gloss = video_path.parent.name
    video_id = video_path.stem
    
    print(f"Processing: {gloss}/{video_id}.mp4")
    print(f"  Path: {video_path}")
    
    try:
        # Extract first 30 frames for testing
        landmarks_seq = extractor.extract_video(str(video_path), max_frames=30)
        
        if len(landmarks_seq) > 0:
            print(f"  ✓ Success! Extracted {len(landmarks_seq)} frames")
            
            # Show first frame info
            first = landmarks_seq[0]
            print(f"    Timestamp: {first.timestamp:.3f}s")
            print(f"    Left hand: {first.left_hand.shape}")
            print(f"    Right hand: {first.right_hand.shape}")
            print(f"    Face: {first.face.shape}")
            print(f"    Pose: {first.pose.shape}")
        else:
            print(f"  ✗ No landmarks extracted")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

print("Test complete!")
