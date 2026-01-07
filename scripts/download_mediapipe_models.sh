#!/bin/bash
# Download MediaPipe model files for Tasks API

MODELS_DIR="/home/alex/Documents/asl-translation-framework/models/mediapipe"
mkdir -p "$MODELS_DIR"

echo "Downloading MediaPipe models..."

# Hand Landmarker
if [ ! -f "$MODELS_DIR/hand_landmarker.task" ]; then
    echo "  Downloading hand_landmarker.task..."
    wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
        -O "$MODELS_DIR/hand_landmarker.task"
    echo "  ✓ Hand landmarker downloaded"
fi

# Face Landmarker
if [ ! -f "$MODELS_DIR/face_landmarker.task" ]; then
    echo "  Downloading face_landmarker.task..."
    wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
        -O "$MODELS_DIR/face_landmarker.task"
    echo "  ✓ Face landmarker downloaded"
fi

# Pose Landmarker
if [ ! -f "$MODELS_DIR/pose_landmarker_full.task" ]; then
    echo "  Downloading pose_landmarker_full.task..."
    wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task \
        -O "$MODELS_DIR/pose_landmarker_full.task"
    echo "  ✓ Pose landmarker downloaded"
fi

echo "✓ All models downloaded to: $MODELS_DIR"
ls -lh "$MODELS_DIR"
