#!/bin/bash
#
# Full Training Experiment
# Trains ASL recognition model on WLASL dataset
#

set -e  # Exit on error

echo "=============================================================================="
echo "ASL Recognition - Full Training Experiment"
echo "=============================================================================="
echo ""

# Configuration
DATA_ROOT="/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100"
METADATA="/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json"
CHECKPOINT_DIR="/home/alex/Documents/asl-translation-framework/checkpoints/stage2_full"
MAX_SAMPLES=200  # Use 200 samples (subset of available data for faster iteration)
BATCH_SIZE=4
NUM_EPOCHS=30
LEARNING_RATE=0.001
HIDDEN_DIM=128
NUM_LAYERS=2
DEVICE="cpu"  # Change to "cuda" if GPU available

echo "Configuration:"
echo "  Data root: $DATA_ROOT"
echo "  Max samples: $MAX_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Num layers: $NUM_LAYERS"
echo "  Device: $DEVICE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo ""
echo "=============================================================================="
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Run training
cd /home/alex/Documents/asl-translation-framework

python3.11 src/training/stage2_ctc.py \
    --data-root "$DATA_ROOT" \
    --metadata "$METADATA" \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --hidden-dim $HIDDEN_DIM \
    --num-layers $NUM_LAYERS \
    --max-samples $MAX_SAMPLES \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --num-workers 0

echo ""
echo "=============================================================================="
echo "Training Complete!"
echo "=============================================================================="
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Best model: $CHECKPOINT_DIR/best_model.pt"
echo ""
