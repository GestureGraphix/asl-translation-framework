#!/usr/bin/env python3.11
"""
Test Inference on WLASL Videos

Loads trained model and tests predictions on training/validation samples.

Usage:
    python3.11 scripts/test_inference.py checkpoints/real_mediapipe_cached/best_model.pt
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.wlasl_dataset import WLASLDataset, collate_fn
from training.stage2_ctc import Stage2Trainer


def test_on_dataset(trainer: Stage2Trainer, dataset: WLASLDataset, split_name: str, max_samples: int = 10):
    """
    Test model on dataset and show predictions.

    Args:
        trainer: Trained model
        dataset: Dataset to test on
        split_name: Name of split (for display)
        max_samples: Maximum samples to test
    """
    print(f"\n{'='*70}")
    print(f"Testing on {split_name} set ({len(dataset)} samples)")
    print(f"{'='*70}\n")

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=1,  # Test one at a time for clarity
        shuffle=False,
        collate_fn=collate_fn,
    )

    trainer.model.eval()

    predictions = []
    ground_truth = []
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_samples:
                break

            if batch is None:
                continue

            # Move to device
            features = batch['features'].to(trainer.device)
            lengths = batch['lengths'].to(trainer.device)
            gloss_ids = batch['gloss_ids'].to(trainer.device)
            glosses = batch['glosses']
            video_ids = batch['video_ids']

            # Forward pass
            logits, output_lengths = trainer.model(features, lengths)

            # Decode
            decoded = trainer.decoder.greedy_decode(logits, output_lengths)

            # Get prediction
            if len(decoded[0]) > 0:
                pred_id = decoded[0][0]
                pred_gloss = dataset.id_to_gloss.get(pred_id, f"<UNK:{pred_id}>")
            else:
                pred_id = -1
                pred_gloss = "<BLANK>"

            true_id = gloss_ids[0].item()
            true_gloss = glosses[0]

            predictions.append(pred_id)
            ground_truth.append(true_id)

            is_correct = pred_id == true_id
            if is_correct:
                correct += 1
            total += 1

            # Print sample
            status = "✓" if is_correct else "✗"
            print(f"{status} Sample {i+1} ({video_ids[0]}):")
            print(f"  True:      {true_gloss} (ID: {true_id})")
            print(f"  Predicted: {pred_gloss} (ID: {pred_id})")
            print(f"  Features:  {features.shape}, length={lengths[0].item()}")

            # Show logit statistics
            top5_probs, top5_ids = torch.topk(torch.softmax(logits[0].mean(0), dim=0), k=5)
            print(f"  Top-5 predictions:")
            for j, (prob, idx) in enumerate(zip(top5_probs, top5_ids)):
                gloss = dataset.id_to_gloss.get(idx.item(), f"<UNK:{idx.item()}>")
                print(f"    {j+1}. {gloss} (ID: {idx.item()}, prob: {prob.item():.3f})")
            print()

    accuracy = correct / total if total > 0 else 0.0

    print(f"{'='*70}")
    print(f"{split_name} Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"{'='*70}\n")

    # Analyze prediction patterns
    analyze_predictions(predictions, ground_truth, dataset)

    return accuracy


def analyze_predictions(predictions, ground_truth, dataset):
    """Analyze prediction patterns to understand model behavior."""
    print(f"\n{'='*70}")
    print("Prediction Analysis")
    print(f"{'='*70}\n")

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Count unique predictions
    unique_preds = set(predictions)
    print(f"Unique predictions: {len(unique_preds)} out of {dataset.vocab_size}")

    # Most common predictions
    from collections import Counter
    pred_counts = Counter(predictions)
    print(f"\nTop 5 predicted glosses:")
    for pred_id, count in pred_counts.most_common(5):
        gloss = dataset.id_to_gloss.get(pred_id, f"<UNK:{pred_id}>")
        print(f"  {gloss} (ID: {pred_id}): {count} times")

    # Check if always predicting same thing
    if len(unique_preds) == 1:
        pred_id = list(unique_preds)[0]
        gloss = dataset.id_to_gloss.get(pred_id, f"<UNK:{pred_id}>")
        print(f"\n⚠ WARNING: Model always predicts '{gloss}' (ID: {pred_id})")

    # Check if predicting blank (ID 0)
    blank_count = np.sum(predictions == 0)
    if blank_count > 0:
        print(f"\n⚠ Model predicted blank {blank_count}/{len(predictions)} times")

    # Check for random vs systematic errors
    if len(predictions) > 1:
        # Are predictions close to ground truth IDs?
        id_diff = np.abs(predictions - ground_truth)
        avg_diff = np.mean(id_diff)
        print(f"\nAverage ID difference: {avg_diff:.1f}")
        if avg_diff < 10:
            print("  → Predictions are close to correct IDs (systematic bias?)")
        else:
            print("  → Predictions are far from correct IDs (more random)")

    print(f"{'='*70}\n")


def main():
    """Main inference testing."""
    import argparse

    parser = argparse.ArgumentParser(description='Test inference on WLASL videos')
    parser.add_argument('checkpoint', type=str,
                       help='Path to checkpoint file')
    parser.add_argument('--data-root', type=str,
                       default='/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100',
                       help='Root directory for videos')
    parser.add_argument('--metadata', type=str,
                       default='/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json',
                       help='Path to metadata.json')
    parser.add_argument('--train-cache', type=str,
                       default='data/features/train_features_real.pkl',
                       help='Training feature cache')
    parser.add_argument('--val-cache', type=str,
                       default='data/features/val_features_real.pkl',
                       help='Validation feature cache')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Max samples to test per split')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Inference Testing")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model from checkpoint...")
    trainer = Stage2Trainer.load_checkpoint(args.checkpoint, device=args.device)

    # Create datasets
    print("\nLoading training dataset...")
    train_dataset = WLASLDataset(
        data_root=args.data_root,
        metadata_path=args.metadata,
        split='train',
        max_samples=20,  # Use same as training
        feature_cache_path=args.train_cache,
        extract_features=True,
    )

    print("\nLoading validation dataset...")
    val_dataset = WLASLDataset(
        data_root=args.data_root,
        metadata_path=args.metadata,
        split='val',
        max_samples=10,  # Use same as training
        feature_cache_path=args.val_cache,
        extract_features=True,
    )

    # Test on training set
    train_acc = test_on_dataset(trainer, train_dataset, "Training", args.max_samples)

    # Test on validation set
    val_acc = test_on_dataset(trainer, val_dataset, "Validation", args.max_samples)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Training accuracy:   {train_acc:.2%}")
    print(f"Validation accuracy: {val_acc:.2%}")
    print(f"{'='*70}\n")

    # Diagnosis
    if train_acc > 0.5 and val_acc < 0.1:
        print("Diagnosis: OVERFITTING - model memorized training data")
        print("  Recommendations:")
        print("  1. Use more training data (currently only 20 samples!)")
        print("  2. Reduce model size (1.18M params is too large)")
        print("  3. Add regularization (dropout, weight decay)")
        print("  4. Early stopping (stop at epoch 3)")
    elif train_acc < 0.1 and val_acc < 0.1:
        print("Diagnosis: UNDERFITTING - model not learning")
        print("  Recommendations:")
        print("  1. Check feature extraction (are features meaningful?)")
        print("  2. Verify labels are correct")
        print("  3. Try simpler model first")
        print("  4. Increase learning rate or train longer")
    elif train_acc > 0.5 and val_acc > 0.5:
        print("Diagnosis: Model is working! ✓")
    else:
        print("Diagnosis: Unclear - investigate further")

    print()


if __name__ == "__main__":
    main()
