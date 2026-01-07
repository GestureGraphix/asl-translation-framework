#!/usr/bin/env python3.11
"""
Simple Phase 1 Evaluation - Detailed Per-Class Results

Reuses the training script's validation code to show per-class accuracy.

Usage:
    python3.11 scripts/eval_phase1.py --checkpoint checkpoints/phase1/best_model.pt
"""

import sys
from pathlib import Path
import torch
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.wlasl_dataset import WLASLDataset, collate_fn
from models.encoder import ASLEncoder
from models.ctc_head import CTCModel, CTCLoss, CTCDecoder
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase 1 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for evaluation')

    args = parser.parse_args()

    print("="*70)
    print("Phase 1 Model Evaluation")
    print("="*70)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Reconstruct model
    encoder = ASLEncoder(checkpoint['encoder_config'])
    model = CTCModel(encoder, checkpoint['ctc_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device).eval()

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train accuracy: {checkpoint['train_accs'][-1]*100:.2f}%")
    print(f"  Val accuracy: {checkpoint['val_accs'][-1]*100:.2f}%")

    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = WLASLDataset(
        data_root='/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100',
        metadata_path='/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json',
        split='val',
        feature_cache_path='data/cache/phase1/val_20glosses.pkl',
        extract_features=False
    )

    # Dataset is already filtered by the cache

    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Vocabulary size: {val_dataset.vocab_size}")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    # Create loss and decoder
    ctc_loss = CTCLoss(checkpoint['ctc_config'])
    decoder = CTCDecoder(checkpoint['ctc_config'])

    # Run validation
    print("\nRunning validation...")
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    class_predictions = {}  # Track what each class is predicted as

    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if batch is None:
                continue

            features = batch['features'].to(args.device)
            lengths = batch['lengths'].to(args.device)
            gloss_ids = batch['gloss_ids'].to(args.device)
            glosses = batch['glosses']

            # Forward pass
            logits, output_lengths = model(features, lengths)

            # CTC loss
            targets = gloss_ids
            target_lengths = torch.ones(len(targets), dtype=torch.long, device=args.device)
            loss = ctc_loss(logits, targets, output_lengths, target_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Decode predictions
            decoded = decoder.greedy_decode(logits, output_lengths)

            # Compute accuracy
            for i, pred_seq in enumerate(decoded):
                true_gloss = glosses[i]
                true_id = gloss_ids[i].item()

                # Initialize tracking
                if true_gloss not in class_total:
                    class_total[true_gloss] = 0
                    class_correct[true_gloss] = 0
                    class_predictions[true_gloss] = []

                class_total[true_gloss] += 1

                # Get prediction
                if len(pred_seq) > 0:
                    pred_id = pred_seq[0]
                    class_predictions[true_gloss].append(pred_id)

                    if pred_id == true_id:
                        correct += 1
                        class_correct[true_gloss] += 1
                else:
                    class_predictions[true_gloss].append(0)  # blank

                total += 1

    # Print results
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY RESULTS")
    print("="*70)

    # Calculate and sort by accuracy
    class_results = []
    for gloss in sorted(class_total.keys()):
        acc = class_correct.get(gloss, 0) / class_total[gloss] if class_total[gloss] > 0 else 0
        class_results.append((gloss, acc, class_correct.get(gloss, 0), class_total[gloss]))

    class_results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Status'}")
    print("-" * 70)

    for rank, (gloss, acc, corr, tot) in enumerate(class_results, 1):
        if acc >= 0.7:
            status = "✓ Excellent"
        elif acc >= 0.5:
            status = "✓ Good"
        elif acc >= 0.3:
            status = "⚠ Medium"
        else:
            status = "✗ Poor"

        print(f"{rank:<6} {gloss:<15} {corr:<10} {tot:<10} {acc*100:>6.1f}%      {status}")

    # Overall
    overall_acc = correct / total if total > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    print("-" * 70)
    print(f"{'':6} {'OVERALL':<15} {correct:<10} {total:<10} {overall_acc*100:>6.1f}%")
    print(f"\nAverage Loss: {avg_loss:.4f}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    accuracies = [r[1] for r in class_results]
    print(f"\nBest class: {class_results[0][0]} ({class_results[0][1]*100:.1f}%)")
    print(f"Worst class: {class_results[-1][0]} ({class_results[-1][1]*100:.1f}%)")
    print(f"Mean accuracy: {sum(accuracies)/len(accuracies)*100:.1f}%")
    print(f"Median accuracy: {sorted(accuracies)[len(accuracies)//2]*100:.1f}%")

    # Count by status
    excellent = sum(1 for r in class_results if r[1] >= 0.7)
    good = sum(1 for r in class_results if 0.5 <= r[1] < 0.7)
    medium = sum(1 for r in class_results if 0.3 <= r[1] < 0.5)
    poor = sum(1 for r in class_results if r[1] < 0.3)

    print(f"\nClasses by performance:")
    print(f"  Excellent (≥70%): {excellent}")
    print(f"  Good (50-70%): {good}")
    print(f"  Medium (30-50%): {medium}")
    print(f"  Poor (<30%): {poor}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if overall_acc >= 0.5:
        print("\n✓ Phase 1 goal achieved (>50%)!")
        print("  Next: Scale to more classes or implement Stage 1 pre-training")
    elif overall_acc >= 0.35:
        print("\n⚠ Close to goal (need >50%)")
        print("  Try:")
        print("  1. Train longer (100 epochs)")
        print("  2. Learning rate scheduler")
        print("  3. Higher dropout (0.6)")
    else:
        print("\n✗ Significantly below goal")
        print("  Recommendations:")
        print("  1. Check feature quality (are features discriminative?)")
        print("  2. Inspect failed classes (confusions?)")
        print("  3. May need Stage 1 pre-training for better features")

    if poor > len(class_results) / 2:
        print("\n⚠ Warning: More than half of classes perform poorly (<30%)")
        print("  This suggests fundamental feature or model issues")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
