#!/usr/bin/env python3.11
"""
Detailed Phase 1 Analysis - Per-Class Accuracy and Confusion Matrix

Analyzes what's failing in Phase 1 training:
- Per-class accuracy
- Confusion matrix
- Feature quality analysis
- Most confused signs

Usage:
    python3.11 scripts/analyze_phase1_detailed.py \
        --checkpoint checkpoints/phase1/best_model.pt \
        --train-cache data/cache/phase1/features_train_phase1.pkl \
        --val-cache data/cache/phase1/features_val_phase1.pkl
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.encoder import ASLEncoder
from models.ctc_head import CTCModel


def load_checkpoint_and_model(checkpoint_path: str, device: str = 'cpu'):
    """Load checkpoint and reconstruct model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model - CTCModel wraps both encoder and ctc_head
    encoder = ASLEncoder(checkpoint['encoder_config'])
    model = CTCModel(encoder, checkpoint['ctc_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # Extract encoder and ctc_head for separate analysis
    encoder = model.encoder
    ctc_head = model.ctc_head

    return encoder, ctc_head, checkpoint


def load_cached_data(cache_path: str) -> Dict:
    """Load cached features."""
    print(f"Loading cached features: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    print(f"  Loaded {len(cache)} cached samples")
    return cache


def get_predictions(encoder, ctc_head, features: torch.Tensor, device: str) -> torch.Tensor:
    """Get model predictions for features."""
    with torch.no_grad():
        features = features.to(device)
        # Add batch dimension
        features = features.unsqueeze(0)  # [1, T, D]
        lengths = torch.tensor([features.size(1)], dtype=torch.long, device=device)

        # Forward pass
        encoded = encoder(features, lengths)  # [1, T, hidden]
        logits = ctc_head(encoded)  # [1, T, vocab_size]

        # Greedy decode
        log_probs = torch.log_softmax(logits, dim=-1)
        predictions = log_probs.argmax(dim=-1)  # [1, T]

        # CTC decode (collapse blanks and repeats)
        pred_seq = predictions[0].cpu().numpy()
        blank_id = 0
        decoded = []
        prev = blank_id
        for p in pred_seq:
            if p != blank_id and p != prev:
                decoded.append(p)
            prev = p

        return decoded, log_probs[0].cpu()  # Return decoded sequence and log probs


def analyze_per_class(encoder, ctc_head, val_cache: Dict, gloss2idx: Dict, device: str):
    """Analyze per-class accuracy."""
    print("\n" + "="*70)
    print("Per-Class Accuracy Analysis")
    print("="*70 + "\n")

    idx2gloss = {v: k for k, v in gloss2idx.items()}

    # Track predictions per class
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_predictions = defaultdict(list)  # Store what each class was predicted as

    all_true = []
    all_pred = []

    for video_id, features in val_cache.items():
        # Get true label from video_id (format: "video_id/gloss")
        if '/' in video_id:
            gloss = video_id.split('/')[-1]
        else:
            # Try to extract gloss from filename
            continue

        if gloss not in gloss2idx:
            continue

        true_idx = gloss2idx[gloss]

        # Get prediction
        features_tensor = torch.tensor(features, dtype=torch.float32)
        decoded, log_probs = get_predictions(encoder, ctc_head, features_tensor, device)

        # Get most likely prediction (first decoded symbol or most probable)
        if len(decoded) > 0:
            pred_idx = decoded[0]
        else:
            # If nothing decoded, take most probable non-blank
            probs = log_probs.exp().mean(dim=0)  # Average over time
            probs[0] = 0  # Zero out blank
            pred_idx = probs.argmax().item()

        all_true.append(true_idx)
        all_pred.append(pred_idx)

        class_total[gloss] += 1
        class_predictions[gloss].append(pred_idx)

        if pred_idx == true_idx:
            class_correct[gloss] += 1

    # Print per-class results
    print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Status'}")
    print("-" * 60)

    class_accuracies = []
    for gloss in sorted(class_total.keys()):
        correct = class_correct[gloss]
        total = class_total[gloss]
        acc = correct / total if total > 0 else 0
        class_accuracies.append((gloss, acc, correct, total))

        status = ""
        if acc >= 0.7:
            status = "✓ Good"
        elif acc >= 0.4:
            status = "⚠ Medium"
        else:
            status = "✗ Poor"

        print(f"{gloss:<15} {correct:<10} {total:<10} {acc*100:>6.1f}%    {status}")

    # Overall accuracy
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0

    print("-" * 60)
    print(f"{'OVERALL':<15} {total_correct:<10} {total_samples:<10} {overall_acc*100:>6.1f}%")

    # Best and worst classes
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    print(f"\n✓ Best performing classes:")
    for gloss, acc, correct, total in class_accuracies[:5]:
        print(f"   {gloss}: {acc*100:.1f}% ({correct}/{total})")

    print(f"\n✗ Worst performing classes:")
    for gloss, acc, correct, total in class_accuracies[-5:]:
        print(f"   {gloss}: {acc*100:.1f}% ({correct}/{total})")

    return all_true, all_pred, idx2gloss


def plot_confusion_matrix(all_true: List[int], all_pred: List[int],
                         idx2gloss: Dict, output_path: str):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix

    # Get unique classes
    classes = sorted(set(all_true + all_pred))
    class_names = [idx2gloss.get(i, f"ID_{i}") for i in classes]

    # Compute confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=classes)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix (Normalized by True Label)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {output_path}")

    # Find most confused pairs
    print(f"\nMost Confused Sign Pairs:")
    print(f"{'True':<15} {'Predicted As':<15} {'Frequency':<10}")
    print("-" * 45)

    confusion_pairs = []
    for i, true_idx in enumerate(classes):
        for j, pred_idx in enumerate(classes):
            if i != j and cm_normalized[i, j] > 0.1:  # More than 10% confusion
                confusion_pairs.append((
                    idx2gloss.get(true_idx, f"ID_{true_idx}"),
                    idx2gloss.get(pred_idx, f"ID_{pred_idx}"),
                    cm_normalized[i, j]
                ))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_gloss, pred_gloss, freq in confusion_pairs[:10]:
        print(f"{true_gloss:<15} {pred_gloss:<15} {freq*100:>6.1f}%")


def analyze_feature_quality(train_cache: Dict, val_cache: Dict):
    """Analyze feature quality - are features actually different between classes?"""
    print("\n" + "="*70)
    print("Feature Quality Analysis")
    print("="*70 + "\n")

    # Group features by gloss
    gloss_features = defaultdict(list)

    for video_id, features in {**train_cache, **val_cache}.items():
        if '/' in video_id:
            gloss = video_id.split('/')[-1]
            gloss_features[gloss].append(features)

    # Compute statistics per class
    print(f"{'Class':<15} {'Samples':<10} {'Mean Norm':<15} {'Std Norm':<15}")
    print("-" * 60)

    class_stats = []
    for gloss in sorted(gloss_features.keys()):
        features_list = gloss_features[gloss]
        # Stack all features for this gloss
        all_feats = np.vstack(features_list)  # [total_frames, 36]

        # Compute statistics
        mean_norm = np.mean(np.linalg.norm(all_feats, axis=1))
        std_norm = np.std(np.linalg.norm(all_feats, axis=1))

        class_stats.append((gloss, len(features_list), mean_norm, std_norm))
        print(f"{gloss:<15} {len(features_list):<10} {mean_norm:<15.3f} {std_norm:<15.3f}")

    # Check if features are distinguishable
    mean_norms = [s[2] for s in class_stats]
    std_norms = [s[3] for s in class_stats]

    print(f"\nOverall Statistics:")
    print(f"  Mean norm range: [{min(mean_norms):.3f}, {max(mean_norms):.3f}]")
    print(f"  Std norm range: [{min(std_norms):.3f}, {max(std_norms):.3f}]")

    # If all features have very similar norms, they might not be distinguishable
    norm_variation = (max(mean_norms) - min(mean_norms)) / np.mean(mean_norms)
    print(f"  Norm variation: {norm_variation*100:.1f}%")

    if norm_variation < 0.1:
        print("\n⚠ Warning: Features have very similar norms across classes!")
        print("  This suggests features may not be discriminative enough.")
    else:
        print("\n✓ Features show reasonable variation across classes.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Detailed Phase 1 analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--train-cache', type=str, required=True,
                       help='Path to training cache file')
    parser.add_argument('--val-cache', type=str, required=True,
                       help='Path to validation cache file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--output-dir', type=str, default='checkpoints/phase1',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Load model
    encoder, ctc_head, checkpoint = load_checkpoint_and_model(args.checkpoint, args.device)

    # Load cached data
    train_cache = load_cached_data(args.train_cache)
    val_cache = load_cached_data(args.val_cache)

    # Get vocabulary from checkpoint
    vocab_size = checkpoint['ctc_config'].vocab_size
    print(f"\nVocabulary size: {vocab_size}")

    # Build gloss2idx from dataset
    # For now, extract from cache keys
    glosses = set()
    for video_id in {**train_cache, **val_cache}.keys():
        if '/' in video_id:
            gloss = video_id.split('/')[-1]
            glosses.add(gloss)

    glosses = sorted(glosses)
    gloss2idx = {gloss: idx + 1 for idx, gloss in enumerate(glosses)}  # +1 because 0 is blank
    print(f"Found {len(glosses)} unique glosses: {glosses[:10]}...")

    # Analyze feature quality
    analyze_feature_quality(train_cache, val_cache)

    # Analyze per-class accuracy
    all_true, all_pred, idx2gloss = analyze_per_class(
        encoder, ctc_head, val_cache, gloss2idx, args.device
    )

    # Plot confusion matrix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(all_true, all_pred, idx2gloss, str(confusion_path))

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
