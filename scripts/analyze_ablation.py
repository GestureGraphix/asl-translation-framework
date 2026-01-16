#!/usr/bin/env python3.11
"""
Analyze ablation study results.

Compares baseline (random init) vs pre-trained (Stage 1 init) experiments.
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys
from scipy import stats
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "checkpoints/ablation"
RESULTS_FILE = RESULTS_DIR / "ablation_results.json"

def extract_metrics_from_checkpoint(checkpoint_path: Path) -> dict:
    """Extract validation metrics from checkpoint."""
    if not checkpoint_path.exists():
        return {'val_loss': None, 'val_acc': None}

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        val_loss = checkpoint.get('val_loss', None)

        # Try to get validation losses list
        val_losses = checkpoint.get('val_losses', [])
        best_val_loss = min(val_losses) if val_losses else val_loss

        return {
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'epoch': checkpoint.get('epoch', None),
        }
    except Exception as e:
        print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
        return {'val_loss': None, 'val_acc': None}

def read_accuracy_from_logs(checkpoint_dir: Path) -> float:
    """
    Try to extract validation accuracy from training logs or checkpoint.

    For now, we'll manually extract from the best_model.pt checkpoint.
    The validation accuracy is printed during training but not saved to checkpoint.
    We'll need to re-run validation or extract from logs.
    """
    # Check if there's a results file
    results_file = checkpoint_dir / "results.txt"
    if results_file.exists():
        with open(results_file, 'r') as f:
            for line in f:
                if 'Val Acc:' in line:
                    # Parse line like "  Val Acc:    26.09% (12/46)"
                    acc_str = line.split('Val Acc:')[1].strip().split('%')[0]
                    return float(acc_str)

    # If no results file, we need to compute accuracy manually
    # This requires loading the model and running validation
    return None

def compute_validation_accuracy(checkpoint_dir: Path, val_cache_path: Path) -> float:
    """
    Load checkpoint and compute validation accuracy.

    Args:
        checkpoint_dir: Directory containing checkpoint
        val_cache_path: Path to validation cache

    Returns:
        Validation accuracy as percentage
    """
    best_checkpoint = checkpoint_dir / "best_model.pt"
    if not best_checkpoint.exists():
        return None

    try:
        # Import necessary modules
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from training.stage2_ctc import Stage2Trainer
        from models.encoder import EncoderConfig
        from models.ctc_head import CTCConfig
        from utils.wlasl_dataset import WLASLDataset, collate_fn
        from torch.utils.data import DataLoader

        # Load checkpoint
        checkpoint = torch.load(best_checkpoint, map_location='cpu', weights_only=False)

        # Get configs
        encoder_config = checkpoint['encoder_config']
        ctc_config = checkpoint['ctc_config']

        # Create trainer
        trainer = Stage2Trainer(
            encoder_config=encoder_config,
            ctc_config=ctc_config,
            device='cpu',
        )

        # Load model weights
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()

        # Load validation data
        val_dataset = WLASLDataset(
            data_root=None,
            metadata_path=None,
            split='val',
            feature_cache_path=str(val_cache_path),
            extract_features=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Run validation
        metrics = trainer.validate(val_loader, epoch=0)

        return metrics['accuracy'] * 100  # Convert to percentage

    except Exception as e:
        print(f"Error computing validation accuracy for {checkpoint_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Analyze ablation study results."""

    if not RESULTS_FILE.exists():
        print(f"❌ Error: Results file not found at {RESULTS_FILE}")
        print(f"Run experiments first: python3.11 scripts/run_ablation_study.py")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY ANALYSIS")
    print(f"{'='*70}\n")

    # Load results
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    # Separate baseline and pre-trained
    baseline_results = [r for r in results if not r['use_pretrained']]
    pretrained_results = [r for r in results if r['use_pretrained']]

    print(f"Loaded {len(results)} experiments:")
    print(f"  Baseline (random init): {len(baseline_results)}")
    print(f"  Pre-trained (Stage 1): {len(pretrained_results)}")

    # Compute validation accuracies
    val_cache = PROJECT_ROOT / "data/cache/phase1/val_20glosses.pkl"

    print(f"\nComputing validation accuracies...")
    print(f"(This may take a few minutes...)\n")

    baseline_accs = []
    for r in baseline_results:
        print(f"  Baseline seed {r['seed']}...", end=' ', flush=True)
        checkpoint_dir = Path(r['checkpoint_dir'])
        acc = compute_validation_accuracy(checkpoint_dir, val_cache)
        if acc is not None:
            baseline_accs.append(acc)
            print(f"{acc:.2f}%")
        else:
            print("Failed")

    pretrained_accs = []
    for r in pretrained_results:
        print(f"  Pre-trained seed {r['seed']}...", end=' ', flush=True)
        checkpoint_dir = Path(r['checkpoint_dir'])
        acc = compute_validation_accuracy(checkpoint_dir, val_cache)
        if acc is not None:
            pretrained_accs.append(acc)
            print(f"{acc:.2f}%")
        else:
            print("Failed")

    if len(baseline_accs) == 0 or len(pretrained_accs) == 0:
        print("\n❌ Error: Could not compute accuracies for all experiments")
        sys.exit(1)

    # Compute statistics
    baseline_mean = np.mean(baseline_accs)
    baseline_std = np.std(baseline_accs, ddof=1)

    pretrained_mean = np.mean(pretrained_accs)
    pretrained_std = np.std(pretrained_accs, ddof=1)

    improvement_abs = pretrained_mean - baseline_mean
    improvement_rel = (improvement_abs / baseline_mean) * 100

    # Statistical test
    t_stat, p_value = stats.ttest_ind(baseline_accs, pretrained_accs)

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}\n")

    print(f"Baseline (Random Initialization):")
    print(f"  Accuracies: {', '.join(f'{acc:.2f}%' for acc in baseline_accs)}")
    print(f"  Mean: {baseline_mean:.2f}%")
    print(f"  Std Dev: {baseline_std:.2f}%")
    print()

    print(f"Pre-trained (Stage 1 Initialization):")
    print(f"  Accuracies: {', '.join(f'{acc:.2f}%' for acc in pretrained_accs)}")
    print(f"  Mean: {pretrained_mean:.2f}%")
    print(f"  Std Dev: {pretrained_std:.2f}%")
    print()

    print(f"Improvement:")
    print(f"  Absolute: {improvement_abs:+.2f}% ({baseline_mean:.2f}% → {pretrained_mean:.2f}%)")
    print(f"  Relative: {improvement_rel:+.1f}%")
    print()

    print(f"Statistical Significance:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}", end=' ')

    if p_value < 0.001:
        print("***  (highly significant)")
        sig_text = "p < 0.001 (highly significant)"
    elif p_value < 0.01:
        print("**   (very significant)")
        sig_text = "p < 0.01 (very significant)"
    elif p_value < 0.05:
        print("*    (significant)")
        sig_text = "p < 0.05 (significant)"
    else:
        print("n.s. (not significant)")
        sig_text = "p ≥ 0.05 (not significant)"

    print()

    # Interpretation
    print(f"{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}\n")

    if p_value < 0.05:
        print(f"✅ Pre-training provides statistically significant improvement ({sig_text})")
        print(f"   Self-supervised Stage 1 pre-training improves Stage 2 CTC accuracy")
        print(f"   by {improvement_abs:.2f} percentage points ({improvement_rel:+.1f}% relative).")
        print()
        print(f"   Conclusion: Paper's 3-stage curriculum validated ✓")
        print(f"   Recommendation: Always use Stage 1 pre-training before Stage 2.")
    else:
        print(f"⚠️  No statistically significant improvement ({sig_text})")
        print(f"   Pre-training may not be beneficial for this task/data size.")
        print()
        print(f"   Possible reasons:")
        print(f"   - Small dataset (20 signs, 243 samples) doesn't benefit from pre-training")
        print(f"   - Random variation dominates signal")
        print(f"   - Pre-training hyperparameters need tuning")

    # Create visualization
    print(f"\n{'='*70}")
    print(f"Creating visualization...")
    print(f"{'='*70}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot with error bars
    conditions = ['Baseline\n(Random)', 'Pre-trained\n(Stage 1)']
    means = [baseline_mean, pretrained_mean]
    stds = [baseline_std, pretrained_std]

    bars = ax1.bar(conditions, means, yerr=stds, capsize=10, alpha=0.7,
                   color=['#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Ablation Study: Pre-training Benefit')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}% ± {std:.2f}%',
                ha='center', va='bottom', fontsize=10)

    # Add significance indicator
    if p_value < 0.05:
        y_max = max(means) + max(stds) + 2
        ax1.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        ax1.text(0.5, y_max + 0.5, '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*',
                ha='center', va='bottom', fontsize=14)

    # Individual data points
    for i, (cond, accs) in enumerate([(conditions[0], baseline_accs),
                                       (conditions[1], pretrained_accs)]):
        x = np.random.normal(i, 0.04, size=len(accs))
        ax2.scatter(x, accs, alpha=0.6, s=100)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(conditions)
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Individual Experiment Results')
    ax2.grid(axis='y', alpha=0.3)

    # Add mean lines
    ax2.axhline(baseline_mean, xmin=0, xmax=0.4, color='#ff7f0e',
                linestyle='--', linewidth=2, label=f'Baseline mean: {baseline_mean:.2f}%')
    ax2.axhline(pretrained_mean, xmin=0.6, xmax=1.0, color='#2ca02c',
                linestyle='--', linewidth=2, label=f'Pre-trained mean: {pretrained_mean:.2f}%')
    ax2.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save plot
    plot_path = RESULTS_DIR / "ablation_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_path}")

    # Save summary report
    report_path = RESULTS_DIR / "ablation_report.txt"
    with open(report_path, 'w') as f:
        f.write("ABLATION STUDY: Pre-training Benefit Validation\n")
        f.write("="*70 + "\n\n")

        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")

        f.write(f"Baseline (Random Initialization):\n")
        f.write(f"  Accuracies: {', '.join(f'{acc:.2f}%' for acc in baseline_accs)}\n")
        f.write(f"  Mean: {baseline_mean:.2f}%\n")
        f.write(f"  Std Dev: {baseline_std:.2f}%\n\n")

        f.write(f"Pre-trained (Stage 1 Initialization):\n")
        f.write(f"  Accuracies: {', '.join(f'{acc:.2f}%' for acc in pretrained_accs)}\n")
        f.write(f"  Mean: {pretrained_mean:.2f}%\n")
        f.write(f"  Std Dev: {pretrained_std:.2f}%\n\n")

        f.write(f"Improvement:\n")
        f.write(f"  Absolute: {improvement_abs:+.2f}% ({baseline_mean:.2f}% → {pretrained_mean:.2f}%)\n")
        f.write(f"  Relative: {improvement_rel:+.1f}%\n\n")

        f.write(f"Statistical Significance:\n")
        f.write(f"  t-statistic: {t_stat:.3f}\n")
        f.write(f"  p-value: {p_value:.4f} ({sig_text})\n\n")

        f.write("CONCLUSION\n")
        f.write("-"*70 + "\n\n")

        if p_value < 0.05:
            f.write(f"✅ Pre-training provides statistically significant improvement.\n\n")
            f.write(f"Self-supervised Stage 1 pre-training improves Stage 2 CTC accuracy\n")
            f.write(f"by {improvement_abs:.2f} percentage points ({improvement_rel:+.1f}% relative).\n\n")
            f.write(f"Paper's 3-stage curriculum validated ✓\n")
            f.write(f"Recommendation: Always use Stage 1 pre-training before Stage 2.\n")
        else:
            f.write(f"⚠️  No statistically significant improvement.\n\n")
            f.write(f"Pre-training may not be beneficial for this task/data size.\n")

    print(f"✓ Saved report to: {report_path}")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}\n")

    print(f"Summary:")
    print(f"  • Baseline: {baseline_mean:.2f}% ± {baseline_std:.2f}%")
    print(f"  • Pre-trained: {pretrained_mean:.2f}% ± {pretrained_std:.2f}%")
    print(f"  • Improvement: {improvement_abs:+.2f}% ({improvement_rel:+.1f}% relative)")
    print(f"  • Significance: {sig_text}")
    print()

if __name__ == "__main__":
    main()
