#!/usr/bin/env python3.11
"""
Parse ablation study training log to extract best validation accuracy.
"""

import re
from pathlib import Path

LOG_FILE = Path("/tmp/claude/-home-alex-Documents-asl-translation-framework-src-utils/tasks/b1592b3.output")

def parse_log():
    """Parse training log and extract best val accuracy for each experiment."""

    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()

    # Find "Training Complete!" markers to split experiments
    complete_markers = []
    for i, line in enumerate(lines):
        if "Training Complete!" in line:
            complete_markers.append(i)

    if len(complete_markers) != 6:
        print(f"Warning: Found {len(complete_markers)} experiments, expected 6")

    # Define experiment names based on the order we know they ran
    exp_names = [
        "Baseline seed 42",
        "Baseline seed 43",
        "Baseline seed 44",
        "Pre-trained seed 42",
        "Pre-trained seed 43",
        "Pre-trained seed 44",
    ]

    # Split log into sections
    experiments = []
    start_idx = 0

    for i, end_idx in enumerate(complete_markers):
        exp_text = ''.join(lines[start_idx:end_idx+1])
        experiments.append((exp_names[i], exp_text))
        start_idx = end_idx + 1

    # Extract best validation accuracy from each experiment
    results = []

    for name, log_text in experiments:
        # Find all validation accuracy lines
        acc_matches = re.findall(r'Val Acc:\s+([\d.]+)%\s+\((\d+)/(\d+)\)', log_text)

        if acc_matches:
            # Convert to floats and find maximum
            accuracies = [float(acc) for acc, correct, total in acc_matches]
            best_acc = max(accuracies)
            best_idx = accuracies.index(best_acc)
            correct, total = acc_matches[best_idx][1], acc_matches[best_idx][2]

            results.append({
                'name': name,
                'best_acc': best_acc,
                'correct': int(correct),
                'total': int(total)
            })
        else:
            results.append({
                'name': name,
                'best_acc': None,
                'correct': None,
                'total': None
            })

    return results

if __name__ == "__main__":
    results = parse_log()

    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}\n")

    baseline_accs = []
    pretrained_accs = []

    for result in results:
        name = result['name']
        acc = result['best_acc']
        correct = result['correct']
        total = result['total']

        if acc is not None:
            print(f"{name:25s}: {acc:5.2f}% ({correct}/{total})")

            if 'Baseline' in name:
                baseline_accs.append(acc)
            else:
                pretrained_accs.append(acc)
        else:
            print(f"{name:25s}: No data")

    if baseline_accs and pretrained_accs:
        import numpy as np
        from scipy import stats

        print(f"\n{'='*70}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*70}\n")

        print(f"Baseline (Random Init):")
        print(f"  Accuracies: {baseline_accs}")
        print(f"  Mean: {np.mean(baseline_accs):.2f}%")
        print(f"  Std:  {np.std(baseline_accs, ddof=1):.2f}%")

        print(f"\nPre-trained (Stage 1 Init):")
        print(f"  Accuracies: {pretrained_accs}")
        print(f"  Mean: {np.mean(pretrained_accs):.2f}%")
        print(f"  Std:  {np.std(pretrained_accs, ddof=1):.2f}%")

        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(baseline_accs, pretrained_accs)

        print(f"\nIndependent Samples t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.4f}")

        if p_value < 0.05:
            print(f"  Result:      ✓ Statistically significant (p < 0.05)")

            improvement = np.mean(pretrained_accs) - np.mean(baseline_accs)
            rel_improvement = (improvement / np.mean(baseline_accs)) * 100

            print(f"\nImprovement:")
            print(f"  Absolute: +{improvement:.2f}%")
            print(f"  Relative: +{rel_improvement:.1f}%")
        else:
            print(f"  Result:      ✗ Not significant (p >= 0.05)")

        print(f"\n{'='*70}\n")
