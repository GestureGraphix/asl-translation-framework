# Diagnostic Code - Run this in Colab to diagnose training issues
# Add as a new cell after training completes

import torch
import numpy as np
from collections import Counter

# 1. Check if Stage 1 checkpoint was loaded
print("="*70)
print("DIAGNOSIS 1: Stage 1 Checkpoint Status")
print("="*70)
print(f"USE_PRETRAINED flag: {USE_PRETRAINED}")
if USE_PRETRAINED:
    print("✓ Stage 1 checkpoint was loaded")
else:
    print("✗ Stage 1 checkpoint was NOT loaded - training from random init!")
    print("  This is likely the main cause of poor results.")
    print("  Solution: Upload checkpoint and retrain")
print()

# 2. Check model predictions
print("="*70)
print("DIAGNOSIS 2: Model Predictions")
print("="*70)
model.eval()
with torch.no_grad():
    sample_batch = next(iter(val_loader))
    features = sample_batch['features'].to(DEVICE)
    lengths = sample_batch['lengths']
    gloss_ids = sample_batch['gloss_ids']
    
    logits = model(features, lengths)
    predictions = logits.argmax(dim=-1)
    
    # Check blank ratio
    blank_count = (predictions == 0).sum().item()
    total_count = predictions.numel()
    blank_ratio = blank_count / total_count
    
    # Check unique predictions
    unique_preds = torch.unique(predictions)
    pred_counts = torch.bincount(predictions.flatten().cpu())
    most_common_preds = pred_counts.argsort(descending=True)[:10]
    
    print(f"Blank predictions: {blank_ratio*100:.1f}%")
    print(f"Unique predictions: {len(unique_preds)} out of {vocab_size} classes")
    print(f"Most common prediction IDs: {most_common_preds.tolist()}")
    print(f"Most common prediction counts: {pred_counts[most_common_preds].tolist()}")
    
    # Check per-sample accuracy
    correct = 0
    for i in range(len(gloss_ids)):
        pred_seq = predictions[i, :lengths[i]].cpu().numpy()
        non_blank = pred_seq[pred_seq != 0]
        if len(non_blank) > 0:
            pred_id = np.bincount(non_blank).argmax()
        else:
            pred_id = 0
        true_id = gloss_ids[i].item()
        if pred_id == true_id:
            correct += 1
    
    print(f"Sample batch accuracy: {correct}/{len(gloss_ids)} ({correct/len(gloss_ids)*100:.1f}%)")
print()

# 3. Check data distribution
print("="*70)
print("DIAGNOSIS 3: Data Distribution")
print("="*70)
train_gloss_ids = [sample['gloss_id'] for sample in train_dataset.samples]
train_gloss_counts = Counter(train_gloss_ids)
val_gloss_ids = [sample['gloss_id'] for sample in val_dataset.samples]
val_gloss_counts = Counter(val_gloss_ids)

print(f"Train: {len(train_dataset)} samples, {len(train_gloss_counts)} unique signs")
print(f"Val: {len(val_dataset)} samples, {len(val_gloss_counts)} unique signs")
print(f"Avg samples per sign (train): {len(train_dataset) / len(train_gloss_counts):.1f}")
print(f"Avg samples per sign (val): {len(val_dataset) / len(val_gloss_counts):.1f}")
print(f"Signs with <3 train samples: {sum(1 for c in train_gloss_counts.values() if c < 3)}")
print(f"Signs with <2 val samples: {sum(1 for c in val_gloss_counts.values() if c < 2)}")
print()

# 4. Model capacity analysis
print("="*70)
print("DIAGNOSIS 4: Model Capacity")
print("="*70)
num_params = sum(p.numel() for p in model.parameters())
params_per_sample = num_params / len(train_dataset)
print(f"Model parameters: {num_params:,}")
print(f"Train samples: {len(train_dataset)}")
print(f"Params per sample: {params_per_sample:.0f}:1")
if params_per_sample > 100:
    print("⚠️ Warning: High params/sample ratio (>100:1) - risk of overfitting")
    print("  Recommendation: Reduce model size (hidden_dim: 128→64, layers: 2→1)")
print()

# 5. Training curve analysis
print("="*70)
print("DIAGNOSIS 5: Training Curves")
print("="*70)
if len(train_losses) > 0:
    print(f"Initial train loss: {train_losses[0]:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Best val accuracy: {max(val_accs):.2f}% (epoch {val_accs.index(max(val_accs))+1})")
    print(f"Final val accuracy: {val_accs[-1]:.2f}%")
    
    # Check for overfitting
    if train_losses[-1] < 1.0 and val_losses[-1] > 5.0:
        print("✗ Severe overfitting detected!")
        print("  Train loss very low but val loss very high")
    elif val_accs[-1] < val_accs[0]:
        print("✗ Accuracy decreased over training!")
        print("  Model got worse - possible collapse or too high learning rate")
print()

# 6. Recommendations
print("="*70)
print("RECOMMENDATIONS")
print("="*70)
recommendations = []

if not USE_PRETRAINED:
    recommendations.append("1. UPLOAD STAGE 1 CHECKPOINT - Most critical fix!")
    recommendations.append("   - Upload checkpoints/stage1/checkpoint_best.pt to Google Drive")
    recommendations.append("   - Place at: /content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt")
    recommendations.append("   - Re-run Cell 12 and retrain")

if params_per_sample > 100:
    recommendations.append("2. REDUCE MODEL SIZE - Too many parameters")
    recommendations.append("   - Change: HIDDEN_DIM = 64 (was 128)")
    recommendations.append("   - Change: NUM_LAYERS = 1 (was 2)")
    recommendations.append("   - Change: DROPOUT = 0.5 (was 0.3)")

if blank_ratio > 0.3:
    recommendations.append("3. ADDRESS BLANK COLLAPSE - Too many blank predictions")
    recommendations.append("   - Increase: BLANK_PENALTY = 0.75 (was 0.5)")
    recommendations.append("   - Check if model is learning anything")

if val_accs[-1] < val_accs[0] if len(val_accs) > 1 else False:
    recommendations.append("4. LOWER LEARNING RATE - Model getting worse over time")
    recommendations.append("   - Change: LEARNING_RATE = 5e-4 (was 1e-3)")
    recommendations.append("   - Add early stopping")

if len(recommendations) == 0:
    print("✓ No obvious issues found. Check data quality or try different architecture.")
else:
    for rec in recommendations:
        print(rec)

print("="*70)
