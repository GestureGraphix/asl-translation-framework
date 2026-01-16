# Training Diagnosis - Poor Results (0.60% accuracy)

**Problem**: Severe overfitting - train loss 0.48 vs val loss 11.38, accuracy dropped from 3.57% → 0.60%

---

## 🔍 Symptoms

- **Train Loss**: 0.4767 (very low - model memorizing training data)
- **Val Loss**: 11.3826 (very high - not generalizing)
- **Best Val Accuracy**: 3.57% (epoch 20)
- **Final Val Accuracy**: 0.60% (got worse over time!)
- **Blank Penalty**: 0.2719 (~27% blanks, better than before but still high)

**Diagnosis**: Severe overfitting + potential model collapse

---

## 🎯 Key Questions to Answer

### 1. Was Stage 1 checkpoint loaded?

**Check**: Look at the notebook output after Cell 12 (Stage 1 loading)

**Expected output if loaded**:
```
✓ Found 'encoder_state_dict' in checkpoint
✓ Successfully loaded pre-trained encoder weights!
✓ Encoder initialized with Stage 1 pre-trained features
```

**If NOT loaded**: You're training from random initialization, which explains poor results.

**Action**: Verify Stage 1 checkpoint was loaded. If not, upload it and retrain.

---

### 2. What is the model actually predicting?

Run this diagnostic code to see what's happening:

```python
# Diagnostic: Check predictions
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
    
    print(f"Blank predictions: {blank_ratio*100:.1f}%")
    print(f"Unique predictions: {len(unique_preds)} out of {vocab_size} classes")
    print(f"Most common predictions: {torch.bincount(predictions.flatten().cpu()).argsort(descending=True)[:10].tolist()}")
    
    # Check per-sample predictions
    print(f"\nSample predictions vs true labels:")
    for i in range(min(5, len(gloss_ids))):
        pred_seq = predictions[i, :lengths[i]].cpu().numpy()
        non_blank = pred_seq[pred_seq != 0]
        if len(non_blank) > 0:
            pred_id = np.bincount(non_blank).argmax()
        else:
            pred_id = 0
        true_id = gloss_ids[i].item()
        print(f"  Sample {i}: Predicted={pred_id}, True={true_id}, Match={pred_id==true_id}")
```

**What to look for**:
- Is model predicting mostly blanks? (>50% = problem)
- Is model predicting only 1-2 classes? (collapse = problem)
- Are predictions random or systematic?

---

### 3. Model Capacity vs Data

**Current model**: ~130K parameters (128 hidden, 2 layers, 100 classes)
**Training samples**: 806
**Ratio**: ~160:1 (parameters per sample)

**Issue**: For 100 classes, this ratio might still be too high. Consider:
- Reducing model size (hidden_dim: 128 → 64, layers: 2 → 1)
- Increasing regularization (dropout: 0.3 → 0.5)
- More data augmentation

---

## 🔧 Recommended Fixes

### Fix 1: Verify Stage 1 Checkpoint Was Loaded

**If NOT loaded**: Upload checkpoint and retrain
- Location: `/content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt`
- Re-run Cell 12 to load it
- Re-run training

### Fix 2: Reduce Model Size (if overfitting)

**Option A**: Smaller model
```python
HIDDEN_DIM = 64  # Reduced from 128
NUM_LAYERS = 1   # Reduced from 2
DROPOUT = 0.5    # Increased from 0.3
```

**Option B**: More regularization
```python
DROPOUT = 0.5
LEARNING_RATE = 5e-4  # Reduced from 1e-3
weight_decay=1e-3     # Increased from 1e-4
```

### Fix 3: Early Stopping

The model got worse after epoch 20 (3.57% → 0.60%). Add early stopping:

```python
# Add after validation
if val_acc < best_val_acc - 1.0:  # If accuracy drops by 1%
    print(f"  ⚠️ Validation accuracy dropped significantly")
    patience += 1
    if patience >= 5:
        print(f"  Early stopping at epoch {epoch}")
        break
else:
    patience = 0
```

### Fix 4: Lower Learning Rate

High learning rate (1e-3) might cause instability. Try:

```python
LEARNING_RATE = 5e-4  # Or even 1e-4
```

### Fix 5: Check Data Quality

Verify data is correct:

```python
# Check data distribution
from collections import Counter
gloss_counts = Counter([id_to_gloss[gid.item()] for gid in train_dataset.samples[:]['gloss_id']])
print(f"Most common signs: {gloss_counts.most_common(10)}")
print(f"Total unique signs: {len(gloss_counts)}")
print(f"Average samples per sign: {len(train_dataset) / len(gloss_counts):.1f}")
```

**Expected**: ~8 samples per sign on average (806 samples / 100 signs)

---

## 🎯 Immediate Action Plan

### Step 1: Diagnose (5 min)
1. Check if Stage 1 checkpoint was loaded (look at Cell 12 output)
2. Run diagnostic code to see what model is predicting
3. Check data distribution

### Step 2: Fix Based on Diagnosis

**If Stage 1 NOT loaded**:
→ Upload checkpoint, retrain (most likely issue!)

**If model collapsing to blanks/few classes**:
→ Reduce model size, increase regularization

**If severe overfitting**:
→ Smaller model, more dropout, lower learning rate

**If data issues**:
→ Check feature extraction quality

### Step 3: Retrain with Fixes (1-2 hours)

Use best checkpoint (epoch 20: 3.57%) as starting point if needed.

---

## 📊 Expected Results After Fixes

**With Stage 1 + smaller model**:
- Train Loss: 1.5-3.0 (not too low)
- Val Loss: 2.5-4.0 (closer to train)
- Val Accuracy: 10-25% (realistic for 100 signs)

**Without Stage 1 but with smaller model**:
- Train Loss: 2.0-4.0
- Val Loss: 3.0-5.0
- Val Accuracy: 5-15% (lower but still better than 0.60%)

---

## 🔬 Root Cause Analysis

**Most Likely Issue**: Stage 1 checkpoint not loaded
- If training from random init on 100 classes with only 806 samples
- Model struggles to learn with such sparse data
- Needs pre-trained features (Stage 1) for better initialization

**Secondary Issue**: Model too large
- 130K params for 806 samples = ~160:1 ratio
- Should aim for <100:1 ratio
- Reduce to ~80K params (64 hidden, 1 layer)

**Tertiary Issue**: Learning rate too high
- 1e-3 might cause instability
- Lower to 5e-4 or 1e-4 for more stable training

---

**Next Step**: Check if Stage 1 checkpoint was loaded, then apply appropriate fixes.
