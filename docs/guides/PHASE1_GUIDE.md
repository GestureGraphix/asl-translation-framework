# Phase 1 Training Guide
## Prove the System Works

**Goal**: Achieve >50% accuracy on 20 sign classes to validate infrastructure.

**Timeline**: 1-2 hours total

---

## 🎯 What Phase 1 Fixes

| Problem (Initial Training) | Solution (Phase 1) |
|----------------------------|-------------------|
| Only 20 samples | **~1000 samples** (50 per class × 20 classes) |
| 1.18M parameters | **~50K parameters** (64 hidden, 1 layer) |
| 2000 classes (vocab) | **21 classes** (20 signs + blank) |
| 0% accuracy (blank collapse) | **Blank penalty** prevents collapse |
| Massive overfitting | **Better regularization** (dropout 0.5, weight decay) |

---

## 📋 Step-by-Step Instructions

### Step 1: Cache Features (~30-45 min)

Extract and cache phonological features from 1000 videos:

```bash
python3.11 scripts/cache_phase1_features.py --max-glosses 20
```

**What this does:**
- Selects 20 most common sign classes from WLASL
- Extracts features from ~50 train + ~25 val videos per class
- Saves to `data/cache/phase1/train_20glosses.pkl` and `val_20glosses.pkl`

**Expected output:**
```
Creating training dataset...
Including 20 glosses: ['after', 'all', 'before', 'book', 'but', ...]
Samples: 983

Extracting Training Features
100%|████████████| 983/983 [25:30<00:00]
✓ Saved 983 cached features

Extracting Validation Features
100%|████████████| 492/492 [12:45<00:00]
✓ Saved 492 cached features
```

**Monitoring:**
- Watch GPU temp: should stay <70°C
- Check VRAM: MediaPipe uses ~500MB
- Press Ctrl+C if needed (progress is saved)

---

### Step 2: Train Model (~10-15 min)

Train with cached features on GPU:

```bash
python3.11 scripts/train_phase1.py \
  --use-cache \
  --max-glosses 20 \
  --device cuda \
  --batch-size 8 \
  --num-epochs 30
```

**What this does:**
- Loads cached features (instant loading!)
- Trains BiLSTM encoder (64 hidden, 1 layer, ~50K params)
- Uses blank penalty (0.1) to prevent blank collapse
- Saves checkpoints to `checkpoints/phase1/`

**Expected output:**
```
Phase 1 Trainer:
  Device: cuda
  Model parameters: 51,329
  Target: ~50K params ✓
  GPU: NVIDIA GeForce MX550
  VRAM: 2.0 GB

Epoch 1/30:
  Train Loss: 45.2341
  Val Loss:   32.1234
  Val Acc:    15.2% (75/492)

Epoch 5/30:
  Train Loss: 12.4532
  Val Loss:   18.3421
  Val Acc:    35.8% (176/492)
  ✓ New best accuracy!

Epoch 10/30:
  Train Loss: 4.2341
  Val Loss:   12.5432
  Val Acc:    52.4% (258/492)
  ✓ New best accuracy!

✓ Reached 50% accuracy - Phase 1 goal achieved!
```

**GPU monitoring:**
- Temp: Should stay 55-65°C (way cooler than initial 97°C on CPU!)
- VRAM: ~100-200 MB allocated
- Speed: ~30 sec/epoch

---

## 🎓 Understanding the Results

### Success Criteria

✅ **50%+ validation accuracy** - System works!
- Features are meaningful
- CTC training successful
- No blank collapse

⚠️ **30-50% validation accuracy** - Partial success
- System working but needs tuning
- Try: lower learning rate, more epochs, different classes

❌ **<30% validation accuracy** - Still issues
- Check feature extraction (visualize some samples)
- Verify labels are correct
- Try even simpler model (hidden_dim=32)

### What to Check

**1. Training curves** (auto-generated):
```bash
python3.11 scripts/analyze_training.py checkpoints/phase1/ --plot
```
- Should see train loss decreasing steadily
- Val loss should decrease then plateau (not increase)
- Val accuracy should increase

**2. Predictions**:
```bash
python3.11 scripts/quick_test.py \
  --checkpoint checkpoints/phase1/best_model.pt \
  --cache-dir data/cache/phase1
```
- Should NOT always predict blank
- Should see variety in predictions
- Some classes should have >70% accuracy

**3. Per-class performance**:
- Training script prints top-5 classes by accuracy
- Identify which signs are easy/hard
- Hard signs might need more data

---

## 🔧 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size
```bash
python3.11 scripts/train_phase1.py --use-cache --batch-size 4
```

---

### Issue: Feature caching too slow

**Solution:** Reduce number of classes or samples
```bash
# Fewer classes (10 instead of 20)
python3.11 scripts/cache_phase1_features.py --max-glosses 10

# OR fewer samples per class (30 instead of 50)
python3.11 scripts/cache_phase1_features.py --samples-per-gloss 30
```

---

### Issue: Still getting blank collapse (all predictions are blank)

**Solutions:**
1. Increase blank penalty:
```bash
python3.11 scripts/train_phase1.py --use-cache --blank-penalty 0.5
```

2. Check features are actually different:
```bash
python3.11 -c "
import pickle
with open('data/cache/phase1/train_20glosses.pkl', 'rb') as f:
    cache = pickle.load(f)

for vid, feat in list(cache.items())[:3]:
    print(f'{vid}: shape={feat.shape}, mean={feat.mean():.3f}, std={feat.std():.3f}')
"
```
Features should have different means/stds, not all zeros.

---

### Issue: Accuracy stuck at random guessing (~5% for 20 classes)

**Solutions:**
1. Train longer:
```bash
python3.11 scripts/train_phase1.py --use-cache --num-epochs 50
```

2. Lower learning rate:
```bash
python3.11 scripts/train_phase1.py --use-cache --lr 5e-4
```

3. Try different optimizer (in code):
Change `optim.Adam` to `optim.SGD(lr=0.01, momentum=0.9)`

---

## 📊 Hyperparameter Guide

### Model Size

| Config | Params | When to Use |
|--------|--------|-------------|
| `--hidden-dim 32 --num-layers 1` | ~15K | Smallest model, debugging |
| `--hidden-dim 64 --num-layers 1` | **~50K** | **Phase 1 default** ✓ |
| `--hidden-dim 128 --num-layers 1` | ~200K | More capacity if >50% achieved |
| `--hidden-dim 64 --num-layers 2` | ~100K | Deeper model for complex patterns |

### Data Amount

| Config | Samples | When to Use |
|--------|---------|-------------|
| `--max-glosses 10 --samples-per-gloss 30` | ~300 | Quick test |
| `--max-glosses 20 --samples-per-gloss 50` | **~1000** | **Phase 1 default** ✓ |
| `--max-glosses 50 --samples-per-gloss 30` | ~1500 | More variety |
| `--max-glosses 20 --samples-per-gloss 100` | ~2000 | More data per class |

### Training

| Hyperparameter | Default | Range | Notes |
|----------------|---------|-------|-------|
| `--lr` | 1e-3 | 1e-4 to 1e-2 | Lower if unstable |
| `--batch-size` | 8 | 4 to 16 | Limited by VRAM |
| `--blank-penalty` | 0.1 | 0.0 to 1.0 | Increase if blank collapse |
| `--num-epochs` | 30 | 10 to 100 | Auto-stops at 50% |

---

## ✅ Next Steps After Phase 1 Success

Once you achieve >50% accuracy:

**Option 1: Scale Gradually** (Recommended)
```bash
# Try 50 classes
python3.11 scripts/cache_phase1_features.py --max-glosses 50
python3.11 scripts/train_phase1.py --use-cache --max-glosses 50 --hidden-dim 128

# Then 100 classes
python3.11 scripts/cache_phase1_features.py --max-glosses 100
python3.11 scripts/train_phase1.py --use-cache --max-glosses 100 --hidden-dim 256
```

**Option 2: Implement Stage 1 (Paper's Approach)**
- Self-supervised pre-training on unlabeled videos
- Learn better phonological features
- Then scale to 500-2000 classes

**Option 3: Add Improvements**
- Data augmentation (temporal, spatial)
- Attention mechanism
- Ensemble models

---

## 📈 Expected Timeline

| Task | Time | Notes |
|------|------|-------|
| Feature caching (20 classes) | 30-45 min | One-time cost |
| Training (30 epochs) | 10-15 min | Can iterate quickly |
| Analysis | 5 min | Check results |
| **Total** | **~1 hour** | Plus debugging time |

---

## 🎯 Success Definition

**Phase 1 Complete** when:
- ✅ Val accuracy >50% on 20 classes
- ✅ Model NOT always predicting blank
- ✅ Train loss decreasing steadily
- ✅ Some classes achieving >70% accuracy
- ✅ GPU training stable (<70°C, no OOM)

**Then**: Move to Phase 2 (scaling) or implement Stage 1 (paper's pre-training)

---

## 📝 Files Created

```
scripts/
  cache_phase1_features.py    # Step 1: Cache features
  train_phase1.py             # Step 2: Train model
  analyze_training.py         # Analyze results
  quick_test.py              # Test predictions

src/
  utils/wlasl_dataset_filtered.py  # Filtered vocabulary dataset
  models/ctc_head.py               # Updated with blank penalty

data/cache/phase1/
  train_20glosses.pkl         # Cached train features
  val_20glosses.pkl           # Cached val features

checkpoints/phase1/
  checkpoint_epoch_*.pt       # Training checkpoints
  best_model.pt              # Best model
```

---

Ready to start? Run:
```bash
# Step 1: Cache features
python3.11 scripts/cache_phase1_features.py --max-glosses 20

# Step 2: Train (after caching completes)
python3.11 scripts/train_phase1.py --use-cache --max-glosses 20 --device cuda
```
