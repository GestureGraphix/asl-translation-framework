# Quick Start: Train WLASL100 on Google Colab

**Problem**: Local training takes 30+ hours for 100 signs  
**Solution**: Use Google Colab (1-2 hours, 15-30x faster)

---

## 🎯 Goal

Train a 100-sign ASL recognition model following the paper's 3-stage curriculum, using Google Colab for faster training.

---

## ✅ Prerequisites

1. **Features extracted** ✅ (already done)
   - Location: `/content/drive/MyDrive/asl_data/extracted_features/`
   - Files: `features_train_wlasl100.pkl`, `features_val_wlasl100.pkl`, `vocabulary.json`

2. **Stage 1 checkpoint** (optional but recommended)
   - Location: `checkpoints/stage1/checkpoint_best.pt` (on your laptop)
   - Upload to: `/content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt`

3. **Colab notebook** ✅ (already exists)
   - File: `notebooks/WLASL100_Training_Colab.ipynb`

---

## 📋 Step-by-Step Instructions

### Step 1: Open Colab Notebook (5 min)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/WLASL100_Training_Colab.ipynb`
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)

### Step 2: Upload Stage 1 Checkpoint (5 min)

**Option A: Upload from laptop** (recommended)
1. On your laptop, locate: `checkpoints/stage1/checkpoint_best.pt`
2. Upload to Google Drive at: `/content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt`
3. Or use Colab's file upload in the notebook

**Option B: Skip Stage 1** (faster, but may get lower accuracy)
- The notebook will train from random initialization
- Expected: 8-15% accuracy (vs 10-30% with pre-training)

### Step 3: Run Training (1-2 hours)

1. **Mount Google Drive** (Cell 5)
   - Mounts your Drive to access features

2. **Load Features** (Cell 5)
   - Loads cached features from Drive
   - Should show: 806 train, 168 val samples

3. **Load Pre-trained Encoder** (Cell 25) - **IMPORTANT**
   - Uncomment the code in Cell 25
   - This loads Stage 1 pre-trained weights
   - **This is why the previous run got 3.57% - it skipped this step!**

4. **Create Model** (Cell 11)
   - Creates encoder + CTC head
   - If Stage 1 loaded: encoder has pre-trained weights ✅
   - If not: encoder is random initialization ⚠️

5. **Train** (Cell 14)
   - Runs training loop (50 epochs)
   - Monitor progress in output
   - Best model saved automatically

### Step 4: Save Results (5 min)

1. **Save to Drive** (Cell 16)
   - Copies checkpoints to Google Drive
   - Location: `/content/drive/MyDrive/asl_data/checkpoints_wlasl100/`

2. **Download** (optional)
   - Download checkpoints to your laptop for local evaluation

---

## 🔧 Key Fixes Needed

### Fix 1: Load Pre-trained Encoder (CRITICAL)

**Current issue**: Notebook skips Stage 1 pre-training → 3.57% accuracy

**Fix**: Uncomment and run Cell 25 before creating the model (Cell 11)

```python
# In Cell 25, uncomment this code:
stage1_checkpoint_path = Path('/content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt')

if stage1_checkpoint_path.exists():
    # ... (rest of loading code)
    encoder.load_state_dict(encoder_state, strict=False)
    print("✓ Loaded pre-trained encoder!")
```

**Why this matters**:
- Paper's curriculum: Stage 1 → Stage 2
- Pre-trained encoder learns good features
- Without it: training from scratch on limited data

### Fix 2: Better Hyperparameters (OPTIONAL)

**Current settings** (Cell 10):
- `HIDDEN_DIM = 128`
- `BLANK_PENALTY = 0.1`
- `LEARNING_RATE = 1e-3`

**Recommended for 100 signs**:
```python
HIDDEN_DIM = 256  # More capacity for 100 classes
BLANK_PENALTY = 0.5  # More aggressive blank suppression
LEARNING_RATE = 5e-4  # More stable training
```

### Fix 3: Use Improved Training Function (OPTIONAL)

The notebook has an improved training function (Cell 20) with better blank penalty computation. You can replace the training function in Cell 13 with the one from Cell 20.

---

## 📊 Expected Results

### With Stage 1 Pre-training (Following Paper)
- **Top-1 Accuracy**: 15-30%
- **Top-5 Accuracy**: 35-50%
- **Training Time**: 1-2 hours on Colab
- **Status**: ✅ Follows paper's curriculum

### Without Stage 1 (Baseline)
- **Top-1 Accuracy**: 8-15%
- **Top-5 Accuracy**: 20-35%
- **Training Time**: 1-2 hours on Colab
- **Status**: ⚠️ Doesn't follow paper, but faster

### Current (Broken - No Pre-training + Wrong Setup)
- **Top-1 Accuracy**: 3.57% ❌
- **Issue**: Skipped Stage 1, wrong blank penalty computation

---

## 🐛 Troubleshooting

### Issue: "Checkpoint not found"
**Solution**: Make sure you uploaded `checkpoints/stage1/checkpoint_best.pt` to Google Drive at the correct path.

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in Cell 10:
```python
BATCH_SIZE = 8  # Instead of 16
```

### Issue: "Low accuracy (<5%)"
**Possible causes**:
1. Didn't load Stage 1 checkpoint → Load it (Cell 25)
2. Blank collapse → Increase `BLANK_PENALTY` to 0.5
3. Model too small → Increase `HIDDEN_DIM` to 256

### Issue: "Training too slow"
**Solution**: Make sure GPU is enabled (Runtime → Change runtime type → GPU)

---

## 📈 After Training

### Evaluate Results

1. **Check accuracy**:
   - Best model saved in Cell 14
   - Look for: "Best Val Accuracy: X.XX%"

2. **Compare with baseline**:
   - 20-sign baseline: 21.74% (with pre-training: 26.09%)
   - 100-sign target: 10-30% (realistic for larger vocabulary)

3. **Analyze per-class**:
   - Which signs work best?
   - Which signs are hardest?

### Next Steps

**If accuracy ≥15%**:
- ✅ System scales to 100 signs
- Next: Scale to 500 signs (Option C in NEXT_STEPS_ANALYSIS.md)

**If accuracy 8-15%**:
- ⚠️ System works but needs improvement
- Try: Larger model, more data, better hyperparameters

**If accuracy <8%**:
- ❌ System struggling
- Debug: Check data quality, model architecture, training setup

---

## ⏱️ Time Estimate

| Step | Time | Notes |
|------|------|-------|
| Setup (upload notebook, enable GPU) | 5 min | One-time |
| Upload Stage 1 checkpoint | 5 min | One-time |
| Load features | 1 min | Automatic |
| Load pre-trained encoder | 1 min | **Critical step** |
| Create model | 30 sec | Automatic |
| **Training** | **1-2 hours** | **Main time** |
| Save results | 5 min | Automatic |
| **Total** | **~2-3 hours** | **vs 30+ hours locally** |

---

## 🎓 Key Takeaways

1. **Use Colab for training** - 15-30x faster than local GPU
2. **Load Stage 1 checkpoint** - Critical for following paper and getting good results
3. **Monitor accuracy** - Should be 10-30% for 100 signs
4. **Save checkpoints** - Download to laptop for evaluation

---

## 📚 References

- **Full analysis**: `NEXT_STEPS_ANALYSIS.md`
- **Paper curriculum**: `CLAUDE.md` (Section 7.2)
- **Current status**: `STATUS.md`
- **Ablation study**: `ABLATION_STUDY_STATUS.md` (pre-training doesn't help on 20 signs, but might on 100)

---

**Ready to train?** Open the Colab notebook and follow Steps 1-4 above!
