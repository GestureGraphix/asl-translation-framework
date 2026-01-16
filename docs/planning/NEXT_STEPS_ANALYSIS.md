# Next Steps Analysis - Training Time Optimization

**Date**: January 2026  
**Issue**: Local training takes too much time (30+ hours for 100 signs)  
**Context**: WLASL100 features extracted, Colab notebook ready, paper's 3-stage curriculum to follow

---

## Executive Summary

**Problem**: Your local GPU (MX550 2GB) is too slow for larger datasets:
- 20 signs: ~10-15 min ✅ (acceptable)
- 100 signs: ~30+ hours ❌ (too slow)
- 500+ signs: Days/weeks ❌ (impractical)

**Solution**: **Use Google Colab for all training** (already set up, 10-20x faster)

**Recommendation**: Fix the Colab training notebook to properly follow the paper's 3-stage curriculum, then scale to 500-2000 signs on Colab.

---

## Current State Assessment

### ✅ What's Working

1. **Infrastructure Validated** (Phase 1, 20 signs):
   - 21.74% baseline accuracy (4.3x random baseline)
   - 26.09% with pre-training (5.2x random baseline)
   - All components functional (MediaPipe, features, encoder, CTC)

2. **Feature Extraction Complete**:
   - WLASL100: 1,077 videos extracted on Google Colab ✅
   - Features saved to Google Drive
   - Ready for training

3. **Colab Setup Ready**:
   - Training notebook exists (`WLASL100_Training_Colab.ipynb`)
   - Feature extraction working on Colab
   - GPU access available (Tesla T4, 16GB VRAM)

### ⚠️ Issues Identified

1. **Training Notebook Problems**:
   - Got only 3.57% accuracy (should be 10-30%)
   - **Skipped Stage 1 pre-training** (violates paper's curriculum)
   - Training from random initialization instead of pre-trained encoder
   - Missing multi-task loss components

2. **Local Training Bottleneck**:
   - MX550 2GB GPU too slow for 100+ signs
   - Estimated 30+ hours for WLASL100
   - Not scalable to 500-2000 signs

3. **Paper Compliance**:
   - Not following Section 7.2 (3-stage curriculum)
   - Stage 1 should train first (self-supervised)
   - Stage 2 should use pre-trained encoder weights

---

## Paper's 3-Stage Curriculum (Section 7.2)

### Stage 1: Self-Supervised Phonological Pre-training
**Purpose**: Learn robust phonological features from unlabeled video

**Components**:
- Contrastive learning (temporal coherence)
- Product Vector Quantization (phonological codebooks)
- Loss: `L_contrast + L_phon`

**Output**: Pre-trained encoder weights

**Status**: ✅ Implemented, checkpoint available (`checkpoints/stage1/checkpoint_best.pt`)

### Stage 2: End-to-End CTC Training
**Purpose**: Fine-tune encoder for gloss prediction

**Components**:
- **Should load Stage 1 pre-trained encoder** ⚠️ (currently skipped)
- CTC loss for gloss sequences
- Multi-task loss: `L_CTC + λ_seg L_seg + λ_phon L_phon`
- Currently only using `L_CTC`

**Status**: ⚠️ Partially implemented (missing pre-trained init, multi-task loss)

### Stage 3: WFST Fine-tuning (Future)
**Purpose**: Discriminative training with full pipeline

**Components**:
- WFST decoder cascade
- Discourse tracking
- Morphological fusion
- Not yet needed for 100-sign validation

---

## Ablation Study Findings (Important!)

**Result**: Pre-training does NOT help on 20-sign task (p=0.37, not significant)

**Implications**:
- With ~12 samples/class, supervised data is sufficient
- Pre-training benefits may only emerge at larger vocabularies
- **For 100 signs**: Pre-training might help (fewer samples/class)

**Decision**: 
- **For 20 signs**: Can skip Stage 1 (validated)
- **For 100+ signs**: Should use Stage 1 (per paper, and data sparsity increases)

---

## Recommended Next Steps

### Option A: Fix Colab Notebook + Train WLASL100 (RECOMMENDED) ⭐

**Goal**: Properly train 100-sign model following paper's curriculum

**Steps**:

1. **Fix Training Notebook** (1-2 hours):
   - Add Stage 1 pre-training (or load existing checkpoint)
   - Load pre-trained encoder weights for Stage 2
   - Add multi-task loss components (optional for now)
   - Fix blank penalty computation (ratio-based, not mean-based)

2. **Upload Stage 1 Checkpoint** (5 min):
   - Upload `checkpoints/stage1/checkpoint_best.pt` to Google Drive
   - Place at: `/content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt`

3. **Train on Colab** (1-2 hours):
   - Run Stage 1 pre-training on WLASL100 features (if needed)
   - Run Stage 2 CTC with pre-trained encoder
   - Expected accuracy: 10-30% (vs 3.57% current)

4. **Evaluate**:
   - Compare with 20-sign baseline
   - Test if pre-training helps at 100-sign scale
   - Document findings

**Time**: 3-4 hours total  
**Risk**: Low (infrastructure proven)  
**Value**: High (validates scaling, follows paper)

---

### Option B: Skip Stage 1, Train Baseline (FASTEST)

**Goal**: Quick validation that 100-sign model works

**Steps**:
1. Fix notebook (remove Stage 1 requirement)
2. Train Stage 2 from random initialization
3. Compare with 20-sign baseline

**Time**: 1-2 hours  
**Risk**: Low  
**Value**: Medium (quick validation, but doesn't follow paper)

**Note**: Ablation study suggests this might work fine for 100 signs too, but paper recommends Stage 1.

---

### Option C: Scale to 500 Signs on Colab (AMBITIOUS)

**Goal**: Realistic vocabulary size matching paper's scope

**Steps**:
1. Extract features for 500 signs (3-5 hours on Colab)
2. Scale model architecture (256 hidden, 2 layers, ~1M params)
3. Run full 3-stage curriculum
4. Compare with published WLASL baselines

**Time**: 1-2 weeks (mostly waiting for feature extraction)  
**Risk**: Medium (larger model, more data)  
**Value**: Very High (publishable results)

**Prerequisites**: Complete Option A first (validate on 100 signs)

---

## Detailed Action Plan: Option A (Recommended)

### Step 1: Fix Colab Training Notebook

**File**: `notebooks/WLASL100_Training_Colab.ipynb`

**Changes Needed**:

1. **Add Stage 1 Pre-training Cell** (or load checkpoint):
   ```python
   # Option 1: Load existing Stage 1 checkpoint
   stage1_checkpoint_path = Path('/content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt')
   
   # Option 2: Run Stage 1 pre-training on WLASL100 features
   # (if checkpoint not available or want to retrain)
   ```

2. **Load Pre-trained Encoder Weights**:
   ```python
   if stage1_checkpoint_path.exists():
       checkpoint = torch.load(stage1_checkpoint_path, map_location=DEVICE)
       encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
       print("✓ Loaded pre-trained encoder from Stage 1")
   ```

3. **Fix Blank Penalty** (use ratio-based, not mean-based):
   ```python
   # Current (wrong): blank_penalty * blank_log_probs.mean()
   # Fixed (correct): blank_penalty * compute_blank_ratio(logits)
   ```

4. **Improve Hyperparameters**:
   - Increase blank penalty: 0.1 → 0.5
   - Consider larger model: hidden_dim 128 → 256 (for 100 classes)
   - Lower learning rate: 1e-3 → 5e-4 (more stable)

### Step 2: Upload Stage 1 Checkpoint

**From local machine**:
```bash
# If you have the checkpoint locally
# Upload to Google Drive at:
# /content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt
```

**Or re-run Stage 1 on Colab** (if checkpoint not available):
- Use WLASL100 features for pre-training
- Takes ~30-60 min on Colab GPU
- Saves checkpoint for Stage 2

### Step 3: Train and Evaluate

**Expected Results**:
- **With Stage 1**: 15-30% accuracy (following paper)
- **Without Stage 1**: 8-15% accuracy (baseline)
- **Current (broken)**: 3.57% accuracy

**Metrics to Track**:
- Validation accuracy (top-1, top-5)
- Per-class accuracy (which signs work best)
- Training time (should be 1-2 hours on Colab)

---

## Training Time Comparison

| Dataset | Local (MX550) | Colab (T4) | Speedup |
|---------|---------------|------------|---------|
| 20 signs | 10-15 min | 5-10 min | 1.5-2x |
| 100 signs | 30+ hours | 1-2 hours | **15-30x** |
| 500 signs | Days/weeks | 4-6 hours | **100x+** |

**Conclusion**: Colab is essential for scaling beyond 20 signs.

---

## Decision Matrix

| Option | Time | Follows Paper | Risk | Value | Recommended? |
|--------|------|---------------|------|-------|--------------|
| **A: Fix + Train WLASL100** | 3-4 hours | ✅ Yes | Low | High | ⭐ **YES** |
| B: Skip Stage 1 | 1-2 hours | ❌ No | Low | Medium | Maybe (quick test) |
| C: Scale to 500 | 1-2 weeks | ✅ Yes | Medium | Very High | Later (after A) |

---

## Immediate Actions (This Week)

### Day 1: Fix Notebook
- [ ] Update `WLASL100_Training_Colab.ipynb` with Stage 1 loading
- [ ] Fix blank penalty computation
- [ ] Test notebook setup (load features, create model)

### Day 2: Upload Checkpoint
- [ ] Upload Stage 1 checkpoint to Google Drive
- [ ] Or run Stage 1 pre-training on Colab (if needed)

### Day 3: Train and Evaluate
- [ ] Run Stage 2 training on Colab
- [ ] Monitor training (1-2 hours)
- [ ] Evaluate results, compare with 20-sign baseline

### Day 4-5: Analysis
- [ ] Analyze per-class accuracy
- [ ] Test if pre-training helped (vs ablation study)
- [ ] Document findings
- [ ] Decide: Scale to 500 signs or optimize further?

---

## Long-Term Strategy

### Phase 1: Validation (Current)
- ✅ 20 signs: Complete
- 🎯 100 signs: In progress (fix notebook, train on Colab)

### Phase 2: Scaling (Next 2-4 weeks)
- Extract 500-sign features on Colab
- Train with full 3-stage curriculum
- Compare with published baselines
- Expected: 10-20% top-1 accuracy

### Phase 3: Full Implementation (Months 2-4)
- Implement Stage 3 (WFST, discourse, morphology)
- Scale to 2000-5000 signs
- Optimize for <200ms latency
- Prepare for publication

---

## Key Questions to Answer

1. **Does Stage 1 pre-training help at 100-sign scale?**
   - Ablation showed no benefit on 20 signs
   - 100 signs = fewer samples/class → might benefit
   - Test by comparing with/without pre-training

2. **What's the accuracy ceiling?**
   - 20 signs: 21.74% baseline, 26.09% with pre-training
   - 100 signs: Target 10-30% (realistic for this task)
   - 500 signs: Target 8-20% (published WLASL baselines)

3. **Is the architecture sufficient?**
   - Current: BiLSTM (128 hidden, 2 layers)
   - For 500+ signs: May need 256 hidden, 3 layers
   - Monitor params/sample ratio (<1000:1)

---

## Summary

**Your Issue**: Local training too slow (30+ hours for 100 signs)

**Solution**: Use Google Colab for all training (already set up, 15-30x faster)

**Next Step**: Fix the Colab training notebook to properly follow the paper's 3-stage curriculum, then train WLASL100 model (3-4 hours total)

**After That**: Scale to 500 signs on Colab (1-2 weeks), then implement full paper (Stage 3, WFST, etc.)

**Key Insight**: The infrastructure is ready, you just need to fix the notebook to load pre-trained weights and train properly on Colab instead of locally.
