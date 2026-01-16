# Paper Alignment Check - Verifying Our Approach

**Paper Reference**: Section 7.2 (Three-Stage Training Pipeline)  
**Date**: January 2026  
**Goal**: Verify our implementation aligns with paper's theoretical framework

---

## 📄 Paper's Three-Stage Training Strategy (Section 7.2)

### Stage 1: Phonological Pre-training

**Paper says** (Section 7.2.1):

> **Objective**: Learn robust phonological quantizers $q$ and encoder $\phi$ from Level 2 data plus self-supervised losses on unlabeled video.

**Components**:

1. **Self-Supervised Pre-training** (Eq 36):
   - Contrastive learning: $\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+\delta})/\tau)}{\sum_{t'} \exp(\text{sim}(z_t, z_{t'})/\tau)}$
   - Uses temporal coherence ($\delta \in [1,5]$)
   - Trains on **unlabeled video corpus** $\mathcal{V}_{\text{unlabeled}}$

2. **Phonological Quantizer Training** (Eq 37):
   - On Level 2 annotated data $\mathcal{D}_2 = \{(V_i, \{Z^*_t\})\}$
   - Loss: $\mathcal{L}_{\text{phon}} = \sum_{t,J} \text{CE}(q_J(\phi_J(X_t)), Z^{*J}_t) + \beta \|\phi(X_t) - \text{sg}(e_{Z^*_t})\|^2$
   - Per phonological component $J \in \{H,L,O,M,N\}$

**Total Loss**: $\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{contrast}} + \mathcal{L}_{\text{phon}}$

**Output**: Pre-trained encoder weights and quantizer codebooks

---

### Stage 2: End-to-End CTC Training

**Paper says** (Section 7.2.2):

> **Objective**: Train encoder + CTC head to predict gloss sequences, jointly optimizing phonological features and sequence modeling.

**Key Requirements**:

1. **Uses Stage 1 pre-trained encoder** (implicit - builds on Stage 1)
2. **Multi-Level Training** (Eq 38):
   $$
   \mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{phon}} \mathcal{L}_{\text{phon}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}
   $$
   - $\mathcal{L}_{\text{CTC}} = -\log p(y \mid Z_{1:T})$ (gloss prediction)
   - $\mathcal{L}_{\text{seg}}$ (boundary supervision from Level 1)
   - $\mathcal{L}_{\text{phon}}$ (phonological supervision from Level 2)
   - $\mathcal{L}_{\text{reg}}$ (regularization)

3. **Data**: Combines Level 0 (gloss-only), Level 1 (boundaries), Level 2 (phonology)

---

### Stage 3: WFST Fine-tuning

**Paper says** (Section 7.2.3):

> Integrate trained encoder with WFST decoder $H\circ C\circ M\circ D\circ L\circ G$, fine-tune with discriminative training and discourse losses.

**Current Status**: Not yet implemented (future work)

---

## ✅ What We're Doing vs Paper

### Stage 1 Implementation

| Paper Requirement | Our Implementation | Status |
|-------------------|-------------------|--------|
| Contrastive learning on unlabeled video | ✅ Implemented (`src/phonology/contrastive_loss.py`) | ✅ |
| Phonological quantizer training | ✅ Implemented (`src/phonology/vq_loss.py`) | ✅ |
| Loss: $\mathcal{L}_{\text{contrast}} + \mathcal{L}_{\text{phon}}$ | ✅ Implemented (`src/training/stage1_phonology.py`) | ✅ |
| Pre-trained encoder weights saved | ✅ Checkpoint at `checkpoints/stage1/checkpoint_best.pt` | ✅ |

**Status**: ✅ **ALIGNED** - Stage 1 correctly implemented

---

### Stage 2 Implementation

| Paper Requirement | Our Implementation | Status |
|-------------------|-------------------|--------|
| **Load Stage 1 pre-trained encoder** | ⚠️ **Code exists but checkpoint may not be loaded** | ⚠️ |
| CTC loss for gloss prediction | ✅ Implemented (`src/models/ctc_head.py`) | ✅ |
| Multi-task loss ($\mathcal{L}_{\text{CTC}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{phon}} \mathcal{L}_{\text{phon}}$) | ❌ Currently only $\mathcal{L}_{\text{CTC}}$ | ❌ |
| Uses Level 0-2 data | ✅ Using gloss-labeled data (Level 0 equivalent) | ✅ |
| Blank penalty for CTC | ✅ Implemented (not in paper but standard practice) | ✅ |

**Status**: ⚠️ **PARTIALLY ALIGNED** - Missing:
1. Stage 1 checkpoint loading (critical!)
2. Multi-task loss components (optional for now)

---

## 🔍 Critical Alignment Issue

### Problem: Stage 1→2 Transfer

**Paper's Expectation**:
- Stage 1 learns robust phonological features from unlabeled + Level 2 data
- Stage 2 **builds on** Stage 1 by fine-tuning pre-trained encoder
- Without Stage 1, Stage 2 trains from random initialization (not paper's approach)

**Our Current Situation**:
- ✅ Stage 1 trained successfully (loss: 2.03)
- ⚠️ Stage 2 may not be loading Stage 1 checkpoint
- ❌ Training from random initialization = **NOT following paper**

**Evidence of Issue**:
- Very poor results (0.60% accuracy)
- Severe overfitting (train loss 0.48 vs val loss 11.38)
- Model not learning (accuracy worse than random in some cases)

**This violates the paper's curriculum!**

---

## ✅ What We're Doing Correctly

1. **Feature Extraction** (Section 2.2):
   - ✅ MediaPipe-based landmarks
   - ✅ Sim(3) normalization
   - ✅ 36D phonological features (matches paper's formulation)

2. **Model Architecture**:
   - ✅ BiLSTM encoder (lightweight, suitable for edge deployment)
   - ✅ CTC head for sequence prediction
   - ✅ Appropriate for paper's goals

3. **Training Infrastructure**:
   - ✅ Three-stage pipeline structure
   - ✅ Checkpoint saving/loading
   - ✅ Colab workflow for scalability

4. **Curriculum Design**:
   - ✅ Stage 1 before Stage 2 (correct order)
   - ✅ Code supports loading pre-trained weights

---

## ❌ What's Missing/Incorrect

### Critical (Must Fix):

1. **Stage 1 Checkpoint Not Loaded** ❌
   - **Impact**: Training from random init, not following paper
   - **Fix**: Upload checkpoint to Colab, verify loading

2. **No Verification of Checkpoint Loading** ❌
   - **Impact**: Unsure if Stage 1 weights actually used
   - **Fix**: Check notebook output from Cell 12

### Important (Should Fix for Full Alignment):

3. **Missing Multi-Task Loss** ⚠️
   - **Impact**: Not using full Stage 2 loss from paper (Eq 38)
   - **Current**: Only $\mathcal{L}_{\text{CTC}}$
   - **Should have**: $\mathcal{L}_{\text{CTC}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{phon}} \mathcal{L}_{\text{phon}}$
   - **Note**: Paper says this is for Level 0-2 data. For Level 0 only (gloss labels), $\mathcal{L}_{\text{CTC}}$ alone is reasonable

4. **No Level 1/2 Data** ⚠️
   - **Impact**: Can't use segmentation/phonological supervision
   - **Current**: Using only gloss labels (Level 0 equivalent)
   - **Note**: Acceptable for validation phase, but not full paper implementation

---

## 🎯 Paper's Goals vs Our Progress

### Paper's Ultimate Goal (Introduction):

> Scale to **5,000-10,000 signs** with **<200ms latency** on edge devices

### Our Progress:

| Component | Paper Goal | Our Status | Alignment |
|-----------|------------|------------|-----------|
| Vocabulary | 5k-10k signs | 100 signs (1-2% of goal) | 🔜 On track |
| Training Strategy | 3-stage curriculum | ⚠️ Stage 1→2 (incomplete) | ⚠️ Needs fix |
| Feature Extraction | Section 2.2 | ✅ Implemented | ✅ Aligned |
| Spatial Discourse | Section 3 | ❌ Not implemented | 🔜 Future |
| Morphology | Section 4 | ❌ Not implemented | 🔜 Future |
| WFST Decoder | Section 5 | ❌ Not implemented | 🔜 Future |
| Latency | <200ms | ❌ Not tested | 🔜 Future |

**Assessment**: ✅ **Structure aligned**, ⚠️ **Execution needs Stage 1 checkpoint loading**

---

## 📋 Verification Checklist

### Must Verify (Critical):

- [ ] **Stage 1 checkpoint uploaded to Google Drive**
  - Path: `/content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt`
  - **Action**: Check if file exists in Drive

- [ ] **Stage 1 checkpoint loaded in notebook**
  - **Action**: Check Cell 12 output - should say "USING PRE-TRAINED"
  - If says "RANDOM INIT" → **NOT following paper!**

- [ ] **Encoder initialized with pre-trained weights**
  - **Action**: Check Cell 13 output - should confirm weights loaded
  - If not → retrain with checkpoint

### Should Verify (Important):

- [ ] **Model architecture matches Stage 1**
  - Hidden dim: 128 (should match Stage 1)
  - Layers: 2 (should match Stage 1)
  - **If mismatch**: Weights won't load correctly

- [ ] **Training loss makes sense**
  - Should start lower with pre-training (vs random init)
  - **If training from scratch**: Loss starts high, learns slowly

---

## 🎓 Paper's Philosophy vs Our Approach

### Paper's Core Philosophy:

> "Mathematical linguistics core that (i) formalizes signs and spatial reference, (ii) enforces well-formedness with automata/grammars, and (iii) composes these with efficient decoders."

**Our Alignment**:
- ✅ (i) Phonological formalization (Section 2)
- ⏳ (ii) Automata/grammars (WFST - Stage 3, future)
- ⏳ (iii) Efficient decoders (WFST - Stage 3, future)

**Current Focus**: Validating foundation (phonological features) before building full system

### Paper's Training Strategy:

> "Three-stage training strategy designed for practical implementation"

**Our Alignment**:
- ✅ Stage 1 structure correct
- ⚠️ Stage 2 structure correct, but execution may skip Stage 1
- 🔜 Stage 3 future work

---

## ✅ Conclusion: Are We Following the Paper?

### Current Status: **MOSTLY YES, BUT CRITICAL ISSUE**

**What's Correct** ✅:
1. Stage 1 implementation aligns with paper
2. Stage 2 structure aligns with paper
3. Feature extraction matches paper's Section 2.2
4. Model architecture appropriate for paper's goals

**What's Wrong** ❌:
1. **Stage 1 checkpoint may not be loaded** (CRITICAL)
2. Multi-task loss simplified (acceptable for Level 0 data)
3. Missing Level 1/2 annotations (acceptable for validation)

**Critical Fix Needed**:
- **Verify Stage 1 checkpoint is loaded** before Stage 2 training
- If not loaded, upload it and retrain
- This is the **most important alignment issue**

**Impact**:
- Without Stage 1: **NOT following paper's curriculum**
- With Stage 1: **Following paper's approach** ✅

---

## 🔧 Immediate Action

1. **Check Cell 12 output** in notebook
   - Look for: "USING PRE-TRAINED" or "RANDOM INIT"
   - If "RANDOM INIT": Upload Stage 1 checkpoint and retrain

2. **Verify checkpoint path**:
   ```python
   stage1_path = Path('/content/drive/MyDrive/asl_data/checkpoints/stage1/checkpoint_best.pt')
   print(f"Checkpoint exists: {stage1_path.exists()}")
   ```

3. **If checkpoint missing**: 
   - Upload from local: `checkpoints/stage1/checkpoint_best.pt`
   - Re-run Cell 12 (Stage 1 loading)
   - Re-run training

**Bottom Line**: The paper's approach requires Stage 1→2 transfer. Without it, we're not following the curriculum. This likely explains poor results.
