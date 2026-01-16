# ASL Translation Framework - Implementation Guide

## 🎯 Project Overview

This repository implements the mathematical framework described in **"Mathematical Linguistics and Scalable Modeling for Real-Time ASL Translation"** by Alex Hernandez Juarez (September 2025).

**Core Innovation**: A compositional approach to ASL translation that combines:
- Phonological factorization with geometric invariance
- Spatial discourse algebra for referent tracking  
- Non-associative morphological fusion
- WFST-based decoding cascade
- Information-theoretic multimodal integration

**Goal**: Scale to 5k-10k signs with <200ms latency on edge devices while maintaining mathematical rigor.

---

## 🚧 CURRENT STATUS - WLASL100 Training on Google Colab

**Status**: Feature extraction complete, ready for training
**Date**: January 2026
**Branch**: main
**Platform**: Google Colab (GPU) for faster training

### What We've Accomplished

**Phase 1 Goal**: Prove the system works before scaling to full paper implementation. ✅ **COMPLETE**

**WLASL100 Extraction** (Google Colab): ✅ **COMPLETE**
1. ✅ **Fixed MediaPipe model download** - Corrected pose landmarker URL path
2. ✅ **Extracted features for 100 signs** - 1,077 videos processed
   - 806 training features (55.9% success rate)
   - 168 validation features (49.7% success rate)
   - 103 test features (39.9% success rate)
3. ✅ **Saved to Google Drive** - All features available at `/content/drive/MyDrive/asl_data/extracted_features/`
4. 🎯 **Next: Train on Google Colab** - Faster than laptop, ready to scale

**Previous Phase 1 Work** (20-sign dataset):
1. ✅ **Fixed infrastructure** - All type errors resolved, codebase clean
2. ✅ **Analyzed root cause** - 1.18M params vs 20 samples = 59,000:1 ratio (should be <1000:1)
3. ✅ **Cached features** - 243 train / 46 val samples (20 unique signs)
4. ✅ **Trained Phase 1** - Smaller model (~50K params) with proper data balance → 21.74% accuracy

### Current Task: Train WLASL100 Model on Google Colab

**Hyperparameters**:
```python
# Data
Train samples: 243 (was: 20)
Val samples: 46 (was: 10)
Vocab size: 21 (was: 2001)
Signs: 20 unique (was: 2)

# Model (reduce from 1.18M → 50K params)
Hidden dim: 64 (was: 128)
Layers: 1 (was: 2)
Dropout: 0.5 (was: 0.3)
Params/sample: ~200:1 (was: 59,000:1) ✓

# Training
Epochs: 30
Batch size: 8
Learning rate: 1e-3
Blank penalty: 0.1 (NEW - prevents blank collapse)
Weight decay: 1e-4 (NEW)
```

**Feature Cache Location** (Google Drive):
- `/content/drive/MyDrive/asl_data/extracted_features/features_train_wlasl100.pkl` (806 samples)
- `/content/drive/MyDrive/asl_data/extracted_features/features_val_wlasl100.pkl` (168 samples)
- `/content/drive/MyDrive/asl_data/extracted_features/features_test_wlasl100.pkl` (103 samples)
- `/content/drive/MyDrive/asl_data/extracted_features/vocabulary.json` (100 gloss vocabulary)

**Previous Phase 1 Cache** (Local):
- `data/cache/phase1/features_train_phase1.pkl` (243 samples, 20 signs)
- `data/cache/phase1/features_val_phase1.pkl` (46 samples, 20 signs)

**Training**: Use Google Colab notebook for WLASL100 (see `notebooks/`)

**Previous Phase 1 Results** (20 signs):
- Random baseline: 5%
- Achieved: 21.74% accuracy ✅
- Validated infrastructure successfully

**WLASL100 Training Goals**:
- Scale to 100 signs (5x vocabulary increase)
- Use Colab GPU for faster training
- Compare performance with 20-sign baseline

### Files Modified This Session

**Created**:
- `scripts/analyze_training.py` - Training analysis tool
- `scripts/quick_test.py` - Fast inference testing
- `scripts/train_phase1.py` - Phase 1 training script
- `TRAINING_ANALYSIS.md` - Comprehensive diagnosis report
- `checkpoints/real_mediapipe_cached/training_curves.png` - Loss visualization

**Fixed** (Type Safety - 37 diagnostics → 10):
- `src/phonology/features.py` - Float casts
- `src/phonology/quantizer.py` - Optional handling
- `src/models/encoder.py` - None assertions
- `src/models/ctc_head.py` - Return types
- `src/training/stage2_ctc.py` - Optimizer types
- `.vscode/settings.json` - Pylance config

### Next Steps (After Phase 1 Success)

**Week 2-3: Stage 1 Pre-training** (Follow Paper)
- Implement self-supervised phonological learning
- Train quantizer on unlabeled videos
- Use learned features for better CTC

**Week 4+: Scale to Production**
- Scale to 500-2000 signs
- Implement Stage 2 (CTC with pre-trained features)
- Add Stage 3 (WFST, discourse, morphology)
- Optimize for <200ms latency

### Key Lessons Learned

1. **Data > Model Size**: 20 samples insufficient for any model
2. **Monitor Predictions**: Loss curves hide blank collapse
3. **Start Simple**: Validate on 20 classes before 2000
4. **Infrastructure First**: Fix types, caching, tools before scaling

### GPU Setup

**Local Hardware**: NVIDIA GeForce MX550 (2GB VRAM)
- Status: Working ✓ (for 20-sign dataset)
- CUDA: 12.8
- Training time (Phase 1, 20 signs): ~10-15 min/run
- Temperature: 56-60°C (comfortable)
- **Limitation**: Too slow for 100-sign dataset (requires 30+ hours)

**Google Colab GPU** (for WLASL100):
- Status: Working ✓ (for feature extraction)
- GPU: Tesla T4 or better (16GB VRAM)
- Training time: Much faster than laptop (estimated 1-2 hours for 100 signs)
- **Advantage**: Free GPU access, faster iteration

---

## 📚 Paper-to-Code Mapping

### Section 2: Phonology (`src/phonology/`)
**Mathematical Objects**:
- Phonological alphabet: $\Sigma = \Sigma_H \times \Sigma_L \times \Sigma_O \times \Sigma_M \times \Sigma_N$
- Observation map: $\phi: (\mathbb{R}^3)^m \to \mathbb{R}^k$
- Quantizer: $q: \mathbb{R}^k \to \Sigma$

**Implementation**:
- `mediapipe_extractor.py`: Raw landmark extraction → normalized features
- `features.py`: $\phi$ implementation (palm centers, normals, velocities)
- `quantizer.py`: Product VQ with learned codebooks

**Key Guarantees**:
- Proposition 1 (Noise robustness): Lipschitz $\phi$ + margin $\gamma$ → stable quantization
- Proposition 2 (Product VQ sample complexity): $\tilde{\mathcal{O}}(\sum_j \sqrt{d_j \log k_j / n})$ error

**Tests**: `tests/test_quantizer.py` validates invariance under Sim(3) transformations

---

### Section 3: Spatial Discourse (`src/spatial/`)
**Mathematical Objects**:
- Locus space: $\mathcal{L}_t = \{(r, \hat{\ell}_r) : r \in \mathcal{R}_t\}$
- Assignment: $\mathcal{A}(r, \hat{\ell})$
- Retrieval: $\mathcal{R}(g(t)) = \argmax_r p(r \mid g(t))$

**Implementation**:
- `locus.py`: Voxelized locus set, novel-locus test (Proposition 4)
- `retrieval.py`: Bayesian sensor fusion over pointing/gaze/eyebrow
- `discourse.py`: State tracking $\mathcal{L}_t$, update rules

**Key Guarantees**:
- Lemma 1 (Deterministic uniqueness): Angular separation → $|\Gamma_t| \leq 1$
- Proposition 4 (Locus test): UMP decision for assignment vs. retrieval

**Tests**: `tests/test_spatial.py` validates uniqueness bounds, retrieval accuracy

---

### Section 4: Morphology (`src/morphology/`)
**Mathematical Objects**:
- Fusion operator: $\otimes: \Sigma \times \Sigma \to \Sigma$
- Component functions: $f_H, f_L, f_O, f_M, f_N$

**Implementation**:
- `fusion.py`: Non-associative $\otimes$, gated activation $\alpha_t$
- `lookup_tables.py`: $\text{SELECT}(h, c)$, $\text{ROTATE}(o, n, a)$

**Key Guarantees**:
- Proposition 3 (Non-associativity): Role sensitivity → $(s_1 \otimes s_2) \otimes s_3 \neq s_1 \otimes (s_2 \otimes s_3)$
- Theorem 1 (Bayes risk decomposition): $\Delta A = \rho (e_{\text{fuse}}^{(0)} - e_{\text{fuse}}^{(*)})$

**Tests**: `tests/test_fusion.py` validates non-associativity, coverage, gating accuracy

---

### Section 5: WFST Decoding (`src/decoder/`)
**Mathematical Objects**:
- Cascade: $H \circ C \circ M \circ D \circ L \circ G$
- Beam search with pruning threshold $\Delta$

**Implementation**:
- `wfst.py`: FST composition, determinization, minimization
- `beam_search.py`: Time-synchronous beam decoding
- `transducers/`: Individual FST constructors

**Key Guarantees**:
- Soundness/Completeness propositions
- Complexity: $\mathcal{O}(TB\bar{d}c)$
- Lemma 2 (Pruning-loss tail): Sub-exponential score gaps

**Tests**: `tests/test_decoder.py` validates composition correctness, beam stability

---

### Section 6-7: Training (`src/training/`)
**Three-Stage Curriculum**:

**Stage 1** (`stage1_phonology.py`):
- Self-supervised pre-training on unlabeled video
- Phonological quantizer learning on Level 2 data
- Loss: $\mathcal{L}_{\text{contrast}} + \mathcal{L}_{\text{phon}}$

**Stage 2** (`stage2_ctc.py`):
- End-to-end CTC training on Level 0-2 data
- Multi-task objective: $\mathcal{L}_{\text{CTC}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{phon}} \mathcal{L}_{\text{phon}}$
- GradNorm balancing

**Stage 3** (`stage3_wfst.py`):
- Discriminative lattice training
- Discourse loss $\mathcal{L}_{\text{locus}}$, morphology loss $\mathcal{L}_{\text{morph}}$
- Fine-tune encoder + WFST integration

**Key Guarantees**:
- Theorem 2 (Joint training convergence): $\mathbb{E}[\|\nabla \mathcal{L}\|] = O(1/\sqrt{T})$

---

## 🛠️ Development Workflow

### Phase 1: Foundation Components (Weeks 1-2)
```bash
# Start with phonology - cleanest interface
cd src/phonology/
# Implement in order:
# 1. mediapipe_extractor.py (Section 2.2)
# 2. features.py (Section 2.2, equations 1-6)
# 3. quantizer.py (Section 2.3, Proposition 2)

# Validate each step
pytest tests/test_quantizer.py -v
```

**Deliverables**:
- [ ] MediaPipe integration working on sample video
- [ ] Feature extraction passes Sim(3) invariance tests
- [ ] Product VQ trained on synthetic phonological data

---

### Phase 2: Sequence Modeling (Weeks 3-4)
```bash
cd src/models/
# Implement encoder + CTC head
# 1. encoder.py (BiLSTM baseline)
# 2. ctc_head.py (CTC loss + decoding)

cd src/training/
# Stage 1 + Stage 2 training
python stage1_phonology.py --config configs/stage1.yaml
python stage2_ctc.py --config configs/stage2.yaml
```

**Deliverables**:
- [ ] Encoder trained on WLASL subset (500 signs)
- [ ] CTC decoding achieves baseline WER
- [ ] Segmentation module identifies boundaries

---

### Phase 3: Spatial Discourse (Weeks 5-6)
```bash
cd src/spatial/
# Implement locus tracking
# 1. locus.py (Section 3.3)
# 2. retrieval.py (Section 3.2)
# 3. discourse.py (Algorithm 1 in paper)

# Test on synthetic discourse examples
pytest tests/test_spatial.py -v
```

**Deliverables**:
- [ ] Locus assignment/retrieval working on annotated examples
- [ ] Bayesian fusion of pointing + gaze implemented
- [ ] Discourse state tracking passes Lemma 1 validation

---

### Phase 4: WFST Integration (Weeks 7-8)
```bash
cd src/decoder/
# Build individual transducers
cd transducers/
# Implement in order (dependencies):
# 1. L_lexicon.py (no dependencies)
# 2. G_language_model.py (no dependencies)
# 3. C_context.py (needs L)
# 4. M_morphology.py (needs L, Section 4)
# 5. D_discourse.py (needs spatial/, Section 3)
# 6. H_observation.py (needs phonology/)

# Compose and test
python wfst.py --compose-all --test-beam
```

**Deliverables**:
- [ ] Each transducer passes unit tests
- [ ] Full cascade $H \circ C \circ M \circ D \circ L \circ G$ compiles
- [ ] Beam search decodes sample sequences correctly

---

### Phase 5: End-to-End Training (Weeks 9-10)
```bash
cd src/training/
# Stage 3: Discriminative training with full pipeline
python stage3_wfst.py --config configs/stage3.yaml \
    --encoder-checkpoint stage2_best.pt \
    --wfst-path data/wfst_cache/composed.fst
```

**Deliverables**:
- [ ] Multi-task loss training stable (Theorem 2)
- [ ] Morphological gate activation accurate
- [ ] Discourse tracking integrated into decoding

---

## 🧪 Testing Strategy

### Unit Tests (per module)
```bash
# Phonology
pytest tests/test_quantizer.py::test_sim3_invariance
pytest tests/test_quantizer.py::test_margin_robustness

# Spatial
pytest tests/test_spatial.py::test_uniqueness_guarantee
pytest tests/test_spatial.py::test_locus_test

# Morphology
pytest tests/test_fusion.py::test_nonassociativity
pytest tests/test_fusion.py::test_coverage

# Decoder
pytest tests/test_decoder.py::test_soundness
pytest tests/test_decoder.py::test_beam_stability
```

### Integration Tests
```bash
# End-to-end pipeline
pytest tests/test_integration.py::test_video_to_gloss
pytest tests/test_integration.py::test_latency_budget
```

### Mathematical Validation
Each test should verify a proposition/lemma:
- **Proposition 1**: Perturb landmarks by $\varepsilon$, check $q(\phi(X')) = q(\phi(X))$
- **Lemma 1**: Sample random loci, verify $|\Gamma_t| \leq 1$ with probability $\geq 1-\alpha$
- **Proposition 3**: Construct triples, verify $(s_1 \otimes s_2) \otimes s_3 \neq s_1 \otimes (s_2 \otimes s_3)$

---

## 📊 Data Requirements

### Annotation Levels (Section 7.1)
| Level | Content | Example Dataset | Purpose |
|-------|---------|-----------------|---------|
| 0 | Gloss sequences | WLASL, PHOENIX | LM training, vocab coverage |
| 1 | Boundaries + glosses | Custom annotation | Segmentation supervision |
| 2 | Full phonology $(H,L,O,M,N)$ | Expert annotation | Quantizer training |
| 3 | Discourse (loci, referents) | Custom annotation | Spatial tracking |

### Dataset Allocation (Section 7.1)
- 60% Level 0 (gloss-only)
- 25% Level 1 (boundaries)
- 10% Level 2 (phonology)
- 5% Level 3 (discourse)

---

## 🚀 Deployment Targets

### Latency Budget (Section 8.1)
$$\text{Lat} = \text{KP} + \text{Enc} + \text{Dec} + \text{Post} < 200\,\text{ms}$$

**Breakdown**:
- MediaPipe keypoints: ~30ms
- Encoder forward pass: ~50ms
- WFST beam search: ~100ms
- Post-processing: ~20ms

### Platforms
- **Edge**: Mobile (iOS/Android), tablets
- **Web**: WebAssembly + MediaPipe.js
- **Cloud**: Batch processing

### Optimizations
- INT8 quantization for encoder
- Sparse FST representation
- Pre-compiled $C \circ M \circ D \circ L \circ G$ offline

---

## 🔬 Evaluation Metrics (Section 9)

### Recognition Accuracy
- Top-k accuracy ($k \in \{1,5,10\}$)
- Word Error Rate (WER)
- BLEU score

### Component-Specific
- Segmentation: Precision, Recall, F1 on boundaries
- Discourse: Referent retrieval accuracy
- Morphology: Fused vs. plain form accuracy

### Deployment
- Latency (ms/frame)
- Throughput (FPS)
- Memory footprint (MB)

### Vocabulary Scaling
Test at: 500, 2000, 5000, 10000 signs

---

## 🎓 Key Mathematical Results (Quick Reference)

| Result | Statement | Code Validation |
|--------|-----------|-----------------|
| **Proposition 1** | $\|\eta\| < \gamma/L \Rightarrow q(\phi(X')) = q(\phi(X))$ | `test_noise_robustness()` |
| **Lemma 1** | Angular separation $> 2\tau \Rightarrow |\Gamma_t| \leq 1$ | `test_uniqueness_bound()` |
| **Proposition 3** | Role sensitivity $\Rightarrow$ non-associative $\otimes$ | `test_nonassociativity()` |
| **Proposition 4** | $T(g) > \tau \Rightarrow$ novel locus (UMP test) | `test_locus_decision()` |
| **Theorem 1** | Oracle gating: $\Delta A = \rho (e_0 - e_*)$ | `test_fusion_gain()` |
| **Theorem 2** | Joint training: $\mathbb{E}[\|\nabla \mathcal{L}\|] = O(1/\sqrt{T})$ | `test_convergence()` |

---

## 🏗️ Implementation Philosophy

### Hybrid Approach
- **Use frameworks** (PyTorch/TensorFlow) for:
  - Neural network layers (LSTM, convolutions)
  - Automatic differentiation
  - GPU acceleration
  - Standard optimizers

- **Implement from scratch**:
  - Phonological quantizer (novel contribution)
  - Spatial discourse algebra (research novelty)
  - Morphological fusion operator (core innovation)
  - Custom WFST integration (adaptation to ASL)

### Design Principles
1. **Mathematical rigor first**: Every implementation must map to a definition/proposition
2. **Test-driven development**: Write tests before implementation
3. **Modular architecture**: Clean interfaces between components
4. **Reproducibility**: Fixed random seeds, deterministic training
5. **Documentation**: Inline equations from paper in docstrings

---

## 📖 Code Style Guide

### Docstring Format
```python
def quantize_phonology(features: np.ndarray) -> Tuple[int, int, int, int, int]:
    """Product vector quantization for phonological features.
    
    Implements Section 2.3, Equation (7).
    
    Mathematical Formulation:
        Z_t = (z^H_t, z^L_t, z^O_t, z^M_t, z^N_t) ∈ Σ
        where each z^J_t = argmin_k ||u^J_t - c^J_k||
    
    Args:
        features: Concatenated feature vector [u^H; u^L; u^O; u^M; u^N]
                  Dimensions: (10 + 6 + 6 + 9 + 5) = 36
    
    Returns:
        Tuple (z^H, z^L, z^O, z^M, z^N) with codebook indices
        
    Guarantees:
        Proposition 2 (margin robustness): If ||η|| < γ/L, output unchanged
    """
    pass
```

### Variable Naming
Follow paper notation where possible:
- `phi` for feature extractor $\phi$
- `Z_t` for quantized phonology
- `ell_hat` for locus directions $\hat{\ell}$
- `L_t` for discourse state $\mathcal{L}_t$
- `alpha_t` for morphological gate $\alpha_t$

---

## 🤝 Contributing

1. Read the paper (ASL_Modeling.tex) thoroughly
2. Identify which section/proposition you're implementing
3. Write tests first (referencing specific mathematical results)
4. Implement with inline equation comments
5. Validate against propositions/lemmas

---

## 📚 References

### Core Papers (in docs/references/)
- Liddell (2003): Spatial grammar foundations
- Stokoe (1960): ASL phonology
- Mohri et al. (2002): WFST fundamentals

### Datasets
- WLASL: https://dxli94.github.io/WLASL/
- PHOENIX: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

### Tools
- MediaPipe: https://google.github.io/mediapipe/
- OpenFST: http://www.openfst.org/

---

## 🎯 Current Status

**⚠️ SEE "IN PROGRESS" SECTION AT TOP OF FILE FOR LATEST STATUS ⚠️**

**Quick Summary** (Updated: January 2026):
- ✅ Phase: Foundation complete (MediaPipe, features, encoder, CTC)
- ✅ Infrastructure: Type-safe, GPU training, feature caching working
- ✅ Phase 1 (20 signs): Complete - 21.74% accuracy validates system
- ✅ WLASL100 feature extraction: Complete on Google Colab (1,077 videos)
- 🎯 Current: Ready to train on 100-sign dataset using Google Colab
- 📍 Next: Train Stage 2 CTC model on WLASL100, compare with 20-sign results
- 🚀 Long-term: Follow paper's 3-stage curriculum for scaling to 5k+ signs

**Training Commands**:

**Local (20 signs, Phase 1)**:
```bash
python3.11 scripts/train_phase1.py \
  --train-cache data/cache/phase1/features_train_phase1.pkl \
  --val-cache data/cache/phase1/features_val_phase1.pkl \
  --device cuda --num-epochs 30
```

**Google Colab (100 signs, WLASL100)**:
- Use `notebooks/WLASL100_Training_Colab.ipynb` (to be created)
- Features available at: `/content/drive/MyDrive/asl_data/extracted_features/`

**Long-term Goal**: Full system demonstrating <200ms latency at 5k+ vocabulary