# Implementation Notes

## Current Status: Foundation Phase

**Date**: December 24, 2025  
**Goal**: Build and validate phonological quantization pipeline (Section 2)

---

## Immediate Next Steps (This Week)

### Step 1: Environment Setup
```bash
# Create project structure
mkdir -p src/{phonology,spatial,morphology,decoder,training,models,utils}
mkdir -p tests data/{raw,processed,annotations,codebooks,wfst_cache}
mkdir -p configs notebooks docs/references

# Install dependencies
pip install torch torchvision mediapipe numpy scipy
pip install pytest tensorboard wandb  # for tracking
pip install openfst-python  # for WFST work later
```

### Step 2: MediaPipe Prototype (Today/Tomorrow)
**File**: `notebooks/prototype_phonology.ipynb`

Goal: Verify MediaPipe works on sample ASL video

```python
import mediapipe as mp
import cv2
import numpy as np

# Test on single video frame
mp_holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Extract landmarks: L_t (21×3), R_t (21×3), F_t (468×3), B_t (33×3)
# Verify we can recover all 1623 landmarks
```

**Success Criteria**:
- [ ] Landmarks extracted from test video
- [ ] Coordinate system understood (world vs. image coordinates)
- [ ] Missing landmark handling strategy defined

---

### Step 3: Sim(3) Normalization Implementation
**File**: `src/phonology/features.py`

Implement equations from Section 2.2:

```python
def normalize_sim3(L_t, R_t, F_t, B_t):
    """
    Normalize landmarks to Sim(3)-invariant frame.
    
    Section 2.2, Equation (3):
        X̃_t = (X_t - T_t) R_t^T / s_t
    
    Where:
        s_t = ||B_t[RS] - B_t[LS]||  (shoulder width)
        T_t = B_t[NE]                 (neck position)
        R_t = yaw rotation matrix     (align shoulders to x-axis)
    """
    # Extract key body points
    right_shoulder = B_t[11]  # MediaPipe pose landmark index
    left_shoulder = B_t[12]
    neck = B_t[0]  # or average of shoulders
    
    # Compute scale
    s_t = np.linalg.norm(right_shoulder - left_shoulder)
    
    # Compute rotation to align shoulder line with x-axis
    shoulder_vec = right_shoulder - left_shoulder
    yaw_angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    R_t = rotation_matrix_z(-yaw_angle)  # rotate to canonical orientation
    
    # Apply transformation to all landmarks
    X_t_normalized = ...
    
    return X_t_normalized
```

**Tests**: `tests/test_quantizer.py::test_sim3_invariance`

```python
def test_sim3_invariance():
    """Validate Proposition 1: G-invariance of q∘φ"""
    # Generate synthetic landmarks
    X = generate_synthetic_hand()
    
    # Apply random Sim(3) transformations
    transformations = [
        translate(X, [1, 2, 3]),
        rotate(X, axis='z', angle=30),
        scale(X, factor=1.5),
        compose(translate, rotate, scale)
    ]
    
    phi = FeatureExtractor()
    q = ProductVectorQuantizer()
    
    for g in transformations:
        X_transformed = g(X)
        assert q(phi(X)) == q(phi(X_transformed)), \
            f"Quantization not invariant under {g}"
```

---

## Design Decisions Log

### Decision 1: Deep Learning Framework
**Date**: Dec 24, 2025  
**Choice**: PyTorch  
**Rationale**:
- Better debugging experience (eager execution)
- Stronger community support for research projects
- Easier integration with MediaPipe (NumPy interface)
- TorchScript for deployment optimization

**Implications**:
- All encoder/decoder models in PyTorch
- WFST integration will be Python bindings to OpenFST (C++)
- Training loops use PyTorch Lightning for structure

---

### Decision 2: Phonological Codebook Sizes
**Date**: Dec 24, 2025  
**Choice**: $(64, 128, 32, 64, 32)$ for $(H, L, O, M, N)$  
**Rationale**:
- From Section 2.3: "codebooks of sizes (64,128,32,64,32)"
- Location needs highest resolution (128) for spatial precision
- Movement can be coarser (64) due to temporal smoothing
- Non-manuals lowest (32) - binary/ternary features

**Validation Strategy**:
- Train on increasing codebook sizes: {16, 32, 64, 128}
- Measure reconstruction error vs. downstream WER
- Select knee of curve (likely paper values are optimal)

---

### Decision 3: MediaPipe Model Selection
**Date**: Dec 24, 2025  
**Choice**: Holistic model (not separate hand/pose/face)  
**Rationale**:
- Single forward pass (faster)
- Temporally coherent across modalities
- Aligned coordinate systems

**Trade-off**:
- Slightly lower hand accuracy vs. dedicated hand model
- Acceptable given invariance from Proposition 1

---

## Mathematical Validation Checklist

As we implement each module, we must validate the corresponding mathematical claims:

### Phonology Module
- [ ] **Proposition 1** (Noise robustness): Add Gaussian noise $\eta$ with $\|\eta\| < \gamma/L$, verify quantization unchanged
- [ ] **Proposition 2** (Product VQ sample complexity): Train on varying $n$, plot convergence rate vs. $\sqrt{d_j \log k_j / n}$
- [ ] **Proposition 3** (Joint margin robustness): Test with multiple perturbation magnitudes, verify threshold behavior

### Spatial Module (Phase 3)
- [ ] **Lemma 1** (Deterministic uniqueness): Sample loci uniformly on sphere, count cases where $|\Gamma_t| > 1$
- [ ] **Probabilistic uniqueness**: Monte Carlo with varying $m$, $\tau$, verify bound holds
- [ ] **Proposition 4** (Locus test): ROC curve for assignment vs. retrieval under controlled $\tau$

### Morphology Module (Phase 4)
- [ ] **Proposition 3** (Non-associativity): Construct explicit counter-examples
- [ ] **Proposition 5** (Fusion coverage): Enumerate $(v,n)$ pairs, verify all have defined $\otimes$
- [ ] **Theorem 1** (Bayes risk decomposition): Simulate with oracle vs. learned gating, measure $\Delta A$

### Decoder Module (Phase 4)
- [ ] **Soundness**: Generate random paths, verify all accepted outputs are well-formed
- [ ] **Completeness**: Generate valid $(z,v)$ pairs, verify paths exist
- [ ] **Lemma 2** (Pruning-loss tail): Measure score gaps empirically, compare to tail bound

### Training Module (Phase 5)
- [ ] **Theorem 2** (Convergence): Track $\|\nabla \mathcal{L}\|$ during training, verify $O(1/\sqrt{T})$ rate

---

## Open Questions / Research Decisions

### Q1: Temporal Feature Window
**Paper**: "Define $\Delta f_t = f_t - f_{t-1}$, $\Delta^2 f_t = \Delta f_t - \Delta f_{t-1}$"  
**Question**: What window size for velocity/acceleration? 1 frame, 3 frames, 5 frames?

**Experiments Needed**:
- Test $\{1, 3, 5, 7\}$ frame windows
- Measure: phonological stability, boundary detection precision
- Likely answer: 3 frames (trade-off responsiveness vs. noise)

---

### Q2: CTC Blank Symbol Strategy
**Paper**: "The blank symbol $\epsilon$ absorbs transitions and coarticulation"  
**Question**: Should blank be phonologically null or a learned "transition" class?

**Options**:
1. Standard CTC blank (no emission)
2. Learned transition codebook (captures coarticulation explicitly)

**Decision**: Start with (1), add (2) if coarticulation errors persist

---

### Q3: Discourse State Representation
**Paper**: "$D$ tracks locus assignments and retrieval"  
**Question**: How to encode $\mathcal{L}_t$ as FST states?

**Challenge**: Infinite possible locus sets → can't enumerate states

**Solution**: 
- Discretize space into $N$ voxels (e.g., $N=27$ for 3×3×3 grid)
- Each FST state represents active voxel set (bit vector)
- Limit to $K$ simultaneous referents (e.g., $K=4$)
- State space: $\binom{N}{K} \approx 17{,}550$ for $N=27, K=4$

---

## Code Review Checklist (Before Merging)

For each module, verify:

- [ ] **Docstrings** cite specific paper sections/equations
- [ ] **Type hints** on all functions
- [ ] **Unit tests** pass with 100% coverage
- [ ] **Mathematical validation** tests pass (Propositions/Lemmas)
- [ ] **Integration tests** work with upstream/downstream modules
- [ ] **Performance profiling** shows no obvious bottlenecks
- [ ] **Memory profiling** shows no leaks
- [ ] **Documentation** updated in `docs/architecture.md`

---

## Performance Benchmarks (Target vs. Actual)

| Component | Target Latency | Actual | Status |
|-----------|----------------|--------|--------|
| MediaPipe extraction | 30ms | TBD | ⏳ |
| Feature computation $\phi$ | 10ms | TBD | ⏳ |
| Quantization $q$ | 5ms | TBD | ⏳ |
| Encoder forward pass | 50ms | TBD | ⏳ |
| WFST beam search (B=32) | 100ms | TBD | ⏳ |
| Post-processing | 20ms | TBD | ⏳ |
| **Total** | **<200ms** | **TBD** | ⏳ |

---

## Next Milestones

### Milestone 1: Phonology Complete (Week 2)
- MediaPipe integration working
- Feature extractor passes invariance tests
- Product VQ trained on synthetic data
- **Deliverable**: Can convert video → phonological sequence $Z_{1:T}$

### Milestone 2: CTC Baseline (Week 4)
- Encoder trained on WLASL subset
- CTC decoding achieves >50% top-5 accuracy on 500 signs
- **Deliverable**: End-to-end video → gloss pipeline (no WFST yet)

### Milestone 3: Spatial Discourse (Week 6)
- Locus tracking working on synthetic examples
- Bayesian retrieval implemented
- **Deliverable**: Demo of pronoun resolution in controlled examples

### Milestone 4: WFST Integration (Week 8)
- All transducers implemented and composed
- Beam search running
- **Deliverable**: Full $H \circ C \circ M \circ D \circ L \circ G$ pipeline

### Milestone 5: End-to-End Training (Week 10)
- Stage 3 discriminative training stable
- Multi-task losses balanced
- **Deliverable**: System achieving competitive WER on PHOENIX benchmark

---

## Resources & References

### Immediate Reading (Next 2 Days)
1. MediaPipe Holistic docs: https://google.github.io/mediapipe/solutions/holistic.html
2. PyTorch VQ-VAE tutorial: https://pytorch.org/tutorials/intermediate/vq_vae_tutorial.html
3. CTC loss explanation: https://distill.pub/2017/ctc/

### For Later Phases
- OpenFST tutorial: http://www.openfst.org/twiki/bin/view/FST/FstQuickTour
- Kaldi WFST decoding: https://kaldi-asr.org/doc/graph.html
- GradNorm paper: https://arxiv.org/abs/1711.02257

---

## Notes to Self

- **Keep equations in comments**: Future you will thank current you
- **Test mathematical claims immediately**: Don't accumulate "TODO: validate Prop X"
- **Profile early**: Latency budget is tight, no room for surprises later
- **Document design choices**: Why PyTorch? Why these codebook sizes? Write it down!
- **Ask for help on discourse FST**: State space explosion is subtle, might need advice

---

## Contact / Collaboration

- **Primary contact**: Alex Hernandez Juarez
- **Advisor**: Professor Raja Kushalnagar (feedback on mathematical formalization)
- **Community engagement**: Plan outreach to Deaf community for validation (see Section 10.3 of paper)

---

**Last Updated**: December 24, 2025  
**Next Review**: After Milestone 1 completion