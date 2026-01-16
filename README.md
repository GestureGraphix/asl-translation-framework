# ASL Translation Framework

**Mathematical Linguistics and Scalable Modeling for Real-Time ASL Translation**

> Implementation of the framework described in the paper by Alex Hernandez Juarez (September 2025)


---

## Overview

This project implements a **compositional, mathematically grounded** approach to American Sign Language (ASL) translation that:

- **Factorizes signs** into phonological primitives with provable invariance guarantees
- **Tracks spatial discourse** using formal locus algebra with uniqueness bounds  
- **Fuses morphology** via non-associative operators for classifier constructions
- **Decodes efficiently** using WFST composition with <200ms latency
- **Scales to 5k-10k signs** while maintaining sample efficiency

### Key Innovation

Rather than treating signs as atomic units, we decompose them into:
```
Sign = (Handshape × Location × Orientation × Movement × Non-manuals)
```
with each component learned via vector quantization over geometric invariants.

This enables **compositional generalization** - recognizing novel sign combinations without retraining.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/asl-translation-framework.git
cd asl-translation-framework

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Test MediaPipe Extraction

```python
from src.phonology.mediapipe_extractor import MediaPipeExtractor

# Initialize extractor
extractor = MediaPipeExtractor()

# Process video
landmarks = extractor.extract_video("path/to/asl_video.mp4")

print(f"Extracted {len(landmarks)} frames")
```

### Extract Phonological Features

```python
from src.phonology.features import FeatureExtractor

feature_extractor = FeatureExtractor()

# Extract features from landmarks
features = feature_extractor.extract_features(landmarks[0])

print(f"Feature vector shape: {features.concatenate().shape}")  # (36,)
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_quantizer.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Project Structure

```
asl-translation-framework/
├── README.md                          # This file
├── requirements.txt
│
├── docs/                              # All documentation (organized by category)
│   ├── guides/                        # How-to guides and tutorials
│   ├── status/                        # Status tracking and progress
│   ├── planning/                      # Planning documents and roadmaps
│   ├── reference/                     # Reference documentation (includes CLAUDE.md)
│   ├── paper/                         # Research paper (ASL Modeling.tex)
│   └── README.md                      # Documentation index
│
├── src/                               # Source code
│   ├── phonology/                     # Section 2: Feature extraction
│   │   ├── mediapipe_extractor.py    # ✅ MediaPipe integration
│   │   ├── features.py               # ✅ Sim(3) normalization
│   │   └── quantizer.py              # ⏳ Product VQ (TODO)
│   ├── spatial/                       # Section 3: Discourse tracking
│   │   ├── locus.py                  # ⏳ Locus assignment (TODO)
│   │   └── retrieval.py              # ⏳ Bayesian fusion (TODO)
│   ├── morphology/                    # Section 4: Fusion operator
│   │   └── fusion.py                 # ⏳ Non-associative ⊗ (TODO)
│   ├── decoder/                       # Section 5: WFST decoding
│   │   └── wfst.py                   # ⏳ FST composition (TODO)
│   └── training/                      # Section 7: Training pipeline
│       ├── stage1_phonology.py       # ⏳ Pre-training (TODO)
│       ├── stage2_ctc.py             # ⏳ CTC training (TODO)
│       └── stage3_wfst.py            # ⏳ Fine-tuning (TODO)
│
├── tests/                             # Unit tests
│   └── test_quantizer.py             # ✅ Phonology tests
│
└── data/                              # Datasets
    ├── raw/                           # Original videos
    ├── processed/                     # Extracted features
    └── annotations/                   # Level 0-3 labels
```

**Legend**: ✅ Implemented | ⏳ TODO | 📝 Design

---

## Implementation Status

### Phase 1: Foundation (Current)
- [x] MediaPipe landmark extraction
- [x] Sim(3) normalization with invariance tests
- [x] Geometric feature extraction (36-dim vectors)
- [ ] Product vector quantization (in progress)
- [ ] Phonological codebook training

### Phase 2: Sequence Modeling
- [ ] BiLSTM encoder implementation
- [ ] CTC loss and decoding
- [ ] Boundary detection module
- [ ] Stage 1-2 training scripts

### Phase 3: Spatial Discourse
- [ ] Locus tracking state machine
- [ ] Bayesian retrieval with sensor fusion
- [ ] Discourse transducer $D$

### Phase 4: WFST Decoding
- [ ] Individual transducers $(H, C, M, D, L, G)$
- [ ] FST composition and determinization
- [ ] Beam search decoder

### Phase 5: End-to-End Training
- [ ] Stage 3 discriminative training
- [ ] Multi-task loss balancing
- [ ] Full pipeline integration

---

## Mathematical Foundations

### Core Results

| Proposition | Statement | Implementation | Test |
|------------|-----------|----------------|------|
| **Prop 1** | Noise robustness: $\\|\eta\\| < \gamma/L \Rightarrow q(\phi(X')) = q(\phi(X))$ | `features.py` | `test_noise_robustness()` |
| **Lemma 1** | Uniqueness: $\angle(\hat{\ell}_1, \hat{\ell}_2) > 2\tau \Rightarrow \|\Gamma_t\| \leq 1$ | `locus.py` | `test_uniqueness_bound()` |
| **Prop 3** | Non-associativity: Role sensitivity $\Rightarrow (s_1 \otimes s_2) \otimes s_3 \neq s_1 \otimes (s_2 \otimes s_3)$ | `fusion.py` | `test_nonassociativity()` |
| **Thm 1** | Risk decomposition: $\Delta A = \rho(e_0 - e_*)$ | `fusion.py` | `test_fusion_gain()` |
| **Thm 2** | Convergence: $\mathbb{E}[\|\nabla \mathcal{L}\|] = O(1/\sqrt{T})$ | `stage3_wfst.py` | `test_convergence()` |

### Key Equations

**Sim(3) Normalization** (Section 2.2):
```math
X̃_t = (X_t - T_t) R_t^T / s_t
```
where $s_t = \\|B_t[\text{RS}] - B_t[\text{LS}]\\|$ (shoulder width)

**Phonological Features** (Section 2.2):
```math
\begin{align}
c^L_t &= \frac{1}{5} \sum_{j \in \{0,5,9,13,17\}} L̃_t[j] \\
n^L_t &= \frac{(L̃_t[5] - L̃_t[0]) \times (L̃_t[17] - L̃_t[0])}{\\|·\\|}
\end{align}
```

**WFST Cascade** (Section 5):
```math
H \circ C \circ M \circ D \circ L \circ G
```

---

## Datasets

### Required Annotations

| Level | Content | Purpose | Allocation |
|-------|---------|---------|-----------|
| 0 | Gloss sequences | Language model, vocabulary | 60% |
| 1 | Boundaries + glosses | Segmentation supervision | 25% |
| 2 | Full phonology $(H,L,O,M,N)$ | Quantizer training | 10% |
| 3 | Discourse (loci, referents) | Spatial tracking | 5% |

### Supported Datasets

- **WLASL**: 2000 glosses, 21K videos (gloss-level)
- **PHOENIX**: 1200 glosses, continuous signing
- **Custom annotations**: Add to `data/annotations/`

---

## Development Workflow

### 1. Start with Documentation
- **Getting Started**: [Quick Start Guide](docs/guides/QUICK_START_COLAB.md) for Colab training
- **Implementation Reference**: [CLAUDE.md](docs/reference/CLAUDE.md) - Complete implementation guide
- **Current Status**: [Status](docs/status/STATUS.md) - Latest project status
- **Documentation Index**: [docs/README.md](docs/README.md) - All documentation organized

### 2. Implement a Module
```bash
# Create implementation
vim src/phonology/quantizer.py

# Write tests FIRST
vim tests/test_quantizer.py

# Run tests
pytest tests/test_quantizer.py -v
```

### 3. Validate Mathematical Properties
Each module must pass proposition/lemma tests:
```bash
# Example: Validate Sim(3) invariance
pytest tests/test_quantizer.py::test_sim3_invariance -v
```

### 4. Profile Performance
```bash
# Check latency budget (<200ms target)
python -m cProfile -o profile.stats src/pipeline.py
python -m pstats profile.stats
```

---

## Latency Budget

**Target**: <200ms end-to-end (Section 8.1)

| Component | Budget | Current | Status |
|-----------|--------|---------|--------|
| MediaPipe | 30ms | TBD | ⏳ |
| Feature extraction $\phi$ | 10ms | TBD | ⏳ |
| Quantization $q$ | 5ms | TBD | ⏳ |
| Encoder | 50ms | TBD | ⏳ |
| WFST beam search | 100ms | TBD | ⏳ |
| Post-processing | 20ms | TBD | ⏳ |
| **Total** | **<200ms** | **TBD** | ⏳ |

---

## Contributing

### Code Style
- **Black** for Python formatting
- **Type hints** on all functions
- **Docstrings** cite paper sections/equations
- **Tests** validate propositions/lemmas

### Pull Request Checklist
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Mathematical validation tests included
- [ ] Docstrings reference paper equations
- [ ] Performance profiling shows no bottlenecks
- [ ] `docs/implementation_notes.md` updated

---

## Citation

If you use this work, please cite:

```bibtex
@article{hernandez2025asl,
  title={Mathematical Linguistics and Scalable Modeling for Real-Time ASL Translation},
  author={Hernandez Juarez, Alex},
  year={2025},
  month={September}
}
```

---

## Ethical Considerations

**This is a technical framework requiring validation by the Deaf community.**

### Key Principles
- **Nothing about us without us**: Deaf researchers must be involved in all stages
- **Linguistic sovereignty**: ASL has its own grammar, not "visual English"  
- **Regional variation**: Support multiple ASL dialects
- **Consent & privacy**: On-device processing, informed consent for data

### Inappropriate Uses
❌ Replacing human interpreters in medical/legal settings  
❌ Employment screening or evaluation  
❌ Surveillance without consent  

### Appropriate Uses
✅ ASL learning tools (with pedagogical design)  
✅ Accessibility features (user-controlled, opt-in)  
✅ Linguistic research (with community partnership)  

See Section 10.3 of the paper for full discussion.

---

## License

MIT License - see LICENSE file for details.

**Note**: This project is for research purposes. Production deployment requires extensive validation with the Deaf community.

---

## Contact

- **Author**: Alex Hernandez Juarez
- **Advisor**: Professor Raja Kushalnagar
- **Issues**: [GitHub Issues](https://github.com/yourusername/asl-translation-framework/issues)

---

## Acknowledgments

This work builds on:
- Stokoe (1960): ASL phonology foundations
- Liddell (2003): Spatial grammar theory
- Mohri et al. (2002): WFST decoding frameworks

**Community Feedback**: We welcome feedback from ASL linguists, Deaf researchers, and native signers to refine this mathematical formalization.

---

**Last Updated**: December 24, 2025  
**Status**: Foundation phase - phonology module implemented