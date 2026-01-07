"""
Product Vector Quantization for ASL Phonology
Section 2.3: "Quantization and the Phonological Alphabet"

Implements the quantizer q: ℝ^36 → Σ where Σ = Σ_H × Σ_L × Σ_O × Σ_M × Σ_N

Key components:
    - Product structure: Quantize each phonological component separately
    - Codebook learning: K-means clustering on training data
    - Margin-based robustness: Proposition 1 (noise tolerance)

Mathematical formulation (Section 2.3):
    φ(X_t) = [u^H; u^L; u^O; u^M; u^N] ∈ ℝ^36

    For each component J ∈ {H, L, O, M, N}:
        z^J_t = argmin_k ||u^J_t - c^J_k||_2

    Output: Z_t = (z^H_t, z^L_t, z^O_t, z^M_t, z^N_t) ∈ Σ

Proposition 2 (Sample complexity):
    Learning error bounded by O(Σ_j √(d_j log k_j / n))
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import pickle


@dataclass
class CodebookConfig:
    """Configuration for phonological component codebooks."""
    # Codebook sizes (k_j for each component)
    k_handshape: int = 64      # |Σ_H| = 64 (Section 2.3)
    k_location: int = 32       # |Σ_L| = 32
    k_orientation: int = 32    # |Σ_O| = 32
    k_movement: int = 16       # |Σ_M| = 16
    k_nonmanual: int = 16      # |Σ_N| = 16

    # Feature dimensions (d_j for each component)
    d_handshape: int = 10
    d_location: int = 6
    d_orientation: int = 6
    d_movement: int = 9
    d_nonmanual: int = 5

    # Training parameters
    max_iter: int = 100        # K-means iterations
    n_init: int = 10           # K-means random initializations
    random_state: int = 42

    def total_alphabet_size(self) -> int:
        """Total size of product alphabet |Σ| = Π |Σ_j|."""
        return (self.k_handshape * self.k_location * self.k_orientation *
                self.k_movement * self.k_nonmanual)

    def get_component_sizes(self) -> Dict[str, int]:
        """Get codebook sizes as dictionary."""
        return {
            'handshape': self.k_handshape,
            'location': self.k_location,
            'orientation': self.k_orientation,
            'movement': self.k_movement,
            'nonmanual': self.k_nonmanual,
        }

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions as dictionary."""
        return {
            'handshape': self.d_handshape,
            'location': self.d_location,
            'orientation': self.d_orientation,
            'movement': self.d_movement,
            'nonmanual': self.d_nonmanual,
        }


@dataclass
class PhonologicalCode:
    """
    Discrete phonological code Z_t ∈ Σ.

    Represents a quantized phonological observation at time t.
    Each component is an integer index into the corresponding codebook.
    """
    handshape: int      # z^H_t ∈ {0, ..., k_H-1}
    location: int       # z^L_t ∈ {0, ..., k_L-1}
    orientation: int    # z^O_t ∈ {0, ..., k_O-1}
    movement: int       # z^M_t ∈ {0, ..., k_M-1}
    nonmanual: int      # z^N_t ∈ {0, ..., k_N-1}

    def to_tuple(self) -> Tuple[int, int, int, int, int]:
        """Convert to tuple representation."""
        return (self.handshape, self.location, self.orientation,
                self.movement, self.nonmanual)

    def __repr__(self) -> str:
        return f"PhonologicalCode(H={self.handshape}, L={self.location}, O={self.orientation}, M={self.movement}, N={self.nonmanual})"


class ProductVectorQuantizer:
    """
    Product vector quantizer for phonological features.

    Implements Section 2.3 with separate codebooks for each component:
        q: ℝ^36 → Σ_H × Σ_L × Σ_O × Σ_M × Σ_N

    Properties (Proposition 1):
        - Lipschitz continuity: ||φ(X) - φ(X')|| < ε implies stable quantization
        - Margin-based robustness: γ-margin ensures noise tolerance

    Training (Proposition 2):
        - K-means clustering on each component independently
        - Sample complexity: O(Σ_j √(d_j log k_j / n))
    """

    def __init__(self, config: Optional[CodebookConfig] = None):
        """
        Initialize quantizer with configuration.

        Args:
            config: Codebook configuration (uses defaults if None)
        """
        self.config = config or CodebookConfig()

        # Codebooks: c^J_k for each component J and codeword k
        # Each codebook is a (k_j, d_j) array
        self.codebooks: Dict[str, Optional[np.ndarray]] = {
            'handshape': None,
            'location': None,
            'orientation': None,
            'movement': None,
            'nonmanual': None,
        }

        # Training metadata
        self.is_trained = False
        self.training_stats: Dict[str, Dict] = {}

        # Feature slicing indices (where each component starts in the 36D vector)
        self._feature_slices = {
            'handshape': slice(0, 10),
            'location': slice(10, 16),
            'orientation': slice(16, 22),
            'movement': slice(22, 31),
            'nonmanual': slice(31, 36),
        }

    def fit(self, features: np.ndarray, verbose: bool = True):
        """
        Learn codebooks from training features.

        Section 2.3: "We learn codebooks {c^J_k} via K-means on Level 2 data"

        Args:
            features: Training features, shape (n_samples, 36)
            verbose: Print training progress

        Raises:
            ValueError: If feature dimensions don't match expected size
        """
        if features.shape[1] != 36:
            raise ValueError(f"Expected 36 features, got {features.shape[1]}")

        n_samples = features.shape[0]

        if verbose:
            print(f"\n{'='*70}")
            print(f"Training Product Vector Quantizer")
            print(f"{'='*70}")
            print(f"Training samples: {n_samples}")
            print(f"Total alphabet size: {self.config.total_alphabet_size():,}")
            print(f"Component sizes: {self.config.get_component_sizes()}")
            print(f"{'='*70}\n")

        # Train each component independently
        components = ['handshape', 'location', 'orientation', 'movement', 'nonmanual']

        for component in components:
            if verbose:
                print(f"Training {component} codebook...")

            # Extract component features
            feature_slice = self._feature_slices[component]
            component_features = features[:, feature_slice]

            # Get codebook size
            k = self.config.get_component_sizes()[component]

            # Learn codebook via K-means
            codebook, stats = self._learn_codebook_kmeans(
                component_features, k, component
            )

            self.codebooks[component] = codebook
            self.training_stats[component] = stats

            if verbose:
                print(f"  ✓ Codebook shape: {codebook.shape}")
                print(f"  ✓ Inertia: {stats['inertia']:.4f}")
                print(f"  ✓ Iterations: {stats['n_iter']}")
                print()

        self.is_trained = True

        if verbose:
            print(f"{'='*70}")
            print("✓ Training Complete!")
            print(f"{'='*70}\n")

    def _learn_codebook_kmeans(
        self,
        features: np.ndarray,
        k: int,
        _component_name: str,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Learn codebook using K-means clustering.

        Args:
            features: Component features, shape (n_samples, d_j)
            k: Number of cluster centers (codebook size)
            _component_name: Name for debugging (unused but kept for API compatibility)

        Returns:
            codebook: Learned codewords, shape (k, d_j)
            stats: Training statistics
        """
        from sklearn.cluster import KMeans

        # K-means clustering
        kmeans = KMeans(
            n_clusters=k,
            max_iter=self.config.max_iter,
            n_init=self.config.n_init,
            random_state=self.config.random_state,
            verbose=0,
        )

        kmeans.fit(features)

        codebook = kmeans.cluster_centers_  # Shape: (k, d_j)

        stats = {
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_,
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'n_clusters': k,
        }

        return codebook, stats

    def quantize(self, features: np.ndarray) -> PhonologicalCode:
        """
        Quantize continuous features to discrete phonological code.

        Section 2.3, Equation (7):
            z^J_t = argmin_k ||u^J_t - c^J_k||_2

        Args:
            features: Feature vector, shape (36,) or PhonologicalFeatures object

        Returns:
            PhonologicalCode with indices for each component

        Raises:
            RuntimeError: If quantizer not trained
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer not trained. Call fit() first.")

        # Handle PhonologicalFeatures object
        if hasattr(features, 'concatenate'):
            features = features.concatenate()  # type: ignore[union-attr]

        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != 36:
            raise ValueError(f"Expected 36 features, got {features.shape[1]}")

        # Quantize each component
        codes = {}
        for component in ['handshape', 'location', 'orientation', 'movement', 'nonmanual']:
            # Extract component features
            feature_slice = self._feature_slices[component]
            component_features = features[0, feature_slice]

            # Find nearest codeword
            codebook = self.codebooks[component]
            distances = np.linalg.norm(codebook - component_features, axis=1)
            code_idx = np.argmin(distances)

            codes[component] = int(code_idx)

        return PhonologicalCode(**codes)

    def quantize_batch(self, features: np.ndarray) -> List[PhonologicalCode]:
        """
        Quantize batch of features.

        Args:
            features: Feature matrix, shape (n_samples, 36)

        Returns:
            List of PhonologicalCode objects
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer not trained. Call fit() first.")

        codes = []
        for i in range(features.shape[0]):
            code = self.quantize(features[i])
            codes.append(code)

        return codes

    def dequantize(self, code: PhonologicalCode) -> np.ndarray:
        """
        Reconstruct continuous features from discrete code.

        Inverse operation (approximate): Z_t → φ̂(X_t)

        Args:
            code: Phonological code

        Returns:
            Reconstructed features, shape (36,)
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer not trained. Call fit() first.")

        # Lookup codewords for each component
        reconstructed = []

        for component in ['handshape', 'location', 'orientation', 'movement', 'nonmanual']:
            code_idx = getattr(code, component)
            codebook = self.codebooks[component]
            assert codebook is not None, f"Codebook {component} not trained"
            codeword = codebook[code_idx]
            reconstructed.append(codeword)

        return np.concatenate(reconstructed)

    def compute_quantization_error(self, features: np.ndarray) -> float:
        """
        Compute reconstruction error: ||φ(X) - φ̂(X)||_2

        Used for validating Proposition 2 (sample complexity bound).

        Args:
            features: Feature matrix, shape (n_samples, 36)

        Returns:
            Mean squared reconstruction error
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer not trained. Call fit() first.")

        errors = []

        for i in range(features.shape[0]):
            # Quantize and dequantize
            code = self.quantize(features[i])
            reconstructed = self.dequantize(code)

            # Compute error
            error = np.linalg.norm(features[i] - reconstructed)
            errors.append(error)

        return float(np.mean(errors))

    def save(self, path: str):
        """Save trained quantizer to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained quantizer")

        save_dict = {
            'config': self.config,
            'codebooks': self.codebooks,
            'training_stats': self.training_stats,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"✓ Saved quantizer to {path}")

    @classmethod
    def load(cls, path: str) -> 'ProductVectorQuantizer':
        """Load trained quantizer from disk."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        quantizer = cls(config=save_dict['config'])
        quantizer.codebooks = save_dict['codebooks']
        quantizer.training_stats = save_dict['training_stats']
        quantizer.is_trained = True

        print(f"✓ Loaded quantizer from {path}")
        return quantizer


# ============================================================================
# Testing / Validation
# ============================================================================

def test_quantizer_basic():
    """Basic functionality test."""
    print("\n" + "="*70)
    print("Testing Product Vector Quantizer - Basic")
    print("="*70 + "\n")

    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    features = np.random.randn(n_samples, 36)

    # Train quantizer
    config = CodebookConfig(
        k_handshape=8,  # Smaller for testing
        k_location=4,
        k_orientation=4,
        k_movement=4,
        k_nonmanual=4,
    )

    quantizer = ProductVectorQuantizer(config)
    quantizer.fit(features, verbose=True)

    # Test quantization
    test_feature = features[0]
    code = quantizer.quantize(test_feature)

    print(f"Test quantization:")
    print(f"  Input feature: {test_feature[:5]}... (first 5)")
    print(f"  Output code: {code}")

    # Test dequantization
    reconstructed = quantizer.dequantize(code)
    error = np.linalg.norm(test_feature - reconstructed)

    print(f"\nReconstruction:")
    print(f"  Reconstructed: {reconstructed[:5]}... (first 5)")
    print(f"  Error: {error:.4f}")

    # Test batch quantization
    batch_codes = quantizer.quantize_batch(features[:10])
    print(f"\nBatch quantization:")
    print(f"  Quantized {len(batch_codes)} samples")
    print(f"  First 3 codes:")
    for i, code in enumerate(batch_codes[:3]):
        print(f"    {i}: {code}")

    # Compute overall quantization error
    total_error = quantizer.compute_quantization_error(features[:100])
    print(f"\nOverall quantization error (100 samples): {total_error:.4f}")

    print("\n" + "="*70)
    print("✓ Basic tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_quantizer_basic()
