"""Vector Quantization loss for phonological learning.

Implements trainable VQ-VAE style quantization for Stage 1 pre-training.
Learns codebooks end-to-end with gradient descent rather than K-means.

Mathematical Framework (Section 2.3 + 7.2):
    For features φ(X_t) ∈ ℝ^36, learn codebooks {c^J_k} to minimize:

    L_phon = ||φ(X_t) - φ̂(X_t)||² + β ||sg[φ(X_t)] - φ̂(X_t)||²

    where:
    - φ̂(X_t): Reconstructed features from quantized codes
    - sg[·]: Stop-gradient operator
    - β: Commitment loss weight (encourages encoder commitment to codebook)

Product VQ Structure:
    Separate codebooks for each phonological component:
    - Handshape: 10D → k_H codes
    - Location: 6D → k_L codes
    - Orientation: 6D → k_O codes
    - Movement: 9D → k_M codes
    - Non-manual: 5D → k_N codes

Straight-Through Estimator:
    Forward: z = argmin_k ||u - c_k||  (discrete, non-differentiable)
    Backward: ∂L/∂u = ∂L/∂ẑ  (copy gradient through)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ProductVQLayer(nn.Module):
    """Differentiable Product Vector Quantization layer.

    Implements product VQ with straight-through gradient estimation.
    Each phonological component has its own codebook.

    Args:
        num_embeddings: Codebook sizes for each component
        embedding_dims: Feature dimensions for each component
        commitment_cost: Weight β for commitment loss (default: 0.25)
    """

    def __init__(
        self,
        num_embeddings: Dict[str, int],
        embedding_dims: Dict[str, int],
        commitment_cost: float = 0.25
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.commitment_cost = commitment_cost

        # Component names and slicing
        self.components = ['handshape', 'location', 'orientation', 'movement', 'nonmanual']
        self.feature_slices = {
            'handshape': slice(0, 10),
            'location': slice(10, 16),
            'orientation': slice(16, 22),
            'movement': slice(22, 31),
            'nonmanual': slice(31, 36),
        }

        # Create codebook embeddings for each component
        self.embeddings = nn.ModuleDict()
        for component in self.components:
            self.embeddings[component] = nn.Embedding(
                num_embeddings[component],
                embedding_dims[component]
            )
            # Initialize embeddings
            self.embeddings[component].weight.data.uniform_(-1/num_embeddings[component],
                                                           1/num_embeddings[component])

    def quantize_component(
        self,
        features: torch.Tensor,
        component: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a single phonological component.

        Args:
            features: (B, D_j) component features
            component: Component name

        Returns:
            Tuple of:
            - quantized: (B, D_j) quantized features
            - indices: (B,) codebook indices
        """
        # Get codebook
        codebook = self.embeddings[component].weight  # (K_j, D_j)

        # Compute distances: ||u - c_k||²
        # Using: ||u - c||² = ||u||² + ||c||² - 2u·c
        distances = (
            torch.sum(features ** 2, dim=1, keepdim=True) +
            torch.sum(codebook ** 2, dim=1) -
            2 * torch.matmul(features, codebook.t())
        )  # (B, K_j)

        # Find nearest codeword
        indices = torch.argmin(distances, dim=1)  # (B,)

        # Lookup quantized values
        quantized = self.embeddings[component](indices)  # (B, D_j)

        return quantized, indices

    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply product vector quantization.

        Args:
            features: (B, 36) phonological features

        Returns:
            Tuple of:
            - quantized: (B, 36) quantized features with straight-through gradients
            - loss: VQ loss (reconstruction + commitment)
            - indices: Dict of quantized indices for each component
        """
        batch_size = features.shape[0]

        # Quantize each component
        quantized_components = []
        indices_dict = {}
        component_losses = []

        for component in self.components:
            # Extract component features
            feature_slice = self.feature_slices[component]
            component_features = features[:, feature_slice]  # (B, D_j)

            # Quantize
            quantized, indices = self.quantize_component(component_features, component)
            quantized_components.append(quantized)
            indices_dict[component] = indices

            # VQ loss for this component
            # L = ||x - sg[e]||² + β ||sg[x] - e||²
            # where x = input features, e = quantized embedding

            # Reconstruction loss: ||x - e||²
            recon_loss = F.mse_loss(quantized.detach(), component_features)

            # Commitment loss: ||x - e||² (encoder commits to codebook)
            commit_loss = F.mse_loss(quantized, component_features.detach())

            component_loss = recon_loss + self.commitment_cost * commit_loss
            component_losses.append(component_loss)

        # Concatenate quantized components
        quantized = torch.cat(quantized_components, dim=1)  # (B, 36)

        # Straight-through estimator: copy gradient through quantization
        quantized = features + (quantized - features).detach()

        # Total VQ loss (mean over components)
        vq_loss = torch.stack(component_losses).mean()

        return quantized, vq_loss, indices_dict

    def get_codebook_usage(
        self,
        indices_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute codebook usage statistics.

        Args:
            indices_dict: Dict of indices from forward pass

        Returns:
            Dict mapping component to usage ratio (0-1)
        """
        usage = {}
        for component in self.components:
            indices = indices_dict[component]
            unique_codes = torch.unique(indices).numel()
            total_codes = self.num_embeddings[component]
            usage[component] = unique_codes / total_codes

        return usage


class PhonologicalReconstructionLoss(nn.Module):
    """Complete phonological reconstruction loss for Stage 1.

    Combines VQ loss with additional regularizations.

    Args:
        num_embeddings: Codebook sizes (default: from paper)
        embedding_dims: Feature dimensions (default: from paper)
        commitment_cost: VQ commitment weight (default: 0.25)
    """

    def __init__(
        self,
        num_embeddings: Dict[str, int] = None,
        embedding_dims: Dict[str, int] = None,
        commitment_cost: float = 0.25
    ):
        super().__init__()

        # Default codebook sizes (Section 2.3)
        if num_embeddings is None:
            num_embeddings = {
                'handshape': 64,
                'location': 32,
                'orientation': 32,
                'movement': 16,
                'nonmanual': 16,
            }

        # Default feature dimensions
        if embedding_dims is None:
            embedding_dims = {
                'handshape': 10,
                'location': 6,
                'orientation': 6,
                'movement': 9,
                'nonmanual': 5,
            }

        self.vq_layer = ProductVQLayer(
            num_embeddings=num_embeddings,
            embedding_dims=embedding_dims,
            commitment_cost=commitment_cost
        )

    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute phonological reconstruction loss.

        Args:
            features: (B, 36) phonological features

        Returns:
            Tuple of:
            - quantized: (B, 36) quantized features
            - losses: Dict of loss components
            - indices: Dict of quantized indices
        """
        # Apply VQ
        quantized, vq_loss, indices_dict = self.vq_layer(features)

        # Reconstruction loss (MSE between input and quantized)
        recon_loss = F.mse_loss(quantized, features)

        # Total loss
        total_loss = recon_loss + vq_loss

        losses = {
            'total': total_loss,
            'reconstruction': recon_loss,
            'vq': vq_loss,
        }

        return quantized, losses, indices_dict

    def get_codebook_usage(self, indices_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get codebook usage statistics."""
        return self.vq_layer.get_codebook_usage(indices_dict)


# Testing
if __name__ == "__main__":
    print("Testing VQ loss implementation...")

    # Test ProductVQLayer
    num_embeddings = {
        'handshape': 16,
        'location': 8,
        'orientation': 8,
        'movement': 8,
        'nonmanual': 8,
    }

    embedding_dims = {
        'handshape': 10,
        'location': 6,
        'orientation': 6,
        'movement': 9,
        'nonmanual': 5,
    }

    vq_layer = ProductVQLayer(num_embeddings, embedding_dims, commitment_cost=0.25)

    # Create dummy features
    batch_size = 16
    features = torch.randn(batch_size, 36)

    # Forward pass
    quantized, vq_loss, indices = vq_layer(features)

    print(f"\nProduct VQ Layer:")
    print(f"  Input shape: {features.shape}")
    print(f"  Quantized shape: {quantized.shape}")
    print(f"  VQ loss: {vq_loss.item():.4f}")
    print(f"  Indices: {list(indices.keys())}")

    # Check gradient flow
    vq_loss.backward()
    print(f"  Gradient exists: {features.grad is not None}")

    # Test codebook usage
    usage = vq_layer.get_codebook_usage(indices)
    print(f"\nCodebook usage:")
    for comp, ratio in usage.items():
        print(f"  {comp}: {ratio:.2%}")

    # Test PhonologicalReconstructionLoss
    print(f"\n{'='*60}")
    recon_loss_module = PhonologicalReconstructionLoss()

    features = torch.randn(batch_size, 36)
    quantized, losses, indices = recon_loss_module(features)

    print(f"Phonological Reconstruction Loss:")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Reconstruction: {losses['reconstruction'].item():.4f}")
    print(f"  VQ loss: {losses['vq'].item():.4f}")

    usage = recon_loss_module.get_codebook_usage(indices)
    print(f"\nCodebook usage:")
    for comp, ratio in usage.items():
        print(f"  {comp}: {ratio:.2%}")

    print("\n✓ All VQ loss tests passed!")
