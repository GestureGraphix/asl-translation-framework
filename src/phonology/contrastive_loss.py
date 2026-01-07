"""Contrastive loss for self-supervised phonological learning.

Implements NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss
for learning temporally coherent representations (Section 7.2).

Mathematical Framework:
    For anchor-positive pair (z_i, z_i^+) and negatives {z_j}_{j≠i}:

    L_contrast = -log( exp(sim(z_i, z_i^+) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )

    where:
    - sim(u, v) = cosine similarity = u·v / (||u|| ||v||)
    - τ: temperature parameter (controls concentration)
    - k ranges over all samples in batch except i

Key Properties:
    - Invariance to augmentations (by design)
    - Encourages temporally coherent features
    - Prevents collapse through negative sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    Used for contrastive learning in Stage 1 pre-training.
    Learns representations invariant to temporal augmentations.

    Args:
        temperature: Temperature parameter τ (default: 0.07)
                    Lower τ → sharper distribution, harder negatives
        reduction: 'mean' or 'sum' for batch reduction
    """

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute NT-Xent loss for anchor-positive pairs.

        Args:
            anchor: (B, D) normalized embeddings from view 1
            positive: (B, D) normalized embeddings from view 2 (same videos)
            mask: Optional (B, B) mask for valid pairs

        Returns:
            Scalar loss value

        Mathematical Formulation:
            For each anchor i:
            - Positive: sample i from other view
            - Negatives: all other samples j≠i from both views

            L_i = -log( exp(z_i · z_i^+ / τ) / Σ_k exp(z_i · z_k / τ) )

            Total loss: L = (1/2B) Σ_i [L_i + L_i^+]
                       (symmetric - compute from both views)
        """
        batch_size = anchor.shape[0]
        device = anchor.device

        # Concatenate anchor and positive to create full batch
        # Shape: (2B, D)
        embeddings = torch.cat([anchor, positive], dim=0)

        # Compute all pairwise similarities
        # Shape: (2B, 2B)
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        ) / self.temperature

        # Create mask for positive pairs
        # For anchor i, positive is at position i + batch_size
        # Shape: (2B, 2B)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        for i in range(batch_size):
            # Anchor i -> Positive i+B
            pos_mask[i, i + batch_size] = 1
            # Positive i+B -> Anchor i
            pos_mask[i + batch_size, i] = 1

        # Create mask to exclude self-similarity
        # Shape: (2B, 2B)
        self_mask = torch.eye(2 * batch_size, device=device)

        # Negative mask: all samples except self and positive
        neg_mask = 1 - pos_mask - self_mask

        # Compute log-sum-exp for numerical stability
        # For each row: log( Σ_k exp(sim_ik) ) for negatives
        neg_logits = similarity_matrix * neg_mask + (1 - neg_mask) * (-1e9)
        neg_exp_sum = torch.logsumexp(neg_logits, dim=1, keepdim=True)

        # Positive logits
        pos_logits = (similarity_matrix * pos_mask).sum(dim=1, keepdim=True)

        # NT-Xent loss: -log( exp(pos) / (exp(pos) + Σ exp(neg)) )
        # = -pos + log(exp(pos) + Σ exp(neg))
        # = -pos + log_sum_exp([pos, neg])
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        loss_per_sample = -pos_logits + torch.logsumexp(all_logits, dim=1, keepdim=True)

        # Only compute loss for samples with valid positives
        valid_samples = pos_mask.sum(dim=1) > 0
        loss_per_sample = loss_per_sample[valid_samples]

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Projects encoder outputs to normalized embeddings for contrastive loss.
    Common architecture: Linear -> ReLU -> Linear -> L2 normalize

    Args:
        input_dim: Dimension of encoder outputs
        hidden_dim: Dimension of hidden layer (default: 2048)
        output_dim: Dimension of projection space (default: 128)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize embeddings.

        Args:
            x: (B, D_in) encoder outputs

        Returns:
            (B, D_out) normalized embeddings
        """
        projected = self.projection(x)
        # L2 normalization for cosine similarity
        return F.normalize(projected, dim=1, p=2)


class ContrastiveLearner(nn.Module):
    """Complete contrastive learning module.

    Combines encoder, projection head, and NT-Xent loss for Stage 1 training.

    Args:
        encoder: Feature encoder (e.g., BiLSTM)
        input_dim: Encoder output dimension
        projection_dim: Projection space dimension (default: 128)
        temperature: NT-Xent temperature (default: 0.07)
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        self.encoder = encoder
        self.projection_head = ProjectionHead(
            input_dim=input_dim,
            output_dim=projection_dim
        )
        self.contrastive_loss = NTXentLoss(temperature=temperature)

    def encode(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode features to representations.

        Args:
            features: (B, T, D) input features
            lengths: (B,) actual sequence lengths

        Returns:
            (B, D_enc) encoded representations (using final hidden state)
        """
        # Encode sequence - ASLEncoder returns (encoded, lengths)
        encoded, _ = self.encoder(features, lengths)  # (B, T, D_enc)

        # Use final hidden state (mean pooling could also work)
        # Get last valid timestep for each sequence
        batch_size = features.shape[0]
        final_outputs = torch.stack([
            encoded[i, lengths[i] - 1] for i in range(batch_size)
        ])

        return final_outputs  # (B, D_enc)

    def forward(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        anchor_lengths: torch.Tensor,
        positive_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss for anchor-positive pairs.

        Args:
            anchor_features: (B, T1, D) features from view 1
            positive_features: (B, T2, D) features from view 2
            anchor_lengths: (B,) actual lengths of anchors
            positive_lengths: (B,) actual lengths of positives

        Returns:
            Scalar contrastive loss
        """
        # Encode both views
        anchor_encoded = self.encode(anchor_features, anchor_lengths)  # (B, D_enc)
        positive_encoded = self.encode(positive_features, positive_lengths)  # (B, D_enc)

        # Project to contrastive space
        anchor_proj = self.projection_head(anchor_encoded)  # (B, D_proj)
        positive_proj = self.projection_head(positive_encoded)  # (B, D_proj)

        # Compute contrastive loss
        loss = self.contrastive_loss(anchor_proj, positive_proj)

        return loss


# Testing and example usage
if __name__ == "__main__":
    print("Testing contrastive loss implementation...")

    # Test NT-Xent loss
    batch_size = 8
    embedding_dim = 128

    # Create dummy embeddings
    anchor = F.normalize(torch.randn(batch_size, embedding_dim), dim=1)
    positive = F.normalize(torch.randn(batch_size, embedding_dim), dim=1)

    # Compute loss
    loss_fn = NTXentLoss(temperature=0.07)
    loss = loss_fn(anchor, positive)

    print(f"NT-Xent loss: {loss.item():.4f}")
    assert loss > 0, "Loss should be positive"

    # Test projection head
    input_dim = 256
    projection_head = ProjectionHead(input_dim=input_dim, output_dim=128)

    encoder_output = torch.randn(batch_size, input_dim)
    projected = projection_head(encoder_output)

    print(f"\nProjection head:")
    print(f"  Input shape: {encoder_output.shape}")
    print(f"  Output shape: {projected.shape}")
    print(f"  Output norm: {projected.norm(dim=1).mean():.4f}")  # Should be ~1.0

    # Test with similar pairs (should have lower loss)
    anchor = F.normalize(torch.randn(batch_size, embedding_dim), dim=1)
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_dim)  # Small perturbation
    positive = F.normalize(positive, dim=1)

    loss_similar = loss_fn(anchor, positive)
    print(f"\nLoss with similar pairs: {loss_similar.item():.4f}")
    assert loss_similar < loss, "Similar pairs should have lower loss"

    print("\n✓ All contrastive loss tests passed!")
