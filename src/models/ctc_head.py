"""
CTC Head and Loss for ASL Recognition
Section 6: "CTC Training"

Implements Connectionist Temporal Classification (CTC) for sequence-to-sequence
learning without frame-level alignment.

CTC allows the model to learn mappings from input sequences (phonological features)
to output sequences (gloss sequences) without requiring precise frame-level labels.

Key components:
    - CTC projection head: Maps encoder outputs to vocabulary logits
    - CTC loss: Marginalizes over all valid alignments
    - CTC decoding: Greedy and beam search decoding

Mathematical formulation (Section 6):
    p(y|x) = Σ_{π ∈ B^{-1}(y)} Π_t p(π_t | x)

    where B is the CTC blanking function that removes blanks and repeats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CTCConfig:
    """Configuration for CTC head."""

    # Vocabulary
    vocab_size: int              # Number of output tokens (glosses)
    blank_id: int = 0            # Index for CTC blank token

    # Model dimensions
    encoder_dim: int = 512       # Dimension from encoder output

    # Decoding
    beam_width: int = 10         # Beam width for beam search decoding


class CTCHead(nn.Module):
    """
    CTC projection head.

    Projects encoder outputs to vocabulary logits for CTC loss computation.

    Forward:
        Input: (batch_size, seq_len, encoder_dim)
        Output: (batch_size, seq_len, vocab_size)
    """

    def __init__(self, config: CTCConfig):
        """
        Initialize CTC head.

        Args:
            config: CTC configuration
        """
        super().__init__()
        self.config = config

        # Projection to vocabulary
        # Maps encoder outputs to logits over vocabulary + blank
        self.projection = nn.Linear(config.encoder_dim, config.vocab_size)

        # Initialize with small weights (helps CTC convergence)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Project encoder outputs to CTC logits.

        Args:
            encoder_output: Encoder outputs (batch_size, seq_len, encoder_dim)

        Returns:
            logits: CTC logits (batch_size, seq_len, vocab_size)
        """
        logits = self.projection(encoder_output)
        return logits


class CTCLoss(nn.Module):
    """
    CTC loss wrapper.

    Computes CTC loss given encoder outputs and target sequences.

    Uses PyTorch's built-in CTCLoss which efficiently marginalizes over
    all valid alignments using dynamic programming.
    """

    def __init__(self, config: CTCConfig, reduction: str = 'mean', blank_penalty: float = 0.0):
        """
        Initialize CTC loss.

        Args:
            config: CTC configuration
            reduction: Reduction mode ('mean', 'sum', 'none')
            blank_penalty: Penalty weight for blank predictions (0.0 = no penalty)
        """
        super().__init__()
        self.config = config
        self.blank_penalty = blank_penalty

        # PyTorch CTCLoss
        # - blank: index of blank label
        # - zero_infinity: set infinite losses to zero (prevents NaN gradients)
        # - reduction: how to aggregate batch losses
        self.ctc_loss = nn.CTCLoss(
            blank=config.blank_id,
            zero_infinity=True,
            reduction=reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            logits: Model outputs (batch_size, seq_len, vocab_size)
            targets: Target sequences (batch_size, max_target_len) or flattened (sum(target_lengths),)
            input_lengths: Length of each input sequence (batch_size,)
            target_lengths: Length of each target sequence (batch_size,)

        Returns:
            loss: CTC loss (scalar if reduction='mean', else per-sample)
        """
        # CTC loss expects log probabilities in shape (seq_len, batch_size, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (seq_len, batch, vocab)

        # Flatten targets if not already (CTC expects 1D target tensor)
        if targets.dim() == 2:
            # Concatenate all targets into single 1D tensor
            target_list = []
            for i in range(targets.size(0)):
                target_list.append(targets[i, :target_lengths[i]])
            targets = torch.cat(target_list)

        # Compute CTC loss
        loss = self.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        # Add blank penalty if configured
        if self.blank_penalty > 0:
            # Compute ratio of blank predictions
            # log_probs shape: (seq_len, batch, vocab)
            blank_log_probs = log_probs[:, :, self.config.blank_id]  # (seq_len, batch)
            blank_ratio = torch.exp(blank_log_probs).mean()  # Average blank probability

            # Penalize high blank ratios
            loss = loss + self.blank_penalty * blank_ratio

        return loss


class CTCDecoder:
    """
    CTC decoder for inference.

    Implements:
        - Greedy decoding: Select most probable token at each timestep
        - Beam search decoding: Search over likely alignment paths
    """

    def __init__(self, config: CTCConfig):
        """
        Initialize decoder.

        Args:
            config: CTC configuration
        """
        self.config = config

    def greedy_decode(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """
        Greedy CTC decoding.

        At each timestep, selects the most probable token, then removes
        blanks and consecutive duplicates.

        Args:
            logits: Model outputs (batch_size, seq_len, vocab_size)
            lengths: Sequence lengths (batch_size,) - if None, use full sequences

        Returns:
            decoded: List of decoded sequences (batch_size,)
        """
        # Get most probable tokens
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

        batch_size = predictions.size(0)
        decoded = []

        for i in range(batch_size):
            seq_len = lengths[i] if lengths is not None else predictions.size(1)
            pred_seq = predictions[i, :seq_len].tolist()

            # Remove blanks and consecutive duplicates
            decoded_seq = self._remove_blanks_and_duplicates(pred_seq)
            decoded.append(decoded_seq)

        return decoded

    def _remove_blanks_and_duplicates(self, sequence: List[int]) -> List[int]:
        """
        Apply CTC blanking operation.

        Removes blank tokens and consecutive duplicates.

        Args:
            sequence: Raw predicted sequence with blanks

        Returns:
            cleaned: Sequence with blanks and duplicates removed
        """
        cleaned = []
        prev = None

        for token in sequence:
            # Skip blanks
            if token == self.config.blank_id:
                prev = None
                continue

            # Skip consecutive duplicates
            if token != prev:
                cleaned.append(token)
                prev = token

        return cleaned

    def beam_search_decode(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        beam_width: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Beam search CTC decoding.

        Maintains top-k hypotheses at each timestep and expands them.

        Args:
            logits: Model outputs (batch_size, seq_len, vocab_size)
            lengths: Sequence lengths (batch_size,)
            beam_width: Beam width (default: from config)

        Returns:
            decoded: List of best decoded sequences (batch_size,)
        """
        beam_width = beam_width or self.config.beam_width

        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq, vocab)

        batch_size = log_probs.size(0)
        decoded = []

        # Decode each sequence independently
        for i in range(batch_size):
            seq_len = lengths[i] if lengths is not None else log_probs.size(1)
            seq_log_probs = log_probs[i, :seq_len, :]  # (seq_len, vocab)

            best_path = self._beam_search_single(seq_log_probs, beam_width)
            decoded.append(best_path)

        return decoded

    def _beam_search_single(
        self,
        log_probs: torch.Tensor,
        beam_width: int,
    ) -> List[int]:
        """
        Beam search for single sequence.

        Simplified beam search that maintains hypotheses and their scores.

        Args:
            log_probs: Log probabilities for single sequence (seq_len, vocab_size)
            beam_width: Beam width

        Returns:
            best_path: Best decoded sequence
        """
        # Initialize beam with empty hypothesis
        # Each hypothesis: (sequence, log_prob)
        # Use -1 as initial marker (will be filtered out)
        beam = [([-1], 0.0)]

        # Process each timestep
        for t in range(log_probs.size(0)):
            candidates = []

            for seq, score in beam:
                # Expand with all possible tokens
                for token_id in range(self.config.vocab_size):
                    token_score = log_probs[t, token_id].item()
                    new_score = score + token_score

                    # Apply CTC blanking
                    if token_id == self.config.blank_id:
                        new_seq = seq  # Blank doesn't extend sequence
                    elif len(seq) > 0 and seq[-1] == token_id:
                        new_seq = seq  # Duplicate doesn't extend sequence
                    else:
                        new_seq = seq + [token_id]

                    candidates.append((new_seq, new_score))

            # Keep top-k hypotheses
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Return best hypothesis (remove initial marker)
        best_seq, _ = beam[0]
        return [t for t in best_seq if t >= 0]


# ============================================================================
# Complete Model
# ============================================================================

class CTCModel(nn.Module):
    """
    Complete CTC model combining encoder and CTC head.

    Useful wrapper for training and inference.
    """

    def __init__(self, encoder: nn.Module, ctc_config: CTCConfig):
        """
        Initialize CTC model.

        Args:
            encoder: Encoder module (e.g., ASLEncoder)
            ctc_config: CTC configuration
        """
        super().__init__()

        self.encoder = encoder
        self.ctc_head = CTCHead(ctc_config)
        self.ctc_config = ctc_config

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and CTC head.

        Args:
            x: Input features or codes
            lengths: Sequence lengths

        Returns:
            logits: CTC logits (batch_size, seq_len, vocab_size)
            lengths: Sequence lengths after encoding
        """
        # Encode
        encoded, out_lengths = self.encoder(x, lengths)

        # Project to vocabulary
        logits = self.ctc_head(encoded)

        return logits, out_lengths


# ============================================================================
# Testing
# ============================================================================

def test_ctc_head():
    """Test CTC head."""
    print("\n" + "="*70)
    print("Testing CTC Head")
    print("="*70 + "\n")

    config = CTCConfig(
        vocab_size=100,  # 100 glosses
        blank_id=0,
        encoder_dim=256,
    )

    # Create head
    head = CTCHead(config)
    print(f"CTC Head config:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Encoder dim: {config.encoder_dim}")
    print(f"  Blank ID: {config.blank_id}\n")

    # Test forward
    batch_size = 4
    seq_len = 50
    encoder_output = torch.randn(batch_size, seq_len, config.encoder_dim)

    logits = head(encoder_output)
    print(f"Input shape: {encoder_output.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print("\n✓ CTC head test passed!\n")


def test_ctc_loss():
    """Test CTC loss."""
    print("\n" + "="*70)
    print("Testing CTC Loss")
    print("="*70 + "\n")

    config = CTCConfig(vocab_size=10, blank_id=0, encoder_dim=256)
    loss_fn = CTCLoss(config)

    # Create sample data
    batch_size = 2
    seq_len = 20
    logits = torch.randn(batch_size, seq_len, config.vocab_size)

    # Targets: [1, 2, 3] and [4, 5]
    targets = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    target_lengths = torch.tensor([3, 2])
    input_lengths = torch.tensor([20, 15])

    print(f"Logits shape: {logits.shape}")
    print(f"Targets: {targets.tolist()}")
    print(f"Target lengths: {target_lengths.tolist()}")
    print(f"Input lengths: {input_lengths.tolist()}\n")

    # Compute loss
    loss = loss_fn(logits, targets, input_lengths, target_lengths)

    print(f"CTC loss: {loss.item():.4f}")
    assert loss.item() >= 0, "Loss should be non-negative"

    print("\n✓ CTC loss test passed!\n")


def test_ctc_decoding():
    """Test CTC decoding."""
    print("\n" + "="*70)
    print("Testing CTC Decoding")
    print("="*70 + "\n")

    config = CTCConfig(vocab_size=10, blank_id=0, beam_width=5)
    decoder = CTCDecoder(config)

    # Create sample logits (batch_size=2, seq_len=10, vocab=10)
    batch_size = 2
    seq_len = 10
    logits = torch.randn(batch_size, seq_len, config.vocab_size)
    lengths = torch.tensor([10, 8])

    print(f"Logits shape: {logits.shape}")
    print(f"Lengths: {lengths.tolist()}\n")

    # Greedy decoding
    print("Greedy decoding:")
    greedy_decoded = decoder.greedy_decode(logits, lengths)
    for i, seq in enumerate(greedy_decoded):
        print(f"  Sample {i}: {seq}")

    # Beam search decoding
    print("\nBeam search decoding:")
    beam_decoded = decoder.beam_search_decode(logits, lengths, beam_width=3)
    for i, seq in enumerate(beam_decoded):
        print(f"  Sample {i}: {seq}")

    print("\n✓ CTC decoding test passed!\n")


if __name__ == "__main__":
    test_ctc_head()
    test_ctc_loss()
    test_ctc_decoding()

    print("="*70)
    print("✓ All CTC tests passed!")
    print("="*70 + "\n")
