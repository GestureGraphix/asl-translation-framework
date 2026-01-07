"""
BiLSTM Encoder for ASL Recognition
Section 6: "Neural Architecture"

Implements the temporal encoder that processes phonological feature sequences
and produces frame-level representations for CTC decoding.

Architecture:
    Input: Phonological features φ(X_t) ∈ ℝ^36 or quantized codes Z_t ∈ Σ

    Embedding (if using quantized codes):
        E: Σ_j → ℝ^{d_emb} for each component j

    BiLSTM Encoder:
        h_t = BiLSTM(x_t, h_{t-1})

    Output: Frame-level representations h_t ∈ ℝ^{d_hidden}

This provides the sequence model for Stage 2 CTC training.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration for BiLSTM encoder."""

    # Input configuration
    input_type: str = "features"  # "features" or "codes"
    input_dim: int = 36           # Feature dimension (36 for continuous features)

    # Embedding configuration (for quantized codes)
    vocab_sizes: Optional[Dict[str, int]] = None  # Codebook sizes per component
    embedding_dim: int = 64       # Embedding dimension per component

    # LSTM configuration
    hidden_dim: int = 256         # Hidden dimension for LSTM
    num_layers: int = 3           # Number of LSTM layers
    dropout: float = 0.3          # Dropout rate
    bidirectional: bool = True    # Use BiLSTM (recommended)

    # Output dimension
    @property
    def output_dim(self) -> int:
        """Output dimension from encoder."""
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def __post_init__(self):
        """Validate configuration."""
        if self.input_type not in ["features", "codes"]:
            raise ValueError(f"Invalid input_type: {self.input_type}")

        if self.input_type == "codes" and self.vocab_sizes is None:
            raise ValueError("vocab_sizes required for input_type='codes'")


class ASLEncoder(nn.Module):
    """
    BiLSTM encoder for ASL feature sequences.

    Processes temporal sequences of phonological features or quantized codes
    and produces frame-level representations for CTC decoding.

    Forward pass:
        Input: (batch_size, seq_len, input_dim)
        Output: (batch_size, seq_len, output_dim)
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize encoder.

        Args:
            config: Encoder configuration
        """
        super().__init__()
        self.config = config

        # Input processing
        if config.input_type == "features":
            # Direct feature input (continuous 36D vectors)
            self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
            lstm_input_dim = config.hidden_dim

        else:  # "codes"
            # Embed quantized codes
            # Each phonological component gets its own embedding
            assert config.vocab_sizes is not None, "vocab_sizes required for codes input"
            self.embeddings = nn.ModuleDict()

            for component, vocab_size in config.vocab_sizes.items():
                self.embeddings[component] = nn.Embedding(
                    vocab_size,
                    config.embedding_dim
                )

            # Project concatenated embeddings
            total_embedding_dim = len(config.vocab_sizes) * config.embedding_dim
            self.input_projection = nn.Linear(total_embedding_dim, config.hidden_dim)
            lstm_input_dim = config.hidden_dim

        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,  # Input: (batch, seq, feature)
        )

        # Layer normalization (stabilizes training)
        self.layer_norm = nn.LayerNorm(config.output_dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence.

        Args:
            x: Input tensor
               - If input_type="features": (batch_size, seq_len, 36)
               - If input_type="codes": (batch_size, seq_len, 5) with integer codes
            lengths: Sequence lengths for each batch element (batch_size,)
                     Used for packing padded sequences

        Returns:
            encoded: Encoded features (batch_size, seq_len, output_dim)
            lengths: Sequence lengths (same as input or inferred)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Process input
        if self.config.input_type == "features":
            # Direct projection of continuous features
            x_proj = self.input_projection(x)  # (batch, seq, hidden_dim)

        else:  # "codes"
            # Embed each component separately
            # Assuming x is (batch, seq, 5) with [H, L, O, M, N] codes
            assert self.config.vocab_sizes is not None, "vocab_sizes required for codes"
            embedded = []
            component_names = list(self.config.vocab_sizes.keys())

            for i, component in enumerate(component_names):
                component_codes = x[:, :, i].long()  # (batch, seq)
                component_emb = self.embeddings[component](component_codes)  # (batch, seq, emb_dim)
                embedded.append(component_emb)

            # Concatenate embeddings
            x_embedded = torch.cat(embedded, dim=-1)  # (batch, seq, 5*emb_dim)
            x_proj = self.input_projection(x_embedded)

        # Pack sequences if lengths provided (for efficiency with variable-length sequences)
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_proj,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            lstm_out, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out,
                batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x_proj)
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)

        # Normalize and dropout
        encoded = self.layer_norm(lstm_out)
        encoded = self.dropout(encoded)

        return encoded, lengths


class ASLEncoderWithProjection(nn.Module):
    """
    Encoder with additional projection layers.

    Useful for multi-task learning where we need different projections
    for different objectives (CTC, segmentation, phonology prediction).
    """

    def __init__(self, config: EncoderConfig):
        """Initialize encoder with projection heads."""
        super().__init__()

        self.encoder = ASLEncoder(config)
        self.config = config

        # Projection heads can be added as needed
        # For example:
        # self.ctc_projection = nn.Linear(config.output_dim, vocab_size)
        # self.segmentation_projection = nn.Linear(config.output_dim, 2)  # boundary/non-boundary

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple outputs.

        Returns:
            Dictionary with different representations
        """
        encoded, lengths = self.encoder(x, lengths)

        outputs = {
            'encoded': encoded,
            'lengths': lengths,
        }

        return outputs


# ============================================================================
# Testing
# ============================================================================

def test_encoder_features():
    """Test encoder with continuous features."""
    print("\n" + "="*70)
    print("Testing ASL Encoder - Continuous Features")
    print("="*70 + "\n")

    # Configuration
    config = EncoderConfig(
        input_type="features",
        input_dim=36,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
    )

    # Create encoder
    encoder = ASLEncoder(config)
    print(f"Encoder config:")
    print(f"  Input type: {config.input_type}")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Output dim: {config.output_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Bidirectional: {config.bidirectional}")

    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {n_params:,}\n")

    # Test forward pass
    batch_size = 4
    seq_len = 50
    x = torch.randn(batch_size, seq_len, 36)
    lengths = torch.tensor([50, 45, 40, 35])

    print(f"Input shape: {x.shape}")
    print(f"Lengths: {lengths.tolist()}\n")

    with torch.no_grad():
        encoded, out_lengths = encoder(x, lengths)

    print(f"Output shape: {encoded.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.output_dim})")
    print(f"Output lengths: {out_lengths.tolist()}")

    assert encoded.shape == (batch_size, seq_len, config.output_dim)
    assert torch.equal(out_lengths, lengths)

    print("\n✓ Continuous features test passed!\n")


def test_encoder_codes():
    """Test encoder with quantized codes."""
    print("\n" + "="*70)
    print("Testing ASL Encoder - Quantized Codes")
    print("="*70 + "\n")

    # Configuration
    vocab_sizes = {
        'handshape': 64,
        'location': 32,
        'orientation': 32,
        'movement': 16,
        'nonmanual': 16,
    }

    config = EncoderConfig(
        input_type="codes",
        vocab_sizes=vocab_sizes,
        embedding_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
    )

    # Create encoder
    encoder = ASLEncoder(config)
    print(f"Encoder config:")
    print(f"  Input type: {config.input_type}")
    print(f"  Vocab sizes: {vocab_sizes}")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Output dim: {config.output_dim}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {n_params:,}\n")

    # Test forward pass with codes
    batch_size = 4
    seq_len = 50

    # Generate random codes (5 components per timestep)
    x = torch.zeros(batch_size, seq_len, 5, dtype=torch.long)
    x[:, :, 0] = torch.randint(0, vocab_sizes['handshape'], (batch_size, seq_len))
    x[:, :, 1] = torch.randint(0, vocab_sizes['location'], (batch_size, seq_len))
    x[:, :, 2] = torch.randint(0, vocab_sizes['orientation'], (batch_size, seq_len))
    x[:, :, 3] = torch.randint(0, vocab_sizes['movement'], (batch_size, seq_len))
    x[:, :, 4] = torch.randint(0, vocab_sizes['nonmanual'], (batch_size, seq_len))

    lengths = torch.tensor([50, 45, 40, 35])

    print(f"Input shape: {x.shape}")
    print(f"Sample codes (first timestep): {x[0, 0].tolist()}")
    print(f"Lengths: {lengths.tolist()}\n")

    with torch.no_grad():
        encoded, _ = encoder(x, lengths)

    print(f"Output shape: {encoded.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.output_dim})")

    assert encoded.shape == (batch_size, seq_len, config.output_dim)

    print("\n✓ Quantized codes test passed!\n")


if __name__ == "__main__":
    test_encoder_features()
    test_encoder_codes()

    print("="*70)
    print("✓ All encoder tests passed!")
    print("="*70 + "\n")
