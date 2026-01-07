#!/usr/bin/env python3.11
"""
End-to-End Encoder + CTC Pipeline Test

Tests the complete sequence modeling pipeline:
    Video → Phonology → Encoder → CTC → Glosses

This validates Phase 1 (Phonology) + Phase 2 (Sequence Modeling):
    1. Extract landmarks and features (Section 2)
    2. Encode temporal sequence (Section 6)
    3. CTC decoding to glosses (Section 6)

Usage:
    python3.11 scripts/test_encoder_ctc_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phonology.mediapipe_extractor_v2 import MediaPipeExtractor
from phonology.features import FeatureExtractor
from phonology.quantizer import ProductVectorQuantizer, CodebookConfig

from models.encoder import ASLEncoder, EncoderConfig
from models.ctc_head import CTCModel, CTCConfig, CTCLoss, CTCDecoder

import torch
import numpy as np


def test_full_pipeline(video_path: str, max_frames: int = 100):
    """
    Test complete pipeline from video to glosses.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process
    """
    print(f"\n{'='*70}")
    print(f"End-to-End Encoder + CTC Pipeline Test")
    print(f"{'='*70}")
    print(f"Video: {Path(video_path).name}")
    print(f"Max frames: {max_frames}")
    print(f"{'='*70}\n")

    # ========================================================================
    # Step 1: Extract phonological features
    # ========================================================================
    print("Step 1: Extracting phonological features...")
    print("-" * 70)

    extractor_mp = MediaPipeExtractor()
    landmarks_sequence = extractor_mp.extract_video(video_path, max_frames=max_frames)

    feature_extractor = FeatureExtractor()
    feature_sequence = []

    for landmarks in landmarks_sequence:
        features = feature_extractor.extract_features(landmarks, include_temporal=True)
        feature_sequence.append(features)

    feature_matrix = np.array([f.concatenate() for f in feature_sequence])
    print(f"✓ Extracted features: {feature_matrix.shape}\n")

    # ========================================================================
    # Step 2: Initialize encoder
    # ========================================================================
    print("Step 2: Initializing encoder...")
    print("-" * 70)

    encoder_config = EncoderConfig(
        input_type="features",
        input_dim=36,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
    )

    encoder = ASLEncoder(encoder_config)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"✓ Encoder created")
    print(f"  Hidden dim: {encoder_config.hidden_dim}")
    print(f"  Output dim: {encoder_config.output_dim}")
    print(f"  Parameters: {n_params:,}\n")

    # ========================================================================
    # Step 3: Initialize CTC head
    # ========================================================================
    print("Step 3: Initializing CTC head...")
    print("-" * 70)

    # Vocabulary: Let's simulate 100 glosses
    vocab_size = 100  # Including blank
    ctc_config = CTCConfig(
        vocab_size=vocab_size,
        blank_id=0,
        encoder_dim=encoder_config.output_dim,
        beam_width=10,
    )

    ctc_model = CTCModel(encoder, ctc_config)
    print(f"✓ CTC model created")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Blank ID: {ctc_config.blank_id}\n")

    # ========================================================================
    # Step 4: Forward pass through model
    # ========================================================================
    print("Step 4: Forward pass through model...")
    print("-" * 70)

    # Convert features to PyTorch tensor
    features_tensor = torch.FloatTensor(feature_matrix).unsqueeze(0)  # (1, seq_len, 36)
    lengths = torch.tensor([feature_matrix.shape[0]])

    print(f"Input shape: {features_tensor.shape}")
    print(f"Sequence length: {lengths.item()}\n")

    # Forward pass
    with torch.no_grad():
        logits, out_lengths = ctc_model(features_tensor, lengths)

    print(f"Logits shape: {logits.shape}")
    print(f"Output length: {out_lengths.item()}")
    print(f"✓ Forward pass successful\n")

    # ========================================================================
    # Step 5: CTC Decoding
    # ========================================================================
    print("Step 5: CTC decoding...")
    print("-" * 70)

    decoder = CTCDecoder(ctc_config)

    # Greedy decoding
    greedy_decoded = decoder.greedy_decode(logits, out_lengths)
    print(f"Greedy decoded: {greedy_decoded[0][:20]}... (first 20 tokens)")
    print(f"Sequence length: {len(greedy_decoded[0])}")

    # Beam search decoding
    beam_decoded = decoder.beam_search_decode(logits, out_lengths, beam_width=5)
    print(f"Beam decoded: {beam_decoded[0][:20]}... (first 20 tokens)")
    print(f"Sequence length: {len(beam_decoded[0])}\n")

    # ========================================================================
    # Step 6: Test CTC loss (simulated training)
    # ========================================================================
    print("Step 6: Testing CTC loss (simulated training)...")
    print("-" * 70)

    # Create dummy target (pretend the gloss sequence is [1, 2, 3, 4, 5])
    dummy_targets = torch.tensor([[1, 2, 3, 4, 5]])
    target_lengths = torch.tensor([5])

    loss_fn = CTCLoss(ctc_config)
    loss = loss_fn(logits, dummy_targets, out_lengths, target_lengths)

    print(f"Dummy target: {dummy_targets[0].tolist()}")
    print(f"CTC loss: {loss.item():.4f}")
    print(f"✓ Loss computation successful\n")

    # ========================================================================
    # Step 7: Test with quantized codes (alternative pipeline)
    # ========================================================================
    print("Step 7: Testing with quantized codes...")
    print("-" * 70)

    # Train quantizer
    quantizer_config = CodebookConfig(
        k_handshape=16, k_location=8, k_orientation=8,
        k_movement=8, k_nonmanual=8
    )
    quantizer = ProductVectorQuantizer(quantizer_config)
    quantizer.fit(feature_matrix, verbose=False)

    # Quantize features
    code_sequence = quantizer.quantize_batch(feature_matrix)

    # Convert to tensor
    codes_array = np.array([c.to_tuple() for c in code_sequence])
    codes_tensor = torch.LongTensor(codes_array).unsqueeze(0)  # (1, seq_len, 5)

    print(f"Quantized codes shape: {codes_tensor.shape}")
    print(f"Sample codes: {codes_tensor[0, 0].tolist()}")

    # Create encoder for codes
    code_encoder_config = EncoderConfig(
        input_type="codes",
        vocab_sizes=quantizer_config.get_component_sizes(),
        embedding_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
    )

    code_encoder = ASLEncoder(code_encoder_config)
    code_ctc_model = CTCModel(code_encoder, ctc_config)

    # Forward pass with codes
    with torch.no_grad():
        code_logits, _ = code_ctc_model(codes_tensor, lengths)

    print(f"Code logits shape: {code_logits.shape}")
    print(f"✓ Quantized code pipeline works!\n")

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"{'='*70}")
    print("✓ End-to-End Pipeline Test Complete!")
    print(f"{'='*70}")
    print(f"\nPipeline Summary:")
    print(f"  Video ({len(landmarks_sequence)} frames)")
    print(f"  → Landmarks: {landmarks_sequence[0].concatenate().shape}")
    print(f"  → Features:  {feature_matrix.shape}")
    print(f"  → Encoder:   {logits.shape}")
    print(f"  → Decoded:   {len(greedy_decoded[0])} glosses")
    print(f"\nAlternative with quantization:")
    print(f"  → Codes:     {codes_tensor.shape}")
    print(f"  → Encoder:   {code_logits.shape}")
    print(f"\n{'='*70}")
    print("SEQUENCE MODELING COMPLETE!")
    print(f"{'='*70}")
    print("✓ Phonology module (Phase 1)")
    print("✓ BiLSTM encoder (Phase 2)")
    print("✓ CTC head and loss (Phase 2)")
    print("✓ CTC decoding (Phase 2)")
    print("\nNext steps:")
    print("  1. Implement data loader for WLASL")
    print("  2. Implement training loop (Stage 1 & 2)")
    print("  3. Train on real data!")
    print(f"{'='*70}\n")

    return True


def main():
    """Run encoder+CTC pipeline tests."""

    # Find a sample video
    video_base = Path("/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100")

    for gloss_dir in sorted(video_base.iterdir()):
        if gloss_dir.is_dir():
            videos = list(gloss_dir.glob("*.mp4"))
            if videos:
                video_path = videos[0]
                success = test_full_pipeline(str(video_path), max_frames=100)

                if success:
                    return 0
                else:
                    return 1

    print("❌ ERROR: No videos found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
