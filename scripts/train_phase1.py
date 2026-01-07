#!/usr/bin/env python3.11
"""
Phase 1 Training: Prove the System Works

Goals:
  1. Fix blank collapse with proper data/model balance
  2. Achieve >50% accuracy on 20-class task
  3. Validate infrastructure before scaling

Changes from previous training:
  - More data: 243 train (was 20)
  - Smaller model: ~50K params (was 1.18M)
  - Filtered vocab: 21 classes (was 2001)
  - Blank penalty: Prevent collapse
  - Better regularization

Usage:
    python3.11 scripts/train_phase1.py --device cuda --num-epochs 30
"""

import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.wlasl_dataset import WLASLDataset, collate_fn
from models.encoder import ASLEncoder, EncoderConfig
from models.ctc_head import CTCModel, CTCConfig, CTCLoss, CTCDecoder


def compute_blank_ratio(logits, blank_id=0):
    """Compute ratio of blank predictions (for penalty)."""
    predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)
    blank_count = (predictions == blank_id).sum().float()
    total_count = predictions.numel()
    return blank_count / total_count


def train_epoch(model, train_loader, optimizer, ctc_loss, decoder, device, epoch, blank_penalty=0.1):
    """Train for one epoch with blank penalty."""
    model.train()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_blank_penalty = 0.0
    num_batches = 0

    correct = 0
    total = 0

    from tqdm import tqdm
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        if batch is None:
            continue

        # Move to device
        features = batch['features'].to(device)
        lengths = batch['lengths'].to(device)
        gloss_ids = batch['gloss_ids'].to(device)

        # Forward pass
        logits, output_lengths = model(features, lengths)

        # CTC loss
        targets = gloss_ids
        target_lengths = torch.ones(len(targets), dtype=torch.long, device=device)
        ctc_loss_value = ctc_loss(logits, targets, output_lengths, target_lengths)

        # Blank penalty (discourage always predicting blank)
        blank_ratio = compute_blank_ratio(logits, blank_id=0)
        blank_penalty_value = blank_penalty * blank_ratio

        # Total loss
        loss = ctc_loss_value + blank_penalty_value

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_ctc_loss += ctc_loss_value.item()
        total_blank_penalty += blank_penalty_value.item()
        num_batches += 1

        # Train accuracy
        decoded = decoder.greedy_decode(logits, output_lengths)
        for i, pred_seq in enumerate(decoded):
            if len(pred_seq) > 0:
                pred_gloss = pred_seq[0]
                true_gloss = gloss_ids[i].item()
                if pred_gloss == true_gloss:
                    correct += 1
            total += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'ctc': f'{ctc_loss_value.item():.3f}',
            'blank': f'{blank_ratio.item():.2f}',
            'acc': f'{100*correct/total:.1f}%' if total > 0 else '0%'
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_ctc_loss = total_ctc_loss / num_batches if num_batches > 0 else 0.0
    avg_blank_penalty = total_blank_penalty / num_batches if num_batches > 0 else 0.0
    train_acc = correct / total if total > 0 else 0.0

    return {
        'loss': avg_loss,
        'ctc_loss': avg_ctc_loss,
        'blank_penalty': avg_blank_penalty,
        'accuracy': train_acc,
    }


def validate(model, val_loader, ctc_loss, decoder, device, epoch):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    correct = 0
    total = 0

    # Track per-class accuracy
    class_correct = {}
    class_total = {}

    from tqdm import tqdm
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in pbar:
            if batch is None:
                continue

            # Move to device
            features = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            gloss_ids = batch['gloss_ids'].to(device)
            glosses = batch['glosses']

            # Forward pass
            logits, output_lengths = model(features, lengths)

            # CTC loss
            targets = gloss_ids
            target_lengths = torch.ones(len(targets), dtype=torch.long, device=device)
            loss = ctc_loss(logits, targets, output_lengths, target_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Decode predictions
            decoded = decoder.greedy_decode(logits, output_lengths)

            # Compute accuracy
            for i, pred_seq in enumerate(decoded):
                true_gloss = glosses[i]
                true_id = gloss_ids[i].item()

                # Track per-class
                if true_gloss not in class_total:
                    class_total[true_gloss] = 0
                    class_correct[true_gloss] = 0
                class_total[true_gloss] += 1

                if len(pred_seq) > 0:
                    pred_id = pred_seq[0]
                    if pred_id == true_id:
                        correct += 1
                        class_correct[true_gloss] += 1
                total += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100*correct/total:.1f}%' if total > 0 else '0%'
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_correct': class_correct,
        'class_total': class_total,
    }


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Phase 1 Training')
    parser.add_argument('--train-cache', type=str, required=True,
                       help='Path to training feature cache')
    parser.add_argument('--val-cache', type=str, required=True,
                       help='Path to validation feature cache')
    parser.add_argument('--metadata', type=str,
                       default='/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json',
                       help='Path to metadata.json')
    parser.add_argument('--num-epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (8 fits in 2GB VRAM)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension (64 = ~50K params)')
    parser.add_argument('--num-layers', type=int, default=1,
                       help='Number of LSTM layers')
    parser.add_argument('--blank-penalty', type=float, default=0.1,
                       help='Blank prediction penalty')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='checkpoints/phase1',
                       help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to train on')
    parser.add_argument('--use-lr-scheduler', action='store_true',
                       help='Use ReduceLROnPlateau scheduler')
    parser.add_argument('--lr-patience', type=int, default=5,
                       help='LR scheduler patience (epochs)')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                       help='LR reduction factor')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--keep-full-vocab', action='store_true',
                       help='Use full 2001-class vocabulary (like successful 36.96%% run)')

    args = parser.parse_args()

    # Check GPU availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"\n{'='*70}")
    print(f"Phase 1 Training: Prove the System Works")
    print(f"{'='*70}")
    print(f"Goal: >50% accuracy on 20-class task")
    print(f"Device: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*70}\n")

    # Load datasets
    print("Loading datasets...")
    from utils.wlasl_dataset import WLASLDataset

    train_dataset = WLASLDataset(
        data_root='/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100',
        metadata_path=args.metadata,
        split='train',
        feature_cache_path=args.train_cache,
        extract_features=False,  # Use cached features only
        keep_full_vocab=args.keep_full_vocab,  # Option to use full 2001-class vocab
    )

    val_dataset = WLASLDataset(
        data_root='/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100',
        metadata_path=args.metadata,
        split='val',
        feature_cache_path=args.val_cache,
        extract_features=False,
        keep_full_vocab=args.keep_full_vocab,
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Vocab: {train_dataset.vocab_size} classes")
    print(f"  Classes: {len(train_dataset.gloss_to_id)} unique signs")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # No multiprocessing for cached data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create model config
    encoder_config = EncoderConfig(
        input_type='features',
        input_dim=36,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,  # Configurable dropout
        bidirectional=True,
    )

    ctc_config = CTCConfig(
        vocab_size=train_dataset.vocab_size,
        blank_id=0,
        encoder_dim=encoder_config.output_dim,
        beam_width=10,
    )

    # Build model
    encoder = ASLEncoder(encoder_config).to(args.device)
    model = CTCModel(encoder, ctc_config).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created:")
    print(f"  Parameters: {num_params:,} (~{num_params/1000:.0f}K)")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Params/sample ratio: {num_params/len(train_dataset):.0f}:1")

    if num_params / len(train_dataset) > 1000:
        print(f"  ⚠ Warning: High params/sample ratio (>1000:1)")
    else:
        print(f"  ✓ Good params/sample ratio (<1000:1)")

    # Create loss and optimizer
    ctc_loss = CTCLoss(ctc_config)
    decoder = CTCDecoder(ctc_config)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Create LR scheduler (optional)
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize validation accuracy
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=1e-6,
        )
        print(f"  ✓ LR Scheduler enabled: ReduceLROnPlateau")

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n{'='*70}")
    print(f"Training")
    print(f"{'='*70}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Blank penalty: {args.blank_penalty}")
    if args.use_lr_scheduler:
        print(f"LR Scheduler: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"Early stopping: patience={args.patience} epochs")
    print(f"{'='*70}\n")

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, ctc_loss, decoder,
            args.device, epoch, args.blank_penalty
        )

        # Validate
        val_metrics = validate(
            model, val_loader, ctc_loss, decoder,
            args.device, epoch
        )

        # Track history
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])
        val_accs.append(val_metrics['accuracy'])

        # Print summary
        print(f"\nEpoch {epoch}/{args.num_epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (CTC: {train_metrics['ctc_loss']:.4f}, Blank: {train_metrics['blank_penalty']:.4f})")
        print(f"  Train Acc:  {train_metrics['accuracy']:.2%}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.2%} ({val_metrics['correct']}/{val_metrics['total']})")

        # Save checkpoint
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'encoder_config': encoder_config,
                'ctc_config': ctc_config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                # CRITICAL: Save vocab mapping for reproducibility
                'gloss_to_id': train_dataset.gloss_to_id,
                'id_to_gloss': train_dataset.id_to_gloss,
            }

            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

            best_path = Path(args.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)

            print(f"  ✓ New best accuracy! Saved checkpoint.")
        else:
            epochs_without_improvement += 1

        # Show per-class accuracy for best epoch
        if val_metrics['accuracy'] == best_val_acc and epoch % 5 == 0:
            print(f"\n  Per-class accuracy:")
            for gloss in sorted(val_metrics['class_total'].keys()):
                total = val_metrics['class_total'][gloss]
                correct = val_metrics['class_correct'].get(gloss, 0)
                acc = correct / total if total > 0 else 0.0
                print(f"    {gloss:15s}: {acc:.1%} ({correct}/{total})")

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_metrics['accuracy'])

        # Early stopping check
        if epochs_without_improvement >= args.patience:
            print(f"\n⏹ Early stopping triggered (no improvement for {args.patience} epochs)")
            print(f"  Best epoch: {best_epoch} ({best_val_acc:.2%})")
            break

    # Final summary
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"Target: >50%")

    if best_val_acc > 0.5:
        print(f"✓ SUCCESS! System validated - ready for Stage 1 pre-training")
    elif best_val_acc > 0.3:
        print(f"✓ GOOD! Significant improvement over random (5%)")
        print(f"  Consider: more data, longer training, or hyperparameter tuning")
    elif best_val_acc > 0.1:
        print(f"⚠ PARTIAL SUCCESS - better than blank collapse (0%)")
        print(f"  Need: more data or better regularization")
    else:
        print(f"✗ FAILURE - investigate blank collapse or data issues")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
