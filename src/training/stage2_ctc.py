"""
Stage 2: CTC Training
Section 6: "End-to-End CTC Training"

Implements multi-task CTC training with:
    - Primary: CTC loss for gloss sequence prediction
    - Auxiliary: Phonology prediction loss (optional)

Training objective:
    L = L_CTC + λ_phon * L_phon

This is the main training stage that produces a working ASL recognizer.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.wlasl_dataset import WLASLDataset, collate_fn
from models.encoder import ASLEncoder, EncoderConfig
from models.ctc_head import CTCModel, CTCConfig, CTCLoss, CTCDecoder


class Stage2Trainer:
    """
    Trainer for Stage 2 CTC training.

    Trains end-to-end ASL recognition model with CTC loss.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        ctc_config: CTCConfig,
        device: str = 'cpu',
    ):
        """
        Initialize trainer.

        Args:
            encoder_config: Encoder configuration
            ctc_config: CTC configuration
            device: Device to train on ('cpu' or 'cuda')
        """
        self.device = device
        self.encoder_config = encoder_config
        self.ctc_config = ctc_config

        # Build model
        encoder = ASLEncoder(encoder_config).to(device)
        self.model = CTCModel(encoder, ctc_config).to(device)

        # Loss function
        self.ctc_loss = CTCLoss(ctc_config)

        # Decoder for evaluation
        self.decoder = CTCDecoder(ctc_config)

        # Optimizer (will be set in train())
        self.optimizer: Optional["Optimizer"] = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        print(f"Stage 2 Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        assert self.optimizer is not None, "Optimizer not initialized - call train() first"
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for _, batch in enumerate(pbar):
            if batch is None:  # Skip empty batches
                continue

            # Move to device
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            gloss_ids = batch['gloss_ids'].to(self.device)

            # Forward pass
            logits, output_lengths = self.model(features, lengths)

            # Compute CTC loss
            # Target: single gloss ID per video (not a sequence)
            # For single-gloss classification, target_lengths = 1
            targets = gloss_ids  # (batch_size,)
            target_lengths = torch.ones(len(targets), dtype=torch.long, device=self.device)

            loss = self.ctc_loss(logits, targets, output_lengths, target_lengths)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                if batch is None:
                    continue

                # Move to device
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                gloss_ids = batch['gloss_ids'].to(self.device)

                # Forward pass
                logits, output_lengths = self.model(features, lengths)

                # Compute loss
                targets = gloss_ids
                target_lengths = torch.ones(len(targets), dtype=torch.long, device=self.device)
                loss = self.ctc_loss(logits, targets, output_lengths, target_lengths)

                total_loss += loss.item()
                num_batches += 1

                # Decode predictions
                decoded = self.decoder.greedy_decode(logits, output_lengths)

                # Compute accuracy (for single-gloss classification)
                for i, pred_seq in enumerate(decoded):
                    if len(pred_seq) > 0:
                        pred_gloss = pred_seq[0]  # Take first predicted gloss
                        true_gloss = gloss_ids[i].item()

                        if pred_gloss == true_gloss:
                            correct += 1
                    total += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 1e-3,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate: Learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Create checkpoint directory
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"{'='*70}\n")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader, epoch)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            self.val_losses.append(val_loss)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:.2%} ({val_metrics['correct']}/{val_metrics['total']})")

            # Save checkpoint
            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, epoch, val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  ✓ New best validation loss!")

        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")

    def _save_checkpoint(self, checkpoint_dir: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        assert self.optimizer is not None, "Optimizer not initialized"
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'encoder_config': self.encoder_config,
            'ctc_config': self.ctc_config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss,
        }

        torch.save(checkpoint, checkpoint_path)

        # Also save best model
        if val_loss == self.best_val_loss:
            best_path = Path(checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        trainer = cls(
            encoder_config=checkpoint['encoder_config'],
            ctc_config=checkpoint['ctc_config'],
            device=device,
        )

        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.train_losses = checkpoint['train_losses']
        trainer.val_losses = checkpoint['val_losses']

        print(f"✓ Loaded checkpoint from epoch {trainer.current_epoch}")
        return trainer


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2 CTC Training')
    parser.add_argument('--data-root', type=str,
                        default='/home/alex/Downloads/asl_datasets/wlasl/WLASL/videos_100',
                        help='Root directory for videos')
    parser.add_argument('--metadata', type=str,
                        default='/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json',
                        help='Path to metadata.json')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to use (for debugging)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/stage2',
                        help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers')
    parser.add_argument('--feature-cache', type=str, default=None,
                        help='Path to cached training features file')
    parser.add_argument('--val-feature-cache', type=str, default=None,
                        help='Path to cached validation features file')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Stage 2: CTC Training")
    print(f"{'='*70}\n")

    # Create datasets
    print("Loading datasets...")
    train_dataset = WLASLDataset(
        data_root=args.data_root,
        metadata_path=args.metadata,
        split='train',
        max_samples=args.max_samples,
        feature_cache_path=args.feature_cache,
        extract_features=True,  # Extract features on-the-fly if not cached
    )

    val_dataset = WLASLDataset(
        data_root=args.data_root,
        metadata_path=args.metadata,
        split='val',
        max_samples=args.max_samples // 2 if args.max_samples else None,
        feature_cache_path=args.val_feature_cache,  # Use separate val cache
        extract_features=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # Create model config
    encoder_config = EncoderConfig(
        input_type='features',
        input_dim=36,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.3,
        bidirectional=True,
    )

    ctc_config = CTCConfig(
        vocab_size=train_dataset.vocab_size,
        blank_id=0,
        encoder_dim=encoder_config.output_dim,
        beam_width=10,
    )

    # Create trainer
    trainer = Stage2Trainer(
        encoder_config=encoder_config,
        ctc_config=ctc_config,
        device=args.device,
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("✓ Training complete!")


if __name__ == "__main__":
    main()
