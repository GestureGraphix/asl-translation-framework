"""Stage 1: Self-Supervised Phonological Pre-training

Implements Section 7.2 of the paper:
"Stage 1: Phonological Pre-training on Unlabeled Video"

Training Objective:
    L_total = L_contrast + λ_phon * L_phon

    where:
    - L_contrast: NT-Xent contrastive loss for temporal coherence
    - L_phon: Phonological reconstruction loss (VQ)
    - λ_phon: Weight for phonological loss (default: 1.0)

Key Components:
    1. Temporal augmentations (crop, speed, noise)
    2. Encoder network (BiLSTM)
    3. Product VQ quantizer for phonology
    4. Contrastive learning (positive/negative pairs)

Data Requirements:
    - Unlabeled sign language videos
    - No gloss labels needed
    - Leverages all available WLASL data

Output:
    - Pre-trained encoder weights
    - Learned phonological codebooks
    - Ready for Stage 2 CTC training
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.encoder import ASLEncoder, EncoderConfig
from src.phonology.augmentations import TemporalAugmentation, create_contrastive_batch
from src.phonology.contrastive_loss import ContrastiveLearner
from src.phonology.vq_loss import PhonologicalReconstructionLoss


class UnlabeledVideoDataset(Dataset):
    """Dataset for unlabeled sign language videos.

    Loads cached MediaPipe features without requiring gloss labels.
    Used for self-supervised Stage 1 pre-training.
    """

    def __init__(self, feature_cache_path: str):
        """Load cached features.

        Args:
            feature_cache_path: Path to pickle file with features
        """
        import pickle

        print(f"Loading unlabeled features from {feature_cache_path}...")

        with open(feature_cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        # Handle different cache formats
        self.features_list = []
        if isinstance(cache_data, dict):
            # Dict format: {video_id: features}
            for video_id, features in cache_data.items():
                if isinstance(features, np.ndarray):
                    self.features_list.append(torch.FloatTensor(features))
                elif isinstance(features, dict) and 'features' in features:
                    self.features_list.append(torch.FloatTensor(features['features']))
        elif isinstance(cache_data, list):
            # List format: [{'features': ...}, ...]
            for item in cache_data:
                if isinstance(item, dict) and 'features' in item:
                    self.features_list.append(torch.FloatTensor(item['features']))
                elif isinstance(item, np.ndarray):
                    self.features_list.append(torch.FloatTensor(item))

        print(f"  Loaded {len(self.features_list)} videos")
        print(f"  Average length: {np.mean([len(f) for f in self.features_list]):.1f} frames")

    def __len__(self) -> int:
        return len(self.features_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features_list[idx]


def collate_fn_stage1(batch):
    """Collate function for Stage 1 training.

    Creates augmented pairs and pads sequences.
    """
    # Return list of tensors (will be augmented in training loop)
    return batch


class Stage1Trainer:
    """Trainer for Stage 1 self-supervised phonological pre-training."""

    def __init__(
        self,
        config: Dict,
        encoder: nn.Module,
        vq_loss: PhonologicalReconstructionLoss,
        contrastive_learner: ContrastiveLearner,
        augmenter: TemporalAugmentation,
        device: torch.device
    ):
        self.config = config
        self.encoder = encoder
        self.vq_loss = vq_loss
        self.contrastive_learner = contrastive_learner
        self.augmenter = augmenter
        self.device = device

        # Hyperparameters
        self.lambda_phon = config['lambda_phon']

        # Optimizer (train encoder + VQ + projection head together)
        self.optimizer = optim.Adam(
            list(encoder.parameters()) +
            list(vq_loss.parameters()) +
            list(contrastive_learner.projection_head.parameters()),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )

        # Tracking
        self.train_losses = []
        self.best_loss = float('inf')
        self.epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader for unlabeled videos

        Returns:
            Dict of average losses
        """
        self.encoder.train()
        self.vq_loss.train()
        self.contrastive_learner.train()

        epoch_losses = {
            'total': 0.0,
            'contrastive': 0.0,
            'phonological': 0.0,
            'vq': 0.0,
            'reconstruction': 0.0,
        }

        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}")

        for batch in progress_bar:
            # Batch is a list of variable-length feature tensors
            # Create augmented pairs
            batch_size = self.config['batch_size']

            # Sample and create pairs
            anchors_list = []
            positives_list = []
            anchor_lengths = []
            positive_lengths = []

            for features in batch:
                # Move to device
                features = features.to(self.device)

                # Create augmented pair
                view1, view2 = self.augmenter.create_positive_pair(features)

                anchors_list.append(view1)
                positives_list.append(view2)
                anchor_lengths.append(len(view1))
                positive_lengths.append(len(view2))

            # Pad sequences
            max_anchor_len = max(anchor_lengths)
            max_positive_len = max(positive_lengths)

            anchors = torch.zeros(len(batch), max_anchor_len, 36, device=self.device)
            positives = torch.zeros(len(batch), max_positive_len, 36, device=self.device)

            for i, (anchor, positive) in enumerate(zip(anchors_list, positives_list)):
                anchors[i, :len(anchor)] = anchor
                positives[i, :len(positive)] = positive

            anchor_lengths = torch.tensor(anchor_lengths, device=self.device)
            positive_lengths = torch.tensor(positive_lengths, device=self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # 1. Contrastive loss
            contrastive_loss = self.contrastive_learner(
                anchors, positives,
                anchor_lengths, positive_lengths
            )

            # 2. Phonological reconstruction loss (on anchors)
            # Flatten for VQ (process all timesteps)
            batch_frames = []
            for i in range(len(batch)):
                batch_frames.append(anchors[i, :anchor_lengths[i]])

            all_frames = torch.cat(batch_frames, dim=0)  # (T_total, 36)

            # Apply VQ
            quantized, phon_losses, indices = self.vq_loss(all_frames)

            phonological_loss = phon_losses['total']

            # 3. Total loss
            total_loss = contrastive_loss + self.lambda_phon * phonological_loss

            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.vq_loss.parameters()) +
                list(self.contrastive_learner.projection_head.parameters()),
                max_norm=self.config.get('grad_clip', 1.0)
            )
            self.optimizer.step()

            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['contrastive'] += contrastive_loss.item()
            epoch_losses['phonological'] += phonological_loss.item()
            epoch_losses['vq'] += phon_losses['vq'].item()
            epoch_losses['reconstruction'] += phon_losses['reconstruction'].item()

            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'contrast': contrastive_loss.item(),
                'phon': phonological_loss.item(),
            })

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Step scheduler
        self.scheduler.step()

        self.epoch += 1

        return epoch_losses

    def save_checkpoint(self, checkpoint_dir: str, is_best: bool = False):
        """Save training checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
            is_best: Whether this is the best model so far
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'vq_state_dict': self.vq_loss.state_dict(),
            'projection_head_state_dict': self.contrastive_learner.projection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'config': self.config,
        }

        # Save latest
        latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best checkpoint (loss: {self.best_loss:.4f})")

        # Save periodic
        if self.epoch % self.config.get('save_every', 10) == 0:
            epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{self.epoch}.pt')
            torch.save(checkpoint, epoch_path)


def main(args):
    """Main training function."""

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Stage 1: Self-Supervised Phonological Pre-training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"{'='*70}\n")

    # Create dataset
    dataset = UnlabeledVideoDataset(args.train_cache)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_stage1
    )

    # Create models
    encoder_config = EncoderConfig(
        input_type="features",
        input_dim=36,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.3),
        bidirectional=True
    )
    encoder = ASLEncoder(encoder_config).to(device)

    vq_loss = PhonologicalReconstructionLoss(
        commitment_cost=config.get('commitment_cost', 0.25)
    ).to(device)

    contrastive_learner = ContrastiveLearner(
        encoder=encoder,
        input_dim=config['hidden_dim'] * 2,  # BiLSTM output
        projection_dim=config.get('projection_dim', 128),
        temperature=config.get('temperature', 0.07)
    ).to(device)

    # Create augmenter
    augmenter = TemporalAugmentation(
        crop_ratio=tuple(config.get('crop_ratio', [0.7, 1.0])),
        speed_range=tuple(config.get('speed_range', [0.8, 1.2])),
        noise_std=config.get('noise_std', 0.05),
        seed=config.get('seed', 42)
    )

    # Create trainer
    trainer = Stage1Trainer(
        config=config,
        encoder=encoder,
        vq_loss=vq_loss,
        contrastive_learner=contrastive_learner,
        augmenter=augmenter,
        device=device
    )

    # Training loop
    print(f"Training for {config['num_epochs']} epochs...")
    print(f"{'='*70}\n")

    for epoch in range(config['num_epochs']):
        epoch_losses = trainer.train_epoch(dataloader)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Total loss: {epoch_losses['total']:.4f}")
        print(f"  Contrastive: {epoch_losses['contrastive']:.4f}")
        print(f"  Phonological: {epoch_losses['phonological']:.4f}")
        print(f"  VQ: {epoch_losses['vq']:.4f}")
        print(f"  Reconstruction: {epoch_losses['reconstruction']:.4f}")

        # Track and save
        trainer.train_losses.append(epoch_losses)

        is_best = epoch_losses['total'] < trainer.best_loss
        if is_best:
            trainer.best_loss = epoch_losses['total']

        trainer.save_checkpoint(args.checkpoint_dir, is_best=is_best)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best loss: {trainer.best_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Phonological Pre-training")
    parser.add_argument('--config', type=str, default='configs/stage1.yaml',
                       help='Path to config file')
    parser.add_argument('--train-cache', type=str, required=True,
                       help='Path to training feature cache')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/stage1',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()
    main(args)
