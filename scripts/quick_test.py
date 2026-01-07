#!/usr/bin/env python3.11
"""
Quick test using cached features only.
"""

import sys
from pathlib import Path
import torch
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.stage2_ctc import Stage2Trainer


def main():
    print("Loading checkpoint...")
    trainer = Stage2Trainer.load_checkpoint(
        "checkpoints/real_mediapipe_cached/checkpoint_epoch_3.pt",
        device='cpu'
    )
    trainer.model.eval()

    print("\nLoading cached features...")
    with open("data/processed/features_train.pkl", "rb") as f:
        train_cache = pickle.load(f)
    with open("data/processed/features_val.pkl", "rb") as f:
        val_cache = pickle.load(f)

    print(f"Train cache: {len(train_cache)} videos")
    print(f"Val cache: {len(val_cache)} videos")

    # Load metadata to get gloss IDs
    import json
    with open("/home/alex/Documents/asl-translation-framework/data/raw/wlasl/metadata.json", "r") as f:
        metadata = json.load(f)

    # Build gloss mapping
    gloss_to_id = {}
    id_to_gloss = {}
    all_glosses = sorted(set(item['gloss'] for item in metadata))
    for idx, gloss in enumerate(all_glosses, start=1):
        gloss_to_id[gloss] = idx
        id_to_gloss[idx] = gloss

    # Find which glosses correspond to cached videos
    video_to_gloss = {}
    for item in metadata:
        gloss = item['gloss']
        for inst in item['instances']:
            video_id = inst['video_id']
            video_to_gloss[video_id] = gloss

    print("\nTesting on TRAIN set:")
    print("="*60)
    correct = 0
    total = 0

    for video_id, features in list(train_cache.items())[:5]:
        if video_id not in video_to_gloss:
            continue

        true_gloss = video_to_gloss[video_id]
        true_id = gloss_to_id[true_gloss]

        # Prepare input
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, T, 36)
        lengths = torch.LongTensor([len(features)])

        # Inference
        with torch.no_grad():
            logits, output_lengths = trainer.model(features_tensor, lengths)
            decoded = trainer.decoder.greedy_decode(logits, output_lengths)

        # Get prediction
        if len(decoded[0]) > 0:
            pred_id = decoded[0][0]
            pred_gloss = id_to_gloss.get(pred_id, f"<UNK:{pred_id}>")
        else:
            pred_id = 0
            pred_gloss = "<BLANK>"

        is_correct = pred_id == true_id
        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} {video_id}:")
        print(f"   True: {true_gloss} (ID:{true_id})")
        print(f"   Pred: {pred_gloss} (ID:{pred_id})")

        # Top-3
        probs = torch.softmax(logits[0].mean(0), dim=0)
        top3_vals, top3_ids = torch.topk(probs, k=3)
        print(f"   Top-3:")
        for i, (val, idx) in enumerate(zip(top3_vals, top3_ids)):
            g = id_to_gloss.get(idx.item(), f"<UNK:{idx.item()}>")
            print(f"      {i+1}. {g} ({val.item():.3f})")

    print(f"\nTrain Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

    print("\n\nTesting on VAL set:")
    print("="*60)
    correct = 0
    total = 0

    for video_id, features in list(val_cache.items())[:5]:
        if video_id not in video_to_gloss:
            continue

        true_gloss = video_to_gloss[video_id]
        true_id = gloss_to_id[true_gloss]

        # Prepare input
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        lengths = torch.LongTensor([len(features)])

        # Inference
        with torch.no_grad():
            logits, output_lengths = trainer.model(features_tensor, lengths)
            decoded = trainer.decoder.greedy_decode(logits, output_lengths)

        # Get prediction
        if len(decoded[0]) > 0:
            pred_id = decoded[0][0]
            pred_gloss = id_to_gloss.get(pred_id, f"<UNK:{pred_id}>")
        else:
            pred_id = 0
            pred_gloss = "<BLANK>"

        is_correct = pred_id == true_id
        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} {video_id}:")
        print(f"   True: {true_gloss} (ID:{true_id})")
        print(f"   Pred: {pred_gloss} (ID:{pred_id})")

        # Top-3
        probs = torch.softmax(logits[0].mean(0), dim=0)
        top3_vals, top3_ids = torch.topk(probs, k=3)
        print(f"   Top-3:")
        for i, (val, idx) in enumerate(zip(top3_vals, top3_ids)):
            g = id_to_gloss.get(idx.item(), f"<UNK:{idx.item()}>")
            print(f"      {i+1}. {g} ({val.item():.3f})")

    print(f"\nVal Accuracy: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()
