#!/usr/bin/env python3.11
"""Analyze WLASL100 downloaded dataset"""

import json
from pathlib import Path
from collections import Counter

VIDEO_DIR = Path('data/raw/wlasl/videos')
METADATA = Path('data/raw/wlasl/metadata.json')

with open(METADATA) as f:
    data = json.load(f)

# Count downloaded videos per gloss
downloaded = Counter()
total_expected = Counter()

for gloss_dir in VIDEO_DIR.iterdir():
    if gloss_dir.is_dir():
        gloss = gloss_dir.name
        count = len(list(gloss_dir.glob('*.mp4')))
        downloaded[gloss] = count

# Expected counts from metadata (first 100 glosses)
for gloss_entry in data[:100]:
    gloss = gloss_entry['gloss']
    total_expected[gloss] = len(gloss_entry['instances'])

print("="*70)
print("WLASL100 Dataset Analysis")
print("="*70)
print(f"\nTotal glosses: {len(downloaded)}")
print(f"Total videos downloaded: {sum(downloaded.values())}")
print(f"Total videos expected: {sum(total_expected.values())}")
print(f"Success rate: {sum(downloaded.values())/sum(total_expected.values())*100:.1f}%")

print(f"\n{'Gloss':<15} {'Downloaded':<12} {'Expected':<10} {'Rate'}")
print("-"*70)
for gloss in sorted(downloaded.keys())[:20]:
    d = downloaded[gloss]
    e = total_expected[gloss]
    rate = d/e*100 if e > 0 else 0
    print(f"{gloss:<15} {d:<12} {e:<10} {rate:>5.1f}%")

print("\n... (showing first 20 glosses)")
print("="*70)
