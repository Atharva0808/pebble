"""
PEBBLE Data Preparation Pipeline.

Downloads open-source datasets, trains the BPE tokenizer,
and pre-tokenizes everything into binary format — all in one script.

Usage:
    python prepare_data.py --output_dir data/
"""

import argparse
import os
import json
import random
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("[Pebble] Installing 'datasets' library...")
    os.system("pip install datasets")
    from datasets import load_dataset

from pebble.tokenizer import train_tokenizer, PebbleTokenizer
from pebble.dataset import pretokenize_to_binary


def download_and_extract(output_dir: str, max_samples: int = 500_000):
    """Download open-source training data from HuggingFace.

    Uses a combination of:
    - Cosmopedia (synthetic textbooks by HuggingFaceTB)
    - OpenWebText (high-quality web text)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    all_texts = []

    # ── Source 1: Cosmopedia (Synthetic Textbooks) ─────────────────────
    print("[Pebble] Downloading Cosmopedia (synthetic textbooks)...")
    try:
        ds = load_dataset(
            "HuggingFaceTB/cosmopedia",
            "auto_math_text",
            split="train",
            streaming=True,
        )
        count = 0
        for example in ds:
            if count >= max_samples // 2:
                break
            text = example.get("text", "")
            if len(text) > 200:
                all_texts.append(text)
                count += 1
            if count % 10000 == 0 and count > 0:
                print(f"  Cosmopedia: {count:,} samples collected")
        print(f"  Cosmopedia: {count:,} total samples")
    except Exception as e:
        print(f"  [Warning] Cosmopedia failed: {e}")
        print("  Falling back to alternative dataset...")

    # ── Source 2: WikiText (Always available, good baseline) ───────────
    print("[Pebble] Downloading WikiText-103...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        count = 0
        for example in ds:
            text = example.get("text", "")
            if len(text) > 200:
                all_texts.append(text)
                count += 1
            if count >= max_samples // 2:
                break
        print(f"  WikiText: {count:,} samples")
    except Exception as e:
        print(f"  [Warning] WikiText failed: {e}")

    # ── Source 3: TinyStories (Reasoning-dense, small) ─────────────────
    print("[Pebble] Downloading TinyStories...")
    try:
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        count = 0
        for example in ds:
            if count >= max_samples // 4:
                break
            text = example.get("text", "")
            if len(text) > 100:
                all_texts.append(text)
                count += 1
            if count % 10000 == 0 and count > 0:
                print(f"  TinyStories: {count:,} samples collected")
        print(f"  TinyStories: {count:,} total samples")
    except Exception as e:
        print(f"  [Warning] TinyStories failed: {e}")

    if not all_texts:
        print("[Pebble] ERROR: No data was downloaded. Check your internet connection.")
        return None

    # Shuffle
    random.seed(42)
    random.shuffle(all_texts)

    # Split: 95% train, 5% validation
    split_idx = int(len(all_texts) * 0.95)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]

    # Save to text files
    train_path = raw_dir / "train.txt"
    val_path = raw_dir / "val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        for text in train_texts:
            f.write(text.strip() + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for text in val_texts:
            f.write(text.strip() + "\n")

    print(f"\n[Pebble] Data saved:")
    print(f"  Train: {len(train_texts):,} samples → {train_path}")
    print(f"  Val:   {len(val_texts):,} samples → {val_path}")

    return str(train_path), str(val_path)


def run_pipeline(args):
    """Run the complete data preparation pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download Data ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Downloading Training Data")
    print("=" * 60)

    result = download_and_extract(args.output_dir, max_samples=args.max_samples)
    if result is None:
        return

    train_txt, val_txt = result

    # ── Step 2: Train Tokenizer ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Training BPE Tokenizer")
    print("=" * 60)

    tokenizer_path = str(output_dir / "tokenizer.json")
    train_tokenizer(
        data_files=[train_txt],
        vocab_size=args.vocab_size,
        save_path=tokenizer_path,
    )

    # ── Step 3: Pre-tokenize to Binary ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Pre-tokenizing to Binary Format")
    print("=" * 60)

    tokenizer = PebbleTokenizer(tokenizer_path)

    train_bin = str(output_dir / "train.bin")
    val_bin = str(output_dir / "val.bin")

    train_tokens = pretokenize_to_binary(
        [train_txt], tokenizer, train_bin,
        max_tokens=args.max_tokens,
    )
    val_tokens = pretokenize_to_binary(
        [val_txt], tokenizer, val_bin,
    )

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Tokenizer:    {tokenizer_path} (vocab={tokenizer.vocab_size})")
    print(f"  Train data:   {train_bin} ({train_tokens:,} tokens)")
    print(f"  Val data:     {val_bin} ({val_tokens:,} tokens)")
    print(f"\n  Ready for training:")
    print(f"    python train.py --data_path {train_bin} --val_data_path {val_bin}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for Pebble.")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_samples", type=int, default=500000,
                        help="Max samples to download per source.")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max tokens for training data (None = use all).")
    args = parser.parse_args()

    run_pipeline(args)
