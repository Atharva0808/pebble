"""
PEBBLE — Complete Kaggle Training Notebook.

This is a self-contained script designed to run end-to-end on Kaggle
with 2× T4 GPUs (free tier). It handles:

  1. Install dependencies
  2. Download and prepare data
  3. Train BPE tokenizer
  4. Pre-tokenize to binary
  5. Train the Mamba-2 model
  6. Evaluate on benchmarks
  7. Save artifacts to Kaggle output

Upload this single file + the pebble/ package to Kaggle and run it.

Estimated runtime: ~8-12 hours on 2× T4 for 30k steps.
"""

import os
import sys
import subprocess

# ── Step 0: Install Dependencies ───────────────────────────────────────────
print("=" * 60)
print("  PEBBLE — Kaggle Training Pipeline")
print("  Step 0: Installing dependencies")
print("=" * 60)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "tokenizers", "datasets", "wandb", "polars"])

import json
import math
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add parent dir to path for pebble package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pebble.config import PebbleConfig
from pebble.model import PebbleLMHeadModel
from pebble.tokenizer import train_tokenizer, PebbleTokenizer
from pebble.dataset import TextDataset, pretokenize_to_binary
from pebble.utils import (
    get_device, count_parameters, save_checkpoint,
    format_time, print_system_info,
)

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = "/kaggle/working/pebble_output"
DATA_DIR = "/kaggle/working/pebble_data"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"

# Training hyperparameters
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 16
SEQ_LEN = 512
MAX_STEPS = 30000
LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
DTYPE = "float16"
LOG_INTERVAL = 25
SAVE_INTERVAL = 100
EVAL_INTERVAL = 100
MAX_DATA_SAMPLES = 300000
VOCAB_SIZE = 32000

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Step 1: System Info ────────────────────────────────────────────────────
print_system_info()

device = get_device()

# ── Step 2: Download & Prepare Data ────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 2: Downloading Training Data")
print("=" * 60)

from datasets import load_dataset

all_texts = []

# WikiText-103
print("[Pebble] Downloading WikiText-103...")
try:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    for example in ds:
        text = example.get("text", "").strip()
        if len(text) > 200:
            all_texts.append(text)
    print(f"  WikiText: {len(all_texts):,} samples")
except Exception as e:
    print(f"  [Warning] WikiText failed: {e}")

# TinyStories
print("[Pebble] Downloading TinyStories...")
try:
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    count = 0
    for example in ds:
        if count >= MAX_DATA_SAMPLES // 3:
            break
        text = example.get("text", "").strip()
        if len(text) > 100:
            all_texts.append(text)
            count += 1
    print(f"  TinyStories: {count:,} samples")
except Exception as e:
    print(f"  [Warning] TinyStories failed: {e}")

print(f"\n  Total samples: {len(all_texts):,}")

# Shuffle and split
random.seed(42)
random.shuffle(all_texts)
split_idx = int(len(all_texts) * 0.95)

raw_dir = Path(DATA_DIR) / "raw"
raw_dir.mkdir(exist_ok=True)

train_txt = str(raw_dir / "train.txt")
val_txt = str(raw_dir / "val.txt")

with open(train_txt, "w", encoding="utf-8") as f:
    for text in all_texts[:split_idx]:
        f.write(text + "\n")

with open(val_txt, "w", encoding="utf-8") as f:
    for text in all_texts[split_idx:]:
        f.write(text + "\n")

print(f"  Train: {split_idx:,} samples")
print(f"  Val:   {len(all_texts) - split_idx:,} samples")

# ── Step 3: Train Tokenizer ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 3: Training BPE Tokenizer")
print("=" * 60)

tokenizer_path = f"{DATA_DIR}/tokenizer.json"
train_tokenizer(
    data_files=[train_txt],
    vocab_size=VOCAB_SIZE,
    save_path=tokenizer_path,
)

tokenizer = PebbleTokenizer(tokenizer_path)

# ── Step 4: Pre-tokenize ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 4: Pre-tokenizing to Binary")
print("=" * 60)

train_bin = f"{DATA_DIR}/train.bin"
val_bin = f"{DATA_DIR}/val.bin"

pretokenize_to_binary([train_txt], tokenizer, train_bin)
pretokenize_to_binary([val_txt], tokenizer, val_bin)

# ── Step 5: Initialize Model ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 5: Initializing Pebble Model")
print("=" * 60)

config = PebbleConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=SEQ_LEN,
)
model = PebbleLMHeadModel(config).to(device)

param_info = count_parameters(model)
print(f"  Parameters: {param_info['total_millions']}M")

# ── Step 6: Training ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 6: Training")
print("=" * 60)

train_dataset = TextDataset(train_bin, seq_len=SEQ_LEN)
val_dataset = TextDataset(val_bin, seq_len=SEQ_LEN)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=True,
)

# Optimizer
decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=LR, betas=(0.9, 0.95),
)

# Mixed precision
use_amp = DTYPE != "float32" and device.type == "cuda"
pt_dtype = torch.float16 if DTYPE == "float16" else torch.bfloat16
dtype_ctx = torch.autocast(device.type, dtype=pt_dtype) if use_amp else torch.autocast(device.type, enabled=False)
scaler = GradScaler(enabled=(DTYPE == "float16"))


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        if n >= 50:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with dtype_ctx:
            result = model(input_ids=ids, labels=labels)
        total_loss += result["loss"].item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# Training loop
model.train()
optimizer.zero_grad()
step = 0
tokens_processed = 0
t_start = time.time()
running_loss = 0.0
best_val_loss = float("inf")
data_iter = iter(train_loader)

print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Training chunks: {len(train_dataset):,}")
print(f"  Max steps: {MAX_STEPS}")
print()

while step < MAX_STEPS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with dtype_ctx:
        result = model(input_ids=input_ids, labels=labels)
        loss = result["loss"] / GRAD_ACCUM_STEPS

    scaler.scale(loss).backward()
    running_loss += loss.item()
    tokens_processed += input_ids.numel()

    if (step + 1) % GRAD_ACCUM_STEPS == 0 or step == MAX_STEPS - 1:
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    step += 1

    if step % LOG_INTERVAL == 0:
        elapsed = time.time() - t_start
        tok_s = tokens_processed / elapsed
        avg_loss = running_loss / LOG_INTERVAL * GRAD_ACCUM_STEPS
        lr = get_lr(step)
        print(
            f"  step {step:>6d}/{MAX_STEPS} | "
            f"loss {avg_loss:.4f} | lr {lr:.2e} | "
            f"tok/s {tok_s:.0f} | {format_time(elapsed)}"
        )
        running_loss = 0.0

    if step % EVAL_INTERVAL == 0:
        val_loss = evaluate()
        print(f"  [eval] step {step} | val_loss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, None, 0, step, val_loss,
                           CHECKPOINT_DIR, config)
            print(f"  [eval] New best! Saved checkpoint.")

    if step % SAVE_INTERVAL == 0:
        save_checkpoint(model, optimizer, None, 0, step, running_loss,
                       CHECKPOINT_DIR, config)

# Final save
save_checkpoint(model, optimizer, None, 0, step, running_loss,
               CHECKPOINT_DIR, config)

total_time = time.time() - t_start
print(f"\n  Training complete: {step} steps in {format_time(total_time)}")

# ── Step 7: Copy artifacts to output ───────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 7: Saving Artifacts")
print("=" * 60)

import shutil
shutil.copy2(tokenizer_path, f"{OUTPUT_DIR}/tokenizer.json")
print(f"  Tokenizer → {OUTPUT_DIR}/tokenizer.json")
print(f"  Checkpoints → {CHECKPOINT_DIR}/")

print("\n" + "=" * 60)
print("  PEBBLE TRAINING COMPLETE")
print("=" * 60)
print(f"  Model:     {param_info['total_millions']}M parameters")
print(f"  Steps:     {step}")
print(f"  Best Val:  {best_val_loss:.4f}")
print(f"  Time:      {format_time(total_time)}")
print(f"  Output:    {OUTPUT_DIR}")
print("=" * 60)