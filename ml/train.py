"""
PEBBLE Training Script.

Full training pipeline with:
  - Mixed precision (FP16/BF16) for memory efficiency
  - Gradient accumulation to simulate large batch sizes
  - Cosine annealing learning rate schedule
  - Weights & Biases logging
  - Periodic checkpointing
  - Resumable training from checkpoints

Optimized to run on free-tier Kaggle T4 GPUs (16GB VRAM).

Usage:
    python train.py --data_path data/train.bin --output_dir checkpoints/
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pebble.config import PebbleConfig
from pebble.model import PebbleLMHeadModel
from pebble.dataset import TextDataset
from pebble.utils import (
    get_device,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    format_time,
    print_system_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Pebble SLM.")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to pre-tokenized .bin training data.")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to pre-tokenized .bin validation data.")

    # Model
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--seq_len", type=int, default=1024)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum).")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Precision
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])

    # Logging & Saving
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default="pebble")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from.")

    return parser.parse_args()


def get_lr(step: int, args) -> float:
    """Cosine annealing with linear warmup."""
    if step < args.warmup_steps:
        return args.lr * step / args.warmup_steps
    if step >= args.max_steps:
        return args.min_lr

    decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


@torch.no_grad()
def evaluate(model, val_loader, device, dtype_ctx, max_batches: int = 50) -> float:
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with dtype_ctx:
            result = model(input_ids=input_ids, labels=labels)

        total_loss += result["loss"].item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(args):
    """Main training loop."""
    print_system_info()

    device = get_device()

    # ── Model Configuration ────────────────────────────────────────────
    config = PebbleConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
    )

    model = PebbleLMHeadModel(config).to(device)
    model.backbone.enable_gradient_checkpointing()
    param_info = count_parameters(model)
    print(f"\n[Pebble] Model initialized: {param_info['total_millions']}M parameters")
    print(f"[Pebble] Gradient checkpointing: ENABLED")
    print(f"[Pebble] Config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"d_inner={config.d_inner}, d_state={config.d_state}")

    # ── Dataset & DataLoader ───────────────────────────────────────────
    train_dataset = TextDataset(args.data_path, seq_len=args.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[Pebble] Training data: {len(train_dataset):,} chunks of {args.seq_len} tokens")

    val_loader = None
    if args.val_data_path:
        val_dataset = TextDataset(args.val_data_path, seq_len=args.seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    # ── Optimizer ──────────────────────────────────────────────────────
    # Separate weight decay for different parameter groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    # ── Mixed Precision ────────────────────────────────────────────────
    use_amp = args.dtype != "float32" and device.type == "cuda"
    pt_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    dtype_ctx = torch.autocast(device.type, dtype=pt_dtype) if use_amp else torch.autocast(device.type, enabled=False)
    scaler = torch.amp.GradScaler(device.type, enabled=(args.dtype == "float16"))

    # ── Resume from Checkpoint ─────────────────────────────────────────
    start_step = 0
    if args.resume_from:
        ckpt_info = load_checkpoint(args.resume_from, model, optimizer, device=device)
        start_step = ckpt_info["step"]

    # ── Weights & Biases ───────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"pebble-{param_info['total_millions']}M",
            config={**vars(args), **{"param_count": param_info["total_millions"]}},
        )

    # ── Training Loop ──────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    step = start_step
    tokens_processed = 0
    t_start = time.time()
    running_loss = 0.0
    grad_norm = 0.0

    print(f"\n[Pebble] Starting training from step {start_step}...")
    print(f"[Pebble] Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"[Pebble] Mixed precision: {args.dtype}")
    print()

    data_iter = iter(train_loader)

    while step < args.max_steps:
        # ── Get Batch ──────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # ── Forward Pass ───────────────────────────────────────────
        with dtype_ctx:
            result = model(input_ids=input_ids, labels=labels)
            loss = result["loss"] / args.grad_accum_steps

        # ── Backward Pass ──────────────────────────────────────────
        scaler.scale(loss).backward()

        running_loss += loss.item()
        tokens_processed += input_ids.numel()

        # ── Optimizer Step (after accumulation) ────────────────────
        if (step + 1) % args.grad_accum_steps == 0 or step == args.max_steps - 1:
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Update learning rate
            lr = get_lr(step, args)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        step += 1

        # ── Logging ────────────────────────────────────────────────
        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            tok_per_sec = tokens_processed / elapsed
            avg_loss = running_loss / args.log_interval * args.grad_accum_steps
            lr = get_lr(step, args)

            print(
                f"  step {step:>6d}/{args.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.2f} | "
                f"tok/s {tok_per_sec:.0f} | "
                f"elapsed {format_time(elapsed)}"
            )

            if args.use_wandb:
                import wandb
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/step": step,
                })

            running_loss = 0.0

        # ── Evaluation ─────────────────────────────────────────────
        if val_loader and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, dtype_ctx)
            print(f"  [eval] step {step} | val_loss {val_loss:.4f}")

            if args.use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/step": step})

        # ── Checkpointing ──────────────────────────────────────────
        if step % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, None,
                epoch=0, step=step,
                loss=running_loss,
                save_dir=args.output_dir,
                config=config,
            )

    # ── Final Save ─────────────────────────────────────────────────────
    save_checkpoint(
        model, optimizer, None,
        epoch=0, step=step,
        loss=running_loss,
        save_dir=args.output_dir,
        config=config,
    )

    total_time = time.time() - t_start
    print(f"\n[Pebble] Training complete. {step} steps in {format_time(total_time)}")
    print(f"[Pebble] Final checkpoint saved to {args.output_dir}")

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
