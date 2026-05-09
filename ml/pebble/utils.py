"""
PEBBLE Utilities.

Helper functions for logging, checkpointing, and system diagnostics.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters broken down by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_millions": round(total / 1e6, 2),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    save_dir: str,
    config: Optional[object] = None,
):
    """Save a training checkpoint.

    Saves model weights, optimizer state, scheduler state,
    and training metadata for resumption.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    ckpt_path = save_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, ckpt_path)

    # Save a "latest" symlink/copy
    latest_path = save_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save config alongside
    if config is not None:
        config_path = save_dir / "config.json"
        import dataclasses
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)

    print(f"[Pebble] Checkpoint saved: {ckpt_path} (step={step}, loss={loss:.4f})")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load a training checkpoint.

    Returns:
        dict with 'epoch', 'step', and 'loss'.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(
        f"[Pebble] Checkpoint loaded: step={checkpoint['step']}, "
        f"loss={checkpoint['loss']:.4f}"
    )

    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "loss": checkpoint["loss"],
    }


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_system_info():
    """Print system and hardware diagnostics."""
    print("=" * 60)
    print("  PEBBLE — System Diagnostics")
    print("=" * 60)
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM:     {mem:.1f} GB")
    print(f"  Device:   {get_device()}")
    print("=" * 60)
