"""
PEBBLE Benchmark Evaluation.

Evaluates the trained Pebble model on standard NLP benchmarks:
  - Perplexity on WikiText-103
  - HellaSwag (commonsense reasoning)
  - LAMBADA (long-range dependencies)

Usage:
    python benchmark.py --checkpoint checkpoints/checkpoint_latest.pt \
                        --tokenizer data/tokenizer.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from datasets import load_dataset
except ImportError:
    import os
    os.system("pip install datasets")
    from datasets import load_dataset

from pebble.config import PebbleConfig
from pebble.model import PebbleLMHeadModel
from pebble.tokenizer import PebbleTokenizer
from pebble.utils import get_device


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    config_path = Path(checkpoint_path).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = PebbleConfig(**{
            k: v for k, v in config_dict.items()
            if k in PebbleConfig.__dataclass_fields__
        })
    else:
        config = PebbleConfig()

    model = PebbleLMHeadModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def eval_perplexity(model, tokenizer, device, max_samples=100, seq_len=512):
    """Evaluate perplexity on WikiText-103 test set."""
    print("\n[Benchmark] Perplexity on WikiText-103...")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    total_loss = 0.0
    total_tokens = 0
    n_samples = 0

    for example in ds:
        text = example.get("text", "").strip()
        if len(text) < 50:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 10:
            continue

        ids = ids[:seq_len + 1]
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        labels = torch.tensor([ids[1:]], dtype=torch.long, device=device)

        result = model(input_ids=input_ids, labels=labels)
        loss = result["loss"].item()

        if not math.isnan(loss) and not math.isinf(loss):
            total_loss += loss * (len(ids) - 1)
            total_tokens += len(ids) - 1
            n_samples += 1

        if n_samples >= max_samples:
            break

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"  Samples:    {n_samples}")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Avg Loss:   {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    return perplexity


@torch.no_grad()
def eval_lambada(model, tokenizer, device, max_samples=500):
    """Evaluate accuracy on LAMBADA (predict last word)."""
    print("\n[Benchmark] LAMBADA (last word prediction)...")

    try:
        ds = load_dataset("lambada", split="test")
    except Exception:
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")

    correct = 0
    total = 0

    for example in ds:
        text = example.get("text", "").strip()
        if not text:
            continue

        words = text.split()
        if len(words) < 3:
            continue

        context = " ".join(words[:-1])
        target_word = words[-1]

        context_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(" " + target_word, add_special_tokens=False)

        if not target_ids:
            continue

        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        result = model(input_ids=input_ids)
        logits = result["logits"][0, -1, :]

        predicted_token = logits.argmax().item()

        if predicted_token == target_ids[0]:
            correct += 1
        total += 1

        if total >= max_samples:
            break

    accuracy = correct / max(total, 1) * 100
    print(f"  Samples:  {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")

    return accuracy


@torch.no_grad()
def eval_hellaswag(model, tokenizer, device, max_samples=200):
    """Evaluate on HellaSwag (commonsense reasoning)."""
    print("\n[Benchmark] HellaSwag (commonsense reasoning)...")

    try:
        ds = load_dataset("Rowan/hellaswag", split="validation")
    except Exception:
        print("  [Warning] Could not load HellaSwag dataset.")
        return 0.0

    correct = 0
    total = 0

    for example in ds:
        if total >= max_samples:
            break

        context = example.get("ctx", "")
        endings = example.get("endings", [])
        label = int(example.get("label", 0))

        if not context or len(endings) < 2:
            continue

        # Score each ending by its log-likelihood given the context
        scores = []
        context_ids = tokenizer.encode(context, add_special_tokens=False)

        for ending in endings:
            ending_ids = tokenizer.encode(" " + ending, add_special_tokens=False)
            full_ids = context_ids + ending_ids

            if len(full_ids) > 1024:
                full_ids = full_ids[:1024]

            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

            result = model(input_ids=input_ids)
            logits = result["logits"]

            # Only score the ending tokens
            start = len(context_ids) - 1
            if start < 0:
                start = 0

            ending_logits = logits[0, start:, :]
            ending_targets = target_ids[0, start:]

            log_probs = F.log_softmax(ending_logits, dim=-1)
            token_log_probs = log_probs.gather(1, ending_targets.unsqueeze(1)).squeeze(1)
            score = token_log_probs.mean().item()
            scores.append(score)

        predicted = scores.index(max(scores))
        if predicted == label:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1) * 100
    print(f"  Samples:  {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")

    return accuracy


def run_benchmarks(args):
    """Run all benchmarks."""
    device = get_device()

    print("=" * 60)
    print("  PEBBLE — Benchmark Suite")
    print("=" * 60)

    model, config = load_model(args.checkpoint, device)
    tokenizer = PebbleTokenizer(args.tokenizer)

    param_count = model.count_parameters()
    print(f"  Model:      {param_count / 1e6:.1f}M parameters")
    print(f"  Device:     {device}")

    results = {}

    # Perplexity
    ppl = eval_perplexity(model, tokenizer, device, max_samples=args.max_samples)
    results["perplexity_wikitext103"] = round(ppl, 2)

    # LAMBADA
    lambada_acc = eval_lambada(model, tokenizer, device, max_samples=args.max_samples)
    results["lambada_accuracy"] = round(lambada_acc, 2)

    # HellaSwag
    hellaswag_acc = eval_hellaswag(model, tokenizer, device, max_samples=args.max_samples)
    results["hellaswag_accuracy"] = round(hellaswag_acc, 2)

    # Summary
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Perplexity (WikiText-103):  {results['perplexity_wikitext103']}")
    print(f"  LAMBADA Accuracy:           {results['lambada_accuracy']}%")
    print(f"  HellaSwag Accuracy:         {results['hellaswag_accuracy']}%")
    print("=" * 60)

    # Save results
    results_path = Path(args.checkpoint).parent / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the Pebble model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    run_benchmarks(args)
