"""
PEBBLE Text Generation.

Autoregressive text generation with:
  - Temperature sampling
  - Top-k filtering
  - Top-p (nucleus) sampling
  - Repetition penalty

Supports both full-context and cached (fast) generation modes.

Usage:
    python generate.py --checkpoint checkpoints/checkpoint_latest.pt \
                       --tokenizer tokenizer.json \
                       --prompt "The future of AI is"
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pebble.config import PebbleConfig
from pebble.model import PebbleLMHeadModel
from pebble.tokenizer import PebbleTokenizer
from pebble.utils import get_device


def top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Apply top-k and top-p (nucleus) filtering to logits.

    Args:
        logits: (vocab_size,) raw logits for the next token.
        top_k: Keep only the top-k tokens.
        top_p: Keep tokens with cumulative probability <= top_p.

    Returns:
        Filtered logits with invalid tokens set to -inf.
    """
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float = 1.2,
) -> torch.Tensor:
    """Apply repetition penalty to discourage repeated tokens."""
    if penalty == 1.0:
        return logits

    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

    return logits


@torch.no_grad()
def generate(
    model: PebbleLMHeadModel,
    tokenizer: PebbleTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    use_cache: bool = True,
    device: torch.device = None,
) -> str:
    """Generate text autoregressively.

    Args:
        model: The Pebble language model.
        tokenizer: The Pebble tokenizer.
        prompt: Input text to continue from.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_k: Top-k filtering parameter.
        top_p: Nucleus sampling threshold.
        repetition_penalty: Penalty for repeated tokens.
        use_cache: Use SSM state caching for faster generation.
        device: Device to run on.

    Returns:
        The generated text (prompt + completion).
    """
    if device is None:
        device = get_device()

    model.eval()

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    generated_ids = list(input_ids)

    if use_cache:
        # Prefill: process the entire prompt at once to populate the cache
        cache = model.init_cache()
        # Expand cache batch dimension
        for layer_cache in cache:
            for key in layer_cache:
                layer_cache[key] = layer_cache[key].to(device)

        # Process prompt tokens one by one to build up the cache
        for i, token_id in enumerate(input_ids):
            token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
            result = model(input_ids=token_tensor, cache=cache)

        # Generate new tokens
        for _ in range(max_new_tokens):
            logits = result["logits"][0, -1, :]  # (vocab_size,)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply repetition penalty
            logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

            # Apply top-k and top-p filtering
            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop on EOS
            if next_token == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)

            # Next step with cache
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            result = model(input_ids=token_tensor, cache=cache)
    else:
        # No cache: recompute full sequence each step (slower but simpler)
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
            result = model(input_ids=input_tensor)
            logits = result["logits"][0, -1, :]

            if temperature > 0:
                logits = logits / temperature

            logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text with Pebble.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    device = get_device()

    # Load config
    config_path = Path(args.checkpoint).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = PebbleConfig(**{
            k: v for k, v in config_dict.items()
            if k in PebbleConfig.__dataclass_fields__
        })
    else:
        config = PebbleConfig()

    # Load model
    model = PebbleLMHeadModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Pebble] Model loaded from {args.checkpoint}")

    # Load tokenizer
    tokenizer = PebbleTokenizer(args.tokenizer)
    print(f"[Pebble] Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Generate
    print(f"\n{'─' * 60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'─' * 60}\n")

    t_start = time.time()
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        use_cache=not args.no_cache,
        device=device,
    )
    elapsed = time.time() - t_start

    print(output)
    print(f"\n{'─' * 60}")
    print(f"Generated in {elapsed:.2f}s")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
