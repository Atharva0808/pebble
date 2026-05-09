"""
PEBBLE Tokenizer.

Trains and loads a custom Byte-Pair Encoding (BPE) tokenizer
optimized for general-purpose text. Uses HuggingFace's fast
tokenizers library for production-grade performance.
"""

import json
from pathlib import Path
from typing import Optional, Union

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors


def train_tokenizer(
    data_files: list[str],
    vocab_size: int = 32000,
    save_path: str = "tokenizer.json",
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a custom BPE tokenizer from scratch.

    Args:
        data_files: List of paths to text files for training.
        vocab_size: Target vocabulary size.
        save_path: Where to save the trained tokenizer.
        min_frequency: Minimum frequency for a token to be included.

    Returns:
        The trained Tokenizer instance.
    """
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenization: split on whitespace and punctuation (GPT-2 style)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: byte-level for roundtrip consistency
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <bos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", 1),
            ("<eos>", 2),
        ],
    )

    # Define the trainer
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train
    tokenizer.train(files=data_files, trainer=trainer)

    # Save
    tokenizer.save(save_path)
    print(f"[Pebble] Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")

    return tokenizer


class PebbleTokenizer:
    """Wrapper around the HuggingFace fast tokenizer for Pebble."""

    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> list[int]:
        """Encode a string into token IDs."""
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids = encoding.ids
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into a string."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: list[str],
        max_length: int = 2048,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> dict:
        """Batch encode multiple strings with padding.

        Returns:
            dict with 'input_ids' and 'attention_mask' (list of lists).
        """
        encodings = self.tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )

        all_ids = []
        all_masks = []

        for enc in encodings:
            ids = enc.ids[:max_length]
            mask = [1] * len(ids)

            if padding:
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len

            all_ids.append(ids)
            all_masks.append(mask)

        return {"input_ids": all_ids, "attention_mask": all_masks}
