"""
PEBBLE Dataset.

Provides a memory-efficient, streaming-capable dataset for training
the Pebble SLM. Supports both local text files and HuggingFace datasets.
"""

import os
import random
from typing import Optional

import torch
from torch.utils.data import Dataset, IterableDataset


class TextDataset(Dataset):
    """Pre-tokenized text dataset stored as a single binary file.

    This is the most memory-efficient approach for training:
    1. Pre-tokenize all data into a flat array of token IDs.
    2. Save as a memory-mapped binary file.
    3. Slice fixed-length chunks during training.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 2048,
        dtype=torch.int32,
    ):
        """
        Args:
            data_path: Path to the .bin file of pre-tokenized data.
            seq_len: Length of each training sequence.
            dtype: Data type of the stored token IDs.
        """
        self.seq_len = seq_len

        # Memory-map the binary file (no RAM overhead for large files)
        self.data = torch.from_numpy(
            __import__("numpy").memmap(data_path, dtype="int32", mode="r")
        ).long()

        self.n_chunks = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for the target (next token)

        chunk = self.data[start:end]

        return {
            "input_ids": chunk[:-1],   # (seq_len,)
            "labels": chunk[1:],       # (seq_len,) — shifted by 1
        }


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for processing raw text files on-the-fly.

    Useful for initial experimentation before creating the binary dataset.
    """

    def __init__(
        self,
        text_files: list[str],
        tokenizer,
        seq_len: int = 2048,
        shuffle: bool = True,
    ):
        self.text_files = text_files
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle = shuffle

    def __iter__(self):
        files = list(self.text_files)
        if self.shuffle:
            random.shuffle(files)

        buffer = []

        for fpath in files:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    tokens = self.tokenizer.encode(line, add_special_tokens=False)
                    buffer.extend(tokens)

                    # Yield chunks of seq_len + 1
                    while len(buffer) >= self.seq_len + 1:
                        chunk = torch.tensor(buffer[: self.seq_len + 1], dtype=torch.long)
                        buffer = buffer[self.seq_len :]

                        yield {
                            "input_ids": chunk[:-1],
                            "labels": chunk[1:],
                        }


def pretokenize_to_binary(
    text_files: list[str],
    tokenizer,
    output_path: str,
    max_tokens: Optional[int] = None,
) -> int:
    """Pre-tokenize text files and save as a flat binary file.

    This is a one-time preprocessing step that converts raw text
    into a compact binary format for efficient training.

    Args:
        text_files: List of paths to raw text files.
        tokenizer: PebbleTokenizer instance.
        output_path: Path for the output .bin file.
        max_tokens: Optional cap on total tokens.

    Returns:
        Total number of tokens written.
    """
    import numpy as np

    all_tokens = []
    total = 0

    for fpath in text_files:
        print(f"[Pebble] Tokenizing: {fpath}")
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = tokenizer.encode(line, add_special_tokens=False)
                all_tokens.extend(tokens)
                total += len(tokens)

                if max_tokens and total >= max_tokens:
                    break
        if max_tokens and total >= max_tokens:
            break

    if max_tokens:
        all_tokens = all_tokens[:max_tokens]

    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(output_path)

    print(f"[Pebble] Saved {len(arr):,} tokens to {output_path}")
    return len(arr)
