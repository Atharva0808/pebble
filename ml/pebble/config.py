"""
PEBBLE Configuration Module.

Defines the architecture hyperparameters for the Pebble Mamba-2 SLM.
Default configuration yields a ~120M parameter model optimized for
training on free-tier Kaggle T4 GPUs.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PebbleConfig:
    """Configuration for the Pebble Mamba-2 language model."""

    # ── Core Architecture ──────────────────────────────────────────────
    d_model: int = 768
    n_layers: int = 24
    vocab_size: int = 32000

    # ── Mamba SSM Parameters ───────────────────────────────────────────
    d_state: int = 16           # SSM latent state dimension (N)
    d_conv: int = 4             # Local convolution kernel width
    expand: int = 2             # Inner dimension expansion factor
    dt_rank: Optional[int] = None  # Rank of delta projection (auto = ceil(d_model/16))

    # ── Regularization ─────────────────────────────────────────────────
    dropout: float = 0.0
    bias: bool = False

    # ── Normalization ──────────────────────────────────────────────────
    rms_norm_eps: float = 1e-5

    # ── Tokenizer & Special Tokens ─────────────────────────────────────
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Sequence ───────────────────────────────────────────────────────
    max_seq_len: int = 2048
    tie_embeddings: bool = True

    # ── Initialization ─────────────────────────────────────────────────
    initializer_range: float = 0.02

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)

    @property
    def num_params_estimate(self) -> int:
        """Rough estimate of total parameter count."""
        embed = self.vocab_size * self.d_model
        per_layer = (
            self.d_model * self.d_inner * 2          # in_proj
            + self.d_inner * self.d_conv               # conv1d
            + self.d_inner * (self.d_state * 2 + self.dt_rank)  # x_proj
            + self.dt_rank * self.d_inner              # dt_proj
            + self.d_inner * self.d_state              # A_log
            + self.d_inner                             # D
            + self.d_inner * self.d_model              # out_proj
            + self.d_model                             # norm
        )
        total = embed + (per_layer * self.n_layers) + self.d_model
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model
        return total
