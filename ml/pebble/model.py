"""
PEBBLE Model Architecture.

Implements the Mamba-2 Selective State Space Model in pure PyTorch.
This is the core of the project — a complete, from-scratch implementation
of the SSM architecture with:
  - RMSNorm (pre-norm residual blocks)
  - Selective Scan (the heart of Mamba)
  - Causal 1D Convolution
  - Gated output projection
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PebbleConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm — skips the mean-centering step.
    Used in Llama, Mamba-2, and other modern architectures.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Selective Scan (The Core SSM Operation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def selective_scan(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Selective scan operation — the heart of the Mamba architecture.

    Implements the discretized state-space recurrence:
        h_t = Ā * h_{t-1} + B̄ * x_t
        y_t = C_t · h_t + D · x_t

    where Ā = exp(Δ · A) and B̄ = Δ · B (Zero-Order Hold discretization).

    Args:
        x:     (batch, seq_len, d_inner)   Input signal.
        delta: (batch, seq_len, d_inner)   Step sizes (input-dependent).
        A:     (d_inner, d_state)          State transition matrix (negative, log-space input).
        B:     (batch, seq_len, d_state)   Input-dependent input matrix.
        C:     (batch, seq_len, d_state)   Input-dependent output matrix.
        D:     (d_inner,)                  Skip connection (residual).

    Returns:
        y:     (batch, seq_len, d_inner)   Output signal.
    """
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # ── Discretization (Zero-Order Hold) ───────────────────────────────
    # Ā = exp(Δ · A), where A is already negative for stability
    delta_A = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    )  # (B, L, d_inner, d_state)

    # B̄·x = Δ · B · x
    delta_B_x = (
        delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    )  # (B, L, d_inner, d_state)

    # ── Sequential Scan ────────────────────────────────────────────────
    # h_t = Ā_t * h_{t-1} + B̄_t * x_t
    h = torch.zeros(
        batch, d_inner, d_state, device=x.device, dtype=x.dtype
    )
    outputs = []

    for t in range(seq_len):
        h = delta_A[:, t] * h + delta_B_x[:, t]
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)  # (B, L, d_inner)

    # ── Skip Connection ────────────────────────────────────────────────
    y = y + x * D.unsqueeze(0).unsqueeze(0)

    return y


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mamba Block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MambaBlock(nn.Module):
    """A single Mamba block.

    Architecture:
        x ─→ in_proj ─→ split(x, z) ─→ conv1d(x) ─→ SiLU(x) ─→ SSM(x)
                                         z ─→ SiLU(z)
                                         y = x * z ─→ out_proj ─→ output
    """

    def __init__(self, config: PebbleConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_inner = config.d_inner
        d_state = config.d_state
        d_conv = config.d_conv
        dt_rank = config.dt_rank

        # ── Projections ────────────────────────────────────────────────
        # Projects d_model → 2 * d_inner (split into x and gate z)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=config.bias)

        # ── Causal Convolution ─────────────────────────────────────────
        # Depthwise 1D conv for local context mixing
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # ── SSM Parameter Projections ──────────────────────────────────
        # Projects d_inner → dt_rank + 2*d_state (for delta, B, C)
        self.x_proj = nn.Linear(
            d_inner, dt_rank + d_state * 2, bias=False
        )

        # Delta (step size) projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # ── Learnable SSM Parameters ───────────────────────────────────
        # A: State transition matrix (stored in log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: Skip connection coefficient
        self.D = nn.Parameter(torch.ones(d_inner))

        # ── Output Projection ─────────────────────────────────────────
        self.out_proj = nn.Linear(d_inner, d_model, bias=config.bias)
        # Flag for scaled initialization (GPT-2/3 technique)
        self.out_proj._is_residual = True

        # Initialize dt bias for proper step-size range [0.001, 0.1]
        self._init_dt_bias()

    def _init_dt_bias(self):
        """Initialize dt_proj bias so initial step sizes are in [0.001, 0.1]."""
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.config.d_inner)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:     (batch, seq_len, d_model)
            cache: Optional dict for autoregressive generation.

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        d_state = self.config.d_state
        dt_rank = self.config.dt_rank

        # ── Input Projection & Split ───────────────────────────────────
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # ── Causal Convolution ─────────────────────────────────────────
        if cache is not None and "conv_state" in cache:
            # Inference mode: single-step with cached conv state
            conv_state = cache["conv_state"]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = x_branch.squeeze(1)
            cache["conv_state"] = conv_state
            x_branch = (conv_state * self.conv1d.weight.squeeze(1)).sum(dim=-1)
            x_branch = x_branch + self.conv1d.bias
            x_branch = x_branch.unsqueeze(1)
        else:
            # Training mode: full causal convolution
            x_conv = x_branch.transpose(1, 2)  # (B, d_inner, L)
            x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # causal trim
            x_branch = x_conv.transpose(1, 2)  # (B, L, d_inner)

        x_branch = F.silu(x_branch)

        # ── SSM Parameters ─────────────────────────────────────────────
        x_ssm = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = x_ssm.split(
            [dt_rank, d_state, d_state], dim=-1
        )

        # Project delta to full inner dimension + softplus for positivity
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        # A is stored in log-space; negate for stability (eigenvalues < 0)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # ── Selective Scan ─────────────────────────────────────────────
        if cache is not None and "ssm_state" in cache:
            # Single-step recurrence for generation
            h = cache["ssm_state"]
            delta_t = delta.squeeze(1)  # (B, d_inner)
            B_t = B.squeeze(1)          # (B, d_state)
            C_t = C.squeeze(1)          # (B, d_state)
            x_t = x_branch.squeeze(1)   # (B, d_inner)

            delta_A = torch.exp(delta_t.unsqueeze(-1) * A)  # (B, d_inner, d_state)
            delta_B_x = (
                delta_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)
            )
            h = delta_A * h + delta_B_x
            y = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            y = y + x_t * self.D
            y = y.unsqueeze(1)  # (B, 1, d_inner)
            cache["ssm_state"] = h
        else:
            y = selective_scan(x_branch, delta, A, B, C, self.D)

        # ── Gated Output ───────────────────────────────────────────────
        y = y * F.silu(z)
        output = self.out_proj(y)  # (B, L, d_model)

        return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Residual Block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ResidualBlock(nn.Module):
    """Pre-norm residual block: Norm → Mamba → Add.

    Uses residual scaling (1/√n_layers) for stable training at depth,
    following GPT-3 and PaLM methodology.
    """

    def __init__(self, config: PebbleConfig, layer_idx: int = 0):
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mamba = MambaBlock(config)
        # Scale residuals by 1/√n_layers for stable deep training
        self.residual_scale = 1.0 / math.sqrt(config.n_layers)

    def forward(
        self, x: torch.Tensor, cache: Optional[dict] = None
    ) -> torch.Tensor:
        return x + self.residual_scale * self.mamba(self.norm(x), cache=cache)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PebbleModel(nn.Module):
    """The Pebble backbone: Embedding → N × ResidualBlock → FinalNorm.

    Supports gradient checkpointing for memory-efficient training
    on resource-constrained hardware (Kaggle T4, etc.).
    """

    def __init__(self, config: PebbleConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [ResidualBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing: trades compute for ~60% memory savings."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for inference."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) — token indices.
            cache: Optional list of per-layer cache dicts for generation.

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        x = self.embedding(input_ids)
        x = self.drop(x)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if self.gradient_checkpointing and self.training and cache is None:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, layer_cache, use_reentrant=False
                )
            else:
                x = layer(x, cache=layer_cache)

        x = self.norm_f(x)
        return x


class PebbleLMHeadModel(nn.Module):
    """Pebble Language Model: Backbone + LM Head for next-token prediction."""

    def __init__(self, config: PebbleConfig):
        super().__init__()
        self.config = config
        self.backbone = PebbleModel(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embedding weights with lm_head
        if config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Re-initialize dt_proj biases (apply() above zeroed them out)
        for layer in self.backbone.layers:
            layer.mamba._init_dt_bias()

    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution.

        Output projections that feed into residual connections use
        scaled initialization (1/√(2·n_layers)) following GPT-2/3.
        """
        if isinstance(module, nn.Linear):
            std = self.config.initializer_range
            # Scale residual-contributing projections (GPT-2/3 technique)
            if hasattr(module, '_is_residual') and module._is_residual:
                std *= (2 * self.config.n_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        # Note: Conv1d uses PyTorch defaults (kaiming_uniform), which is fine.
        # MambaBlock._init_dt_bias() is called again below to restore the
        # carefully computed dt_proj bias that apply() would have zeroed out.

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cache: Optional[list] = None,
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) — shifted targets for CE loss.
            cache: Optional list of per-layer caches.

        Returns:
            dict with 'logits' and optionally 'loss'.
        """
        hidden = self.backbone(input_ids, cache=cache)
        logits = self.lm_head(hidden)  # (B, L, vocab_size)

        result = {"logits": logits}

        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            result["loss"] = loss

        return result

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_cache(self) -> list:
        """Initialize empty cache for autoregressive generation."""
        cache = []
        for _ in range(self.config.n_layers):
            cache.append({
                "conv_state": torch.zeros(
                    1, self.config.d_inner, self.config.d_conv,
                    device=next(self.parameters()).device,
                    dtype=next(self.parameters()).dtype,
                ),
                "ssm_state": torch.zeros(
                    1, self.config.d_inner, self.config.d_state,
                    device=next(self.parameters()).device,
                    dtype=next(self.parameters()).dtype,
                ),
            })
        return cache
