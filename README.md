# Pebble

**A 120M parameter general-purpose language model built from scratch on the Mamba-2 Selective State Space architecture.**

Small model, serious intelligence.

---

## What is Pebble?

Pebble is a complete, from-scratch implementation of a Selective State Space Model (SSM) language model. While the industry iterates on Transformer architectures, Pebble moves to the next generation — replacing quadratic self-attention with **linear-time recurrence**.

### Key Properties

| Property | Value |
| :--- | :--- |
| Parameters | ~120M |
| Architecture | Mamba-2 (Selective State Space) |
| Inference Complexity | O(n) linear |
| Memory Complexity | O(1) constant |
| Context Window | Theoretically infinite |
| Training Cost | $0 (Kaggle free tier) |
| Deployment | WebGPU (browser-native) |

---

## Project Structure

```
pebble/
├── ml/                         # ML Engine
│   ├── pebble/
│   │   ├── config.py           # Model hyperparameters
│   │   ├── model.py            # Mamba-2 architecture (from scratch)
│   │   ├── tokenizer.py        # Custom BPE tokenizer
│   │   ├── dataset.py          # Memory-mapped data pipeline
│   │   └── utils.py            # Checkpointing, diagnostics
│   ├── train.py                # Training script (AMP, grad accum, cosine LR)
│   ├── generate.py             # Text generation (top-k, top-p, temperature)
│   ├── export_onnx.py          # ONNX export for WebGPU deployment
│   └── requirements.txt
│
└── web/                        # Next.js Frontend
    └── app/
        ├── layout.tsx          # Root layout with SEO
        ├── globals.css         # Swiss editorial design system
        └── page.tsx            # Landing page
```

---

## Architecture

Pebble implements the **Mamba-2 Selective State Space Model**:

```
Input → Embedding → [RMSNorm → Conv1D → SiLU → Selective SSM → Gate → Project] × 24 → Norm → LM Head
```

### The Selective Scan 
```
h_t = exp(Δ_t · A) · h_{t-1} + (Δ_t · B_t) · x_t
y_t = C_t · h_t + D · x_t
```

Where Δ (delta) is **input-dependent** — the model learns what to remember and what to forget at every timestep.

---

## Quick Start

### ML Engine

```bash
# Install dependencies
cd ml
pip install -r requirements.txt

# Verify the model
python -c "
from pebble import PebbleConfig, PebbleLMHeadModel
config = PebbleConfig()
model = PebbleLMHeadModel(config)
print(f'Pebble: {model.count_parameters() / 1e6:.1f}M parameters')
"

# Train
python train.py \
    --data_path data/train.bin \
    --output_dir checkpoints/ \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --max_steps 50000 \
    --use_wandb

# Generate
python generate.py \
    --checkpoint checkpoints/checkpoint_latest.pt \
    --tokenizer tokenizer.json \
    --prompt "The future of artificial intelligence"
```

### Web Frontend

```bash
cd web
bun install
bun run dev
```

---

## Training Pipeline

1. **Data Curation**: Synthetic high-reasoning data via knowledge distillation from frontier models
2. **Tokenization**: Custom BPE tokenizer (32k vocab) trained on the curated corpus
3. **Pre-tokenization**: Convert to memory-mapped binary format for zero-overhead data loading
4. **Training**: Mixed precision (FP16), gradient accumulation, cosine annealing LR
5. **Evaluation**: MMLU, HellaSwag, and custom reasoning benchmarks
6. **Export**: ONNX → WebGPU for browser-native inference

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| Language | Python 3.10+ |
| Framework | PyTorch 2.2+ |
| Tokenizer | HuggingFace Tokenizers |
| Tracking | Weights & Biases |
| Training | Kaggle (2× T4 GPUs, free) |
| Frontend | Next.js + Bun |
| Deployment | ONNX + WebGPU |
| Design | Swiss Editorial (Vanilla CSS) |

---

## Why Mamba?

Transformers have a fundamental bottleneck: **quadratic attention**. As context length grows, memory and compute explode. This limits practical deployment on consumer hardware.

Mamba replaces attention with a **Selective State Space** recurrence:
- **Linear time**: Process any sequence length at constant speed
- **Constant memory**: Hidden state compresses the full context
- **Hardware efficient**: No KV-cache, no attention matrix materialization

Pebble proves this architecture works at the small model scale — delivering competitive reasoning with 10× fewer parameters than Transformer baselines.

