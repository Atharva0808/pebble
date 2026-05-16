# Pebble

A 120M-parameter general-purpose language model built entirely from scratch on the **Mamba-2 Selective State Space** architecture.

Pebble bypasses the quadratic constraints of Transformer models entirely — delivering **linear-time inference**, **theoretically infinite context**, and a **constant memory footprint** during generation. The entire implementation is written in raw PyTorch with zero external Mamba libraries.

> **Key Innovation:** While Transformers must re-examine every previous token for each new word generated (O(n²) attention), Pebble compresses context into a fixed-size hidden state that evolves with each token — achieving O(n) time complexity with O(1) memory.

## Technical Highlights

| Feature | Details |
|---|---|
| **Architecture** | Mamba-2 Selective State Space Model (24 layers, 768d) |
| **Parameters** | 120M (weight-tied embedding + LM head) |
| **Inference** | O(n) time, O(1) memory — no KV-cache |
| **Context** | Theoretically infinite (no position embeddings) |
| **Training** | Mixed-precision FP16, gradient checkpointing, cosine LR |
| **Deployment** | ONNX export → WebGPU → runs entirely in-browser |

## Architecture

Each of the 24 residual layers implements the full Mamba-2 block:

```
Input → RMSNorm → [In-Proj → Split(x, z)]
                       ↓           ↓
                  Conv1D(x)     SiLU(z)
                       ↓           ↓
                  SiLU(x)          │
                       ↓           │
              Selective SSM(x)     │
                       ↓           │
                    x * z ←────────┘
                       ↓
                   Out-Proj → + Residual → Output
```

### Core Components

- **Selective Scan (SSM Core):** Input-dependent recurrence `h_t = Ā·h_{t-1} + B̄·x_t` with Zero-Order Hold discretization. The step size Δ is a learned function of the input, enabling adaptive temporal resolution.
- **RMSNorm:** Pre-normalization for stable gradient flow (more efficient than LayerNorm).
- **Causal Conv1D:** Depthwise convolution (k=4) for local context mixing before the SSM.
- **Gated Output:** SiLU-gated projection that controls information flow from the SSM back into the residual stream.

### Training Optimizations

- **Residual Scaling:** All residual connections scaled by `1/√n_layers` (GPT-3/PaLM technique) for stable 24-layer training.
- **Scaled Initialization:** Output projections initialized with `std / √(2·n_layers)` (GPT-2/3 technique) to prevent gradient explosion at depth.
- **Gradient Checkpointing:** Trades compute for ~60% VRAM savings, enabling 1024-token sequences on free-tier T4 GPUs.
- **Cosine LR Schedule:** With linear warmup and minimum LR floor for optimal convergence.
- **Graceful Shutdown:** SIGTERM handler catches Kaggle/cloud timeout signals and saves a final checkpoint before process termination.

## Repository Structure

### `/ml` — Machine Learning Engine

| File | Purpose |
|---|---|
| `pebble/model.py` | Complete Mamba-2 architecture (SSM, Conv1D, RMSNorm, gated output) |
| `pebble/config.py` | Architecture hyperparameters as a typed dataclass |
| `pebble/tokenizer.py` | Custom BPE tokenizer (32k vocab, byte-level) |
| `pebble/dataset.py` | Memory-mapped binary dataset with numpy memmap |
| `pebble/utils.py` | Checkpointing, diagnostics, and training utilities |
| `kaggle_train.py` | End-to-end cloud training pipeline (Kaggle/Colab) |
| `train.py` | Local training with WandB integration |
| `generate.py` | Autoregressive generation (top-k, top-p, temperature, repetition penalty) |
| `benchmark.py` | Evaluation suite (WikiText perplexity, LAMBADA, HellaSwag) |
| `export_onnx.py` | PyTorch → ONNX conversion for web deployment |

### `/web` — Frontend & WebGPU Inference

- **Next.js** App Router with TypeScript
- **Framer Motion** for fluid animations and transitions
- **ONNX Runtime Web** for browser-native model inference via WebGPU
- **Swiss editorial design system** — vanilla CSS, no framework dependencies

## Getting Started

### Training

```bash
cd ml
pip install -r requirements.txt

# Local training
python prepare_data.py
python train.py --data_path data/train.bin --val_data_path data/val.bin

# Cloud training (Kaggle/Colab)
python kaggle_train.py
```

### Web Interface

```bash
cd web
bun install
bun run dev
```

The interface will be available at `http://localhost:3000`.

### ONNX Export

```bash
python export_onnx.py --checkpoint checkpoints/checkpoint_latest.pt --output pebble.onnx
```

## Why Mamba Over Transformers?

| Property | Transformer | Pebble (Mamba-2) |
|---|---|---|
| **Attention** | O(n²) quadratic | O(n) linear recurrence |
| **Memory** | Grows with sequence | Constant (fixed state) |
| **Context** | Fixed window (2k–8k) | Theoretically infinite |
| **Generation** | KV-cache dependent | Native recurrent mode |
| **Hardware** | Requires server GPU | Runs in your browser |

## License

MIT
