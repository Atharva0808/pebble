# Pebble

Pebble is a 120M-parameter general-purpose small language model (SLM) built entirely from scratch based on the Mamba-2 Selective State Space architecture. 

It was engineered to completely bypass the quadratic constraints of traditional Transformer models, offering theoretically infinite context length, linear-time inference, and a constant memory footprint during generation. 

The project includes the raw PyTorch implementation of the Mamba-2 architecture, a comprehensive training and data pipeline, and a highly optimized Next.js frontend capable of running the model entirely in the browser via WebGPU.

## Features

* Linear-Time Inference: The selective scan mechanism processes tokens in O(n) time, guaranteeing that inference speed remains constant regardless of the context size.
* Constant Memory Generation: The hidden state compresses the context into a fixed-size vector. During generation, memory usage is O(1), completely eliminating the KV-cache explosion found in Transformers.
* Zero-Order Hold Discretization: Continuous-time state equations are discretized using input-dependent step sizes, allowing the model to dynamically control its temporal resolution per token.
* Pure PyTorch Implementation: Every layer, convolution, and optimization is implemented natively in PyTorch without relying on external pre-built Mamba libraries.
* Browser-Native Execution: The model can be exported to ONNX format and executed locally in any modern browser via WebGPU, removing all backend latency and server costs.

## Architecture

The model implements a deep residual architecture with 24 layers. Each layer is constructed as follows:

1. Token Embedding: Translates the 32k vocabulary into a 768-dimensional space.
2. RMSNorm: Pre-normalization applied before the main block for stable gradient flow.
3. Causal Conv1D: Extracts local context across sequence tokens.
4. Selective SSM: The core recurrent block utilizing input-dependent selection parameters to learn what to remember and what to forget.
5. Gated Output Projection: Uses a SiLU activation to gate the recurrent output back into the primary residual stream.

## Repository Structure

The repository is divided into two main environments: the machine learning backend and the frontend client.

### /ml (Machine Learning Engine)
Contains the core PyTorch model, dataset processing, and training routines.
* pebble/: The Python package containing the model architecture, configuration, and custom tokenizer.
* prepare_data.py: Downloads and tokenizes the training corpus, saving it in chunked binary formats for high-speed disk reads.
* train.py: The main training loop featuring Automatic Mixed Precision (AMP), gradient accumulation, and learning rate scheduling.
* kaggle_train.py: An orchestrated script designed to run the entire pipeline end-to-end on cloud environments like Kaggle.
* export_onnx.py: Converts the trained .pt PyTorch weights into ONNX format for web deployment.

### /web (Frontend & WebGPU Inference)
A Next.js application that serves as the landing page and inference engine.
* Built with Next.js App Router and TypeScript.
* Uses Framer Motion for UI interactions and transitions.
* Implements ONNX Runtime Web to execute the model locally on the user's GPU.
* The frontend uses a highly deliberate, minimalist Swiss-editorial design system built with vanilla CSS.

## Getting Started

### Training the Model
To train the model from scratch, navigate to the `ml` directory. Ensure you have PyTorch installed with CUDA support.

```bash
cd ml
pip install -r requirements.txt
python prepare_data.py
python train.py
```

For cloud training (e.g., Kaggle or Google Colab), simply upload the `ml` directory and run:
```bash
python kaggle_train.py
```

### Running the Web Interface
To run the frontend and the interactive model playground locally:

```bash
cd web
bun install
bun run dev
```
The interface will be available at http://localhost:3000.

## License
MIT License
