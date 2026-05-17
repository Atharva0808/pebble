"use client";

import { useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import Image from "next/image";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.1,
      duration: 0.7,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ease: [0.16, 1, 0.3, 1] as any,
    },
  }),
};

const stagger = {
  visible: { transition: { staggerChildren: 0.06 } },
};

// ── Simulated generation for demo mode (before ONNX model is loaded) ──
const DEMO_RESPONSES: Record<string, string> = {
  default:
    "I'm running in demo mode right now — my full model is a 120M parameter Mamba-2 SSM trained from scratch. In this mode, I can answer questions about my architecture, how I was trained, and why I'm different from Transformers. Try one of the suggested prompts below, or ask me something like 'what is a state space model?' or 'how does Pebble train?'",
  hi:
    "Hello! I'm Pebble, a small language model built from scratch. I don't use attention like GPT or Llama — instead, I process text through a selective state space recurrence that runs in linear time. Ask me anything about how I work, or try one of the suggested prompts below.",
  hello:
    "Hey there! I'm Pebble — a 120M parameter language model based on the Mamba-2 architecture. Unlike Transformers, I maintain a fixed-size hidden state that compresses context, giving me constant-memory inference and theoretically infinite context length. What would you like to know?",
  "who are":
    "I'm Pebble, a small language model with 120 million parameters. I was built entirely from scratch — every layer, every convolution, every optimization is written in raw PyTorch. My architecture is based on Mamba-2, a Selective State Space Model that replaces the quadratic attention mechanism found in Transformers with a linear-time recurrence. I was trained for free on Kaggle's T4 GPUs.",
  "tell me about":
    "Pebble is a general-purpose language model that takes a fundamentally different approach to text processing. While most modern language models use Transformer attention (which scales quadratically with sequence length), Pebble uses a Selective State Space Model that processes each token in constant time. The result is a model that maintains consistent speed and memory usage regardless of how long the conversation gets.",
  "what is":
    "At its core, a Selective State Space Model is a continuous-time dynamical system that has been discretized for sequence processing. The key innovation is the selection mechanism — the model learns input-dependent parameters that control how information flows through the hidden state. When a token is relevant, the step size Δ increases, allowing more information to be absorbed. When a token is noise, Δ shrinks, effectively ignoring it. This gives the model the ability to selectively remember and forget — a capability that standard RNNs lack.",
  how: "The training process begins with data curation — we use a combination of WikiText-103 and TinyStories to create a diverse corpus of high-quality text. A custom BPE tokenizer with 32,000 vocabulary entries is trained on this data. The corpus is then pre-tokenized into a memory-mapped binary file for zero-overhead data loading. Training uses mixed-precision (FP16) with gradient accumulation to simulate large batch sizes on free-tier T4 GPUs, with a cosine annealing learning rate schedule that warms up over the first 150 steps.",
  why: "Transformers have dominated NLP since 2017, but they carry a fundamental inefficiency: quadratic attention. For every token generated, the model must re-examine every previous token. This means that as conversations get longer, Transformers get slower and consume more memory. The Mamba architecture eliminates this bottleneck entirely. By replacing attention with a selective state space recurrence, we achieve linear-time inference with constant memory — making it possible to process arbitrarily long sequences at consistent speed.",
  "explain":
    "The Mamba-2 architecture works by maintaining a hidden state vector that gets updated with each input token. Think of it as a compressed summary of everything the model has read so far. At each step, the model decides how much of the new token to absorb and how much of the old state to retain — this is the 'selective' part. The math behind it comes from control theory: a continuous-time linear system (dx/dt = Ax + Bu) is discretized using Zero-Order Hold, giving us a recurrence h_t = Ā·h_{t-1} + B̄·x_t that can be computed in linear time.",
  code: 'def selective_scan(x, delta, A, B, C, D):\n    """The core SSM recurrence."""\n    h = torch.zeros(batch, d_inner, d_state)\n    outputs = []\n    for t in range(seq_len):\n        # Discretize: Ā = exp(Δ·A)\n        h = torch.exp(delta[:, t] * A) * h\n        h = h + delta[:, t] * B[:, t] * x[:, t]\n        y = (h * C[:, t]).sum(-1)\n        outputs.append(y)\n    return torch.stack(outputs, dim=1) + x * D',
};

function getResponse(prompt: string): string {
  const lower = prompt.toLowerCase().trim();
  for (const [key, value] of Object.entries(DEMO_RESPONSES)) {
    if (key !== "default" && lower.includes(key)) {
      return value;
    }
  }
  return DEMO_RESPONSES.default;
}

export default function PlaygroundPage() {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [tokensGenerated, setTokensGenerated] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(256);
  const outputRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef(false);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setOutput("");
    setTokensGenerated(0);
    abortRef.current = false;

    const response = getResponse(prompt);
    const words = response.split(" ");
    const startTime = performance.now();

    // Simulate token-by-token generation with realistic timing
    for (let i = 0; i < words.length && i < maxTokens; i++) {
      if (abortRef.current) break;

      await new Promise((r) => setTimeout(r, 20 + Math.random() * 30));
      setOutput((prev) => (prev ? prev + " " + words[i] : words[i]));
      setTokensGenerated(i + 1);
      setElapsedMs(performance.now() - startTime);

      if (outputRef.current) {
        outputRef.current.scrollTop = outputRef.current.scrollHeight;
      }
    }

    setElapsedMs(performance.now() - startTime);
    setIsGenerating(false);
  }, [prompt, isGenerating, maxTokens]);

  const handleStop = () => {
    abortRef.current = true;
    setIsGenerating(false);
  };

  const tokPerSec =
    elapsedMs > 0 ? ((tokensGenerated / elapsedMs) * 1000).toFixed(1) : "0";

  return (
    <>
      {/* ── Navigation ────────────────────────────────────────────── */}
      <nav className="nav">
        <Link href="/" className="nav-logo" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Image src="/pebble_logo_v3.png" alt="Pebble Logo" width={26} height={26} priority />
          Pebble
        </Link>
        <ul className="nav-links">
          <li>
            <Link href="/#architecture">Architecture</Link>
          </li>
          <li>
            <Link href="/#benchmarks">Benchmarks</Link>
          </li>
          <li>
            <Link href="/#code">Code</Link>
          </li>
          <li>
            <Link href="/#comparison">Comparison</Link>
          </li>
          <li>
            <Link href="/playground" style={{ color: "var(--stone-950)" }}>
              Playground
            </Link>
          </li>
        </ul>
      </nav>

      {/* ── Playground ────────────────────────────────────────────── */}
      <section
        className="section container"
        style={{ paddingTop: "calc(var(--nav-height) + var(--space-4xl))" }}
      >
        <motion.div
          initial="hidden"
          animate="visible"
          variants={stagger}
        >
          <motion.span className="label" custom={0} variants={fadeUp}>
            Playground
          </motion.span>
          <motion.h2
            custom={1}
            variants={fadeUp}
            style={{ marginTop: "var(--space-md)" }}
          >
            Talk to Pebble.
          </motion.h2>
          <motion.p
            custom={2}
            variants={fadeUp}
            style={{ marginTop: "var(--space-md)" }}
          >
            This demo simulates Pebble&apos;s generation behavior. When the
            trained ONNX model is loaded, inference will run entirely in your
            browser via WebGPU — no server, no API calls, zero latency.
          </motion.p>

          <motion.div
            custom={3}
            variants={fadeUp}
            style={{ marginTop: "var(--space-3xl)" }}
          >
            {/* ── Controls ──────────────────────────────────────── */}
            <div className="playground-controls">
              <div className="control-group">
                <label className="control-label">Temperature</label>
                <div className="control-row">
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="control-slider"
                  />
                  <span className="control-value">{temperature.toFixed(1)}</span>
                </div>
              </div>
              <div className="control-group">
                <label className="control-label">Max Tokens</label>
                <div className="control-row">
                  <input
                    type="range"
                    min="32"
                    max="512"
                    step="32"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                    className="control-slider"
                  />
                  <span className="control-value">{maxTokens}</span>
                </div>
              </div>
            </div>

            {/* ── Input ─────────────────────────────────────────── */}
            <div className="playground-input-wrapper">
              <textarea
                className="playground-input"
                placeholder="Enter a prompt... (try: What is a state space model?)"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleGenerate();
                  }
                }}
                rows={3}
              />
              <div className="playground-actions">
                {isGenerating ? (
                  <button className="btn-stop" onClick={handleStop}>
                    Stop
                  </button>
                ) : (
                  <button
                    className="btn-generate"
                    onClick={handleGenerate}
                    disabled={!prompt.trim()}
                  >
                    Generate
                  </button>
                )}
              </div>
            </div>

            {/* ── Output ────────────────────────────────────────── */}
            <div className="playground-output" ref={outputRef}>
              {output ? (
                <p className="output-text">{output}</p>
              ) : (
                <p className="output-placeholder">
                  Generation will appear here...
                </p>
              )}
            </div>

            {/* ── Stats ─────────────────────────────────────────── */}
            <div className="playground-stats">
              <div className="stat-item">
                <span className="stat-label">Tokens</span>
                <span className="stat-value">{tokensGenerated}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Speed</span>
                <span className="stat-value">{tokPerSec} tok/s</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Elapsed</span>
                <span className="stat-value">
                  {(elapsedMs / 1000).toFixed(2)}s
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Runtime</span>
                <span className="stat-value">WebGPU (Demo)</span>
              </div>
            </div>
          </motion.div>

          {/* ── Suggested Prompts ──────────────────────────────── */}
          <motion.div
            custom={4}
            variants={fadeUp}
            style={{ marginTop: "var(--space-3xl)" }}
          >
            <span className="label">Try These</span>
            <div className="suggestions">
              {[
                "What is a state space model?",
                "How does Pebble train on free GPUs?",
                "Why is Mamba faster than a Transformer?",
                "Show me the code for selective scan",
              ].map((s) => (
                <button
                  key={s}
                  className="suggestion-chip"
                  onClick={() => {
                    setPrompt(s);
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────── */}
      <footer className="footer container">
        <div className="footer-content">
          <div className="footer-left">
            <div className="footer-logo" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <Image src="/pebble_logo_v3.png" alt="Pebble Logo" width={32} height={32} />
              Pebble
            </div>
            <p className="footer-tagline">
              A 120M parameter Mamba-2 language model. Built from scratch.
              Trained for free. Deployed everywhere.
            </p>
          </div>
          <div className="footer-links">
            <Link href="/">Home</Link>
            <a href="https://github.com/Atharva0808/pebble" target="_blank" rel="noopener" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
                <path d="M9 18c-4.51 2-5-2-7-2" />
              </svg>
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </>
  );
}
