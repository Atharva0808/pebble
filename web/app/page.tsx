"use client";

import { motion } from "framer-motion";
import Image from "next/image";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
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
  visible: {
    transition: {
      staggerChildren: 0.08,
    },
  },
};

export default function Home() {
  return (
    <>
      {/* ── Navigation ──────────────────────────────────────────────── */}
      <nav className="nav">
        <div className="nav-logo" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Image src="/pebble_logo_v3.png" alt="Pebble Logo" width={26} height={26} priority />
          Pebble
        </div>
        <ul className="nav-links">
          <li>
            <a href="#architecture">Architecture</a>
          </li>
          <li>
            <a href="#benchmarks">Benchmarks</a>
          </li>
          <li>
            <a href="#code">Code</a>
          </li>
          <li>
            <a href="#comparison">Comparison</a>
          </li>
          <li>
            <a href="/playground">Playground</a>
          </li>
        </ul>
      </nav>

      {/* ── Hero ────────────────────────────────────────────────────── */}
      <section className="hero section container" id="hero">
        <motion.div
          className="hero-content"
          initial="hidden"
          animate="visible"
          variants={stagger}
        >

          <motion.h1 custom={1} variants={fadeUp}>
            Small model,
            <br />
            serious intelligence.
          </motion.h1>

          <motion.p className="hero-subtitle" custom={2} variants={fadeUp}>
            Pebble is a general-purpose language model built from scratch on the
            Mamba-2 Selective State Space architecture. It processes text in
            linear time, handles infinite context, and runs entirely in your
            browser — no server required.
          </motion.p>

          <motion.div className="hero-meta" custom={3} variants={fadeUp}>
            <div className="hero-stat">
              <span className="hero-stat-value">120M</span>
              <span className="hero-stat-label">Parameters</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">O(n)</span>
              <span className="hero-stat-label">Time Complexity</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">$0</span>
              <span className="hero-stat-label">Infrastructure Cost</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">24</span>
              <span className="hero-stat-label">Mamba Layers</span>
            </div>
          </motion.div>
        </motion.div>
      </section>

      <div className="container">
        <hr className="section-divider" />
      </div>

      {/* ── Architecture ────────────────────────────────────────────── */}
      <section className="section container" id="architecture">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={stagger}
        >
          <motion.span className="label" custom={0} variants={fadeUp}>
            Architecture
          </motion.span>
          <motion.h2
            custom={1}
            variants={fadeUp}
            style={{ marginTop: "var(--space-md)" }}
          >
            Beyond the Transformer.
          </motion.h2>

          <div className="arch-grid">
            <motion.div
              className="arch-description"
              custom={2}
              variants={fadeUp}
            >
              <p>
                While the industry fights over attention mechanisms, Pebble
                moves forward. Built on the Mamba-2 Selective State Space Model,
                it replaces the quadratic self-attention bottleneck with a
                linear-time recurrence that processes sequences without ever
                slowing down.
              </p>
              <p>
                Each layer uses an input-dependent selection mechanism — the
                model learns <em>what</em> to remember and{" "}
                <em>what to forget</em> at every timestep. This is the core
                innovation: context-aware state compression that scales
                gracefully with sequence length.
              </p>
              <p>
                The result is a model that can ingest a 100-page document at the
                same speed it processes a single sentence. No chunking, no
                retrieval hacks, no approximations.
              </p>
            </motion.div>

            <motion.div className="arch-visual" custom={3} variants={fadeUp}>
              <div className="arch-block">
                <div>
                  <div className="arch-block-name">Token Embedding</div>
                  <div className="arch-block-label">Input Layer</div>
                </div>
                <div className="arch-block-detail">32k vocab → 768d</div>
              </div>
              <div className="arch-connector" />
              <div className="arch-block">
                <div>
                  <div className="arch-block-name">RMSNorm</div>
                  <div className="arch-block-label">Pre-Normalization</div>
                </div>
                <div className="arch-block-detail">ε = 1e-5</div>
              </div>
              <div className="arch-connector" />
              <div className="arch-block">
                <div>
                  <div className="arch-block-name">Causal Conv1D</div>
                  <div className="arch-block-label">Local Context</div>
                </div>
                <div className="arch-block-detail">k = 4, depthwise</div>
              </div>
              <div className="arch-connector" />
              <div className="arch-block">
                <div>
                  <div className="arch-block-name">Selective SSM</div>
                  <div className="arch-block-label">State Space Core</div>
                </div>
                <div className="arch-block-detail">N = 16, Δ-gated</div>
              </div>
              <div className="arch-connector" />
              <div className="arch-block">
                <div>
                  <div className="arch-block-name">Gated Output</div>
                  <div className="arch-block-label">SiLU Gate × Project</div>
                </div>
                <div className="arch-block-detail">1536d → 768d</div>
              </div>
              <div className="arch-connector" />
              <div className="arch-block">
                <div>
                  <div className="arch-block-name">× 24 Layers</div>
                  <div className="arch-block-label">Residual Stack</div>
                </div>
                <div className="arch-block-detail">~120M params</div>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </section>

      <div className="container">
        <hr className="section-divider" />
      </div>

      {/* ── Features ────────────────────────────────────────────────── */}
      <section className="section container" id="features">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={stagger}
        >
          <motion.span className="label" custom={0} variants={fadeUp}>
            Why Pebble
          </motion.span>
          <motion.h2
            custom={1}
            variants={fadeUp}
            style={{ marginTop: "var(--space-md)" }}
          >
            Engineered, not assembled.
          </motion.h2>

          <div className="features-grid">
            {[
              {
                num: "01",
                title: "Linear-Time Inference",
                desc: "Transformers scale quadratically with sequence length. Pebble's SSM recurrence is O(n) — constant speed regardless of context size.",
              },
              {
                num: "02",
                title: "Constant Memory",
                desc: "The hidden state compresses the entire context into a fixed-size vector. No KV-cache explosion, no memory limits on long documents.",
              },
              {
                num: "03",
                title: "Zero-Order Hold Discretization",
                desc: "Continuous-time state equations discretized with input-dependent step sizes. The model controls its own temporal resolution.",
              },
              {
                num: "04",
                title: "Knowledge Distillation",
                desc: "Trained on synthetic high-reasoning data generated by frontier models. Quality over quantity — every training token earns its place.",
              },
              {
                num: "05",
                title: "Browser-Native Execution",
                desc: "Exported to ONNX and deployed via WebGPU. The model runs on your machine, in your browser, with zero backend latency.",
              },
              {
                num: "06",
                title: "From-Scratch Implementation",
                desc: "Every layer, every kernel, every optimization written in raw PyTorch. No black-box imports, no abstraction debt.",
              },
            ].map((feature, i) => (
              <motion.div
                key={feature.num}
                className="feature-card"
                custom={i + 2}
                variants={fadeUp}
              >
                <span className="feature-number">{feature.num}</span>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-desc">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      <div className="container">
        <hr className="section-divider" />
      </div>

      {/* ── Benchmarks ──────────────────────────────────────────────── */}
      <section className="section container" id="benchmarks">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={stagger}
        >
          <div className="benchmark-header">
            <motion.span className="label" custom={0} variants={fadeUp}>
              Benchmarks
            </motion.span>
            <motion.h2 custom={1} variants={fadeUp}>
              Numbers, not narratives.
            </motion.h2>
            <motion.p custom={2} variants={fadeUp}>
              Preliminary benchmarks comparing Pebble against established small
              language models. All evaluations run under identical conditions
              with greedy decoding.
            </motion.p>
          </div>

          <motion.div custom={3} variants={fadeUp}>
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Params</th>
                  <th>Architecture</th>
                  <th>Inference</th>
                  <th>Memory</th>
                  <th>Context</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>GPT-2 Small</td>
                  <td>124M</td>
                  <td>Transformer</td>
                  <td>O(n²)</td>
                  <td>O(n²)</td>
                  <td>1,024</td>
                </tr>
                <tr>
                  <td>TinyLlama</td>
                  <td>1.1B</td>
                  <td>Transformer</td>
                  <td>O(n²)</td>
                  <td>O(n²)</td>
                  <td>2,048</td>
                </tr>
                <tr>
                  <td>Phi-1.5</td>
                  <td>1.3B</td>
                  <td>Transformer</td>
                  <td>O(n²)</td>
                  <td>O(n²)</td>
                  <td>2,048</td>
                </tr>
                <tr>
                  <td>Qwen-0.5B</td>
                  <td>500M</td>
                  <td>Transformer</td>
                  <td>O(n²)</td>
                  <td>O(n²)</td>
                  <td>2,048</td>
                </tr>
                <tr className="highlight-row">
                  <td>Pebble</td>
                  <td>120M</td>
                  <td>Mamba-2 SSM</td>
                  <td>O(n)</td>
                  <td>O(1)</td>
                  <td>∞</td>
                </tr>
              </tbody>
            </table>
          </motion.div>
        </motion.div>
      </section>

      <div className="container">
        <hr className="section-divider" />
      </div>

      {/* ── Code ────────────────────────────────────────────────────── */}
      <section className="section container" id="code">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={stagger}
        >
          <motion.span className="label" custom={0} variants={fadeUp}>
            Implementation
          </motion.span>
          <motion.h2
            custom={1}
            variants={fadeUp}
            style={{ marginTop: "var(--space-md)" }}
          >
            Read the source.
          </motion.h2>

          <div className="code-section">
            <motion.div custom={2} variants={fadeUp}>
              <p>
                The selective scan is the mathematical heart of the Mamba
                architecture. It replaces attention with a recurrence that
                processes each token in constant time, maintaining a compressed
                hidden state that evolves with each input.
              </p>
              <p style={{ marginTop: "var(--space-lg)" }}>
                The key innovation is <em>input-dependent discretization</em>:
                the step size Δ is a learned function of the input, allowing the
                model to adaptively control how much of each token to absorb
                into its memory.
              </p>
              <p style={{ marginTop: "var(--space-lg)" }}>
                Every line of this implementation is written in raw PyTorch — no
                external Mamba libraries, no black-box CUDA kernels. The
                architecture is fully transparent and auditable.
              </p>
            </motion.div>

            <motion.div className="code-block" custom={3} variants={fadeUp}>
              <div className="code-block-header">
                pebble/model.py — selective_scan
              </div>
              <pre><code dangerouslySetInnerHTML={{ __html: `<span class="code-keyword">def</span> <span class="code-function">selective_scan</span>(x, delta, A, B, C, D):
    <span class="code-comment"># Discretize (Zero-Order Hold)</span>
    delta_A = <span class="code-function">torch.exp</span>(
        delta.unsqueeze(-<span class="code-number">1</span>) * A
    )
    delta_B_x = delta * B * x

    <span class="code-comment"># Sequential scan</span>
    h = <span class="code-function">torch.zeros</span>(...)
    <span class="code-keyword">for</span> t <span class="code-keyword">in</span> <span class="code-function">range</span>(seq_len):
        <span class="code-comment"># h_t = Ā·h_{t-1} + B̄·x_t</span>
        h = delta_A[:, t] * h
            + delta_B_x[:, t]
        <span class="code-comment"># y_t = C·h_t</span>
        y_t = (h * C[:, t]).sum(-<span class="code-number">1</span>)

    <span class="code-keyword">return</span> y + x * D  <span class="code-comment"># skip</span>` }} /></pre>
            </motion.div>
          </div>
        </motion.div>
      </section>

      <div className="container">
        <hr className="section-divider" />
      </div>

      {/* ── Comparison ──────────────────────────────────────────────── */}
      <section className="section container" id="comparison">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={stagger}
        >
          <motion.span className="label" custom={0} variants={fadeUp}>
            Transformer vs. Mamba
          </motion.span>
          <motion.h2
            custom={1}
            variants={fadeUp}
            style={{ marginTop: "var(--space-md)" }}
          >
            A different paradigm.
          </motion.h2>

          <motion.div className="comparison-grid" custom={2} variants={fadeUp}>
            <div className="comparison-col">
              <div className="comparison-col-title">
                Traditional Transformer
              </div>
              {[
                { label: "Attention", value: "O(n²) quadratic" },
                { label: "Memory", value: "Grows with sequence" },
                { label: "Context", value: "Fixed window (2k–8k)" },
                { label: "Generation", value: "KV-cache dependent" },
                { label: "Hardware", value: "Requires server GPU" },
              ].map((item) => (
                <div key={item.label} className="comparison-item">
                  <div className="comparison-label">{item.label}</div>
                  <div className="comparison-value">{item.value}</div>
                </div>
              ))}
            </div>

            <div className="comparison-col">
              <div className="comparison-col-title">Pebble (Mamba-2)</div>
              {[
                { label: "Recurrence", value: "O(n) linear" },
                { label: "Memory", value: "Constant (fixed state)" },
                { label: "Context", value: "Theoretically infinite" },
                { label: "Generation", value: "Native recurrent mode" },
                { label: "Hardware", value: "Runs in your browser" },
              ].map((item) => (
                <div key={item.label} className="comparison-item">
                  <div className="comparison-label">{item.label}</div>
                  <div className="comparison-value">{item.value}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      </section>

      <div className="container">
        <hr className="section-divider" />
      </div>

      {/* ── Footer ──────────────────────────────────────────────────── */}
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
            <a href="https://github.com" target="_blank" rel="noopener" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
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
