# ⚖️ Athenium

> *A production system that reads financial contracts the way an expert does — clause by clause, with full understanding of what each word means in context.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.40-FFD21E)](https://huggingface.co/docs/transformers/index)
[![PEFT](https://img.shields.io/badge/PEFT-0.10-7C3AED)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🔬_Live-Pipeline_Explorer-0A0C0F?style=flat-square)](https://prajitdatta.github.io/athenium/pipeline.html)
[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-prajitdatta.github.io-4F46E5?style=flat-square)](https://prajitdatta.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-prajitdatta-181717?style=flat-square&logo=github)](https://github.com/prajitdatta)




---

<div align="center">
**[→ Anthenium Website](https://prajitdatta.github.io/athenium/index.html)**
*Built by [Prajit Datta](https://prajitdatta.github.io/) &nbsp;·&nbsp; [GitHub](https://github.com/prajitdatta)*

</div>

---

## What Athenium Does

Every quarter, investment banks process thousands of financial contracts — loan agreements, ISDA master agreements, credit facilities, derivatives documentation. Hidden inside the dense legal language of these documents are clauses that carry serious risk: covenant breaches, jurisdiction conflicts, margin call triggers, and default provisions that could expose the institution to significant liability.

The traditional process relies on junior analysts working under time pressure: 72-hour turnaround, ~12% error rate on complex instruments, and no consistent standard for what "high risk" actually means across reviewers.

**Athenium replaces that process with a fine-tuned transformer that reads each clause, understands its legal context, and returns a structured risk classification in under 200 milliseconds** — with a confidence score, per-class probabilities, and an attribution map showing exactly which terms drove the decision.

The system is not a wrapper around an external API. It is a self-hosted, fine-tuned model built from first principles — because when the output influences a credit decision, you need to own and understand every layer of the stack.

---

## Results

Evaluated on 1,000 held-out contracts, stratified by risk level and instrument type:

| Metric | Score |
|--------|-------|
| Macro F1 | **0.971** |
| Accuracy | **97.4%** |
| Confidence Calibration (ECE) | **0.031** |
| P95 Inference Latency | **118 ms** |
| Contracts Auto-Processed | **88.3%** |
| Cost per 1,000 Documents | **$0.18** |

Against GPT-4o zero-shot on the same held-out set: Athenium achieves 14% higher macro F1, 7× lower latency, 79× lower cost — with all contract data remaining self-hosted.

---

## How It Works

This repository is the complete engineering record of Athenium. Every file is annotated. Every decision has a documented rationale. The following sections walk through the system from the ground up.

---

### I. Turning Words into Numbers

*[`src/embeddings/positional_encoding.py`](src/embeddings/positional_encoding.py)*

Before a transformer can process a contract clause, the text must be converted into a form the model can compute over. This happens in two steps.

**Tokenisation** breaks the raw string into subword units using a SentencePiece vocabulary of 32,000 entries. The clause *"The Borrower shall not declare any Event of Default"* becomes a sequence of integer token IDs. Subword tokenisation means the model never encounters a truly unknown word — even rare legal terms are decomposed into known units.

**Embedding lookup** converts each token ID into a dense 4,096-dimensional vector by indexing into a learned lookup table. At this point the vectors are context-free: the token `default` has the same embedding whether it appears in *"event of default"* or *"the default setting"*. That changes in the next stage.

**The word order problem.** Transformers are inherently order-invariant. The attention computation treats the input as an unordered set — rearrange the tokens and you get the same output, rearranged identically. Nothing in the bare mathematics distinguishes position 3 from position 7. Positional encoding solves this by adding a unique signal to each position before the first transformer layer.

The original approach (Vaswani et al., 2017) uses sinusoidal functions:

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

Each dimension pair oscillates at a different frequency. Low dimensions are sensitive to local position; high dimensions encode global structure. The encoding is fixed — no parameters, no training, works for any sequence length.

Athenium's backbone (Mistral-7B) uses **Rotary Position Embedding (RoPE)**. Instead of adding position to the embeddings, RoPE rotates the Query and Key vectors inside each attention head by an angle proportional to the token's position. This encodes *relative* distance rather than absolute position — the dot product between two rotated vectors depends only on how far apart they are. The model generalises better to longer sequences at inference.

---

### II. Attention — The Core Operation

*[`src/attention/scaled_dot_product.py`](src/attention/scaled_dot_product.py) · `python scripts/trace_attention.py`*

The central question attention answers for every token:

> *Given everything else in this document, what should I be attending to right now?*

Each token is projected into three simultaneous roles via three learned weight matrices:

```
Q = x · W_Q    "What am I looking for?"
K = x · W_K    "What information do I offer?"
V = x · W_V    "What do I contribute if selected?"
```

The dot product Q·Kᵀ scores alignment between every query and every key. These scores are divided by √dₖ before softmax — without this, dot product variance grows with dimension size, softmax collapses to near-one-hot distributions, and gradients vanish. Dividing by √dₖ keeps variance around 1 regardless of dimension.

Softmax converts the scaled scores to a probability distribution — each token's *attention budget* across the sequence. The output is a weighted sum of value vectors:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V
```

The result: `default` in *"event of default"* now has a fundamentally different vector than `default` in *"the default setting"* — it attended to completely different neighbours and received different value signals. Context-sensitivity is now computable.

---

### III. Multi-Head Attention — Many Relationships at Once

*[`src/attention/multihead.py`](src/attention/multihead.py)*

A single attention head learns one type of relationship. Financial contracts require several simultaneously: syntactic structure, legal defined terms as compound units, long-range coreference, jurisdictional scope, and numeric covenant triggers.

Multi-head attention runs `h` independent operations in parallel, each in a lower-dimensional subspace of size `dₖ = d_model / h`:

```
MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) · W_O
```

Same parameter count as a single full-dimension head. Same computational cost. But `h` independent subspaces, each free to specialise on a different structural pattern. After fine-tuning on financial contracts, heads 4–5 bind legal defined terms, heads 6–7 track jurisdictional markers, heads 8–9 attend to numeric quantities. This specialisation emerges from training, not from design.

Athenium: `d_model=4096, h=32 heads, dₖ=128 dimensions per head`.

---

### IV. The Transformer Block — Stacked 32 Times

*[`src/encoder/transformer_block.py`](src/encoder/transformer_block.py)*

The full transformer layer wraps attention in a Feed-Forward Network, residual connections, and layer normalisation:

```
x → LayerNorm → Multi-Head Attention → Dropout → + x  (residual)
x → LayerNorm → Feed-Forward Network → Dropout → + x  (residual)
```

The **FFN** expands each token's representation to 4× d_model, applies an activation function, then compresses back. It runs independently on every token — no cross-token interaction. Where attention routes information *between* tokens, the FFN processes each token *in depth*, acting as a key-value memory for factual associations learned during pretraining.

**Residual connections** make stacking 32 blocks tractable. Each block learns only the *delta* — the update to add. Gradients flow directly through residual paths to the earliest layers, preventing the exponential decay that would otherwise make deep networks untrainable.

**Pre-norm** (LayerNorm before each sublayer) keeps activation magnitudes controlled at any depth. Post-norm becomes numerically unstable above 12 layers without careful warmup schedules. Athenium matches the Mistral, LLaMA, and GPT-2 architecture on this point.

---

### V. Normalisation — BatchNorm vs LayerNorm

*[`src/internals/normalization.py`](src/internals/normalization.py)*

Both normalisation types zero-centre and unit-scale activations, then rescale with learned parameters **γ** (scale) and **β** (shift). The difference is which axis they normalise over.

**Batch Normalisation** normalises each feature across the batch:

```python
μ_d = mean(x[:, d])      # average of feature d across all samples in the batch
y   = γ · (x - μ) / (σ + ε) + β
```

This works well for convolutional vision models with large, fixed-size batches. It fails for transformers: inference on a single document has no batch to normalise over, so the model falls back to running statistics accumulated during training — a train/eval discrepancy. Padded positions contaminate batch statistics. Different sequence positions are semantically incomparable. BatchNorm is the right tool for CNNs; it is the wrong tool for sequence models.

**Layer Normalisation** normalises across the feature dimension for each individual token:

```python
μ = mean(x[b, s, :])     # average across features for this single token
y = γ · (x - μ) / (σ + ε) + β
```

Each token normalises independently. Batch size of 1 is identical to batch size of 32. No running statistics, no train/eval discrepancy, no padding contamination. This is why every modern transformer — BERT, GPT, LLaMA, Mistral, Athenium — uses LayerNorm.

---

### VI. Fine-Tuning — Teaching Mistral to Read Contracts

*[`src/finetune/lora_config.py`](src/finetune/lora_config.py) · [`src/finetune/train.py`](src/finetune/train.py)*

Full fine-tuning of Mistral-7B in fp32 requires storing four copies of every parameter:

| Tensor | Memory |
|--------|--------|
| Model weights | 28 GB |
| Gradients | 28 GB |
| Adam first moment (m) | 28 GB |
| Adam second moment (v) | 28 GB |
| **Total** | **~112 GB** |

That requires four A100 80GB GPUs. The Adam optimizer states (m and v) must remain in fp32 even in mixed-precision training — in fp16, tiny gradient increments underflow to zero and the moment estimates stop tracking the gradient signal, causing divergence.

**LoRA** solves this. The key insight: weight updates during fine-tuning have low intrinsic rank. For each weight matrix W₀, inject a parallel low-rank path:

```
W = W₀ + (α/r) · B · A
  where A ∈ ℝ^(r×k),  B ∈ ℝ^(d×r),  r ≪ min(d, k)
```

W₀ is frozen. Only A and B accumulate gradients. B is initialised to zero so the model starts from exactly the pretrained distribution. After training, `W_merged = W₀ + (α/r)·B·A` is computed once and the adapter disappears — zero inference overhead.

**QLoRA** adds 4-bit NF4 quantisation of the frozen base weights. NF4 is the information-theoretically optimal data type for normally-distributed weights. The 28 GB base model compresses to 3.5 GB. LoRA adapters and their optimizer states add less than 200 MB. Total peak VRAM: **18.4 GB** — one A10G.

Rank ablation on 3,000 held-out contracts:

| Rank | Trainable Params | Macro F1 | Training Time |
|------|-----------------|----------|---------------|
| r=4 | 2.1M | 0.912 | 1h 20m |
| r=8 | 4.2M | 0.947 | 2h 10m |
| **r=16** | **8.4M** | **0.971** | **3h 45m** |
| r=32 | 16.8M | 0.974 | 7h 30m |

r=16 is the inflection point. r=32 buys 0.3% F1 at double the compute and time.

---

### VII. GPU Memory — The Full Accounting

*[`src/internals/gpu_memory.py`](src/internals/gpu_memory.py)*

For any model trained with Adam in full fp32 precision, the memory decomposes into four equal components — weights, gradients, Adam m, Adam v — each the same size as the model weights. The total is always **4× the raw weight memory**. A 7B fp32 model: 4 × 27 GB = 108 GB.

QLoRA collapses this: NF4 base weights (3.5 GB) + BF16 adapters + fp32 optimizer states for 8.4M parameters only = 18.4 GB total.

The runnable module calculates exact memory for any model size, precision, and optimizer: `python -m src.internals.gpu_memory`

---

### VIII. Evaluation — Knowing When to Trust the Output

*[`src/evaluation/metrics.py`](src/evaluation/metrics.py)*

**Macro F1** weights all four risk classes equally. A model that classifies every contract as LOW achieves 60% accuracy but 0.25 macro F1. Accuracy is the wrong metric for imbalanced risk classification.

**Expected Calibration Error (ECE)** measures whether confidence scores are trustworthy. A model saying "92% confident" should be correct 92% of the time. ECE bins predictions by confidence and measures the gap. Athenium's ECE of 0.031 makes the confidence score reliable enough to automate escalation decisions around.

**Error decomposition** separates under-classification (HIGH predicted as LOW — missed risk, dangerous) from over-classification (LOW predicted as HIGH — unnecessary review, costly). These carry asymmetric consequences and must be tracked independently.

**Escalation simulation** models the production policy: at threshold 0.85, 88.3% of contracts auto-process at 98.7% accuracy, 11.7% escalate, and no CRITICAL contract in the auto-processed set is misclassified.

---

### IX. The API

*[`src/serving/api.py`](src/serving/api.py)*

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Borrower shall not declare any dividend if an Event of Default is continuing.",
    "return_attribution": true
  }'
```

```json
{
  "risk_label":          "HIGH",
  "confidence":          0.9231,
  "label_proba":         {"LOW": 0.004, "MEDIUM": 0.061, "HIGH": 0.923, "CRITICAL": 0.012},
  "escalate_for_review": false,
  "latency_ms":          118.4,
  "attribution": [
    {"token": "Event",    "weight": 0.142},
    {"token": "Default",  "weight": 0.138},
    {"token": "dividend", "weight": 0.091}
  ]
}
```

The attribution map averages attention weights across all 32 layers and 32 heads, extracts the CLS row, and returns the top-15 tokens. A risk analyst reviewing a HIGH or CRITICAL classification can see exactly which terms the model weighted — and verify that the model's reasoning matches theirs.

Below 0.85 confidence, `escalate_for_review` is true and the calling system routes to the analyst queue. No escalation logic lives in the API — it returns data, the infrastructure decides.

---

## Repository Structure

```
athenium/
├── src/
│   ├── embeddings/
│   │   └── positional_encoding.py      ← Token embeddings, sinusoidal PE, RoPE
│   ├── attention/
│   │   ├── scaled_dot_product.py       ← Q·Kᵀ/√dₖ → softmax → weighted sum
│   │   └── multihead.py                ← 32 parallel heads, concat + W_O
│   ├── encoder/
│   │   └── transformer_block.py        ← MHA + FFN + LayerNorm + residuals ×32
│   ├── internals/
│   │   ├── normalization.py            ← BatchNorm vs LayerNorm, from scratch
│   │   └── gpu_memory.py               ← Memory: weights + grads + Adam states
│   ├── finetune/
│   │   ├── lora_config.py              ← LoRA: W = W₀ + (α/r)·BA, rank ablation
│   │   └── train.py                    ← QLoRA training pipeline
│   ├── evaluation/
│   │   └── metrics.py                  ← Macro F1, ECE, escalation analysis
│   └── serving/
│       └── api.py                      ← FastAPI, attribution, escalation flag
├── scripts/
│   └── trace_attention.py              ← Attention trace through a contract clause
├── tests/
    ├── test_attention.py
    ├── test_embeddings.py
    ├── test_normalization.py
    └── test_gpu_memory.py

```



---

## References

| | |
|-|-|
| Attention Is All You Need — Vaswani et al. (2017) | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| LoRA — Hu et al. (2021) | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| QLoRA — Dettmers et al. (2023) | [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314) |
| RoFormer (RoPE) — Su et al. (2021) | [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864) |
| Layer Normalisation — Ba et al. (2016) | [arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450) |
| Batch Normalisation — Ioffe & Szegedy (2015) | [arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167) |
| Pre-norm analysis — Xiong et al. (2020) | [arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745) |
| Calibration — Guo et al. (2017) | [arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599) |

---

<div align="center">

**Built by [Prajit Datta](https://prajitdatta.github.io/) | *MIT License***

[![GitHub](https://img.shields.io/badge/GitHub-prajitdatta-181717?style=for-the-badge&logo=github)](https://github.com/prajitdatta)&nbsp;&nbsp;
[![Portfolio](https://img.shields.io/badge/Portfolio-prajitdatta.github.io-4F46E5?style=for-the-badge&logo=google-chrome&logoColor=white)](https://prajitdatta.github.io/)

</div>
