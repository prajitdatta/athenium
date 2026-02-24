"""
trace_attention.py
──────────────────
Athenium — Attention Layer Trace

Traces a real financial contract clause through one complete attention
layer: Q/K/V projection → raw scores → scaling → softmax → output.

Prints every intermediate tensor shape and the full attention weight matrix.

Run with:   python scripts/trace_attention.py
Requires:   torch  (no GPU, no model download, completes in seconds)
"""

import math
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(42)


def rule(char="─", width=70): print(char * width)
def header(title):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def main():
    print("\n" + "═" * 70)
    print("  ATHENIUM — Attention Layer Trace")
    print("  Clause: 'The borrower shall not declare any event of default'")
    print("═" * 70)

    # Tokens from the clause (simplified — 11 tokens including special)
    tokens = ["<s>", "The", "borrow", "er", "shall", "not",
              "declare", "any", "event", "of", "default"]
    n       = len(tokens)
    d_model = 64    # simplified (production: d_model=4096, d_k=128 per head)
    d_k     = 64

    header(f"Input — {n} tokens, d_model={d_model}")
    print(f"  Tokens: {tokens}")
    X = torch.randn(1, n, d_model)
    print(f"  X.shape = {list(X.shape)}   (batch=1, seq={n}, d_model={d_model})")

    # ── Step 1: Project to Q, K, V ────────────────────────────────────────────
    header("Step 1 — Three learned projections: Q, K, V")
    print("""
  Q = X · W_Q    "What am I looking for?"
  K = X · W_K    "What do I advertise?"
  V = X · W_V    "What do I give you if you select me?"

  Three roles, three independent weight matrices, same input X.
  The weights are learned during training — not hard-coded.
""")
    W_Q = torch.nn.Linear(d_model, d_k, bias=False)
    W_K = torch.nn.Linear(d_model, d_k, bias=False)
    W_V = torch.nn.Linear(d_model, d_k, bias=False)

    Q = W_Q(X)
    K = W_K(X)
    V = W_V(X)

    print(f"  Q.shape = {list(Q.shape)}   ← what each token is looking for")
    print(f"  K.shape = {list(K.shape)}   ← what each token advertises")
    print(f"  V.shape = {list(V.shape)}   ← what each token contributes")

    # ── Step 2: Raw alignment scores ─────────────────────────────────────────
    header("Step 2 — Compute alignment scores: Q · Kᵀ")
    print("""
  scores[i][j] = dot product of query i with key j
               = "how much should token i attend to token j?"

  High score → strong alignment → high attention weight (after softmax)
""")
    scores_raw = Q @ K.transpose(-2, -1)
    print(f"  scores_raw.shape = {list(scores_raw.shape)}   (seq × seq)")
    print(f"\n  Top-left 5×5 corner of raw score matrix:")
    print(f"  {scores_raw[0, :5, :5].detach().numpy().round(2)}")

    # ── Step 3: Scale ─────────────────────────────────────────────────────────
    header(f"Step 3 — Scale by √d_k = √{d_k} = {math.sqrt(d_k):.1f}")
    print(f"""
  Problem: dot products grow with d_k (variance = {d_k} without scaling).
  Large scores → softmax saturates → one weight ≈ 1.0, rest ≈ 0.0
  → gradients vanish → model stops learning.

  Fix: divide by √d_k to restore variance to ~1.0.
""")
    scores_scaled = scores_raw / math.sqrt(d_k)
    print(f"  Mean |score| before scaling: {scores_raw[0].abs().mean().item():.4f}")
    print(f"  Mean |score| after  scaling: {scores_scaled[0].abs().mean().item():.4f}")

    # ── Step 4: Softmax ───────────────────────────────────────────────────────
    header("Step 4 — Softmax → attention weight matrix")
    print("""
  softmax(scores, dim=-1) converts each row to a probability distribution.
  Each row sums to 1.0 — this is token i's "attention budget".
""")
    weights = F.softmax(scores_scaled, dim=-1)
    print(f"  weights.shape = {list(weights.shape)}")
    print(f"\n  Attention weight matrix (rows = query tokens, cols = key tokens):\n")

    w = weights[0].detach().numpy()
    col_hdr = "          " + "".join(f"{t[:6]:>8}" for t in tokens)
    print(col_hdr)
    for i, tok in enumerate(tokens):
        row = "  ".join(f"{w[i, j]:6.3f}" for j in range(n))
        print(f"  {tok[:8]:>8}  {row}")

    print(f"\n  Row sums (each should be 1.000):")
    print(f"  {weights[0].sum(dim=-1).detach().numpy().round(4)}")

    # ── Step 5: Weighted sum of values ────────────────────────────────────────
    header("Step 5 — Weighted aggregation: weights @ V → contextualised output")
    print("""
  output = weights @ V

  Each token's new representation is a blend of all value vectors,
  weighted by how much attention it paid to each position.

  "default" (position 10) after attention:
    It has drawn heavily from "event" (pos 8) and "of" (pos 9).
    Its output embedding now encodes the legal defined term
    "event of default" as a compound unit — not the word in isolation.
    This is context-sensitivity made computable.
""")
    output = weights @ V
    print(f"  output.shape = {list(output.shape)}")
    print(f"\n  Output vector for 'default' (position 10), first 8 dims:")
    print(f"  {output[0, 10, :8].detach().numpy().round(4)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    header("Summary — Tensor flow through one attention layer")
    print(f"""
  X           {list(X.shape)}       raw token embeddings
  Q, K, V     {list(Q.shape)}       projected to query / key / value roles
  scores_raw  {list(scores_raw.shape)}  Q · Kᵀ  (alignment scores)
  scores_sc   {list(scores_scaled.shape)}  scaled by √d_k = {math.sqrt(d_k):.1f}
  weights     {list(weights.shape)}  softmax → probability distributions
  output      {list(output.shape)}       context-aware token representations

  Parameters in this single-head layer (d_model={d_model}, d_k={d_k}):
    W_Q: {d_model:,} × {d_k:,} = {d_model * d_k:,}
    W_K: {d_model:,} × {d_k:,} = {d_model * d_k:,}
    W_V: {d_model:,} × {d_k:,} = {d_model * d_k:,}
    ─────────────────────────────────────────
    Total: {3 * d_model * d_k:,} parameters (this single head, simplified)

  In Athenium (d_model=4096, 32 heads, d_k=128):
    Q/K/V per layer:  3 × 4096 × 4096 = {3 * 4096 * 4096:,} parameters
    × 32 layers:      {32 * 3 * 4096 * 4096:,} attention parameters
""")
    print("  ✓ Trace complete.")
    print("  → See src/attention/scaled_dot_product.py for the full annotated implementation.\n")


if __name__ == "__main__":
    main()
