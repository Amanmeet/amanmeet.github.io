---
title: Attention Is All You Need
date: 2025-12-15
tags:
  - transformers
  - NLP
updates:
  - date: 2025-12-20
    note: Added personal notes on multi-head attention elegance
---

# Attention Is All You Need

**Authors:** Vaswani et al. (2017)
**Venue:** NeurIPS 2017

---

## Key Idea

The paper introduces the **Transformer** architecture, which relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions. This enables significantly more parallelization during training.

## Architecture Overview

The Transformer uses an encoder-decoder structure:

- **Encoder**: 6 identical layers, each with multi-head self-attention and a position-wise feed-forward network.
- **Decoder**: 6 identical layers with an additional cross-attention sub-layer attending to encoder output.
- **Positional Encoding**: Since there is no recurrence, sinusoidal positional encodings are added to input embeddings.

### Multi-Head Attention

Instead of a single attention function, the model projects queries, keys, and values into `h` different subspaces and computes attention in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

## Strengths

- Massive parallelism compared to RNNs -- training time reduced significantly
- Attention weights provide some interpretability of what the model "looks at"
- Scales well with data and compute (as later demonstrated by BERT, GPT, etc.)

## Weaknesses / Open Questions

- Quadratic memory and compute cost in sequence length (`O(n^2)`)
- Positional encodings are somewhat ad-hoc -- later work (RoPE, ALiBi) improved this
- No inherent inductive bias for sequential structure

## Personal Notes

This is one of those papers where the impact far exceeded what the authors likely imagined. The title turned out to be almost literally true for the next decade of ML research. The key insight -- that attention alone, without recurrence, can model sequential data -- seems obvious in hindsight but was genuinely surprising at the time.

I find the multi-head attention mechanism particularly elegant: it's a simple idea (project into subspaces, attend, concatenate) that gives the model a rich capacity for capturing different types of relationships.
