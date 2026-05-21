# Blobs and Multi-Token Prediction: Two Sides of the Same Coin

Meta's multi-token prediction (Gloeckle et al., 2024) trains models to predict N tokens simultaneously. This forces the shared trunk to encode "what happens over the next N steps" in a single hidden state. The result: faster inference (fewer autoregressive steps) and better internal representations (the model must plan ahead).

Our blob retrieval (independently developed, 2025-2026) does the same thing at the input side. Instead of re-reading an entire passage token by token, the model retrieves one primitive that encodes "what the passage means" in a single vector. One cosine query, one alpha-compositing pass, done.

The parallel is structural:
- Both operate at coarser grain than the atomic unit (the token).
- Both amortize computation. Multi-token prediction amortizes output steps. Blob retrieval amortizes input processing.
- Both force more compositional internal representations. The model must compress richer information into fewer primitives.

The difference is where the savings occur. Multi-token prediction reduces the number of generation steps (output). Blob retrieval reduces the amount of context that must be re-processed (input). They are complementary, not competing.

The deeper connection: we proved that alpha-compositing (the rendering equation for blobs) is a strict superset of softmax attention (how transformers aggregate tokens). Same mathematical operation, different granularity. Multi-token prediction and multi-chunk retrieval are two expressions of the same principle: amortize computation by operating at a coarser semantic grain.

We identified this parallel after the fact. The convergence was independent.

```
Multi-Token Prediction            Blob Retrieval
========================          ========================

Input tokens                      Input tokens + query
    |                                 |
    v                                 v
[Shared model trunk]              [Blob store: cosine query]
    |                                 |
    v                                 v
Predict tokens 1, 2, ..., N       Retrieve top-k blobs
simultaneously from one state         |
    |                                 v
    |                             [Alpha-compositing pass]
    |                             Render blob meaning
    |                                 |
    |                                 v
    |                             [Word-level rendering]
    |                             Fill remaining capacity
    |                                 |
    v                                 v
Output: N tokens at once          Output: next token prediction

Amortizes: output steps           Amortizes: input re-reading
Grain: N tokens                   Grain: entire passages
Mechanism: multi-head output      Mechanism: alpha-compositing
```

Nikita Gorshkov / Radiance Labs
