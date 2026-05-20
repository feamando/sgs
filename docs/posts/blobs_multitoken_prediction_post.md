# LinkedIn Post: Blobs as Multi-Token Prediction

---

Meta's multi-token prediction (Gloeckle et al., 2024): predict the next N tokens simultaneously. Forces the model to encode "what happens over the next N steps" in one hidden state.

Our SGS blobs do the same thing, but for retrieval instead of generation.

A blob is a single Gaussian primitive that encodes what an entire paragraph means. One query retrieves it. One alpha-compositing pass integrates it. The model doesn't re-read the paragraph token by token; it gets the chunk-level meaning in one shot.

The parallel:
- Multi-token prediction amortizes autoregressive steps at the output.
- Blob retrieval amortizes context processing at the input.

Both force more compositional representations. Both operate at a coarser grain than the atomic unit (token).

The deeper connection: we recently proved that alpha-compositing (how blobs are rendered) is a strict superset of softmax attention (how transformers aggregate). Same mathematical operation, different scale. Multi-token prediction and multi-chunk retrieval are two faces of the same coin.

Currently evaluating on Planck 1.4, where each conversation turn becomes a retrievable blob. Flat cost per turn regardless of conversation length.

---

Nikita Gorshkov
Radiance Labs
