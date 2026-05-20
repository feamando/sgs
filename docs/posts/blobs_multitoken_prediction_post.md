# LinkedIn Post: Blobs as Multi-Token Prediction

---

Something clicked recently while reading Meta's multi-token prediction paper (Gloeckle et al., 2024).

Their core insight: instead of predicting one token at a time, train the model to predict the next N tokens simultaneously. The auxiliary prediction heads force the shared trunk to build richer representations, representations that encode not just "what comes next" but "what comes next at multiple scales."

We have been building something structurally analogous in our Semantic Gaussian Splatting work, but in a completely different domain.

**SGS blobs are multi-token predictions over meaning.**

In a standard language model, each token contributes a tiny slice of meaning to the output. Context accumulates one position at a time. The model's hidden state is a compressed summary of everything so far.

In SGS, a "blob" is a pre-computed chunk of meaning, a Gaussian distribution in semantic space that summarizes an entire passage, paragraph, or concept. When the model encounters a query, it retrieves the top-k most relevant blobs and renders them via transmittance-weighted compositing (the same alpha-compositing equation used in volume rendering and, as we recently proved formally, a strict superset of softmax attention).

The parallel to multi-token prediction:

- Multi-token prediction: one forward pass produces N future tokens simultaneously, forcing the model to represent "what happens over the next N steps" in its hidden state.
- Blob retrieval: one query retrieves a blob that encodes "what an entire paragraph means" in a single Gaussian primitive, forcing the retrieval mechanism to match at a semantic chunk level rather than token-by-token.

Both are forms of "predict/retrieve at a coarser grain than the atomic unit." Both force the underlying representations to be more compositional. Both amortize computation: multi-token prediction amortizes autoregressive steps, blob retrieval amortizes context processing.

The difference: multi-token prediction is trained end-to-end with a shared trunk. Blob retrieval separates the "chunking" step (offline, at index time) from the "retrieval + integration" step (online, at inference time). This separation is what lets a 100M parameter model access knowledge that would otherwise require a much larger context window.

We are currently evaluating this on Planck 1.4 (conversation-memory blobs, where each dialogue turn becomes a retrievable blob). Early architecture is live; eval results incoming.

The deeper question: if multi-token prediction at the output forces better internal representations, does multi-chunk retrieval at the input do the same? We think yes, and the formal connection between alpha-compositing and attention (our JMLR submission) provides the theoretical grounding for why these two seemingly different ideas are actually the same mathematical operation applied at different scales.

---

Nikita Gorshkov
Radiance Labs
github.com/feamando/sgs
