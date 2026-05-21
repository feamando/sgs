# What is a Blob? Chunked Meaning as a First-Class Primitive

Language models process one token at a time. Context grows linearly. A 100-turn conversation means re-reading 100 turns of tokens at every step. This is expensive, and it scales poorly.

We asked: what if you could compress an entire passage into a single primitive, retrieve it by similarity, and render it in one operation?

That primitive is a blob. A Gaussian distribution in semantic space (mu in d_s=128 splatting space, features in d_f=1000 semantic space) that encodes what a full paragraph or concept means. Offline, we mean-pool token embeddings from a passage into a single blob vector and store it. Online, the model queries the blob store, retrieves the top-k matches by cosine similarity, and renders them via alpha-compositing in a single pass. Word-level rendering fills in the remaining specifics.

The key insight: blobs are not RAG chunks with string matching. They live in the same mathematical space as word tokens and are rendered by the same transmittance-weighted equation. The model does not "read" the blob. It "sees" it the way it sees any word, just at a coarser grain. There is no separate retrieval pathway bolted on after the fact. Retrieval and generation share one unified rendering pass.

We proved in Lean 4 that alpha-compositing (how blobs render) is a strict superset of softmax attention (how transformers aggregate). The math subsumes standard attention as a special case.

```
[Passage: "The capital of France is Paris..."]
        |
        v
    [Encoder: mean-pool token embeddings]
        |
        v
    Blob: (mu, features)        ---> stored in BlobStore
                                          |
                                          |
Query: "What is the capital of France?"   |
        |                                 |
        v                                 |
    [Encoder] --> query vector            |
        |                                 |
        v                                 v
    [Cosine similarity match across all blobs]
        |
        v
    Retrieved: top-k blobs
        |
        v
    [Alpha-compositing: render blobs + word tokens]
        |
        v
    Output token prediction
```

Nikita Gorshkov / Radiance Labs
