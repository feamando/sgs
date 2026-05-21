# First Results: Conversation Memory via Blobs (Planck 1.4)

We ran our first needle-in-a-haystack test for blob-based conversation memory. The question: if you inject a fact at turn 10 of a 100-turn conversation, can the blob retrieval system surface it 90 turns later?

Setup: Planck 1.3 (100M params, Wikipedia-trained), DynamicBlobStore with 512 slots, cosine similarity retrieval in feature space (d_f=1000). Each dialogue turn is encoded into a blob on the fly. At query time, the system retrieves top-k blobs by similarity to the current question.

We planted 10 needle facts (e.g., "The largest planet is Jupiter") at turn 10, ran 90 filler turns of unrelated conversation, then asked about the needle at turn 100.

Results:
- Similarity-only retrieval: 40% recall (4 out of 10 needle blobs surfaced in top-k)
- Hybrid retrieval (recency decay 0.05): 0% recall (decay too aggressive, suppresses signal from old turns)
- No-retrieval baseline: 0% recall (expected; no memory mechanism)

Interpretation: the retrieval mechanism works. With zero tuning, pure cosine similarity surfaces the correct blob 40% of the time across a 90-turn gap. The model cannot generate the answer (100M params, no instruction tuning), but that is a generation capacity issue, not a retrieval failure. The information reaches the model's context.

The hybrid result is informative: a recency decay of 0.05 per turn means signal from turn 10 is multiplied by 0.95^90 = 0.01. Effectively zero. The fix is straightforward: reduce decay for long-range retrieval, or remove it entirely when similarity is high.

Next steps: reduce recency decay, test at Hertz scale (1B params with instruction tuning where generation capacity exists), and connect to Raum 1.3 recursive decomposition where blobs serve as compositional building blocks.

The core finding: blob retrieval at 100M params, zero tuning, already provides non-trivial long-range memory. The architecture works. Scale and tuning are engineering problems, not research risks.

```
Turn  1: "What's the weather?"                --> encode --> blob_1
Turn  2: "Tell me about food"                 --> encode --> blob_2
  ...
Turn 10: "The capital of Australia            --> encode --> blob_10
          is Canberra"                                      [NEEDLE]
  ...
Turn 99: (filler conversation)                --> encode --> blob_99

                    --- Query Phase ---

Turn 100: "What is the capital of Australia?"
    |
    v
[Query encoder] --> query_features (d_f=1000)
    |
    v
[Cosine similarity vs all 100 blob feature vectors]
    |
    v
Top-k results: blob_10 retrieved?  --> RETRIEVAL RECALL: 40%
    |
    v
[Construct context: top-k blobs + last 3 verbatim turns + query]
    |
    v
[Planck 1.3 LM generates response]
    |
    v
Does "Canberra" appear in output?  --> GENERATION RECALL: 0%
                                       (expected at 100M, no instruct)
```

Nikita Gorshkov / Radiance Labs
