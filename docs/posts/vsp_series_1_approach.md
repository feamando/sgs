# LinkedIn Series 3/3 — Post 1: the approach

Illustration: docs/posts/img/vsp_series_1_approach.svg
Tone: plain, first-person, understated. No em dashes. Plain text below for paste.

---

Language models inherit a quiet bug from their tokenizer: "crane" the bird and "crane" the machine get the same token. The word collapses two meanings into one, and the model has to pull them apart from context, every time, from scratch.

For a small language model I have been building (SGS), I tested a different token representation. Instead of one text vector per word, bundle three signals per SENSE into one vector:

- S, the semantic/text embedding (what a token already is)
- V, a visual grounding: generate an image of the sense, embed that
- P, a physical grounding: predict material properties (hard/soft, dense/light) for the sense

The bet: if the sense is already in the representation, the model does not spend capacity learning to disambiguate. Smaller model, multimodal understanding built in rather than bolted on.

First question, with a clear kill condition: can the bundle separate senses that text alone merges? Text embeddings score 0 on this, they cannot separate the senses at all, because both meanings of a written word get the exact same vector by construction. That is the whole problem, stated as a number. The V+S+P bundle scores 0.37 across 20 polysemous words, entirely auto-derived, nothing hand-labeled. The senses pull apart, carried mostly by the visual signal (a picture of a bird and a picture of a construction crane are just far apart).

So the representation works. That clears the gate to the real question, and the one that actually matters: does putting this into a language model make it better at disambiguation?

That is post 2. It did not go how I expected.
