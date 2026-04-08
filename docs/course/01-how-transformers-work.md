# Article 1: How Transformers Actually Work

*The machine behind every modern language model — and what it can't do*

---

## The Problem Transformers Solve

Language is sequential. "The cat sat on the mat" has meaning because of the *order* of words and the *relationships* between them. "Cat" relates to "sat" (the cat is doing the sitting). "Mat" relates to "on" (the location). A language model needs to capture these relationships to understand — and generate — text.

Before transformers, models processed language word-by-word in sequence (RNNs, LSTMs). This was slow and forgetful — by the time the model reached the end of a long sentence, it had largely forgotten the beginning.

Transformers (Vaswani et al., 2017) solved this by processing all words simultaneously and letting each word "look at" every other word to figure out what's relevant. This mechanism is called **attention**.

---

## Step 1: Words Become Vectors

A transformer can't read the word "cat." It reads a vector — a list of numbers. Each word in the vocabulary is mapped to a vector of, say, 768 numbers. This mapping is called an **embedding**.

```
"cat"  → [0.12, -0.34, 0.87, ..., 0.05]   (768 numbers)
"sat"  → [0.45, 0.22, -0.11, ..., 0.93]
"mat"  → [0.09, -0.56, 0.44, ..., 0.17]
```

These vectors aren't random. During training, words that appear in similar contexts end up with similar vectors. "Cat" and "dog" will have vectors pointing in roughly the same direction. "Cat" and "democracy" will point in very different directions.

The embedding is a **point** in a 768-dimensional space. Each word gets exactly one point. This is important — we'll come back to why this is a limitation.

---

## Step 2: Positional Encoding

Since the transformer processes all words at once (not sequentially), it doesn't inherently know that "cat" comes before "sat." To fix this, a **positional encoding** is added to each word's vector — a pattern of numbers that encodes "this is the 1st word," "this is the 2nd word," etc.

After adding positional encoding, the model can distinguish "the cat sat on the mat" from "the mat sat on the cat."

---

## Step 3: Attention — The Core Mechanism

This is where transformers earn their name. For each word, the model computes three vectors:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

These are computed by multiplying the word's embedding by three learned matrices:

```
Q = embedding × W_Q
K = embedding × W_K
V = embedding × W_V
```

Then for each word, attention computes:

```
score(word_i, word_j) = Q_i · K_j / √d
```

This dot product measures "how relevant is word j to word i?" High score = very relevant. The scores are passed through **softmax** (which turns them into probabilities that sum to 1), and the result is used to take a weighted average of all the Value vectors:

```
output_i = Σ_j softmax(score_ij) × V_j
```

**In plain English:** each word looks at every other word, decides which ones are relevant, and constructs a new representation by blending information from the relevant words.

For the sentence "The cat sat on the mat":
- When processing "sat," attention might assign high weight to "cat" (the subject doing the sitting) and "mat" (via "on") — and low weight to "the" (not very informative).
- When processing "mat," attention might weight "on" heavily (the spatial relationship).

---

## Step 4: Multi-Head Attention

One set of Q/K/V might capture one type of relationship (e.g., "who is doing the action?"). But language has many simultaneous relationships: syntactic roles, semantic associations, co-references, sentiment. 

So transformers use **multiple attention heads** — typically 12 — each with its own W_Q, W_K, W_V. Each head independently computes attention, capturing a different aspect. The outputs are concatenated and mixed:

```
MultiHead = Concat(head_1, head_2, ..., head_12) × W_out
```

Research has shown that different heads do specialize: some track syntactic structure, others handle co-reference, others capture semantic similarity.

---

## Step 5: Feedforward Network + Stacking

After attention, each word's representation passes through a **feedforward network** (FFN) — two linear transformations with a nonlinearity between them. This adds computational depth beyond just looking at other words.

Then the whole process repeats. A transformer is **stacked layers** — 12 layers for BERT-base, 96 for GPT-3. Each layer applies attention + FFN + residual connection. Early layers capture local syntax. Later layers build abstract semantics.

```
Layer 1: raw embeddings → basic syntax
Layer 6: syntactic structure → semantic roles
Layer 12: abstract meaning, sentiment, discourse structure
```

The depth is crucial. Complex understanding — disambiguation, multi-hop reasoning, inference — emerges from this depth.

---

## Step 6: Generating Text

For language models like GPT, generation is **autoregressive** — one token at a time:

1. Feed in the prompt: "The cat sat on the"
2. Run through all layers → get a vector for the last position
3. Project that vector to vocabulary-sized probabilities → "mat" has the highest probability
4. Append "mat" to the input, repeat

Each token depends on all previous tokens. This left-to-right dependency is why autoregressive models dominate: they naturally model the conditional probability P(next word | all previous words).

---

## What Transformers Get Right

- **Parallelism**: All words processed simultaneously during training (unlike sequential RNNs)
- **Long-range dependencies**: Any word can attend to any other, regardless of distance
- **Scalability**: Performance improves predictably with more data and compute (scaling laws)
- **Versatility**: The same architecture works for translation, summarization, code, reasoning

---

## What Transformers Get Wrong

Despite their success, transformers have fundamental limitations:

**1. Quadratic cost.** Every word attends to every other word. For n words, this is n^2 operations. Double the text length → 4x the computation. This limits context windows and makes long documents expensive.

**2. Words are points.** Each word gets a single vector (a point in 768D space). But words aren't points — "bank" means at least three different things (financial, river, verb). The model has to smash all meanings into one vector and disambiguate through context. There's no native way to represent "this word has broad meaning" or "I'm uncertain about this word."

**3. No explicit geometry.** The model learns implicit geometric relationships (similar words cluster together), but this geometry is not architecturally enforced — it's an emergent property that can be fragile and hard to interpret.

**4. Compositional generalization failure.** Transformers struggle to systematically combine known primitives in new ways. If you train a model on "jump," "jump twice," and "turn left," it may fail to correctly handle "turn left twice" — even though the composition rule is obvious to humans.

**5. Black box.** When a transformer outputs "The capital of France is Paris," you can't easily inspect *why*. Attention weights provide some clues but are notoriously unreliable as explanations.

---

## The Key Insight for What Comes Next

The transformer's attention mechanism is, at its core, a **weighted sum of values**:

```
output = Σ_j  weight_j × value_j
```

Where the weights come from softmax over dot-product scores. This is a very specific mathematical choice. There are other ways to compute weights. There are other ways to compose values. The question this course explores is: **what if we used a completely different mathematical framework for computing those weights — one borrowed from computer graphics?**

That framework is the rendering equation from 3D Gaussian Splatting. Before we get there, we need to understand how words become geometry.

---

*Next: [Article 2 — Words as Points, Words as Regions](02-words-as-geometry.md)*
