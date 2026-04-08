# Article 2: Words as Points, Words as Regions

*Why meaning is geometric — and why a dot on a map isn't enough to describe a city*

---

## The Discovery That Changed NLP

In 2013, Tomas Mikolov and colleagues at Google published a paper that accidentally revealed something profound about language: **meaning has geometry.**

They trained a simple neural network (Word2Vec) to predict words from their neighbors. The network learned a 300-dimensional vector for each word. That was expected. What was unexpected was what those vectors could do:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

This isn't a trick. The vector difference between "king" and "man" encodes something like "royalness" — a *direction* in 300-dimensional space. Adding that direction to "woman" lands you near "queen." The relationship is geometric: a parallelogram in meaning-space.

Other regularities emerged:
- "Paris" - "France" + "Germany" ≈ "Berlin" (capital-of relationship)
- "walking" - "walk" + "swim" ≈ "swimming" (tense relationship)

This means **semantic relationships are encoded as directions and distances** in a high-dimensional space. Language has a geometry.

---

## Words as Points: What We Gain

Word2Vec, GloVe (2014), and later transformer embeddings all represent words as **points** — single locations in high-dimensional space.

What this gives us:

**Similarity = Proximity.** "Cat" and "dog" are nearby. "Cat" and "quantum" are far apart. You can measure similarity with cosine distance (the angle between vectors).

**Relationships = Directions.** The direction from "man" to "woman" is roughly parallel to the direction from "king" to "queen." Semantic relationships are consistent vector offsets.

**Clusters = Categories.** Animals cluster together. Countries cluster together. Professions cluster together. The space self-organizes into a semantic geography.

---

## Words as Points: What We Lose

But a point is a *precise location with zero extent*. Think of a point on a map — it marks a single spot. Now think of a word like "bank."

"Bank" means:
1. A financial institution
2. The side of a river
3. To tilt an aircraft
4. A collection of objects ("a bank of monitors")

In a point-based embedding, "bank" gets ONE vector — a single dot on the map, somewhere in the middle of all its meanings. This is like placing a pin for "San Francisco" and "banking district" and "waterfront" at the exact same point. It's a compromise that loses information.

**What we can't represent with points:**

| Property | Example | Why Points Fail |
|---|---|---|
| **Uncertainty** | "The flibberjabber is blue" — what is a flibberjabber? | A point has no "I don't know" — it's either somewhere or nowhere |
| **Breadth of meaning** | "Animal" covers dogs, cats, whales, insects | A point can't express that "animal" *encompasses a region* while "poodle" is a precise spot |
| **Polysemy** | "Spring" = season, coil, water source, to jump | A single point must compromise between unrelated meanings |
| **Asymmetric relations** | "Dog" IS-A "animal" but "animal" is NOT "dog" | Cosine similarity is symmetric — it can't distinguish containment from equivalence |

---

## Linguistics Predicted This

Before ML discovered word vectors, linguists had already described meaning in spatial, region-based terms.

**Eleanor Rosch's Prototype Theory (1973-1978):** Categories don't have sharp boundaries. A robin is a *very typical* bird. A penguin is a *less typical* bird. An ostrich is even less typical. Typicality decreases gradually from a central prototype — exactly like the density of a Gaussian distribution fading from its center.

Rosch measured this experimentally. When asked "Is X a bird?", people respond faster for robins (prototypical) than penguins (peripheral). The response time correlates with distance from the category prototype.

**Peter Gärdenfors' Conceptual Spaces (2000):** Gärdenfors formalized the geometry. Concepts are **regions**, not points. The concept "red" is a region in color space. The concept "dog" is a region in animal-feature space. He proved that "natural categories are convex regions" — meaning they have a smooth, blob-like shape without holes.

**Semantic Field Theory (Trier, 1930s):** Words don't carry meaning in isolation — they belong to overlapping semantic fields that "constantly blend into one another without rigid demarcation." This is exactly what overlapping Gaussian distributions look like.

---

## Words as Gaussians: The Better Primitive

In 2015, Luke Vilnis and Andrew McCallum proposed a simple but powerful idea: **represent each word not as a point, but as a Gaussian distribution** — a "cloud" in embedding space.

A Gaussian in d dimensions is defined by two things:
1. **Mean (μ):** The center — the "typical" meaning
2. **Covariance (Σ):** The shape and spread — how much the meaning extends in each direction

Visually (in 2D for simplicity):

```
Point embedding:          Gaussian embedding:
      •                      ╭───╮
     "cat"                  │ • │  ← "cat" (tight, precise)
                            ╰───╯

      •                   ╭─────────╮
    "animal"              │    •    │  ← "animal" (broad, encompasses many species)
                          ╰─────────╯

      •                  ╭──╮   ╭──╮
    "bank"               │• │   │• │  ← "bank" (two separate meaning regions)
                         ╰──╯   ╰──╯
                       financial  river
```

**What Gaussian embeddings solve:**

| Problem | Point Embedding | Gaussian Embedding |
|---|---|---|
| **Uncertainty** | No representation | Broad covariance = "I'm uncertain about this word's meaning" |
| **Breadth** | "Animal" and "poodle" are both single dots | "Animal" has large Σ (broad region); "poodle" has small Σ (precise spot) |
| **Polysemy** | One compromised point | Multiple Gaussians — one per sense (Gaussian mixture) |
| **Asymmetry** | Cosine similarity is symmetric | KL divergence between Gaussians is asymmetric: "dog" fits inside "animal" but not vice versa |

The last point is particularly elegant. To check if "dog IS-A animal," you compute the KL divergence from the "dog" Gaussian to the "animal" Gaussian. If "animal" is broad and "dog" is narrow and contained within it, the divergence is small → entailment. The reverse (animal IS-A dog) would have high divergence → not entailment. Point embeddings can't do this because cosine similarity is the same in both directions.

---

## A Decade of Validation (2015-2026)

Word2Gauss wasn't a one-off. It spawned a productive lineage:

- **Athiwaratkun & Wilson (2017):** Extended to Gaussian *mixtures* — multiple clouds per word — capturing polysemy. "Bank" gets two separate Gaussians, one for each meaning.
- **Athiwaratkun & Wilson (2018):** Showed that the *spread* of a Gaussian encodes generality/specificity. Hypernyms ("animal") have large determinant; hyponyms ("poodle") have small determinant.
- **Chen et al. (2015, GMSG):** Dynamic Gaussian mixtures that automatically split into more components during training when a single Gaussian can't capture all senses.
- **Yoda et al. (2023, GaussCSE):** Extended Gaussians from words to entire *sentences* — proving the representation scales beyond the word level.
- **Yuksel et al. (2021):** Used Gaussian covariance to track how word meanings *change over time* — the covariance shifts capture semantic drift.

The evidence is clear: **Gaussians are a more natural mathematical primitive for meaning than points.**

---

## The Missing Piece: Composition

Here's the gap. We know:
- Words as Gaussians: validated (2015-2026, 7+ papers)
- Sentences as Gaussians: validated (GaussCSE, 2023)

But **how do word Gaussians combine into sentence Gaussians?**

When you hear "the big red ball," how do the Gaussians for "big," "red," and "ball" compose into the meaning of the phrase? Simple averaging? Some kind of blending? A learned operation?

Word2Gauss computes pairwise similarity between word Gaussians (via KL divergence). GaussCSE represents the final sentence as a Gaussian. But neither specifies the *composition mechanism* — the operation that takes multiple word Gaussians and produces sentence meaning.

This is where 3D Gaussian Splatting enters the picture. In computer graphics, there's a well-established mathematical framework for taking many overlapping Gaussians and compositing them into a single coherent output. It's called the **rendering equation**, and it uses **alpha-compositing** — a technique for blending overlapping elements, accounting for occlusion and transparency.

The core idea of Semantic Gaussian Splatting: **use the rendering equation from 3D Gaussian Splatting as the composition mechanism for semantic Gaussians.** Sentence meaning is "rendered" from word Gaussians the way an image is rendered from scene Gaussians.

To understand why this might work, we first need to understand what radiance fields and Gaussian splatting actually are.

---

*Next: [Article 3 — What Is a Radiance Field?](03-radiance-fields-explained.md)*
