# Article 5: The Leap — Radiance Fields for Meaning

*What it means to "render" a sentence, and why this isn't just a metaphor*

---

## The Core Idea

In 3D Gaussian Splatting, a scene is a collection of Gaussians in physical space. Each Gaussian has a position (where it is), a shape (how it extends), a weight (how prominent it is), and features (what it looks like). You point a camera at the scene, and the rendering equation composites these Gaussians into an image.

The SGS proposal: **language meaning works the same way.**

A vocabulary is a collection of Gaussians in *semantic* space. Each Gaussian has a position (core meaning), a shape (semantic breadth), a weight (salience), and features (rich semantic attributes). You pose a *query* to the vocabulary, and the rendering equation composites these Gaussians into a *meaning vector*.

This article is about whether this analogy is deep or shallow — whether "rendering meaning" is a genuine computational framework or a pretty metaphor that collapses under scrutiny.

---

## What Is a Semantic Radiance Field?

A visual radiance field answers: "What color and density exist at point (x, y, z) in 3D space?"

A **semantic radiance field** answers: "What meaning and importance exist at point μ in semantic space?"

In the visual case, the field is populated by Gaussians representing physical objects — surfaces, textures, materials. Where Gaussians overlap, appearances blend.

In the semantic case, the field is populated by Gaussians representing words and concepts. Where Gaussians overlap, *meanings* blend.

Consider the word "warm." Its Gaussian sits in semantic space with a certain center (the prototypical sense of warmth — temperature) and a covariance that extends toward:
- Physical warmth ("warm water")
- Emotional warmth ("a warm smile")
- Color warmth ("warm tones")
- Mild intensity ("warm applause")

Nearby, you find Gaussians for "hot" (overlapping on the temperature dimension but more intense), "cozy" (overlapping on the emotional dimension), "tepid" (overlapping on temperature but less positive). These Gaussians overlap and blend in exactly the way Trier's semantic field theory predicted: "fields constantly overlap and blend into one another without rigid demarcation."

The semantic radiance field is this entire landscape of overlapping word Gaussians. It's not built for one sentence — it's the model's entire knowledge of word meanings. A sentence activates a *subset* of this field, just as a camera view captures a *portion* of a visual scene.

---

## How a Sentence Becomes a Scene

When you input the sentence "The warm coffee sat on the old wooden table," the SGS model:

**1. Activates Gaussians.** Each token looks up its Gaussian(s) from the vocabulary. "Warm" activates its Gaussian (or Gaussians, if polysemous). "Coffee" activates its own. "Table" activates its own. These activated Gaussians form the **sentence scene** — a subset of the full semantic field.

**2. Applies positional modulation.** Just as 4D Gaussian Splatting shifts Gaussians over time, SGS shifts Gaussians based on their position in the sentence. "Warm" at position 2 gets a slight displacement in semantic space that encodes "I'm the second word, likely an adjective." This is analogous to adding positional encoding in a transformer — but instead of adding a vector to a point, we're shifting a Gaussian's center and adjusting its salience.

```
After activation, the sentence scene contains ~9 Gaussians
(one per token, possibly more for polysemous words)
floating in 64-dimensional semantic space
```

**3. The scene is now "renderable."** Different queries will extract different information from this same scene — just as different camera angles show different views of the same 3D arrangement.

---

## Rendering Meaning: A Worked Example

Let's render the meaning of "The warm coffee sat on the old wooden table" from a specific query viewpoint.

The query might be: "What is the main subject and its properties?" — formalized as a query vector q pointing toward the "agent/subject" region of semantic space.

The rendering equation evaluates:

```
Meaning(q) = Σᵢ fᵢ · αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ
```

Compositing in sequence order (left to right):

**"The"** (position 1)
- K(q, μ_the) = 0.05 — "the" is far from the agent/subject region in semantic space
- T₁ = 1.0
- Contribution: 0.05 × α_the × T₁ = very small
- Transmittance barely drops

**"warm"** (position 2)
- K(q, μ_warm) = 0.30 — "warm" is moderately relevant to subject properties
- T₂ ≈ 0.97
- Contribution: moderate — it's a property of the subject

**"coffee"** (position 3)
- K(q, μ_coffee) = 0.85 — "coffee" is very close to the agent/subject query!
- T₃ ≈ 0.88
- Contribution: high — this IS the subject
- Transmittance drops significantly (coffee "absorbs" a lot of the query's capacity)

**"sat"** (position 4)
- K(q, μ_sat) = 0.40 — the verb is relevant to what the subject does
- T₄ ≈ 0.35 (much of the transmittance was consumed by "coffee")
- Contribution: moderate but reduced by transmittance

**"on the old wooden table"** (positions 5-9)
- These Gaussians are in the location/object region of semantic space — farther from the "agent" query
- Plus, transmittance is already partly depleted
- They contribute little to this particular query

**Result:** The rendered meaning vector is dominated by "coffee" (the subject), with secondary contributions from "warm" (its property) and "sat" (its action). The spatial/location information ("on the old wooden table") is de-emphasized for this query.

**Now render from a different viewpoint** — query: "Where is the action taking place?"

This query vector points toward the spatial/location region of semantic space. Now:
- "Coffee" has low kernel value (it's an object, not a location)
- "Table" has high kernel value (it's the location)
- "Old" and "wooden" contribute as properties of the location

**Same scene, different viewpoint, different rendered meaning.** This is the multi-view property — and it's what makes SGS fundamentally different from bag-of-words approaches. The sentence scene contains *all* the information; the viewpoint determines which aspects are extracted.

---

## Why This Isn't Just Averaging

Consider what simple averaging would produce:

```
Average = (f_the + f_warm + f_coffee + f_sat + f_on + f_the + f_old + f_wooden + f_table) / 9
```

This gives one vector for the whole sentence — a semantic centroid that smashes together subject, verb, properties, and location. It can't distinguish "what is the subject?" from "where did it happen?" — you always get the same average.

The rendering equation adds three mechanisms that averaging lacks:

**1. Locality (Gaussian kernel).** The query selects which Gaussians are relevant. A "subject" query naturally weights nouns; a "location" query naturally weights prepositional phrases. This selection is automatic — it falls out of the geometry of where Gaussians sit in semantic space relative to the query.

**2. Occlusion (transmittance).** Once a highly relevant Gaussian (like "coffee" for a subject query) contributes, it absorbs transmittance — reducing the contribution of later words. This prevents redundant information from swamping the output. If there were two subjects ("Coffee and tea sat on the table"), both would contribute before transmittance depletes — naturally handling coordination.

**3. Order (sequence compositing).** The compositing order (left to right) gives earlier words a structural advantage, which the model can learn to adjust via opacity. In English, subjects typically precede verbs, so the default order bias aligns with syntactic structure.

---

## The Analogy Table: Where It Holds

| Visual Rendering | Semantic Rendering | Analogy Quality |
|---|---|---|
| Camera sees objects from an angle | Query extracts meaning from a perspective | **Strong** — multi-view = multi-aspect extraction |
| Nearby objects dominate the view | Nearby-in-semantic-space words dominate the meaning | **Strong** — locality is meaningful in both domains |
| Opaque objects block what's behind | Salient words absorb query capacity | **Medium** — transmittance is a capacity model, not physical occlusion |
| Gaussian shape = object geometry | Gaussian shape = semantic breadth | **Strong** — broad Gaussians in both domains mean "covers a large area" |
| View-dependent color (specular highlights) | Context-dependent meaning (polysemy resolution) | **Strong** — appearance changes with viewpoint in both domains |
| Scene is a persistent 3D structure | Vocabulary is a persistent semantic structure | **Strong** — both are queried, not consumed |

---

## The Analogy Table: Where It Strains

| Visual Rendering | Semantic Rendering | Issue |
|---|---|---|
| Depth ordering is physically determined | Sequence ordering is a convention | **The physics grounds visual depth; sentence order is syntactic, not geometric** |
| Alpha-blending adds light contributions | Alpha-blending adds semantic contributions | **"Not happy" requires subtraction, not addition. Light doesn't negate.** |
| A scene has ~10⁶ Gaussians | A sentence has ~10-100 Gaussians | **Visual scenes are richer; sentences are sparse. Does the equation work with so few elements?** |
| Pixel color is continuous | Token prediction is discrete | **The rendering output must eventually become a discrete word choice** |

These strains are real and addressed in the full architecture (operator Gaussians for negation, autoregressive decoding for discrete output, multi-pass for depth of computation). But the core rendering-as-composition operation is mathematically clean for the ~85% of language that IS additive composition.

---

## A Deeper Way to See It: Meaning as a Field You Query

Forget the 3DGS analogy for a moment. Think about it from the language side.

You have a sentence. It contains information. Different questions about that sentence extract different information. "Who?" gives you the subject. "Where?" gives you the location. "How?" gives you the manner.

In a transformer, all of this is encoded in the same sequence of hidden-state vectors. Different attention patterns extract different information.

In SGS, the sentence is a **semantic field** — a distribution of meaning-density over semantic space. The word "coffee" creates a peak of meaning-density in the beverage/object region. "Table" creates a peak in the furniture/surface region. "Warm" creates a peak in the temperature/sensation region.

When you query "What is the subject?", you evaluate this field at the agent-region of semantic space. The field has high density there (because "coffee" is nearby), so you get a strong signal. When you query "Where?", you evaluate at the location-region, finding "table."

**The sentence IS a radiance field of meaning.** Different queries shine light on different parts of it. The rendering equation is how you read the field.

This is not a metaphor. It's a computation: evaluate Gaussians, weight by proximity and transmittance, sum the features. The same math that composites visual scenes composites semantic scenes. The question is whether this math — designed for light — is also correct for meaning.

That's an empirical question. The answer comes from experiments. But the theoretical alignment between:
- Linguistics (concepts are convex regions — Gärdenfors)
- NLP (word embeddings have geometric structure — Mikolov)
- Probabilistic ML (words are well-modeled as Gaussians — Vilnis)
- Computer graphics (Gaussians compose via the rendering equation — Kerbl)

...is too tight to be coincidental. These four independent traditions converge on the same mathematical structure: overlapping Gaussians in a continuous space, composed by a well-defined blending operation.

SGS is the proposal to close the loop — to use the composition mechanism from computer graphics (rendering) for the composition task in NLP (sentence meaning from word meanings).

---

*Next: [Article 6 — SGS Architecture: The Full Machine](06-sgs-architecture.md)*
