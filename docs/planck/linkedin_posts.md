# Planck 1.1 LinkedIn Posts

Four-part LinkedIn series on Planck 1.1 and the H-SGS (Hierarchical
Semantic Gaussian Splatting) blob concept. Continues the SGS thread
that already covers Semantic Gaussian Splatting for text, Planck 1.0,
and the three Klang posts on the audio track.

Tone: formal, technical narrative. No em dashes. No "not X, but Y"
contrastives. No evaluative AI adverbs. Plain language for the idea,
precise numbers for the results.

---

## Post 1. The blob concept, and why the architecture is already a RAG

Continuing the SGS series.

Earlier posts introduced Semantic Gaussian Splatting as a
representational primitive for sequence modelling, and Planck as the
small language model built on that foundation. This post introduces the
next layer in the stack. We call them blobs.

The idea is straightforward. A trained Planck model already produces a
continuous hidden-state vector at every position. If we take a
pretrained model, run it across a large text corpus, and cluster the
resulting hidden states, each cluster is a region of the model's
internal meaning space that the model already uses to explain a class of
contexts. We keep one Gaussian per cluster, parameterised with a centre,
a covariance, an opacity, and a feature payload. That collection of
Gaussians is the blob store.

At inference time, the blob pathway runs in parallel with the normal
forward pass. Each query position looks up its top-k blobs by
Mahalanobis distance, composites their feature payloads under the same
transmittance rule that SGS uses elsewhere, and hands the result to a
gate that decides how much of the blob signal to mix into the
prediction. The gate is learned. The blobs themselves can be frozen.

Three properties follow directly from this construction.

The first is that the blob store is a retrieval index by construction.
Top-k Mahalanobis is nearest-neighbour search in the model's own
embedding space. The gate decides when the retrieval is relevant and
when to ignore it. This is the same shape as retrieval-augmented
generation, except the index sits inside the model, uses the model's
own representations, and is trained against the model's own loss. There
is no separate retriever, no separate embedder, no bolt-on.

The second is that the index can be extended without retraining the
base model. New blobs are new Gaussians. Appending them does not change
the weights of the base, and the gate has already learned how to down-
weight irrelevant lookups. This gives us a path to in-model memory
that grows over time, on a model that keeps its original behaviour on
anything the new blobs do not explain.

The third is that the blobs are frequency-weighted by construction. A
concept that appears often in the training data produces many hidden
states in its cluster, which produces a higher-opacity blob. Rare
concepts get low-opacity blobs and a lower effective contribution at
inference time. The model's confidence on common ground is structurally
higher than on uncommon ground, and the blob store makes that
asymmetry explicit.

The targeted use cases come out of those properties. Factual answers
to the same question should be consistent, and blobs produce
consistency structurally because similar contexts resolve to the same
top-k. Code generation benefits from the same property for a different
reason: given two near-identical prompts, a user wants near-identical
code, not two distinct creative takes. Search and question-answering
benefit because the blobs can encode the background knowledge the base
model would otherwise have to approximate.

The next post describes how we tested whether this actually works on a
real model.

---

## Post 2. Planck 1.1, what we were trying to prove

Continuing the SGS series.

The previous post described H-SGS blobs as an architectural
retrieval layer built into an SGS language model. Planck 1.1 is the
first concrete test of that claim.

Setup. Take Planck 1.0, our 100M-parameter SGS model trained on
TinyStories. Run it across the TinyStories training set and collect its
hidden states. Cluster those hidden states into 50,000 Gaussians using
k-means++. Each Gaussian is stored with its centre, log-variance,
opacity, and the mean feature vector of its cluster. That is the
initial blob store.

Model. On top of Planck 1.0, add two components. The first is a small
projection that maps the composited blob feature to the model's meaning
space. The second is a learned gate that takes the local context and
produces a scalar per position: how much of the blob signal should mix
into this prediction. Top-k is fixed at 8, the transmittance cap at
0.3.

Training. Fine-tune the resulting model on TinyStories with the base
unfrozen. This is Planck 1.1.

The experimental question has four parts, and we wrote a four-gate
validator to answer them.

Gate 3, perplexity: does adding blobs lower the validation loss
relative to Planck 1.0? This is the headline question. If blobs do not
help the likelihood, nothing else matters.

Gate 2, utilisation: is the blob pathway actually being used? It is
possible for a learned gate to down-weight the blob signal to near-zero
and let the base handle everything. We measure the mean effective blob
weight across validation batches and require it to be materially above
zero.

Gate 1, base intactness: if we silence the blobs at inference, does
the model collapse back to something close to Planck 1.0? A pass means
the blob pathway is additive, not a crutch.

Gate 4, generation quality: does the generated text show more looping
or copy-paste within a single sample? The retrieval pathway could in
principle bias completions toward high-probability continuations in a
pathological way. We needed to measure that directly.

What the four gates together test is whether blobs are a real
architectural feature rather than a side-effect of more training. The
next post reports what the first run found.

---

## Post 3. Preliminary results

Continuing the SGS series.

The previous post set out four validation gates for Planck 1.1. Here
are the numbers from the first full run.

Gate 3, perplexity. Planck 1.0 validation loss was 1.7504, perplexity
5.76. Planck 1.1 validation loss was 1.6672, perplexity 5.30. The
improvement is 0.083 in log-loss, a 14 percent relative reduction in
perplexity. Gate 3 passes.

Gate 2, utilisation. The mean effective blob weight across validation
batches was 0.059, above the 0.05 floor. The blob pathway is
contributing at inference, not being gated out by the learned mixer.
Gate 2 passes.

Gate 1, base intactness. This one needs care. The gate evaluates
Planck 1.1 with the blob pathway silenced (transmittance cap set to
zero) and compares the resulting validation loss to Planck 1.0's. The
pass condition is that the two losses sit within 0.05 of each other.
The observed delta was 0.083, and the direction was downward: Planck
1.1 with blobs off was better than Planck 1.0.

That is a gate-design issue, not a model regression. The Planck 1.1
fine-tune updated the base weights as well as the blob head. So
"Planck 1.1 with blobs off" is "Planck 1.0 after an additional pass of
training", not "Planck 1.0". Some of the observed improvement is the
extra training. Some is the blobs. This gate, as specified, cannot
separate the two. A clean version requires training the base-frozen
variant, which is queued for Planck 1.3.

Gate 4, repetition. The first version of this gate counted repeated
4-grams across 50 generated samples per model. Planck 1.0 produced 258,
Planck 1.1 produced 417. The headline read: more repetition, gate fails.

On closer reading, that number conflates two different behaviours. One
is looping inside a single generation, which is a real failure mode.
The other is consistency across different completions of the same
prompt, which is the main upside of the blob architecture for factual
and code-style output. The aggregate count cannot tell them apart.

We have rewritten Gate 4 as two separate measurements. Gate 4a counts
repeated 4-grams within each generation and averages per sample. Gate
4b measures overlap across samples of the same prompt via unique-n-gram
ratio and pairwise Jaccard. Gate 4a is the new hard gate. Gate 4b is
reported but not treated as pass or fail, because high cross-sample
agreement is exactly the property we want for retrieval-style output.

A rerun on the new gate is in flight. The next post reports the final
numbers and what they imply for Planck 1.3.

---

## Post 4. Gate results and the Planck 1.3 plan

Continuing the SGS series.

The previous post explained why the original Gate 4 was split into
Gate 4a (intra-sample repetition, hard gate) and Gate 4b (cross-sample
diversity, informational), and why Gate 1 as specified is not yet
diagnostic. Here are the updated numbers from the rerun.

Gate 4a, intra-sample repetition. Mean repeated 4-grams per generation
over 50 matched samples: Planck 1.0 = 5.16, Planck 1.1 = 8.34. Delta
+3.18, a 62 percent relative increase. The pass condition was Planck
1.1 less-than-or-equal to Planck 1.0. Result: fail.

Gate 4b, cross-sample diversity. Unique-4-gram ratio across the 50
samples: Planck 1.0 = 0.864, Planck 1.1 = 0.825. Mean pairwise Jaccard
over the same sets: Planck 1.0 = 0.0133, Planck 1.1 = 0.0122. The
direction is the one we wanted (lower unique ratio and lower Jaccard
mean together indicate slightly more agreement across runs of the same
prompt), but the magnitude is small. Unique ratio drops 4.5 percent,
Jaccard drops 8.8 percent. An order of magnitude smaller than the Gate
4a regression it would have to offset.

Full picture. Gate 3 passed (val loss 1.7504 to 1.6672, perplexity 5.76
to 5.30). Gate 2 passed (mean effective blob weight 0.059, above the
0.05 floor). Gate 4a failed. Gate 1 remains a false fail as specified
and will stay that way until we retrain the ablation with the base
frozen.

Net reading. The mechanism works. Blobs lower the likelihood,
measurably and non-trivially, and the learned gate is using them at
inference. But in the Planck 1.1 configuration, the retrieval pathway
pulls generations into loops inside a single sample more than it pulls
different samples of the same prompt into agreement. The property we
wanted, retrieval-as-consistency, shows up only in a weak form. The
property we did not want, retrieval-as-repetition, shows up strongly.
That is why Gate 4a is the right gate.

Decision. H-SGS does not enter Hertz 1.2 in this configuration. Hertz
1.2 ships on the acceleration recipe already planned, without blobs.
Blobs move into the Planck 1.3 track, and the question of whether they
make Hertz at all is answered after 1.3.

Planck 1.3 plan. Five items, ordered by cost and by how directly they
attack the Gate 4a failure.

First, a top-k and transmittance-cap sweep on the existing Planck 1.1
checkpoint. No retraining. We try top-k in four, eight, sixteen and a
transmittance cap in 0.1, 0.3, 0.5 at inference and measure Gate 4a on
each combination. This tells us, cheaply, whether the looping can be
dialled out at decode or whether it is baked into the fine-tuned
weights.

Second, a frozen-base retrain for the Gate 1 ablation. Train a 1.1
variant with the base weights frozen and only the blob head learning.
That is the checkpoint Gate 1 should be comparing against. Independent
of the Gate 4a work, cheap, and it cleans up the outstanding
methodological issue.

Third, a blob-count sweep. Fifty thousand to two hundred thousand to
five hundred thousand, everything else held fixed. The working
hypothesis for the Gate 4a failure is blob collision: when the store is
small relative to the diversity of contexts, similar contexts resolve
to the same top-k and the retrieval pathway pulls the decoder toward a
narrow set of continuations that then repeat. More blobs spread that
pull across more attractors. If the hypothesis is right, Gate 4a
approaches Planck 1.0 levels as the store grows, without losing the
Gate 3 improvement. This is the decisive experiment.

Fourth, live blob addition. Freeze the model and the blob projection,
append new blobs built from a held-out slice of TinyStories, and
measure three things. Do new prompts route to new blobs? Does
perplexity on the held-out slice improve without retraining? Do Gate 2
and Gate 4a stay stable on the original slice? This is the
architectural test of the built-in RAG framing. Worth running only
once Gate 4a is under control from step one or step three.

Fifth, an inference-time transmittance dial exposed as a CLI flag. A
small plumbing change on top of step one. One checkpoint, two modes: a
high cap for factual and code generation where consistency is wanted,
and a low cap for creative output where variety is wanted.

Per-domain blob shards, separate stores for code, for factual recall,
and for creative writing selected by a classifier or a prompt marker,
move to Hertz 2.x. The idea needs the larger model and a more diverse
corpus than TinyStories to be testable.

The headline from Planck 1.1 stands. The retrieval pathway is real and
useful for likelihood. The question Planck 1.3 asks is whether it can
be made safe for generation at the same time. That is the condition
for blobs rejoining the Hertz track.

**Images**

Suggested slots, all already in the repo or straightforward to generate
from `results/planck11_validation.json`:

- Reconstruction-style comparison: Planck 1.0 vs Planck 1.1 sample
  outputs side by side on the same prompts.
- Gate 2 / Gate 3 bar chart: val loss, perplexity, mean effective blob
  weight.
- Gate 4 diagram (to be generated after the rerun): two stacked bars,
  intra-sample repeats and cross-sample overlap, for Planck 1.0 and
  Planck 1.1.
- Architecture diagram: base forward pass + parallel blob retrieval +
  gate, annotated to highlight the "RAG by construction" framing.
