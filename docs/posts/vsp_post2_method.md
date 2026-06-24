# LinkedIn Post 2 — the method (four experiments, one honest decision)

Illustration: docs/posts/img/vsp_grounding.svg
Tone: rigorous but accessible, no em dashes. Plain text below for paste.

---

How do you prove an idea is real before you spend a month training a model on it?

We had a claim: a token that carries what a thing looks like (V), means (S), and is made of (P) should separate senses that a normal model collapses. "Crane the bird" from "crane the machine." The test is whether V, S, and P together push the two senses apart when language alone holds them together.

Instead of training first and hoping, we ran the cheapest version of the experiment four ways. Each one taught us something, and three of them said no.

1. Hand-authored V and P. Separation worked (gain 0.48). But we wrote the labels, so we were partly grading our own homework. Promising, not proof.

2. V and P derived from the word's text embedding. Failed (0.13). The cases that most need help, the ones with identical spelling, collapsed completely, because text is the very signal that confuses them. Circular.

3. V from a curated 3D-asset category id. Passed (0.41), but the separation was almost tautological: different ids are different by construction. It also caps us at a few dozen object categories.

4. V from a CLIP reading of the word. Failed (0.18). Even a vision-language model, fed text, still sees the shared word and merges the senses.

The pattern is unmistakable. Every grounding that starts from TEXT collapses. The thing that cannot collapse is a picture: an image of a bird and an image of a construction crane are visually unrelated, no matter that they share a name.

So the direction is set. We ground the visual component in GENERATED IMAGES. We ask a model to enumerate a word's distinct senses, generate views of each, embed them, and keep only the senses that occupy genuinely different regions of image space. Open vocabulary, no fixed category list, and senses that separate because they look different, not because we labeled them.

The decisive test runs next, on real generated views. If it holds, it becomes the basis for Radiance Planck 2.0 and a paper on grounding meaning in rendered images.

The lesson we keep relearning: a cheap experiment that says no is worth more than an expensive one that says what you hoped.

Meaning, rendered.
