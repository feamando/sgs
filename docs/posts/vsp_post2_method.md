# LinkedIn Post 2: the method (four experiments, one decision)

Illustration: docs/posts/img/vsp_grounding.svg
Tone: plain, factual, understated. No em dashes. Plain text below for paste.

---

A follow-up on the token representation we are testing at Radiance Labs.

The claim to check: a token carrying what a thing looks like (V), means (S), and is made of (P) should separate senses that a normal embedding merges. The test case is "crane the bird" versus "crane the machine".

Before training anything, we ran a small separation test four ways and measured how far apart the two senses landed.

1. Hand-written V and P: separated well (0.48), but we wrote the labels, so this mostly confirms the setup, not the idea.
2. V and P derived from the word's text embedding: did not separate (0.13). The hardest cases share spelling, and text alone cannot tell them apart.
3. V from a fixed 3D-object category id: separated (0.41), but largely because different ids are different by definition. It also limits us to a small set of categories.
4. V from a vision-language model reading the word as text: did not separate (0.18).

The pattern: anything derived from text collapses the two senses. Images do not, since a picture of a bird and a picture of a construction crane are visually unrelated.

So the next step is to ground the visual part in generated images rather than text or a fixed category list. The plan is to enumerate a word's senses, generate views of each, embed them, and keep the senses that occupy clearly different regions of image space.

That test runs next on real generated views. If it holds, it informs our next model and a short write-up. If it does not, we will say so.
