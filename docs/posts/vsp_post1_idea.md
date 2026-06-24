# LinkedIn Post 1: the VSP idea (why a token should be more than a word)

Illustration: docs/posts/img/vsp_token.svg
Tone: rigorous but accessible, no em dashes. Plain text below for paste.

---

A language model thinks the word "crane" is one thing.

It is at least two. A crane is a bird. A crane is a machine that lifts steel. They are spelled the same, so a normal tokenizer gives them ONE vector and lets context sort it out later. Most of the time that works. Sometimes it does not, and the model has quietly merged a bird with a construction site.

We are building language models a different way at Radiance Labs. Our models compose meaning using the rendering equation from 3D Gaussian Splatting, the same math a graphics engine uses to compose light. That gives us a natural question the usual approach never asks: if a token can be rendered, what should it actually contain?

Our answer is VSP. Every grounded token carries three things, bundled from birth:

V, the visual. What the thing looks like, taken from images, not from its spelling.
S, the semantic. What the word means in language, the usual embedding.
P, the physical. What it is made of and how it behaves: hardness, density, transparency.

A bird and a construction crane have the same S, because they share a word. But their V is completely different (feathers versus steel girders) and their P is completely different (light and soft versus heavy and rigid). Bundle all three and the two senses pull apart on their own. You do not need a disambiguation rule. The representation does it, because meaning was never only semantic.

This is the foundation for our next model, Radiance Planck 2.0. The bet is simple to state and hard to earn: a model that knows what words LOOK like and are MADE of, not just what they mean, should understand them more precisely.

We are testing that bet now, and being honest about what passes and what does not. More on that next.

Meaning, rendered.
