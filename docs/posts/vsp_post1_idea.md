# LinkedIn Post 1: the VSP idea (why a token should be more than a word)

Illustration: docs/posts/img/vsp_token.svg
Tone: plain, factual, understated. No em dashes. Plain text below for paste.

---

A note on something we are working on at Radiance Labs.

Most language models give the word "crane" a single vector. But a crane is a bird and also a machine that lifts steel. Same spelling, two unrelated things. The model relies on surrounding context to tell them apart, which usually works and sometimes does not.

We are testing a different representation. Our models compose meaning using the rendering equation from 3D Gaussian Splatting, so a token can carry more than a word embedding. We are looking at giving each grounded token three parts:

V, visual: what the thing looks like, derived from images.
S, semantic: the usual language embedding.
P, physical: material properties like hardness and density.

The bird and the machine share the same S. Their V and P are different. If you combine all three, the two senses end up far apart without any special rule to separate them.

This is an experiment, not a result yet. We are checking whether it actually holds before building a model on it. Notes to follow.
