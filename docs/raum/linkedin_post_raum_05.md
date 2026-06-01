# Raum 0.5 LinkedIn Post — the fix worked

Short follow-up to the Raum 1.4 article ("A castle made of the wrong-sized
boxes"). Tone: formal, plain, no em dashes, no "not X but Y", precise numbers.

---

Last week I shipped a negative result: Raum 1.4 could generate dense 3D scenes
from a sentence, but a castle came out as a pile of wrong-sized boxes. The
post-mortem isolated the bottleneck to one stage, the decomposer, and made a
bet: fidelity is a decomposition problem before it is a rendering problem.
Prove the target is reachable deterministically first, then teach a model to
hit it.

Raum 0.5 is the first half of that bet, and it worked.

I wrote a scene grammar that builds a castle on a hill out of atomic parts.
Walls and towers are courses of small stones. Tops carry crenellations. The
gate is a wall segment with the center stones left out and an arch over the
gap. Four towers sit at the corners of a square wall ring, the keep stands
taller in the middle, the whole thing sits on a domed hill, and conifers
scatter on the slope outside the walls. No model in the loop. The same fill,
densify, refine, and render stages from 1.4, fed a tree whose leaves are small
enough that a simple shape is the honest answer.

The result is a castle on a hill. A green dome, four stone towers with red
conical roofs, crenellated walls between them, a keep, trees on the slope.
About 2,600 atomic parts densify to roughly 51,000 splats. It is low fidelity
and it is unmistakably a castle. The §3 gate, "recognizable to someone who was
not told what it is", is met.

Two things matter here beyond the picture.

First, it confirms the diagnosis. The 1.4 plumbing was never the problem. Feed
the exact same pipeline a tree decomposed to the right granularity and it
produces the right scene. The bottleneck was always how deep we decomposed.

Second, the grammar is not a demo dead end. It is the training-data generator
for the next step. Raum 1.5 retrains the decomposer on thousands of trees the
grammar produces, so a text prompt learns to emit the structure the grammar
proves looks right. The render path is shared, so a trained model that emits
"tower, wall, keep, hill, trees" expands to the identical atomic geometry you
see here.

0.5 is the proof. 1.5 is the model that reproduces it from a sentence.
