# Raum 1.7 LinkedIn Post — the model places the castle, not the grammar

Follow-up to the Raum 1.6 / 0.6 post. Tone: formal, plain, no em dashes, no
"not X but Y" contrastives, precise numbers, honest about what broke.

---

Raum 1.6 could generate a castle from a sentence. The model named the right
parts, four towers, four walls, a keep, but it was noisy about where they go and
how big they are. So the model chose the composition and a hand-written grammar
snapped the layout: towers to the ring corners, walls to the faces, keep to the
centre, all from fixed constants we picked by eye.

That snapping was the ceiling. It works for castles and generalises to nothing.

Raum 1.7 removes it. The model now emits the positions and scales itself, and
they are good enough to render with no snapping at all. Here is how, what broke,
and the bug that almost made us call it a failure.

The approach, in three stages.

We did not train the model on the grammar's own constants. That is circular: it
just moves the magic numbers into the weights. The layout had to be learned
against a signal that does not come from the grammar.

Stage 1. We stood up a differentiable Gaussian-splat renderer with a Score
Distillation Sampling loss from Stable Diffusion, and asked one question on a
fixed scene: does the gradient improve a render against a text prompt without
dissolving the geometry? It does, once positions are frozen. Free positions let
the splats scatter chasing texture. The signal is real.

Stage 2. We made the castle's proportions free parameters, about five numbers:
ring radius, tower scale, wall height, keep scale, the seat height on the hill,
and optimised them against the render score with a small evolution strategy. No
grammar constant is read. The structure is fixed by construction so the search
cannot produce a disconnected castle, but every proportion is learned.

Stage 3. We fine-tuned the decomposer on the frozen Planck 1.3 base to emit
those learned proportions directly, so inference needs no per-scene
optimisation. Then we turned the snapping off and rendered the model's raw
output. Four towers on a ring, walls between them, a keep in the centre, placed
by the model.

What broke, and what it taught.

The search collapsed the walls. The proportion optimiser drove wall height to
its floor, because shorter walls gave the renderer less to disagree with. The
fix was to bound that one parameter to a sane range. The lesson is the same one
from Stage 1: give a search a degree of freedom that can break the structure and
it will use it.

The first proportion design scattered the towers. An early version let each
tower move freely while the walls stayed pinned, so the optimiser pulled the
towers off the ring and the walls no longer met them. Constraining the search to
proportions that keep the four-fold symmetry made a broken castle unreachable.

And the bug that almost ended the project. With snapping off, the first renders
showed a hill and a single tower, nothing else. It looked like the model could
not hold the full layout, that the base model was simply too small. We were one
decision away from shutting it down.

Instead we read what the model actually emitted, before rendering, and counted
the parts. The terse prompt "a castle on a hill" produced one tower. The richer
prompt "a stone castle on a green hill", the phrasing the training leaned on,
produced the full nine-part castle. The model had learned the layout. It was
sensitive to the wording, and we had been testing it on its weakest prompt, the
viewer's default. The failure was in the prompt, not the model.

The general lesson is to instrument the output, not the picture. A render
conflates "the model did not emit it" with "the model emitted it in the wrong
place". Counting the raw emitted parts told us which one we had, and the answer
changed the verdict completely.

Where it landed.

The model emits a coherent castle with no snapping and no magic constants. The
scaffolding that was the ceiling in 1.6 is gone. The proportions are learned
end to end from a render-scored objective.

What is still rough. The terse prompt under-recalls, which is a data weighting
problem, now being fixed by showing that phrasing more often. The hill lost its
colour in the shallow training format, fixed. Trees are not in the layout yet.
These are polish, not the thesis.

The thesis held. A model can learn where the parts of a scene go, and how big
they are, from a signal that is not the grammar that drew the training data.
That is the difference between a system that renders castles and a system that
could render anything.
