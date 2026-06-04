# Raum 1.6 / 0.6 LinkedIn Post — a castle from a sentence

Follow-up to the Raum 1.5 post ("castle from a sentence", but collapsing every
prompt onto one template). Tone: formal, plain, no em dashes, no "not X but Y"
contrastives, precise numbers, honest about what broke.

---

Raum 1.5 could generate a castle from the words "a castle on a hill". It had
two problems. It collapsed every prompt onto the same template, so "a wall with
a gate" also produced a whole castle. And it rendered as flat coloured dots.

Raum 1.6 + 0.6 fixed both. Here is what changed, what broke along the way, and
where it landed.

What we changed from 1.5.

Accuracy. The 1.5 decomposer was trained on two scenes (castle, tower), castle
dominant, so it learned one answer. We rebuilt the training data around a
SceneSpec: a structured description that generates the tree AND the prompt text
from the same source, so they can never disagree. Every element became a
standalone labelled scene (wall, gate, door, window, arrow slit, arch, tower,
keep, gatehouse, hill, cliff, tree), and the prompts describe counts and
presence ("a castle with two towers", "a fort without trees"). 21,000 records,
retrained the decomposer on the frozen Planck 1.3 base, validation loss 0.107.

Realism. The viewer drew every Gaussian as a flat screen-facing disc. We
replaced it with lit, oriented ellipsoid splats sized per Gaussian, with sun,
fill, and hemisphere light for shading. Walls became courses of flat
slab-stones. Windows and arrow slits are carved into the wall faces.

What we encountered and learned.

Most of the work was downstream of the model. The decomposer was fine. The
failures were in the plumbing between a clean tree and a rendered scene, and
each one taught something.

The model derails. At temperature 0.3 it occasionally picked a bad token
mid-tree and never recovered, emitting broken JSON. Structured output wants
near-greedy decoding. Greedy first, with a retry, removed the flakiness.

The model is reliable about WHAT, noisy about WHERE. It names the right parts
(four towers, four walls, a keep) but places them loosely. So the model chooses
composition and the grammar snaps the layout: towers to the wall-ring corners,
walls to the faces, keep to the centre. Snapping position alone was not enough;
the model's noisy scales had to be snapped too, or towers came out stretched.

Refinement was hurting, not helping. An SGS step pulled clusters toward
matched template scans by Chamfer distance. It was built for messy inputs.
Against clean snapped geometry it only dragged things toward random scans and
distorted them. Turning it off was a fix.

Real scans are not a free win. We routed parts to the 109 real architecture
scans, expecting a realism jump. The scans are low-resolution ruin fragments,
and stretched to a part's footprint they read as worse than the hand-built
geometry. The lesson is that "use real data" only helps when the data is good
enough. It stays an option, off by default.

And one clean bug. Walls kept cutting through the castle centre instead of
ringing the towers. The scene flattener applied a node's scale to its child
positions but never its rotation, so a wall yawed to run along one axis kept
running along the other. One missing quaternion rotation.

The result.

"a castle on a hill" typed into the demo produces a castle on a hill. Four
corner towers with conical roofs, walls forming the perimeter between them, a
keep, on a dense green dome with trees on the slope. "a wall with a gate"
produces a gated wall. "a tower on a hill" produces a tower. The prompt
controls the scene.

The honest limit is that the geometry is hand-authored procedural primitives,
and hand-tuning constants (tower proportion, roof height, hill size) is the
ceiling. It does not generalise to arbitrary structures, and the scan
experiment showed that swapping in low-quality real geometry is not the answer.
The next iteration is about learning proportions and geometry rather than
tuning them by eye.

For now: text in, a recognisable castle out, rendered as lit stone. That was
the goal.
