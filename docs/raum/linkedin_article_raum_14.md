# A castle made of the wrong-sized boxes: what Raum 1.4 taught us about fidelity

*Part of the Semantic Gaussian Splatting series. Earlier entries covered SGS
for text (Planck), audio (Klang), and the first 3D scenes (Raum 0.x).*

Tone note for the file: formal technical narrative, no em dashes, no
"not X, but Y" contrastives, no evaluative adverbs, plain language for the
idea and precise numbers for the results. Honest about failure.

---

The earlier Raum posts ended on a working but coarse demo. Raum 0.1 could
take a sentence and place a handful of template objects in 3D space. "A red
sphere above a blue cube" came out right. A castle came out as a single blob
the size of a castle. The skeleton was there. The surface was not.

Raum 1.4 is the iteration that went after the surface. This is the story of
what we aimed for, how the system works, where it broke, the pivot it forced,
and where it actually landed. The short version: we hit our density target,
missed our fidelity target, and learned exactly which stage to fix. That is a
result worth writing down.

## What we set out to do

Three goals.

Density. Raum 0.1 scenes were a few thousand Gaussians. A real 3D Gaussian
Splatting scene is tens of thousands. We wanted a pipeline that takes a sparse
skeleton and grows it into a dense cloud, the way the original 3DGS densifies a
point cloud during optimization. Target: 50,000 plus splats per scene.

Shape resolution. A node in the scene tree called "tower" should read as a
tower, not as a sphere standing in for one. Each semantic part needs a
geometric primitive that matches its shape, and the parts need to subdivide
into enough detail that the silhouette is recognizable.

Surface, not points. A dense cloud of tiny dots still looks like dust. The
splats have to overlap and merge into something that reads as a solid surface
when you orbit it.

The version says 1.4 because this is the fourth pass on the 3D bridge, and the
first one aimed at fidelity rather than correctness. The thesis underneath is
unchanged. If SGS is a real multimodal primitive, the same compositing math
that renders meaning in Planck and sound in Klang renders 3D scenes in Raum.
The job of this iteration was to make those scenes dense and shaped enough to
look at.

## How the system works

Raum 1.4 is a four-stage pipeline. A sentence goes in. A dense Gaussian cloud
comes out, rendered live in the browser.

Stage 1, decomposition. A small Planck-class language model reads the prompt
and emits a composition tree. "A castle on a hill" becomes a tree with a scene
root, a castle node with child nodes for the main building, towers, and roof,
and a hill node beside it. Each node carries a name, a position relative to its
parent, and a scale. This is the semantic skeleton. The model was trained on
procedurally generated trees plus a set of LLM-generated trees for variety.

Stage 2, primitive fill. Each leaf node gets filled with a cloud of Gaussians
shaped to its name. "Tower" becomes a cylinder. "Roof" becomes a cone. "Hill"
becomes a dome. A building body becomes a solid box. An open space like a
courtyard becomes a flat plane. This is a lookup from word to shape.

Stage 3, subdivision and densification. The filled skeleton is still coarse, a
few hundred Gaussians. Subdivision expands each Gaussian into a small local
cluster, using a library of real architecture scans where a node name matches a
template category, and a procedural box fill where it does not. Densification
then runs a gradient-driven loop that clones and splits Gaussians in
under-covered regions, the same mechanism 3DGS uses during reconstruction,
until the cloud reaches tens of thousands of splats. A castle scene grows from
around 400 skeleton Gaussians to roughly 62,000.

Stage 4, refinement and render. An SGS-native pass nudges each spatial cluster
toward its nearest architecture template by Chamfer distance, fills the worst
gaps, and pushes opacity toward visible. The result streams to a WebGL viewer.
Each splat is drawn as an opaque round disc sized by its own Gaussian scale, so
overlapping splats merge into a surface rather than reading as separate dots.

The whole round trip runs locally on one GPU in about ten seconds. No
photographs, no multi-view capture, no frontier model in the loop. A sentence,
a small model, and the splatting math.

## What went wrong

Raum 1.4 hit its density target and missed its fidelity target. The scenes are
dense. They do not look like what you asked for. Three failures, each
instructive.

Shape resolution. The primitive fill maps a word to a single coarse shape.
"Tower" becomes one cylinder. A real tower has walls, a top, a base, windows.
We never had the resolution inside a part to express any of that, so every part
reads as its crudest possible approximation. A castle is a box with cylinders
on it. Recognizable as architecture, not as a castle.

Lack of training where it counted. The decomposer was trained on generic
procedural trees and a scatter of LLM-generated ones. It knows how to nest a
tree and place children, in the abstract. It does not know what a castle is
actually made of. So it emits plausible-looking trees with generic node names,
and the geometry that comes out is generic. The model never saw enough castles
to learn castle structure.

Shapes too big. Subdivision and densification operate on parts that are already
large relative to the detail we want. When a single oversized Gaussian gets
densified, it grows a blob, not a feature. The result is mass in roughly the
right place with none of the fine structure that makes a silhouette legible.
The tower scene came out looking more like a tree than a tower, because a fat
cone on a stalk is what the pipeline actually produced.

The honest read: the pipeline is sound end to end. Decompose, fill, densify,
refine, render, all working, all fast, 33,000 to 62,000 splats per scene
depending on the prompt. The problem is upstream of the plumbing. We were
resolving meaning at the wrong granularity. A castle is not a labeled box. It
is an arrangement of smaller things, and we were never decomposing down to
those smaller things.

## The pivot

The failures pointed at one root cause. We were stopping the decomposition too
early. A castle node filled with a box is a castle the way a stick figure is a
person. The fix is to keep decomposing until you reach parts small enough that
a simple shape is actually the right answer.

Call it atomic decomposition. Instead of "castle is a box," the target is
"castle is walls and towers and a gate and a roof, walls are courses of stone,
a tower is a stack of stone with crenellations on top." You descend the tree
until the leaf is something a primitive can honestly represent. A single stone.
A single brick. A single tree in a forest. At that level a box or a small
cluster is not an approximation. It is the thing.

This changes what we train and on what. The decomposer was learning from
ShapeNet-style generic object shapes and procedural trees. We are moving it onto
castle scans: real architecture, decomposed into its actual parts, so the model
learns that a castle has a curtain wall and corner towers and a keep and a
gatehouse, in roughly the right arrangement, rather than emitting a flat list of
generic children. Rather than hand-label thousands of castle trees, we generate
the decompositions and use them as training targets, then refine the generator
against what actually renders well. The supervision is the tree itself, so the
model learns to produce trees that subdivide cleanly all the way down to atomic
parts.

The bet is that fidelity is a decomposition problem before it is a rendering
problem. If the tree goes deep enough and the leaves are atomic, the existing
fill, densify, and render stages already produce a surface. We saw that in 1.4.
The surfaces are solid. They are just solid in the shape of the wrong,
too-shallow tree.

## The final result

Here is where Raum 1.4 actually landed. This is the honest version, and it is
not where we wanted to get to.

"A castle on a hill" produces 62,280 splats. It renders as a solid blocky mass
with leg-like columns underneath and a vague tower on top. You can tell it is a
built structure on a raised ground plane. You cannot tell it is a castle.

"A tower" produces 33,240 splats. It renders as a cone on a thin stalk with
clusters around the base. It reads as a tree more than a tower.

"A gate in a wall" produces 62,463 splats. This one comes closest. A flat slab
with an opening, recognizable as a wall. The simplest prompt gave the most
legible result, which tells you exactly where the ceiling is. The pipeline
handles a single flat part well and loses coherence as the structure gets more
compositional.

What worked: the plumbing. Every scene is dense, in the tens of thousands of
splats. The splats merge into solid surfaces now, after the viewer was fixed to
size each splat by its own Gaussian scale and draw it as an opaque disc instead
of a fixed dot. Ten seconds per scene, end to end, locally. The skeleton is in
the right place. The mass is in the right place.

What did not: the shapes. Still blocks. No fine structure, no crenellations, no
windows, nothing that turns a box into a building. The decomposition stops three
or four levels too high, so the geometry is honest about a tree that was never
detailed enough.

This is a useful negative result. It isolates the problem to one stage. The
bridge, the densifier, the renderer are not the bottleneck. The decomposer is.
That is the thing to fix, and it is a tractable thing to fix.

## What comes next

The plan to push fidelity, in rough order of leverage.

Retrain the decomposer on castles and hills only. Narrow the domain hard. A
model that has seen ten thousand castle decompositions and nothing else will
produce far better castle trees than a generalist that has seen a little of
everything. Domain-specific resolution first, breadth later, one domain at a
time.

Atomic primitive training. Add training for the smallest building blocks. A
brick. A stone. A roof tile. A single tree. Once the decomposer can reach these
as leaves and the fill stage has matching atomic primitives, a wall becomes
courses of stone instead of a smooth slab, and the surface carries real texture.

Compositional awareness. Teach the decomposer how parts relate, not just that
they exist. Towers sit at the corners of a curtain wall. A gate sits in the
middle of a wall face. A roof caps a building and does not float beside it.
Right now the tree nests but the spatial arrangement is loose. Compositional
constraints in the training targets would tighten it.

Scale discipline in the fill stage. The "shapes too big" failure was partly a
fill problem. Seed leaf Gaussians small relative to their parent so
densification grows features instead of blobs. A cheap change with visible
payoff.

A feedback loop from render to tree. The refinement stage already matches
clusters to real architecture scans by Chamfer distance. That signal can flow
back into the decomposer as a reward. Trees that render close to real
architecture get reinforced. This closes the loop between how we decompose and
how it looks.

Per-part primitive variety. One cylinder for every tower is the problem in
miniature. A small library of tower shapes, wall shapes, roof shapes, selected
by the bridge the way Raum 0.1 selected object templates, would break the
monotony without needing full atomic decomposition everywhere.

The through line is the same as the pivot. Fidelity is a decomposition problem.
Make the tree deep, domain-specific, atomic, and spatially aware, and the
rendering stages we already have will draw the rest.

Raum 1.4 is a negative result with a clear next move. That is the kind worth
shipping.
