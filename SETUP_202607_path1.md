# SGS — Path A (v1, current architecture) Setup, July 2026

*Windows + RTX 4090, 1 TB drive. Other platforms not validated.*

Path A scales the WORKING Raum stack (text -> tree -> fill -> render, proven at
Raum 1.7) to Hertz sizes. No new architecture. Three model upgrades:

1. **Hertz as the prompt interpreter** (replaces Planck 1.3 as the decomposer base)
2. **Decomposer scaled to Hertz**
3. **Learned FILL** (conditional part -> Gaussian generator, replaces the
   hand-built grammar `_fill_gaussians`)

Roadmap row: `10-raum-0-8`. Runs parallel to Path B (SETUP_202607_path2.md) and
shares VSPS tokenization. Origin: 2026-06-22 planning, see
[[project_sgs_post17_directions]].

## 0. General setup

### 0.0 Environment

Reuse the existing venvs (see SETUP_202606_04.md §0): main `.venv` (torch 2.6 /
py3.12) for serving + data, `.venv-sds` (py3.10 / torch 2.4.1+cu124 / gsplat /
diffusers) for the differentiable-render / SDS supervision the learned fill
needs.

```powershell
cd sgs
.venv-sds\Scripts\Activate.ps1
python -c "import torch; print('cuda', torch.cuda.is_available())"   # must be True
python -c "import gsplat, diffusers; print('render+SDS deps ok')"
```

### 0.1 Gemma 4 for cheap data generation

Gemma 4 (huggingface.co/collections/google/gemma-4) generates new grammar +
training data on the cheap. It is multimodal (Any-to-Any), so it can emit a
part DESCRIPTION and a reference IMAGE together, useful for render supervision.
Pick by VRAM headroom alongside the SGS model on the 4090: start with
`gemma-4-E4B-it` (8B) or `gemma-4-12B-it`; the larger 26B-A4B / 31B only if the
4090 has room or generation runs on a separate pass. Accept Google's license on
the HF repo first.

```powershell
# download with the hf CLI (accept license on the repo page first)
hf download google/gemma-4-E4B-it --local-dir models/gemma-4-e4b-it
```

PRINCIPLE (do not skip): **fill richness caps usable data richness.** Gemma can
describe a billion parts; the learned fill can only learn the ones you can
render-supervise. Generate data as rich as the fill can render, not as rich as
Gemma can describe. Otherwise you teach the model to emit parts that render as
nothing.

## 1. Hertz demo run (interpreter sanity)

Before retraining anything, confirm the running Hertz 1.2 base (640M, see
[[project_sgs_hertz_maxtokens_trap]]) can serve as a prompt interpreter: load
the checkpoint, run a few prompts, confirm coherent next-token behaviour.

```powershell
# inspect the latest Hertz checkpoint + run a quick generation
python scripts/eval_lm.py --checkpoint checkpoints/hertz12/best.pt `
  --tokenizer data/wikipedia/tokenizer.model --prompt "a castle on a hill"
```

GATE: Hertz produces coherent output and loads as an SGS LM. If Hertz 1.2 is
still mid-run, use the latest rotated checkpoint (read-only; do NOT interrupt
the training run -- it shares the GPU).

NOTE (512 lever): the decomposer's OUTPUT tree is capped by generation context.
Blobs (src/blob_store.py) overcome the INPUT-knowledge limit (a retrieved blob
injects d_f content into the meaning render without spending token positions)
but do NOT extend output length. If trees need to be longer than context allows,
that is a hierarchical/chunked-emission problem, tracked separately.

## 2. Hertz decomposer training

Re-run the Raum 1.7 Stage 3 recipe (build_stage3_dataset.py +
train_decomposer.py) but on the HERTZ base instead of Planck 1.3. The decomposer
is the same prompt->tree job, just a bigger interpreter.

```powershell
# 1. build the dataset (weighted paraphrases + non-castle anti-forgetting mix
#    + Gemma-4-generated NEW scene types beyond castles)
python scripts/build_stage3_dataset.py --params output/layout_opt.params.json `
  --variants 8 --repeat 16 --out data/decomposition_trees/path1_train.json

# 2. fine-tune the decomposer on the HERTZ base
python scripts/train_decomposer.py `
  --data data/decomposition_trees/path1_train.json `
  --checkpoint checkpoints/hertz12/best.pt `
  --tokenizer data/wikipedia/tokenizer.model `
  --save-dir checkpoints/hertz_decomposer --epochs 40

# 3. judge with snapping OFF (the 1.7 gate, now at Hertz scale)
python scripts/infer_decomposer.py --checkpoint checkpoints/hertz_decomposer/best.pt `
  --tokenizer data/wikipedia/tokenizer.model --serve --port 8003 --no-snap
```

GATE: Hertz decomposer emits coherent trees (snap off) across MORE scene types
than the Planck castle, since the bigger interpreter + Gemma-expanded data
should generalize past one scene. Use diagnose_emission.py to count raw parts
per prompt before trusting renders (the 1.7 lesson: instrument output not picture).

## 3. Hertz learned fill

The real upgrade. Replace the hand-built grammar fill (`_fill_gaussians` ->
`expand_part`) with a LEARNED conditional generator: given a part token + pose
(position, scale, rotation), emit N Gaussians. This is "decompose to particles,
reconstruct in 100Ks of splats."

### 3.1 Generate (part -> Gaussian-cloud) training pairs

```powershell
# the grammar is the cheap data generator (same 0.5->1.5 trick): render every
# known part at varied scales/seeds to get (part-token+pose -> gaussian-set)
# pairs; Gemma-4 proposes NEW parts beyond the grammar, rendered by whatever
# can be grounded.
python scripts/build_fill_dataset.py --out data/fill/path1_fill.json   # TO BUILD
```

### 3.2 Train the fill model, render-score supervised

The supervision signal is the differentiable render / SDS path from
`sds_refine.py` (NOT template Chamfer matching -- Raum 0.6 proved Chamfer-to-scan
distorts clean geometry). Train so parts LOOK right under render, not just match
a template.

```powershell
.venv-sds\Scripts\Activate.ps1
python scripts/train_fill.py --data data/fill/path1_fill.json `
  --render-supervision sds --out checkpoints/fill_model   # TO BUILD
```

GATE: the learned fill renders a known part (tower) at least as well as the
grammar fill, AND can render a part the grammar never had a builder for. That
second half is the point: it lifts the grammar ceiling (the decomposer was only
ever as expressive as `expand_part`).

## What needs to be built (honest gap list)

- `scripts/build_fill_dataset.py` -- (part+pose -> gaussian-set) pair generator
- `scripts/train_fill.py` -- the conditional fill generator + SDS supervision loop
- Gemma-4 generation harness for new scene types / parts (extends
  generate_decomposition_trees.py, which currently calls Claude)
- Decomposer training already exists (train_decomposer.py); only the base
  checkpoint swaps Planck 1.3 -> Hertz

## Sequencing

| Phase | What | Depends on |
|-------|------|------------|
| 1 | Hertz demo run (interpreter sanity) | Hertz 1.2 checkpoint |
| 2 | Hertz decomposer training | Phase 1 + Gemma data |
| 3 | Hertz learned fill | Phase 2 + render-supervision harness |
| (par) | Product hotswap + Satz | any working interpreter/decomposer/fill |
