# SGS Gemma-Decomposer — Raum with a pretrained base, July 2026

*Windows + RTX 4090. Other platforms not validated.*

**Objective:** stand up **Gemma 4 as a Raum decomposer**, trained through the same
decomposition path we used for Hertz and Planck, and expose it as a selectable
model in the Raum/Satz app alongside them. Question this answers: does a strong
off-the-shelf pretrained base beat our task-saturated custom decomposer on the
prompt to structure-tree job, with little or no training?

Origin: path1 showed the Hertz decomposer hit PARITY with Planck (task saturated,
no scaling win, see [[project_sgs_path1_outcome]]) and Planck 2.0's VSP bundle LOST
its phase-4 gate by 3.8pts at full compute (see [[project_sgs_vsp_gate]]). Both say
the same thing: on a fixed, low-entropy task, our small custom bases are the ceiling.
Gemma 4 is the opposite lever, a large model that already knows language and JSON.
If it decomposes castles cleanly with a few examples, that is a cheap, honest win and
a live comparison baseline inside Raum.

DISCIPLINE: gate-and-kill, same as every other track. Prove the cheap step before
the expensive one. Few-shot prompting is Gate 0. Do NOT LoRA-tune before few-shot
reliability is measured, and do NOT build a per-request hotswap before launch-time
selection works.

## 0. What we KNOW going in (carry these forward)

Settled facts from the code map (2026-07-22) and prior tracks. The build must
respect them.

1. **The decomposer output is a STRUCTURE-ONLY tree, no gaussians.** Root
   `{"name":"scene","children":[...]}`; each node has `name`, `position [x,y,z]`,
   `scale` (float), optional `rotation` (quat [w,x,y,z]), optional `color [r,g,b]`,
   and either `children` or nothing (shallow leaf). Gaussians are added downstream
   by `_fill_gaussians()` (grammar `expand_part`, or the learned `FillModel`, or
   scanned splats). Schema is `CompositionNode` in `src/raum/decomposition.py`
   (`to_dict`/`from_dict`). Gemma only has to emit this JSON. Fill and render are
   untouched.
2. **Gemma already emits this format.** `scripts/generate_trees_gemma.py` prompts
   `google/gemma-4-E4B-it` to produce shallow-skeleton trees constrained to the
   fill-renderable `EXPANDABLE_PARTS` vocab, and we already used it to GENERATE
   training data. So the base is proven capable of the output shape; the open
   question is per-prompt reliability and coherence as a live decomposer.
3. **The training format is a plain causal-LM string.** `train_decomposer.py`
   feeds `f"DECOMPOSE: {prompt}\nTREE: {tree_json}"` (compact JSON). The dataset is
   records of `{"prompt": str, "tree": {...}}`. The Stage-3 castle dataset already
   exists at `data/decomposition_trees/path1_train.json` (built by
   `build_stage3_dataset.py`). Gemma reuses the SAME dataset, just wrapped in its
   chat template.
4. **The existing decomposer trainer is a FULL fine-tune of `SGSLanguageModel`.**
   `train_decomposer.py` builds the model from `infer_arch(state)` and runs
   `AdamW(model.parameters())`, no freezing, no adapter (the LoRA-adapter docstring
   is stale, the code does full FT). This path CANNOT load Gemma (Gemma is not an
   `SGSLanguageModel`). Gemma needs its own trainer.
5. **Gemma-4-E4B is ~8B.** Full fine-tune will not fit a 4090. Use **LoRA** (PEFT).
   This is also, ironically, what the SGS trainer's docstring falsely claims to do.
6. **The model registry and both apps are `SGSLanguageModel`-shaped.**
   `satz/app.py MODELS` and `infer_decomposer.py Decomposer` both construct
   `SGSLanguageModel` and expect a `.pt` state dict. Gemma is loaded via
   `transformers.AutoModelForCausalLM`. The clean fix is a **backend field**
   (`"sgs"` vs `"hf"`) dispatched behind one common `generate_tree(prompt) -> dict`
   interface, so everything downstream (fill, render, the selector UI) is unchanged.
7. **Tokenizer discipline.** SGS bases need their matching SentencePiece
   (`data/hertz12_data/tokenizer.model` for Hertz, `data/wikipedia/tokenizer.model`
   for Planck). Gemma brings its OWN tokenizer from its HF folder, never a
   SentencePiece `.model` from `data/`. Do not cross them.

## 0.1 Environment

```powershell
cd sgs
.venv\Scripts\Activate.ps1
# Gemma 4 needs a modern transformers (its tokenizer.json uses a newer schema;
# <4.45 fails with "data did not match any variant of untagged enum ModelWrapper")
pip install -U "transformers>=4.50" "tokenizers>=0.21" accelerate peft
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__)"
```

- Run Gemma work in the MAIN `.venv` (torch 2.6 / py3.12), NOT `.venv-sds`
  (diffusers pins would conflict).
- Do NOT mix with the SDS-path pin in `SETUP_202606_04.md`
  (`transformers==4.40.2` etc.). Different world.
- `peft` is the only new dependency vs path1 (LoRA). It is py3.11/3.12 clean and
  ships no heavy transitive deps (see [[feedback_tooling_precheck]]).

## 0.2 Assets (all git-ignored, live only on the training box)

| Asset | Path | Source |
|-------|------|--------|
| Gemma 4 weights | `models/gemma-4-e4b-it` | HF `google/gemma-4-E4B-it`, downloaded local |
| Decomposer dataset | `data/decomposition_trees/path1_train.json` | `build_stage3_dataset.py` (already built) |
| Eval prompts | `scripts/assets/gemma_scene_prompts.txt` | existing starter list |
| Grammar parts vocab | `src/raum/castle_grammar.py EXPANDABLE_PARTS` | existing |

If `path1_train.json` is stale or absent, rebuild it (this is the same command the
Hertz decomposer used):

```powershell
python scripts/build_stage3_dataset.py --params output/layout_opt.params.json `
  --variants 8 --repeat 16 --out data/decomposition_trees/path1_train.json
```

## 1. Gate 0 — few-shot Gemma decomposer, ZERO training (TO BUILD: scripts/gemma_decomposer.py)

The cheapest possible decomposer: prompt the instruction-tuned base with a few
`prompt -> tree` exemplars and parse the JSON it returns. Reuse the load + chat-
template + generate path from `generate_trees_gemma.py` (proven), plus a strict
JSON extractor (balance the first top-level object, same early-stop logic as
`infer_decomposer.py generate_tree`, lines ~168-181).

`scripts/gemma_decomposer.py` defines `class GemmaDecomposer` with the SAME public
method the SGS `Decomposer` exposes:

```python
class GemmaDecomposer:
    def __init__(self, model_path, max_new=1024, temperature=0.1, n_shot=3): ...
    def generate_tree(self, prompt: str) -> dict:   # returns a CompositionNode dict
        # apply_chat_template([system + n_shot exemplars + user prompt]) -> generate
        # -> extract first balanced {...} -> json.loads -> validate against schema
```

- System prompt states the schema and constrains part names to
  `EXPANDABLE_PARTS` (so `_fill_gaussians` can render every leaf).
- `n_shot` exemplars drawn from `path1_train.json` (a castle, a non-castle, one with
  nested children). This is retrieval-free, fixed exemplars.
- Near-greedy (`temperature 0.1`) like the SGS decomposer, JSON wants determinism.

```powershell
python scripts/gemma_decomposer.py --model models/gemma-4-e4b-it `
  --eval scripts/assets/gemma_scene_prompts.txt --out results/gemma_decomp_fewshot.json
```

GATE 0 (measured, not eyeballed): on the eval prompt set, report
(a) **JSON-valid rate** (parses + matches schema), and
(b) **part-vocab rate** (every leaf name in `EXPANDABLE_PARTS`, so it renders), and
(c) a coherence spot-check via the diagnose path (below).
- **PASS (>= ~0.9 valid AND renders coherent castles):** skip §2 entirely, go
  straight to §3 wiring. A strong base with no training beating our tuned decomposer
  is the headline result.
- **PARTIAL (valid but drifting vocab / occasional malformed):** proceed to §2 LoRA
  to lock the format.
- **FAIL (can't hold the schema even few-shot):** kill. Gemma is the wrong tool for
  this output; record and stop.

Coherence check reuses the existing tooling (counts RAW emitted parts, the path1
lesson: instrument the output, not the picture, see [[project_sgs_raum_17]]):

```powershell
python scripts/diagnose_emission.py --trees results/gemma_decomp_fewshot.json
```

## 2. Gate 1 — LoRA fine-tune Gemma on the decomposer dataset (TO BUILD: scripts/train_decomposer_gemma.py)

Only if Gate 0 was PARTIAL. Fine-tune Gemma on the SAME `path1_train.json` the
Hertz/Planck decomposers used, so the comparison is apples-to-apples.

`scripts/train_decomposer_gemma.py` (mirrors `train_decomposer.py`'s CLI where it
makes sense, HF/PEFT internals):

- `--data` (default `data/decomposition_trees/path1_train.json`)
- `--model` (default `models/gemma-4-e4b-it`)
- `--save-dir` (default `checkpoints/gemma_decomposer`)
- `--epochs` (default 3, big base needs few), `--lr` (default 2e-4, LoRA range),
  `--batch-size` (default 1 + grad-accum), `--max-len` (default 512),
  `--lora-r` (default 16), `--lora-alpha` (default 32), `--lora-dropout` (0.05)
- `--val-split` (0.1)

Mechanics:
- Load with `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16,
  device_map="auto")`, wrap with PEFT `LoraConfig` on attention + MLP proj
  modules, `task_type="CAUSAL_LM"`.
- Training string reuses the decomposer format inside Gemma's chat template: user
  turn = `f"DECOMPOSE: {prompt}"`, assistant turn = the compact `tree_json`. Mask
  loss to the assistant turn only (label -100 on the prompt tokens).
- Save the LoRA ADAPTER (small) to `--save-dir`, not merged weights. Record
  `{adapter, args, val_loss}`.

```powershell
python scripts/train_decomposer_gemma.py `
  --data data/decomposition_trees/path1_train.json `
  --model models/gemma-4-e4b-it --save-dir checkpoints/gemma_decomposer --epochs 3
```

GATE 1: JSON-valid + part-vocab rate on the eval set beats Gate 0's few-shot
numbers AND the trees render coherent castles under `diagnose_emission.py`. If LoRA
does not beat few-shot, ship the few-shot decomposer (simpler, no checkpoint) and
note it.

## 3. Wire Gemma into the Raum decomposer app (EDIT: scripts/infer_decomposer.py)

Add an HF backend to the serving path so Gemma is selectable exactly like the SGS
bases. The downstream fill + render pipeline is unchanged (it consumes a tree dict).

- Add `--backend {sgs,hf}` (default `sgs`) and let `--checkpoint` accept a Gemma
  folder path when `--backend hf`. When `hf`, construct `GemmaDecomposer` (few-shot,
  or with `--adapter checkpoints/gemma_decomposer` for the LoRA variant) INSTEAD of
  the `SGSLanguageModel`-based `Decomposer`.
- Both classes expose `generate_tree(prompt) -> dict`, so the FastAPI `/generate`
  handler, `_fill_gaussians`, `snap_layout`, and every appearance control work
  as-is. This is the whole point of the backend abstraction.

```powershell
# few-shot Gemma decomposer, served in Raum's decomposer UI (port 8003)
python scripts/infer_decomposer.py --backend hf --checkpoint models/gemma-4-e4b-it `
  --serve --port 8003 --no-snap
# LoRA-tuned variant
python scripts/infer_decomposer.py --backend hf --checkpoint models/gemma-4-e4b-it `
  --adapter checkpoints/gemma_decomposer --serve --port 8003 --no-snap
```

GATE 2: the Raum decomposer UI renders a coherent castle from Gemma at parity-or-
better vs the Hertz decomposer on the same prompt, snap OFF (learned/emitted
geometry, not grammar constants). Compare side by side against
`checkpoints/hertz_decomposer/best.pt`.

## 4. Register Gemma in the model selector (EDIT: satz/app.py + static/app.js)

Add Gemma to the `MODELS` registry so the selector lists it next to Planck and
Hertz, with a `backend` key the runtime dispatches on.

```python
# satz/app.py MODELS, new entry
"gemma": {
    "label": "Gemma 4 E4B (8B, pretrained base)",
    "backend": "hf",                      # NEW field; existing entries get "sgs"
    "model_path": "models/gemma-4-e4b-it",
    "adapter": None,                      # or "checkpoints/gemma_decomposer"
    "tokenizer": None,                    # HF folder brings its own
    "blobs_dir": None,
    "arch": None,                         # inferred by HF, no infer_arch
},
```

- `SatzRuntime` / `RuntimeManager.load()` branches on `backend`: `"sgs"` keeps the
  current `SGSLanguageModel` + `infer_arch` + SentencePiece path; `"hf"` loads via
  `transformers` and wraps `GemmaDecomposer`. `RuntimeManager` already lazy-loads +
  caches per model name, so hotswap comes for free once the branch exists.
- `static/app.js loadModels()` already populates the `<select>` from `/models`; no
  UI change needed beyond the new entry showing up. Grey the blob panel for Gemma
  (`has_blobs=false`), same as Hertz.

```powershell
python -m satz.app --model gemma       # boot straight into the Gemma decomposer
```

GATE 3: switching the selector to "Gemma 4 E4B" in the running app produces a valid
tree/scene on the next Generate, and switching back to Hertz/Planck still works
(no runtime desync, caches hold).

## Sequencing (gate-and-kill)

| Phase | What | Status / kill-if |
|-------|------|------------------|
| 0.1 | env: transformers>=4.50 + peft in main .venv | prereq |
| 0 | few-shot Gemma decomposer (scripts/gemma_decomposer.py) | **TO BUILD.** KILL if Gemma can't hold the JSON schema even few-shot. PASS (>=~0.9 valid + coherent) -> skip §2. |
| 1 | LoRA fine-tune (train_decomposer_gemma.py) | **TO BUILD, only if §0 PARTIAL.** KILL claim if LoRA does not beat few-shot; then ship few-shot. |
| 2 | HF backend in infer_decomposer.py --serve | **TO BUILD.** Parity-or-better vs Hertz decomposer, snap OFF. |
| 3 | register in satz/app.py MODELS + selector | **TO BUILD.** Hotswap works both directions. |

## Papers / product this feeds

- **Product (Raum 0.8 hotswap, roadmap 10-raum-0-8):** a third selectable decomposer
  backend, and the first NON-`SGSLanguageModel` one, proving the interpreter/
  decomposer/fill hotswap generalizes to arbitrary HF bases.
- **Finding (feeds no new paper yet):** whether a large pretrained base beats a
  purpose-built small model on a saturated structured-emission task. If Gemma wins
  few-shot with zero training, that is a clean data point for the "representation/
  base is the lever" thread ([[project_sgs_path1_outcome]], [[project_sgs_vsp_gate]]).

## Honest scope reminder

This does NOT prove Gemma is a better SGS model, only a better (or cheaper)
DECOMPOSER on the castle task. The task is low-entropy and saturated (path1), so a
Gemma win is expected and modest, its value is a strong, zero-training baseline in
the hotswap and a sanity floor for the custom bases, not a new capability. The real
levers stay representation (VSP) and fill. Keep fill = grammar for shippable renders;
Gemma changes only the tree emitter.
