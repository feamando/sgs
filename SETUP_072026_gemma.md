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
| Decomposer dataset | `data/decomposition_trees/path1_train.json` | `build_stage3_dataset.py` (BUILT 2026-07-22: 160 castle + 400 mix = 560 records) |
| Eval prompts | `scripts/assets/gemma_scene_prompts.txt` | existing starter list |
| Grammar parts vocab | `src/raum/castle_grammar.py EXPANDABLE_PARTS` | existing |

If `path1_train.json` is stale or absent, rebuild it (this is the same command the
Hertz decomposer used):

```powershell
python scripts/build_stage3_dataset.py --params output/layout_opt.params.json `
  --variants 8 --repeat 16 --out data/decomposition_trees/path1_train.json
```

## 1. Gate 0 — few-shot Gemma decomposer, ZERO training (BUILT: scripts/gemma_decomposer.py)

The cheapest possible decomposer: prompt the instruction-tuned base with a few
`prompt -> tree` exemplars and parse the JSON it returns. Reuses the load + chat-
template + generate path from `generate_trees_gemma.py` (proven), and imports that
module's `parse_tree` + `validate_skeleton` directly, so few-shot output stays
format-consistent with the data generator (same JSON extractor, same part-vocab
drop rule).

`scripts/gemma_decomposer.py` (BUILT 2026-07-22, pure-python helpers smoke-tested)
defines `class GemmaDecomposer` with the SAME public method the SGS `Decomposer`
exposes:

```python
class GemmaDecomposer:
    def __init__(self, model_path, exemplars_path=None, n_shot=3,
                 max_new=1024, temperature=0.1): ...
    def generate_tree(self, prompt: str) -> dict | None:  # CompositionNode dict, or None if unparseable
        # apply_chat_template([system + n_shot exemplars + user prompt]) -> generate
        # -> parse_tree (first balanced {...}) -> validate_skeleton -> dict
    def generate_tree_verbose(self, prompt: str) -> dict:  # + parsed/n_leaves/n_unknown for the gate
```

- System prompt states the schema and constrains part names to
  `EXPANDABLE_PARTS` (so `_fill_gaussians` can render every leaf). Reuses the
  data generator's rules verbatim.
- `n_shot` exemplars auto-drawn from `path1_train.json` via `load_exemplars`
  (castle-first spread: one castle, one non-castle, then fill). Retrieval-free,
  fixed exemplars.
- Near-greedy (`temperature 0.1`) like the SGS decomposer, JSON wants determinism.
- Loss of `diagnose_emission.py`: it needs a checkpoint (it GENERATES trees from a
  model), so it can't read a trees file. The recall/structure counts it would give
  are computed INLINE here (`n_leaves`, `n_unknown` per prompt); `--dump-tree`
  writes the first valid tree for a model-free `infer_decomposer.py --scene-file`
  render.

```powershell
python scripts/gemma_decomposer.py --model models/gemma-4-e4b-it `
  --eval scripts/assets/gemma_scene_prompts.txt --out results/gemma_decomp_fewshot.json `
  --dump-tree output/gemma_scene_0.json
```

GATE 0 (measured, not eyeballed): the script reports, over the eval prompt set,
(a) **JSON-valid rate** (parses + matches the shallow-skeleton schema), and
(b) **part-vocab rate** (fraction of emitted leaves whose name is in
`EXPANDABLE_PARTS`, so the fill stage can render them), plus per-prompt leaf +
unknown-leaf counts (the RECALL/STRUCTURE signal `diagnose_emission.py` gives, but
computed inline from the emitted trees, no model reload).
- **PASS (>= ~0.9 valid AND renders coherent castles):** skip §2 entirely, go
  straight to §3 wiring. A strong base with no training beating our tuned decomposer
  is the headline result.
- **PARTIAL (valid but drifting vocab / occasional malformed):** proceed to §2 LoRA
  to lock the format.
- **FAIL (can't hold the schema even few-shot):** kill. Gemma is the wrong tool for
  this output; record and stop.

Visual coherence spot-check (path1 lesson: instrument the output, not just eyeball
a render, see [[project_sgs_raum_17]]). `diagnose_emission.py` needs a checkpoint +
tokenizer (it GENERATES trees from a model) so it does NOT read a pre-made trees
file; the recall/structure counts are already in the summary above. To eyeball one
scene, `--dump-tree` writes the first valid tree, then render it model-free:

```powershell
python scripts/infer_decomposer.py --scene-file output/gemma_scene_0.json --no-snap `
  --serve --port 8003
```

### GATE 0 RESULT — PASS (run 2026-07-27 on the 4090)

`results/gemma_decomp_fewshot.json`, 20 eval prompts, 3-shot, temp 0.1:
- **JSON-valid rate: 100%** (20/20). Gemma never broke the schema.
- **part-vocab rate: 97.9%** (92/94 leaves renderable).
- Rich prompts decompose well: "fortress with four towers" -> 9 parts,
  "ruined castle on a hill" -> 10, "hilltop fortress with trees" -> 14, all 0-unknown.

DECISION: PASS clears the bar -> **skip §2 LoRA**, wire the HF backend (§3).

ONE recall soft-spot (NOT a format/capacity failure): the 2 dropped leaves are both
on TERSE prompts that name only the whole object -- "a castle on a hill" emitted
`[hill, "castle"]` and "a small fort on a hill" emitted a monolithic `"fort"`; both
non-renderable, dropped -> only a hill renders. This is the **Raum 1.7 lesson**
verbatim ([[project_sgs_raum_17]]): terse default prompts under-recall; concrete
sub-part phrasing emits everything. FIX (decided): enrich the decomposer UI default
prompt to "a stone castle on a green hill" (the phrasing that worked for Hertz),
done in §3. Left as an open item: optionally add a castle->towers+walls+keep
few-shot exemplar so "castle"/"fort" never emit monolithic regardless of phrasing.

## 2. Gate 1 — LoRA fine-tune Gemma on the decomposer dataset (SKIPPED: Gate 0 passed)

**SKIPPED 2026-07-27** — Gate 0 passed at 100% valid / 97.9% vocab with zero
training, so per the gate-and-kill rule this phase is not built. Kept below as the
recorded fallback: if the terse-prompt recall soft-spot proves to need more than a
UI prompt change (i.e. the monolithic-"castle" emission shows up on real usage), the
cheap fix is a castle-decomposition few-shot exemplar; LoRA (this section) is the
heavier fallback if few-shot exemplars can't lock it. `train_decomposer_gemma.py`
is TO BUILD only if that happens. The `--adapter` path is already wired in §3.

Fine-tune Gemma on the SAME `path1_train.json` the
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

## 3. Wire Gemma into the Raum decomposer app (DONE 2026-07-27: scripts/infer_decomposer.py)

HF backend added to the serving path so Gemma is a launch-time-selectable decomposer
exactly like the SGS bases (this is Raum's decomposer-selector: interpreter/decomposer
are heavy loads chosen at launch via `--checkpoint`, only FILL hotswaps per-request).
The downstream fill + render pipeline is unchanged (it consumes a tree dict).

DONE:
- `--backend {sgs,hf}` (default `sgs`) added; with `--backend hf`, `--checkpoint`
  is the Gemma model folder and `--adapter` optionally loads a LoRA dir. When `hf`,
  `main()` constructs `GemmaDecomposer` instead of the `SGSLanguageModel`-based
  `Decomposer`.
- `GemmaDecomposer.generate_tree(prompt, max_new, temperature, top_k, retries)` is
  signature-compatible with the SGS `Decomposer.generate_tree`, and exposes
  `.last_raw` / `.scan_library` / `.vocab_size` so the FastAPI `/generate` handler,
  `_fill_gaussians`, `snap_layout`, and appearance controls work unchanged.
- The two parse-failure debug blocks (which poke `.sp`/`.model`, SGS-only) are
  guarded by `backend`; the HF branch dumps `.last_raw` instead.
- Recall fix (Gate 0 soft-spot): the decomposer UI default prompt is enriched to
  "a stone castle on a green hill" (was terse "a castle on a hill"), so terse
  under-recall doesn't hit the default scene. Applies to ALL backends.

```powershell
# few-shot Gemma decomposer served in the Raum decomposer UI (port 8003)
python scripts/infer_decomposer.py --backend hf --checkpoint models/gemma-4-e4b-it `
  --serve --port 8003 --no-snap
# LoRA variant (only if §2 ever gets built)
python scripts/infer_decomposer.py --backend hf --checkpoint models/gemma-4-e4b-it `
  --adapter checkpoints/gemma_decomposer --serve --port 8003 --no-snap
```

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
| 0.a | decomposer dataset (build_stage3_dataset.py) | **DONE 2026-07-22:** path1_train.json, 160 castle + 400 mix = 560 records. |
| 0 (§1) | few-shot Gemma decomposer (scripts/gemma_decomposer.py) | **PASS 2026-07-27: 100% valid / 97.9% vocab**, 20 prompts, zero training. Terse-prompt recall soft-spot (fixed via UI default prompt). |
| 1 (§2) | LoRA fine-tune (train_decomposer_gemma.py) | **SKIPPED** — Gate 0 passed, no training needed. Fallback if the recall soft-spot resists few-shot exemplars. |
| 2 (§3) | HF backend in infer_decomposer.py --serve | **DONE 2026-07-27:** --backend hf + --adapter; GemmaDecomposer signature-compatible; parse-failure debug guarded by backend; UI default prompt enriched. Live render vs Hertz decomposer = next RUN on the box. |
| 3 (§4) | register in satz/app.py MODELS + selector | **TO BUILD** (optional): Satz chat selector. Raum's decomposer selector is already covered by §3's --backend. |

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
