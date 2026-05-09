# Raum 1.2 — quality pass: more data, more epochs, OOV policy

*Status: open. Created 2026-05-09. Follows Raum 1.1 (§3 in SETUP).*

Raum 1.1 proved the frozen-encoder + blob-library architecture works
(100% on procedural scenes, functional on Objaverse blobs). But demo
quality is low: the bridge often misclassifies user-typed words because
(a) training was short (30 epochs, 5k samples), and (b) words not in
the blob vocabulary crash or produce garbage predictions.

Raum 1.2 is a quality pass, not an architecture change:

1. **More training data** (15k-50k samples)
2. **More epochs** (100+)
3. **OOV policy** (cosine-NN fallback for unknown words)

No new model components. Same bridge, same encoder, same blob library.
Ships before Satz 0.1.

---

## 1. Training scale-up

| parameter | Raum 1.1 | Raum 1.2 |
|---|---|---|
| `--n-train` | 5000 | 50000 |
| `--n-val` | 500 | 2000 |
| `--n-test` | 500 | 2000 |
| `--epochs` | 30 | 100 |
| `--n-objects-max` | 5 | 5 |
| `--with-relation-head` | yes | yes |
| estimated wall-clock | ~10 min | ~2-4 hours |

The bridge is 5.15M params. At batch 32, 50k scenes generate ~1500
steps/epoch. 100 epochs = 150k steps. At 4090 speed this is 2-4 hours.

---

## 2. OOV (out-of-vocabulary) policy

When the user types a word not in the blob library (e.g. "sphere",
"dog", "house"), the bridge's blob_id head picks whichever class has
the nearest embedding. This is noisy.

Fix: at inference time (demo), after the bridge predicts a blob_id,
check cosine similarity between the input word's GloVe embedding and
the predicted blob class's GloVe embedding. If cosine < threshold
(default 0.3), mark as unresolved instead of stamping a bad match.

Additionally, offer a nearest-neighbor fallback: find the blob class
whose GloVe embedding is closest to the input word. If cosine > 0.3,
use that class. This handles "couch" -> "sofa", "automobile" -> "car",
etc.

Implementation:
- `src/raum/oov.py` (new): `OOVPolicy` class with `resolve(word, predicted_blob_id) -> blob_id | None`
- Wired into `assemble_scene` and the demo's `generate()` path
- No training change needed

---

## 3. Steps

| step | what | time |
|---|---|---|
| 3.1 | Retrain with --n-train 50000 --epochs 100 | ~2-4 hours |
| 3.2 | Implement OOV policy (src/raum/oov.py) | ~30 min code |
| 3.3 | Wire OOV into demo + analyzer | ~30 min |
| 3.4 | Analyze + gate | ~10 min |
| 3.5 | Publish | ~5 min |

---

## 4. Gates

- blob_id accuracy > 85% on 5-object held-out scenes
- direction accuracy > 90%
- relation accuracy > 85% (with --with-relation-head)
- Demo: typing known blob words produces correct objects > 80% of the time
- Demo: OOV words either resolve via NN to a sensible blob OR show "unresolved"

---

## 5. Commands

### 3.1 Retrain

```powershell
python scripts\train_raum_bridge.py --glove data\glove.6B.300d.txt `
  --n-objects-max 5 `
  --encoder-checkpoint checkpoints\planck13\best.pt --freeze-encoder `
  --blobs-dir data\blobs `
  --with-relation-head `
  --n-train 50000 --n-val 2000 --n-test 2000 --epochs 100 `
  --d-model 256 --n-layers 6 --n-heads 8 `
  --save-dir checkpoints\raum_12
```

### 3.4 Analyze

```powershell
python scripts\analyze_raum_bridge.py `
  --checkpoint checkpoints\raum_12\best.pt `
  --glove data\glove.6B.300d.txt `
  --encoder-checkpoint checkpoints\planck13\best.pt `
  --n-objects-max 5
```

### 3.5 Demo

```powershell
python -m demo.app `
  --checkpoint checkpoints\raum_12\best.pt `
  --glove data\glove.6B.300d.txt `
  --encoder-checkpoint checkpoints\planck13\best.pt `
  --blobs-dir data\blobs
```
