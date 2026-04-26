# Raum demo v0

Local web app: prompt in, 3D Gaussian cloud out. Runs entirely on your
laptop, nothing is hosted.

## What's in the box

- **Backend**: FastAPI (`demo/app.py`). Loads a trained `RaumBridge`
  checkpoint and a frozen SGS semantic vocabulary (initialised from
  GloVe), exposes `POST /generate` that returns the Gaussian parameters
  for a prompt.
- **Frontend**: static HTML/CSS/JS in `demo/static/`. Three.js renders
  each Gaussian as a soft point sprite (good enough for a few thousand
  splats at 60fps). No build step, uses the unpkg CDN for Three.
- **UI**: Radiance Labs dark palette, logo top-left, viewer fills the
  top, chat bar at the bottom. Drag to orbit, scroll to zoom.

## Install (once)

From the repo root, in your existing Raum Python environment:

```powershell
pip install -r demo\requirements.txt
```

(Assumes `torch`, `numpy`, and the project itself are already
installed, same env you use to train.)

## Run (Windows)

```powershell
python -m demo.app `
  --checkpoint checkpoints\raum_c_pos\best.pt `
  --glove data\glove.6B.300d.txt
```

Open <http://localhost:8000> in a browser.

The first request after boot will take a couple of seconds (GloVe load
+ vocab build happens at startup, but torch cold-starts its CUDA
context on first generate). Subsequent prompts are snappy.

### Flags

| flag | default | purpose |
| --- | --- | --- |
| `--checkpoint` | required | `RaumBridge` `best.pt` |
| `--glove` | required | `glove.6B.300d.txt` |
| `--d-s` | 64 | must match training |
| `--K` | 32 | must match training |
| `--host` | 127.0.0.1 | bind address |
| `--port` | 8000 | |
| `--max-tokens` | 32 | prompt length cap |

## Prompting

The vocabulary the bridge was trained on is small and specific (see
`src/raum/vocab.py`):

- **objects**: sphere, cube, cylinder, cone, plane, torus
- **colours**: red, blue, green, yellow, white, black, orange, purple
- **sizes**: tiny, small, medium, large, huge
- **relations**: above, below, left, right, behind, on, beside

Prompts that stay close to the training distribution work best:

- `a red sphere above a blue cube`
- `a yellow cone left a green cylinder`
- `a huge white sphere behind a small orange cube`
- `a purple torus on a black plane`

Out-of-vocabulary words are mapped to `<unk>` and tend to drag the
scene toward the origin, but the demo still runs.

## Architecture

```
browser (Three.js)  <-- http://localhost:8000 --> FastAPI (demo/app.py)
       |                                               |
       |  POST /generate { prompt }                    |
       |  <-- { splats: {...}, words: [...] }          v
                                                 RaumBridge forward
                                                  (coarse_means → cloud)
```

The backend loads the bridge + vocab once at startup and reuses them
for every request. Requests are CPU-bound through tokenisation and
GPU-bound for the forward pass (auto-selects CUDA if available).

## Scope (v0 is deliberately small)

Shipped:
- Single-prompt generation
- Single viewer
- Default scene on page load

Deferred:
- Editing, undo, persistence
- Multi-scene comparison
- Real 3DGS tile-based splatting (current renderer is sprite-based;
  fine up to a few thousand points, will need upgrading if the bridge
  grows much denser)
- Electron wrapper for a desktop binary
- Auth, deployment, any form of hosting

## Troubleshooting

**`ModuleNotFoundError: src`**
You launched from the wrong directory. Run `python -m demo.app` from
the repo root (not from inside `demo/`).

**`RuntimeError: size mismatch` on load_state_dict**
`--d-s` or `--K` don't match the checkpoint. They must be identical
to the values passed to `train_raum_bridge.py`.

**Blank viewer, no errors**
Check the browser console and the server log. The default prompt
should auto-submit on page load; if it fails the header meta text
will say `error: ...`.
