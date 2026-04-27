# Radiance Labs Roadmap Visualizer

A zero-build, local-storage dashboard that reads `roadmap.md` (at repo
root) and renders Model / Product swimlanes with status cards.

## Run

From the **repo root**, start any static file server:

```powershell
# Windows
python -m http.server 8000
```

```bash
# macOS / Linux
python3 -m http.server 8000
```

Then open:

```
http://localhost:8000/pm/
```

> Serve from the repo root, not from `pm/`. The dashboard fetches
> `../roadmap.md` relative to this folder.

## How it works

- `index.html` is a single page. No framework, no build.
- `app.js` fetches `../roadmap.md`, finds the `## Swimlanes` and
  `## Entries` pipe-tables by their headings, and renders lanes.
- `logo.js` is the animated Gaussian-splat logo (shared with the Raum
  demo).
- Status colours: grey = `open`, amber = `in progress`, green = `done`.
- Refresh the page after editing `roadmap.md` to pick up changes.

## Adding a new version

Edit `roadmap.md`:

```
| 1-planck-1-3 | Planck 1.3 | model | open | 2026-05-05 | Short note |
```

- `id` is `<swimlane-id>-<slug>` in kebab-case. Dots in version numbers
  become dashes (`1.3` → `1-3`).
- `type` must be `model` or `product` (matches the lane's `kind`).
- `status` is one of `open`, `in progress`, `done`.

Every new entry must also be reflected in `SETUP.md` and vice-versa.
See the "Sync rules" block at the top of `roadmap.md`.
