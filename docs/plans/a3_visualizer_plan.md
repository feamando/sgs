# A3: Interactive SGS Visualizer — Plan (v2)

## Concept

A web app where you type a sentence and watch meaning being "rendered" from word Gaussians — like a real-time 3D splatting demo but for language.

---

## What the User Sees

1. **Type a sentence** → "The warm coffee sat on the table"
2. **3D view** shows Gaussian ellipsoids appearing in semantic space (projected to 3D via PCA)
   - Each word is a colored blob, sized by its covariance
   - Opacity pulses to show salience (α)
   - Colors encode semantic features (sentiment=red/blue, POS=shape)
3. **Rendering animation** plays left to right:
   - Transmittance bar depletes as each word contributes
   - Weight of each word shown as a bar chart
   - "Rendered meaning" vector builds up incrementally
4. **Compare** side-by-side with softmax attention weights
5. **Query viewpoint** slider: move the "camera" through semantic space and watch different words light up

---

## Architecture

### MVP (Tier 1 — 5 days, no backend)

```
GitHub Pages (static)
├── index.html
├── app.js (React + Three.js)
├── data/
│   ├── vocab_gaussians.json    (top 10K GloVe Gaussians: μ, Σ, α)
│   ├── pca_projection.json     (300d → 3d for visualization)
│   └── examples/
│       ├── sentence_001.json   (pre-computed weights, transmittance)
│       └── ...50-100 examples
└── assets/
```

- All inference pre-computed in Python, exported as JSON
- No server, no GPU, no costs
- Users can only explore pre-computed examples (not type custom sentences)
- ~50-100 curated examples covering interesting phenomena (polysemy, ordering effects, composition)

### Full Version (Tier 2 — 12 days, with backend)

```
Vercel (frontend)                    Railway/Fly.io (backend)
├── React + Three.js UI              ├── FastAPI
├── Text input (custom sentences)    ├── PyTorch SGS model (CPU)
├── Real-time 3D rendering           ├── GloVe vocab (in memory)
└── Comparison views                 └── /api/render endpoint
```

- Users type any sentence → backend runs SGS inference → returns Gaussian params + weights
- CPU-only inference (SGS is lightweight — <100ms per sentence on CPU)
- Backend cost: ~$5-10/month on Railway free tier

---

## Implementation Phases

### Phase 1: Data Export (Day 1)

```python
# Export script: export_viz_data.py
1. Load trained SGS-2pass model
2. For top 10K vocab words, export: μ (3d PCA), Σ (3d PCA), α, f_summary
3. For 100 example sentences:
   a. Run SGS forward pass
   b. Export per-word: kernel value K, weight w, transmittance T
   c. Export rendered meaning vector
   d. Run softmax attention for comparison
4. Save all as JSON
```

### Phase 2: 3D Renderer (Days 2-3)

- Three.js scene with instanced ellipsoid meshes
- Each Gaussian = scaled, rotated sphere
- Color mapped from feature vector (first 3 PCA components → RGB)
- Opacity from α value
- Camera orbit controls

### Phase 3: Animation (Day 4)

- Transmittance bar: horizontal bar that depletes left-to-right
- Word weight chart: vertical bars per word, animating as rendering proceeds
- Meaning vector: growing arrow in 3D showing the accumulated output
- Play/pause/step controls

### Phase 4: UI (Day 5 for MVP, Days 5-9 for full)

- Sentence selector (dropdown of 100 examples) → MVP
- Text input field → Full version only
- Split view: SGS rendering (left) vs softmax attention (right)
- Info panel: hover a word to see its Gaussian params
- "About" modal explaining the concept (link to paper)

### Phase 5: Backend (Days 6-7, full version only)

```python
# FastAPI endpoint
@app.post("/api/render")
async def render_sentence(text: str):
    tokens = tokenize(text, word2idx)
    mu, log_var, alpha, features = vocab.get_params(tokens)
    # Run SGS rendering
    K = gaussian_kernel_diag(query, mu, log_var, tau)
    meaning, weights = render(features, alpha, K)
    # Project to 3D for visualization
    mu_3d = pca_3d.transform(mu)
    return {
        "words": [...],
        "gaussians": {"mu_3d": ..., "sigma_3d": ..., "alpha": ..., "color": ...},
        "rendering": {"weights": ..., "transmittance": ..., "meaning_3d": ...},
    }
```

### Phase 6: Polish + Deploy (Days 8-10)

- Mobile responsive
- Performance optimization (instanced rendering for many Gaussians)
- Share button (link with encoded sentence)
- Open Graph meta tags for social sharing previews
- README with screenshots

---

## Testing

| What | How | When |
|---|---|---|
| Data export correctness | Compare JSON values to Python model output | Phase 1 |
| 3D rendering accuracy | Visual check: do ellipsoids match Gaussian params? | Phase 2 |
| Animation correctness | Step through rendering: weights match pre-computed values? | Phase 3 |
| Cross-browser | Chrome, Firefox, Safari, mobile Safari | Phase 6 |
| Performance | 60fps with 50 Gaussians on mid-range laptop | Phase 6 |
| Backend latency | <200ms response for any sentence | Phase 5 |
| Backend load | Locust load test: 10 concurrent users | Phase 5 |

---

## Monitoring & Telemetry

### MVP (Static Site)

- **Plausible Analytics** (privacy-friendly, no cookies): page views, time on page, example sentence clicks
- No server = no backend monitoring needed
- GitHub Issues for bug reports (link in footer)

### Full Version

| Layer | Tool | What |
|---|---|---|
| Frontend | Plausible Analytics | Visits, unique users, popular sentences |
| Frontend | Sentry (JS) | JavaScript errors, rendering crashes, browser/device info |
| Backend | Sentry (Python) | Exceptions, slow requests |
| Backend | Built-in `/health` endpoint | Uptime monitoring via UptimeRobot (free) |
| Backend | Request logging | Log: sentence text, response time, token count (no PII) |

### User Feedback

- **"Report Issue" button** in bottom-right corner → opens GitHub Issue with pre-filled template:
  ```
  Sentence: [auto-filled]
  Browser: [auto-detected]
  What happened: [user fills in]
  Expected: [user fills in]
  Screenshot: [optional upload]
  ```
- **"This looks wrong" per-word flag** — user can click a word's Gaussian and report "this weight seems off" → logs to a feedback CSV

---

## Rollout

| Phase | Audience | Goal |
|---|---|---|
| **Alpha** | Self + 2-3 friends | Catch obvious bugs |
| **Beta** | Post in 1 ML Discord/Slack | 50 users, collect feedback |
| **Launch** | Twitter/X post + HN submission | Virality attempt |
| **Sustain** | Link from paper, README, conference poster | Long-tail traffic |

### Launch Checklist

- [ ] 50+ curated example sentences covering interesting cases
- [ ] Mobile works (3D renders on phone)
- [ ] Open Graph image (screenshot of the visualization)
- [ ] 30-second screen recording GIF for Twitter
- [ ] "About" page links to paper and GitHub repo
- [ ] No broken links, no console errors
- [ ] Load time < 3 seconds on 4G connection

---

## Cost

| Tier | Monthly Cost |
|---|---|
| MVP (GitHub Pages) | $0 |
| Full (Vercel + Railway) | $5-15/month |
| Domain name (optional) | $12/year |
