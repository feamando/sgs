# A3: Interactive SGS Visualizer — Plan

## Concept

A web app where you type a sentence and watch meaning being "rendered" from word Gaussians — like a real-time 3D splatting demo but for language.

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

## Tech Stack

- **Frontend:** Three.js for 3D Gaussian rendering, React for UI
- **Backend:** FastAPI serving PyTorch SGS model
- **Model:** Pre-trained SGS-2pass with GloVe (the model we already have)
- **Deployment:** Vercel (frontend) + a small GPU instance or CPU-only (inference is fast)

## Implementation Steps

| Phase | What | Effort |
|---|---|---|
| **1** | Export pre-trained SGS weights to JSON | 1 day |
| **2** | Three.js Gaussian ellipsoid renderer | 3 days |
| **3** | Transmittance animation + weight bar chart | 2 days |
| **4** | FastAPI backend for SGS inference | 1 day |
| **5** | React UI: text input, comparison view, query slider | 3 days |
| **6** | Polish: mobile, performance, examples | 2 days |
| **Total** | | **~12 days** |

## MVP (Minimal Shareable Version)

Skip the backend entirely. Pre-compute SGS weights for 50-100 example sentences. Pure frontend, no server needed. Deployable on GitHub Pages.

- Effort: ~5 days
- Shareable: yes, link goes viral potential in ML Twitter

## Viral Hooks

- "What if we rendered sentences like 3D scenes?"
- Side-by-side: SGS rendering vs transformer attention for the same sentence
- Interactive: users can type their own sentences and see the Gaussians
- Educational: best visualization of alpha-compositing ever made
