# Physical Gaussians — Overleaf project

Third SGS paper. Upload `physical_gaussians_overleaf.zip` to Overleaf
(New Project → Upload Project).

## Files
- `physical_gaussians.tex` — main paper (JMLR `article` format)
- `jmlr2e.sty` — JMLR style file (must sit beside the .tex)

## Compile
- Engine: **pdfLaTeX** (default in Overleaf)
- The bibliography is a hand-written `thebibliography` block, so no separate
  BibTeX pass is needed; one or two pdfLaTeX runs resolve all refs.

## Source provenance
- Prose: `papers/physical-gaussians/physical_gaussians.md`
- Formal claims: `docs/proofs/physical_gaussians_math.md`
- Lean appendix proofs: authoritative `docs/proofs/results/{P1,P5}_dir/lean_aristotle/`
  (both verified 0 `sorry`, Lean 4 v4.28.0)

## Before external submission
- Confirm venues/years for the 2024–2026 related-work citations against
  published versions (several come from the literature review).
- The author block matches the JMLR theorem-paper submission; adjust if the
  target venue differs.
