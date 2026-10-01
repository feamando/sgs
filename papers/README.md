# Papers

One folder per paper. Each folder holds the manuscript (`.md` and/or `.tex`), any cover letters, and the Overleaf bundles (`.zip`) used to build PDFs.

| Folder | Paper | Status | Main file |
|--------|-------|--------|-----------|
| [sgs-core](sgs-core/) | Semantic Gaussian Splatting: Alpha-Compositing as a Composition Mechanism for Language | Working paper (v3), plus two orthogonal challenges | `semantic_gaussian_splatting.md` |
| [alpha-compositing-theorem](alpha-compositing-theorem/) | On the Expressiveness of Alpha-Compositing: A Strict Superset of Softmax Attention | Submitted to JMLR 2026-06-01 | `softmax_subset_alpha_compositing.tex` |
| [vsp-negative-result](vsp-negative-result/) | Grounded Token Bundles Separate Word Senses but Do Not Help a Language Model (VSP negative result) | JMLR draft + cover letter | `vsp_negative_result.tex` |
| [physical-gaussians](physical-gaussians/) | Physical Gaussians: Extending the Semantic Gaussian Splatting Primitive with Material Semantics | Whitepaper + literature review + Overleaf draft | `physical_gaussians.md` |
| [recursive-sgs-decomposition](recursive-sgs-decomposition/) | Recursive Semantic-to-Geometric Decomposition via Gaussian Splatting (RSGD) | Draft | `recursive_sgs_decomposition.tex` |
| [training-acceleration](training-acceleration/) | Training Acceleration for Semantic Gaussian Splatting | Research (v1, v2) | `sgs_training_acceleration_v2.md` |
| [hierarchical-sgs](hierarchical-sgs/) | Orthogonal challenge on the H-SGS whitepaper (`docs/whitepaper/hierarchical_sgs.md`) | Review | `orthogonal_challenge_hsgs.md` |
| [raum](raum/) | Raum concept article, scene fidelity, Raum 1.3 literature review, GS scan datasets survey | Articles / reviews | `raum_concept_article.md` |

## Notes

- **alpha-compositing-theorem:** `overleaf_paper.zip` + `overleaf_cover_letter.zip` are the bundles used for the 2026-06-01 submission. `softmax_subset_alpha_compositing_jmlr.zip` and the current `.tex` include the 2026-07-31 edits.
- **recursive-sgs-decomposition:** `rsgd_paper_overleaf.zip` is the LaTeX bundle (`main.tex`); `rsgd_paper_md.zip` only wraps the Markdown draft.
- `jmlr2e.sty` is copied into each LaTeX paper folder so every folder compiles on its own.
