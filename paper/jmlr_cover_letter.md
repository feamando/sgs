# JMLR Cover Letter

> **Paper:** *On the Expressiveness of Alpha-Compositing: A Strict Superset of Softmax Attention*
> **Author:** Nikita Gorshkov
> **Status:** DRAFT — only two open items remain (preprint/arXiv status in §1;
> filling reviewer affiliations/emails in §5). Everything else is complete.
> **Reference:** JMLR author guidelines, https://www.jmlr.org/author-info.html (verified 2026-06-01).
>
> JMLR requires the cover letter to be a PDF or plain-text file accompanying the
> manuscript PDF. The six numbered items below are the JMLR-mandated contents.
> This stub also tracks the manuscript-side requirements (style file, abstract ≤200
> words, 5 keywords, running title ≤50 chars) that gate acceptance.

---

To the Editors of the Journal of Machine Learning Research,

I am submitting the manuscript *On the Expressiveness of Alpha-Compositing: A
Strict Superset of Softmax Attention* for consideration for publication in JMLR.

**Summary.** The paper proves that the set of weight vectors achievable by
alpha-compositing (the rendering equation of volume rendering and 3D Gaussian
Splatting) strictly contains those achievable by softmax attention: every
softmax weighting can be exactly reproduced by alpha-compositing via an explicit
constructive map, but not conversely. Alpha-compositing additionally realizes
exact zero weights (hard sparsity) and sub-unity weight sums (residual
capacity). The complete result is formally verified in Lean 4 with Mathlib
(v4.28.0), with zero `sorry` statements and only the standard axioms. The
contribution sits at the intersection of attention-mechanism theory and computer
graphics, and adds a third characterization to the softmax-attention literature
alongside the Hopfield-network (Ramsauer et al., 2021) and kernel (Katharopoulos
et al., 2020; Choromanski et al., 2021) views.

---

## 1. Overlapping or prior publications

- This manuscript has **not** been published in, or simultaneously submitted to,
  any other journal or conference.
- `[TODO: confirm preprint status]` A preprint version `[is / is not]` posted at
  `[arXiv ID or "n/a"]`. (arXiv preprints are permitted by JMLR.)
- Related but distinct work by the author: *Semantic Gaussian Splatting:
  Alpha-Compositing as a Composition Mechanism for Language* (Gorshkov, 2026,
  preprint), cited as the empirical companion in §4.2. That paper reports
  empirical NLP results; **this** paper is the formal/theoretical result and
  shares no overlapping theorems, figures, or text beyond standard background.
  `[TODO: confirm the companion is not under review at a venue that would create
  a dual-submission conflict.]`

## 2. Co-author awareness and consent

- Single-author submission (Nikita Gorshkov). No co-authors to notify.

## 3. Conflicts of interest

- The author declares **no conflicts of interest** with the suggested action
  editors or reviewers (no family/close-friend, advisor/advisee, or
  collaboration within the past three years).
- The author declares **no funding and no competing interests** for the prior
  36 months.

## 4. Suggested Action Editors (3–5, from the current JMLR AE list)

> Drawn from the current JMLR Action Editor list (verified against the editorial
> board page, 2026-06-01). No AE lists "attention/transformers" or "formal
> verification" explicitly, so these are chosen from learning theory,
> theoretical deep learning, and kernel methods — the paper's actual lanes
> (expressiveness theory + the softmax-as-kernel connection in §4.3). Confirm
> each is still a current AE and conflict-free before submitting.

1. **Mehryar Mohri** — Learning theory (all aspects); strong fit for an
   expressiveness/set-inclusion result.
2. **Joan Bruna** — Deep learning theory; fit for the architectural-inductive-bias
   framing (§4.1–4.2).
3. **Gabor Lugosi** — Statistical learning theory, online prediction.
4. **Bharath Sriperumbudur** — Kernel methods, regularization; fit for the
   softmax-as-kernel / linear-attention connection (§4.3).
5. **Maxim Raginsky** — Theory of deep learning, statistical learning (optional).

## 5. Suggested Reviewers (3–5, conflict-free)

> Suggested from the paper's own citation neighborhood and adjacent expertise.
> The author has no collaboration with any of these. Emails below are
> best-effort from public pages (verified 2026-06-01 where marked ✓); confirm
> each before submission.

1. **Angelos Katharopoulos** — Apple Machine Learning Research. Author of
   "Transformers are RNNs" (linear attention); fit for the weighted-aggregation
   hierarchy (§4.3). Email: `a_katharopoulos@apple.com` ✓
2. **Krzysztof Choromanski** — Google DeepMind (New York) and Columbia
   University. Author of Performers / the kernel view of attention; fit for §4.3.
   `[email: public DeepMind/Columbia address — confirm]`
3. **Sepp Hochreiter** — Institute for Machine Learning, JKU Linz. Senior author
   of "Hopfield Networks is All You Need"; directly relevant to the
   attention-as-aggregation framing (§4.3). `[email: @ml.jku.at — confirm]`
4. **George Drettakis** — Research Director, Inria (GraphDeco, Sophia Antipolis).
   Co-author of 3D Gaussian Splatting; fit for the alpha-compositing /
   rendering-equation side (§4.4). (optional) `[email: @inria.fr — confirm]`
5. *(Optional fifth)* a Lean 4 / formal-verification researcher to vet the
   mechanized proof, e.g. from the Lean FRO or a Mathlib maintainer.
   `[name + email TODO, or omit — four suffices]`

> Note: emails are not strictly required by JMLR for reviewer suggestions
> (name + affiliation + area is enough for the editor to identify the person),
> so an unconfirmed email can be omitted rather than guessed.

## 6. Keywords

attention mechanisms; alpha-compositing; volume rendering; expressiveness;
formal verification (Lean 4)

*(Matches the manuscript's five keywords.)*

---

Thank you for considering this submission.

Sincerely,

Nikita Gorshkov (corresponding author)
Senefelderstr. 29a, 10437 Berlin, Germany
ngorshkov@protom.me

---

## Pre-submission checklist (manuscript side — not part of the letter text)

- [ ] Manuscript compiled in the **JMLR LaTeX style** (papers not in JMLR style
      are rejected without review). For the JMLR *journal*, that is
      `\documentclass[twoside,11pt]{article}` + `\usepackage{jmlr2e}` (the
      `jmlr2e.sty` package, downloadable from jmlr.org/format). Do NOT use the
      PMLR `jmlr.cls` / `\documentclass[wcp]{jmlr}` (that is for proceedings).
      **Current state:** the LaTeX source `paper/softmax_subset_alpha_compositing.tex`
      exists and is Overleaf-ready (`overleaf_paper.zip`), but it is plain
      `article`, **not yet `jmlr2e`**. Before final submission, add `jmlr2e.sty`
      and switch the preamble. (Title + author block already match the cover
      letter.) `theorem_paper.md` is the Markdown working copy; the `.tex` is the
      submission artifact.
- [x] Cover letter as compilable LaTeX: `paper/jmlr_cover_letter.tex`
      (`overleaf_cover_letter.zip`). Markdown `jmlr_cover_letter.md` is the
      source of record; keep both in sync.
- [ ] PDF only; archive multiple files as tar/zip; total **< 5 MB**.
- [ ] Title page: corresponding author name + **postal address** + email.
- [ ] **Running title ≤ 50 characters.** Suggested: `Alpha-Compositing ⊋ Softmax Attention` (37 chars). OK.
- [ ] **Abstract ≤ 200 words.** Current abstract is 132 words. OK (recount after any edits).
- [ ] **Five keywords** present (see §6).
- [ ] Page count ≤ 35 (incl. appendices) for normal handling; >50 needs
      justification here and risks desk rejection. Current draft is short, so OK.
- [ ] Lean source linked and reproducible: `docs/proofs/lean/claim_3_5_softmax_subset_alpha.lean`.
      `[TODO: include a release form if the Lean code is submitted as an online appendix.]`
- [ ] Notify JMLR of any prior publication at submission time (see §1).
- [ ] Register / log in to the JMLR submission system and upload via
      "submit manuscript."
