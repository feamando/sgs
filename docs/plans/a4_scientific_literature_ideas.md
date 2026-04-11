# A4-SciLit: SGS for Scientific Literature Triage

## The Opportunity

New research fields (AI safety, CRISPR therapeutics, quantum ML, synthetic biology) produce papers faster than taxonomies can be built. Systematic reviews cost $100K-500K and take 6-18 months. The bottleneck: **semantic matching of papers in domains where no labeled similarity data exists.** SGS's zero-shot composition advantage directly applies.

---

## Concrete Product Ideas

### Idea 1: "EmergingLit" — Zero-Shot Paper Matching for New Fields

**What:** Given a query paper abstract (e.g., on AI safety), find the most relevant papers across all of arXiv — even in subfields that didn't exist when models were trained.

**How:**
1. Represent each paper abstract as an SGS-rendered meaning vector
2. For a query abstract, render and match via Gaussian kernel similarity
3. The covariance of each paper's Gaussian captures its "scope" — survey papers have large Σ, narrow technical papers have small Σ

**Data (all public):**
- arXiv API: ~2.5M paper abstracts with metadata
- Semantic Scholar API: citations, paper embeddings for comparison
- For evaluation: RELISH dataset (biomedical paper relevance), SciDocs benchmark

**Evaluation:**
- Hold out papers from 2025-2026 (unseen fields/topics)
- Query with recent papers, evaluate retrieval quality
- Compare SGS vs SPECTER2 vs SciNCL vs SBERT zero-shot

**Why SGS wins here:** SPECTER2 and SciNCL train on citation graphs — they can only match papers that are connected to the existing citation network. Papers in genuinely new fields have no citations yet. SGS composes meaning from the abstract text alone, without citation signals.

**PoC effort:** 3-5 days. arXiv data freely accessible via API.

---

### Idea 2: "Cross-Field Bridge Finder" — Discover Hidden Connections

**What:** Find papers in Field A that are semantically similar to papers in Field B, even when the two fields don't cite each other.

**Example:** A quantum computing paper on "error correction via surface codes" might be semantically close to a coding theory paper on "LDPC codes for noisy channels" — but they share zero citations and different vocabulary.

**How:**
1. Encode papers from both fields via SGS
2. The Gaussian kernel measures proximity in the learned splatting space, which captures meaning regardless of field-specific jargon
3. Rank cross-field pairs by kernel similarity
4. Highlight: "These papers from different fields are solving the same problem"

**Why SGS:** The Gaussian representation captures that "error correction" in quantum computing and "error correction" in coding theory share a semantic region despite different vocabularies. The covariance structure can align similar concepts across field boundaries.

**PoC effort:** 5-7 days. Requires selecting two fields and evaluating bridge quality.

---

### Idea 3: "ReviewBot" — Automated Systematic Review Screening

**What:** Given a systematic review protocol (inclusion/exclusion criteria), automatically screen thousands of paper abstracts for relevance.

**How:**
1. Encode the inclusion criteria as a Gaussian scene (the "query scene")
2. For each candidate abstract, render and compare
3. Rank by rendering similarity
4. Human reviewer validates top-N (workload reduction from 10,000 → 500)

**Why SGS:** Systematic review criteria are often complex multi-sentence descriptions. Mean-pooling loses the compositional structure (e.g., "randomized controlled trial" AND "type 2 diabetes" AND "HbA1c outcome"). SGS's rendering equation composes these requirements through alpha-compositing, where transmittance naturally handles the conjunction (each criterion "absorbs" capacity, and only papers matching ALL criteria retain high total weight).

**PoC effort:** 5-7 days. Requires a systematic review dataset with labeled inclusions/exclusions (CLEF eHealth provides this).

---

### Idea 4: "TrendSplat" — Visualize Research Field Evolution

**What:** Map an entire research field as a Gaussian scene and watch it evolve over time. Each topic is a Gaussian blob — see how topics emerge, split, merge, and die.

**How:**
1. Cluster paper abstracts by year using SGS representations
2. Each cluster = a topic Gaussian (position = topic center, covariance = topic breadth, opacity = paper count)
3. Track Gaussians over time: new blobs emerge (new topics), existing blobs split (field diversification), blobs merge (fields converging)
4. Visualize as an animated 3D Gaussian scene — literally splatting research

**Why SGS:** This is the most natural application — the Gaussian scene IS the research field. Each paper is a Gaussian, topics are clusters of Gaussians, and the field is a scene. The visualizer (A3) could be repurposed directly.

**PoC effort:** 7-10 days. Data from arXiv. Most effort in visualization.

---

## Recommended Starting Point

**Idea 1 (EmergingLit)** — same reasoning as cybersecurity:
- Free data (arXiv API)
- Standard evaluation (SciDocs benchmark)
- Clear narrative: "find relevant papers in fields too new for citation-trained models"
- Natural extension of our existing STS-B work (paper similarity is sentence similarity)

---

## Technical Implementation

```python
# scripts/scilit_poc.py

# 1. Download arXiv abstracts (via API or Kaggle dataset)
abstracts = load_arxiv(categories=["cs.AI", "cs.CL", "q-bio", "quant-ph"])

# 2. Encode all abstracts via SGS
# (batch process — each abstract is a "sentence" for SGS)
for abstract in abstracts:
    tokens = tokenize(abstract, word2idx)
    meaning = sgs_encode(tokens)
    store(meaning, metadata)

# 3. Query: given a recent paper, find most similar
query = sgs_encode(tokenize(new_paper_abstract))
similarities = gaussian_kernel(query, all_meanings, all_covs, tau)
top_k = similarities.topk(20)

# 4. Evaluate on SciDocs
# Compare retrieval metrics: nDCG, MAP, Recall@20
# Baselines: SPECTER2, SBERT, BM25
```

---

## Shared Infrastructure with Cybersecurity

Both A4-Cyber and A4-SciLit share:
- The same SGS encoder (pre-trained on AllNLI or GloVe)
- The same evaluation framework (retrieval metrics)
- The same code structure (encode corpus → query → rank)

Building one makes the other almost free. **Recommend building the shared retrieval framework first, then plugging in both domains.**
