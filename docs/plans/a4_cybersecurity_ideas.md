# A4-Cyber: SGS for Cybersecurity Threat Intelligence

## The Opportunity

Every new vulnerability (CVE), zero-day exploit, and APT campaign is a **zero-shot classification problem** — it has never been seen before, yet must be matched to known attack techniques, existing mitigations, and similar past incidents. SGS's proven zero-shot advantage (+0.08) directly applies.

---

## Concrete Product Ideas

### Idea 1: "ZeroDay Match" — Zero-Shot CVE-to-MITRE Mapping

**What:** Given a new CVE description (text), automatically map it to MITRE ATT&CK techniques without any labeled examples of that specific CVE.

**How:**
1. Represent each MITRE ATT&CK technique (currently ~200 techniques) as a Gaussian in semantic space, initialized from technique descriptions
2. When a new CVE arrives, encode it via SGS rendering
3. Match to nearest techniques via kernel evaluation (not cosine — the Gaussian kernel)
4. Output: ranked list of relevant ATT&CK techniques + confidence (residual transmittance = uncertainty)

**Data (all public):**
- MITRE ATT&CK framework: ~200 techniques with descriptions
- NVD/CVE database: ~200K CVEs with descriptions
- Mapped CVE-to-ATT&CK pairs (from MITRE's existing mappings): ~5K labeled pairs for evaluation

**Evaluation:**
- Hold out 2024-2026 CVEs (not in training)
- Measure: Recall@5 of correct ATT&CK technique
- Compare SGS vs SBERT vs SecBERT zero-shot

**Why SGS wins here:** The Gaussian representation naturally captures that "Buffer Overflow" is a broad technique (large Σ) encompassing specific variants, while "CVE-2026-XXXX" is a precise instance (small Σ). The kernel evaluation directly measures inclusion/entailment — a hypernym-hyponym relationship that dot-product similarity misses.

**PoC effort:** 3-5 days. Data is free. Evaluation framework is standard.

---

### Idea 2: "Threat Narrative Clustering" — Group Incidents by Campaign

**What:** Given a stream of security incident reports (text), cluster them by underlying campaign/actor without prior labels.

**How:**
1. Encode each incident report via SGS
2. Cluster in the Gaussian space using the kernel as a similarity metric
3. The covariance of the cluster Gaussian represents the campaign's "breadth"
4. New incidents are matched to clusters via kernel evaluation

**Why SGS:** Gaussian clustering is more natural than cosine-based clustering because each cluster IS a Gaussian. No need to compute centroids and distances — the cluster parameters (μ, Σ) directly define the cluster shape.

**PoC effort:** 5-7 days. Requires incident report data (harder to access than CVEs).

---

### Idea 3: "Anomaly Narrative Scoring" — How Novel Is This Threat?

**What:** Given a threat description, score how "novel" it is — how far it falls from all known techniques.

**How:**
1. Build a Gaussian scene of all known ATT&CK techniques
2. For a new threat report, evaluate kernel values against all techniques
3. If max kernel value is low, the threat is novel (falls in empty space)
4. The residual transmittance T_{n+1} serves as a natural novelty score

**Why SGS:** The transmittance-based rendering naturally produces an "unaccounted meaning" residual. High residual = the threat doesn't fit known categories = novel. This is architecturally native to SGS, not a post-hoc add-on.

**PoC effort:** 2-3 days (simplest of the three).

---

## Recommended Starting Point

**Idea 1 (ZeroDay Match)** — it has:
- Freely available data (MITRE + NVD)
- Clear evaluation protocol (hold-out by year)
- Interpretable output (ranked ATT&CK techniques)
- Direct comparison to existing tools
- A compelling demo: paste a CVE, see matched techniques with Gaussian confidence

---

## Technical Implementation

```python
# scripts/cyber_poc.py

# 1. Load MITRE ATT&CK techniques as Gaussians
for technique in attack_techniques:
    # Initialize Gaussian from technique description
    mu = embed(technique.description)  # GloVe → PCA → d_s=64
    sigma = initial_covariance(technique.scope)  # broader for tactics, narrower for sub-techniques
    alpha = technique.prevalence  # common techniques more salient

# 2. For each test CVE, encode via SGS rendering
meaning = sgs_render(cve_description, technique_gaussians)

# 3. Match by kernel evaluation
scores = gaussian_kernel(meaning, technique_means, technique_covs, tau)
top_k = scores.topk(5)

# 4. Evaluate
recall_at_5 = (ground_truth_technique in top_k)
```
