# A4: Few-Shot Domain Application — Research

## Top Domains Where SGS's Zero-Shot Advantage Matters Most

Ranked by intersection of data scarcity, task fit, commercial value, and zero-shot relevance.

---

### 1. Rare Disease Diagnosis (Clinical NLP)

**Task:** Matching patient phenotype descriptions to rare disease profiles.

**Why data is scarce:** ~7,000 rare diseases, many with <100 documented cases worldwide. Annotation requires specialist clinicians ($200-500/hr). HIPAA restricts sharing.

**Current SOTA:** BioLinkBERT/PubMedBERT — collapses on ultra-rare conditions absent from pre-training. Rule-based HPO matching (Exomiser) misses semantic paraphrases.

**Market:** $7B+ rare disease diagnostics. Average diagnostic odyssey: 5-7 years.

**SGS angle:** +0.08 zero-shot Spearman in phenotype-to-disease matching could reduce misdiagnosis for conditions with near-zero training examples. Every rare disease is a zero-shot problem.

---

### 2. Low-Resource Language NLP (100+ languages)

**Task:** Sentence similarity, cross-lingual transfer, text classification for languages with <10K labeled pairs (Yoruba, Quechua, Tigrinya, etc).

**Why data is scarce:** No commercial annotation incentive. XLM-R/mBERT degrade sharply below ~50K monolingual pre-training sentences.

**Market:** 1.2B+ people with inadequate NLP tooling. Humanitarian NLP, education access, emerging-market fintech.

**SGS angle:** Zero-shot composition from limited vocabulary embeddings. The +0.08 gap is largest exactly where multilingual models fail.

---

### 3. Cybersecurity Threat Intelligence

**Task:** Matching novel threat descriptions to MITRE ATT&CK techniques. Linking incident narratives to known attack patterns.

**Why data is scarce:** Threat intel is proprietary/classified. New attack vectors have zero prior labeled examples *by definition*. Jargon shifts quarterly.

**Market:** $180B+ cybersecurity. Mean-time-to-detect averages 204 days.

**SGS angle:** Every zero-day is literally a zero-shot problem. Matching never-before-seen attack descriptions to technique taxonomies is the core use case.

---

### 4. Legal Document Analysis (Niche Jurisdictions)

**Task:** Clause similarity, contract classification, case matching in non-English jurisdictions or niche regulatory domains (GDPR enforcement, EU AI Act compliance).

**Why data is scarce:** Annotation requires licensed attorneys ($200-600/hr). Cross-jurisdictional transfer is poor. Novel regulations have zero labeled data.

**Market:** $28B+ legal tech, 8% CAGR.

**SGS angle:** Few-shot clause matching for novel regulatory frameworks where firms cannot wait for dataset creation.

---

### 5. Scientific Literature Triage (Emerging Fields)

**Task:** Classifying/matching papers in new research areas (synthetic bio, quantum computing, AI safety) where taxonomies don't exist yet.

**Why data is scarce:** New subfields produce literature faster than labels can be created. Citation-based training is backward-looking.

**Market:** Systematic reviews cost $100K-500K each. Pharma literature screening is a $2B+ bottleneck.

**SGS angle:** Zero-shot abstract similarity in fields too new for fine-tuning data to exist.

---

### 6. Financial Regulatory Filings

**Task:** Similarity matching across jurisdictions, sanctions screening, ESG disclosure classification.

**Why data is scarce:** Each jurisdiction's regulatory language is unique. New frameworks (EU DORA, MiCA) have no training data.

**Market:** $12B+ RegTech. Banks spend $180B/yr on compliance.

**SGS angle:** Few-shot classification under new regulations where data won't exist for years.

---

## Recommendation

**Start with #3 (Cybersecurity) or #5 (Scientific Literature).** Reasons:

- **Data accessibility:** MITRE ATT&CK and arXiv are freely available. Rare disease data requires IRB/ethics approval. Legal data is paywalled.
- **Fast evaluation:** Can build a PoC in days using existing public datasets (CVE descriptions, arXiv abstracts).
- **Clear narrative:** "SGS matches novel cyber threats with zero training examples" or "SGS finds relevant papers in fields that didn't exist last year."
- **Commercial interest:** Both have clear buyers (SOCs, pharma R&D teams).

#1 (Rare Disease) has the highest humanitarian impact but requires clinical partnerships and data access. Better as a Phase 2 after proving the approach on accessible data.
