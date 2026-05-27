# Aristotle Proof Tracker — SGS Mathematical Claims

**Submitted:** 2026-04-07
**Total claims:** 12

## Submitted Proofs

| # | Claim | Priority | Aristotle ID | Status |
|---|---|---|---|---|
| 1 | **1.1** Cholesky LL^T + εI is SPD | P7 | `890a7029-8fe2-434d-a1aa-776dac000340` | QUEUED |
| 2 | **3.1** Blending weights sum ≤ 1 (telescoping) | P1 | `e7ed426e-8c60-482c-9524-1bff4338100b` | QUEUED |
| 3 | **2.2** E[Mahalanobis] = d (chi-squared) | P3 | `4d0ff1ca-3626-4b1c-b1c3-b0a9a4b30461` | QUEUED |
| 4 | **4.1** Projected covariance PΣP^T is SPD | P8 | `9254439e-6764-42cf-9098-0b3f8db4eeda` | QUEUED |
| 5 | **3.3** Complete gradient flow through compositing | P2 | `c440d891-93e3-491f-8f6c-9b6e2e0eefc7` | QUEUED |
| 6 | **2.1** Anisotropic Gaussian is Mercer kernel | P9 | `fc1437d1-337d-45fd-b837-e5c3894b4ca3` | QUEUED |
| 7 | **3.2** Monotonic transmittance | P6 | `418ce11a-8191-440e-8043-923e0c671b70` | QUEUED |
| 8 | **3.4** Order-dependence (constructive) | P11 | `2297d38b-3eea-47c1-bbe0-129e313f6fcc` | IN_PROGRESS |
| 9 | **5.2** Opacity monotonicity across passes | P12 | `7e2bae05-f1a6-42b2-8cbc-c34a995d3ae2` | QUEUED |
| 10 | **5.1** Multi-pass bounded (tanh) | P6 | `2f241daf-690a-4fee-a18c-dd93b68f4792` | QUEUED |
| 11 | **2.4** Sparsity bound S(q) = O(N·(r/R)^d) | P4 (CRITICAL) | `11e20423-9092-4114-ba31-1cfc81b6084a` | QUEUED |
| 12 | **3.5** Rendering ↔ Attention relationship | P5 (NOVEL) | `efb72d79-54a2-49ed-a263-d7b9ce34dc33` | QUEUED |
| 13 | **7.1** Split doubles mass → halve opacity | P13 | `62241506-c9dc-4fa4-a1a6-ca75a809589c` | QUEUED |

## H-SGS Proofs (Submitted 2026-04-14)

| # | Claim | Priority | Aristotle ID | Status |
|---|---|---|---|---|
| 14 | **H1** Two-pass partition equivalence | P1 (CRITICAL) | `d760d6d8-f53a-435a-95ee-ded40f844d9b` | COMPLETE |
| 15 | **H2** Expressiveness under T_max cap | P2 (CRITICAL) | `d3ba9b30-d03e-45b2-a3a0-d5a3258e9048` | COMPLETE |
| 16 | **H4** Total weight is permutation-invariant | P3 (HIGH) | `6fa7e757-9065-41e7-80a0-dcf746bd6e69` | COMPLETE |

## Physical Gaussians Proofs (Prepared 2026-05-27)

| # | Claim | Priority | Aristotle ID | Status |
|---|---|---|---|---|
| 17 | **P1** Covariance SPD from quat + scale | P1 (CRITICAL) | `6fce0b10-b73c-4e88-bc58-23c83a2046da` | QUEUED |
| 18 | **P2** Material similarity bounded [0,1], self=1 | P3 (HIGH) | `466538bb-8100-4791-aad1-14fae6d08224` | QUEUED |
| 19 | **P4** Covariance similarity well-defined for SPD | P4 (HIGH) | `28420e6a-ab3d-4a1c-a893-a383adfecb07` | QUEUED |
| 20 | **P5** Physics prediction continuity (MLP) | P6 (MEDIUM) | — | PENDING |
| 21 | **P6** Correlation hypothesis (info-theoretic) | P2 (CRITICAL) | — | PENDING (hardest) |
| 22 | **P8** Densification preserves material coherence | P5 (HIGH) | — | PENDING |
| 23 | **P9** Dimensionality sufficiency for K classes | P7 (MEDIUM) | — | PENDING |

Formal specification: `docs/proofs/physical_gaussians_math.md`

## Commands

```bash
# Check status
export ARISTOTLE_API_KEY='arstl_oztlKEDfQez7GVY52wyG99zIkpdYvKHTLAHs9jSwC8M'
aristotle list

# Get result for a specific proof
aristotle result <ID> --destination ./proofs/<claim>/

# Cancel a proof
aristotle cancel <ID>
```

## Notes
- Aristotle works as a Lean 4 formal prover — it will attempt to formalize these English statements into Lean 4 proofs
- No local Lean installation required — Aristotle runs its own Lean environment
- Results will contain Lean 4 source files with formal proofs (or failure reports)
- Claims 2.4 (sparsity) and 3.5 (rendering↔attention) are the hardest — may require human guidance if Aristotle struggles
