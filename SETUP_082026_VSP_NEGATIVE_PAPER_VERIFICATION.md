# SETUP 08/2026 — VSP Negative-Result Paper: Verification for JMLR

Target venue: **JMLR** (same as the alpha-compositing paper). A main-venue
negative only publishes if it credibly rules out "you did it wrong." A red-team
sweep (2026-07-27) surfaced the gaps; this doc closes the two that need GPU. The
editorial fixes (metric mislabel 0.13->0.00, table error, underpowered-null
reframing, clustering caveat) are already in `paper/vsp_negative_result.tex`
(commit 8bcc32b). What remains is evidence.

## One command

From the repo root on the 4090 box:

```bash
bash scripts/run_paper_verification.sh
```

It trains what's missing, evals each checkpoint on the 105-pair gate, runs
`verify_paper.py`, writes `results/paper_verification_summary.txt`, and pushes
the new result JSONs. Idempotent: existing checkpoints/results are skipped, so a
re-run after any interruption resumes. `--no-push` computes only; `--quick`
runs the 2k control only (skips all 40k, ~4h) for a smoke test.

**GPU budget:** the full grid is dominated by 3x 40k runs (~10h each) plus one
40k scrambled run; ~30h wall on one 4090. `--quick` is ~4h.

## What it runs, and the reviewer concern each closes

| ID | Control | What it answers | Runs added |
|----|---------|-----------------|------------|
| C1 | **Scrambled-bundle init** | Is the failure the SIGNAL or the init pipeline? Init from bundles permuted across the vocab (real stats, wrong token). If grounded ≈ scrambled, the grounding carries nothing usable; the negative is about the signal, not a bad warm start. | scrambled 2k s0 (+40k s0) |
| C2 | **40k reproduction** | The headline −3.8 is a single seed on code that had a since-fixed init-scale bug, and the paper's own ethos is reproduction. 3 seeds of grounded vs baseline at full compute on the rescaled init give a mean ± CI instead of one point. | vsp/baseline 40k s1, s2 (s0 reused) |

C3 (low-context held-out −0.004) needs no new compute; `verify_paper.py` reports
it for completeness.

**Not run here (deliberate):** WiC calibration. WiC is a different task (binary
same-sense across two sentences, scored on hidden-state reps, not next-token
logprob) and needs an external dataset + a new scoring head, too fragile for an
unattended run and only a reviewer *nice-to-have*, not a blocking control. Do it
by hand if a reviewer asks. Same for the oracle-sense upper bound and the
incremental-margin regression (diagnosis-sharpening, not gating).

## Mechanism (what changed in the code)

`scripts/train_planck2.py`:
- `--shuffle-bundles`: in `init_from_bundles`, permute the bundle ROWS before the
  projection. Preserves marginal statistics, dimensionality, projection, and the
  native-std rescaling; destroys only the sense↔token correspondence. Verified:
  scrambled init differs from grounded but has identical std (0.0200) and
  mean-abs (0.0160).
- `--seed` (added earlier) seeds model init + the bundle projection + the
  permutation + the data shuffle, so every arm/seed is a genuine independent draw.

`scripts/verify_paper.py`: reads all `results/disambig_<arm>_<regime>_s<seed>.json`
(and the legacy names), reports the inventory, C1/C2/C3 verdicts, per-seed
deltas, and 95% CIs. GPU-free; safe to run any time.

`scripts/run_paper_verification.sh`: the idempotent driver above.

## File / naming convention

```
checkpoints/planck2_<arm>_<regime>_s<seed>/final.pt
results/disambig_<arm>_<regime>_s<seed>.json      arm ∈ {vsp, baseline, scrambled}
legacy s0 40k:  results/disambig_{vsp,baseline}.json   (aliased by the runner)
legacy s0 2k:   results/disambig_{vsp,baseline}_2k.json
```

## Decision rules when it finishes

Read `results/paper_verification_summary.txt`.

- **C1 grounded − scrambled ≈ 0 (CI includes 0):** confirms the signal is
  useless, not the pipeline. This is the expected, paper-supporting outcome.
  Update paper Discussion "what is bounded" to cite the measured control.
- **C1 grounded > scrambled (CI excludes 0):** the grounding DOES carry
  something the init can use; the earlier negative may be a delivery problem.
  This would reopen the line, report honestly.
- **C2 40k 3-seed mean −3.8-ish with a tight CI:** the headline reproduces;
  replace the single-seed −3.8 with mean ± CI in the paper.
- **C2 CI now spans 0 or flips positive:** the −3.8 was partly the init-scale
  bug; the full-compute claim weakens to "no reliable effect," update Section 5.

Then hand the numbers back and I fold them into `paper/vsp_negative_result.tex`
(Table 1, Section 5, Discussion) + regenerate the summary.

## Provenance

Red-team findings and the editorial fixes: see `SETUP_202607_VSP_v1.md`
RESULT 1–4 and the Brain memory `project_sgs_vsp_gate`. Paper: `paper/
vsp_negative_result.tex`. Prior verification tooling: `aggregate_disambig_seeds.py`,
`rerank_disambiguation.py`.
