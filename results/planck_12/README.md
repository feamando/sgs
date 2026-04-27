# Planck 1.2 — Ablation results

*Placeholder. Populated by `scripts/validate_planck12.py` runs.*

See:
- `docs/plans/planck_12_plan.md` — design
- `docs/plans/planck_12_validation.md` — ablation matrix + gate
- `docs/plans/planck_12_runbook.md` — how to run

Each run's stdout is teed to `<run_id>/train_log.txt`. The
aggregate JSON at `ablation.json` is the source of truth for the
summary table.

## Summary table

*(Filled in once runs complete.)*

| run | label | status | val loss | tok/s | wall (s) | speedup |
|-----|-------|--------|----------|-------|----------|---------|
| baseline | Plain CE, 3 passes | pending | – | – | – | 1.00× |
| tl | §2.1 only | pending | – | – | – | – |
| ap | §2.2 only | pending | – | – | – | – |
| sk | §2.3 only | pending | – | – | – | – |
| shk | §2.4 only | pending | – | – | – | – |
| all | all four composed | pending | – | – | – | – |

## Gate

Compound `all` vs. `baseline`:
- Sample efficiency ≥ 1.43× (same val-loss target on ≤70% tokens): **pending**
- Wall-clock speedup ≥ 1.8×: **pending**

## Notes

*(Populated after the runs with anomalies, additivity audit, and
recommendation.)*
