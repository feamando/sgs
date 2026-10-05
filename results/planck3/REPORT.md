# Planck 3.0 results (2026-10-05 19:04)

## G0 (seed benchmark)

| run | n | success | answered | wrong when answered | ECE | depth evidence | search-only @3 | $/correct | LLM tok/task | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| g0_gemma | 47 | 0.681 | 0.915 | 0.256 | 0.137 | 1.000 | 0.957 | 0.008738 | 5486 | PASS |
| g0_gemma_closedbook | 47 | 0.957 | 1.000 | 0.043 | 0.043 |  |  | 0.000110 | 82 | COMPARATOR |
| g0_gemma_closedbook_quick | 12 | 1.000 | 1.000 | 0.000 | 0.000 |  |  | 0.000173 | 136 | COMPARATOR |
| g0_gemma_quick | 12 | 0.667 | 1.000 | 0.333 | 0.183 | 0.667 | 0.417 | 0.015154 | 9337 | PASS |
| g0_heuristic | 47 | 0.617 | 0.979 | 0.370 | 0.132 | 1.000 | 0.851 | 0.000000 | 0 | BASELINE |
| g0_heuristic_quick | 12 | 0.250 | 0.833 | 0.700 | 0.278 | 0.667 | 0.417 | 0.000000 | 0 | BASELINE |

**g0_gemma vs base chat:** success 0.681 vs 0.957 (0.71x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct 79.8x MORE expensive (5486 vs 82 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_gemma_quick vs base chat:** success 0.667 vs 1.000 (0.67x); depth evidence 0.667 vs none (base chat shows no sources); cost per correct 87.7x MORE expensive (9337 vs 136 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.

## G1 Wikiracing (g1_eval_s0)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1192 | 0.001 | 1.000 | 0.202 | 0.111 | 0.000 |
| lexical | 1192 | 0.153 | 1.226 | 0.459 | 0.128 | 0.014 |
| head:hash | 1192 | 0.146 | 1.251 | 0.580 | 0.085 | 0.651 |
| head:planck | 1192 | 0.211 | 1.272 | 0.624 | 0.095 | 0.432 |
| gemma | 100 | 0.300 | 1.347 |  |  | 222.664 |

Cold CPU latency (encode target + all candidate titles): median 94 ms, p90 536 ms, 50 candidates on average

**Verdict:** FAIL (try the Hertz encoder). planck-vs-hash head delta +0.065 (<=0 means Planck features add nothing); head/teacher = 0.70 (pass >= 0.8); warm decision 0.4 ms (pass < 100.0)

## G1 Wikiracing (g1_eval_s0_quick)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 100 | 0.000 |  | 0.190 | 0.105 | 0.000 |
| lexical | 100 | 0.160 | 1.403 | 0.460 | 0.136 | 0.015 |
| head:hash | 100 | 0.040 | 1.000 | 0.490 | 0.069 | 0.553 |
| head:planck | 100 | 0.110 | 1.358 | 0.517 | 0.105 | 0.426 |
| gemma | 20 | 0.200 | 1.250 |  |  | 272.297 |

Cold CPU latency (encode target + all candidate titles): median 101 ms, p90 584 ms, 50 candidates on average

**Verdict:** FAIL (try the Hertz encoder). planck-vs-hash head delta +0.070 (<=0 means Planck features add nothing); head/teacher = 0.55 (pass >= 0.8); warm decision 0.4 ms (pass < 100.0)

