# Planck 3.0 results (2026-10-07 15:35)

## G0 (seed benchmark)

| run | n | success | answered | wrong when answered | ECE | depth evidence | search-only @3 | answer fetches/task | from snippet | gated | $/correct | LLM tok/task | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g0_gemma | 47 | 0.681 | 0.915 | 0.256 | 0.137 | 1.000 | 0.957 | 2.809 |  |  | 0.008738 | 5486 | PASS |
| g0_gemma_closedbook | 47 | 0.957 | 1.000 | 0.043 | 0.043 |  |  | 0.000 |  |  | 0.000110 | 82 | COMPARATOR |
| g0_gemma_closedbook_quick | 12 | 1.000 | 1.000 | 0.000 | 0.000 |  |  | 0.000 |  |  | 0.000173 | 136 | COMPARATOR |
| g0_gemma_quick | 12 | 0.667 | 1.000 | 0.333 | 0.183 | 0.667 | 0.417 | 4.333 |  |  | 0.015154 | 9337 | PASS |
| g0_gemma_snip | 47 | 0.851 | 0.936 | 0.091 | 0.039 | 0.957 | 0.957 | 0.000 | 0.894 | 0.000 | 0.005360 | 4206 | PASS |
| g0_heuristic | 47 | 0.617 | 0.979 | 0.370 | 0.132 | 1.000 | 0.851 | 2.702 |  |  | 0.000000 | 0 | BASELINE |
| g0_heuristic_quick | 12 | 0.250 | 0.833 | 0.700 | 0.278 | 0.667 | 0.417 | 3.917 |  |  | 0.000000 | 0 | BASELINE |
| g0_heuristic_snip | 47 | 0.936 | 1.000 | 0.064 | 0.057 | 0.957 | 0.851 | 0.021 | 0.830 | 0.000 | 0.000000 | 0 | BASELINE |
| g0_planck-g2-hash | 47 | 0.000 | 0.000 | 0.000 |  | 1.000 | 1.000 | 3.213 | 0.000 | 0.000 |  | 0 | G2 (rule B) |
| g0_planck-g2-hash_quick | 12 | 0.000 | 0.000 | 0.000 |  | 1.000 | 1.000 | 5.667 | 0.000 | 0.000 |  | 0 | G2 (rule B) |
| g0_planck-g2-planck | 47 | 0.000 | 0.000 | 0.000 |  | 1.000 | 1.000 | 3.213 | 0.000 | 0.000 |  | 0 | G2 (rule B) |
| g0_planck-g2-planck_quick | 12 | 0.000 | 0.000 | 0.000 |  | 1.000 | 1.000 | 5.667 | 0.000 | 0.000 |  | 0 | G2 (rule B) |
| g0f_gemma_closedbook | 143 | 0.049 | 0.986 | 0.950 | 0.950 |  |  | 0.000 |  |  | 0.002297 | 75 | COMPARATOR |
| g0f_gemma_closedbook_quick | 12 | 0.000 | 1.000 | 1.000 | 1.000 |  |  | 0.000 |  |  |  | 110 | COMPARATOR |
| g0f_gemma_snip | 143 | 0.056 | 0.084 | 0.333 | 0.267 | 0.063 | 0.077 | 0.000 | 0.084 | 0.000 | 0.015598 | 765 | ROUND 3 (rules A/B) |
| g0f_gemma_snip_quick | 12 | 0.667 | 1.000 | 0.333 | 0.267 | 0.750 | 0.917 | 0.000 | 1.000 | 0.000 | 0.007831 | 4830 | ROUND 3 (rules A/B) |
| g0f_heuristic_snip | 143 | 0.056 | 0.091 | 0.385 | 0.356 | 0.070 | 0.084 | 0.021 | 0.077 | 0.000 | 0.000000 | 0 | ROUND 3 (rules A/B) |
| g0f_heuristic_snip_quick | 12 | 0.667 | 1.000 | 0.333 | 0.302 | 0.750 | 0.917 | 0.250 | 0.833 | 0.000 | 0.000000 | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-hash | 143 | 0.000 | 0.000 | 0.000 |  | 0.070 | 0.084 | 0.287 | 0.000 | 0.000 |  | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-hash_quick | 12 | 0.000 | 0.000 | 0.000 |  | 0.750 | 0.917 | 3.250 | 0.000 | 0.000 |  | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-planck | 143 | 0.000 | 0.000 | 0.000 |  | 0.070 | 0.084 | 0.287 | 0.000 | 0.000 |  | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-planck_quick | 12 | 0.000 | 0.000 | 0.000 |  | 0.750 | 0.917 | 3.250 | 0.000 | 0.000 |  | 0 | ROUND 3 (rules A/B) |

**g0_gemma vs base chat:** success 0.681 vs 0.957 (0.71x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct 79.8x MORE expensive (5486 vs 82 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_gemma_quick vs base chat:** success 0.667 vs 1.000 (0.67x); depth evidence 0.667 vs none (base chat shows no sources); cost per correct 87.7x MORE expensive (9337 vs 136 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_gemma_snip vs base chat:** success 0.851 vs 0.957 (0.89x); depth evidence 0.957 vs none (base chat shows no sources); cost per correct 48.9x MORE expensive (4206 vs 82 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_heuristic vs base chat:** success 0.617 vs 0.957 (0.64x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct 8567.8x cheaper.
**g0_heuristic_quick vs base chat:** success 0.250 vs 1.000 (0.25x); depth evidence 0.667 vs none (base chat shows no sources); cost per correct 3112.9x cheaper.
**g0_heuristic_snip vs base chat:** success 0.936 vs 0.957 (0.98x); depth evidence 0.957 vs none (base chat shows no sources); cost per correct 113295.6x cheaper.
**g0_planck-g2-hash vs base chat:** success 0.000 vs 0.957 (0.00x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct n/a.
**g0_planck-g2-hash_quick vs base chat:** success 0.000 vs 1.000 (0.00x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct n/a.
**g0_planck-g2-planck vs base chat:** success 0.000 vs 0.957 (0.00x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct n/a.
**g0_planck-g2-planck_quick vs base chat:** success 0.000 vs 1.000 (0.00x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct n/a.
**g0f_gemma_snip vs base chat:** success 0.056 vs 0.049 (1.14x); depth evidence 0.063 vs none (base chat shows no sources); cost per correct 6.8x MORE expensive (765 vs 75 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0f_gemma_snip_quick vs base chat:** success 0.667 vs 0.000 (666666666.67x); depth evidence 0.750 vs none (base chat shows no sources); cost per correct n/a. An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0f_heuristic_snip vs base chat:** success 0.056 vs 0.049 (1.14x); depth evidence 0.070 vs none (base chat shows no sources); cost per correct 530579.8x cheaper.
**g0f_heuristic_snip_quick vs base chat:** success 0.667 vs 0.000 (666666666.67x); depth evidence 0.750 vs none (base chat shows no sources); cost per correct n/a.
**g0f_planck-g2-hash vs base chat:** success 0.000 vs 0.049 (0.00x); depth evidence 0.070 vs none (base chat shows no sources); cost per correct n/a.
**g0f_planck-g2-hash_quick vs base chat:** success 0.000 vs 0.000 (0.00x); depth evidence 0.750 vs none (base chat shows no sources); cost per correct n/a.
**g0f_planck-g2-planck vs base chat:** success 0.000 vs 0.049 (0.00x); depth evidence 0.070 vs none (base chat shows no sources); cost per correct n/a.
**g0f_planck-g2-planck_quick vs base chat:** success 0.000 vs 0.000 (0.00x); depth evidence 0.750 vs none (base chat shows no sources); cost per correct n/a.

### By regime (fresh = 2026 facts; long_tail = <= 3 Wikipedia editions)

| run | per regime: success (n, wrong when answered) |
|---|---|
| g0f_gemma_closedbook | fresh: 0.000 (n=60, wrong 1.000) | long_tail: 0.084 (n=83, wrong 0.915) |
| g0f_gemma_closedbook_quick | fresh: 0.000 (n=6, wrong 1.000) | long_tail: 0.000 (n=6, wrong 1.000) |
| g0f_gemma_snip | fresh: 0.050 (n=60, wrong 0.500) | long_tail: 0.060 (n=83, wrong 0.167) |
| g0f_gemma_snip_quick | fresh: 0.500 (n=6, wrong 0.500) | long_tail: 0.833 (n=6, wrong 0.167) |
| g0f_heuristic_snip | fresh: 0.050 (n=60, wrong 0.500) | long_tail: 0.060 (n=83, wrong 0.286) |
| g0f_heuristic_snip_quick | fresh: 0.500 (n=6, wrong 0.500) | long_tail: 0.833 (n=6, wrong 0.167) |
| g0f_planck-g2-hash | fresh: 0.000 (n=60, wrong 0.000) | long_tail: 0.000 (n=83, wrong 0.000) |
| g0f_planck-g2-hash_quick | fresh: 0.000 (n=6, wrong 0.000) | long_tail: 0.000 (n=6, wrong 0.000) |
| g0f_planck-g2-planck | fresh: 0.000 (n=60, wrong 0.000) | long_tail: 0.000 (n=83, wrong 0.000) |
| g0f_planck-g2-planck_quick | fresh: 0.000 (n=6, wrong 0.000) | long_tail: 0.000 (n=6, wrong 0.000) |

## G2 heads (learned from known answers)

| head | train points | val points with a right candidate | learned choice accuracy | deterministic ranker | 'none' correct (points without one) | val ECE | answer accuracy when p >= 0.5 |
|---|---|---|---|---|---|---|---|
| g2_head_hash_s0 | 84 | 5 | 0.400 | 0.200 | 1.000 (3) | 0.281 | 0.000 |
| g2_head_planck_s0 | 84 | 5 | 0.600 | 0.200 | 1.000 (3) | 0.306 | 0.000 |

## G1 Wikiracing (g1_eval_s0)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1192 | 0.001 | 1.000 | 0.202 | 0.111 | 0.000 |
| lexical | 1192 | 0.153 | 1.226 | 0.459 | 0.128 | 0.015 |
| head:hash | 1192 | 0.146 | 1.251 | 0.580 | 0.085 | 0.632 |
| head:planck-rank | 1192 | 0.271 | 1.282 | 0.635 | 0.112 | 0.458 |
| head:planck | 1192 | 0.211 | 1.272 | 0.624 | 0.095 | 0.445 |
| gemma | 300 | 0.337 | 1.348 |  |  | 226.268 |

Cold CPU latency (encode target + all candidate titles): median 95 ms, p90 553 ms, 50 candidates on average

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.163 | 0.337 | 0.49 | 21 | 73 | 6.65e-08 |
| head:hash vs lexical | 1192 | 0.146 | 0.153 |  | 112 | 120 | 0.646 |
| head:planck-rank vs gemma | 300 | 0.267 | 0.337 | 0.79 | 39 | 60 | 0.0439 |
| head:planck-rank vs head:hash | 1192 | 0.271 | 0.146 |  | 222 | 73 | 1.21e-18 |
| head:planck-rank vs lexical | 1192 | 0.271 | 0.153 |  | 227 | 86 | 8.04e-16 |
| head:planck vs gemma | 300 | 0.177 | 0.337 | 0.52 | 24 | 72 | 9.7e-07 |
| head:planck vs head:hash | 1192 | 0.211 | 0.146 |  | 161 | 84 | 9.88e-07 |
| head:planck vs lexical | 1192 | 0.211 | 0.153 |  | 171 | 102 | 3.53e-05 |
| head:planck-rank vs head:planck | 1192 | 0.271 | 0.211 |  | 174 | 102 | 1.73e-05 |

**Verdict:** FAIL. head:planck-vs-hash +0.065 (McNemar p=9.88e-07); paired head/teacher = 0.52 on 300 shared races (pass >= 0.8); warm decision 0.4 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.49, head:planck-rank 0.79

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

## G1 Wikiracing (g1_eval_s1)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1192 | 0.001 | 1.000 | 0.202 | 0.111 | 0.000 |
| lexical | 1192 | 0.153 | 1.226 | 0.459 | 0.128 | 0.013 |
| head:hash | 1192 | 0.146 | 1.227 | 0.584 | 0.106 | 0.636 |
| head:planck-rank | 1192 | 0.277 | 1.287 | 0.636 | 0.113 | 0.445 |
| head:planck | 1192 | 0.247 | 1.299 | 0.628 | 0.102 | 0.426 |
| gemma | 300 | 0.337 | 1.348 |  |  | 238.269 |

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.127 | 0.337 | 0.38 | 14 | 77 | 1.05e-11 |
| head:hash vs lexical | 1192 | 0.146 | 0.153 |  | 120 | 128 | 0.657 |
| head:planck-rank vs gemma | 300 | 0.283 | 0.337 | 0.84 | 38 | 54 | 0.117 |
| head:planck-rank vs head:hash | 1192 | 0.277 | 0.146 |  | 234 | 78 | 2.93e-19 |
| head:planck-rank vs lexical | 1192 | 0.277 | 0.153 |  | 246 | 98 | 8.08e-16 |
| head:planck vs gemma | 300 | 0.200 | 0.337 | 0.59 | 28 | 69 | 3.8e-05 |
| head:planck vs head:hash | 1192 | 0.247 | 0.146 |  | 204 | 83 | 6.41e-13 |
| head:planck vs lexical | 1192 | 0.247 | 0.153 |  | 218 | 105 | 3.06e-10 |
| head:planck-rank vs head:planck | 1192 | 0.277 | 0.247 |  | 152 | 117 | 0.038 |

**Verdict:** FAIL. head:planck-vs-hash +0.102 (McNemar p=6.41e-13); paired head/teacher = 0.59 on 300 shared races (pass >= 0.8); warm decision 0.4 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.38, head:planck-rank 0.84

## G1 Wikiracing (g1_eval_s2)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1192 | 0.001 | 1.000 | 0.202 | 0.111 | 0.000 |
| lexical | 1192 | 0.153 | 1.226 | 0.459 | 0.128 | 0.013 |
| head:hash | 1192 | 0.138 | 1.283 | 0.580 | 0.082 | 0.620 |
| head:planck-rank | 1192 | 0.270 | 1.310 | 0.632 | 0.104 | 0.431 |
| head:planck | 1192 | 0.231 | 1.292 | 0.622 | 0.084 | 0.414 |
| gemma | 300 | 0.337 | 1.348 |  |  | 238.269 |

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.133 | 0.337 | 0.40 | 13 | 74 | 1.59e-11 |
| head:hash vs lexical | 1192 | 0.138 | 0.153 |  | 103 | 121 | 0.256 |
| head:planck-rank vs gemma | 300 | 0.240 | 0.337 | 0.71 | 34 | 63 | 0.00423 |
| head:planck-rank vs head:hash | 1192 | 0.270 | 0.138 |  | 231 | 73 | 2.85e-20 |
| head:planck-rank vs lexical | 1192 | 0.270 | 0.153 |  | 234 | 94 | 6.39e-15 |
| head:planck vs gemma | 300 | 0.213 | 0.337 | 0.63 | 30 | 67 | 0.000219 |
| head:planck vs head:hash | 1192 | 0.231 | 0.138 |  | 200 | 89 | 5.71e-11 |
| head:planck vs lexical | 1192 | 0.231 | 0.153 |  | 196 | 103 | 8.21e-08 |
| head:planck-rank vs head:planck | 1192 | 0.270 | 0.231 |  | 141 | 94 | 0.00262 |

**Verdict:** FAIL. head:planck-vs-hash +0.093 (McNemar p=5.71e-11); paired head/teacher = 0.63 on 300 shared races (pass >= 0.8); warm decision 0.4 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.40, head:planck-rank 0.71

## G1 Wikiracing (g1_eval_s3_t1)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1200 | 0.001 | 1.500 | 0.197 | 0.103 | 0.000 |
| lexical | 1200 | 0.144 | 1.231 | 0.463 | 0.128 | 0.015 |
| head:hash | 1200 | 0.108 | 1.208 | 0.580 | 0.105 | 0.556 |
| head:planck-rank | 1200 | 0.235 | 1.275 | 0.636 | 0.104 | 0.442 |
| head:planck | 1200 | 0.224 | 1.288 | 0.635 | 0.112 | 0.422 |
| gemma | 300 | 0.337 | 1.335 |  |  | 234.415 |

Cold CPU latency (encode target + all candidate titles): median 87 ms, p90 298 ms, 20 candidates on average

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.093 | 0.337 | 0.28 | 11 | 84 | 4.52e-15 |
| head:hash vs lexical | 1200 | 0.108 | 0.144 |  | 80 | 123 | 0.0031 |
| head:planck-rank vs gemma | 300 | 0.227 | 0.337 | 0.67 | 23 | 56 | 0.000264 |
| head:planck-rank vs head:hash | 1200 | 0.235 | 0.108 |  | 215 | 63 | 1.35e-20 |
| head:planck-rank vs lexical | 1200 | 0.235 | 0.144 |  | 198 | 89 | 1.1e-10 |
| head:planck vs gemma | 300 | 0.220 | 0.337 | 0.65 | 25 | 60 | 0.000187 |
| head:planck vs head:hash | 1200 | 0.224 | 0.108 |  | 198 | 59 | 9.81e-19 |
| head:planck vs lexical | 1200 | 0.224 | 0.144 |  | 180 | 84 | 3.4e-09 |
| head:planck-rank vs head:planck | 1200 | 0.235 | 0.224 |  | 134 | 121 | 0.452 |

**Verdict:** FAIL. head:planck-rank-vs-hash +0.127 (McNemar p=1.35e-20); paired head/teacher = 0.67 on 300 shared races (pass >= 0.8); warm decision 0.4 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.28, head:planck 0.65

## G1 Wikiracing (g1_eval_s4_t1)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1200 | 0.001 | 1.500 | 0.197 | 0.103 | 0.000 |
| lexical | 1200 | 0.144 | 1.231 | 0.463 | 0.128 | 0.015 |
| head:hash | 1200 | 0.111 | 1.239 | 0.587 | 0.120 | 0.650 |
| head:planck-rank | 1200 | 0.230 | 1.296 | 0.633 | 0.102 | 0.467 |
| head:planck | 1200 | 0.207 | 1.253 | 0.632 | 0.090 | 0.451 |
| gemma | 300 | 0.337 | 1.335 |  |  | 251.616 |

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.080 | 0.337 | 0.24 | 8 | 85 | 2.26e-17 |
| head:hash vs lexical | 1200 | 0.111 | 0.144 |  | 83 | 123 | 0.00644 |
| head:planck-rank vs gemma | 300 | 0.270 | 0.337 | 0.80 | 34 | 54 | 0.0422 |
| head:planck-rank vs head:hash | 1200 | 0.230 | 0.111 |  | 208 | 65 | 1.28e-18 |
| head:planck-rank vs lexical | 1200 | 0.230 | 0.144 |  | 187 | 84 | 3.58e-10 |
| head:planck vs gemma | 300 | 0.230 | 0.337 | 0.68 | 21 | 53 | 0.000256 |
| head:planck vs head:hash | 1200 | 0.207 | 0.111 |  | 185 | 69 | 2.03e-13 |
| head:planck vs lexical | 1200 | 0.207 | 0.144 |  | 173 | 97 | 4.37e-06 |
| head:planck-rank vs head:planck | 1200 | 0.230 | 0.207 |  | 111 | 84 | 0.0623 |

**Verdict:** PASS. head:planck-rank-vs-hash +0.119 (McNemar p=1.28e-18); paired head/teacher = 0.80 on 300 shared races (pass >= 0.8); warm decision 0.5 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.24, head:planck 0.68

## G1 Wikiracing (g1_eval_s5_t1)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1200 | 0.001 | 1.500 | 0.197 | 0.103 | 0.000 |
| lexical | 1200 | 0.144 | 1.231 | 0.463 | 0.128 | 0.015 |
| head:hash | 1200 | 0.102 | 1.221 | 0.580 | 0.118 | 0.623 |
| head:planck-rank | 1200 | 0.241 | 1.322 | 0.635 | 0.099 | 0.464 |
| head:planck | 1200 | 0.223 | 1.306 | 0.629 | 0.095 | 0.465 |
| gemma | 300 | 0.337 | 1.335 |  |  | 251.616 |

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.080 | 0.337 | 0.24 | 8 | 85 | 2.26e-17 |
| head:hash vs lexical | 1200 | 0.102 | 0.144 |  | 64 | 115 | 0.00017 |
| head:planck-rank vs gemma | 300 | 0.223 | 0.337 | 0.66 | 34 | 68 | 0.000987 |
| head:planck-rank vs head:hash | 1200 | 0.241 | 0.102 |  | 224 | 57 | 1.43e-24 |
| head:planck-rank vs lexical | 1200 | 0.241 | 0.144 |  | 206 | 90 | 1.28e-11 |
| head:planck vs gemma | 300 | 0.183 | 0.337 | 0.54 | 24 | 70 | 2.2e-06 |
| head:planck vs head:hash | 1200 | 0.223 | 0.102 |  | 213 | 67 | 6.81e-19 |
| head:planck vs lexical | 1200 | 0.223 | 0.144 |  | 193 | 98 | 2.72e-08 |
| head:planck-rank vs head:planck | 1200 | 0.241 | 0.223 |  | 126 | 105 | 0.188 |

**Verdict:** FAIL. head:planck-rank-vs-hash +0.139 (McNemar p=1.43e-24); paired head/teacher = 0.66 on 300 shared races (pass >= 0.8); warm decision 0.5 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.24, head:planck 0.54

## G1 across seeds (g1_aggregate: seeds [0, 1, 2])

| policy | mean rollout success | sd | 95% CI | per seed |
|---|---|---|---|---|
| gemma | 0.337 | 0.000 | [0.337, 0.337] | 0.337, 0.337, 0.337 |
| head:hash | 0.143 | 0.005 | [0.131, 0.155] | 0.146, 0.146, 0.138 |
| head:planck | 0.230 | 0.018 | [0.184, 0.276] | 0.211, 0.247, 0.231 |
| head:planck-rank | 0.273 | 0.004 | [0.264, 0.282] | 0.271, 0.277, 0.270 |
| lexical | 0.153 | 0.000 | [0.153, 0.153] | 0.153, 0.153, 0.153 |
| random | 0.001 | 0.000 | [0.001, 0.001] | 0.001, 0.001, 0.001 |

head:hash: paired ratio vs teacher 0.42 (per seed 0.49, 0.38, 0.40)

head:planck: paired ratio vs teacher 0.58 (per seed 0.52, 0.59, 0.63)

head:planck-rank: paired ratio vs teacher 0.78 (per seed 0.79, 0.84, 0.71)

**Verdict (seeds):** FAIL. mean paired head:planck/teacher = 0.58 over 3 seeds (pass >= 0.8); beats head:hash in every seed: True

## G1 across seeds (g1_aggregate_t1: seeds [3, 4, 5])

| policy | mean rollout success | sd | 95% CI | per seed |
|---|---|---|---|---|
| gemma | 0.337 | 0.000 | [0.337, 0.337] | 0.337, 0.337, 0.337 |
| head:hash | 0.107 | 0.005 | [0.095, 0.119] | 0.108, 0.111, 0.102 |
| head:planck | 0.218 | 0.009 | [0.195, 0.242] | 0.224, 0.207, 0.223 |
| head:planck-rank | 0.235 | 0.005 | [0.222, 0.249] | 0.235, 0.230, 0.241 |
| lexical | 0.144 | 0.000 | [0.144, 0.144] | 0.144, 0.144, 0.144 |
| random | 0.001 | 0.000 | [0.001, 0.001] | 0.001, 0.001, 0.001 |

head:hash: paired ratio vs teacher 0.25 (per seed 0.28, 0.24, 0.24)

head:planck: paired ratio vs teacher 0.63 (per seed 0.65, 0.68, 0.54)

head:planck-rank: paired ratio vs teacher 0.71 (per seed 0.67, 0.80, 0.66)

**Verdict (seeds):** FAIL. mean paired head:planck-rank/teacher = 0.71 over 3 seeds (pass >= 0.8); beats head:hash in every seed: True

