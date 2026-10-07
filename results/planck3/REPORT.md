# Planck 3.0 results (2026-10-07 12:31)

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

**g0_gemma vs base chat:** success 0.681 vs 0.957 (0.71x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct 79.8x MORE expensive (5486 vs 82 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_gemma_quick vs base chat:** success 0.667 vs 1.000 (0.67x); depth evidence 0.667 vs none (base chat shows no sources); cost per correct 87.7x MORE expensive (9337 vs 136 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_gemma_snip vs base chat:** success 0.851 vs 0.957 (0.89x); depth evidence 0.957 vs none (base chat shows no sources); cost per correct 48.9x MORE expensive (4206 vs 82 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.

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

