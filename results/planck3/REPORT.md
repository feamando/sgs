# Planck 3.0 results (2026-10-17 15:37)

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
| g0_planck-g2-hash | 47 | 0.021 | 0.021 | 0.000 | 0.507 | 0.851 | 0.553 | 2.617 | 0.000 | 0.000 | 0.000006 | 0 | G2 (rule B) |
| g0_planck-g2-hash_quick | 12 | 0.000 | 0.000 | 0.000 |  | 1.000 | 1.000 | 5.667 | 0.000 | 0.000 |  | 0 | G2 (rule B) |
| g0_planck-g2-hash_wikipedia | 47 | 0.000 | 0.021 | 1.000 | 0.843 | 0.851 | 0.574 | 2.681 | 0.000 | 0.000 |  | 0 | G2 (rule B) |
| g0_planck-g2-planck | 47 | 0.021 | 0.106 | 0.800 | 0.431 | 0.851 | 0.574 | 2.574 | 0.043 | 0.000 | 0.000128 | 0 | G2 (rule B) |
| g0_planck-g2-planck_quick | 12 | 0.000 | 0.000 | 0.000 |  | 1.000 | 1.000 | 5.667 | 0.000 | 0.000 |  | 0 | G2 (rule B) |
| g0_planck-g2-planck_wikipedia | 47 | 0.021 | 0.021 | 0.000 | 0.286 | 0.851 | 0.574 | 2.681 | 0.000 | 0.000 | 0.000098 | 0 | G2 (rule B) |
| g0_planck-g2v2-hash | 47 | 0.617 | 0.745 | 0.171 | 0.161 | 0.979 | 0.851 | 0.830 | 0.660 | 0.000 | 0.000000 | 0 | G2 (rule B) |
| g0_planck-g2v2-hash_ddgs | 47 | 0.426 | 0.489 | 0.130 | 0.109 | 0.979 | 0.830 | 1.191 | 0.404 | 0.000 | 0.000000 | 0 | G2 (rule B) |
| g0_planck-g2v2-hash_ddgs_quick | 12 | 0.000 | 0.333 | 1.000 | 0.739 | 1.000 | 0.667 | 1.750 | 0.250 | 0.000 |  | 0 | G2 (rule B) |
| g0_planck-g2v2-planck | 47 | 0.638 | 0.787 | 0.189 | 0.182 | 1.000 | 0.872 | 0.660 | 0.702 | 0.000 | 0.000002 | 0 | G2 (rule B) |
| g0_planck-g2v2-planck_ddgs | 47 | 0.298 | 0.340 | 0.125 | 0.109 | 0.979 | 0.872 | 1.596 | 0.298 | 0.000 | 0.000005 | 0 | G2 (rule B) |
| g0_planck-g2v2-planck_ddgs_quick | 12 | 0.000 | 0.000 | 0.000 |  | 1.000 | 0.917 | 4.083 | 0.000 | 0.000 |  | 0 | G2 (rule B) |
| g0f_gemma_closedbook | 143 | 0.049 | 0.986 | 0.950 | 0.950 |  |  | 0.000 |  |  | 0.002297 | 75 | COMPARATOR |
| g0f_gemma_closedbook_quick | 12 | 0.000 | 1.000 | 1.000 | 1.000 |  |  | 0.000 |  |  |  | 110 | COMPARATOR |
| g0f_gemma_snip | 143 | 0.196 | 0.294 | 0.333 | 0.257 | 0.322 | 0.273 | 0.042 | 0.294 | 0.014 | 0.015155 | 2737 | ROUND 3 (rules A/B) |
| g0f_gemma_snip_ddgs | 143 | 0.601 | 0.797 | 0.246 | 0.177 | 0.839 | 0.846 | 0.063 | 0.797 | 0.007 | 0.006522 | 3635 | ROUND 4 (rules S/P) |
| g0f_gemma_snip_ddgs_quick | 12 | 0.833 | 0.917 | 0.091 | 0.036 | 0.917 | 0.917 | 0.083 | 0.917 | 0.000 | 0.005969 | 4576 | ROUND 4 (rules S/P) |
| g0f_gemma_snip_quick | 12 | 0.667 | 1.000 | 0.333 | 0.267 | 0.750 | 0.917 | 0.000 | 1.000 | 0.000 | 0.007831 | 4830 | ROUND 3 (rules A/B) |
| g0f_gemma_snip_wikipedia | 143 | 0.140 | 0.210 | 0.333 | 0.253 | 0.252 | 0.189 | 0.056 | 0.210 | 0.014 | 0.020229 | 2607 | ROUND 3 (rules A/B) |
| g0f_heuristic_snip | 143 | 0.210 | 0.818 | 0.744 | 0.471 | 0.336 | 0.273 | 0.699 | 0.308 | 0.000 | 0.000000 | 0 | ROUND 3 (rules A/B) |
| g0f_heuristic_snip_ddgs | 143 | 0.510 | 0.958 | 0.467 | 0.319 | 0.734 | 0.734 | 0.133 | 0.860 | 0.000 | 0.000000 | 0 | ROUND 4 (rules S/P) |
| g0f_heuristic_snip_ddgs_quick | 12 | 0.333 | 1.000 | 0.667 | 0.414 | 0.667 | 0.667 | 0.250 | 0.750 | 0.000 | 0.000000 | 0 | ROUND 4 (rules S/P) |
| g0f_heuristic_snip_ddgs_trust | 143 | 0.566 | 1.000 | 0.434 | 0.297 | 0.832 | 0.846 | 0.077 | 0.944 | 0.000 | 0.000000 | 0 | ROUND 4 (rules S/P) |
| g0f_heuristic_snip_ddgs_trust_quick | 12 | 0.583 | 1.000 | 0.417 | 0.300 | 0.917 | 0.917 | 0.083 | 0.917 | 0.000 | 0.000000 | 0 | ROUND 4 (rules S/P) |
| g0f_heuristic_snip_quick | 12 | 0.667 | 1.000 | 0.333 | 0.302 | 0.750 | 0.917 | 0.250 | 0.833 | 0.000 | 0.000000 | 0 | ROUND 3 (rules A/B) |
| g0f_heuristic_snip_wikipedia | 143 | 0.154 | 0.797 | 0.807 | 0.510 | 0.273 | 0.189 | 0.755 | 0.259 | 0.000 | 0.000000 | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-hash | 143 | 0.021 | 0.063 | 0.667 | 0.358 | 0.287 | 0.189 | 1.951 | 0.028 | 0.000 | 0.000005 | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-hash_quick | 12 | 0.000 | 0.000 | 0.000 |  | 0.750 | 0.917 | 3.250 | 0.000 | 0.000 |  | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-hash_wikipedia | 143 | 0.007 | 0.042 | 0.833 | 0.377 | 0.287 | 0.189 | 1.958 | 0.014 | 0.000 | 0.000016 | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-planck | 143 | 0.091 | 0.189 | 0.519 | 0.287 | 0.287 | 0.189 | 1.706 | 0.133 | 0.000 | 0.000020 | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-planck_quick | 12 | 0.000 | 0.000 | 0.000 |  | 0.750 | 0.917 | 3.250 | 0.000 | 0.000 |  | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2-planck_wikipedia | 143 | 0.098 | 0.182 | 0.462 | 0.241 | 0.287 | 0.189 | 1.734 | 0.112 | 0.000 | 0.000015 | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2v2-hash | 143 | 0.035 | 0.056 | 0.375 | 0.345 | 0.343 | 0.273 | 1.916 | 0.042 | 0.000 | 0.000003 | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2v2-hash_ddgs | 143 | 0.245 | 0.294 | 0.167 | 0.102 | 0.839 | 0.853 | 1.580 | 0.217 | 0.000 | 0.000000 | 0 | ROUND 4 (rules S/P) |
| g0f_planck-g2v2-hash_ddgs_quick | 12 | 0.167 | 0.500 | 0.667 | 0.382 | 0.917 | 0.917 | 2.083 | 0.250 | 0.000 | 0.000001 | 0 | ROUND 4 (rules S/P) |
| g0f_planck-g2v2-planck | 143 | 0.084 | 0.119 | 0.294 | 0.094 | 0.343 | 0.273 | 1.783 | 0.084 | 0.000 | 0.000024 | 0 | ROUND 3 (rules A/B) |
| g0f_planck-g2v2-planck_ddgs | 143 | 0.294 | 0.322 | 0.087 | 0.022 | 0.839 | 0.853 | 1.580 | 0.245 | 0.000 | 0.000004 | 0 | ROUND 4 (rules S/P) |
| g0f_planck-g2v2-planck_ddgs_quick | 12 | 0.083 | 0.167 | 0.500 | 0.351 | 0.917 | 0.917 | 3.333 | 0.167 | 0.000 | 0.000027 | 0 | ROUND 4 (rules S/P) |

**g0_gemma vs base chat:** success 0.681 vs 0.957 (0.71x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct 79.8x MORE expensive (5486 vs 82 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_gemma_quick vs base chat:** success 0.667 vs 1.000 (0.67x); depth evidence 0.667 vs none (base chat shows no sources); cost per correct 87.7x MORE expensive (9337 vs 136 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_gemma_snip vs base chat:** success 0.851 vs 0.957 (0.89x); depth evidence 0.957 vs none (base chat shows no sources); cost per correct 48.9x MORE expensive (4206 vs 82 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0_heuristic vs base chat:** success 0.617 vs 0.957 (0.64x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct 8567.8x cheaper.
**g0_heuristic_quick vs base chat:** success 0.250 vs 1.000 (0.25x); depth evidence 0.667 vs none (base chat shows no sources); cost per correct 3112.9x cheaper.
**g0_heuristic_snip vs base chat:** success 0.936 vs 0.957 (0.98x); depth evidence 0.957 vs none (base chat shows no sources); cost per correct 113295.6x cheaper.
**g0_planck-g2-hash vs base chat:** success 0.021 vs 0.957 (0.02x); depth evidence 0.851 vs none (base chat shows no sources); cost per correct 17.2x cheaper.
**g0_planck-g2-hash_quick vs base chat:** success 0.000 vs 1.000 (0.00x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct n/a.
**g0_planck-g2-hash_wikipedia vs base chat:** success 0.000 vs 0.957 (0.00x); depth evidence 0.851 vs none (base chat shows no sources); cost per correct n/a.
**g0_planck-g2-planck vs base chat:** success 0.021 vs 0.957 (0.02x); depth evidence 0.851 vs none (base chat shows no sources); cost per correct 1.2x MORE expensive (0 vs 82 LLM tokens/task).
**g0_planck-g2-planck_quick vs base chat:** success 0.000 vs 1.000 (0.00x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct n/a.
**g0_planck-g2-planck_wikipedia vs base chat:** success 0.021 vs 0.957 (0.02x); depth evidence 0.851 vs none (base chat shows no sources); cost per correct 1.1x cheaper.
**g0_planck-g2v2-hash vs base chat:** success 0.617 vs 0.957 (0.64x); depth evidence 0.979 vs none (base chat shows no sources); cost per correct 1282.3x cheaper.
**g0_planck-g2v2-hash_ddgs vs base chat:** success 0.426 vs 0.957 (0.44x); depth evidence 0.979 vs none (base chat shows no sources); cost per correct 564.9x cheaper.
**g0_planck-g2v2-hash_ddgs_quick vs base chat:** success 0.000 vs 1.000 (0.00x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct n/a.
**g0_planck-g2v2-planck vs base chat:** success 0.638 vs 0.957 (0.67x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct 72.8x cheaper.
**g0_planck-g2v2-planck_ddgs vs base chat:** success 0.298 vs 0.957 (0.31x); depth evidence 0.979 vs none (base chat shows no sources); cost per correct 23.8x cheaper.
**g0_planck-g2v2-planck_ddgs_quick vs base chat:** success 0.000 vs 1.000 (0.00x); depth evidence 1.000 vs none (base chat shows no sources); cost per correct n/a.
**g0f_gemma_snip vs base chat:** success 0.196 vs 0.049 (4.00x); depth evidence 0.322 vs none (base chat shows no sources); cost per correct 6.6x MORE expensive (2737 vs 75 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0f_gemma_snip_ddgs vs base chat:** success 0.601 vs 0.049 (12.29x); depth evidence 0.839 vs none (base chat shows no sources); cost per correct 2.8x MORE expensive (3635 vs 75 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0f_gemma_snip_ddgs_quick vs base chat:** success 0.833 vs 0.000 (833333333.33x); depth evidence 0.917 vs none (base chat shows no sources); cost per correct n/a. An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0f_gemma_snip_quick vs base chat:** success 0.667 vs 0.000 (666666666.67x); depth evidence 0.750 vs none (base chat shows no sources); cost per correct n/a. An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0f_gemma_snip_wikipedia vs base chat:** success 0.140 vs 0.049 (2.86x); depth evidence 0.252 vs none (base chat shows no sources); cost per correct 8.8x MORE expensive (2607 vs 75 LLM tokens/task). An LLM driving the tools is the expensive path by design; the cheap path is the distilled Planck policy (G2), which this run does not include.
**g0f_heuristic_snip vs base chat:** success 0.210 vs 0.049 (4.29x); depth evidence 0.336 vs none (base chat shows no sources); cost per correct 136268.0x cheaper.
**g0f_heuristic_snip_ddgs vs base chat:** success 0.510 vs 0.049 (10.43x); depth evidence 0.734 vs none (base chat shows no sources); cost per correct 530110.6x cheaper.
**g0f_heuristic_snip_ddgs_quick vs base chat:** success 0.333 vs 0.000 (333333333.33x); depth evidence 0.667 vs none (base chat shows no sources); cost per correct n/a.
**g0f_heuristic_snip_ddgs_trust vs base chat:** success 0.566 vs 0.049 (11.57x); depth evidence 0.832 vs none (base chat shows no sources); cost per correct 915042.4x cheaper.
**g0f_heuristic_snip_ddgs_trust_quick vs base chat:** success 0.583 vs 0.000 (583333333.33x); depth evidence 0.917 vs none (base chat shows no sources); cost per correct n/a.
**g0f_heuristic_snip_quick vs base chat:** success 0.667 vs 0.000 (666666666.67x); depth evidence 0.750 vs none (base chat shows no sources); cost per correct n/a.
**g0f_heuristic_snip_wikipedia vs base chat:** success 0.154 vs 0.049 (3.14x); depth evidence 0.273 vs none (base chat shows no sources); cost per correct 69225.3x cheaper.
**g0f_planck-g2-hash vs base chat:** success 0.021 vs 0.049 (0.43x); depth evidence 0.287 vs none (base chat shows no sources); cost per correct 440.8x cheaper.
**g0f_planck-g2-hash_quick vs base chat:** success 0.000 vs 0.000 (0.00x); depth evidence 0.750 vs none (base chat shows no sources); cost per correct n/a.
**g0f_planck-g2-hash_wikipedia vs base chat:** success 0.007 vs 0.049 (0.14x); depth evidence 0.287 vs none (base chat shows no sources); cost per correct 146.9x cheaper.
**g0f_planck-g2-planck vs base chat:** success 0.091 vs 0.049 (1.86x); depth evidence 0.287 vs none (base chat shows no sources); cost per correct 112.8x cheaper.
**g0f_planck-g2-planck_quick vs base chat:** success 0.000 vs 0.000 (0.00x); depth evidence 0.750 vs none (base chat shows no sources); cost per correct n/a.
**g0f_planck-g2-planck_wikipedia vs base chat:** success 0.098 vs 0.049 (2.00x); depth evidence 0.287 vs none (base chat shows no sources); cost per correct 151.3x cheaper.
**g0f_planck-g2v2-hash vs base chat:** success 0.035 vs 0.049 (0.71x); depth evidence 0.343 vs none (base chat shows no sources); cost per correct 738.7x cheaper.
**g0f_planck-g2v2-hash_ddgs vs base chat:** success 0.245 vs 0.049 (5.00x); depth evidence 0.839 vs none (base chat shows no sources); cost per correct 6108.9x cheaper.
**g0f_planck-g2v2-hash_ddgs_quick vs base chat:** success 0.167 vs 0.000 (166666666.67x); depth evidence 0.917 vs none (base chat shows no sources); cost per correct n/a.
**g0f_planck-g2v2-planck vs base chat:** success 0.084 vs 0.049 (1.71x); depth evidence 0.343 vs none (base chat shows no sources); cost per correct 94.2x cheaper.
**g0f_planck-g2v2-planck_ddgs vs base chat:** success 0.294 vs 0.049 (6.00x); depth evidence 0.839 vs none (base chat shows no sources); cost per correct 537.5x cheaper.
**g0f_planck-g2v2-planck_ddgs_quick vs base chat:** success 0.083 vs 0.000 (83333333.33x); depth evidence 0.917 vs none (base chat shows no sources); cost per correct n/a.

### Round 4 paired comparisons (same questions; exact McNemar)

| rule | run | vs | metric | run | vs | only run | only vs | p |
|---|---|---|---|---|---|---|---|---|
| v2 ddgs head vs 3b head | g0_planck-g2v2-hash_ddgs | g0_planck-g2v2-hash | correct | 0.426 | 0.617 | 2 | 11 | 0.0225 |
| v2 ddgs head vs 3b head | g0_planck-g2v2-planck_ddgs | g0_planck-g2v2-planck | correct | 0.298 | 0.638 | 4 | 20 | 0.00154 |
| S2 | g0f_gemma_snip_ddgs | g0f_gemma_snip_wikipedia | serp_visible | 0.846 | 0.189 | 94 | 0 | 1.01e-28 |
| S2 | g0f_gemma_snip_ddgs | g0f_gemma_snip_wikipedia | correct | 0.601 | 0.140 | 70 | 4 | 1.29e-16 |
| A1 | g0f_gemma_snip_ddgs | g0f_gemma_closedbook | correct | 0.601 | 0.049 | 83 | 4 | 3.02e-20 |
| S2 | g0f_heuristic_snip_ddgs | g0f_heuristic_snip_wikipedia | serp_visible | 0.734 | 0.189 | 78 | 0 | 6.62e-24 |
| S2 | g0f_heuristic_snip_ddgs | g0f_heuristic_snip_wikipedia | correct | 0.510 | 0.154 | 53 | 2 | 8.55e-14 |
| T | g0f_heuristic_snip_ddgs_trust | g0f_heuristic_snip_ddgs | correct | 0.566 | 0.510 | 9 | 1 | 0.0215 |
| T (both web-served) | g0f_heuristic_snip_ddgs_trust | g0f_heuristic_snip_ddgs | correct | 0.575 | 0.575 | 0 | 0 | 1 |
| v2 ddgs head vs 3b head | g0f_planck-g2v2-hash_ddgs | g0f_planck-g2v2-hash | correct | 0.245 | 0.035 | 31 | 1 | 1.54e-08 |
| v2 ddgs head vs 3b head | g0f_planck-g2v2-planck_ddgs | g0f_planck-g2v2-planck | correct | 0.294 | 0.084 | 32 | 2 | 6.94e-08 |

### Answer tiers (what the card shows: confident / low confidence / evidence only)

| run | confident: share | precision | low confidence: share | precision | none: share | its evidence has the answer |
|---|---|---|---|---|---|---|
| g0_planck-g2v2-hash_ddgs | 0.489 | 0.870 | 0.511 | 0.500 | 0.000 |  |
| g0_planck-g2v2-hash_ddgs_quick | 0.333 | 0.000 | 0.667 | 0.250 | 0.000 |  |
| g0_planck-g2v2-planck_ddgs | 0.340 | 0.875 | 0.660 | 0.710 | 0.000 |  |
| g0_planck-g2v2-planck_ddgs_quick | 0.000 |  | 1.000 | 0.083 | 0.000 |  |
| g0f_gemma_snip_ddgs | 0.797 | 0.754 | 0.203 | 0.379 | 0.000 |  |
| g0f_gemma_snip_ddgs_quick | 0.917 | 0.909 | 0.083 | 1.000 | 0.000 |  |
| g0f_heuristic_snip_ddgs | 0.909 | 0.562 | 0.084 | 0.000 | 0.007 | 0.000 |
| g0f_heuristic_snip_ddgs_quick | 0.833 | 0.400 | 0.167 | 0.000 | 0.000 |  |
| g0f_heuristic_snip_ddgs_trust | 0.979 | 0.571 | 0.021 | 0.333 | 0.000 |  |
| g0f_heuristic_snip_ddgs_trust_quick | 1.000 | 0.583 | 0.000 |  | 0.000 |  |
| g0f_planck-g2v2-hash_ddgs | 0.294 | 0.833 | 0.706 | 0.287 | 0.000 |  |
| g0f_planck-g2v2-hash_ddgs_quick | 0.333 | 0.250 | 0.667 | 0.250 | 0.000 |  |
| g0f_planck-g2v2-planck_ddgs | 0.322 | 0.913 | 0.678 | 0.495 | 0.000 |  |
| g0f_planck-g2v2-planck_ddgs_quick | 0.167 | 0.500 | 0.833 | 0.400 | 0.000 |  |

### By regime (fresh = 2026 facts; long_tail = <= 3 Wikipedia editions)

| run | per regime: success (n, wrong when answered) |
|---|---|
| g0f_gemma_closedbook | fresh: 0.000 (n=60, wrong 1.000) | long_tail: 0.084 (n=83, wrong 0.915) |
| g0f_gemma_closedbook_quick | fresh: 0.000 (n=6, wrong 1.000) | long_tail: 0.000 (n=6, wrong 1.000) |
| g0f_gemma_snip | fresh: 0.083 (n=60, wrong 0.545) | long_tail: 0.277 (n=83, wrong 0.258) |
| g0f_gemma_snip_ddgs | fresh: 0.567 (n=60, wrong 0.244) | long_tail: 0.627 (n=83, wrong 0.246) |
| g0f_gemma_snip_ddgs_quick | fresh: 1.000 (n=6, wrong 0.000) | long_tail: 0.667 (n=6, wrong 0.200) |
| g0f_gemma_snip_quick | fresh: 0.500 (n=6, wrong 0.500) | long_tail: 0.833 (n=6, wrong 0.167) |
| g0f_gemma_snip_wikipedia | fresh: 0.033 (n=60, wrong 0.600) | long_tail: 0.217 (n=83, wrong 0.280) |
| g0f_heuristic_snip | fresh: 0.133 (n=60, wrong 0.855) | long_tail: 0.265 (n=83, wrong 0.645) |
| g0f_heuristic_snip_ddgs | fresh: 0.250 (n=60, wrong 0.737) | long_tail: 0.699 (n=83, wrong 0.275) |
| g0f_heuristic_snip_ddgs_quick | fresh: 0.167 (n=6, wrong 0.833) | long_tail: 0.500 (n=6, wrong 0.500) |
| g0f_heuristic_snip_ddgs_trust | fresh: 0.333 (n=60, wrong 0.667) | long_tail: 0.735 (n=83, wrong 0.265) |
| g0f_heuristic_snip_ddgs_trust_quick | fresh: 0.333 (n=6, wrong 0.667) | long_tail: 0.833 (n=6, wrong 0.167) |
| g0f_heuristic_snip_quick | fresh: 0.500 (n=6, wrong 0.500) | long_tail: 0.833 (n=6, wrong 0.167) |
| g0f_heuristic_snip_wikipedia | fresh: 0.083 (n=60, wrong 0.909) | long_tail: 0.205 (n=83, wrong 0.712) |
| g0f_planck-g2-hash | fresh: 0.000 (n=60, wrong 0.000) | long_tail: 0.036 (n=83, wrong 0.667) |
| g0f_planck-g2-hash_quick | fresh: 0.000 (n=6, wrong 0.000) | long_tail: 0.000 (n=6, wrong 0.000) |
| g0f_planck-g2-hash_wikipedia | fresh: 0.000 (n=60, wrong 1.000) | long_tail: 0.012 (n=83, wrong 0.800) |
| g0f_planck-g2-planck | fresh: 0.000 (n=60, wrong 1.000) | long_tail: 0.157 (n=83, wrong 0.480) |
| g0f_planck-g2-planck_quick | fresh: 0.000 (n=6, wrong 0.000) | long_tail: 0.000 (n=6, wrong 0.000) |
| g0f_planck-g2-planck_wikipedia | fresh: 0.017 (n=60, wrong 0.900) | long_tail: 0.157 (n=83, wrong 0.188) |
| g0f_planck-g2v2-hash | fresh: 0.000 (n=60, wrong 1.000) | long_tail: 0.060 (n=83, wrong 0.167) |
| g0f_planck-g2v2-hash_ddgs | fresh: 0.000 (n=60, wrong 0.000) | long_tail: 0.422 (n=83, wrong 0.167) |
| g0f_planck-g2v2-hash_ddgs_quick | fresh: 0.333 (n=6, wrong 0.000) | long_tail: 0.000 (n=6, wrong 1.000) |
| g0f_planck-g2v2-planck | fresh: 0.017 (n=60, wrong 0.500) | long_tail: 0.133 (n=83, wrong 0.267) |
| g0f_planck-g2v2-planck_ddgs | fresh: 0.000 (n=60, wrong 0.000) | long_tail: 0.506 (n=83, wrong 0.087) |
| g0f_planck-g2v2-planck_ddgs_quick | fresh: 0.167 (n=6, wrong 0.500) | long_tail: 0.000 (n=6, wrong 0.000) |

## Round 5: read -> weigh -> write (SETUP_planck_20261009.md)

| run | n | value shown is right | high: n / precision | good | low | sceptical | none | monotone | writer faithful | fallback | LLM tok/task | read ms | write ms | changed by personal | question type acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r5_gemma_gemma | 243 | 0.531 | 41 / 0.927 | 92 / 0.772 | 25 / 0.560 | 12 / 0.500 | 73 | True | 0.371 | 0.629 | 1898 | 2663 | 1977 | 0.000 | 0.979 |
| r5_gemma_gemma_quick | 12 | 0.750 | 4 / 1.000 | 6 / 0.667 | 1 / 1.000 | 0 /  | 1 | True | 0.333 | 0.667 | 1811 | 2160 | 2233 | 0.000 | 1.000 |
| r5_heuristic_template | 243 | 0.391 | 27 / 0.815 | 70 / 0.614 | 61 / 0.295 | 59 / 0.203 | 26 | True | 1.000 | 0.000 | 0 | 0 | 0 | 0.000 | 0.979 |
| r5_heuristic_template_quick | 12 | 0.500 | 1 / 1.000 | 4 / 0.750 | 4 / 0.250 | 2 / 0.500 | 1 | True | 1.000 | 0.000 | 0 | 0 | 0 | 0.000 | 1.000 |
| r5_planck_hertz | 243 | 0.387 | 28 / 0.786 | 74 / 0.581 | 59 / 0.339 | 81 / 0.111 | 1 | True | 0.460 | 0.540 | 0 | 17 | 267 | 0.000 | 0.979 |
| r5_planck_hertz_brave | 243 | 0.514 | 39 / 0.641 | 66 / 0.621 | 71 / 0.535 | 66 / 0.318 | 1 | True | 0.540 | 0.460 | 0 | 22 | 344 | 0.000 | 0.979 |
| r5_planck_hertz_brave_quick | 12 | 0.667 | 2 / 1.000 | 2 / 0.500 | 7 / 0.714 | 1 / 0.000 | 0 | True | 0.385 | 0.615 | 0 | 55 | 329 | 0.000 | 1.000 |
| r5_planck_hertz_contrarian | 243 | 0.391 | 0 /  | 4 / 0.750 | 96 / 0.635 | 142 / 0.218 | 1 | True | 0.480 | 0.520 | 0 | 23 | 251 | 0.473 | 0.979 |
| r5_planck_hertz_contrarian_quick | 12 | 0.583 | 0 /  | 0 /  | 5 / 1.000 | 7 / 0.286 | 0 | True | 0.538 | 0.462 | 0 | 63 | 333 | 0.500 | 1.000 |
| r5_planck_hertz_cpu | 10 | 0.300 | 2 / 0.500 | 2 / 0.500 | 3 / 0.333 | 3 / 0.000 | 0 | True | 0.500 | 0.500 | 0 | 934 | 27765 | 0.000 | 0.900 |
| r5_planck_hertz_cpu_quick | 12 | 0.500 | 0 /  | 5 / 0.600 | 5 / 0.600 | 2 / 0.000 | 0 | True | 0.231 | 0.769 | 0 | 499 | 20399 | 0.000 | 1.000 |
| r5_planck_hertz_quick | 12 | 0.583 | 0 /  | 5 / 0.800 | 5 / 0.600 | 2 / 0.000 | 0 | True | 0.308 | 0.692 | 0 | 44 | 294 | 0.000 | 1.000 |
| r5_planck_planck | 243 | 0.391 | 29 / 0.793 | 73 / 0.589 | 59 / 0.339 | 81 / 0.111 | 1 | True | 0.472 | 0.528 | 0 | 16 | 99 | 0.000 | 0.979 |
| r5_planck_planck_quick | 12 | 0.583 | 0 /  | 5 / 0.800 | 5 / 0.600 | 2 / 0.000 | 0 | True | 0.385 | 0.615 | 0 | 36 | 83 | 0.000 | 1.000 |
| r5_planck_template | 243 | 0.383 | 28 / 0.786 | 73 / 0.575 | 59 / 0.339 | 82 / 0.110 | 1 | True | 1.000 | 0.000 | 0 | 68 | 0 | 0.000 | 0.979 |
| r5_planck_template_quick | 12 | 0.583 | 0 /  | 5 / 0.800 | 5 / 0.600 | 2 / 0.000 | 0 | True | 1.000 | 0.000 | 0 | 53 | 0 | 0.000 | 1.000 |

| run | accuracy per benchmark regime |
|---|---|
| r5_gemma_gemma | fresh: 0.750 (n=60) | general_info: 0.250 (n=40) | long_tail: 0.819 (n=83) | news: 0.100 (n=60) |
| r5_gemma_gemma_quick | fresh: 1.000 (n=3) | general_info: 1.000 (n=1) | long_tail: 1.000 (n=5) | news: 0.000 (n=3) |
| r5_heuristic_template | fresh: 0.350 (n=60) | general_info: 0.225 (n=40) | long_tail: 0.723 (n=83) | news: 0.083 (n=60) |
| r5_heuristic_template_quick | fresh: 0.667 (n=3) | general_info: 1.000 (n=1) | long_tail: 0.600 (n=5) | news: 0.000 (n=3) |
| r5_planck_hertz | fresh: 0.350 (n=60) | general_info: 0.175 (n=40) | long_tail: 0.759 (n=83) | news: 0.050 (n=60) |
| r5_planck_hertz_brave | fresh: 0.400 (n=60) | general_info: 0.550 (n=40) | long_tail: 0.663 (n=83) | news: 0.400 (n=60) |
| r5_planck_hertz_brave_quick | fresh: 0.000 (n=3) | general_info: 1.000 (n=1) | long_tail: 1.000 (n=5) | news: 0.667 (n=3) |
| r5_planck_hertz_contrarian | fresh: 0.333 (n=60) | general_info: 0.200 (n=40) | long_tail: 0.771 (n=83) | news: 0.050 (n=60) |
| r5_planck_hertz_contrarian_quick | fresh: 0.333 (n=3) | general_info: 1.000 (n=1) | long_tail: 1.000 (n=5) | news: 0.000 (n=3) |
| r5_planck_hertz_cpu | fresh: 0.000 (n=3) | general_info: 0.000 (n=1) | long_tail: 1.000 (n=3) | news: 0.000 (n=3) |
| r5_planck_hertz_cpu_quick | fresh: 0.667 (n=3) | general_info: 1.000 (n=1) | long_tail: 0.600 (n=5) | news: 0.000 (n=3) |
| r5_planck_hertz_quick | fresh: 0.667 (n=3) | general_info: 1.000 (n=1) | long_tail: 0.800 (n=5) | news: 0.000 (n=3) |
| r5_planck_planck | fresh: 0.350 (n=60) | general_info: 0.200 (n=40) | long_tail: 0.759 (n=83) | news: 0.050 (n=60) |
| r5_planck_planck_quick | fresh: 0.667 (n=3) | general_info: 1.000 (n=1) | long_tail: 0.800 (n=5) | news: 0.000 (n=3) |
| r5_planck_template | fresh: 0.350 (n=60) | general_info: 0.175 (n=40) | long_tail: 0.759 (n=83) | news: 0.033 (n=60) |
| r5_planck_template_quick | fresh: 0.667 (n=3) | general_info: 1.000 (n=1) | long_tail: 0.800 (n=5) | news: 0.000 (n=3) |

| paired (same questions) | n | run | vs | only run right | only vs right | McNemar p |
|---|---|---|---|---|---|---|
| r5_planck_hertz vs r5_gemma_gemma | 243 | 0.387 | 0.531 | 11 | 46 | 3.31e-06 |
| r5_planck_hertz vs r5_heuristic_template | 243 | 0.387 | 0.391 | 24 | 25 | 1 |
| r5_planck_hertz vs r5_planck_planck | 243 | 0.387 | 0.391 | 0 | 1 | 1 |
| r5_planck_hertz vs r5_planck_template | 243 | 0.387 | 0.383 | 1 | 0 | 1 |
| r5_planck_hertz_brave vs r5_planck_hertz | 243 | 0.514 | 0.387 | 54 | 23 | 0.000539 |
| r5_planck_hertz_brave_quick vs r5_planck_hertz | 12 | 0.667 | 0.583 | 3 | 2 | 1 |
| r5_planck_hertz_contrarian vs r5_planck_hertz | 243 | 0.391 | 0.387 | 6 | 5 | 1 |
| r5_planck_hertz_contrarian_quick vs r5_planck_hertz | 12 | 0.583 | 0.583 | 1 | 1 | 1 |
| r5_planck_hertz_cpu vs r5_planck_hertz | 10 | 0.300 | 0.300 | 0 | 0 | 1 |
| r5_planck_hertz_cpu_quick vs r5_planck_hertz | 12 | 0.500 | 0.583 | 0 | 1 | 1 |
| r5_planck_hertz_quick vs r5_planck_hertz | 12 | 0.583 | 0.583 | 0 | 0 | 1 |

- writer_hertz: 410 training examples (369 train / 41 val), best val loss 0.963, 641M params
- writer_planck: 410 training examples (369 train / 41 val), best val loss 0.988, 101M params

## Round 6: the reader trained once (cross-encoder) (SETUP_planck_20261012.md)

_v2 = round 6b, trained on the top 16 candidates as served (round 6 swapped a deep right candidate into the last slot); _rc = round 7 stage A (public reading sets only); _r7 = round 7 (stage A, then our pools); _p15rc = round 8, Planck 1.5 (Planck's size, Hertz's data) after stage A

| reader | params | train / val / test pools | right only beyond top 16 (test) | test choice acc | ranker | gate AUC | gate ECE | by group (choice vs ranker) |
|---|---|---|---|---|---|---|---|---|
| x-planck | 101M | 5310 / 592 / 654 |  | 0.538 | 0.401 | 0.753 | 0.059 | stable 0.655 vs 0.460 (n=226); recent 0.398 vs 0.229 (n=83); news 0.388 vs 0.418 (n=67); info 0.455 vs 0.400 (n=55) |
| x-hertz | 641M | 5310 / 592 / 654 |  | 0.568 | 0.401 | 0.789 | 0.05 | stable 0.699 vs 0.460 (n=226); recent 0.458 vs 0.229 (n=83); news 0.328 vs 0.418 (n=67); info 0.491 vs 0.400 (n=55) |
| x-planck_v2 | 101M | 5310 / 592 / 654 | 32 | 0.446 | 0.434 | 0.704 | 0.047 | stable 0.512 vs 0.479 (n=217); recent 0.324 vs 0.268 (n=71); news 0.407 vs 0.475 (n=59); info 0.385 vs 0.423 (n=52) |
| x-hertz_v2 | 641M | 5310 / 592 / 654 | 32 | 0.609 | 0.434 | 0.804 | 0.052 | stable 0.737 vs 0.479 (n=217); recent 0.423 vs 0.268 (n=71); news 0.458 vs 0.475 (n=59); info 0.500 vs 0.423 (n=52) |
| x-planck_rc | 101M | 137581 / 6000 / 6000 | 229 | 0.487 | 0.474 | 0.647 | 0.025 | squad 0.560 vs 0.601 (n=813); newsqa 0.464 vs 0.477 (n=859); triviaqa 0.508 vs 0.422 (n=417); searchqa 0.522 vs 0.462 (n=833); nq 0.399 vs 0.389 (n=884); squad2neg 0.000 vs 0.000 (n=0) |
| x-hertz_rc | 641M | 137581 / 6000 / 6000 | 229 | 0.643 | 0.474 | 0.778 | 0.025 | squad 0.654 vs 0.601 (n=813); newsqa 0.580 vs 0.477 (n=859); triviaqa 0.717 vs 0.422 (n=417); searchqa 0.677 vs 0.462 (n=833); nq 0.628 vs 0.389 (n=884); squad2neg 0.000 vs 0.000 (n=0) |
| x-planck_r7 | 101M | 5310 / 592 / 654 | 32 | 0.456 | 0.434 | 0.687 | 0.064 | stable 0.525 vs 0.479 (n=217); recent 0.338 vs 0.268 (n=71); news 0.407 vs 0.475 (n=59); info 0.385 vs 0.423 (n=52) |
| x-hertz_r7 | 641M | 5310 / 592 / 654 | 32 | 0.667 | 0.434 | 0.79 | 0.047 | stable 0.710 vs 0.479 (n=217); recent 0.577 vs 0.268 (n=71); news 0.593 vs 0.475 (n=59); info 0.692 vs 0.423 (n=52) |

## Round 6: read -> weigh -> write on Brave, relevance-checked (SETUP_planck_20261012.md)

| run | n | value shown is right | 143 fresh + long-tail | 100 news + info | high: n / precision | good | low | sceptical | none | monotone | search on topic | writer faithful (any try) | first try | tries | read ms | write ms | changed by personal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r6_gemma_gemma | 243 | 0.634 | 0.657 (n=143) | 0.600 (n=100) | 44 / 0.705 | 128 / 0.695 | 38 / 0.632 | 16 / 0.625 | 17 | True | 0.984 | 0.957 | 0.957 | 1.000 | 2566 | 2747 | 0.000 |
| r6_gemma_gemma_quick | 12 | 0.667 | 0.750 (n=8) | 0.500 (n=4) | 3 / 1.000 | 6 / 0.500 | 2 / 1.000 | 0 /  | 1 | True | 1.000 | 1.000 | 1.000 | 1.000 | 1873 | 2666 | 0.000 |
| r6_gemma_template | 243 | 0.634 | 0.657 (n=143) | 0.600 (n=100) | 44 / 0.705 | 128 / 0.695 | 38 / 0.632 | 16 / 0.625 | 17 | True | 0.984 | 1.000 |  |  | 2874 | 0 | 0.000 |
| r6_gemma_template_quick | 12 | 0.667 | 0.750 (n=8) | 0.500 (n=4) | 3 / 1.000 | 6 / 0.500 | 2 / 1.000 | 0 /  | 1 | True | 1.000 | 1.000 |  |  | 2006 | 0 | 0.000 |
| r6_heuristic_template | 243 | 0.498 | 0.413 (n=143) | 0.620 (n=100) | 33 / 0.758 | 72 / 0.667 | 80 / 0.375 | 57 / 0.316 | 1 | True | 0.984 | 1.000 |  |  | 0 | 0 | 0.000 |
| r6_heuristic_template_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 3 / 1.000 | 3 / 0.333 | 5 / 0.400 | 1 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 0 | 0 | 0.000 |
| r6_planck_template | 243 | 0.514 | 0.552 (n=143) | 0.460 (n=100) | 39 / 0.641 | 66 / 0.621 | 71 / 0.535 | 66 / 0.318 | 1 | True | 0.984 | 1.000 |  |  | 39 | 0 | 0.000 |
| r6_planck_template_quick | 12 | 0.667 | 0.625 (n=8) | 0.750 (n=4) | 2 / 1.000 | 2 / 0.500 | 7 / 0.714 | 1 / 0.000 | 0 | True | 1.000 | 1.000 |  |  | 51 | 0 | 0.000 |
| r6_x-hertz_hertz | 243 | 0.391 | 0.497 (n=143) | 0.240 (n=100) | 12 / 0.750 | 67 / 0.687 | 82 / 0.415 | 28 / 0.214 | 54 | True | 0.984 | 0.858 | 0.690 | 1.492 | 46 | 169 | 0.000 |
| r6_x-hertz_hertz_contrarian_quick | 12 | 0.167 | 0.250 (n=8) | 0.000 (n=4) | 0 /  | 0 /  | 0 /  | 9 / 0.222 | 3 | True | 1.000 | 0.400 | 0.300 | 2.300 | 89 | 258 | 0.750 |
| r6_x-hertz_hertz_cpu_quick | 12 | 0.167 | 0.250 (n=8) | 0.000 (n=4) | 0 /  | 2 / 0.500 | 6 / 0.167 | 1 / 0.000 | 3 | True | 1.000 | 1.000 | 0.600 | 1.600 | 11355 | 3759 | 0.000 |
| r6_x-hertz_hertz_quick | 12 | 0.250 | 0.375 (n=8) | 0.000 (n=4) | 0 /  | 2 / 0.500 | 7 / 0.286 | 0 /  | 3 | True | 1.000 | 1.000 | 0.600 | 1.600 | 73 | 123 | 0.000 |
| r6_x-hertz_planck | 243 | 0.391 | 0.497 (n=143) | 0.240 (n=100) | 12 / 0.750 | 67 / 0.687 | 82 / 0.415 | 28 / 0.214 | 54 | True | 0.984 | 0.883 | 0.746 | 1.421 | 66 | 107 | 0.000 |
| r6_x-hertz_planck_quick | 12 | 0.250 | 0.375 (n=8) | 0.000 (n=4) | 0 /  | 2 / 0.500 | 7 / 0.286 | 0 /  | 3 | True | 1.000 | 0.900 | 0.800 | 1.300 | 68 | 77 | 0.000 |
| r6_x-hertz_template | 243 | 0.391 | 0.497 (n=143) | 0.240 (n=100) | 12 / 0.750 | 67 / 0.687 | 82 / 0.415 | 28 / 0.214 | 54 | True | 0.984 | 1.000 |  |  | 63 | 0 | 0.000 |
| r6_x-hertz_template_contrarian | 243 | 0.399 | 0.503 (n=143) | 0.250 (n=100) | 0 /  | 0 /  | 43 / 0.814 | 146 / 0.425 | 54 | True | 0.984 | 1.000 |  |  | 76 | 0 | 0.634 |
| r6_x-hertz_template_cpu | 20 | 0.450 | 0.500 (n=12) | 0.375 (n=8) | 1 / 1.000 | 8 / 0.625 | 6 / 0.500 | 0 /  | 5 | True | 1.000 | 1.000 |  |  | 10608 | 0 | 0.000 |
| r6_x-hertz_template_quick | 12 | 0.250 | 0.375 (n=8) | 0.000 (n=4) | 0 /  | 2 / 0.500 | 7 / 0.286 | 0 /  | 3 | True | 1.000 | 1.000 |  |  | 71 | 0 | 0.000 |
| r6_x-planck_template | 243 | 0.424 | 0.503 (n=143) | 0.310 (n=100) | 15 / 0.667 | 69 / 0.754 | 82 / 0.390 | 38 / 0.237 | 39 | False | 0.984 | 1.000 |  |  | 47 | 0 | 0.000 |
| r6_x-planck_template_quick | 12 | 0.500 | 0.375 (n=8) | 0.750 (n=4) | 0 /  | 4 / 0.750 | 6 / 0.167 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 68 | 0 | 0.000 |
| r6b_x-hertz_template | 243 | 0.473 | 0.538 (n=143) | 0.380 (n=100) | 14 / 0.857 | 84 / 0.714 | 74 / 0.351 | 40 / 0.425 | 31 | False | 0.984 | 1.000 |  |  | 63 | 0 | 0.000 |
| r6b_x-hertz_template_contrarian | 243 | 0.477 | 0.545 (n=143) | 0.380 (n=100) | 0 /  | 0 /  | 48 / 0.833 | 164 / 0.463 | 31 | True | 0.984 | 1.000 |  |  | 76 | 0 | 0.691 |
| r6b_x-hertz_template_cpu | 20 | 0.550 | 0.583 (n=12) | 0.500 (n=8) | 2 / 1.000 | 6 / 0.833 | 8 / 0.500 | 0 /  | 4 | True | 1.000 | 1.000 |  |  | 11385 | 0 | 0.000 |
| r6b_x-planck_template | 243 | 0.535 | 0.448 (n=143) | 0.660 (n=100) | 30 / 0.800 | 82 / 0.683 | 83 / 0.361 | 43 / 0.465 | 5 | False | 0.984 | 1.000 |  |  | 64 | 0 | 0.000 |
| r7_x-hertz-rc_template | 243 | 0.601 | 0.580 (n=143) | 0.630 (n=100) | 29 / 0.724 | 88 / 0.739 | 77 / 0.506 | 43 / 0.488 | 6 | False | 0.984 | 1.000 |  |  | 71 | 0 | 0.000 |
| r7_x-hertz-rc_template_quick | 12 | 0.500 | 0.375 (n=8) | 0.750 (n=4) | 0 /  | 4 / 0.750 | 6 / 0.167 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 78 | 0 | 0.000 |
| r7_x-hertz_template | 243 | 0.560 | 0.552 (n=143) | 0.570 (n=100) | 33 / 0.758 | 77 / 0.727 | 72 / 0.486 | 44 / 0.455 | 17 | True | 0.984 | 1.000 |  |  | 52 | 0 | 0.000 |
| r7_x-hertz_template_quick | 12 | 0.333 | 0.375 (n=8) | 0.250 (n=4) | 1 / 1.000 | 3 / 0.667 | 4 / 0.250 | 1 / 0.000 | 3 | True | 1.000 | 1.000 |  |  | 78 | 0 | 0.000 |
| r7_x-planck-rc_template | 243 | 0.531 | 0.455 (n=143) | 0.640 (n=100) | 14 / 0.714 | 77 / 0.662 | 106 / 0.443 | 41 / 0.512 | 5 | False | 0.984 | 1.000 |  |  | 54 | 0 | 0.000 |
| r7_x-planck-rc_template_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 1 / 1.000 | 3 / 0.667 | 6 / 0.333 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 40 | 0 | 0.000 |
| r7_x-planck_template | 243 | 0.543 | 0.462 (n=143) | 0.660 (n=100) | 26 / 0.808 | 80 / 0.675 | 91 / 0.396 | 40 / 0.525 | 6 | False | 0.984 | 1.000 |  |  | 42 | 0 | 0.000 |
| r7_x-planck_template_contrarian | 243 | 0.543 | 0.469 (n=143) | 0.650 (n=100) | 0 /  | 3 / 0.667 | 32 / 0.781 | 202 / 0.520 | 6 | True | 0.984 | 1.000 |  |  | 96 | 0 | 0.811 |
| r7_x-planck_template_contrarian_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 0 /  | 0 /  | 0 /  | 12 / 0.583 | 0 | True | 1.000 | 1.000 |  |  | 37 | 0 | 0.833 |
| r7_x-planck_template_cpu | 20 | 0.700 | 0.667 (n=12) | 0.750 (n=8) | 5 / 1.000 | 3 / 1.000 | 7 / 0.571 | 3 / 0.667 | 2 | True | 1.000 | 1.000 |  |  | 1051 | 0 | 0.000 |
| r7_x-planck_template_cpu_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 1 / 1.000 | 3 / 0.667 | 6 / 0.333 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 892 | 0 | 0.000 |
| r7_x-planck_template_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 1 / 1.000 | 3 / 0.667 | 6 / 0.333 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 34 | 0 | 0.000 |
| r8_x-hertz-rc_cpu-fp32-top16 | 20 | 0.700 | 0.667 (n=12) | 0.750 (n=8) | 4 / 1.000 | 5 / 1.000 | 5 / 0.600 | 5 / 0.400 | 1 | True | 1.000 | 1.000 |  |  | 6460 | 0 | 0.000 |
| r8_x-hertz-rc_cpu-fp32-top16_quick | 12 | 0.500 | 0.375 (n=8) | 0.750 (n=4) | 0 /  | 4 / 0.750 | 6 / 0.167 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 11102 | 0 | 0.000 |
| r8_x-hertz-rc_cpu-int8-top16 | 243 | 0.605 | 0.580 (n=143) | 0.640 (n=100) | 27 / 0.704 | 87 / 0.747 | 81 / 0.543 | 43 / 0.442 | 5 | False | 0.984 | 1.000 |  |  | 1331 | 0 | 0.000 |
| r8_x-hertz-rc_cpu-int8-top16_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 0 /  | 4 / 0.750 | 6 / 0.333 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 2458 | 0 | 0.000 |
| r8_x-hertz-rc_cpu-int8-top8 | 243 | 0.588 | 0.552 (n=143) | 0.640 (n=100) | 27 / 0.704 | 90 / 0.722 | 78 / 0.526 | 44 / 0.409 | 4 | False | 0.984 | 1.000 |  |  | 752 | 0 | 0.000 |
| r8_x-hertz-rc_cpu-int8-top8_quick | 12 | 0.500 | 0.375 (n=8) | 0.750 (n=4) | 0 /  | 4 / 0.750 | 6 / 0.167 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 1414 | 0 | 0.000 |
| r8_x-planck15-rc_template_contrarian_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 0 /  | 0 /  | 0 /  | 12 / 0.583 | 0 | True | 1.000 | 1.000 |  |  | 38 | 0 | 0.833 |
| r8_x-planck15-rc_template_cpu_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 1 / 1.000 | 3 / 0.667 | 6 / 0.333 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 838 | 0 | 0.000 |
| r8_x-planck15-rc_template_quick | 12 | 0.583 | 0.500 (n=8) | 0.750 (n=4) | 1 / 1.000 | 3 / 0.667 | 6 / 0.333 | 2 / 1.000 | 0 | True | 1.000 | 1.000 |  |  | 36 | 0 | 0.000 |

| paired (same questions) | n | run | vs | only run right | only vs right | McNemar p |
|---|---|---|---|---|---|---|
| r6_x-hertz_template vs r6_planck_template | 243 | 0.391 | 0.514 | 18 | 48 | 0.000287 |
| r6_x-hertz_template vs r6_gemma_template | 243 | 0.391 | 0.634 | 22 | 81 | 4.07e-09 |
| r6_x-hertz_template vs r6_gemma_gemma | 243 | 0.391 | 0.634 | 22 | 81 | 4.07e-09 |
| r6_x-hertz_template vs r6_heuristic_template | 243 | 0.391 | 0.498 | 24 | 50 | 0.00337 |
| r6_x-hertz_template_contrarian vs r6_x-hertz_template | 243 | 0.399 | 0.391 | 2 | 0 | 0.5 |
| r6_x-hertz_template_cpu vs r6_x-hertz_template | 20 | 0.450 | 0.450 | 0 | 0 | 1 |
| r6_x-hertz_template_quick vs r6_x-hertz_template | 12 | 0.250 | 0.500 | 1 | 4 | 0.375 |
| r6b_x-hertz_template vs r6_planck_template | 243 | 0.473 | 0.514 | 26 | 36 | 0.253 |
| r6b_x-hertz_template vs r6_gemma_template | 243 | 0.473 | 0.634 | 24 | 63 | 3.48e-05 |
| r6b_x-hertz_template vs r6_gemma_gemma | 243 | 0.473 | 0.634 | 24 | 63 | 3.48e-05 |
| r6b_x-hertz_template vs r6_heuristic_template | 243 | 0.473 | 0.498 | 30 | 36 | 0.539 |
| r6b_x-hertz_template vs r6_x-hertz_template | 243 | 0.473 | 0.391 | 25 | 5 | 0.000325 |
| r6b_x-hertz_template_contrarian vs r6b_x-hertz_template | 243 | 0.477 | 0.473 | 1 | 0 | 1 |
| r6b_x-hertz_template_cpu vs r6b_x-hertz_template | 20 | 0.550 | 0.550 | 0 | 0 | 1 |
| r7_x-planck_template vs r6_planck_template | 243 | 0.543 | 0.514 | 38 | 31 | 0.47 |
| r7_x-planck_template vs r6_gemma_template | 243 | 0.543 | 0.634 | 28 | 50 | 0.0169 |
| r7_x-planck_template vs r6_gemma_gemma | 243 | 0.543 | 0.634 | 28 | 50 | 0.0169 |
| r7_x-planck_template vs r6_heuristic_template | 243 | 0.543 | 0.498 | 16 | 5 | 0.0266 |
| r7_x-planck_template vs r6_x-hertz_template | 243 | 0.543 | 0.391 | 63 | 26 | 0.00011 |
| r7_x-planck_template_contrarian vs r7_x-planck_template | 243 | 0.543 | 0.543 | 1 | 1 | 1 |
| r7_x-planck_template_contrarian_quick vs r7_x-planck_template | 12 | 0.583 | 0.583 | 1 | 1 | 1 |
| r7_x-planck_template_cpu vs r7_x-planck_template | 20 | 0.700 | 0.700 | 0 | 0 | 1 |
| r7_x-planck_template_cpu_quick vs r7_x-planck_template | 12 | 0.583 | 0.583 | 1 | 1 | 1 |
| r7_x-planck_template_quick vs r7_x-planck_template | 12 | 0.583 | 0.583 | 1 | 1 | 1 |
| r6_x-planck_template vs r6_planck_template | 243 | 0.424 | 0.514 | 27 | 49 | 0.0154 |
| r6_x-hertz_template vs r6_x-planck_template | 243 | 0.391 | 0.424 | 22 | 30 | 0.332 |
| r6_x-planck_template vs r6_gemma_template | 243 | 0.424 | 0.634 | 21 | 72 | 1.04e-07 |
| r6b_x-planck_template vs r6_x-planck_template | 243 | 0.535 | 0.424 | 52 | 25 | 0.0028 |
| r6b_x-hertz_template vs r6b_x-planck_template | 243 | 0.473 | 0.535 | 26 | 41 | 0.0864 |
| r7_x-planck_template vs r6b_x-planck_template | 243 | 0.543 | 0.535 | 8 | 6 | 0.791 |
| r7_x-planck-rc_template vs r7_x-planck_template | 243 | 0.531 | 0.543 | 4 | 7 | 0.549 |
| r7_x-hertz_template vs r7_x-planck_template | 243 | 0.560 | 0.543 | 26 | 22 | 0.665 |
| r7_x-hertz_template vs r6b_x-hertz_template | 243 | 0.560 | 0.473 | 35 | 14 | 0.0038 |
| r8_x-hertz-rc_cpu-int8-top16 vs r7_x-hertz-rc_template | 243 | 0.605 | 0.601 | 3 | 2 | 1 |
| r8_x-hertz-rc_cpu-int8-top8 vs r7_x-hertz-rc_template | 243 | 0.588 | 0.601 | 2 | 5 | 0.453 |
| r8_x-hertz-rc_cpu-fp32-top16 vs r7_x-hertz-rc_template | 20 | 0.700 | 0.650 | 1 | 0 | 1 |

- r6_product (pre-registered choice): reader x-hertz (Hertz reads >= 3 points better on held-out questions); writer template (no small writer passes rule W: the template writes); rule R FAIL (0.391 vs midpoint 0.574, round 5's reader 0.514, p 0.000287)
- r6b_product (pre-registered choice): reader x-hertz (Hertz reads >= 3 points better on held-out questions); writer template (round 6: no small writer passes rule W); rule R FAIL (0.473 vs midpoint 0.574, round 5's reader 0.514, p 0.253)
- r7_product (pre-registered choice): reader x-planck (fixed before the run: the phone-size model (no selection)); writer template (round 6: no small writer passes rule W); rule R FAIL (0.543 vs midpoint 0.574, round 5's reader 0.514, p 0.47)
- r8_product (pre-registered choice): reader x-planck (fixed before the run: Planck 1.5, the phone-size model); writer template (round 6: no small writer passes rule W); rule R FAIL (a run is missing)
- writer judge (r6_x-hertz_hertz_quick, Gemma, blind): small writer preferred 3, template 6, tie 1 of 10
- KV cache (writer_hertz6_cpu): 100% of 10 greedy answers identical; 4195 ms cached vs 41492 ms uncached (cpu)
- KV cache (writer_hertz6_cuda): 100% of 20 greedy answers identical; 175 ms cached vs 370 ms uncached (cuda)
- KV cache (writer_hertz6_quick_cpu): 100% of 10 greedy answers identical; 2728 ms cached vs 22209 ms uncached (cpu)
- KV cache (writer_hertz6_quick_cuda): 100% of 20 greedy answers identical; 138 ms cached vs 230 ms uncached (cuda)
- KV cache (writer_planck6_cpu): 100% of 10 greedy answers identical; 555 ms cached vs 5034 ms uncached (cpu)
- KV cache (writer_planck6_cuda): 100% of 20 greedy answers identical; 98 ms cached vs 100 ms uncached (cuda)
- KV cache (writer_planck6_quick_cpu): 100% of 10 greedy answers identical; 324 ms cached vs 2021 ms uncached (cpu)
- KV cache (writer_planck6_quick_cuda): 100% of 20 greedy answers identical; 71 ms cached vs 69 ms uncached (cuda)
- writer_hertz6: 1412 training examples (1271 train / 141 val), best val loss 0.734, 641M params
- writer_planck6: 1412 training examples (1271 train / 141 val), best val loss 0.755, 101M params

## Round 7: reading taught as a skill (SETUP_planck_20261014.md)

- **R** FAIL: x-planck r7 0.543 vs midpoint 0.574; round 5's reader 0.514, p 0.47
- **N (rebuild gate, NewsQA)** FAIL: choice 0.464 vs ranker 0.477 (+-1.3 pts, p 0.367, 859 pools)
- **N on SearchQA (reported)** FAIL: choice 0.522 vs ranker 0.462 (+6.0 pts, p 4.27e-05, 833 pools)
- **N for Hertz (reported)** PASS: choice 0.580 vs ranker 0.477 (+10.2 pts, p 2.73e-11, 859 pools)
- **H** (reported): Hertz r7 minus Planck r7 +1.6 pts, p 0.665 -> no clear Hertz advantage
- **C1** FAIL: monotone False, high {'n': 26, 'precision': 0.8076923076923077, 'bar': 0.9, 'judged': True}, good {'n': 80, 'precision': 0.675, 'bar': 0.7, 'judged': True}
- **L** PASS: ms 1051.35
- **E** PASS: drop 0.0, changed 197, flagged 197
- **Outcome (pre-registered):** rebuild: Planck does not read news even after reading training (round 8 rebuilds the base)

## Round 8: Planck 1.5 (Planck's size, Hertz's data) and Hertz on a phone proxy (SETUP_planck_20261015.md)

- **M (Hertz on the phone proxy, AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD)** PASS:
  - cpu-fp32-top16: 6460 ms per question, 1996.5 MB, right 0.700 vs GPU 0.650 on 20 questions (not judged: too few questions)
  - cpu-int8-top16: 1331 ms per question, 888.2 MB, right 0.605 vs GPU 0.601 on 243 questions
  - cpu-int8-top8: 752 ms per question, 888.2 MB, right 0.588 vs GPU 0.601 on 243 questions
- **Outcome (pre-registered):** pending: Planck 1.5 is not evaluated yet (this decision covers the mobile check only)

## G2 v2 heads (choice head + calibrated answerability gate)

| head | points | test points with a right candidate | choice accuracy | deterministic ranker | gate AUC | gate ECE | tau | test coverage at tau | test precision at tau (target) |
|---|---|---|---|---|---|---|---|---|---|
| g2v2_head_hash_s0 | points_searxng.jsonl | 143 | 0.336 | 0.364 | 0.739 | 0.061 | 0.879 | 0.014 | 1.000 (0.9) |
| g2v2_head_hash_s0_ddgs | points_ddgs.jsonl | 243 | 0.494 | 0.481 | 0.809 | 0.046 | 0.843 | 0.126 | 0.917 (0.9) |
| g2v2_head_hash_s0_ddgs_quick | points_ddgs_quick.jsonl | 32 | 0.438 | 0.312 | 0.753 | 0.204 | 0.528 | 0.325 | 0.462 (0.9) |
| g2v2_head_planck_s0 | points_searxng.jsonl | 143 | 0.490 | 0.364 | 0.782 | 0.051 | 0.669 | 0.051 | 1.000 (0.9) |
| g2v2_head_planck_s0_ddgs | points_ddgs.jsonl | 243 | 0.667 | 0.481 | 0.831 | 0.090 | 0.811 | 0.203 | 0.931 (0.9) |
| g2v2_head_planck_s0_ddgs_quick | points_ddgs_quick.jsonl | 32 | 0.531 | 0.312 | 0.875 | 0.184 | 0.911 | 0.150 | 1.000 (0.9) |

## G2 heads (learned from known answers)

| head | train points | val points with a right candidate | learned choice accuracy | deterministic ranker | 'none' correct (points without one) | val ECE | answer accuracy when p >= 0.5 |
|---|---|---|---|---|---|---|---|
| g2_head_hash_s0 | 2546 | 126 | 0.476 | 0.365 | 0.856 (153) | 0.050 | 0.724 |
| g2_head_planck_s0 | 2546 | 126 | 0.603 | 0.365 | 0.837 (153) | 0.057 | 0.708 |

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
| lexical | 1200 | 0.144 | 1.231 | 0.463 | 0.128 | 0.016 |
| head:hash | 1200 | 0.108 | 1.208 | 0.580 | 0.105 | 0.560 |
| head:hertz-rank | 1200 | 0.247 | 1.296 | 0.642 | 0.118 | 0.712 |
| head:hertz | 1200 | 0.190 | 1.312 | 0.626 | 0.094 | 0.688 |
| head:planck-rank | 1200 | 0.235 | 1.275 | 0.636 | 0.104 | 0.434 |
| head:planck | 1200 | 0.224 | 1.288 | 0.635 | 0.112 | 0.417 |
| gemma | 300 | 0.337 | 1.335 |  |  | 251.616 |

Cold CPU latency (encode target + all candidate titles): median 83 ms, p90 286 ms, 20 candidates on average

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.093 | 0.337 | 0.28 | 11 | 84 | 4.52e-15 |
| head:hash vs lexical | 1200 | 0.108 | 0.144 |  | 80 | 123 | 0.0031 |
| head:hertz-rank vs gemma | 300 | 0.237 | 0.337 | 0.70 | 28 | 58 | 0.00161 |
| head:hertz-rank vs head:hash | 1200 | 0.247 | 0.108 |  | 236 | 69 | 1.57e-22 |
| head:hertz-rank vs lexical | 1200 | 0.247 | 0.144 |  | 213 | 89 | 6.97e-13 |
| head:hertz vs gemma | 300 | 0.133 | 0.337 | 0.40 | 17 | 78 | 1.64e-10 |
| head:hertz vs head:hash | 1200 | 0.190 | 0.108 |  | 176 | 78 | 7.17e-10 |
| head:hertz vs lexical | 1200 | 0.190 | 0.144 |  | 156 | 101 | 0.000725 |
| head:planck-rank vs gemma | 300 | 0.227 | 0.337 | 0.67 | 23 | 56 | 0.000264 |
| head:planck-rank vs head:hash | 1200 | 0.235 | 0.108 |  | 215 | 63 | 1.35e-20 |
| head:planck-rank vs lexical | 1200 | 0.235 | 0.144 |  | 198 | 89 | 1.1e-10 |
| head:planck vs gemma | 300 | 0.220 | 0.337 | 0.65 | 25 | 60 | 0.000187 |
| head:planck vs head:hash | 1200 | 0.224 | 0.108 |  | 198 | 59 | 9.81e-19 |
| head:planck vs lexical | 1200 | 0.224 | 0.144 |  | 180 | 84 | 3.4e-09 |
| head:planck-rank vs head:planck | 1200 | 0.235 | 0.224 |  | 134 | 121 | 0.452 |
| head:hertz vs head:planck | 1200 | 0.190 | 0.224 |  | 89 | 130 | 0.00674 |
| head:hertz-rank vs head:planck-rank | 1200 | 0.247 | 0.235 |  | 140 | 125 | 0.39 |

**Verdict:** FAIL. head:planck-rank-vs-hash +0.127 (McNemar p=1.35e-20); paired head/teacher = 0.67 on 300 shared races (pass >= 0.8); warm decision 0.4 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.28, head:hertz-rank 0.70, head:hertz 0.40, head:planck 0.65

## G1 Wikiracing (g1_eval_s4_t1)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1200 | 0.001 | 1.500 | 0.197 | 0.103 | 0.000 |
| lexical | 1200 | 0.144 | 1.231 | 0.463 | 0.128 | 0.014 |
| head:hash | 1200 | 0.111 | 1.239 | 0.587 | 0.120 | 0.624 |
| head:hertz-rank | 1200 | 0.239 | 1.288 | 0.644 | 0.103 | 0.686 |
| head:hertz | 1200 | 0.208 | 1.305 | 0.621 | 0.096 | 0.659 |
| head:planck-rank | 1200 | 0.230 | 1.296 | 0.633 | 0.102 | 0.393 |
| head:planck | 1200 | 0.207 | 1.253 | 0.632 | 0.090 | 0.395 |
| gemma | 300 | 0.337 | 1.335 |  |  | 251.616 |

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.080 | 0.337 | 0.24 | 8 | 85 | 2.26e-17 |
| head:hash vs lexical | 1200 | 0.111 | 0.144 |  | 83 | 123 | 0.00644 |
| head:hertz-rank vs gemma | 300 | 0.243 | 0.337 | 0.72 | 33 | 61 | 0.00508 |
| head:hertz-rank vs head:hash | 1200 | 0.239 | 0.111 |  | 214 | 60 | 1.86e-21 |
| head:hertz-rank vs lexical | 1200 | 0.239 | 0.144 |  | 198 | 84 | 8.94e-12 |
| head:hertz vs gemma | 300 | 0.163 | 0.337 | 0.49 | 25 | 77 | 2.45e-07 |
| head:hertz vs head:hash | 1200 | 0.208 | 0.111 |  | 189 | 72 | 2.77e-13 |
| head:hertz vs lexical | 1200 | 0.208 | 0.144 |  | 178 | 101 | 4.71e-06 |
| head:planck-rank vs gemma | 300 | 0.270 | 0.337 | 0.80 | 34 | 54 | 0.0422 |
| head:planck-rank vs head:hash | 1200 | 0.230 | 0.111 |  | 208 | 65 | 1.28e-18 |
| head:planck-rank vs lexical | 1200 | 0.230 | 0.144 |  | 187 | 84 | 3.58e-10 |
| head:planck vs gemma | 300 | 0.230 | 0.337 | 0.68 | 21 | 53 | 0.000256 |
| head:planck vs head:hash | 1200 | 0.207 | 0.111 |  | 185 | 69 | 2.03e-13 |
| head:planck vs lexical | 1200 | 0.207 | 0.144 |  | 173 | 97 | 4.37e-06 |
| head:planck-rank vs head:planck | 1200 | 0.230 | 0.207 |  | 111 | 84 | 0.0623 |
| head:hertz vs head:planck | 1200 | 0.208 | 0.207 |  | 124 | 123 | 1 |
| head:hertz-rank vs head:planck-rank | 1200 | 0.239 | 0.230 |  | 122 | 111 | 0.512 |

**Verdict:** PASS. head:planck-rank-vs-hash +0.119 (McNemar p=1.28e-18); paired head/teacher = 0.80 on 300 shared races (pass >= 0.8); warm decision 0.4 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.24, head:hertz-rank 0.72, head:hertz 0.49, head:planck 0.68

## G1 Wikiracing (g1_eval_s5_t1)

| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |
|---|---|---|---|---|---|---|
| random | 1200 | 0.001 | 1.500 | 0.197 | 0.103 | 0.000 |
| lexical | 1200 | 0.144 | 1.231 | 0.463 | 0.128 | 0.014 |
| head:hash | 1200 | 0.102 | 1.221 | 0.580 | 0.118 | 0.520 |
| head:hertz-rank | 1200 | 0.225 | 1.270 | 0.641 | 0.126 | 0.658 |
| head:hertz | 1200 | 0.196 | 1.248 | 0.634 | 0.088 | 0.653 |
| head:planck-rank | 1200 | 0.241 | 1.322 | 0.635 | 0.099 | 0.414 |
| head:planck | 1200 | 0.223 | 1.306 | 0.629 | 0.095 | 0.406 |
| gemma | 300 | 0.337 | 1.335 |  |  | 251.616 |

| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |
|---|---|---|---|---|---|---|---|
| head:hash vs gemma | 300 | 0.080 | 0.337 | 0.24 | 8 | 85 | 2.26e-17 |
| head:hash vs lexical | 1200 | 0.102 | 0.144 |  | 64 | 115 | 0.00017 |
| head:hertz-rank vs gemma | 300 | 0.230 | 0.337 | 0.68 | 33 | 65 | 0.0016 |
| head:hertz-rank vs head:hash | 1200 | 0.225 | 0.102 |  | 209 | 61 | 3.86e-20 |
| head:hertz-rank vs lexical | 1200 | 0.225 | 0.144 |  | 187 | 90 | 5.65e-09 |
| head:hertz vs gemma | 300 | 0.173 | 0.337 | 0.51 | 27 | 76 | 1.45e-06 |
| head:hertz vs head:hash | 1200 | 0.196 | 0.102 |  | 182 | 69 | 6.31e-13 |
| head:hertz vs lexical | 1200 | 0.196 | 0.144 |  | 161 | 99 | 0.000145 |
| head:planck-rank vs gemma | 300 | 0.223 | 0.337 | 0.66 | 34 | 68 | 0.000987 |
| head:planck-rank vs head:hash | 1200 | 0.241 | 0.102 |  | 224 | 57 | 1.43e-24 |
| head:planck-rank vs lexical | 1200 | 0.241 | 0.144 |  | 206 | 90 | 1.28e-11 |
| head:planck vs gemma | 300 | 0.183 | 0.337 | 0.54 | 24 | 70 | 2.2e-06 |
| head:planck vs head:hash | 1200 | 0.223 | 0.102 |  | 213 | 67 | 6.81e-19 |
| head:planck vs lexical | 1200 | 0.223 | 0.144 |  | 193 | 98 | 2.72e-08 |
| head:planck-rank vs head:planck | 1200 | 0.241 | 0.223 |  | 126 | 105 | 0.188 |
| head:hertz vs head:planck | 1200 | 0.196 | 0.223 |  | 112 | 145 | 0.0457 |
| head:hertz-rank vs head:planck-rank | 1200 | 0.225 | 0.241 |  | 124 | 143 | 0.271 |

**Verdict:** FAIL. head:planck-rank-vs-hash +0.139 (McNemar p=1.43e-24); paired head/teacher = 0.66 on 300 shared races (pass >= 0.8); warm decision 0.4 ms (pass < 100.0); secondary arms (exploratory): head:hash 0.24, head:hertz-rank 0.68, head:hertz 0.51, head:planck 0.54

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
| head:hertz | 0.198 | 0.009 | [0.175, 0.221] | 0.190, 0.208, 0.196 |
| head:hertz-rank | 0.237 | 0.011 | [0.209, 0.265] | 0.247, 0.239, 0.225 |
| head:planck | 0.218 | 0.009 | [0.195, 0.242] | 0.224, 0.207, 0.223 |
| head:planck-rank | 0.235 | 0.005 | [0.222, 0.249] | 0.235, 0.230, 0.241 |
| lexical | 0.144 | 0.000 | [0.144, 0.144] | 0.144, 0.144, 0.144 |
| random | 0.001 | 0.000 | [0.001, 0.001] | 0.001, 0.001, 0.001 |

head:hash: paired ratio vs teacher 0.25 (per seed 0.28, 0.24, 0.24)

head:hertz: paired ratio vs teacher 0.47 (per seed 0.40, 0.49, 0.51)

head:hertz-rank: paired ratio vs teacher 0.70 (per seed 0.70, 0.72, 0.68)

head:planck: paired ratio vs teacher 0.63 (per seed 0.65, 0.68, 0.54)

head:planck-rank: paired ratio vs teacher 0.71 (per seed 0.67, 0.80, 0.66)

**Verdict (seeds):** FAIL. mean paired head:planck-rank/teacher = 0.71 over 3 seeds (pass >= 0.8); beats head:hash in every seed: True

