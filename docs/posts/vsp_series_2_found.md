# LinkedIn Series 3/3 — Post 2: what we found

Illustration: docs/posts/img/vsp_series_2_found.svg
Tone: plain, first-person, understated. No em dashes. Plain text below for paste.

---

Last post: a grounded token representation (text + a generated image + material properties) separates word senses that a plain embedding merges. This post: I put it into a trained language model to see if it helps. It does not. Here is the honest version, including three times I nearly fooled myself.

I tried two ways to deliver the grounding.

First, initialize the model's token embeddings from the bundle, then train. At full training this LOST by 3.8 points. The two models, grounded and random, ended up 96% identical: training simply overwrote the head start. A warm start only helps if the model would otherwise struggle, and at full scale it does not struggle.

At low compute it looked different: +5.7 points. Promising. But I reran it with a different random seed and got minus 1.0. Six seeds averaged +1.6 with a confidence interval that comfortably includes zero. The +5.7 was just the best of six draws. Two of the six were negative.

Second, skip retraining and use the bundle to rerank the model's guesses at inference. Best case +2.9 points, but not statistically significant. When I sliced to the cases where the model was genuinely unsure, it jumped to +8.3, the right shape, exactly where grounding should help. So I built a harder benchmark, 260 low-context examples designed to keep the model uncertain, to confirm it.

The benchmark worked: model accuracy fell from 0.83 to 0.58, it really was unsure. And the grounding did nothing. Chosen honestly (tune on one half, measure on the other), the gain was minus 0.004. Where the model was truly lost, the grounding was lost too: 54% correct, a hair above a coin flip.

The +8.3 evaporated because those "unsure" cases still had faint text cues the grounding was quietly riding on. Take the text away entirely and the signal is not there. The grounding was echoing language where language already worked, not rescuing it where it failed.

What caught all three false positives was boring discipline: reproduce across seeds, and never report a number tuned on the same data you measured on. Those two habits turned three publishable-looking wins into one honest negative.

Post 3: what that actually buys you, and where the idea is still alive.
