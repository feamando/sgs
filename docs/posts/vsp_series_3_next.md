# LinkedIn Series 3/3 — Post 3: what's next

Illustration: docs/posts/img/vsp_series_3_next.svg
Tone: plain, first-person, understated. No em dashes. Plain text below for paste.

---

Two posts ago: a grounded token representation separates word senses. Last post: it does not help a trained language model, across two mechanisms and three benchmarks. This post: what a clean negative is actually worth, and where the door is still open.

A negative result is not a dead end if you know precisely what died. Here is the split.

Stands: the representation separates senses (0.37 versus 0.13 for text). A picture of a crane the bird and a crane the machine genuinely sit far apart. That is a real finding and I am writing it up as one.

Closed: baking that separation into a token, by initialization or by reranking, does not help a language model. Two mechanisms, three benchmarks, no significant win. The reason is almost mundane, text context already carries the sense the grounding was trying to add. Grounding was redundant where language worked and absent where it did not.

Still open, and I want to be honest that these are untested, not disproven:

- A different visual source. I generated images per sense; real photographs might carry sharper signal.
- Trained in, not bolted on. I injected grounding at the start or the end. A contrastive objective DURING training, where the grounding shapes the space continuously instead of being overwritten, is a different bet.
- A task where language genuinely is not enough. Disambiguation from rich text may just be the wrong place to look, because text is already good at it.

But the lesson I am taking is about method, not about VSP. Three times, a result looked strong enough to publish: +5.7, +2.9, +8.3. Three times, a reproduction across seeds or an honest held-out measurement took it to zero. The negative is trustworthy only because those checks came before I believed the positives, not after.

I would rather ship a correct no than a fragile yes. Full write-up and code in the comments.
