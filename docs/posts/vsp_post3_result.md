# LinkedIn Post 3: the result (the grounded model lost, and why that is still useful)

Illustration: docs/posts/img/vsp_grounding.svg
Tone: plain, factual, understated. No em dashes. Plain text below for paste.

---

A follow-up on the token representation we have been testing at Radiance Labs.

Earlier we showed that a bundled token (what a thing looks like, means, and is made of) could separate word senses that a normal embedding merges: "crane the bird" from "crane the machine". The separation test passed. So we trained the actual model to see whether the separation would help it. It did not. Here is the honest result and what it taught us.

The setup: two identical small language models at matched compute. One starts from the grounded bundle. One starts from random. Then a fair test, 105 sense-disambiguation pairs, does the model prefer the sense-correct next word?

Result: the grounded model scored 0.79, the random baseline 0.83. Grounding lost by about four points, and the gap is not statistically meaningful.

The interesting part is why. We measured how correlated the two trained models were, and they had converged to almost the same function (0.96). Over billions of training tokens, the model overwrote the grounded starting point and learned disambiguation from scratch, the same way the baseline did. The head start did not survive contact with training.

That reframes the bet. Injecting meaning up front can only help if the model would otherwise struggle to learn it. At full training scale it does not struggle, so a warm start is at best neutral, and here slightly negative, from two fixable wiring issues we found in the post-mortem. We ran the experiment in exactly the regime where the idea was designed not to matter.

The real claim was always about efficiency: a smaller model, or less data. That test, the same comparison at a fraction of the compute, is running now. If grounding wins there and ties at full scale, the thesis holds and it is a data-efficiency lever, not a ceiling-raiser. If it loses there too, the "bake it into the embedding" approach is done and we move the grounding elsewhere.

Two things we keep relearning:

A passing probe is not a passing model. The representation separating senses and the representation helping a trained model are different claims. It is tempting to treat the first as evidence for the second. It is not.

A negative result with a clean diagnosis is worth as much as a win. We now know precisely why it failed and which regime to test next, which is more useful than a fluke pass would have been.

Code and the full write-up in the comments.
