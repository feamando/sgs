# Raum 0.1 / 1.2: Shipping a Text-to-3D Demo and Hitting a Fundamental Ceiling

We shipped Raum 0.1, a local text-to-3D demo built on Semantic Gaussian Splatting. Type a scene description, get positioned 3D Gaussian splats with an editable DSL. 300 object classes from Objaverse, relation heads for spatial reasoning, a frozen 100M language model as the encoder. The bridge predicts position, blob ID, color, scale, and role per token. It works.

Then we tried to make it better.

Raum 1.2 was supposed to be a quality pass: 50k training samples, 100 epochs, cosine-NN OOV policy. Same architecture, more data, longer training. The result was 0.1% blob accuracy. A collapse, not a plateau.

The root cause is architectural, not a hyperparameter problem. We use a frozen SentencePiece-tokenized language model (Planck 1.3, 100M params, Wikipedia-trained) as the encoder. SentencePiece decomposes words into subword units. "Cup", "cupcake", and "cupboard" all route through the same internal token. Across our 300-class blob vocabulary, we identified 35 collision groups affecting roughly 120 classes. The bridge receives embeddings that are nearly indistinguishable for objects that happen to share subword prefixes.

This is not a bug. It is a design property of subword tokenization, which optimizes for next-token prediction across open vocabulary, not for fine-grained class discrimination.

The finding: using frozen LM embeddings as-is for per-word object routing hits a fundamental ceiling. Subword tokenization trades word-level distinctness for vocabulary coverage. That tradeoff is correct for language modeling but wrong for routing bridges that need to discriminate between 300+ object classes at the word level.

Raum 0.1 ships with the GloVe path (300-dimensional word vectors, no subword decomposition, one vector per word). GloVe gives clean separation between object classes because each word gets its own learned embedding. This is the proven path from Raum 1.0.

What we learned shapes what comes next. The Planck encoder is not useless for 3D generation. It just should not be used as a word-level classifier. Its proper role is as a compositional decomposer: understanding that "a castle on a hill" implies towers, gates, walls, and a sloped terrain, then recursively expanding each sub-concept until we reach terminal primitives that map to individual Gaussian splats.

That is Raum 1.3: recursive semantic-to-geometric decomposition. The language model decomposes, GloVe discriminates at the leaves.
