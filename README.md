# Metropolis-Hasting-Sentence-Sampling
Sampling / Interpolating sentence using transformers with Metropolis-Hasting like algorithm


## Experiment logs:

- temperature needs to be high
- action based sampling, shorter sentence seems to win given low perplexity, needs to increase semantics a lot
- after sampling, sentence drifts to different language, senseless sentence, high temp not so good either.
- need to influence sampler to take more directed sampling. how to influence?
  - sawp [CLS] emb with target sentence [CLS]
  - average word embeddings
  - randomly inject target sentence words
  - mask target sentence, infuse with source sentence (word swap, logit average)
  - 