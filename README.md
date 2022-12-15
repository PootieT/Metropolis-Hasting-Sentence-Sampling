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

- if embedding interpolation doesn't work well, would be cool to do case study, see cases of such interpolation (probing word classifier)
  - emb-closest-exp interpolation does not work well, produces <s> tokens
  - emb-global/local-polar doesn't work well, especially global, i think you can't interpolate in hidden space too much before anisotropy hits

- fusion in logits
  - works! with closest-polar + dirchlet sampling, quiet nice
  - global does not work as well, too much noise i think
  - local/closest + linear/polar works well


## Intrinsic Evaluations:

