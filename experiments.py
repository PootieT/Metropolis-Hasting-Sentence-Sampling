from typing import *
import random
import numpy as np
from datasets import load_dataset, Dataset

from sampler import *


def draw_random_pairs(dataset: Dataset, subset_per_class_count: Optional[int]=10):
    df = dataset.to_pandas()
    classes = np.unique(df.label)
    if subset_per_class_count:
        df = pd.concat([df[df.label==l].sample(subset_per_class_count) for l in classes])

    result_df = pd.DataFrame()
    for label in classes:
        this_class_df = df[df.label==label].rename(columns={"text": "sent1", "label": "label1"})
        other_class_df = df[df.label!=label].rename(columns={"text": "sent2", "label": "label2"})
        pair_df = df.concat([this_class_df,
                            other_class_df.sample(len(this_class_df),
                                                  replace=len(this_class_df)>len(other_class_df))],
                            axis=1)
        result_df = result_df.append(pair_df)
    return result_df

def augment_dataset():
    pass


def intrinsic_evaluation(
    sentence_pairs: List[List[str]],
    mh_sampler: MetropolisHastingSentenceSampler,
    num_runs: int=5
):
    """
    Given sets of paired sentences, and a sampler
    - How quickly does it converge (semantic score > 0.8)?
        - number of iterations
        - run time
        - how often does it fail to converge
    - How good are samples in between convergence?
        - average fluency (discard one/zero word examples)
        - number of rejected samples
        - average acceptance

    Visualize:
        - semantic score trajectories
        - fluency score trajectories
        - bar plot: action vs. acceptance
        - scatterplot: action vs. replacement_lbd
        - acceptance / rejections over time
    """

    pass


def extrinsic_evaluation():
    """
    Given few shot examples, perform mixup sampling, and finetune a classifier
    on top. 
    """
    pass


if __name__ == "__main__":
    random.seed = 24
    np.random.seed(24)
    dataset = load_dataset("imdb")
    pair_df = draw_random_pairs(dataset["train"], subset_per_class_count=10)
    sentence_pair = pair_df[["sent1", "sent2"]].tolist()

    mhss = MetropolisHastingSentenceSampler(
        sampler_model_name="roberta-base",
        acceptor_semantic_model_name="sentence-transformers/all-mpnet-base-v2",
        # "sentence-transformers/all-MiniLM-L6-v2",
        acceptor_fluency_model_name="distilgpt2",
        method="word_random",
        lambda_semantic=2.0,
        lambda_fluency=0.1,
        target_fusion="logits",
        fusion_aggregation="local",
        fusion_interpolation="linear",
        init_temp=1.0,
        min_temp=0.1,
    )

    intrinsic_evaluation(sentence_pair, mhss, num_runs=5)