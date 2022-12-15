import time
from typing import *
import random
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
import seaborn as sns
import matplotlib.pyplot as plt

from sampler import *


def draw_random_pairs(dataset: Dataset, subset_per_class_count: Optional[int]=10):
    df = dataset.to_pandas()
    classes = np.unique(df.label)
    if subset_per_class_count:
        df = pd.concat([df[df.label==l].sample(subset_per_class_count) for l in classes])

    result_df = pd.DataFrame()
    for label in classes:
        this_class_df = df[df.label==label].rename(columns={"text": "sent1", "label": "label1"}).reset_index(drop=True)
        other_class_df = df[df.label!=label].rename(columns={"text": "sent2", "label": "label2"})
        other_class_df = other_class_df.sample(len(this_class_df),
                                               replace=len(this_class_df)>len(other_class_df)).reset_index(drop=True)
        pair_df = pd.concat([this_class_df, other_class_df], axis=1)
        result_df = result_df.append(pair_df)
    return result_df


def augment_dataset():
    pass


def visualize_sampling_runs(sample_df: pd.DataFrame, postfix: str=""):
    # semantic score trajectory
    plt.figure()
    sample_df = sample_df.rename(columns={"sem_sim": "semantic similarity"})
    sns.lineplot(sample_df, x="step", y="semantic similarity")
    plt.title("Semantic Similarity to Target Sentence During Sampling")
    plt.savefig(f"figures/trajectory_sem{postfix}.png")


def intrinsic_evaluation(
    sentence_pairs: List[Tuple[str, str]],
    mh_sampler: MetropolisHastingSentenceSampler,
    num_runs: int=5,
    visualize: bool=True,
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
        - acceptance trajectories
        - bar plot: action vs. acceptance
        - scatter plot: action vs. replacement_lbd
    """
    meta_stats = []
    failed_df = pd.DataFrame()
    success_df = pd.DataFrame()
    num_steps = 100
    for i, pair in enumerate(sentence_pairs):
        for run_idx in range(num_runs):
            start_time = time.perf_counter()
            result_df = mh_sampler.metropolis_hasting_sample(pair[0], pair[1], steps=num_steps)
            total_time = time.perf_counter() - start_time
            filtered_df = mh_sampler.post_sample_filter(result_df, target_semantic_score=0.8)
            meta_stats.append({
                "total_time": total_time,
                "mixin_steps": num_steps if filtered_df is None else filtered_df.index[-1],
                "sampling_time": total_time if filtered_df is None else total_time*filtered_df.index[-1]/num_steps,
                "cnt_reject": sum([result_df.sentence != result_df.proposal_sentence]) - 1 if filtered_df is None else
                              sum([filtered_df.sentence != filtered_df.proposal_sentence]) - 1,
            })
            if filtered_df is None:
                result_df["run_idx"] = run_idx
                result_df["data_idx"] = i
                result_df["step"] = result_df.index
                failed_df = failed_df.append(result_df)
            else:
                filtered_df["run_idx"] = run_idx
                filtered_df["data_idx"] = i
                filtered_df["step"] = filtered_df.index
                success_df = success_df.append(filtered_df)

    meta_df = pd.DataFrame(meta_stats)
    print("meta stats:\n", meta_df.describe())
    stat_columns = ["avg_perplexity", "acceptance"]
    print("success sample stats:\n", success_df[stat_columns].describe())
    print("failed sample stats:\n", failed_df[stat_columns].describe())
    if visualize:
        visualize_sampling_runs(success_df)


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
    sentence_pair = list(zip(pair_df["sent1"], pair_df["sent2"]))

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