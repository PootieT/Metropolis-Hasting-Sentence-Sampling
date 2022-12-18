import os
import time
from typing import *
import random
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
import seaborn as sns
import matplotlib.pyplot as plt

from sampler import *


def draw_random_pairs(
    df: pd.DataFrame,
    subset_per_class_count: Optional[int]=10,
    max_len: Optional[int]=None
) -> pd.DataFrame:
    classes = np.unique(df.label)
    if max_len:  # roughly filter by number of word size
        df = df[df.text.apply(lambda x: len(x.split())) < max_len-50]
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


def augment_dataset(
    pair_df: pd.DataFrame,
    mh_sampler: Optional[MetropolisHastingSentenceSampler],
    num_aug:int=1,
    margin=0.02,
    aug_data_dir:Optional[str]=None
) -> pd.DataFrame:
    if num_aug > 1:
        pair_df = pd.concat([pair_df]*num_aug)
    pair_df["alpha"] = uniform.rvs(scale=0.5, size=len(pair_df))
    num_class = pair_df.label1.max()+1
    aug_df = []
    if aug_data_dir is not None:
        try:
            exp_df = pd.concat([pd.read_csv(f"{aug_data_dir}/success.csv"), pd.read_csv(f"{aug_data_dir}/fail.csv")])
        except:
            exp_df = pd.read_csv(f"{aug_data_dir}/fail.csv")
    for i, row in pair_df.iterrows():
        sample_found = False
        while not sample_found:
            # max_steps = max(len(row["sent1"].split()), len(row["sent2"].split()))
            # sample_df = mh_sampler.metropolis_hasting_sample(row["sent1"], row["sent2"], max_steps, row["alpha"])
            sample_df = exp_df[exp_df.data_idx==i]
            # if sample_df.semantic_progress.max() >= row["alpha"]-margin:
            # find samples closest to target progress, search within +-% range, pick one with lowest perplexity
            closest_progress = sample_df.iloc[np.argmin(np.abs(sample_df.semantic_progress-row["alpha"]))].semantic_progress
            filt_df = sample_df[(sample_df.semantic_progress > closest_progress - margin) &
                                (sample_df.semantic_progress < closest_progress + margin)]
            # filt_df = sample_df[(sample_df.semantic_progress > row["alpha"]-margin) &
            #                     (sample_df.semantic_progress < row["alpha"]+margin)]
            aug_sentence = filt_df.sentences.values[filt_df["avg_perplexity"].argmin()]
            aug_ppl = filt_df.avg_perplexity.values[filt_df["avg_perplexity"].argmin()]
            aug_label = list(range(num_class))
            aug_label[row.label1], aug_label[row.label2] = 1.0-row["alpha"], row["alpha"]
            aug_df.append({"text": aug_sentence, "label": aug_label, "ppl": aug_ppl})
            aug_df.append({"text": row.sent1, "label": [1 if i==row.label1 else 0 for i in range(num_class)]})
            sample_found = True
    aug_df = pd.DataFrame(aug_df)
    if aug_data_dir is not None:
        aug_df.to_csv(f"{aug_data_dir}/aug.csv")
    return aug_df


def plot_data(exp_str:str):
    df = pd.read_csv(f"dump/{exp_str}/success.csv")
    visualize_sampling_runs(df, exp_str)


def visualize_sampling_runs(sample_df: pd.DataFrame, exp_str: str="baseline"):
    # semantic score trajectory
    plt.figure()
    sample_df = sample_df.rename(columns={"semantic_progress": "semantic similarity to target (normalized)"})
    sns.lineplot(data=sample_df, x="step", y="semantic similarity to target (normalized)", hue="data_idx")
    plt.title("Normalized Semantic Similarity to Target Sentence over Sampling Steps")
    plt.savefig(f"dump/{exp_str}/trajectory_semantics.png")

    # perplexity score trajectory
    plt.figure()
    sample_df = sample_df.rename(columns={"avg_perplexity": "perplexity"})
    sns.lineplot(data=sample_df, x="step", y="perplexity", hue="data_idx")
    plt.title("Perplexity over Sampling Steps")
    plt.savefig(f"dump/{exp_str}/trajectory_perplexity.png")

    # acceptance score trajectory
    plt.figure()
    sns.lineplot(data=sample_df, x="step", y="acceptance", hue="data_idx")
    reject_df = sample_df[sample_df.sentences!=sample_df.proposal_sentence]
    plt.scatter(reject_df.step, reject_df.acceptance, color="r", marker="x")
    plt.title("Acceptance over Sampling Steps")
    plt.savefig(f"dump/{exp_str}/trajectory_acceptance.png")

    # action or replacement lambda over
    plt.figure()
    if np.isnan(sample_df.replacement_lbd[0]):
        sample_df["action"] = sample_df.action.replace({"ins": "insertion", "del": "deletion", "sub": "substitution"})
        sns.barplot(data=sample_df, x="action", y="acceptance", hue="data_idx")
        # plt.tight_layout()
        plt.title("Sampling Action vs. Acceptance")
    else:
        sample_df = sample_df.rename(columns={"replacement_lbd": "replacement lambda"})
        sns.scatterplot(data=sample_df, x="replacement lambda")
        plt.title("Replacement Lambda vs. Acceptance")
    plt.savefig(f"dump/{exp_str}/action_vs_acceptance.png")


def intrinsic_evaluation(
    sentence_pairs: List[Tuple[str, str]],
    mh_sampler: MetropolisHastingSentenceSampler,
    num_runs: int=5,
    visualize: bool=True,
    exp_str: str="baseline"
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
        - scatter plot: replacement lbd vs. acceptance
    """
    print(f"============ starting experiment {exp_str} ===========")
    os.makedirs(f"dump/{exp_str}", exist_ok=False)
    meta_stats = []
    failed_df = pd.DataFrame()
    success_df = pd.DataFrame()
    for i, pair in enumerate(sentence_pairs):
        for run_idx in range(num_runs):
            num_steps = max(len(pair[0].split()), len(pair[1].split()))
            start_time = time.perf_counter()
            result_df = mh_sampler.metropolis_hasting_sample(
                pair[0], pair[1], steps=num_steps, early_stop_semantic_ratio=0.5)
            total_time = time.perf_counter() - start_time
            filtered_df = mh_sampler.post_sample_filter(result_df, target_semantic_ratio=0.5)
            meta_stats.append({
                "run_idx": run_idx,
                "data_idx": i,
                "success": filtered_df is not None,
                "total_time": total_time,
                "mixin_steps": num_steps if filtered_df is None else filtered_df.index[-1],
                "sampling_time": total_time if filtered_df is None else total_time*filtered_df.index[-1]/num_steps,
                "time_per_step": total_time/len(result_df),
                "cnt_reject": sum(result_df.sentences != result_df.proposal_sentence) - 1 if filtered_df is None else
                              sum(filtered_df.sentences != filtered_df.proposal_sentence) - 1,
                "max_semantic_progress": max(result_df.semantic_progress),
                "mean_ppl": result_df.avg_perplexity.mean(),
                "mean_acceptance": result_df.acceptance.mean()
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
    exp_path = f"dump/{exp_str}"
    meta_df.to_csv(f"{exp_path}/meta.csv", index=False)
    success_df.to_csv(f"{exp_path}/success.csv", index=False)
    failed_df.to_csv(f"{exp_path}/fail.csv", index=False)

    # print("meta stats:\n", meta_df[["success", "sampling_time", "mixin_steps", "cnt_reject"]].describe())
    # stat_columns = ["avg_perplexity", "acceptance"]
    # print("success sample stats:\n", success_df[stat_columns].describe())
    # print("failed sample stats:\n", failed_df[stat_columns].describe())

    if visualize:
        visualize_sampling_runs(success_df, exp_str)


def extrinsic_evaluation():
    """
    Given few shot examples, perform mixup sampling, and finetune a classifier
    on top. 
    """
    pass


def interpolation_evaluation():
    """ Are pretrained model's embedding linearly interpretable? """
    print(f"============ starting interpolation experiment ===========")
    pairs = [
        ("Roses are [MASK], violets are blue",
         "Bananas usually have the color [MASK]"),
        ("In the US, cars go on the [MASK] side of the road",
         "Most people are right handed, so it's rare to someone who is [MASK] handed"),
        ("Firefighters are [MASK] people: they save lives while risking their own.",
         "Thieves are [MASK] people: they take advantage of others for their own gain."),
        ("I left the car [MASK] at my desk, so I can't start the car.",
         "Brazil is home to one of the largest wild habitat in the world: the Amazon [MASK].")
    ]
    kwargs = {
        "sampler_model_name": "distilbert-base-uncased",
        "acceptor_semantic_model_name": "sentence-transformers/all-mpnet-base-v2",
        "acceptor_fluency_model_name": "distilgpt2",
        "method": None,  # probing mode, no sampling, assumes input has mask already
        "init_temp": 1,
        "min_temp": 1,
        "fusion_aggregation": "closest"
    }
    exp_kwargs_list = [
        {"target_fusion": "hiddens", "fusion_interpolation": "linear"},
        {"target_fusion": "hiddens", "fusion_interpolation": "exp"},
        {"target_fusion": "hiddens", "fusion_interpolation": "polar"},
        {"target_fusion": "logits",  "fusion_interpolation": "linear"},
        {"target_fusion": "logits", "fusion_interpolation": "exp"},
        {"target_fusion": "logits", "fusion_interpolation": "polar"},
    ]
    all_outputs = []
    for i, pair in enumerate(pairs):
        for exp_kwargs in exp_kwargs_list:
            kwargs.update(exp_kwargs)
            mh_sampler = MetropolisHastingSentenceSampler(**kwargs)
            proposed_sentences = []
            for _ in tqdm(range(10)):
                result_df = mh_sampler.metropolis_hasting_sample(pair[0], pair[1], steps=1)
                proposed_sentences.append(result_df.iloc[1]["proposal_sentence"])
            all_outputs.append({
                "data_idx": i,
                "proposal_sentences": proposed_sentences,
                "target_fusion": kwargs["target_fusion"],
                "fusion_interpolation": kwargs["fusion_interpolation"]
            })
    pd.DataFrame(all_outputs).to_csv("dump/interpolation_results.csv", index=False)


if __name__ == "__main__":
    random.seed = 24
    np.random.seed(24)
    torch.random.manual_seed(24)
    dataset = load_dataset("imdb")["train"].to_pandas()
    pair_df = draw_random_pairs(dataset, subset_per_class_count=5, max_len=512)
    # pair_df = pair_df.drop(columns=["sent2", "label2"]).rename(columns={"sent1": "text", "label1": "label"})
    # pd.to_csv("dump/imdb_5.csv", index=False)
    # sentence_pair = list(zip(pair_df["sent1"], pair_df["sent2"]))

    # mhss = MetropolisHastingSentenceSampler(
    #     sampler_model_name="distilbert-base-uncased",
    #     acceptor_semantic_model_name="sentence-transformers/all-mpnet-base-v2",
    #     acceptor_fluency_model_name="distilgpt2",
    #     method="span_pm",
    #     lambda_semantic=10,
    #     lambda_fluency=10,
    #     target_fusion="logits",
    #     fusion_aggregation="closest",
    #     fusion_interpolation="exp",
    #     init_temp=1.0,
    #     annealing_rate=1e-4,
    #     min_temp=0.1,
    #     cuda=True
    # )
    # intrinsic_evaluation(sentence_pair, mhss, num_runs=5, exp_str="span_pm_ppl10", visualize=False)

    # interpolation_evaluation()

    # exp_kwargs_list = [
    #     # {"target_fusion": "embs", "fusion_aggregation": "closest", "fusion_interpolation": "linear"},
    #     # {"target_fusion": "embs", "fusion_aggregation": "closest", "fusion_interpolation": "exp"},
    #     # {"target_fusion": "embs", "fusion_aggregation": "local_5", "fusion_interpolation": "exp"},
    #     # {"target_fusion": "embs", "fusion_aggregation": "local_5", "fusion_interpolation": "polar"},
    #     # {"target_fusion": "hiddens", "fusion_aggregation": "closest", "fusion_interpolation": "linear"},
    #     {"target_fusion": "hiddens", "fusion_aggregation": "closest", "fusion_interpolation": "exp"},
    #     {"target_fusion": "hiddens", "fusion_aggregation": "closest", "fusion_interpolation": "polar"},
    #     {"target_fusion": "hiddens", "fusion_aggregation": "local_5", "fusion_interpolation": "exp"},
    #     {"target_fusion": "hiddens", "fusion_aggregation": "local_5", "fusion_interpolation": "polar"},
    #     {"target_fusion": "logits", "fusion_aggregation": "closest", "fusion_interpolation": "exp", "init_temp": 1.0},
    #     {"init_temp": 20.0}
    # ]
    # for exp_kwargs in exp_kwargs_list:
    #     kwargs.update(exp_kwargs)
    #     mhss = MetropolisHastingSentenceSampler(**kwargs)
    #     exp_str = f"{kwargs['target_fusion']}_{kwargs['fusion_aggregation']}_{kwargs['fusion_interpolation']}" \
    #         if kwargs["init_temp"] == 10.0 else f"init_temp{kwargs['init_temp']}"
    #     intrinsic_evaluation(
    #         sentence_pair, mhss, num_runs=5,
    #         exp_str=exp_str,
    #         visualize=False
    #     )

    augment_dataset(pair_df, None, 1, aug_data_dir="./dump/init_temp1.0")
