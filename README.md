# Metropolis-Hasting-Sentence-Sampling
Sampling / Interpolating sentence using transformers with Metropolis-Hasting like algorithm

## Code-base Structure
In this repo, you can find the code to reproduce the results in the report

`sampler.py` includes all code related to sampling sentences, from span selection, higher order span score calculation, 
to the actual Metropolis-Hasting sampling algorithm

`proposal_acceptor.py` contains the class which calculates proposal log probs based on semantics and fluency.

`experiments.py` contains all experiment code used to run sampling, probing, and visualization.

`finetune.py` is used to finetune downstream classification model using augmented sentences.

## How this repo works:

I built this repo along with the flow of my experimental procedure:
1. I searched through hyperparameters to find optimal set of parameters. For each experiment, I would run something
like this in the `experiments.py`:
```python
random.seed(24)
np.random.seed(24)
torch.random.manual_seed(24)
dataset = load_dataset("imdb")["train"].to_pandas()
pair_df = draw_random_pairs(dataset, subset_per_class_count=5, max_len=512)
pair_df = pair_df.drop(columns=["sent2", "label2"]).rename(columns={"sent1": "text", "label1": "label"})
pd.to_csv("dump/imdb_5.csv", index=False)
sentence_pair = list(zip(pair_df["sent1"], pair_df["sent2"]))

mhss = MetropolisHastingSentenceSampler(
    sampler_model_name="distilbert-base-uncased",
    acceptor_semantic_model_name="sentence-transformers/all-mpnet-base-v2",
    acceptor_fluency_model_name="distilgpt2",
    method="span_pm",
    lambda_semantic=10,
    lambda_fluency=10,
    target_fusion="logits",
    fusion_aggregation="closest",
    fusion_interpolation="exp",
    init_temp=1.0,
    annealing_rate=1e-4,
    min_temp=0.1,
    cuda=True
)
intrinsic_evaluation(sentence_pair, mhss, num_runs=5, exp_str="my_experiment_trial", visualize=False)
```
This samples 5 data points from IMDB train dataset for each class, and performs 5 runs of sampling experiments for each pair
Each experiment like this can run from 2.5 hrs to 5 hrs long. For span based methods, especially `pm`, 
`mask-one` (or `one`), a GPU is needed to parallelize the model calls. Each run outputs results in `dump/[EXP_STR]`
where you can specific what `EXP_STR` is. `dump/init_temp1.0` is an example output directory, and `dump/baseline` 
contains original sentences used in all interpolation experiments.


2. After intrinsic evaluations, then I can do visualizations if I want on the trial by calling:
```python
plot_data("my_experiment_trial")
```

3. Before extrinsic evaluation, make sure to call generate augmentation data first with set seed in `experiments.py`:
```python
augment_dataset(pair_df, None, 1, aug_data_dir=f"./dump/my_experiment_trial")
```
This way, the augmentation wouldn't select other random samples if you were finetuning with multiple seeds.

4. For extrinsic evaluation, you can call something like this in the `finetune.py` script:
```python
random.seed(24)
np.random.seed(24)
torch.random.manual_seed(24)
model_name = "bert-base-uncased"  
aug_data_dir_list = ["my_experiment_trial"] 
for aug_dir in aug_data_dir_list:
    print(f" =========== start finetuning {aug_dir} ==========")
    for seed in [ 11,12,13, 22,25]: # 11,12,13, 22,25
        print(f"++++++++ seed {seed} ++++++++")
        try:
            res = finetune_model(model_name, aug_dir, seed, 2)
            print(res)
        except:
            print("exception occured")
            res = finetune_model(model_name, aug_dir, seed, 1)
            print(res)
```
Metrics will be logged in WanDB as well as standard output.


