import ast
import os.path
import random
from typing import *

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from typing import Union, Optional, List, Dict

from transformers.utils import PaddingStrategy
import wandb

from experiments import augment_dataset, draw_random_pairs


@dataclass
class OurDataCollatorWithPadding:
    # Cite: SimCSE: https://github.com/princeton-nlp/SimCSE/blob/d868602e4679c0a654e58236d2cfb3e1b6c9c1fc/train.py
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[
        str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask']
        bs = len(features)
        if bs == 0:
            return

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = batch["label"]
        del batch["label"]
        return batch


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        del inputs["labels"]
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
        return (loss, outputs) if return_outputs else loss


def load_train_val_datasets(model: str, exp_str: str) -> Tuple[Dataset, Dataset]:
    aug_data_path = f"dump/{exp_str}/aug.csv"
    if not os.path.exists(aug_data_path):
        dataset = load_dataset("imdb")["train"].to_pandas()
        pair_df = draw_random_pairs(dataset, subset_per_class_count=5, max_len=512)
        augment_dataset(pair_df, None, 1, aug_data_dir=f"./dump/{exp_str}")
    train_df = pd.read_csv(aug_data_path)
    tokenizer = AutoTokenizer.from_pretrained(model)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = load_dataset("imdb")["test"]

    def tokenize_function(examples):
        if isinstance(examples["label"][0], str):
            examples["label"] = torch.tensor([ast.literal_eval(l) for l in examples["label"]])
        elif isinstance(examples["label"][0], int):
            examples["label"] = torch.tensor([[1 if i == ex else 0 for i in range(2)] for ex in examples["label"]])
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True).shuffle(seed=42)
    tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True).shuffle(seed=42)
    return tokenized_train_datasets, tokenized_val_datasets


def custom_compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # print("========= classification report =========")
    # print(classification_report(labels, predictions))
    # print("========= classification report end =========")
    report = classification_report(labels.argmax(1), predictions, target_names=["neg", "pos"], output_dict=True)
    report_flat = {}
    for k, v in report.items():
        if isinstance(v, dict):
            for vk, vv in v.items():
                report_flat[f"{k}_{vk}".replace(' ', '_')] = vv
    report_flat["accuracy"] = sum(labels.argmax(1)== predictions)/len(predictions)
    return report_flat


def finetune_model(model_name, aug_data_dir, seed, bs: 2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train_dataset, val_dataset = load_train_val_datasets(model_name, aug_data_dir)
    training_args = TrainingArguments(
        output_dir=f"dump/{aug_data_dir}",
        evaluation_strategy="epoch",
        eval_delay=29,
        num_train_epochs=30,
        do_train=True,
        do_eval=True,
        do_predict=False,
        learning_rate=3e-5,
        lr_scheduler_type="constant",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=int(8/bs),
        no_cuda=False,
        logging_steps=1,
        seed=seed,
        run_name=f"{aug_data_dir}_seed{seed}"
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=custom_compute_metrics,  # compute_metrics
        data_collator=OurDataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_name))
    )
    result = trainer.train()
    wandb.finish()
    return result


if __name__ == "__main__":
    # random.seed(24)
    np.random.seed(24)
    torch.random.manual_seed(24)
    model_name = "bert-base-uncased"  # "baseline","no_fusion", "hiddens_closest_linear" "init_temp1.0" "word_pm" "span_static" "span_mask_one"
    aug_data_dir_list = [ "span_pm"] #   , ,  ,  , , "span_pm_ppl10"
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

