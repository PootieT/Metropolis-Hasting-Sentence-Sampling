import random
from typing import *

import datasets
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import Trainer, is_datasets_available

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import Dataset, load_metric

import evaluate
from transformers.trainer_utils import seed_worker

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from typing import Union, Optional, List, Dict

from transformers.utils import PaddingStrategy




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
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        # class_weight = 1. / torch.Tensor([328, 2496, 28]).to(model.device)
        class_weight = (1-torch.Tensor([328, 2496, 28]).to(model.device)/(328+2496+28))
        loss_fct = nn.CrossEntropyLoss(weight=class_weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_train_datasets(model: str, exp_str: str) -> Dataset:
    train_df = pd.read_csv(f"dump/{exp_str}/data_5.csv")
    tokenizer = AutoTokenizer.from_pretrained(model)
    train_dataset = Dataset.from_pandas(train_df)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True).shuffle(seed=42)

    return tokenized_train_datasets


def custom_compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # print("========= classification report =========")
    # print(classification_report(labels, predictions))
    # print("========= classification report end =========")
    report = classification_report(labels, predictions, target_names=["neg", "neu", "pos"], output_dict=True)
    report_flat = {}
    for k, v in report.items():
        if isinstance(v, dict):
            for vk, vv in v.items():
                report_flat[f"{k}_{vk}".replace(' ', '_')] = vv
    return report_flat


if __name__ == "__main__":
    random.seed = 24
    np.random.seed(24)
    torch.random.manual_seed(24)
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    training_args = TrainingArguments(
        output_dir="cardi-tweet",
        evaluation_strategy="epoch",
        num_train_epochs=30,
        do_train=True,
        do_eval=True,
        do_predict=False,
        learning_rate=3e-5,
        lr_scheduler_type="fixed",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        no_cuda=False,
        logging_steps=1
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=custom_compute_metrics,  # compute_metrics
        data_collator=OurDataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_name))
    )
    trainer.train()
