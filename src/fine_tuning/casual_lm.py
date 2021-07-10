
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import torch
import sys
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
import math

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train(dataset, path):

    block_size = 128
    model_checkpoint = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        path,
        do_train= True,
        do_eval = True,
        num_train_epochs = 5,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=True,
        push_to_hub_model_id=f"{model_checkpoint}-finetuned-wikitext2",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    perplexity= math.exp(eval_results['eval_loss'])
    print(f"Perplexity: {perplexity:.2f}")

    return model, perplexity, tokenizer
