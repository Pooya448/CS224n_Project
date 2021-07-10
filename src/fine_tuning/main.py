import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import torch
import sys
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import pipeline
from casual_lm import train
import math

os.system("python3 data.py")

start_sents = [
    'coffee',
    'smelly',
    'monica',
    'joey',
    'home',
]

for person in ['chandler', 'phoebe']:

    data_dir = f"../../data/fine_tuning/"

    dev_path = data_dir + f"dev_{person}.txt"
    train_path = data_dir + f"train_{person}.txt"

    data_paths = {}

    if train_path is not None:
        data_paths["train"] = train_path

    if dev_path is not None:
        data_paths["validation"] = dev_path


    extension = train_path.split(".")[-1]
    if extension == "txt":
        extension = "text"

    datasets = load_dataset(extension, data_files=data_paths)

    data = DatasetDict({"train":datasets["train"],
                        "validation":datasets["validation"]})

    save_dir = dir = f"../../models/fine_tuning/{person}.bert_lm"

    model, perplexity, tok = train(data, save_dir)
    model.save_pretrained(save_dir)

    report_dir = "../../reports/fine_tuning/"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    with open(report_dir + "perplexity.txt", "a+") as f:
        f.write(f"{person} -> perplexity = {str(perplexity)}")
        f.write('\n')

    text_generation = pipeline("text-generation", model=model, tokenizer=tok, device=0)

    with open(report_dir + f"generated_{person}.txt", "w+") as f:
        for s in start_sents:
            gen = text_generation(s, max_length = 20)
            f.write(str(gen))
            f.write('\n')
