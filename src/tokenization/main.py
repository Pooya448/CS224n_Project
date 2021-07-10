# I used https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb#scrollTo=ee9W6wGnVteW

import sentencepiece as spm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.plotting import table

temp_dir = "../../temps/tokenization/"

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

data_path = "../../data/sentence_tokenize/ALL.txt"
with open(data_path, 'r') as f:
    raw_data = f.readlines()

data = [x for x in raw_data if len(x) > 1]
n = len(data)
split = np.array_split(data, 5)

print(n)
sizes = [n/100, n/70, n/30, n/20, 11737]
results = []
best_percentage = 100
best_size = 100
best_i = 100

for i, size in enumerate(sizes):
    temp_copy = split.copy()
    test_data = temp_copy.pop(i)
    train_data = np.concatenate(temp_copy).tolist()

    with open("../../temps/tokenization/train.txt", 'w+') as f:
        f.write('\n'.join(train_data))

    spm.SentencePieceTrainer.train(f'--input=../../temps/tokenization/train.txt --model_prefix=../../temps/tokenization/m{i} --vocab_size={size} --model_type=word'.format(i, sizes[i]))

    sp = spm.SentencePieceProcessor()
    sp.load(f'm{i}.model')

    token_count = 0
    unk_count = 0
    for s in test_data:
        encoded = sp.encode_as_ids(s)

        token_count += len(encoded)
        unk_count += encoded.count(0)

    percentage = (float(unk_count) / token_count) * 100

    if percentage < best_percentage:
        best_percentage = percentage
        best_size = size
        best_i = i

    results.append(percentage)

df = pd.DataFrame( {'Sizes': sizes, "Percentage": results})

ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table(ax, df, loc='center')
x = plt.gcf()

save_dir = "../../reports/tokenization/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

x.savefig(f'{save_dir}percentages.png')

model_dir = "../../models/tokenization/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

os.system(f"cp ../../temps/tokenization/m{best_i}.model ../../models/tokenization/tokenization.model")
os.system(f"cp ../../temps/tokenization/m{best_i}.vocab ../../models/tokenization/tokenization.vocab")

print(f'Best unk percentage -> Size:{best_size}')
