#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import FriendsDataset
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os

from word2vec import *
from sgd import *

# Check Python Version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

import pandas as pd
from pandas.plotting import table

from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_similarity(v1, v2):
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

dir = "../../reports/word2vec/"
if not os.path.exists(dir):
    os.makedirs(dir)

word_vectors = {}
tokens = {}


for person in ["phoebe", "chandler"]:

    dataset = FriendsDataset(person=person)
    ts = dataset.tokens()
    tokens[person] = ts
    nWords = len(tokens)

    model_state_path = f"../../models/word2vec/{person}.word2vec.pickle"
    model_params_path = f"../../models/word2vec/{person}.word2vec.npy"

    params = np.load(model_params_path)
    with open(model_state_path, "rb") as f:
        state = pickle.load(f)

    word_vectors[person] = params

l1_tokens = list(tokens['phoebe'].keys())
l2_tokens = list(tokens['chandler'].keys())
common_tokens = list(set.intersection(set(l1_tokens), set(l2_tokens)))
print(f"Number of common words: {len(common_tokens)}")
words_cmp = []
sim_scores = []
for word in common_tokens:
    t1 = tokens['chandler'][word]
    t2 = tokens['phoebe'][word]

    wv1 = word_vectors['chandler'][t1]
    wv2 = word_vectors['phoebe'][t2]

    sim_score = get_cosine_similarity(wv1, wv2)

    words_cmp.append(word)
    sim_scores.append(sim_score)

assert len(sim_scores) == len(words_cmp)

n = (len(words_cmp) // 20) + 1

ax = plt.subplot(frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

for i in range(n):

    w = words_cmp[i*20 : (i+1)*20].copy()
    s = sim_scores[i*20 : (i+1)*20].copy()

    df = pd.DataFrame( { "Word": w, "Cosine Similarity": s })

    table(ax, df, loc='center')
    x = plt.gcf()
    x.savefig(f'../../reports/word2vec/common_words_{i}.png')

