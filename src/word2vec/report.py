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
    tokens = dataset.tokens()
    tokens[person] = tokens
    nWords = len(tokens)

    model_state_path = f"../../models/word2vec/{person}.word2vec.pickle"
    model_params_path = f"../../models/word2vec/{person}.word2vec.npy"

    params = np.load(model_params_path)
    with open(model_state_path, "rb") as f:
        state = pickle.load(f)

    word_vectors[person] = params

l1_tokens = list(tokens['phoebe'].keys())
l2_tokens = list(tokens['chandler'].keys())
common_tokens = list(set.intersection(l1_tokens, l2_tokens))

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

df = pd.DataFrame( { "Word": words_cmp, "Cosine Similarity": sim_scores })

ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table(ax, df, loc='center')
x = plt.gcf()
x.savefig('../../reports/word2vec/common_words.png')
