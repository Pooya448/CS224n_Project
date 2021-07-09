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


def plot(words, tokens, wVecs, figname):
    visualizeIdx = [tokens[word] for word in words]
    visualizeVecs = wVecs[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in range(len(words)):
        plt.text(coord[i,0], coord[i,1], words[i],
            bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    dir = "../../reports/word2vec/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.savefig(dir + f'{figname}_wvecs.png')


wVectors = {}
pTokens = {}

for person in ["phoebe", "chandler"]:
    # Reset the random seed to make sure that everyone gets the same results
    random.seed(314)
    dataset = FriendsDataset(person=person)
    tokens = dataset.tokens()
    pTokens[person] = tokens
    nWords = len(tokens)

    # We are going to train 10-dimensional vectors for this assignment
    dimVectors = 10

    # Context size
    C = 5

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)

    startTime=time.time()
    wordVectors = np.concatenate(
        ((np.random.rand(nWords, dimVectors) - 0.5) /
           dimVectors, np.zeros((nWords, dimVectors))),
        axis=0)
    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
            negSamplingLossAndGradient),
        wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10, label=person)
    # Note that normalization is not called here. This is not a bug,
    # normalizing during training loses the notion of length.

    print("sanity check: cost at convergence should be around or below 10")
    print("training took %d seconds" % (time.time() - startTime))

    # concatenate the input and output word vectors
    wordVectors = np.concatenate(
        (wordVectors[:nWords,:], wordVectors[nWords:,:]),
        axis=0)

    wVectors[person] = wordVectors


common_words = set.intersection(set(pTokens['phoebe'].keys()), set(pTokens['chandler'].keys()))

sim = {}

for word in common_words:

    c_vec = wVectors['chandler'][pTokens['chandler'][word], :]
    p_vec = wVectors['phoebe'][pTokens['phoebe'][word], :]

    from scipy import spatial

    coSimilarity = 1 - spatial.distance.cosine(c_vec, p_vec)
    sim[word] = coSimilarity

words_sorted = sorted(sim.items(), key=lambda item: item[1])

sim_words = [item[0] for item in words_sorted[:10]]
diff_words = [item[0] for item in words_sorted[-10:]]

plot(sim_words, pTokens['chandler'], wVectors['chandler'], 'chandler_similar')
plot(sim_words, pTokens['phoebe'], wVectors['phoebe'], 'phoebe_similar')

plot(diff_words, pTokens['chandler'], wVectors['chandler'], 'chandler_diff')
plot(diff_words, pTokens['phoebe'], wVectors['phoebe'], 'phoebe_diff')
