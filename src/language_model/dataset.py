import torch
import pandas as pd
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        device,
    ):
        self.args = args
        self.person = args.person
        self.device = device
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        with open(f'../../data/sentence_tokenize/{str.upper(self.person)}.txt', 'r') as f:
            train_sents = [line.strip() for line in f.readlines()]
        train_tokens = ' '.join(train_sents).split(' ')
        return train_tokens

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length], device=self.device),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1], device=self.device),
        )
