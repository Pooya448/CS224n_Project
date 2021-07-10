import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
import os

def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--person', type=str)
    parser.add_argument('--genlength', type=int, default=10)
    parser.add_argument('--sequence-length', type=int, default=15)
    args = parser.parse_args()

    model_dir = f"../../models/language_model/"

    dataset = Dataset(args, torch.device('cpu'))
    model = Model(dataset, torch.device('cpu'))

    loaded_states = torch.load(model_dir + f"{args.person}.language_model.pth")
    model.load_state_dict(loaded_states)

    report_dir = "../../reports/language_model/"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    with open(report_dir + f"report_{args.person}.txt", "a+") as fp:
        gen_text = ' '.join(predict(dataset, model, text=args.input, next_words=args.genlength))
        fp.write(str(gen_text))
        fp.write('\n')
