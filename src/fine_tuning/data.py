import os
import math
import random

data_dir = f"../../data/fine_tuning/"
if os.path.exists(data_dir):
    os.system(f"rm -rf {data_dir}")
    os.makedirs(data_dir)
else:
    os.makedirs(data_dir)

for person in ['CHANDLER', 'PHOEBE']:

    sents_path = f"../../data/sentence_tokenize/{person}.txt"
    with open(sents_path, 'r') as f:
        sents = f.readlines()

    len_train = math.floor(0.8 * len(sents))

    random.shuffle(sents)

    train = sents[: len_train].copy()
    dev = sents[len_train : ].copy()

    with open(data_dir + f"dev_{str.lower(person)}.txt", 'w+') as f:
        lines = '\n'.join([x for x in dev if len(x) > 2])
        f.write(lines)

    with open(data_dir + f"train_{str.lower(person)}.txt", 'w+') as f:
        lines = '\n'.join([x for x in train if len(x) > 2])
        f.write(lines)
