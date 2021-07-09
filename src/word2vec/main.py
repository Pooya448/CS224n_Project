import os

os.system("python3 run.py")

dir = "../../models/word2vec/"
if not os.path.exists(dir):
    os.makedirs(dir)

os.system("cp ../../temps/word2vec/chandler.word2vec.5000.npy ../../models/word2vec/chandler.word2vec.npy")
os.system("cp ../../temps/word2vec/chandler.word2vec.5000.pickle ../../models/word2vec/chandler.word2vec.pickle")
os.system("cp ../../temps/word2vec/phoebe.word2vec.5000.npy ../../models/word2vec/phoebe.word2vec.npy")
os.system("cp ../../temps/word2vec/phoebe.word2vec.5000.pickle ../../models/word2vec/phoebe.word2vec.pickle")


os.system("python3 report.py")
