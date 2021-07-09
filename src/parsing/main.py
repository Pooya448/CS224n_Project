import os

os.system("cp ../../data/parsing/test.conll ./data/test.conll")
os.system("cp ../../data/parsing/test.conll ./data/test.gold.conll")

os.system("python3 run.py")

if not os.path.exists("../../models/parsing/"):
    os.makedirs("../../models/parsing/")

os.system("cp ../../temps/parsing/model.weights ../../models/parsing/model.weights")
