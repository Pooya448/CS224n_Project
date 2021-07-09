import os

os.system("cp ../../data/parsing/test.conll ./data/test.conll")
os.system("cp ../../data/parsing/test.conll ./data/test.gold.conll")

os.system("python3 run.py")
