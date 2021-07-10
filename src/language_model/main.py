import os

person = ['phoebe', 'chandler']

for p in person:
    os.system(f"python3 train.py --person {p}"")

os.system("python3 complete_sents.py")
