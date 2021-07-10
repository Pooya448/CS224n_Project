import os

start_sents = [
    'coffee',
    'smelly',
    'monica',
    'joey',
    'home',
]
person = ['phoebe', 'chandler']

for p in person:
    for s in start_sents:
        os.system(f"python3 query.py --person {p} --input {s}")
