import os

start_sents = [
    'coffee',
    'smelly cat',
    'monica',
    'joey',
    'homeless',
    'paleontology',
]
person = ['phoebe', 'chandler']

for p in person:
    for s in start_sents:
        os.system(f"python3 query.py --person {p} --input {s}")
