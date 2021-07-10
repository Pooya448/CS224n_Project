cd src/word2vec
python3 main.py

cd ../tokenization
python3 main.py

cd ../parsing
python3 main.py

cd ../language_model
python3 main.py

cd ../fine_tuning
python3 main.py

cd ../../latex
latexmk -pdf report.tex
