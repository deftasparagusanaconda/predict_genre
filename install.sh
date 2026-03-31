#!/bin/sh

echo installing python module...
echo
pip install -e .

echo extracting audio features...
echo
cd src/predict_genre
python extractor.py ../../data/genres_original/

echo training prediction model...
echo
python trainer.py

echo you may now do \`predict_genre filepath\`
