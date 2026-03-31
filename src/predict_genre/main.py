from .extractor import extract_features
import joblib
import sys

from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'model.pkl'

def main():
    if len(sys.argv) < 2:
        print('usage: predict_genre path/to/music')
        sys.exit(1)
    path = sys.argv[1]

    features = extract_features(path)
    model = joblib.load(MODEL_PATH)

    probs = model.predict_proba([features])[0]
    
    for genre, prob in zip(model.classes_, probs):
        print(f"{genre:12s} {prob * 100:6.0f}%")



