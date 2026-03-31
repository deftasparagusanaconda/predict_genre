from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import joblib
import sys
import pandas as pd

N_SPLITS = 5
DEFAULT_INPUT_PATH = 'features.csv'
DEFAULT_OUTPUT_PATH = 'model.pkl'

if len(sys.argv) > 2:
    input_path = sys.argv[1]
else:
    input_path = DEFAULT_INPUT_PATH 

if len(sys.argv) > 3:
    output_path = sys.argv[2]
else:
    output_path = DEFAULT_OUTPUT_PATH 

df = pd.read_csv(input_path)

model = make_pipeline(
    StandardScaler(),
    HistGradientBoostingClassifier()
)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
x = df.drop(columns=['Unnamed: 0', "genre"]).values
y = df["genre"].values

scores = cross_val_score(model, x, y, cv=cv)

print(f'the model trained on {((N_SPLITS-1)/N_SPLITS)*100}% and tested on {(1/N_SPLITS)*100}% of data')
print(f'accuracy:', scores.mean())
print(f'variance:', scores.std())

model.fit(x, y)
joblib.dump(model, output_path)





from .extractor import extract_features
print(model.predict([extract_features('~/github/predict-genre/data/genres_original/blues/blues.00000.wav')]))
