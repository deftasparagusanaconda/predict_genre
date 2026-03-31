# recurse through a directory tree, writing 

import sys  # for command line arguments
import librosa  # to extract mfccs
import pathlib  # for directory crawling
import numpy as np
import pandas as pd
from collections.abc import Sequence

N_MFCC: int = 40
DEFAULT_OUTPUT: str = 'features.csv'

def extract_features(filepath) -> Sequence[float]:
    y, sr = librosa.load(filepath, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfccs_mean = mfccs.mean(axis=1)
    mfccs_var = mfccs.var(axis=1)
    return np.array([*mfccs_mean] + [*mfccs_var])

if __name__ == '__main__':
# no path given as command line argument
    if len(sys.argv) < 2:
        print("usage: python extractor.py path/to/data/ path/to/output/")
        sys.exit(1)

    path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_OUTPUT

    rows = []

    for dir in pathlib.Path(path).iterdir():
        print('now exploring', dir)
        genre = dir.stem

        #if genre not in mfccs_per_genre:
        #    mfccs_per_genre[genre] = np.array(dtype = float)
        
        for file in dir.iterdir():
            try:
                features = extract_features(file)
                rows.append([genre, *features[0]])
            except Exception as e:
                print(f'skipping {file}: {e}')
                continue

    df = pd.DataFrame(rows, columns = ['genre'] + [f'mfcc_mean_{i}' for i in range(1, N_MFCC + 1)] + [f'mfcc_var_{i}' for i in range(1, N_MFCC + 1)])

# write to output
    df.to_csv(output)
