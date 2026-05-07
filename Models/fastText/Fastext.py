import fasttext
import itertools
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root

# 1) train a first fastText model
# 2) use manual grid search to build another model and then compare
# 3) begin analysis of the vector space with get_nearest_neighbors()

# (no train/val split needed for unsupervised)
nigeria_spoken  = BASE_DIR / "Preprocessing/preprocessed_word2vec/nigeria_spoken.txt"
nigeria_written = BASE_DIR / "Preprocessing/preprocessed_word2vec/nigeria_written.txt"

import shutil

# ── Combine spoken and written ─────────────────────────────────────────────────
nigeria_combined = BASE_DIR / "Preprocessing/preprocessed_word2vec/nigeria_combined.txt"

with open(nigeria_combined, "w", encoding="utf-8") as out:
    for path in [nigeria_spoken, nigeria_written]:
        with open(path, "r", encoding="utf-8") as f:
            shutil.copyfileobj(f, out)

print(f"Combined file created: {nigeria_combined}")

## 1) TRAINING

# Skipgram model :
model = fasttext.train_unsupervised(str(nigeria_combined), model='skipgram')

output_dir = BASE_DIR / "Models/fastText"
output_dir.mkdir(parents=True, exist_ok=True)

# Save first model
model.save_model(str(output_dir / "nigeria_fasttext.bin"))
print(f"Model saved → {output_dir / 'nigeria_fasttext.bin'}")

# Load model
model = fasttext.load_model(str(output_dir / "nigeria_fasttext.bin"))

## 2) MANUAL GRID SEARCH
# Hyperparameter grid eac key maps to a list of values to try.
# itertools.product will generate every possible combination.
parameters = {
    'model':        ['cbow', 'skipgram'],  # cbow: predict word from context; skipgram: predict context from word
    'lr':           [0.05],                # learning rate
    'dim':          [100, 200],            # size of word vectors
    'ws':           [5],                   # context window size (words to the left and right)
    'epoch':        [5],                   # number of passes over the training data
    'minCount':     [5],                   # ignore words appearing fewer than 5 times
    'minn':         [3],                   # minimum character n-gram size (FastText specific)
    'maxn':         [6],                   # maximum character n-gram size (FastText specific)
    'neg':          [5],                   # number of negative samples (only used with loss='ns')
    'loss':         ['ns', 'hs'],          # ns: negative sampling (faster); hs: hierarchical softmax
    'bucket':       [2000000],             # number of buckets for hashing character n-grams
    'lrUpdateRate': [100],                 # how often the learning rate is updated
    't':            [0.0001],              # threshold for downsampling frequent words
}

best_model = None
best_score = -1

# Train one model per parameter combination and keep the best one
for values in itertools.product(*parameters.values()):
    params = dict(zip(parameters.keys(), values))
    m = fasttext.train_unsupervised(str(nigeria_combined), **params)
    
    # Evaluate by checking nearest neighbors of "nation" as a proxy for embedding quality
    # Score = number of unique neighbors returned (higher = more diverse semantic space)
    neighbors = m.get_nearest_neighbors("nation")
    score = len(set(w for _, w in neighbors))
    
    if score > best_score:
        best_score = score
        best_model = m
        print(f"New best: {params} → score {score}")

# Save best model from grid search
best_model.save_model(str(output_dir / "nigeria_fasttext_best.bin"))
print(f"Best model saved → {output_dir / 'nigeria_fasttext_best.bin'}")

## 3) VECTOR SIMILARITY SEARCH
# Returns the 10 nearest neighbors of "nationhood" with their cosine similarity scores
print(model.get_nearest_neighbors("nationhood"))