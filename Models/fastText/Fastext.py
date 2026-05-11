# code from: https://pypi.org/project/fasttext/#installation

import fasttext

# 1) split the data before training
# 2) train a first fastText model
# 3) use gridSearch to build another model and then compare
# 4) begin analysis of the vector space with get_nearest_neighbors()


## 1) SPLIT of the DATA

## 2) TRAINING

# Skipgram model :
model = fasttext.train_unsupervised('data.txt', model='skipgram')

# or, cbow model :
model = fasttext.train_unsupervised('data.txt', model='cbow')


# Save trained model object by calling the function save_model
model.save_model("model_filename.bin")

# Retrieve saved model using function load_model
model = fasttext.load_model("model_filename.bin")



## 3) GRIDSEARCH
from sklearn.model_selection import GridSearchCV

# Define hyperparameters potential values before GridSearch
parameters = {
    'model': ['cbow','skipgram']
    'lr': [0.05],
    'dim': [100, 200],
    'ws': [5],
    'epoch': [5],
    'minCount': [5],
    'minn': [3],
    'maxn': [6],
    'neg': [5],
    'wordNgrams': [1],
    'loss': [ns, hs, softmax, ova],
    'bucket': [20000000],
    'lrUpdateRate': [100],
    't': [0.0001],
    'verbose': [2],
}

svc = svm.SVC()
clf_grid_search = GridSearchCV(svc, parameters)

# Create an instance of fastText with GridSearch and fit the data
clf_grid_search.fit(X_train, y_train)


## 4) VECTOR SIMILARITY SEARCH

model.get_nearest_neighbors("nationhood")
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

india           = BASE_DIR / "Preprocessing/preprocessed_word2vec/india.txt"
usa             = BASE_DIR / "Preprocessing/preprocessed_word2vec/usa.txt"
jamaica         = BASE_DIR / "Preprocessing/preprocessed_word2vec/jamaica.txt"


import shutil

# ── Combine spoken and written ─────────────────────────────────────────────────
nigeria_combined = BASE_DIR / "Preprocessing/preprocessed_word2vec/nigeria_combined.txt"

with open(nigeria_combined, "w", encoding="utf-8") as out:
    for path in [nigeria_spoken, nigeria_written]:
        with open(path, "r", encoding="utf-8") as f:
            shutil.copyfileobj(f, out)

print(f"Combined file created: {nigeria_combined}")

## 1) TRAINING

# Train and save a fasText model given a dataset
def train_and_save(path, corpus_name, output_dir):
    
    model = fasttext.train_unsupervised(str(path), model='skipgram')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_model(str(output_dir / f"{corpus_name}_fasttext.bin"))
    print(f"Model saved → {output_dir} / '{str(corpus_name)}_fasttext.bin'")

    return model


output_dir = BASE_DIR / "Models/fastText"
output_dir.mkdir(parents=True, exist_ok=True)


# 1.1 - NIGERIA MODEL

# train and save model
model_nigeria = train_and_save(nigeria_combined, 'nigeria', output_dir) # skipgram model


# Load model
model_nigeria = fasttext.load_model(str(output_dir / "nigeria_fasttext.bin"))

# 1.2 - JAMAICA MODEL

# train and save model
model_jamaica = train_and_save(jamaica, 'jamaica', output_dir) # skipgram model


# Load model
model_jamaica = fasttext.load_model(str(output_dir / "jamaica_fasttext.bin"))

# 1.3 - USA MODEL

# train and save model
model_usa = train_and_save(usa, 'usa', output_dir) # skipgram model

# Load model
model_usa = fasttext.load_model(str(output_dir / "usa_fasttext.bin"))

# 1.4 - INDIA MODEL

# train and save model
model_india = train_and_save(jamaica, 'india', output_dir) # skipgram model

# Load model
model_india = fasttext.load_model(str(output_dir / "india_fasttext.bin"))


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

corpora = {
        'nigeria_combined': 'nigeria_fasttext_best.bin', 
        'india': 'india_fasttext_best.bin', 
        'jamaica': 'jamaica_fasttext_best.bin', 
        'usa': 'usa_fasttext_best.bin'
}

# Train one model per parameter combination and keep the best one
for corpus_name, path in corpora.items():
    for values in itertools.product(*parameters.values()):
        params = dict(zip(parameters.keys(), values))
        m = fasttext.train_unsupervised(str(corpus), **params)
    
        # Evaluate by checking nearest neighbors of "nation" as a proxy for embedding quality
        # Score = number of unique neighbors returned (higher = more diverse semantic space)
        neighbors = m.get_nearest_neighbors("nation")
        score = len(set(w for _, w in neighbors))
    
        if score > best_score:
            best_score = score
            best_model = m
            print(f"New best: {params} → score {score}")

    # Save best model from grid search
    best_model.save_model(str(output_dir / data_labels[corpus]))
    print(f"Best model saved → {output_dir / data_labels[corpus]}")

## 3) VECTOR SIMILARITY SEARCH
# Returns the 10 nearest neighbors of "nationhood" with their cosine similarity scores
print(model_nigeria.get_nearest_neighbors("nation"))
print(model_india.get_nearest_neighbors("nation"))
print(model_jamaica.get_nearest_neighbors("nation"))
print(model_usa.get_nearest_neighbors("nation"))
