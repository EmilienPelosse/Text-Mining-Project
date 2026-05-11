# Word2Vec model training on ICE Nigeria corpus

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import itertools
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root

# Paths
nigeria_spoken   = BASE_DIR / "Preprocessing/preprocessed_word2vec/nigeria_spoken.txt"
nigeria_written  = BASE_DIR / "Preprocessing/preprocessed_word2vec/nigeria_written.txt"
nigeria_combined = BASE_DIR / "Preprocessing/preprocessed_word2vec/nigeria_combined.txt"

india           = BASE_DIR / "Preprocessing/preprocessed_word2vec/india.txt"
usa             = BASE_DIR / "Preprocessing/preprocessed_word2vec/usa.txt"
jamaica         = BASE_DIR / "Preprocessing/preprocessed_word2vec/jamaica.txt"


output_dir = BASE_DIR / "Models/Word2Vec"
output_dir.mkdir(parents=True, exist_ok=True)

# 1) TRAINING

def train_and_save(path, corpus_name, output_dir):
    # train model
    model = Word2Vec(
            sentences=LineSentence(str(path)),
            vector_size=100,    # size of word vectors
            window=5,           # context window size
            min_count=5,        # ignore words appearing fewer than 5 times
            sg=1,               # 1 = skipgram, 0 = cbow
            epochs=5
    )

    # Save model
    model.save(str(output_dir / f"{corpus_name}_word2vec.model"))
    print(f"Model saved → {output_dir / f'{corpus_name}_word2vec.model'}")

    return model


# Load model
model = Word2Vec.load(str(output_dir / "nigeria_word2vec.model"))


# 2) MANUAL GRID SEARCH
parameters = {
    'sg':           [0, 1],         # 0 = cbow, 1 = skipgram
    'vector_size':  [100, 200],     # size of word vectors
    'window':       [5],            # context window size
    'min_count':    [5],            # minimum word frequency
    'epochs':       [5],            # number of passes over the data
}

best_model = None
best_score = -1

for values in itertools.product(*parameters.values()):
    params = dict(zip(parameters.keys(), values))
    m = Word2Vec(
        sentences=LineSentence(str(nigeria_combined)),
        **params
    )
    # evaluate: check nearest neighbors of "nation" as proxy for quality
    neighbors = m.wv.most_similar("nation")
    score = len(set(w for w, _ in neighbors))
    if score > best_score:
        best_score = score
        best_model = m
        print(f"New best: {params} → score {score}")

# Save best model
best_model.save(str(output_dir / "nigeria_word2vec_best.model"))
print(f"Best model saved → {output_dir / 'nigeria_word2vec_best.model'}")


# 3) VECTOR SIMILARITY SEARCH
keywords = ["nation", "nationhood", "freedom", "border"]

for word in keywords:
    if word in model.wv:
        print(f"\n'{word}' nearest neighbors:")
        for neighbor, score in model.wv.most_similar(word):
            print(f"  {score:.4f}  {neighbor}")
    else:
        print(f"\n'{word}' not in vocabulary")
