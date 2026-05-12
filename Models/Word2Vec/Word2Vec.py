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


corpora = {
        'nigeria_combined': nigeria_combined,
        'india': india,
        'jamaica': jamaica,
        'usa': usa
}

models = {}

for corpus_name, path in corpora.items():
    # train and save model
    models[corpus_name] = train_and_save(path, corpus_name, output_dir)


# Load models
model_nigeria = Word2Vec.load(str(output_dir / "nigeria_combined_word2vec.model"))
model_jamaica = Word2Vec.load(str(output_dir / "jamaica_word2vec.model"))
model_usa = Word2Vec.load(str(output_dir / "usa_word2vec.model"))
model_india = Word2Vec.load(str(output_dir / "india_word2vec.model"))



# 2) MANUAL GRID SEARCH
parameters = {
    'sg':           [0, 1],         # 0 = cbow, 1 = skipgram
    'vector_size':  [100, 200],     # size of word vectors
    'window':       [5],            # context window size
    'min_count':    [5],            # minimum word frequency
    'epochs':       [5],            # number of passes over the data
}


# Train one model per parameter combination and keep the best one
for corpus_name, path in corpora.items():

    best_model = None
    best_score = -1

    for values in itertools.product(*parameters.values()):
        params = dict(zip(parameters.keys(), values))
        m = Word2Vec(
        sentences=LineSentence(str(path)),
        **params
        )

        # evaluate: check nearest neighbors of "nation" as proxy for quality
        neighbors = m.wv.most_similar("nation")
        score = len(set(w for w, _ in neighbors))
        if score > best_score:
            best_score = score
            best_model = m
            print(f"New best {corpus_name}: {params} → score {score}")

    # Save best model from grid search
    best_model.save(str(output_dir / f"{corpus_name}_word2vec_best.model"))
    print(f"Best model saved → {output_dir / f'{corpus_name}_word2vec_best.model'}")


# 3) VECTOR SIMILARITY SEARCH
keywords = ['nation', 'border', 'freedom', 'community', 'government', 'citizen', 'territory', 'politic', 'culture', 'society', 'civilization', 'religion']

for corpus_name, model in models.items():
    print(f"{corpus_name} corpus: vector similarity search")
    for word in keywords:
        if word in model.wv:
            print(f"\n'{word}' nearest neighbors:")
            
            for neighbor, score in model.wv.most_similar(word):
                print(f"  {score:.4f}  {neighbor}")
        else:
            print(f"\n'{word}' not in vocabulary")


