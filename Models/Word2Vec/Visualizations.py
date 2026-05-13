from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec

BASE_DIR = Path(__file__).resolve().parent
matplotlib.use('Agg')  # Use non-interactive backend

def plot_tsne(model, keywords, corpus_name):
    words = []
    vectors = []
    colors = []
    color_map = matplotlib.colormaps["tab10"]

    for i, kw in enumerate(keywords):
        if kw not in model.wv:
            continue
        neighbors = [w for w, _ in model.wv.most_similar(kw, topn=10)]
        for word in [kw] + neighbors:
            if word in model.wv:
                words.append(word)
                vectors.append(model.wv[word])
                colors.append(color_map(i))

    if len(words) < 4:
        print(f"Not enough words for t-SNE in {corpus_name}")
        return

    # perplexity must be less than number of points
    perplexity = min(5, len(words) - 1)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(np.array(vectors))

    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        plt.scatter(coords[i, 0], coords[i, 1], color=colors[i], alpha=0.7)
        plt.annotate(word, coords[i], fontsize=8)
    plt.title(f"t-SNE — {corpus_name}")
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"tsne_{corpus_name}.png", dpi=150, bbox_inches="tight")
    plt.close()  # Close to free memory

# Load your already-trained models
model_nigeria = Word2Vec.load(str(BASE_DIR / "nigeria_combined_word2vec_best.model"))
model_jamaica = Word2Vec.load(str(BASE_DIR / "jamaica_word2vec_best.model"))
model_usa     = Word2Vec.load(str(BASE_DIR / "usa_word2vec_best.model"))
model_india   = Word2Vec.load(str(BASE_DIR / "india_word2vec_best.model"))

keywords = ["nation", "freedom", "border", "culture", "religion", "society"]

plot_tsne(model_nigeria, keywords, "Nigeria")
plot_tsne(model_india,   keywords, "India")
plot_tsne(model_jamaica, keywords, "Jamaica")
plot_tsne(model_usa,     keywords, "USA")