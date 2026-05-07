## FastText Results — Nigeria Corpus

### Model Training
The FastText model was trained on the combined Nigerian spoken and written ICE corpus (~1M words, 10,395 unique words after filtering). A manual grid search was run over combinations of `cbow`/`skipgram`, `dim` (100/200), and loss functions (`ns`/`hs`).

### Best Model
The best performing combination was:
- **Architecture:** cbow
- **Dimensions:** 100
- **Loss:** negative sampling (ns)
- **Window size:** 5

### Nearest Neighbors of "nationhood"
| Word | Cosine Similarity |
|------|------------------|
| nation | 0.91 |
| nationalist | 0.88 |
| nationwide | 0.86 |
| nationalistic | 0.84 |
| nationals | 0.83 |
| imagination | 0.82 |
| donation | 0.81 |
| nations | 0.79 |
| globalisation | 0.78 |
| aspiration | 0.78 |

### Observations
- The top neighbors are semantically coherent, clustering around the nationhood semantic field
- "globalisation" is analytically interesting — suggesting Nigerian English links nationhood to global context
- "donation", "imagination", "aspiration" are noise caused by the shared `-ation` suffix, a known limitation of subword models on small corpora