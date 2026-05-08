# Project Update 2 — 08/05/2026
**Cross-Cultural Semantic Analysis of Nationhood within the ICE Corpora**

*Emma Leschly, Romane Gourlet, Juliette Duriez, Emilien Pelosse*

---

## Progress Since Update 1

### Data Access
- Successfully obtained access to all ICE corpora (Nigeria, India, USA and others)
- All national components are available locally in the repository

### Data Preprocessing
- Completed preprocessing pipeline for all corpora
- Steps implemented: ICE annotation removal, lowercasing, punctuation removal, sentence tokenization
- Output: flat `.txt` files with one sentence per line, ready for model training
- Combined spoken and written into a single file for richer training data
- Preprocessing scripts were reusable for all other corpora

### Model Training

#### FastText
- Trained on `nigeria_combined.txt` (~1M words, 10,395 unique words)
- Manual grid search over: architecture (cbow/skipgram), dimensions (100/200), loss (ns/hs)
- **Best model:** cbow, dim=100, negative sampling
- Models saved locally (excluded from git due to file size — ~771MB each)

#### Word2Vec
- Trained on same combined Nigeria corpus using gensim
- Manual grid search over: architecture (cbow/skipgram), dimensions (100/200)
- **Best model:** cbow, dim=100
- Models saved locally

#### Bert

---

## Initial Results — Nigeria Corpus

### FastText Nearest Neighbors
| Query | Top Neighbors |
|-------|--------------|
| nationhood | nation, nationalist, nationwide, nationalistic, nationals |

> FastText captures morphological variants strongly due to subword learning (sharing character n-grams across "nation", "nationalist", "nationwide").

### Word2Vec Nearest Neighbors
| Query | Top Neighbors |
|-------|--------------|
| nation | country, struggle, economy, democratic, democracy, unity |
| nationhood | federalism, patriotism, legislate, statutes, conflicting |
| freedom | protect, opposition, corrupt, masses, citizenry, undermine |
| border | demarcation, retail, sourced, situated, poorest |

> Word2Vec produces more purely semantic neighbors. Key observations:
> - **"nation"** clusters around political and economic discourse: "struggle", "democracy", "anpp" (Nigerian political party) suggest nationhood is tied to political identity
> - **"nationhood"** appears in legal/civic discourse: "federalism", "legislate", "statutes" suggest formal institutional usage
> - **"freedom"** is framed as threatened or contested: "corrupt", "undermine", "opposition", "masses" suggest resistance discourse
> - **"border"** reflects local geographic context: Nigerian place names ("lagelu", "idere") suggest border refers to local rather than national boundaries

---

## Remaining Work

### Still To Do
- Preprocess corpora for GB, India, and USA
- Train FastText and Word2Vec models for all 4 corpora
- Cross-corpus comparison of nearest neighbors
- Procrustes alignment for direct vector space comparison across models
- Begin BERT analysis (Romane, Emma)
- Write methodology, results, and discussion sections

### Known Limitations
- Nigeria corpus is small (~1M words, 10,395 unique vocabulary after filtering), some noise expected in embeddings
- Morphological noise in FastText (shared `-ation` suffix pulling in "donation", "imagination")

---

## Next Steps
- [ ] Train models on all corpora
- [ ] Run comparative nearest neighbor analysis across all countries
- [ ] Begin writing methodology and results sections
- [ ] Coordinate between BERT and Word2Vec/FastText team on results
