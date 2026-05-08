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

#### Word2Vec versus fastText models

As we were planning on replacing/adding in addition to our word2Vec model with a fastText model, we read documentation to gain a better insight over their different affordances. Although both are very similar, Word2Vec is trained on word level embeddings while fastText is trained on characters n-grams (e.g. 'capable' => 'cap', 'apa', 'pab', 'abl', 'ble', for n=3) in addition to the word embeddings. This technique enables recognizing words from their roots e.g. capable and capability, it handles unknown words (OOV = Out Of Vocabulary) better than Word2Vec. We trained both a word2vec and a fastText model. We reused the preprocessing code we already had for word2vec. Note that a manual GridSearch was performed to tune both models' hyperparameters.

To allow a comparison between the four corpora, we decided to use fastText pretrained embeddings. We also wondered if we should train our fastText model. On the one hand if we were to train our model, the coordinate of vectors for a given word would be different for each corpus. Given the size of our dataset, it is safer to use fastText in order to enlarge the coverage between our dataset and the pre-trained vectors. But we eventually decided to train fastText embeddings for every corpora to preserve their idiosyncrasies. It would preserve the words and expressions specific to a region. Although this decision will make the comparison between vector spaces more difficult (as a given word could have 4 different coordinates), it aligns better with our initial goals for this project. 
For the training, we tuned the hyperparameters of the fasttext function train_unsupervised.

The word2vec model with 100 dimensions and the CBOW architecture produced semantic clusters that were coherent. We used the cosine similarity as a metric to compute the nearest neighbours in the vector space. For instance, the cluster around the  word "nation"  revolves around political and economic discourse, and features nearest neighbours like — "struggle", "democracy", "anpp" (a Nigerian political party). These embeddings suggest that the concept of nationhood is strongly tied to political identity in the Nigerian corpus.
We repeated the process for "nationhood", which seems to appear in a legal and civic discourse context. In particular, some of its closest embeddings are "federalism", "legislate", "statutes", "patriotism" suggesting that the word is used in formal and institutional discussions about the structure and identity of the Nigerian state. By contrast, the embedding for “freedom” among its closest neighbours by seemingly opposite associations which evoke political tensions such as “opposition”, “undermine”, and “corrupt”. It may suggest freedom is framed as something threatened or fought for rather than an established right. More generally, "freedom" is embedded in a discourse of resistance and governance. We also did a first analysis of the embedding for  “border”, which we had considered in our research questions for the update 0. The nearest neighbours of “border” contain local Nigerian place names, notably  "ona", "lagelu", and "idere", and  "retail", "sourced" suggest local trade contexts. This likely reflects that the word "border" in the Nigerian corpus refers more to local geographic and economic boundaries than to national borders in a political sense. More general associations were found, e.g. “demarcation”.

The fastText model was also selected with a manual gridSearch over multiple hyperparameters combination. The most performant model was using 100 dimensions and the CBOW architecture. We also used cosine similarity as the metric to determine the nearest neighbours of an embedding. Among the nearest neighbours of “nationhood”, we found “nation” (0.91), “nationalist” (0.88 ), “nationwide” (0.86), “nationalistic” (0.84), “nationals” (0.83), “imagination” (0.82), “donation” (0.81), “nations” (0.79), “globalisation” (0.78), and “aspiration” (0.78). While some of the previous embeddings do have similar meanings to nationhood, others seem to be part of the noise. As we have seen, the fastText model is trained on characters n-grams. Therefore any word having the similar subwords as nation will be encoded as semantically close, even though it does not imply a direct etymological connection.  For instance, the embeddings "donation", "imagination", "aspiration" are noise caused by the shared `-ation` suffix, a known limitation of subword models on small corpora. By contrast, "globalisation" is analytically interesting as it suggests Nigerian English links nationhood to global context.

To summarize, Word2Vec produces more purely semantic neighbors while FastText captures morphological variants due to its subword model (and introduces noise in the results, due to the common suffixes issue). Both models are complementary for our cross-cultural analysis.

##### FastText
- Trained on `nigeria_combined.txt` (~1M words, 10,395 unique words)
- Manual grid search over: architecture (cbow/skipgram), dimensions (100/200), loss (ns/hs)
- **Best model:** cbow, dim=100, negative sampling
- Models saved locally (excluded from git due to file size — ~771MB each)

##### Word2Vec
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

## Discussion of methods and sources

As we had advanced on the training of our models, we discussed what method we should use to perform our cross-cultural analysis. One challenge we anticipated is the difficulty to compare different vector spaces for our different models and corpora. We will try to further investigate the possibility to align our different vector spaces to allow cross-model/corpus comparisons. We also began our first analysis using the cosine similarity to determine the closest neighbours of some embeddings. We think we will keep this metric, and maybe use it as a baseline

As our project focuses on analyzing different embeddings of “nationhood”, we also need to pin down the different dimensions it encompasses. We found the following excerpt which offers a perspective particularly relevant for such analysis. It distinguishes between different aspects constitutive of nationhood, which may be represented differently in the different vector spaces. It can be analyzed through its cultural, economic, political/legal, historical, territorial dimensions and the excerpt also describes it as a form of collective consciousness.
We could determine directions in the vector spaces that encode these different dimensions, by creating (using Wordnet?) semantic clusters of words, and assume that  the “average” direction should encode most of the dimension we are examining. This method would allow us to compare the relative distances of nationhood to  its dimensions across vector spaces.

> Nation is a more diffuse term, if only because, unlike state, it is not a clearly demarcated, officially recognized, and objective unit. Anthony Smith defines nation as "a named human population **sharing a historic territory**, **common myths and historical memories**, a mass, **public culture**, a **common economy and common legal rights and duties** for all members."[19] [...] They clearly are not, or do not have, a state. Are they a nation? The difficulty of this question should be obvious. In the U.S. they do not occupy a distinct territory, and it is debatable whether they share a **common economy and political culture** (especially one discrete from "white" or general U.S. society); they do share, in the widest sense, an "historic" territory in the form of Africa but originate from a plethora of societies (and nations?) within that geographic field.  
>
> The best answer to our question is that they are a nation if they can acquire the characteristics—and even more, the consciousness—of nationhood. [...] In other words, **nationhood is an achieved status**, at least to a degree.”  
>
> — Eller, Jack David. 1997. “Ethnicity, Culture, and ‘The Past’.”  
> — Smith, Anthony D. 1991. National Identity.

Also, the passage below (from our reading assignment) could be used to support the methodology for our analysis, where we assume that studying the different vector embeddings of some fixed words and their relative distance  can help us understand their particular use in different contexts:

> “Word embeddings, trained only on word co-occurrence in text corpora, serve as a dictionary of sorts for computer programs that would like to use word meaning. First, **words with similar semantic meanings tend to have vectors that are close together**. Second, the **vector differences between words in embeddings have been shown to represent relationships between words** [32, 26]”
> — Bulakbasi et al. Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings



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
