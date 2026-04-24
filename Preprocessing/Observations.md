# Observations/Preprocessing ideas

### Reflections on the Scan.py results:
- How much occurences of our topic (nationhood) appear   in the corpus?
- What components are more important, which ones should we exclude?
- Find percentage of documents containing words from lexical field of 'nationhood'.

### Preprocessing for Word2Vec

### Essential
- Lowercase all text
- Tokenize into words using `nltk.word_tokenize` or spaCy
- Remove punctuation and special characters (keep hyphens in compounds like "nation-state")
- Remove numbers
- **Stopword removal**
- POS-Tagging

### ICE-Specific Cleaning
- Strip speaker tags (e.g. `<$A>`) from spoken files
- Remove corpus annotation codes (e.g. `<ICE-NG:S1A-001#1:1:A>`)
- Track spoken/written subcorpus origin per document

### For Cross-Corpus Comparability
- Apply sentence segmentation before training (Word2Vec uses sentence boundaries as context windows)
- Train one model per country corpus separately
- Align vector spaces post-training using Procrustes alignment or a shared seed vocabulary

### Probably won't do
- **Lemmatization**: may blur distinctions like "nation" vs "nationals" that are analytically relevant
- **Stemming**: too aggressive, hurts embedding quality
