## Word2Vec Results — Nigeria Corpus

### Model Training
The Word2Vec model was trained on the combined Nigerian spoken and written ICE corpus (~1M words, 10,395 unique words after filtering). A manual grid search was run over combinations of `cbow`/`skipgram` and `vector_size` (100/200).

### Best Model
The best performing combination was:
- **Architecture:** cbow (sg=0)
- **Dimensions:** 100
- **Window size:** 5
- **Min count:** 5
- **Epochs:** 5

---

### Nearest Neighbors

#### "nation"
| Word | Cosine Similarity |
|------|------------------|
| country | 0.75 |
| struggle | 0.73 |
| economy | 0.72 |
| democratic | 0.70 |
| continent | 0.69 |
| progress | 0.69 |
| democracy | 0.69 |
| anpp | 0.69 |
| unity | 0.68 |
| citizens | 0.68 |

> "nation" clusters around political and economic discourse — "struggle", "democracy", "anpp" (a Nigerian political party) suggest nationhood is strongly tied to political identity in the Nigerian corpus.

#### "nationhood"
| Word | Cosine Similarity |
|------|------------------|
| conflicting | 0.91 |
| legislate | 0.90 |
| federalism | 0.90 |
| sustaining | 0.90 |
| patriotism | 0.90 |
| deepen | 0.90 |
| statutes | 0.90 |
| gaps | 0.90 |
| realising | 0.89 |
| worker | 0.89 |

> "nationhood" appears in a legal and civic discourse context — "federalism", "legislate", "statutes", "patriotism" suggest it is used in formal, institutional discussions about the structure and identity of the Nigerian state.

#### "freedom"
| Word | Cosine Similarity |
|------|------------------|
| protect | 0.80 |
| positively | 0.80 |
| opposition | 0.79 |
| vision | 0.79 |
| undermine | 0.79 |
| corrupt | 0.78 |
| masses | 0.78 |
| citizenry | 0.78 |
| prosperity | 0.78 |
| slogan | 0.78 |

> "freedom" is embedded in a discourse of resistance and governance — "opposition", "corrupt", "undermine", "masses" suggest freedom is framed as something threatened or fought for rather than an established right.

#### "border"
| Word | Cosine Similarity |
|------|------------------|
| demarcation | 0.94 |
| ona | 0.94 |
| lagelu | 0.94 |
| akin | 0.93 |
| retail | 0.93 |
| sensitisation | 0.93 |
| situated | 0.93 |
| poorest | 0.92 |
| idere | 0.92 |
| sourced | 0.92 |

> "border" associations are mixed — "demarcation" is semantically coherent, but "ona", "lagelu", "idere" are Nigerian place names, and "retail", "sourced" suggest local trade contexts. This likely reflects that "border" in the Nigerian corpus refers more to local geographic and economic boundaries than to national borders in a political sense.

---

### Comparison with FastText
| Aspect | Word2Vec | FastText |
|--------|----------|----------|
| "nation" neighbors | Semantic (country, democracy) | Morphological (nationalist, nationwide) |
| Subword learning | No | Yes |
| Rare word handling | Weak | Strong |
| Overall coherence | Higher for frequent words | Higher for rare/morphological variants |

Word2Vec produces more purely semantic neighbors while FastText captures morphological variants due to its subword model. Both are complementary for the cross-cultural analysis.