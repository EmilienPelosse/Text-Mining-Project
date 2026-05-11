from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import torch
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

model_id = "answerdotai/ModernBERT-base"
output_dir = "ModernBERT-base-finetuned-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

def get_word_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Find the position of the target word
    target_indices = [i for i, t in enumerate(tokens) if target_word in t.lower()]
    if not target_indices:
        return None, None
    
    # Since we aren't training the model, we don't have to compute gradients
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Average the embeddings if the word is split into multiple tokens
    embeddings = outputs.last_hidden_state[0]
    target_embedding = embeddings[target_indices].mean(dim=0).numpy()
    return target_emb, (tokens, embeddings.numpy())

def find_neighbours(file_path, target_word="nation", top_n=10):
    lines = Path(file_path).read_text(encoding="utf-8").strip().split("\n")

    target_embeddings = []
    all_tokens = []
    all_embeddings = []

    for line in lines:
        # Skipping any sentence that doesn't contain the target word
        if target_word not in line.lower():
            continue
            
        # Gets target word embedding + all context embeddings for the sentece
        target_emb, context = get_nation_embedding(line, target_word)
        if target_emb is None:
            continue

        target_embeddings.append(target_emb)
        tokens, embeddings = context
        all_tokens.extend(tokens)
        all_embeddings.extend(embeddings)

    if not target_embeddings:
      print(f"'{target_word}' not found in {file_path}")
      return

    # Average embedding across all occurrences
    mean_target = np.mean(target_embeddings, axis=0, keepdims=True)
    all_embeddings = np.array(all_embeddings)
    similarities = cosine_similarity(mean_target, all_embeddings)[0]

    # Get top N neighbours
    top_indices = similarities.argsort()[::-1]
    seen = set()
    results = []
    for idx in top_indices:
        token = all_tokens[idx]
        
        # Skip subword fragments
        if not token.startswith("ġ") and not token.isalpha():
            continue
        
        # Clean the ġ prefix
        clean_token = token.replace("ġ", "").lower()
        
        # Skip target word, special tokens, very short words
        if target_word in clean_token or len(clean_token) < 3:
            continue
        
        if clean_token not in seen:
            seen.add(clean_token)
            results.append((clean_token, similarities[idx]))
        
        if len(results) >= top_n:
            break

    return results

# I am running this code on google collab to make it faster, which is why I need to have a file picker :
uploaded = files.upload()

varieties = {
    "canada": "canada.txt",
    "east_africa": "east_africa.txt",
    "india": "india.txt",
    "jamaica": "jamaica.txt",
    "nigeria": "nigeria.txt",
    "philippines": "philippines.txt",
    "singapore": "singapore.txt",
    "usa": "usa.txt",
}

# I am only testing for the words "nation" and "freedom" for the moment
for variety, path in varieties.items():
    print(f"\n=== {variety.upper()} ===")
    neighbours = find_neighbours(path, target_word="nation")
    if neighbours:
        for word, score in neighbours:
            print(f"  {word[1:]}: {score:.4f}")


for variety, path in varieties.items():
    print(f"\n=== {variety.upper()} ===")
    neighbours = find_neighbours(path, target_word="freedom")
    if neighbours:
        for word, score in neighbours:
            print(f"  {word[1:]}: {score:.4f}")
