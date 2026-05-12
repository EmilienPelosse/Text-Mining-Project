from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
base_model_id  = "answerdotai/ModernBERT-base"
finetuned_path = BASE_DIR / "models/moder   nbert_nigeria"

print("Loading base model...")
tokenizer_base = AutoTokenizer.from_pretrained(base_model_id)
model_base     = AutoModel.from_pretrained(base_model_id)
model_base.eval()

print("Loading finetuned model...")
tokenizer_ft = AutoTokenizer.from_pretrained(str(finetuned_path), local_files_only=True)
model_ft     = AutoModel.from_pretrained(str(finetuned_path), local_files_only=True)
model_ft.eval()

# ── Same logic as your teammate's code ────────────────────────────────────────
def get_word_embedding(sentence, target_word, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    target_indices = [i for i, t in enumerate(tokens) if target_word in t.lower()]
    if not target_indices:
        return None, None

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[0]
    target_embedding = embeddings[target_indices].mean(dim=0).numpy()
    return target_embedding, (tokens, embeddings.numpy())


def find_neighbours(file_path, target_word, tokenizer, model, top_n=10):
    lines = Path(file_path).read_text(encoding="utf-8").strip().split("\n")

    target_embeddings = []
    all_tokens = []
    all_embeddings = []

    for line in lines:
        if target_word not in line.lower():
            continue

        target_emb, context = get_word_embedding(line, target_word, tokenizer, model)
        if target_emb is None:
            continue

        target_embeddings.append(target_emb)
        tokens, embeddings = context
        all_tokens.extend(tokens)
        all_embeddings.extend(embeddings)

    if not target_embeddings:
        print(f"'{target_word}' not found in file")
        return []

    mean_target  = np.mean(target_embeddings, axis=0, keepdims=True)
    all_embeddings = np.array(all_embeddings)
    similarities = cosine_similarity(mean_target, all_embeddings)[0]

    top_indices = similarities.argsort()[::-1]
    seen = set()
    results = []
    for idx in top_indices:
        token = all_tokens[idx]
        if not token.startswith("Ġ") and not token.isalpha():
            continue
        clean_token = token.replace("Ġ", "").lower()
        if target_word in clean_token or len(clean_token) < 3:
            continue
        if clean_token not in seen:
            seen.add(clean_token)
            results.append((clean_token, similarities[idx]))
        if len(results) >= top_n:
            break

    return results


# ── Run comparison ─────────────────────────────────────────────────────────────
nigeria_file = BASE_DIR / "Preprocessing/preprocessed_word2vec/nigeria_combined.txt"
keywords     = ["nation", "nationhood", "freedom", "border"]

for word in keywords:
    print(f"\n{'='*50}")
    print(f"Keyword: '{word}'")
    print(f"{'='*50}")

    neighbours_base = find_neighbours(nigeria_file, word, tokenizer_base, model_base)
    neighbours_ft   = find_neighbours(nigeria_file, word, tokenizer_ft,   model_ft)

    # side by side comparison
    print(f"{'Base Model':<30} {'Finetuned Model':<30}")
    print(f"{'-'*30} {'-'*30}")
    for i in range(max(len(neighbours_base), len(neighbours_ft))):
        base_str = f"{neighbours_base[i][0]} ({neighbours_base[i][1]:.4f})" if i < len(neighbours_base) else ""
        ft_str   = f"{neighbours_ft[i][0]} ({neighbours_ft[i][1]:.4f})"     if i < len(neighbours_ft)   else ""
        print(f"{base_str:<30} {ft_str:<30}")