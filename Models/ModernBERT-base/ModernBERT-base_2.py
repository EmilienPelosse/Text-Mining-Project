from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import torch
import random
import numpy as np

model_id = "answerdotai/ModernBERT-base"
output_dir = "ModernBERT-base-finetuned-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

def get_word_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Find the position of the target word
    target_indices = [i for i, t in enumerate(tokens) if target_word in t.lower()]
    if not target_indices:
        return None
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Average the embeddings if the word is split into multiple tokens
    embeddings = outputs.last_hidden_state[0]
    target_embedding = embeddings[target_indices].mean(dim=0)
    return target_embedding.numpy()
