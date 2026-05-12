# preprocessing for Word2Vec training on ICE corpora

import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

# Read the file 
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalize line endings
    return text

# Remove ICE annotations
def remove_annotations(text):
    text = re.sub(r"<ICE-[^>]+>", " ", text)   # e.g. <ICE-NG:S1A-001#1:1:A>
    text = re.sub(r"<\$[^>]*>", " ", text)      # speaker tags e.g. <$A>
    text = re.sub(r"<[^>]+>", " ", text)       # any remaining tags
    text = re.sub(r"\r\n|\r", "\n", text) # normalizing endings
    return text

# Lowercase
def lowercase(text):
    return text.lower()

# Remove punctuation and numbers
def remove_punctuation(text):
    text = re.sub(r"(?<!\w)-(?!\w)|[^\w\s-]", " ", text)  # keep inner hyphens!! for words like nation-state
    text = re.sub(r"\d+", " ", text)                       # remove numbers
    text = re.sub(r"\s+", " ", text).strip()             # clean whitespace
    return text

# Tokenize into sentences, each a list of words
def tokenize(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [word_tokenize(line) for line in lines]

# Run all steps on one file
def preprocess_file(file_path):
    text = read_file(file_path)
    text = remove_annotations(text)
    text = lowercase(text)
    text = remove_punctuation(text)
    sentences = tokenize(text)
    return sentences



# Run on a whole folder, save result as flat .txt
def preprocess_folder(folder_path, output_path):
    all_sentences = []
    file_count = 0
    for root, _, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.lower().endswith(".txt"):
                file_count += 1
                file_path = os.path.join(root, filename)
                sentences = preprocess_file(file_path)
                all_sentences.extend(sentences)
    print(f"Files processed : {file_count}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            if sent:
                f.write(" ".join(sent) + "\n")

    print(f"Done: {len(all_sentences)} sentences → {output_path}")

if __name__ == "__main__":
    datasets = [
        ("ICE-Canada/ICE-CAN/Corpus", "canada"),
        ("ICE-Philippines/ICE Philippines/Corpus", "philippines"),
        ("ICE-EastAfrica/ICE East Africa/ICE-EA/corpus", "east_africa"),
        ("ICE-INDIA/ICE India/Corpus", "india"),
        ("ICE-IRELAND/ICE-IRL/ICE-Ireland-version1/ICE-Ireland txt/All ICE files txt", "ireland"),
        ("ICE-JA/ICE-JA/CORPUS", "jamaica"),
        ("ICE-Singapore/ICE SINGAPORE/Corpus", "singapore"),
        ("ICE-USA", "usa"),
        ("ice-nig/txt - without speaker tags", "nigeria")
    ]

    for folder_name, output_name in datasets:
        folder_path = os.path.join(BASE_DIR, "Data", folder_name)
        if os.path.exists(folder_path):
            preprocess_folder(
                folder_path=folder_path,
                output_path=os.path.join(BASE_DIR, "Preprocessing", "preprocessed_word2vec", f"{output_name}.txt")
            )
        else:
            print(f"Skipping {folder_path} (not found)")
