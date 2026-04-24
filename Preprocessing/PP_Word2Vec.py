# preprocessing for Word2Vec training on ICE corpora

import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

# Read the file 
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

# Remove ICE annotations
def remove_annotations(text):
    text = re.sub(r"<ICE-[^>]+>", " ", text)   # e.g. <ICE-NG:S1A-001#1:1:A>
    text = re.sub(r"<\$[^>]*>", " ", text)      # speaker tags e.g. <$A>
    text = re.sub(r"<[^>]+>", " ", text)       # any remaining tags
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
    sentences = sent_tokenize(text)
    return [word_tokenize(sent) for sent in sentences]

# Run all steps on one file
def preprocess_file(file_path):
    text = read_file(file_path)
    text = remove_annotations(text)
    text = lowercase(text)
    text = remove_punctuation(text)
    sentences = tokenize(text)
    return sentences

folder = "../Data/ice-nig/txt - without speaker tags/spoken"
print("Exists:", os.path.exists(folder))
print("Files found:")
for root, _, files in os.walk(folder):
    for f in files:
        print(os.path.join(root, f))

# Run on a whole folder, save result as flat .txt
def preprocess_folder(folder_path, output_path):
    all_sentences = []

    for root, _, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                sentences = preprocess_file(file_path)
                all_sentences.extend(sentences)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            if sent:
                f.write(" ".join(sent) + "\n")

    print(f"Done: {len(all_sentences)} sentences → {output_path}")

if __name__ == "__main__":
    base = os.path.join(BASE_DIR, "Data", "ice-nig", "txt - without speaker tags")

    preprocess_folder(
        folder_path=os.path.join(base, "spoken"),
        output_path=os.path.join(BASE_DIR, "Preprocessing", "preprocessed_word2vec", "nigeria_spoken.txt")
    )
    preprocess_folder(
        folder_path=os.path.join(base, "written"),
        output_path=os.path.join(BASE_DIR, "Preprocessing", "preprocessed_word2vec", "nigeria_written.txt")
    )
