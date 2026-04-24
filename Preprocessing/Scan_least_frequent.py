# This code scans the ICE-Nigeria corpus and counts occurrences
# of selected nation-related keywords in each text file, allowing us to rank
# and filter documents based on their relevance to the research topic.

import os
import re
from collections import Counter
from collections import defaultdict

# path to Data folder (choose between spoken/written)
folder_path1 = "../Data/ice-nig/txt - without speaker tags/spoken"
folder_path2 = "../Data/ice-nig/txt - without speaker tags/written"

# keywords (expandable) stored in txt file
with open("keywords.txt", 'r', encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]


def function_word_counts(file) :
    # store results
    file_counts = {}
    total_counts = Counter()

    for root, dirs, files in os.walk(file):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().lower()

                    counts = {kw: len(re.findall(rf"\b{kw}s?\b", text)) for kw in keywords}

                    total = sum(counts.values())
                    file_counts[file_path] = {"total": total, **counts}

                    total_counts.update(counts)

    # sort files by relevance
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1]["total"], reverse=False)

    print("\n--- Least relevant files ---")
    for fname, counts in sorted_files[:40]:
        print(f"  {counts['total']:>4} total  {fname}")
    return None

function_word_counts(folder_path1)
function_word_counts(folder_path2)
