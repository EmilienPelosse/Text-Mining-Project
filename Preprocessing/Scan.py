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
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1]["total"], reverse=True)

    print("\nTotal counts across corpus:\n", total_counts)

    word_counter = defaultdict(lambda: {"total": 0, "files": {}})

    for fname, counts in file_counts.items():
        for word, count in counts.items():
            if word == "total" or count == 0:
                continue
            word_counter[word]["total"] += count
            word_counter[word]["files"][fname] = count

    # sort words by total occurrences
    sorted_words = sorted(word_counter.items(), key=lambda x: x[1]["total"], reverse=True)

    # print breakdown
    for word, data in sorted_words[:40]:
        print(f"\n'{word}' — {data['total']} occurrences across {len(data['files'])} files")
        for fname, count in sorted(data["files"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {count:>4}x  {fname}")
    return None

function_word_counts(folder_path1)
function_word_counts(folder_path2)
