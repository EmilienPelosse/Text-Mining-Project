from pathlib import Path
import random

# Reproducibility
random.seed(42)

# Base directory
base_dir = Path("/text_mining/Text-Mining-Project/Preprocessing/preprocessed_BERT")

# Input files
input_files = [
    "nigeria_spoken.txt",
    "nigeria_written.txt",
    "britain_spoken.txt",
    "britain_written.txt",
    "india_spoken.txt",
    "india_written.txt",
    "usa_spoken.txt",
    "usa_written.txt",
]

# Create train/val directories
train_dir = base_dir / "train"
val_dir = base_dir / "val"

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Validation split percentage
val_ratio = 0.1

for filename in input_files:

    input_path = base_dir / filename

    print(f"Processing {filename}...")

    # Read file
    text = input_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Split into non-empty lines
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Shuffle lines
    random.shuffle(lines)

    # Compute split index
    split_idx = int(len(lines) * (1 - val_ratio))

    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # Output paths
    train_output = train_dir / filename
    val_output = val_dir / filename

    # Write train file
    train_output.write_text(
        "\n".join(train_lines),
        encoding="utf-8"
    )

    # Write validation file
    val_output.write_text(
        "\n".join(val_lines),
        encoding="utf-8"
    )

    print(
        f"Saved {len(train_lines)} train lines "
        f"and {len(val_lines)} validation lines."
    )

print("Preprocessing complete.")