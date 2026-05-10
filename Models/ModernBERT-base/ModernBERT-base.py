from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

model_id = "answerdotai/ModernBERT-base"

# Fine tune four models using different "Englishes"
corpora = {
    "nigeria": {
        "train": [
            Path("Preprocessing/preprocessed_BERT/train/nigeria_spoken.txt"),
            Path("Preprocessing/preprocessed_BERT/train/nigeria_written.txt"),
        ],
        "validation": [
            Path("Preprocessing/preprocessed_BERT/val/nigeria_spoken.txt"),
            Path("Preprocessing/preprocessed_BERT/val/nigeria_written.txt"),
        ],
    },

    "britain": {
        "train": [
            Path("Preprocessing/preprocessed_BERT/train/britain_spoken.txt"),
            Path("Preprocessing/preprocessed_BERT/train/britain_written.txt"),
        ],
        "validation": [
            Path("Preprocessing/preprocessed_BERT/val/britain_spoken.txt"),
            Path("Preprocessing/preprocessed_BERT/val/britain_written.txt"),
        ],
    },

    "india": {
        "train": [
            Path("Preprocessing/preprocessed_BERT/train/india_spoken.txt"),
            Path("Preprocessing/preprocessed_BERT/train/india_written.txt"),
        ],
        "validation": [
            Path("Preprocessing/preprocessed_BERT/val/india_spoken.txt"),
            Path("Preprocessing/preprocessed_BERT/val/india_written.txt"),
        ],
    },

    "usa": {
        "train": [
            Path("Preprocessing/preprocessed_BERT/train/usa_spoken.txt"),
            Path("Preprocessing/preprocessed_BERT/train/usa_written.txt"),
        ],
        "validation": [
            Path("Preprocessing/preprocessed_BERT/val/usa_spoken.txt"),
            Path("Preprocessing/preprocessed_BERT/val/usa_written.txt"),
        ],
    },
}

output_dir = "ModernBERT-base-finetuned"

# Load text files
def load_texts(file_paths):
    texts = []

    for path in file_paths:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if text:
            texts.append({"text": text})

    return texts

# Load text files and convert them into training examples
# Each non-empty line becomes one text sample
def load_texts(file_paths):

    texts = []

    for path in file_paths:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        # Split file into separate examples
        for line in text.splitlines():

            line = line.strip()

            if line:
                texts.append({"text": line})

    return texts

# Fine-tune one ModernBERT model for a specific English variety
def train_model(variety_name, train_files, valid_files):

    print(f"\nTraining model for {variety_name} English...")

    output_dir = f"models/modernbert_{variety_name}"

    # Create Hugging Face datasets
    train_dataset = Dataset.from_list(load_texts(train_files))
    valid_dataset = Dataset.from_list(load_texts(valid_files))

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset,
    })

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenization function
    def tokenize_function(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            return_special_tokens_mask=True,
        )

    # Tokenize datasets
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Load pretrained ModernBERT model
    model = AutoModelForMaskedLM.from_pretrained(model_id)

    # Dynamically apply masking during training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,

        learning_rate=5e-5,
        num_train_epochs=3,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=50,

        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # Fine-tune model
    trainer.train()

    # Save trained model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Finished training {variety_name} model.")
    print(f"Model saved to: {output_dir}")

# Train one model for each English variety
for variety_name, files in corpora.items():

    train_model(
        variety_name=variety_name,
        train_files=files["train"],
        valid_files=files["validation"],
    )
