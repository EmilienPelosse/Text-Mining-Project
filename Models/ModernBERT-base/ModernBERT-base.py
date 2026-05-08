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

# Add more directories when preprocessing is done
# We need to split the directories into training and validation:

# These folders do not exist yet, but should be added once we get the corpora and finish preprocessing
train_files = [
    Path("Preprocessing/preprocessed_BERT/train/nigeria.txt"),
    Path("Preprocessing/preprocessed_BERT/train/india.txt"),
]

valid_files = [
    Path("Preprocessing/preprocessed_BERT/val/nigeria.txt"),
    Path("Preprocessing/preprocessed_BERT/val/india.txt"),
]

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

# Create dataset 
train_dataset = Dataset.from_list(load_texts(train_files))
valid_dataset = Dataset.from_list(load_texts(valid_files))

dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset,
})

# Load tokenizer and tokenize
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(batch):

    return tokenizer(batch["text"], 
                     truncation=True, 
                     max_length=1024,
                     return_special_tokens_mask=True)

tokenized_dataset = dataset.map(tokenize_function, 
                                batched=True, 
                                remove_columns=["text"])

# Set model
model = AutoModelForMaskedLM.from_pretrained(model_id)

# Use data_collator to mask words during training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15,
)

# Set arguments for training
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

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
