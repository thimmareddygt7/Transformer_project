import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from src.tokenizer import get_tokenizer, prepare_train_features

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./models/bert-squad-finetuned"


def main():
    # 1. Load dataset & tokenizer
    print("Loading dataset and tokenizer...")
    dataset = load_dataset("rajpurkar/squad")
    tokenizer = get_tokenizer(MODEL_NAME)

    # 2. Tokenize dataset
    print("Preprocessing full dataset...")
    tokenized_datasets = dataset.map(
        prepare_train_features,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )

    # 3. Load Model
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # 4. Training Configurations
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    # 5. Train & Save
    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")


if __name__ == "__main__":
    main()