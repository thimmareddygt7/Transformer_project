# =============================================================================
#  ALL FILES CODE - Transformer Question Answering Project
# =============================================================================


# =============================================================================
# FILE: src\__init__.py
# =============================================================================

# Package initialization


# =============================================================================
# FILE: src\data_preparation.py
# =============================================================================

"""
Data Preparation Module for Question Answering tasks.
"""

from datasets import load_dataset


def load_qa_dataset(dataset_name: str = "rajpurkar/squad"):
    """
    Load SQuAD or compatible QA dataset from Hugging Face Datasets Hub.
    """
    print(f"Loading dataset: '{dataset_name}'...")
    return load_dataset(dataset_name)


if __name__ == "__main__":
    dataset = load_qa_dataset()
    print("Dataset Summary:")
    print(dataset)


# =============================================================================
# FILE: src\tokenizer.py
# =============================================================================

from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128


def get_tokenizer(model_name: str = MODEL_NAME):
    """Load and return the pre-trained tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def prepare_train_features(examples, tokenizer, max_length=MAX_LENGTH, doc_stride=DOC_STRIDE):
    """
    Preprocess SQuAD training data by aligning character-level answer spans
    with tokenized sequence offsets and handling sliding window overflows.
    """
    questions = [q.strip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # Fallback to CLS token if there are no answers
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Locate context tokens boundaries
        token_start_idx = 0
        while sequence_ids[token_start_idx] != 1:
            token_start_idx += 1

        token_end_idx = len(input_ids) - 1
        while sequence_ids[token_end_idx] != 1:
            token_end_idx -= 1

        # Check if the answer is completely inside the current window span
        if not (offsets[token_start_idx][0] <= start_char and offsets[token_end_idx][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_idx <= token_end_idx and offsets[token_start_idx][0] <= start_char:
                token_start_idx += 1
            start_positions.append(token_start_idx - 1)

            while token_end_idx >= token_start_idx and offsets[token_end_idx][1] >= end_char:
                token_end_idx -= 1
            end_positions.append(token_end_idx + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_length=MAX_LENGTH, doc_stride=DOC_STRIDE):
    """
    Preprocess SQuAD evaluation data while preserving offset mappings
    and example IDs for logit post-processing.
    """
    questions = [q.strip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(tokenized_examples["input_ids"])):
        sample_index = sample_mapping[i]
        example_ids.append(examples["id"][sample_index])

        sequence_ids = tokenized_examples.sequence_ids(i)
        offset = tokenized_examples["offset_mapping"][i]

        # Mask tokens that are not part of the context
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None) for k, o in enumerate(offset)
        ]

    tokenized_examples["example_id"] = example_ids
    return tokenized_examples


# =============================================================================
# FILE: src\train.py
# =============================================================================

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


# =============================================================================
# FILE: src\evaluate.py
# =============================================================================

import os
import sys

# Prevent self-import module shadowing when file is named evaluate.py
_current_dir = os.path.dirname(os.path.abspath(__file__))
_paths_to_restore = []
for _p in [_current_dir, ""]:
    while _p in sys.path:
        sys.path.remove(_p)
        _paths_to_restore.append(_p)

import evaluate

for _p in reversed(_paths_to_restore):
    sys.path.insert(0, _p)

import collections
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, Trainer
from src.tokenizer import get_tokenizer, prepare_validation_features

MODEL_PATH = "./models/bert-squad-finetuned"
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 30


def _model_is_ready(path: str) -> bool:
    """Return True only if the directory contains a saved model config."""
    return os.path.isfile(os.path.join(path, "config.json"))


def postprocess_qa_predictions(
    examples, features, predictions, n_best_size=N_BEST_SIZE, max_answer_length=MAX_ANSWER_LENGTH
):
    all_start_logits, all_end_logits = predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions_dict = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        valid_answers = []

        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            predictions_dict[example["id"]] = best_answer["text"]
        else:
            predictions_dict[example["id"]] = ""

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions_dict.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    return formatted_predictions, references


def evaluate_model():
    # Use fine-tuned model if available, otherwise default to pretrained BERT
    model_path = MODEL_PATH if _model_is_ready(MODEL_PATH) else "bert-base-uncased"
    if not _model_is_ready(MODEL_PATH):
        print(f"Notice: Fine-tuned model at '{MODEL_PATH}' not found. Evaluating baseline '{model_path}'...")
    else:
        print(f"Evaluating fine-tuned model from '{model_path}'...")

    tokenizer = get_tokenizer("bert-base-uncased")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    # Load dataset validation split
    # Evaluate on a 500-sample slice for quick local CPU testing (~1-2 minutes)
    dataset = load_dataset("rajpurkar/squad")["validation"].select(range(500))

    print("Preprocessing validation data...")
    eval_features = dataset.map(
        prepare_validation_features,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
    )

    trainer = Trainer(model=model, processing_class=tokenizer)
    print("Running predictions...")
    raw_predictions = trainer.predict(eval_features)

    print("Post-processing logits to string answers...")
    formatted_preds, references = postprocess_qa_predictions(
        dataset, eval_features, raw_predictions.predictions
    )

    metric = evaluate.load("squad")
    results = metric.compute(predictions=formatted_preds, references=references)

    print("\n================ EVALUATION RESULTS ================")
    print(f"Exact Match (EM): {results['exact_match']:.2f}%")
    print(f"F1 Score:         {results['f1']:.2f}%")
    print("====================================================")


if __name__ == "__main__":
    evaluate_model()


# =============================================================================
# FILE: src\predict.py
# =============================================================================

import os
import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

MODEL_PATH = "./models/bert-squad-finetuned"


def _model_is_ready(path: str) -> bool:
    """Return True only if the directory contains a saved model config."""
    return os.path.isfile(os.path.join(path, "config.json"))


def load_inference_model():
    """Load the base uncased tokenizer and fine-tuned BERT model weights."""
    model_path = MODEL_PATH if _model_is_ready(MODEL_PATH) else "bert-base-uncased"
    if not _model_is_ready(MODEL_PATH):
        print(f"Notice: Fine-tuned model at '{MODEL_PATH}' not found. Loading baseline '{model_path}'...")
    else:
        print(f"Loading fine-tuned model from '{model_path}'...")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    return tokenizer, model


def answer_question(context: str, question: str) -> str:
    """Extract an answer span from a context passage given a question."""
    tokenizer, model = load_inference_model()

    # Tokenize input context and question
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=384,
        truncation=True,
    )

    # Run forward pass through fine-tuned BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Identify highest-probability start and end token indices
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()

    # Extract predicted token span and convert back to string answer
    if end_idx >= start_idx:
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx : end_idx + 1])
        answer = tokenizer.convert_tokens_to_string(tokens)
    else:
        answer = "No clear answer found."

    return answer


if __name__ == "__main__":
    sample_context = (
        "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based "
        "machine learning technique for natural language processing pre-training developed by Google."
    )
    sample_question = "Who developed BERT?"

    print("\n--- Testing Single-Example Prediction ---")
    print(f"Context:  {sample_context}")
    print(f"Question: {sample_question}")

    predicted_answer = answer_question(sample_context, sample_question)
    print(f"Answer:   {predicted_answer}\n")


# =============================================================================
# FILE: app\app.py
# =============================================================================

import os
import sys
import streamlit as st
import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

MODEL_PATH = "./models/bert-squad-finetuned"


def _model_is_ready(path: str) -> bool:
    """Return True only if the directory contains a saved model config."""
    return os.path.isfile(os.path.join(path, "config.json"))


@st.cache_resource
def load_qa_model():
    is_finetuned = _model_is_ready(MODEL_PATH)
    model_path = MODEL_PATH if is_finetuned else "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, is_finetuned


def main():
    st.set_page_config(
        page_title="Extractive Question Answering",
        page_icon="ðŸ¤–",
        layout="wide",
    )

    st.title("ðŸ¤– Extractive Question Answering System")
    st.markdown(
        "Enter a passage of text (context) and ask a question. "
        "The fine-tuned BERT model will locate and extract the exact answer span from the text."
    )

    tokenizer, model, is_finetuned = load_qa_model()

    if is_finetuned:
        st.info(" Loaded Fine-Tuned Model Weights (`./models/bert-squad-finetuned`)")
    else:
        st.warning(" Notice: Using Baseline Model (`bert-base-uncased`). Fine-tune the model with `python -m src.train` to improve accuracy.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Passage & Question")
        context = st.text_area(
            "Context Passage:",
            height=250,
            placeholder="Paste your context paragraph here...",
        )
        question = st.text_input(
            "Question:",
            placeholder="Ask a question about the context passage...",
        )
        submit_btn = st.button("Extract Answer", type="primary")

    with col2:
        st.subheader("Model Output")
        if submit_btn:
            if not context.strip() or not question.strip():
                st.warning("Please provide both a context passage and a question.")
            else:
                with st.spinner("Processing text and extracting answer..."):
                    inputs = tokenizer(
                        question,
                        context,
                        add_special_tokens=True,
                        return_tensors="pt",
                        max_length=384,
                        truncation=True,
                    )

                    with torch.no_grad():
                        outputs = model(**inputs)

                    start_idx = torch.argmax(outputs.start_logits).item()
                    end_idx = torch.argmax(outputs.end_logits).item()

                    if end_idx >= start_idx:
                        tokens = tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][0][start_idx : end_idx + 1]
                        )
                        answer = tokenizer.convert_tokens_to_string(tokens)
                        st.success(f"**Predicted Answer:** {answer}")
                    else:
                        st.error("No valid answer span could be identified in the provided context.")


if __name__ == "__main__":
    main()
