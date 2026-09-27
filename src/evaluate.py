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