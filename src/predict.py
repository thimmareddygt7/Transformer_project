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