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
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Extractive Question Answering System")
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