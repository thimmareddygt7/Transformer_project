🤖 Transformer Question Answering System
A fine-tuned BERT model that performs extractive question answering — given a passage of text and a question, it locates and extracts the exact answer span directly from the passage.
Built with 🤗 Hugging Face transformers, trained on the SQuAD dataset, and served through an interactive Streamlit web app.
📌 Overview
This project fine-tunes bert-base-uncased on the Stanford Question Answering Dataset (SQuAD) to build a model that can read a paragraph and pinpoint the exact words that answer a given question — no answer generation, just precise extraction.
Example:
Context: "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing pre-training developed by Google."
Question: "Who developed BERT?"
Answer: Google
✨ Features
🔧 End-to-end fine-tuning pipeline for BERT on SQuAD
📊 Evaluation with official Exact Match (EM) and F1 metrics
⚡ Single-example inference via Python
🌐 Interactive web UI built with Streamlit
📓 Exploratory Jupyter notebooks documenting the full development process
🔄 Graceful fallback to the base pretrained model if no fine-tuned checkpoint is found
🗂️ Project Structure
transformer-question-answering/
│
├── app/
│   └── app.py                        # Streamlit web application
│
├── data/
│   └── .gitkeep                      # Datasets are downloaded automatically
│
├── models/
│   └── bert-squad-finetuned/         # Fine-tuned model (created after training)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── notebooks/                        # Exploratory / development notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_tokenization.ipynb
│   ├── 03_bert_baseline.ipynb
│   ├── 04_fine_tuning.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── init.py
│   ├── data_preparation.py           # Loads the SQuAD dataset
│   ├── tokenizer.py                  # Tokenization & answer-span alignment
│   ├── train.py                      # Fine-tuning pipeline
│   ├── evaluate.py                   # EM & F1 evaluation
│   └── predict.py                    # Single-example inference
│
├── requirements.txt
├── README.md
└── .gitignore
🛠️ Tech Stack
Category	Tool
Language	Python 3.9+
Deep Learning	PyTorch
Transformers	Hugging Face transformers
Dataset	Hugging Face datasets (SQuAD)
Evaluation	Hugging Face evaluate
Web App	Streamlit
Base Model	bert-base-uncased
⚙️ Installation
Clone the repository
bash
git clone https://github.com/thimmareddygt7/Transformer_project.git
cd Transformer_project
Create a virtual environment (recommended)
bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
Install dependencies
bash
pip install -r requirements.txt
🚀 Usage
Train the model
Fine-tunes bert-base-uncased on SQuAD and saves the result to models/bert-squad-finetuned/.
bash
python -m src.train
⏱️ Training runs for 2 epochs. A GPU is strongly recommended — CPU training will be significantly slower.
Evaluate the model
Runs the fine-tuned model against a validation sample and prints Exact Match & F1 scores.
bash
python -m src.evaluate
3. Run a single prediction (script)
Quick sanity-check on one example directly from the terminal.
bash
python -m src.predict
4. Launch the web app
Opens an interactive browser UI where you can paste any passage and ask any question.
bash
streamlit run app/app.py
ℹ️ If no fine-tuned model is found, the app and scripts automatically fall back to the base bert-base-uncased model so everything still runs — just with lower accuracy.
📈 Evaluation Metrics
The model is evaluated using the official SQuAD metrics:
Metric	Meaning
Exact Match (EM)	% of predictions that match the ground-truth answer exactly
F1 Score	Token-level overlap score, giving partial credit for close answers
Run python -m src.evaluate to reproduce these scores on your fine-tuned checkpoint.
🔮 How It Works
Data Preparation — SQuAD context/question/answer triples are loaded via datasets.
Tokenization — Text is converted into token IDs; the character-level answer span is mapped to token-level start/end positions (handling long passages via a sliding window).
Fine-Tuning — BERT is trained with a QA head that predicts a start-token and end-token probability distribution over the input.
Post-processing — At inference time, the highest-scoring valid (start, end) span is selected and decoded back into readable text.
Serving — The trained model is loaded once and exposed via a Streamlit interface for interactive use.
📓 Notebooks
The notebooks/ folder documents the step-by-step exploration behind the final src/ code:
Notebook	Purpose
01_dataset_exploration.ipynb	First look at the SQuAD dataset structure
02_tokenization.ipynb	Experiments with tokenization and offset mapping
03_bert_baseline.ipynb	Testing the base, un-fine-tuned BERT model
04_fine_tuning.ipynb	Early fine-tuning experiments
05_evaluation.ipynb	Early evaluation experiments
📋 Requirements
See requirements.txt for the full list. Core dependencies include:
torch
transformers
datasets
evaluate
streamlit
🙏 Acknowledgements
SQuAD Dataset — Rajpurkar et al., Stanford University
Hugging Face Transformers
BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al., Google AI
📄 License
This project is open-source. Add your preferred license here (e.g., MIT).
