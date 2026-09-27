# Transformer Question Answering

A end-to-end Question Answering project using Transformer models (e.g., BERT, RoBERTa) built with PyTorch, Hugging Face Transformers, and Streamlit.

## Project Structure

```text
transformer-question-answering/
│
├── data/                  # Raw and processed datasets (e.g., SQuAD)
│
├── notebooks/             # Step-by-step Jupyter Notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_tokenization.ipynb
│   ├── 03_bert_baseline.ipynb
│   ├── 04_fine_tuning.ipynb
│   └── 05_evaluation.ipynb
│
├── src/                   # Source code modules
│   ├── data_preparation.py # Dataset loading and preprocessing
│   ├── tokenizer.py        # Tokenizer utilities and feature extraction
│   ├── train.py            # Training and fine-tuning pipeline
│   ├── evaluate.py         # Model evaluation (Exact Match & F1)
│   └── predict.py          # QA Inference pipeline
│
├── models/                # Saved model checkpoints and tokenizer files
│
├── app/                   # Web Application
│   └── app.py              # Streamlit demo user interface
│
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── .gitignore             # Git ignore rules
```

## Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
cd transformer-question-answering
pip install -r requirements.txt
```

### 2. Running Notebooks

Launch Jupyter Notebook to explore data, tokenization, training, and evaluation step-by-step:

```bash
jupyter notebook notebooks/
```

### 3. Training & Fine-Tuning

Fine-tune BERT on the SQuAD dataset (saves to `./models/bert-squad-finetuned`):

```bash
python -m src.train
```

### 4. Evaluation

Evaluate the fine-tuned model on the SQuAD validation set (Exact Match & F1):

```bash
python -m src.evaluate
```

### 5. Running Web App

Launch the interactive Streamlit QA demo **from the `transformer-question-answering/` directory**:

```bash
python -m streamlit run app/app.py
```

## License

MIT License
