[Transformer_QA_Project_Explained.pdf](https://github.com/user-attachments/files/32698188/Transformer_QA_Project_Explained.pdf) Detailed Explanation 

# 🤖 Transformer Question Answering System

A fine-tuned BERT model for extractive question answering. Given a passage and a question, the system identifies the exact answer span inside the passage and returns it as a concise answer.

This project is built with Hugging Face Transformers, trained on the Stanford Question Answering Dataset (SQuAD), and served through a simple interactive Streamlit web application.

## Table of Contents

- [Overview](#overview)
- [Demo Example](#demo-example)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Evaluation Metrics](#evaluation-metrics)
- [Notebooks](#notebooks)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

This project fine-tunes `bert-base-uncased` on the Stanford Question Answering Dataset (SQuAD) to build a model that can read a paragraph and locate the exact words that answer a question.

The workflow consists of:

1. Loading and preparing the SQuAD dataset
2. Tokenizing text and aligning answers to model input positions
3. Fine-tuning BERT with a QA classification head
4. Evaluating using SQuAD metrics
5. Serving the model through a user-friendly Streamlit app

## Demo Example

**Context:** "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing pre-training developed by Google."

**Question:** "Who developed BERT?"

**Answer:** "Google"

## Features

- End-to-end fine-tuning pipeline for BERT on SQuAD
- Evaluation using the official Exact Match (EM) and F1 metrics
- Single-example inference with Python
- Interactive web UI built using Streamlit
- Jupyter notebooks documenting the training and experimentation workflow
- Automatic fallback to the base pre-trained model when a checkpoint is unavailable

## Project Structure

```text
Transformer_project/
├── app/
│   └── app.py                         # Streamlit web application
├── data/
│   └── .gitkeep                      # Datasets are downloaded automatically
├── models/
│   └── bert-squad-finetuned/         # Fine-tuned model checkpoint
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_tokenization.ipynb
│   ├── 03_bert_baseline.ipynb
│   ├── 04_fine_tuning.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preparation.py           # Loads the SQuAD dataset
│   ├── tokenizer.py                  # Tokenization and answer-span alignment
│   ├── train.py                      # Fine-tuning pipeline
│   ├── evaluate.py                   # EM & F1 evaluation
│   └── predict.py                    # Single-example inference
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE (optional, to be added)
```

## Tech Stack

| Category | Tool |
| --- | --- |
| Language | Python 3.9+ |
| Deep Learning | PyTorch |
| Transformers | Hugging Face `transformers` |
| Dataset | Hugging Face `datasets` (SQuAD) |
| Evaluation | Hugging Face `evaluate` |
| Web App | Streamlit |
| Base Model | `bert-base-uncased` |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/thimmareddygt7/Transformer_project.git
cd Transformer_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model

This step fine-tunes `bert-base-uncased` on SQuAD and saves the result to `models/bert-squad-finetuned/`.

```bash
python -m src.train
```

> Training runs for 2 epochs. A GPU is strongly recommended because CPU training will be significantly slower.

### 2. Evaluate the model

This runs the fine-tuned model against validation samples and prints the Exact Match and F1 scores.

```bash
python -m src.evaluate
```

### 3. Run a single prediction

Use this as a quick sanity check on one example from the terminal.

```bash
python -m src.predict
```

### 4. Launch the web app

This opens an interactive browser-based UI where you can paste a passage and ask a question.

```bash
streamlit run app/app.py
```

> If no fine-tuned checkpoint is found, the app and scripts automatically fall back to the base `bert-base-uncased` model, so the project still runs but with lower accuracy.

## How It Works

1. Data Preparation  
   The SQuAD dataset is loaded and structured into context-question-answer triples.

2. Tokenization  
   Text is converted to token IDs, and answer spans are mapped from character positions to token-level start and end indices. Long passages are handled using appropriate tokenization strategies.

3. Fine-Tuning  
   BERT is trained with a QA head that predicts the start and end token positions for the answer within the input.

4. Post-Processing  
   At inference time, the highest-scoring valid answer span is selected and decoded back into readable text.

5. Serving  
   The trained model is loaded once and exposed through a Streamlit interface for interactive question answering.

## Evaluation Metrics

The model is evaluated using the official SQuAD-style metrics:

| Metric | Meaning |
| --- | --- |
| Exact Match (EM) | Percentage of predictions that exactly match the ground-truth answer |
| F1 Score | Token-level overlap score that gives partial credit for close answers |

Run the following command to reproduce the evaluation on the fine-tuned checkpoint:

```bash
python -m src.evaluate
```

## Notebooks

The notebooks directory documents the step-by-step exploration behind the final project code.

| Notebook | Purpose |
| --- | --- |
| `01_dataset_exploration.ipynb` | Initial exploration of the SQuAD dataset structure |
| `02_tokenization.ipynb` | Experiments with tokenization and offset mapping |
| `03_bert_baseline.ipynb` | Testing the base, un-fine-tuned BERT model |
| `04_fine_tuning.ipynb` | Early fine-tuning experiments |
| `05_evaluation.ipynb` | Early evaluation experiments |

## Requirements

See `requirements.txt` for the complete dependency list. Core libraries include:

- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `streamlit`

## Acknowledgements

- SQuAD Dataset — Rajpurkar et al., Stanford University
- Hugging Face Transformers
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — Devlin et al., Google AI

## License

This project is open source. Add your preferred license here, such as MIT, if you want to publish the repository with formal licensing terms.

---

If you want, I can also make it even more polished with:

- a GitHub badge section
- a screenshot/demo section
- a contributor section
- a more professional landing-page style README
- a README tailored for a portfolio or hackathon submission
