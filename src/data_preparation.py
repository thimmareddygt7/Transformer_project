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
