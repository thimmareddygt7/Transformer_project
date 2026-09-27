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