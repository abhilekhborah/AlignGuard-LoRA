import os
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SafetyDataset(Dataset):
    """
    Dataset class that includes safety labels for SafeLoRA.

    Args:
        texts: List of input texts
        labels: List of output labels (for language modeling, usually same as texts)
        safety_labels: List of safety labels (0 for safe, 1 for unsafe)
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
    """

    def __init__(self, texts, labels, safety_labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.safety_labels = safety_labels  # 0 for safe, 1 for unsafe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        safety_label = self.safety_labels[idx]

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # For causal language modeling, shift labels
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore pad tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "safety_labels": safety_label,
        }


def load_data_from_csvs(safe_csv_path, unsafe_csv_path):
    """
    Load instruction-response pairs from separate safe and unsafe CSV files.

    Args:
        safe_csv_path: Path to the CSV file with safe instruction-response pairs
        unsafe_csv_path: Path to the CSV file with unsafe instruction-response pairs

    Returns:
        texts: List of texts for training (instruction + response)
        labels: List of target labels
        safety_labels: List of safety labels (0 for safe, 1 for unsafe)
    """
    import pandas as pd

    logger.info(f"Loading safe data from {safe_csv_path}")
    logger.info(f"Loading unsafe data from {unsafe_csv_path}")

    texts = []
    labels = []
    safety_labels = []

    try:
        # Read safe CSV file
        safe_df = pd.read_csv(safe_csv_path)

        # Check if required columns exist
        if 'instruction' not in safe_df.columns or 'response' not in safe_df.columns:
            raise ValueError("CSV must contain 'instruction' and 'response' columns")

        # Format the safe training data
        for _, row in safe_df.iterrows():
            instruction = str(row['instruction']).strip()
            response = str(row['response']).strip()

            # Format as a single text for training
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            texts.append(formatted_text)
            safety_labels.append(0)  # Safe label

        # Read unsafe CSV file
        unsafe_df = pd.read_csv(unsafe_csv_path)

        # Check if required columns exist
        if 'instruction' not in unsafe_df.columns or 'response' not in unsafe_df.columns:
            raise ValueError("CSV must contain 'instruction' and 'response' columns")

        # Format the unsafe training data
        for _, row in unsafe_df.iterrows():
            instruction = str(row['instruction']).strip()
            response = str(row['response']).strip()

            # Format as a single text for training
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            texts.append(formatted_text)
            safety_labels.append(1)  # Unsafe label

        # For causal LM, labels are the same as inputs
        labels = texts.copy()

        logger.info(f"Loaded {len(safe_df)} safe examples and {len(unsafe_df)} unsafe examples")
        logger.info(f"Total dataset size: {len(texts)} examples")

        return texts, labels, safety_labels

    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise
