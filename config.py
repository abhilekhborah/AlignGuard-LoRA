import random
import numpy as np
import torch
from typing import List


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AlignGuardConfig:
    """
    Configuration for AlignGuard model.

    Args:
        lambda_sc: Weight for Silhouette Coefficient regularization
        lambda_null: Weight for null-space projection constraint
        poison_threshold: Threshold for poison detection (Mahalanobis distance)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter (scaling factor)
        lora_dropout: LoRA dropout rate
        target_modules: Target modules for LoRA adaptation
        enable_poison_detection: Whether to use poison detection
    """

    def __init__(
        self,
        lambda_sc: float = 0.1,
        lambda_null: float = 0.1,
        poison_threshold: float = 0.95,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: List[str] = ["q_proj", "v_proj"],
        enable_poison_detection: bool = False,
    ):
        self.lambda_sc = lambda_sc
        self.lambda_null = lambda_null
        self.poison_threshold = poison_threshold
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.enable_poison_detection = enable_poison_detection
