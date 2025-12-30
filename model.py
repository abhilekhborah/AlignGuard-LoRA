import os
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import silhouette_score
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


class SafeLoRAModel:
    """
    SafeLoRA: Silhouette-Aware Fine-Tuning with Parameter-Efficient Learning
    """

    def __init__(
        self,
        model_name_or_path: str,
        config,
        tokenizer=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        is_lora_checkpoint=False,
    ):
        self.device = device
        self.config = config

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = tokenizer

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Load base model
        logger.info(f"Loading base model from {model_name_or_path}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Resize model embeddings to match tokenizer
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Store original weights for SC calculation and null-space projection
        logger.info("Storing original model weights for alignment preservation...")
        self.original_weights = {}
        for name, param in self.base_model.named_parameters():
            if any(target in name for target in config.target_modules):
                self.original_weights[name] = param.detach().clone()

        # Configure LoRA if not loading from a checkpoint
        if not is_lora_checkpoint:
            logger.info(f"Configuring LoRA with rank {config.lora_r}...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
            )

            # Apply LoRA to the model
            self.model = get_peft_model(self.base_model, peft_config)
        else:
            # For checkpoint loading, we'll use the base model directly
            self.model = self.base_model

        self.model.to(device)

        # Initialize cluster tracking for alignment preservation
        self.base_embeddings = None
        self.base_sc = None
        self.safe_centroid = None
        self.unsafe_centroid = None
        self.safe_cov = None
        self.unsafe_cov = None

    def compute_embeddings(self, input_ids, attention_mask=None, use_base_model=False):
        with torch.no_grad():
            model_to_use = self.base_model if use_base_model else self.model

            outputs = model_to_use(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
            embeddings = last_hidden_state[:, -1, :].detach().cpu().numpy()
            return embeddings

    def compute_base_silhouette_coefficient(self, dataloader):
        logger.info("Computing base model's representation separation (Silhouette Coefficient)...")
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                safety_labels = batch["safety_labels"].numpy()

                batch_embeddings = self.compute_embeddings(
                    input_ids, attention_mask, use_base_model=True
                )
                all_embeddings.append(batch_embeddings)
                all_labels.append(safety_labels)

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)

        self.base_embeddings = all_embeddings

        if len(np.unique(all_labels)) < 2:
            logger.warning("Need at least 2 clusters to compute base Silhouette Coefficient")
            return 0.0

        try:
            sc = silhouette_score(all_embeddings, all_labels)
            self.base_sc = sc
            logger.info(f"Base model's Silhouette Coefficient: {sc:.4f}")
            return sc
        except Exception as e:
            logger.error(f"Error computing base Silhouette Coefficient: {e}")
            return 0.0

    def compute_silhouette_coefficient(self, embeddings, labels):
        if len(np.unique(labels)) < 2:
            return 0.0

        try:
            sc = silhouette_score(embeddings, labels)
            return sc
        except Exception:
            return 0.0

    def compute_cluster_metrics(self, embeddings, labels):
        if self.base_embeddings is None or len(np.unique(labels)) < 2:
            return {}

        safe_mask = labels == 0
        unsafe_mask = labels == 1

        current_safe_centroid = np.mean(embeddings[safe_mask], axis=0) if np.any(safe_mask) else None
        current_unsafe_centroid = np.mean(embeddings[unsafe_mask], axis=0) if np.any(unsafe_mask) else None

        if len(self.base_embeddings) >= len(embeddings):
            base_safe_centroid = np.mean(self.base_embeddings[:len(embeddings)][safe_mask], axis=0) if np.any(safe_mask) else None
            base_unsafe_centroid = np.mean(self.base_embeddings[:len(embeddings)][unsafe_mask], axis=0) if np.any(unsafe_mask) else None
        else:
            return {}

        metrics = {}
        current_sc = self.compute_silhouette_coefficient(embeddings, labels)
        metrics["current_sc"] = current_sc

        if (current_safe_centroid is not None and current_unsafe_centroid is not None and base_safe_centroid is not None and base_unsafe_centroid is not None):
            current_distance = np.linalg.norm(current_safe_centroid - current_unsafe_centroid)
            metrics["current_distance"] = current_distance
            base_distance = np.linalg.norm(base_safe_centroid - base_unsafe_centroid)
            metrics["base_distance"] = base_distance
            metrics["distance_ratio"] = current_distance / base_distance if base_distance > 0 else 1.0

        return metrics

    def detect_poison_samples(self, embeddings, labels):
        if not self.config.enable_poison_detection:
            return np.zeros(len(embeddings)), np.ones(len(embeddings))

        safe_indices = np.where(labels == 0)[0]
        unsafe_indices = np.where(labels == 1)[0]

        if len(safe_indices) < 2 or len(unsafe_indices) < 2:
            return np.zeros(len(embeddings)), np.ones(len(embeddings))

        safe_embeddings = embeddings[safe_indices]
        unsafe_embeddings = embeddings[unsafe_indices]

        safe_centroid = np.mean(safe_embeddings, axis=0)
        safe_cov = EmpiricalCovariance().fit(safe_embeddings)

        unsafe_centroid = np.mean(unsafe_embeddings, axis=0)
        unsafe_cov = EmpiricalCovariance().fit(unsafe_embeddings)

        if self.safe_centroid is None:
            self.safe_centroid = safe_centroid
            self.safe_cov = safe_cov
        if self.unsafe_centroid is None:
            self.unsafe_centroid = unsafe_centroid
            self.unsafe_cov = unsafe_cov

        mahalanobis_distances = np.zeros(len(embeddings))
        sample_weights = np.ones(len(embeddings))

        for i, embedding in enumerate(embeddings):
            dist_to_safe = mahalanobis(embedding, safe_centroid, safe_cov.precision_)
            dist_to_unsafe = mahalanobis(embedding, unsafe_centroid, unsafe_cov.precision_)

            expected_label = 0 if dist_to_safe < dist_to_unsafe else 1

            if labels[i] != expected_label:
                if labels[i] == 0:
                    dist = dist_to_safe
                else:
                    dist = dist_to_unsafe

                mahalanobis_distances[i] = dist

                if dist > self.config.poison_threshold:
                    sample_weights[i] = max(0.1, 1.0 - (dist - self.config.poison_threshold) / 10.0)

            else:
                mahalanobis_distances[i] = min(dist_to_safe, dist_to_unsafe)

        return mahalanobis_distances, sample_weights

    def compute_null_space_projection(self, model_output, unsafe_mask):
        if torch.sum(unsafe_mask) == 0:
            return torch.tensor(0.0, device=self.device)

        unsafe_representations = model_output[unsafe_mask]
        null_space_penalty = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if any(target in name for target in self.config.target_modules) and param.requires_grad:
                original_param = self.original_weights.get(name, None)
                if original_param is None:
                    continue

                delta = param - original_param

                if delta.dim() == 2 and unsafe_representations.dim() == 2:
                    if delta.size(0) == unsafe_representations.size(1):
                        projection = torch.norm(delta @ unsafe_representations.T, p=2) ** 2
                    elif delta.size(1) == unsafe_representations.size(1):
                        projection = torch.norm(unsafe_representations @ delta.T, p=2) ** 2
                    else:
                        continue

                    null_space_penalty += projection

        return null_space_penalty

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        batch_size=8,
        num_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        output_dir="./alignguard_output",
        gradient_accumulation_steps=1,
    ):
        from torch.utils.data import DataLoader

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(0.1 * total_steps)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )

        base_sc = self.compute_base_silhouette_coefficient(train_dataloader)

        global_step = 0
        best_eval_loss = float("inf")
        best_alignment_score = float("-inf")

        sc_history = []
        distance_ratio_history = []

        logger.info("Starting SafeLoRA fine-tuning with alignment preservation...")
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_sc_loss = 0.0
            epoch_null_loss = 0.0
            epoch_task_loss = 0.0

            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                safety_labels = batch["safety_labels"].numpy()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )

*** End Comment to=functions.create_file پروگرامically provided content **Continuing*** (adjusting)**