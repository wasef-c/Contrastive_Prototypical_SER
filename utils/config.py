#!/usr/bin/env python3
"""
Configuration system for emotion recognition with contrastive learning
"""

import yaml
from utils.prototypicality import DEFAULT_EXPECTED_VAD


class Config:
    """Clean configuration for contrastive learning experiments"""

    def __init__(self, **kwargs):
        # Dataset
        self.train_dataset = "MSPP"
        self.test_datasets = []  # Empty = all others
        self.val_split = 0.1

        # Model architecture
        self.modality = "both"  # "audio", "text", "both"
        self.audio_dim = 768
        self.text_model_name = "bert-base-uncased"
        self.text_max_length = 128
        self.hidden_dim = 1024
        self.num_classes = 4

        # Fusion (for multimodal)
        self.fusion_type = "cross_attention"
        self.fusion_hidden_dim = 512
        self.num_attention_heads = 8

        # Task
        self.task_type = "classification"  # "classification" or "regression"
        self.vad_output_dim = 3

        # Training
        self.num_epochs = 60
        self.batch_size = 32
        self.learning_rate = 5e-6
        self.weight_decay = 5e-6
        self.dropout = 0.1

        # Prototypicality
        self.expected_vad = DEFAULT_EXPECTED_VAD.copy()

        # Contrastive Learning
        self.use_contrastive = False
        self.contrastive_loss_type = "supervised"  # "supervised", "prototypical_v1", "prototypical_v2", "prototypical_v3"
        self.contrastive_weight = 0.5
        self.contrastive_temperature = 0.07
        self.contrastive_warmup_epochs = 5

        # Prototypicality weighting
        self.prototypical_alpha = 1.0  # For v1: exp(-alpha * difficulty)
        self.prototypical_beta = 0.5   # For v2: pair-level weighting
        self.prototypical_threshold = 1.0  # For v3: binary threshold

        # Logging
        self.wandb_project = "Emotion2Vec_Contrastive"
        self.experiment_name = "baseline"
        self.seed = 42

        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_path):
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Ensure numeric types are correctly parsed
        numeric_fields = [
            'learning_rate', 'weight_decay', 'dropout', 'batch_size', 'num_epochs',
            'contrastive_weight', 'contrastive_temperature', 'prototypical_alpha',
            'prototypical_beta', 'prototypical_threshold', 'val_split',
            'audio_dim', 'hidden_dim', 'num_classes', 'text_max_length',
            'fusion_hidden_dim', 'num_attention_heads', 'vad_output_dim', 'seed'
        ]

        for field in numeric_fields:
            if field in config_dict:
                try:
                    config_dict[field] = float(config_dict[field])
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails

        # Convert integer fields
        int_fields = [
            'batch_size', 'num_epochs', 'num_classes', 'text_max_length',
            'audio_dim', 'hidden_dim', 'fusion_hidden_dim', 'num_attention_heads',
            'vad_output_dim', 'seed'
        ]

        for field in int_fields:
            if field in config_dict:
                try:
                    config_dict[field] = int(float(config_dict[field]))
                except (ValueError, TypeError):
                    pass

        return cls(**config_dict)

    def to_dict(self):
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save_yaml(self, yaml_path):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def __repr__(self):
        return f"Config({self.experiment_name})"
