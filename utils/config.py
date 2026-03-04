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

        # BERT unfreezing
        self.unfreeze_bert_layers = 0  # 0 = fully frozen, 2-4 = unfreeze top N layers
        self.bert_learning_rate = 5e-7  # 10x lower than main LR

        # Audio encoder
        self.audio_encoder_type = "preextracted"  # "preextracted", "wav2vec2", or "emotion2vec"
        self.audio_model_name = "facebook/wav2vec2-base-960h"
        self.unfreeze_audio_layers = 0  # 0 = fully frozen, 2-4 = unfreeze top N layers
        self.audio_learning_rate = 5e-7  # differential LR for wav2vec2/emotion2vec
        self.max_audio_seconds = 40  # truncate waveforms to this length (caps VRAM usage)
        self.emotion2vec_upstream_dir = "/home/rml/Documents/pythontest/emotion2vec/upstream"

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

        # Projection head (for contrastive learning)
        self.projection_dim = 128
        self.projection_hidden_dim = 512

        # Prototypicality weighting
        self.prototypical_alpha = 1.0  # For v1: exp(-alpha * difficulty)
        self.prototypical_beta = 0.5   # For v2: pair-level weighting
        self.prototypical_threshold = 1.0  # For v3: binary threshold

        # Prototype-anchored loss params
        self.margin_base = 0.1  # Min margin for prototypical samples
        self.separation_margin = 2.0  # Min distance between class prototypes
        self.separation_weight = 0.5  # Weight for prototype separation loss
        self.alignment_weight = 1.0  # Weight for cross-domain alignment (multiDS)

        # Prototypicality-weighted primary loss
        self.use_prototypical_weighting = False
        self.prototypical_weighting_alpha = 2.0  # exp(-alpha * difficulty)
        self.use_prototypical_label_smoothing = False
        self.label_smoothing_beta = 0.5  # smoothing = beta * difficulty
        self.label_smoothing_max = 0.6  # cap smoothing to avoid total flattening

        # Auxiliary prototypicality prediction
        self.use_proto_predictor = False
        self.proto_predictor_weight = 0.5  # λ for MSE(pred_proto, actual_proto)
        self.proto_predictor_hidden_dim = 256

        # Domain adversarial training
        self.use_domain_adversarial = False
        self.adversarial_weight = 0.1  # Weight for adversarial loss
        self.adversarial_alpha = 2.0  # Prototypicality weight decay for adversarial
        self.adversarial_hidden_dim = 256  # Hidden dim of domain discriminator
        self.use_prototypical_adversarial = True  # Weight adversarial by prototypicality

        # Two-stage prototypicality training
        self.use_two_stage_training = False
        self.stage1_percentile = 50  # bottom N% most prototypical for stage 1
        self.stage1_epochs = 30     # epochs for stage 1
        self.stage2_lr_factor = 0.1  # reduce LR by this factor for stage 2

        # Early stopping
        self.early_stopping_patience = 10

        # Logging
        self.wandb_project = "Emotion2Vec_Contrastive"
        self.experiment_name = "baseline"
        self.seed = 42
        self.seeds = [42]  # List of seeds for multi-seed runs

        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _fix_numeric_types(cls, config_dict):
        """Ensure numeric types are correctly parsed from YAML"""
        numeric_fields = [
            'learning_rate', 'weight_decay', 'dropout', 'batch_size', 'num_epochs',
            'contrastive_weight', 'contrastive_temperature', 'prototypical_alpha',
            'prototypical_beta', 'prototypical_threshold', 'val_split',
            'audio_dim', 'hidden_dim', 'num_classes', 'text_max_length',
            'fusion_hidden_dim', 'num_attention_heads', 'vad_output_dim', 'seed',
            'projection_dim', 'projection_hidden_dim',
            'margin_base', 'separation_margin', 'separation_weight',
            'early_stopping_patience',
            'adversarial_weight', 'adversarial_alpha', 'adversarial_hidden_dim',
            'prototypical_weighting_alpha', 'label_smoothing_beta', 'label_smoothing_max',
            'proto_predictor_weight', 'proto_predictor_hidden_dim',
            'unfreeze_bert_layers', 'bert_learning_rate',
            'unfreeze_audio_layers', 'audio_learning_rate',
            'stage1_percentile', 'stage1_epochs', 'stage2_lr_factor',
            'max_audio_seconds',
        ]

        for field in numeric_fields:
            if field in config_dict:
                try:
                    config_dict[field] = float(config_dict[field])
                except (ValueError, TypeError):
                    pass

        int_fields = [
            'batch_size', 'num_epochs', 'num_classes', 'text_max_length',
            'audio_dim', 'hidden_dim', 'fusion_hidden_dim', 'num_attention_heads',
            'vad_output_dim', 'seed', 'projection_dim', 'projection_hidden_dim',
            'early_stopping_patience', 'adversarial_hidden_dim',
            'proto_predictor_hidden_dim',
            'unfreeze_bert_layers',
            'unfreeze_audio_layers', 'stage1_epochs',
        ]

        for field in int_fields:
            if field in config_dict:
                try:
                    config_dict[field] = int(float(config_dict[field]))
                except (ValueError, TypeError):
                    pass

        return config_dict

    @classmethod
    def from_yaml(cls, yaml_path, experiment_id=None):
        """
        Load config from YAML file.

        Supports two formats:
        1. Single config: all keys at top level
        2. Multi-experiment: template_config + experiments list
           Uses YAML anchors (&template) and merge keys (<<: *template)

        Args:
            yaml_path: Path to YAML file
            experiment_id: int (index) or str (name) to select experiment.
                          None = load as single config (or error if multi-experiment)
        """
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Check if multi-experiment config
        if "experiments" in yaml_config:
            if experiment_id is None:
                print("Available experiments:")
                for i, exp in enumerate(yaml_config["experiments"]):
                    print(f"   {i}: {exp.get('name', exp.get('id', f'experiment_{i}'))}")
                raise ValueError(
                    "Multi-experiment config detected. Use --experiment <id> or --all"
                )

            # Find the specified experiment
            if isinstance(experiment_id, int):
                if 0 <= experiment_id < len(yaml_config["experiments"]):
                    config_dict = yaml_config["experiments"][experiment_id]
                else:
                    raise ValueError(f"Experiment index {experiment_id} out of range")
            else:
                # Find by name or id
                config_dict = None
                for exp in yaml_config["experiments"]:
                    if exp.get("id") == experiment_id or exp.get("name") == experiment_id:
                        config_dict = exp
                        break
                if config_dict is None:
                    raise ValueError(f"Experiment '{experiment_id}' not found")

            exp_name = config_dict.get('name', config_dict.get('id', experiment_id))
            print(f"Running experiment: {exp_name}")
        else:
            config_dict = yaml_config

        # Remove YAML-only keys that aren't config fields
        config_dict.pop('template_config', None)

        # Map 'name' to 'experiment_name' (YAML uses 'name', Config uses 'experiment_name')
        if 'name' in config_dict and 'experiment_name' not in config_dict:
            config_dict['experiment_name'] = config_dict.pop('name')
        elif 'name' in config_dict:
            config_dict.pop('name')

        config_dict = cls._fix_numeric_types(config_dict)
        return cls(**config_dict)

    @classmethod
    def list_experiments(cls, yaml_path):
        """List all experiments in a multi-experiment YAML file"""
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if "experiments" not in yaml_config:
            return None

        experiments = []
        for i, exp in enumerate(yaml_config["experiments"]):
            experiments.append({
                'index': i,
                'name': exp.get('name', exp.get('id', f'experiment_{i}')),
            })
        return experiments

    def to_dict(self):
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save_yaml(self, yaml_path):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def __repr__(self):
        return f"Config({self.experiment_name})"
