#!/usr/bin/env python3
"""
Simplified emotion dataset loader for cross-corpus experiments
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import math


class EmotionDataset(Dataset):
    """
    Simplified dataset for cross-corpus emotion recognition

    Supports:
    - IEMO, MSPI, MSPP (with VAD annotations)
    - Pre-extracted audio features (Emotion2Vec 768-dim)
    - Text transcripts
    - VAD normalization to [0, 1] range
    - Prototypicality (difficulty) calculation
    """

    # Dataset mappings
    DATASET_MAP = {
        "IEMO": "cairocode/IEMO_Audio_Text_Merged",
        "MSPI": "cairocode/MSPI_Audio_Text_Merged",
        "MSPP": "cairocode/MSPP_Audio_Text_Merged",
        "CMUMOSEI": "cairocode/CMUMOSEI_Emotion2Vec_PrecomputedEncodings",
        "SAMSEMO": "cairocode/SAMSEMO_Emotion2Vec_PrecomputedEncodings",
    }

    # Datasets that have VAD annotations (for regression and prototypicality)
    DATASETS_WITH_VAD = {"IEMO", "MSPI", "MSPP"}

    def __init__(self, dataset_name, split="train", config=None, task_type="classification"):
        """
        Args:
            dataset_name: str - "IEMO", "MSPI", or "MSPP"
            split: str - "train" or "test"
            config: Config object with expected_vad and modality
            task_type: str - "classification" or "regression"
        """
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.task_type = task_type
        self.modality = getattr(config, 'modality', 'both') if config else 'both'

        self.has_vad = dataset_name in self.DATASETS_WITH_VAD

        # For regression, only datasets with VAD are valid
        if task_type == "regression" and not self.has_vad:
            raise ValueError(f"{dataset_name} has no VAD annotations - cannot use for regression")

        # Load dataset from HuggingFace
        if dataset_name not in self.DATASET_MAP:
            raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(self.DATASET_MAP.keys())}")

        dataset_path = self.DATASET_MAP[dataset_name]
        self.hf_dataset = load_dataset(dataset_path, split=split, trust_remote_code=True)

        print(f"📥 Loaded {dataset_name}: {len(self.hf_dataset)} samples")
        print(f"   Columns: {self.hf_dataset.column_names}")

        # Process data
        self.data = []
        skipped_vad_count = 0

        for item in self.hf_dataset:
            # Extract features (store raw, convert to tensor in __getitem__)
            if self.modality in ["audio", "both"]:
                if "emotion2vec_features" not in item or item["emotion2vec_features"] is None:
                    continue
                features = item["emotion2vec_features"][0]["feats"]  # Raw list
            else:
                features = None

            # Extract transcript
            if self.modality in ["text", "both"]:
                transcript = item.get("transcript", item.get("text", "[EMPTY]"))
                if not transcript:
                    transcript = "[EMPTY]"
            else:
                transcript = None

            # Extract label
            label = item["label"]

            # Extract VAD values with multiple naming variants
            valence = item.get("valence", item.get("consensus_valence", item.get("EmoVal", None)))
            arousal = item.get("arousal", item.get("consensus_arousal", item.get("EmoAct", None)))
            dominance = item.get("domination", item.get("consensus_dominance", item.get("EmoDom", None)))

            # For regression, skip samples with missing VAD
            if task_type == "regression":
                if valence is None or (isinstance(valence, float) and math.isnan(valence)):
                    skipped_vad_count += 1
                    continue
                if arousal is None or (isinstance(arousal, float) and math.isnan(arousal)):
                    skipped_vad_count += 1
                    continue
                if dominance is None or (isinstance(dominance, float) and math.isnan(dominance)):
                    skipped_vad_count += 1
                    continue

            # Normalize VAD to [0, 1] range
            if valence is not None and not (isinstance(valence, float) and math.isnan(valence)):
                if dataset_name == "MSPP":
                    valence = (valence - 1) / 6  # 1-7 scale → 0-1
                else:  # IEMO, MSPI use 1-5 scale
                    valence = (valence - 1) / 4  # 1-5 scale → 0-1

            if arousal is not None and not (isinstance(arousal, float) and math.isnan(arousal)):
                if dataset_name == "MSPP":
                    arousal = (arousal - 1) / 6
                else:
                    arousal = (arousal - 1) / 4

            if dominance is not None and not (isinstance(dominance, float) and math.isnan(dominance)):
                if dataset_name == "MSPP":
                    dominance = (dominance - 1) / 6
                else:
                    dominance = (dominance - 1) / 4

            # For classification, use midpoint if missing
            if valence is None:
                valence = 0.5
            if arousal is None:
                arousal = 0.5
            if dominance is None:
                dominance = 0.5

            # Store sample data
            self.data.append({
                "features": features,  # Raw list (convert to tensor in __getitem__)
                "transcript": transcript,
                "label": label,
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "dataset": dataset_name,
            })

        print(f"✅ Loaded {len(self.data)} samples from {dataset_name}")
        if skipped_vad_count > 0:
            print(f"   ⚠️  Skipped {skipped_vad_count} samples with missing/NaN VAD values (regression mode)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample with on-the-fly tensor conversion

        Returns dict with:
            - label: int (0-3)
            - valence, arousal, dominance: float (0-1 normalized)
            - features: tensor [audio_dim] - if audio/both modality
            - transcript: str - if text/both modality
            - dataset: str - dataset name
        """
        item = self.data[idx]

        result = {
            "label": torch.tensor(item["label"], dtype=torch.long),
            "valence": item["valence"],
            "arousal": item["arousal"],
            "dominance": item["dominance"],
            "dataset": item["dataset"],
        }

        # Add audio features (convert to tensor here)
        if self.modality in ["audio", "both"] and item["features"] is not None:
            features = item["features"]
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            result["features"] = features

        # Add text
        if self.modality in ["text", "both"]:
            result["transcript"] = item["transcript"]

        return result


def create_datasets(config):
    """
    Create train and test datasets based on config

    Args:
        config: Config object with train_dataset, test_datasets, task_type

    Returns:
        train_dataset, test_datasets (list)
    """
    task_type = getattr(config, 'task_type', 'classification')

    # Create training dataset
    train_dataset = EmotionDataset(
        config.train_dataset,
        split="train",
        config=config,
        task_type=task_type
    )

    # Create test datasets (cross-corpus evaluation)
    test_dataset_names = getattr(config, 'test_datasets', [])

    # If not specified, use all datasets except training one
    if not test_dataset_names:
        all_datasets = list(EmotionDataset.DATASET_MAP.keys())
        test_dataset_names = [d for d in all_datasets if d != config.train_dataset]

    # For regression, filter out datasets without VAD
    if task_type == "regression":
        test_dataset_names = [d for d in test_dataset_names if d in EmotionDataset.DATASETS_WITH_VAD]

    test_datasets = []
    for dataset_name in test_dataset_names:
        test_dataset = EmotionDataset(
            dataset_name,
            split="train",  # Use full dataset for testing
            config=config,
            task_type=task_type
        )
        test_datasets.append(test_dataset)

    print(f"🚀 Training: {config.train_dataset} → {test_dataset_names}")

    return train_dataset, test_datasets
